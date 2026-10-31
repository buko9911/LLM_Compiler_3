#!/usr/bin/env python3
"""Row chains of hyb-npu-elementwise-distribute (npu.row_distribution).

hyb.quant and the elementwise ops that read a row reduction are split over the cores by token
rows; their single-use row-wise producers (the reduction, its zero fill, the normalization) are
fused into the same core loop. Structure only; the numbers are checked by the runtime tests.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def source(rows=64, columns=2560, quant=True, share_norm=False, phase="npu"):
    x, var = f"tensor<{rows}x{columns}xf16>", f"tensor<{rows}xf32>"
    q, s = f"tensor<{rows}x{columns}xi8>", f"tensor<{rows}x1xf32>"
    results, returned = [], []
    if quant:
        results += [q, s]
        returned += ["%a#0", "%a#1"]
    if share_norm or not quant:
        results.append(x)
        returned.append("%n")
    quantized = f"""%qe = tensor.empty() : {q}
      %se = tensor.empty() : {s}
      %a:2 = hyb.quant %n into %qe, %se : {x}, {q}, {s} -> {q}, {s}""" if quant else ""
    return f"""module {{func.func @phase(%x:{x},%w:tensor<{columns}xf16>) -> ({', '.join(results)})
      attributes {{hyb.phase="{phase}"}} {{
      %zero = arith.constant 0.0 : f32
      %h = arith.constant {float(columns)} : f32
      %eps = arith.constant 1.0e-06 : f32
      %ve = tensor.empty() : {var}
      %vi = linalg.fill ins(%zero : f32) outs(%ve : {var}) -> {var}
      %v = linalg.generic {{indexing_maps = [affine_map<(r,c)->(r,c)>, affine_map<(r,c)->(r)>],
          iterator_types = ["parallel", "reduction"]}} ins(%x : {x}) outs(%vi : {var}) {{
      ^bb0(%e: f16, %acc: f32):
        %f = arith.extf %e : f16 to f32
        %p = arith.mulf %f, %f : f32
        %t = arith.addf %acc, %p : f32
        linalg.yield %t : f32
      }} -> {var}
      %ne = tensor.empty() : {x}
      %n = linalg.generic {{indexing_maps = [affine_map<(r,c)->(r,c)>, affine_map<(r,c)->(r)>,
          affine_map<(r,c)->(c)>, affine_map<(r,c)->(r,c)>], iterator_types = ["parallel", "parallel"]}}
          ins(%x, %v, %w : {x}, {var}, tensor<{columns}xf16>) outs(%ne : {x}) {{
      ^bb0(%e: f16, %sq: f32, %g: f16, %unused: f16):
        %gf = arith.extf %g : f16 to f32
        %ef = arith.extf %e : f16 to f32
        %m = arith.divf %sq, %h : f32
        %d = arith.addf %m, %eps : f32
        %r = math.rsqrt %d : f32
        %y = arith.mulf %ef, %r : f32
        %z = arith.mulf %y, %gf : f32
        %o = arith.truncf %z : f32 to f16
        linalg.yield %o : f16
      }} -> {x}
      {quantized}
      return {', '.join(returned)} : {', '.join(results)}
    }} }}"""


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    # (source args, spec overrides, row chains, fused producers per chain in program order, cores).
    cases = {
        "rmsnorm_quant": ({}, {}, [2], 4),
        "norm_only": ({"quant": False}, {}, [1], 4),
        "shared_norm": ({"share_norm": True}, {}, [1, 0], 4),
        # Tiny shapes exercise the rewrite with no split cost; the cost rule itself is below.
        "row_divisor": ({"rows": 6, "columns": 256}, {"distribute_min_elements": 1, "core_split_us": 0}, [2], 3),
        "not_worth_splitting": ({"rows": 4, "columns": 64}, {"distribute_min_elements": 1}, [], None),
        "one_row": ({"rows": 1}, {}, [], None),
        "small": ({"rows": 8, "columns": 64}, {}, [], None),
        "off": ({}, {"row_distribution": False}, [], None),
        "pim": ({"phase": "pim"}, {}, [], None),
    }
    results = {}
    for name, (kwargs, options, chains, cores) in cases.items():
        src = out / f"{name}.mlir"
        src.write_text(source(**kwargs))
        spec = json.loads((root / "configs/target.json").read_text())
        spec["npu"].update(elementwise_distribution=True, row_distribution=True)
        spec["npu"].update(options)
        target = out / f"{name}.json"
        target.write_text(json.dumps(spec))
        flag = f"--hyb-npu-elementwise-distribute=spec={target.resolve()}"
        command = [str(args.optimizer.resolve()), str(src), flag]
        run = subprocess.run(command, text=True, capture_output=True)
        (out / f"{name}.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 0, (name, run.stderr)
        found = [int(v) for v in re.findall(r"hyb\.row_chain = (\d+)", run.stdout)]
        assert found == chains, (name, found, chains)
        if chains:
            assert run.stdout.count(f"hyb.cores = {cores} : i64") >= len(chains), name
            assert f"hyb.row_distributions = {len(chains)} : i64" in run.stdout, name
            assert f"hyb.row_fused = {sum(chains)} : i64" in run.stdout, name
            # No row work is left on one core: every generic and quant sits inside a core loop.
            body = run.stdout.split("func.func @phase", 1)[1]
            top = [line for line in body.splitlines() if re.match(r"^    %\S+ = (linalg\.generic|hyb\.quant)", line)]
            assert not top, (name, top)
            again = subprocess.run([command[0], "-", flag], input=run.stdout, text=True, capture_output=True)
            assert again.returncode == 0, (name, again.stderr)
            assert again.stdout.count("scf.forall (") == run.stdout.count("scf.forall ("), (name, "double distribution")
        elif "row_distribution" not in options and kwargs.get("phase") != "pim":
            assert "hyb.row_distributions = 0 : i64" in run.stdout, name
        results[name] = dict(passed=True, chains=found)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
