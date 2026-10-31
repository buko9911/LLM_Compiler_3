#!/usr/bin/env python3
"""Structure checks for the prefill layout passes: hyb-npu-rope (npu.rope_split), transpose
distribution (npu.distribute_transposes) and the gate/up pair (npu.pair_gate_up)."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from npu_gemv_optimizations import tensor_source


def rope_source(tokens=16, heads=8, dim=128, transpose=True, body="x*c+r*s", users=1, phase="npu"):
    half = dim // 2
    x, h = f"tensor<{tokens}x{heads}x{dim}xf16>", f"tensor<{tokens}x{heads}x{half}xf16>"
    table = f"tensor<{tokens}x{dim}xf16>"
    first, second = {"x*c+r*s": ("%xc", "%rs"), "r*s+x*c": ("%rs", "%xc")}.get(body, ("%xc", "%rs"))
    combine = "arith.subf" if body == "x*c-r*s" else "arith.addf"
    out, head_major = x, f"tensor<{heads}x{tokens}x{dim}xf16>"
    tail = ""
    results, types = ["%rope"], [out]
    if transpose:
        tail = f"%t_empty = tensor.empty() : {head_major}\n"
        results, types = [], []
        for u in range(users):
            perm = "[1, 0, 2]" if u == 0 else "[1, 0, 2]"
            tail += f"%t{u} = linalg.transpose ins(%rope : {out}) outs(%t_empty : {head_major}) permutation = {perm}\n"
            results.append(f"%t{u}")
            types.append(head_major)
    return f"""module {{ func.func @phase(%x: {x}, %cos: {table}, %sin: {table}) -> ({", ".join(types)})
  attributes {{hyb.phase = "{phase}"}} {{
  %lo = tensor.extract_slice %x[0, 0, 0] [{tokens}, {heads}, {half}] [1, 1, 1] : {x} to {h}
  %hi = tensor.extract_slice %x[0, 0, {half}] [{tokens}, {heads}, {half}] [1, 1, 1] : {x} to {h}
  %n_empty = tensor.empty() : {h}
  %neg = linalg.generic {{indexing_maps = [affine_map<(a, b, c) -> (a, b, c)>, affine_map<(a, b, c) -> (a, b, c)>],
      iterator_types = ["parallel", "parallel", "parallel"]}} ins(%hi : {h}) outs(%n_empty : {h}) {{
  ^bb0(%v: f16, %o: f16):
    %w = arith.extf %v : f16 to f32
    %m = arith.negf %w : f32
    %t = arith.truncf %m : f32 to f16
    linalg.yield %t : f16
  }} -> {h}
  %rot = tensor.concat dim(2) %neg, %lo : ({h}, {h}) -> {x}
  %r_empty = tensor.empty() : {x}
  %rope = linalg.generic {{indexing_maps = [affine_map<(a, b, c) -> (a, b, c)>, affine_map<(a, b, c) -> (a, c)>,
      affine_map<(a, b, c) -> (a, b, c)>, affine_map<(a, b, c) -> (a, c)>, affine_map<(a, b, c) -> (a, b, c)>],
      iterator_types = ["parallel", "parallel", "parallel"]}}
      ins(%x, %cos, %rot, %sin : {x}, {table}, {x}, {table}) outs(%r_empty : {x}) {{
  ^bb0(%xv: f16, %cv: f16, %rv: f16, %sv: f16, %o: f16):
    %xw = arith.extf %xv : f16 to f32
    %cw = arith.extf %cv : f16 to f32
    %rw = arith.extf %rv : f16 to f32
    %sw = arith.extf %sv : f16 to f32
    %rs = arith.mulf %rw, %sw : f32
    %xc = arith.mulf %xw, %cw : f32
    %sum = {combine} {first}, {second} : f32
    %t = arith.truncf %sum : f32 to f16
    linalg.yield %t : f16
  }} -> {x}
  {tail}
  return {", ".join(results)} : {", ".join(types)}
}} }}"""


def transpose_source(rows=64, heads=8, dim=128):
    src, dst = f"tensor<{rows}x{heads}x{dim}xf16>", f"tensor<{heads}x{rows}x{dim}xf16>"
    return f"""module {{ func.func @phase(%v: {src}) -> {dst} attributes {{hyb.phase = "npu"}} {{
  %e = tensor.empty() : {dst}
  %t = linalg.transpose ins(%v : {src}) outs(%e : {dst}) permutation = [1, 0, 2]
  return %t : {dst}
}} }}"""


def pair_source(nb=8, consumer="mulf", phase="npu", extra_use=False):
    """Two same-input projections (gate, up) whose results feed one elementwise op."""
    bodies = []
    for i in range(2):
        text = tensor_source(nb=nb, phase=phase)
        body = text[text.index('  %zero'):text.index('  return')]
        body = re.sub(r'%([a-zA-Z][a-zA-Z0-9_]*)', lambda m: '%' + m[1] + (str(i) if m[1] not in ('q', 's') else ''), body)
        body = body.replace('proj.weight', f'proj{i}.weight')
        bodies.append(f'''%w{i} = hyb.weight "proj{i}.weight" layout "npu_nk32" : tensor<{nb}x64x32xi8>
%ws{i} = hyb.weight "proj{i}.weight" layout "row_scale" : tensor<{nb * 32}xf32>
''' + body)
    n = nb * 32
    out = f"tensor<8x{n}xf16>"
    results, types = ("%c, %y0", f"{out}, {out}") if extra_use else ("%c", out)
    return f"""module {{ func.func @phase(%q: tensor<8x64xi8>, %s: tensor<8x1xf32>) -> ({types})
  attributes {{hyb.phase = "{phase}"}} {{
{''.join(bodies)}
  %c_empty = tensor.empty() : {out}
  %c = linalg.generic {{indexing_maps = [affine_map<(a, b) -> (a, b)>, affine_map<(a, b) -> (a, b)>,
      affine_map<(a, b) -> (a, b)>], iterator_types = ["parallel", "parallel"]}}
      ins(%y0, %y1 : {out}, {out}) outs(%c_empty : {out}) {{
  ^bb0(%g: f16, %u: f16, %o: f16):
    %gw = arith.extf %g : f16 to f32
    %uw = arith.extf %u : f16 to f32
    %p = arith.{consumer} %gw, %uw : f32
    %t = arith.truncf %p : f32 to f16
    linalg.yield %t : f16
  }} -> {out}
  return {results} : {types}
}} }}"""


def run(optimizer, out, name, text, flags):
    src = out / f"{name}.mlir"
    src.write_text(text)
    p = subprocess.run([str(optimizer), str(src), *flags, "--canonicalize", "--verify-each"],
                       text=True, capture_output=True)
    (out / f"{name}.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, (name, p.stderr)
    return p.stdout


def spec(out, name, **options):
    path = out / f"{name}.json"
    path.write_text(json.dumps({"npu": {"cores": 4, **options}}))
    return path.resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=a.output))
    opt = a.optimizer.resolve()
    results = {}

    # hyb-npu-rope: (source, spec, rewritten, head-major output)
    rope_cases = {
        "head_major": (rope_source(), dict(rope_split=True), True, True),
        "token_major": (rope_source(transpose=False), dict(rope_split=True), True, False),
        "add_order": (rope_source(body="r*s+x*c"), dict(rope_split=True), True, True),
        "two_transposes": (rope_source(users=2), dict(rope_split=True), True, False),
        "not_rope": (rope_source(body="x*c-r*s"), dict(rope_split=True), False, False),
        "off": (rope_source(), dict(rope_split=False), False, False),
        "below_threshold": (rope_source(), dict(rope_split=True, rope_split_min_elements=1 << 20), False, False),
        "pim": (rope_source(phase="pim"), dict(rope_split=True), False, False),
    }
    for name, (text, options, rewritten, head_major) in rope_cases.items():
        got = run(opt, out, f"rope_{name}", text, [f"--hyb-npu-rope=spec={spec(out, name, **options)}"])
        if rewritten:
            assert "tensor.concat" not in got and "arith.negf" not in got, (name, got)
            assert len(re.findall(r"hyb\.rope_split(?!s)", got)) == 1, (name, got)
            assert got.count("tensor.parallel_insert_slice") == 2, (name, got)
            assert got.count("arith.subf") == 1 and got.count("arith.addf") == 1, (name, got)
            assert ("linalg.transpose" in got) == (name == "two_transposes"), (name, got)
            shared = re.search(r"scf\.forall \(%\w+\) in \(4\) shared_outs\(%\w+ = %\w+\) -> \(tensor<(\w+)>\)", got)
            assert shared and (shared[1] == ("8x16x128xf16" if head_major else "16x8x128xf16")), (name, got)
            assert "hyb.cores = 4 : i64" in got and "hyb.rope_splits = 1 : i64" in got, (name, got)
        else:
            assert "tensor.concat" in got and not re.findall(r"hyb\.rope_split(?!s)", got), (name, got)
        results[f"rope_{name}"] = dict(passed=True, rewritten=rewritten)

    # npu.distribute_transposes inside hyb-npu-elementwise-distribute
    for name, rows, options, expected in [
        ("transpose_split", 64, dict(distribute_transposes=True), 1),
        ("transpose_off", 64, dict(distribute_transposes=False), 0),
        ("transpose_small", 8, dict(distribute_transposes=True), 0),
    ]:
        sp = spec(out, name, elementwise_distribution=True, distribute_min_elements=16384, **options)
        got = run(opt, out, name, transpose_source(rows=rows), [f"--hyb-npu-elementwise-distribute=spec={sp}"])
        assert got.count("scf.forall (") == expected, (name, got)
        if expected:
            assert "hyb.cores = 4 : i64" in got and "hyb.transposes_distributed = 1 : i64" in got, (name, got)
            assert re.search(r"linalg.transpose ins\(%\S+ : tensor<64x2x128xf16>\)", got), (name, got)
        results[name] = dict(passed=True, distributed=expected)

    # npu.pair_gate_up: projection fusion, then the GEMV epilogue pairs the halves per core.
    for name, text, options, paired in [
        ("pair", pair_source(), dict(pair_gate_up=True), True),
        ("pair_off", pair_source(), dict(pair_gate_up=False), False),
        ("pair_half_also_used", pair_source(extra_use=True), dict(pair_gate_up=True), False),
    ]:
        sp = spec(out, name, projection_fusion=True, gemv_epilogue_fusion=True, **options)
        got = run(opt, out, name, text, [f"--hyb-npu-projection-fusion=spec={sp}", f"--hyb-npu-gemv-epilogue=spec={sp}"])
        loops = got.count("scf.forall (")
        assert loops == 1, (name, got)
        if paired:
            body = got[got.index("scf.forall"):]
            assert "hyb.gemv_pair" in got and body.count("hyb.dequant") == 2, (name, got)
            assert body.count("hyb.npu_gemv") == 2 and body.count("arith.mulf") == 1, (name, got)
            assert "tensor<8x64xf16>" in body, (name, got)          # 8 blocks / 2 halves / 4 cores = 2 blocks
        else:
            assert "hyb.gemv_pair" not in got, (name, got)
        results[name] = dict(passed=True, paired=paired)
    for option in ("rope_split", "distribute_transposes", "pair_gate_up"):
        sp = out / "invalid.json"
        sp.write_text(json.dumps({"npu": {option: 1}}))
        p = subprocess.run([str(opt), str(out / "rope_head_major.mlir"), f"--hyb-npu-rope=spec={sp.resolve()}"],
                           text=True, capture_output=True)
        assert p.returncode != 0 and option in p.stderr, (option, p.stderr)
        results[f"invalid_{option}"] = dict(passed=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
