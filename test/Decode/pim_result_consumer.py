#!/usr/bin/env python3
"""Work on PIM results split over the cores: the dequants and their elementwise consumer in one core loop, the quant of
the next PIM input split by columns (hyb-npu-gemv-epilogue, hyb-pim-boundary, Codegen::lowerQuantCores).

Two PIM phases: x -> gate, up -> gate * up -> down -> y. The same step is compiled with a split cost that makes every
split pay and with one that makes none pay. Checks: the decisions, both programs give byte-equal results on the
simulator, and the split program is faster. Both ranks and one rank.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from safetensors.numpy import save_file
from weight_prefetch import run


def source(m, k, n):
    t = f"tensor<{m}x{n}xf32>"
    def matmul(name, x, w):
        return f"""
    %{name}_wt = tensor.empty() : tensor<{k}x{n}xf32>
    %{name}_t = linalg.transpose ins(%{w} : tensor<{n}x{k}xf32>) outs(%{name}_wt : tensor<{k}x{n}xf32>) permutation = [1, 0]
    %{name}_e = tensor.empty() : {t}
    %{name}_i = linalg.fill ins(%zero : f32) outs(%{name}_e : {t}) -> {t}
    %{name} = linalg.matmul ins(%{x}, %{name}_t : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%{name}_i : {t}) -> {t}"""
    return f"""#id = affine_map<(i, j) -> (i, j)>
module {{
  func.func @decode_step(%x: tensor<{m}x{k}xf32>, %wg: tensor<{n}x{k}xf32>, %wu: tensor<{n}x{k}xf32>, %wd: tensor<{n}x{k}xf32>) -> {t} {{
    %zero = arith.constant 0.0 : f32{matmul("g", "x", "wg")}{matmul("u", "x", "wu")}
    %pe = tensor.empty() : {t}
    %p = linalg.generic {{indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}}
        ins(%g, %u : {t}, {t}) outs(%pe : {t}) {{
      ^bb0(%a: f32, %b: f32, %c: f32):
        %r = arith.mulf %a, %b : f32
        linalg.yield %r : f32
    }} -> {t}{matmul("y", "p", "wd")}
    return %y : {t}
  }}
}}
"""


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    ap.add_argument("--lp6-library", type=Path, help="Ramulator bridge (default core/ramulator2/liblp6_ramulator.so)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "runtime"))
    import workspace
    npu = workspace.NPU
    emu = npu / "transactional_emulator/target/release/transactional_emulator"
    env = dict(os.environ, LP6_RAMULATOR_LIBRARY=str((a.lp6_library or workspace.RAMULATOR_LIBRARY).resolve()))
    settings = out / "settings.toml"
    settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4", (npu / "plena_settings.toml").read_text()))
    m, k, n = 1, 2560, 2560
    model = out / "model"
    model.mkdir()
    rng = np.random.default_rng(2560)
    weight = lambda: (rng.integers(-127, 128, (n, k), dtype=np.int8).astype(np.float16) / np.float16(64))
    x = rng.integers(-127, 128, (m, k), dtype=np.int8).astype(np.float16) / np.float16(64)
    save_file({"gate.weight": weight(), "up.weight": weight(), "down.weight": weight()}, model / "model.safetensors")
    step = out / "step.mlir"
    step.write_text(source(m, k, n))
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
        arguments=[dict(kind="input", name="x"), dict(kind="weight", name="gate.weight"),
                   dict(kind="weight", name="up.weight"), dict(kind="weight", name="down.weight")],
        results=[dict(kind="output", name="y")]), indent=2))
    for concurrent in (True, False):
        results = {}
        for name, cost in (("split", 0.0), ("one_core", 1000.0)):
            case = out / f"{'two' if concurrent else 'one'}_{name}"
            case.mkdir()
            spec = json.loads((root / "configs/target.json").read_text())
            spec["pim"].update(concurrent_ranks=concurrent)
            spec["npu"].update(core_split_no_stream_us=cost, core_split_us=cost)
            target = case / "target.json"
            target.write_text(json.dumps(spec, indent=2))
            build = case / "build"
            run([a.compiler.resolve(), step, "--manifest", manifest, "--spec", target, "--force=pim", "-o", build],
                root, case / "compile.log")
            io = json.loads((build / "io.json").read_text())
            choices = {}
            for p in io["phases"]:
                for d in p.get("decisions", []):
                    choices.setdefault(d["kind"], []).append(d["choice"])
            expected = "split" if name == "split" else "one core"
            assert expected in choices.get("split result consumer", []), (name, choices)
            if name == "split":
                assert "split" in choices.get("split quant", []), choices
            arg = next(v for v in io["args"] if v["name"] == "x")
            with (build / "lp6.bin").open("r+b") as f:
                f.seek(arg["addr"])
                f.write(x.tobytes())
            kernel = json.loads((build / "system.json").read_text())["pim_kernel"]
            run([emu, "--settings", settings, "--lp6-config", root.parent / "ramulator2/configs/lp6/LPDDR6_PIM_Mobile_4CH.ini",
                 "--system-program", "system.json", "--command-report", "commands.json", "--lp6-image", "lp6.bin",
                 "--lp6-size", "16GiB", "--log-level", "off"], build, build / "run.log",
                {**env, "LP6PIM_BANKSET_PINGPONG": str(kernel.get("bankset_pingpong_lead", 0))})
            o = next(v for v in io["outputs"] if v["name"] == "y")
            actual = np.frombuffer((build / "sram_dump.core0.bin").read_bytes(), np.float16, m * n, o["l1"])
            commands = json.loads((build / "commands.json").read_text())
            results[name] = (actual.tobytes(), max(c["end_cycle"] for c in commands["commands"]))
        assert results["split"][0] == results["one_core"][0], "results differ"
        assert results["split"][1] < results["one_core"][1], (results["split"][1], results["one_core"][1])
        print("two ranks" if concurrent else "one rank", "PASS cycles", results["split"][1], "<", results["one_core"][1], flush=True)


if __name__ == "__main__":
    main()
