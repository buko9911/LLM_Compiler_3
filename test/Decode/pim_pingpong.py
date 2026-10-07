#!/usr/bin/env python3
"""A/B real compiled bank-set layouts against independent integer GEMVs."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import save_file
from weight_prefetch import run, sha256

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime"))
from decode import Device, phase_times  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", type=Path, default=ROOT / "build/bin/npu-compile")
    ap.add_argument("--emulator", type=Path, required=True)
    ap.add_argument("--dramsim-library", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/target.json")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["LD_PRELOAD"] = str(a.dramsim_library.resolve())
    result = dict(provenance={k: sha256(getattr(a, k)) for k in ("compiler", "emulator", "dramsim_library")}, cases={})
    for name, m, k, n in [("full_sc", 1, 4096, 4096), ("partial_tail", 2, 768, 1056),
                           ("large_k", 1, 14336, 4096)]:
        case = out / name
        model = case / "model"
        model.mkdir(parents=True)
        rng = np.random.default_rng(129)
        xq = rng.integers(-127, 128, (m, k), dtype=np.int8)
        wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
        xq[:, 0] = wq[:, 0] = 127
        x = xq.astype(np.float16) / np.float16(64)
        save_file({"proj.weight": wq.astype(np.float16) / np.float16(64)}, model / "model.safetensors")
        expected = ((xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32) / 4096).astype(np.float16)
        source = case / "step.mlir"
        source.write_text(f"""module {{
  func.func @decode_step(%x: tensor<{m}x{k}xf32>, %w: tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %wt = tensor.empty() : tensor<{k}x{n}xf32>
    %t = linalg.transpose ins(%w : tensor<{n}x{k}xf32>) outs(%wt : tensor<{k}x{n}xf32>) permutation = [1, 0]
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %y = linalg.matmul ins(%x, %t : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %y : tensor<{m}x{n}xf32>
  }}
}}
""")
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
            arguments=[dict(kind="input", name="x"), dict(kind="weight", name="proj.weight")],
            results=[dict(kind="output", name="y")]), indent=2))
        modes = {}
        for lead in (0, 12):
            build = case / f"lead{lead}"
            run([a.compiler.resolve(), source, "--manifest", manifest, "--spec", a.spec.resolve(),
                 "--force=pim", f"--pim-bankset-pingpong={lead}", "-o", build], ROOT, case / f"compile{lead}.log")
            dev = Device(build, emulator=a.emulator,
                         lp6_config=ROOT.parent / "lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini")
            dev.write("x", x)
            outputs, report, wall = dev.run()
            actual = outputs["y"]
            assert np.isfinite(actual).all() and actual.tobytes() == expected.tobytes(), (name, lead)
            times = phase_times(report)
            modes[str(lead)] = dict(all_outputs_byte_exact=True, wall_seconds=wall,
                total_us=sum(t for _, _, t in times), pim_us=sum(t for _, kind, t in times if kind == "pim"),
                program_sha256=sha256(build / "program.bin"), phase_times=times)
            print(name, lead, modes[str(lead)]["total_us"], "us PASS", flush=True)
        assert modes["0"]["program_sha256"] == modes["12"]["program_sha256"]
        result["cases"][name] = dict(m=m, k=k, n=n, modes=modes)
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
