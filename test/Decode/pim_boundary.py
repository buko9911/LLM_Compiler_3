#!/usr/bin/env python3
"""The choices around a PIM phase: shared input feed, K pieces per rank, several PIM records.

One GEMV (K 2560, N 2560, 1 token), both ranks and one rank. Checks in the compiled program: the feed runs on every
core, a phase is cut into PIM records, each rank's input stores cover only the K pieces it computes (both ranks) or the
whole K (one rank). Runs both on the simulator: results byte-equal to an independent integer matmul, and the records of
a phase overlap (the later ones are read while the earlier ones compute).
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


def stores_per_rank(commands, dense_end):
    bytes_of = {}
    for c in commands:
        if c["kind"] == "gdma_store" and (c["lp6_address"] & ((1 << 33) - 1)) >= dense_end:
            rank = c["lp6_address"] >> 33
            bytes_of[rank] = bytes_of.get(rank, 0) + c["row_bytes"] * c["rows"]
    return bytes_of


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
    sys.path.insert(0, str(npu / "transactional_emulator/tools"))
    from isa_v2 import decode_program
    emu = npu / "transactional_emulator/target/release/transactional_emulator"
    library = (a.lp6_library or workspace.RAMULATOR_LIBRARY).resolve()
    env = dict(os.environ, LP6_RAMULATOR_LIBRARY=str(library))
    settings = out / "settings.toml"
    settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4", (npu / "plena_settings.toml").read_text()))
    m, k, n = 1, 2560, 2560
    model = out / "model"
    model.mkdir()
    rng = np.random.default_rng(2560)
    xq = rng.integers(-127, 128, (m, k), dtype=np.int8)
    wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
    xq[:, 0] = wq[:, 0] = 127
    x = xq.astype(np.float16) / np.float16(64)
    save_file({"proj.weight": wq.astype(np.float16) / np.float16(64)}, model / "model.safetensors")
    expected = ((xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32) / 4096).astype(np.float16)
    source = out / "step.mlir"
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
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
        arguments=[dict(kind="input", name="x"), dict(kind="weight", name="proj.weight")],
        results=[dict(kind="output", name="y")]), indent=2))
    # K split in two forced by the layout search options, so both ranks get their own K piece.
    for name, concurrent, accs, split in (("two_ranks", True, [10], 2), ("one_rank", False, [10], 1)):
        case = out / name
        case.mkdir()
        spec = json.loads((root / "configs/target.json").read_text())
        spec["pim"].update(concurrent_ranks=concurrent, stream_accs=accs, max_split_k=split)
        target = case / "target.json"
        target.write_text(json.dumps(spec, indent=2))
        build = case / "build"
        run([a.compiler.resolve(), source, "--manifest", manifest, "--spec", target, "--force=pim", "-o", build],
            root, case / "compile.log")
        io = json.loads((build / "io.json").read_text())
        decisions = {d["kind"]: d for p in io["phases"] for d in p.get("decisions", [])}
        assert decisions["input feed"]["choice"] == "split", decisions["input feed"]
        records = int(decisions["pim records"]["choice"])
        assert records > 1, decisions["pim records"]
        assert decisions["pim stream layout"]["choice"] == f"A=10 S={split}", decisions["pim stream layout"]
        program = decode_program((build / "program.bin").read_bytes())["commands"]
        assert sum(c["kind"] == "pim" for c in program) == records
        stores = stores_per_rank(program, io["dense_end"])
        # Activation bytes per rank: K x 8 PUs x 8 subchannels, divided among the K pieces when both ranks compute.
        if concurrent:
            assert stores == {0: k * 64 // 2, 1: k * 64 // 2}, stores
        else:
            assert sum(stores.values()) == k * 64 and len(stores) == 1, stores
        cores = {c["target_logical_core"] for c in program if c["kind"] == "core_block"}
        assert cores == {0, 1, 2, 3}, cores
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
        actual = np.frombuffer((build / "sram_dump.core0.bin").read_bytes(), np.float16, m * n, o["l1"]).reshape(m, n)
        assert actual.tobytes() == expected.tobytes(), (name, "integer reference mismatch")
        pims = [c for c in json.loads((build / "commands.json").read_text())["commands"] if c["kind"] == "pim"]
        assert len(pims) == records and all(p["issue_cycle"] < pims[-1]["issue_cycle"] for p in pims[:-1])
        assert pims[-1]["issue_cycle"] < max(p["end_cycle"] for p in pims), "records do not overlap"
        print(name, "PASS", "records", records, "stores", stores, flush=True)


if __name__ == "__main__":
    main()
