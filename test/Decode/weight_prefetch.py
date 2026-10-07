#!/usr/bin/env python3
"""Compile and simulate deterministic GEMVs with/without L1 weight prefetch.

No model download or torch export is needed. Exercises L2 ring wraparound, odd
batch tails, multiple MMA rectangles per tile, and the single-slot fallback.
Requires numpy, safetensors and the sibling NPU simulator. Output must be new.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def run(cmd, cwd, log, env=None):
    with log.open("w") as f:
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{log.read_text()[-4000:]}")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    ap.add_argument("--npu-root", type=Path, default=root.parent / "simulator/npu")
    ap.add_argument("--pim-root", type=Path, default=root.parent / "simulator/dram-pim")
    ap.add_argument("--dramsim-library", type=Path, help="explicit library (overrides the emulator's DT_RPATH)")
    ap.add_argument("--emulator", type=Path, help="frozen simulator executable")
    ap.add_argument("--settings", type=Path, help="simulator settings to copy")
    ap.add_argument("--case", action="append", help="run only these named cases (repeatable)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.npu_root / "transactional_emulator/tools"))
    from isa_v2 import decode_program
    from isa_v2_trace import trace_v2

    settings = a.output / "settings.toml"
    settings.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4",
                               (a.settings or a.npu_root / "plena_settings.toml").read_text()))
    emu = (a.emulator or a.npu_root / "transactional_emulator/target/release/transactional_emulator").resolve()
    env = dict(os.environ, LP6_DRAMSIM3_ROOT=str(a.pim_root.resolve()))
    library = (a.dramsim_library or a.pim_root / "libdramsim3.so").resolve()
    env["LD_PRELOAD"] = str(library)
    linked = subprocess.check_output(["ldd", str(emu)], text=True, env=env)
    selected = [line.strip() for line in linked.splitlines() if "libdramsim3.so" in line]
    assert len(selected) == 1 and str(library) in selected[0], selected
    (a.output / "provenance.json").write_text(json.dumps(dict(
        compiler_sha256=sha256(a.compiler), emulator_sha256=sha256(emu),
        library=str(library), library_sha256=sha256(library),
        loader_library=selected[0].split(" (")[0], settings_sha256=sha256(settings)), indent=2) + "\n")
    results = {}
    # name, M, K, N, weight tile limit, L2 staging per core, should pipeline
    cases = [("qwen_q_tail", 1, 2560, 4 * 32 * 19, 131072, 1048576, True),
             ("qwen_down_tail", 1, 9728, 4 * 32 * 7, 131072, 1048576, True),
             ("multi_rectangle", 2, 256, 4 * 32 * 36, 32768, 1048576, True),
             ("multiple_activation_rectangles", 33, 64, 4 * 32 * 9, 2048, 1048576, True),
             ("two_staging_slots", 1, 2560, 4 * 32 * 5, 81920, 163840, False),
             ("one_staging_slot", 1, 2560, 4 * 32 * 3, 81920, 81920, False),
             ("single_iteration", 1, 2560, 4 * 32, 131072, 1048576, False),
             ("two_iterations", 1, 2560, 4 * 32 * 2, 131072, 1048576, True),
             ("auto_ring", 1, 4096, 4 * 32 * 41, 131072, 0, True),
             ("cross_batch_ring", 1, 4096, 4 * 32 * 41, 131072, 0, True),
             ("cross_batch_down", 1, 14336, 4 * 32 * 7, 131072, 1572864, True),
             ("cross_batch_odd", 2, 256, 4 * 32 * 36, 32768, 1048576, True),
             ("cross_batch_many_m", 33, 64, 4 * 32 * 9, 2048, 1048576, True),
             ("cross_batch_large_join", 1, 64, 4 * 32 * 600, 2048, 1048576, True)]
    if a.case:
        unknown = set(a.case) - {c[0] for c in cases}
        if unknown:
            ap.error(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c[0] in a.case]
    for name, m, k, n, tile, staging, pipeline in cases:
        case = a.output / name
        case.mkdir()
        model = case / "model"
        model.mkdir()
        rng = np.random.default_rng(123)
        xq = rng.integers(-127, 128, size=(m, k), dtype=np.int8)
        wq = rng.integers(-127, 128, size=(n, k), dtype=np.int8)
        xq[:, 0] = wq[:, 0] = 127
        x = xq.astype(np.float16) / np.float16(64)
        weight = wq.astype(np.float16) / np.float16(64)
        save_file({"proj.weight": weight}, model / "model.safetensors")
        # Exact binary scales (1/64) make the independent integer reference unambiguous.
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
        spec = json.loads((root / "configs/target.json").read_text())
        spec["npu"].update(gemv_tile_bytes=tile, l2_staging_bytes_per_core=staging,
                           weight_stream_max_slots=8, weight_prefetch_batch_tiles=0,
                           weight_prefetch_cross_batch=False)
        if name == "auto_ring" or name.startswith("cross_batch_"):
            spec["npu"].update(weight_stream_max_slots=0,
                               weight_prefetch_cross_batch=name.startswith("cross_batch_"))
        if name.startswith("cross_batch_"):
            spec["npu"]["weight_prefetch_batch_tiles"] = 6 if name == "cross_batch_ring" else 2
        if name == "cross_batch_large_join":
            spec["npu"]["weight_prefetch_batch_tiles"] = 256
        spec_path = case / "target.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        modes = {}
        for mode in ("serial", "pipeline"):
            out = case / mode
            run([a.compiler.resolve(), source, "--manifest", manifest, "--spec", spec_path, "--force=npu",
                 f"--npu-weight-prefetch={'true' if mode == 'pipeline' else 'false'}", "-o", out],
                root, case / f"{mode}.compile.log")
            io = json.loads((out / "io.json").read_text())
            streams = [s for phase in io["phases"] for s in phase.get("weight_streams", [])]
            if name in ("auto_ring", "cross_batch_ring"):
                assert max(s["slots"] for s in streams) > 8, (name, streams)
            assert json.loads((out / "system.json").read_text())["l2_bytes_required"] <= spec["npu"]["l2_bytes"]
            image = out / "lp6.bin"
            with image.open("r+b") as f:
                arg = next(arg for arg in io["args"] if arg["name"] == "x")
                f.seek(arg["addr"])
                f.write(x.tobytes())
            image_hash = sha256(image)
            run([emu, "--settings", settings, "--lp6-config", a.pim_root / "configs/LPDDR6_PIM_Mobile_4CH.ini",
                 "--system-program", "system.json", "--command-report", "commands.json",
                 "--lp6-image", "lp6.bin", "--lp6-size", "16GiB", "--log-level", "off",
                 "--timeline-out", "timeline.json", "--timeline-merge-gap", "0"], out, out / "run.log", env)
            o = next(o for o in io["outputs"] if o["name"] == "y")
            dump = (out / "sram_dump.core0.bin").read_bytes()
            actual = np.frombuffer(dump, dtype=np.float16, count=m * n, offset=o["l1"]).reshape(m, n)
            assert actual.tobytes() == expected.tobytes(), (name, mode, "independent integer reference mismatch")
            prog = decode_program((out / "program.bin").read_bytes())
            if name == "cross_batch_large_join" and mode == "pipeline":
                assert sum(c["kind"] == "barrier" for c in prog["commands"]) > 2
                assert all(len(c["dependencies"]) <= 255 for c in prog["commands"])
            blocks = [c for c in prog["commands"] if c["kind"] == "core_block"]
            traces = [trace_v2(prog["words"][c["isa_start_word"]:c["isa_start_word"] + c["isa_word_count"]])
                      for c in blocks]
            async_loads = sum(t[0] == "ldma" and t[1] == 10 for tr in traces for t in tr)
            waits = sum(t[0:2] == ("sync", 4) for tr in traces for t in tr)
            assert async_loads == waits
            if name.startswith("cross_batch_") and mode == "pipeline":
                assert async_loads == sum(s["iterations"] - 1 for s in streams if s["prefetch"]), (name, async_loads, streams)
            assert (async_loads > 0) == (pipeline and mode == "pipeline"), (name, mode, async_loads)
            timeline = json.loads((out / "timeline.json").read_text())
            assert timeline["dropped_events"] == 0, "timeline truncated"
            events = timeline["events"]
            overlap = 0
            for core in range(4):
                # Timeline records identify physical cores independently of system command ids.
                ce = [e for e in events if e["lane"].startswith(f"Core {core} / ")]
                matrix = union((e["begin_cycle"], e["end_cycle"]) for e in ce if e["kind"] == "matrix_compute")
                dma = union((e["begin_cycle"], e["end_cycle"]) for e in ce if e["kind"] in ("ldma", "ldma_read"))
                overlap += sum(max(0, min(y, v) - max(x, u)) for x, y in matrix for u, v in dma)
            rep = json.loads((out / "commands.json").read_text())
            modes[mode] = dict(total_cycles=max(c["end_cycle"] for c in rep["commands"]),
                               core_blocks=len(blocks), async_weight_loads=async_loads, event_waits=waits,
                               ldma_matrix_overlap_cycles=overlap, independent_reference_exact=True,
                               input_image_sha256=image_hash, output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                               mma_slices=sum(t[0] == "mma" for tr in traces for t in tr))
            modes[mode]["weight_streams"] = streams
        assert modes["serial"]["input_image_sha256"] == modes["pipeline"]["input_image_sha256"]
        assert modes["serial"]["output_sha256"] == modes["pipeline"]["output_sha256"]
        assert modes["serial"]["mma_slices"] == modes["pipeline"]["mma_slices"]
        if pipeline:
            assert modes["pipeline"]["ldma_matrix_overlap_cycles"] > 0, name
            assert modes["pipeline"]["core_blocks"] < modes["serial"]["core_blocks"], name
        else:
            assert (case / "serial/program.bin").read_bytes() == (case / "pipeline/program.bin").read_bytes()
        results[name] = dict(shape=dict(m=m, n=n, k=k), results=modes,
                             speedup=modes["serial"]["total_cycles"] / modes["pipeline"]["total_cycles"])
        print(name, json.dumps(results[name]), flush=True)
    (a.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
