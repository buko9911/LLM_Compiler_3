#!/usr/bin/env python3
"""Measure a fresh decoder build against saved step inputs and optional outputs.

Use --image-from to reuse an existing NPU-only weight image after verifying
that all planned operations/addresses and I/O agree. Never overwrites a run.
--inputs-only accepts inputs from prepare_inputs.py; it records output correctness
as unvalidated until results are compared independently (never as a passing check).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from decode import Device, EMU, SETTINGS, BACKEND, phase_times


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build", type=Path)
    ap.add_argument("reference", type=Path, help="stepN.npz: six saved inputs and all expected outputs")
    ap.add_argument("--inputs-only", action="store_true", help="run common saved inputs without expected outputs")
    ap.add_argument("--image-from", type=Path, help="existing NPU-only build with identical planned IR")
    ap.add_argument("--settings", type=Path, default=SETTINGS)
    ap.add_argument("--emulator", type=Path, default=EMU, help="specific simulator executable for reproducible replay")
    ap.add_argument("--lp6-config", type=Path, default=BACKEND / "configs/LPDDR6_PIM_Mobile_4CH.ini")
    ap.add_argument("--dramsim-library", type=Path, default=BACKEND / "libdramsim3.so")
    ap.add_argument("--profile", action="store_true", help="capture full resource timeline, timing and VPU profiles")
    ap.add_argument("--timeline-max-events", type=int, default=12_000_000)
    a = ap.parse_args()
    if a.timeline_max_events <= 0:
        ap.error("--timeline-max-events must be positive")
    out = a.build.resolve()
    if any((out / n).exists() for n in ("commands.json", "benchmark.json", "benchmark_started.json")):
        ap.error("build already has a measurement; use a fresh output directory")
    io = json.loads((out / "io.json").read_text())
    if a.image_from:
        old = a.image_from.resolve()
        before = json.loads((old / "io.json").read_text())
        for key in ("args", "outputs", "dense_end", "dims", "tokens", "capacity", "model", "mask_min"):
            if before[key] != io[key]:
                ap.error(f"cannot reuse image: {key} differs")
        def body(path):
            return "\n".join(line for line in path.read_text().splitlines()
                             if not line.startswith("module attributes "))
        if before["pim_commands"] or io["pim_commands"] or body(old / "compiled.mlir") != body(out / "compiled.mlir"):
            ap.error("image reuse requires identical planned NPU-only operations and addresses")
        image = out / "lp6.bin"
        if image.exists():
            ap.error("destination already has lp6.bin")
        shutil.copyfile(old / "lp6.bin", image)
        # PIM row alignment can change unused trailing padding even in an NPU-only image.
        with image.open("r+b") as f:
            f.truncate(io["image"]["bytes"])
    library = a.dramsim_library.resolve()
    # The emulator has DT_RPATH, which takes precedence over LD_LIBRARY_PATH.
    # Preload the selected SONAME and verify the dynamic loader's actual choice.
    os.environ["LD_PRELOAD"] = str(library)
    os.environ["LD_LIBRARY_PATH"] = str(library.parent) + (":" + os.environ["LD_LIBRARY_PATH"]
                                                         if os.environ.get("LD_LIBRARY_PATH") else "")
    emulator = a.emulator.resolve()
    linked = subprocess.check_output(["ldd", str(emulator)], text=True)
    selected = [line.strip() for line in linked.splitlines() if "libdramsim3.so" in line]
    if len(selected) != 1 or str(library) not in selected[0]:
        raise SystemExit(f"dynamic loader did not select {library}: {selected}")
    dev = Device(out, a.settings, a.lp6_config, emulator=emulator)
    execution_env = dev.execution_env()  # reject a layout mismatch before writing inputs
    with np.load(a.reference) as z:
        saved = {k: z[k] for k in z.files}
    for name in ("x", "cos", "sin", "mask"):
        dev.write(name, saved[name])
    for i in range(saved["K"].shape[0]):
        dev.write(f"past_k.{i}", saved["K"][i])
        dev.write(f"past_v.{i}", saved["V"][i])
    provenance = {"reference": str(a.reference.resolve()), "reference_sha256": sha256(a.reference),
                  "emulator": str(emulator), "emulator_sha256": sha256(emulator), "libdramsim3_sha256": sha256(library),
                  "libdramsim3": str(library), "settings_sha256": sha256(dev.settings),
                  "lp6_config_sha256": sha256(a.lp6_config), "program_sha256": sha256(out / "program.bin"),
                  "input_image_sha256": sha256(out / "lp6.bin"),
                  "pim_kernel": json.loads((out / "system.json").read_text()).get("pim_kernel"),
                  "pim_environment": {k: v for k, v in execution_env.items() if k.startswith("LP6PIM_")},
                  "loader_library": selected[0].split(" (")[0],
                  "profile": a.profile, "timeline_max_events": a.timeline_max_events if a.profile else None,
                  "validation_mode": "inputs_only" if a.inputs_only else "byte_exact_reference",
                  "compiler_head": subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                                            "rev-parse", "HEAD"], text=True).strip()}
    (out / "benchmark_started.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"starting {out.name}: same saved inputs, {len(dev.outputs)} outputs", flush=True)
    outputs, report, wall = dev.run(profile=a.profile, timeline_max_events=a.timeline_max_events)
    matches = {} if a.inputs_only else {
        k: outputs[k].shape == saved[k].shape and outputs[k].dtype == saved[k].dtype and
           outputs[k].tobytes() == saved[k].tobytes() for k in outputs}
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in outputs.items() if not np.isfinite(v).all()}
    np.savez(out / "step0.npz", **{k: saved[k] for k in ("x", "cos", "sin", "mask", "K", "V")}, **outputs)
    times = phase_times(report)
    (out / "phase_times.json").write_text(json.dumps(times, indent=2) + "\n")
    decode_us = sum(t for name, _, t in times if name != "collect")
    partition = json.loads((out / "partition.json").read_text())
    weight_bytes = sum(n * k for g in partition["groups"] for n, k in g["shapes"])
    summary = dict(decode_us=decode_us, collect_us=sum(t for n, _, t in times if n == "collect"),
                   total_us=sum(t for _, _, t in times), wall_seconds=wall,
                   npu_clock_mhz=report["npu_clock_mhz"], weight_bytes=weight_bytes,
                   effective_weight_gbps=weight_bytes / decode_us / 1000,
                   output_count=len(outputs), all_outputs_byte_exact=None if a.inputs_only else all(matches.values()),
                   all_outputs_finite=not nonfinite, nonfinite_output_counts=nonfinite,
                   output_matches=matches, logit_argmax=int(outputs["logits"].reshape(-1).argmax()),
                   provenance=provenance)
    if a.profile:
        # The timeline can be several GB; its header contains the truncation count.
        with (out / "timeline.json").open() as f:
            header = f.read(4096)
        dropped = re.search(r'"dropped_events"\s*:\s*(\d+)', header)
        if dropped is None:
            raise RuntimeError("timeline header is missing dropped_events")
        summary["timeline_dropped_events"] = int(dropped[1])
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("output_matches", "provenance")}), flush=True)
    if nonfinite:
        raise SystemExit("nonfinite outputs; timing is not a validated result: " + ", ".join(nonfinite))
    if not all(matches.values()):
        raise SystemExit("outputs differ: " + ", ".join(k for k, same in matches.items() if not same))
    if a.profile and summary["timeline_dropped_events"]:
        raise SystemExit("timeline was truncated; increase --timeline-max-events")


if __name__ == "__main__":
    main()
