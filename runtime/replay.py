#!/usr/bin/env python3
"""Replay the inputs of a saved step (stepN.npz from decode.py) on another build (e.g. one
compiled with --debug-collect) and save every collected buffer to <build>/replay.npz."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import Device, phase_times  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("build", type=Path)
ap.add_argument("inputs", type=Path)
a = ap.parse_args()
dev = Device(a.build)
z = np.load(a.inputs)
for n in ("x", "cos", "sin", "mask"):
    dev.write(n, z[n])
for i in range(z["K"].shape[0]):
    dev.write(f"past_k.{i}", z["K"][i]); dev.write(f"past_v.{i}", z["V"][i])
outs, rep, wall = dev.run()
np.savez(a.build / "replay.npz", **{k.replace(".", "_"): v for k, v in outs.items()})
for n, k, t in phase_times(rep):
    print(f"{n:10s} {k:4s} {t:9.1f} us")
print(f"wall {wall:.0f}s")
