#!/usr/bin/env python3
"""Fit the cost-model constants of configs/target.json to simulator runs.

  PIM efficiency: every PIM phase of a hybrid run (build dir with partition.json and
    commands.json, the emulator's command report) against its LdMAC count at efficiency 1 (the formula of estimateGemvGroup):
    efficiency = sum(ideal) / sum(measured - phase_fixed).
  NPU LP6 read bandwidth: an NPU-only run of the same step: weight bytes of the NPU GEMVs over
    (NPU-only time - the hybrid run's NPU phase time), i.e. the GEMV weight streaming alone.
Prints the fitted values; --write updates the spec file."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import phase_times  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("hybrid", type=Path)
ap.add_argument("npu_only", type=Path, nargs="?")
ap.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "configs/target.json")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
spec = json.loads(a.spec.read_text())
P = spec["pim"]


def times(build):
    return phase_times(json.loads((build / "commands.json").read_text()))


def ideal_pim_us(shapes, tokens=1):
    rows = P["pus"] * 16 * P["lanes"]
    parallel = P["partitions"] if P.get("concurrent_ranks", True) else min(P["partitions"], P["channels"])
    load = [0] * parallel
    for n, k in shapes:
        for _ in range((n + rows - 1) // rows):
            load[load.index(min(load))] += k // 256
    return max(load) * 16 * 32 * tokens * P["nlld_ck"] / P["clock_mhz"]


groups = json.loads((a.hybrid / "partition.json").read_text())["groups"]
pim_groups = [g for g in groups if g["target"] == "pim"]
t = times(a.hybrid)
pim_t = [x for x in t if x[1] == "pim"]
assert len(pim_t) == len(pim_groups), (len(pim_t), len(pim_groups))
num = den = 0.0
print(f"{'group':34s} {'ideal us':>9s} {'measured':>9s} {'eff':>6s}")
for g, (name, _, us) in zip(pim_groups, pim_t):
    ideal = ideal_pim_us(g["shapes"])
    num += ideal
    den += us - P["phase_fixed_us"]
    print(f"{name + ' ' + ','.join(w.split('.')[-2] for w in g['weights']):34s} {ideal:9.1f} {us:9.1f} {ideal / (us - P['phase_fixed_us']):6.3f}")
eff = num / den
print(f"PIM efficiency (fit): {eff:.4f}   (spec: {P['efficiency']})")
new = {"pim.efficiency": round(eff, 4)}
if a.npu_only:
    npu_groups = json.loads((a.npu_only / "partition.json").read_text())["groups"]
    wbytes = sum(n * k for g in npu_groups if g["target"] == "npu" for n, k in g["shapes"])
    t_npu = sum(us for n, k, us in times(a.npu_only) if k == "npu" and n != "collect")
    t_glue = sum(us for n, k, us in t if k == "npu" and n != "collect")
    bw = wbytes / ((t_npu - t_glue) * 1e3)
    print(f"NPU GEMV weights {wbytes / 1e6:.1f} MB in {t_npu - t_glue:.0f} us -> {bw:.2f} GB/s   (spec: {spec['npu']['lp6_read_gbps']})")
    new["npu.lp6_read_gbps"] = round(bw, 2)
if a.write:
    for k, v in new.items():
        sec, key = k.split(".")
        spec[sec][key] = v
    a.spec.write_text(json.dumps(spec, indent=2) + "\n")
    print("updated", a.spec)
