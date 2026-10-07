#!/usr/bin/env python3
"""Calibrate a shape-specific layout/device cost model with held-out token counts.

Prototype contract: never extrapolate to other tensor shapes or controller
configurations. Each measured component is a nonnegative affine function of M;
large unseen M and unmeasured layouts require new profiles, not invented speedups.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def affine(ms, values):
    a=np.column_stack([np.ones(len(ms)),ms])
    x=np.linalg.lstsq(a,np.array(values),rcond=None)[0]
    if x[1]<0: x=np.array([np.mean(values),0.0])
    if x[0]<0: x=np.array([0.0,np.dot(ms,values)/np.dot(ms,ms)])
    return dict(fixed_us=float(x[0]),per_token_us=float(x[1]))


def predict(model, m):
    return model["fixed_us"]+model["per_token_us"]*m


def select_per_m(sweep):
    """Measured per-M oracle over the enumerated candidates, not a global optimum.

    A shared candidate is one physical weight layout. Device-specific window
    sizes and NPU batch schedules may differ without repacking that weight.
    """
    if sweep["errors"]:
        raise ValueError("complete the failed sweep cases before selecting optima")
    records=sweep["records"]
    if not records or not all(r["all_outputs_int32_exact"] for r in records):
        raise ValueError("only independently verified profiles may be selected")
    p=sweep["provenance"]
    expected=set()
    for m in p["tokens"]:
        for name in p["candidates"]:
            if name=="npu_native":
                expected.update((m,name,f"npu_{order}_b{b}")
                    for order in ("linear","core_interleaved") for b in p["npu_batch_tiles"])
            else:
                expected.update((m,name,f"pim_w{w}") for w in p["read_windows"])
                expected.update((m,name,f"npu_w{w}_b{b}")
                    for w in p["read_windows"] for b in p["npu_batch_tiles"])
    actual={(r["tokens"],r["candidate"],r["variant"]) for r in records}
    if actual!=expected or len(actual)!=len(records):
        raise ValueError("incomplete or duplicate sweep profiles")
    shapes={(r["n"],r["k"]) for r in records}
    if len(shapes)!=1:raise ValueError("different shapes in one sweep")
    hardware=sweep["provenance"]["emulator_sha256"]
    if any(r["emulator_sha256"]!=hardware for r in records):
        raise ValueError("different simulators in one sweep")
    comparisons=[]
    for m in sorted({r["tokens"] for r in records}):
        samples=[r for r in records if r["tokens"]==m]
        if len({r["output_sha256"] for r in samples})!=1:
            raise ValueError(f"different numerical outputs at M={m}")
        reference={d:min((r for r in samples if r["engine"]==d),key=lambda r:r["latency_us"])
                   for d in ("npu","pim")}
        layouts=[]
        for name in sorted({r["candidate"] for r in samples}-{"npu_native"}):
            engines={d:min((r for r in samples if r["candidate"]==name and r["engine"]==d),
                           key=lambda r:r["latency_us"]) for d in ("npu","pim")}
            if engines["npu"]["physical_layout"]!=engines["pim"]["physical_layout"]:
                raise ValueError("a shared candidate must use exactly one physical layout")
            ratios={d:engines[d]["latency_us"]/reference[d]["latency_us"] for d in engines}
            layouts.append(dict(candidate=name,npu=engines["npu"],pim=engines["pim"],
                normalized_latency=ratios,slowdown_percent={d:100*(v-1) for d,v in ratios.items()},
                worst_normalized_latency=max(ratios.values()),mean_normalized_latency=sum(ratios.values())/2))
        layouts.sort(key=lambda x:(x["worst_normalized_latency"],x["mean_normalized_latency"],x["candidate"]))
        for row in layouts:
            row["pareto"]=not any(
                other["npu"]["latency_us"]<=row["npu"]["latency_us"] and
                other["pim"]["latency_us"]<=row["pim"]["latency_us"] and
                (other["npu"]["latency_us"]<row["npu"]["latency_us"] or other["pim"]["latency_us"]<row["pim"]["latency_us"])
                for other in layouts)
        comparisons.append(dict(tokens=m,npu_optimum=reference["npu"],pim_optimum=reference["pim"],
                                balanced=layouts[0],shared_candidates=layouts))
    return dict(schema="lp6.experimental.dual_view.per_m.v1",shape=list(next(iter(shapes))),
                selection="for each M independently: argmin_L max(T_NPU(L,M)/best_NPU(M), T_PIM(L,M)/best_PIM(M))",
                tie_break="mean normalized latency, then candidate name",comparisons=comparisons,
                measured_runs=len(records),provenance=sweep["provenance"],
                limitations=["Best among measured candidates; not a proof of global optimality.",
                    "Each M can select a different resident layout; repacking during a running workload is not free or measured.",
                    "The shared layout uses the experimental 64 KiB read-view controller and may use research bank-set ping-pong.",
                    "Kernel scope excludes quantization/dequantization and does not predict full-model TPS."])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--summaries",type=Path,nargs="+")
    source.add_argument("--sweep",type=Path,help="select independent per-M minima and balanced shared layouts from a measured sweep")
    ap.add_argument("--calibration-m",type=int,nargs="+",default=[1,8])
    ap.add_argument("--decode-probability",type=float,default=.9)
    ap.add_argument("--large-m",type=int,default=8)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.sweep:
        result=select_per_m(json.loads(a.sweep.read_text()))
        result["source_sweep"]=str(a.sweep.resolve())
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+"\n")
        for row in result["comparisons"]:
            b=row["balanced"]
            print(f"M={row['tokens']}: NPU best {row['npu_optimum']['latency_us']:.3f} us, "
                  f"PIM best {row['pim_optimum']['latency_us']:.3f} us; shared {b['candidate']}: "
                  f"NPU {b['npu']['latency_us']:.3f}, PIM {b['pim']['latency_us']:.3f} us; "
                  f"slowdown {b['slowdown_percent']}")
        return
    assert 0<=a.decode_probability<=1
    if a.large_m < 1 or any(m < 1 for m in a.calibration_m):
        ap.error("token counts must be positive")
    profiles={}
    provenance=[]
    hardware=set()
    for path in a.summaries:
        provenance.append(str(path.resolve()))
        source=json.loads((path.parent/"provenance.json").read_text())
        hardware.add(tuple(source[k] for k in ("emulator_sha256","library_sha256","settings_sha256","dram_config_sha256")))
        for key,row in json.loads(path.read_text()).items():
            mtag,layout=key.split("_",1); m=int(mtag[1:])
            engines={"npu":row} if layout=="npu_native" else {d:row[d] for d in ("pim","npu") if d in row}
            for device,value in engines.items():
                if not value["all_outputs_int32_exact"]:
                    raise ValueError(f"incorrect profile: {layout}/{device}/M={m}")
                if value["emulator_sha256"] != source["emulator_sha256"]:
                    raise ValueError(f"profile/provenance emulator mismatch: {path}")
                entry=profiles.setdefault((layout,device),{})
                if m in entry: raise ValueError(f"duplicate profile: {layout}/{device}/M={m}")
                entry[m]=value
    models={}; validation=[]; shapes=set(); contracts=set()
    for (layout,device),samples in profiles.items():
        for r in samples.values():
            shapes.add((r["n"],r["k"]))
            if r["read_view"]:
                c={k:v for k,v in r["read_view"].items() if k!="views"}
                contracts.add(json.dumps(c,sort_keys=True))
        training=sorted(set(samples)&set(a.calibration_m))
        if len(training)<2: continue
        values=[samples[m]["latency_us"] for m in training]
        model=affine(training,values)
        model.update(calibration_tokens=training,min_tokens=min(training),max_tokens=max(training),
                     phase_models={name:affine(training,[samples[m]["phase_us"][name] for m in training])
                                   for name in samples[training[0]]["phase_us"]})
        models.setdefault(layout,{})[device]=model
        for m,r in sorted(samples.items()):
            if m in training: continue
            estimate=predict(model,m)
            validation.append(dict(layout=layout,device=device,tokens=m,measured_us=r["latency_us"],
                                   predicted_us=estimate,error_percent=100*(estimate/r["latency_us"]-1),
                                   extrapolated=m>max(training) or m<min(training)))
    assert len(shapes)==1,"calibrate each shape separately"
    assert len(hardware)==1,"calibrate each simulator/hardware configuration separately"
    assert len(contracts)<=1,"controller variants need separate calibration"
    selection=[]
    for layout,devices in models.items():
        if set(devices)!={"npu","pim"}: continue
        choices=[]; weighted=0.0
        for m,prob in [(1,a.decode_probability),(a.large_m,1-a.decode_probability)]:
            if not all(c["min_tokens"]<=m<=c["max_tokens"] for c in devices.values()):
                raise ValueError(f"M={m} is outside {layout}'s calibration range; measure additional profiles")
            costs={d:predict(c,m) for d,c in devices.items()}
            device=min(costs,key=costs.get)
            weighted+=prob*costs[device]
            choices.append(dict(tokens=m,device=device,costs_us=costs))
        selection.append(dict(layout=layout,weighted_latency_us=weighted,choices=choices))
    selection.sort(key=lambda x:x["weighted_latency_us"])
    decisions=[]
    for layout,devices in models.items():
        if set(devices)!={"pim","npu"}: continue
        tokens=set(profiles[(layout,"pim")])&set(profiles[(layout,"npu")])-set(a.calibration_m)
        for m in sorted(tokens):
            measured={d:profiles[(layout,d)][m]["latency_us"] for d in devices}
            predicted={d:predict(c,m) for d,c in devices.items()}
            actual=min(measured,key=measured.get); estimated=min(predicted,key=predicted.get)
            decisions.append(dict(layout=layout,tokens=m,measured_device=actual,predicted_device=estimated,correct=actual==estimated))
    result=dict(schema="lp6.experimental.dual_view.cost.v1",shape=list(next(iter(shapes))),
                scope="quantized Up-projection kernel, single resident shared weight copy",
                source_summaries=provenance,controller=json.loads(next(iter(contracts))) if contracts else None,
                hardware_fingerprints=list(next(iter(hardware))),models=models,held_out_validation=validation,
                held_out_device_selection=decisions,layout_selection=selection,
                workload=dict(decode_probability=a.decode_probability,larger_m=a.large_m),
                limitations=["Same N/K and hardware only; profiles do not generalize to down/attention.",
                             "Token count is GEMM M, not sequence-length/whole-model TPS.",
                             "A shared layout requires the experimental controller; production compiler has no such hardware contract."])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(best=selection[0] if selection else None,held_out=validation),indent=2))


if __name__=="__main__": main()
