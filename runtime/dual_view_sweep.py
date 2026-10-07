#!/usr/bin/env python3
"""Measure separate NPU, PIM and balanced shared layouts independently for each M.

Every number comes from a complete INT8-in-LP6 -> INT32-in-L2 simulation.
Host parallelism changes wall time only; each simulator has private images/stats.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import dual_view_experiment as exp


def candidates():
    result = {}
    for tile in (64,128,256,512):
        for ping in (False,True):
            result[f"{'pp' if ping else 'n'}{tile}"] = dict(tile_n=tile,pingpong=ping)
        result[f"jointpp{tile}"] = dict(tile_n=tile,pingpong=True,n_order="joint")
    result["groupedpp256"] = dict(tile_n=256,pingpong=True,n_order="grouped")
    for split in (2,4,8):
        result[f"k{split}pp"] = dict(tile_n=256,pingpong=True,k_splits=split)
    return result


def run_case(out, name, m, w, x, expected, cfg, args):
    n,k=w.shape
    base=out/f"m{m}_{name}"
    base.mkdir(exist_ok=False)
    image=base/"lp6.bin"
    rows=[]
    common=dict(k=k,library=out/"binaries/libdramsim3.so",buffer_bytes=65536,
                reorder_bytes_per_cycle=128,fair_windows=True,timeout=args.timeout)
    def run(device, variant, layout=None, **options):
        path=base/variant
        path.mkdir()
        batch=options.pop("batch_tiles",None)
        order=options.pop("native_order","linear")
        if device=="npu":
            exp.npu_program(path,m,n,k,batch_tiles=batch,native_order=order)
        else:
            exp.pim_program(path,layout,"wide")
        value=exp.execute(path,image,out/"settings.toml",out/"binaries/emulator",
                          expected,layout,engine=device,**common,**options)
        value.update(candidate=name,variant=variant,batch_tiles=batch,native_order=order,
                     physical_layout=asdict(layout) if layout else {"kind":"npu_nk32","order":order},
                     directory=str(path.resolve()))
        exp.save(path/"result.json",value)
        rows.append(value)

    if name=="npu_native":
        packed=w.reshape(n//32,32,k).transpose(0,2,1)
        for order in ("linear","core_interleaved"):
            data=packed if order=="linear" else packed.reshape(4,n//128,k,32).transpose(1,0,2,3)
            with image.open("wb") as f:
                f.seek(exp.X_BASE);f.write(x.tobytes())
                f.seek(exp.W_BASE);f.write(data.copy().tobytes())
            for batch in args.batch_tiles:
                run("npu",f"npu_{order}_b{batch}",batch_tiles=batch,native_order=order)
    else:
        layout=exp.Layout(n,k,m,**cfg)
        layout.pack(image,w,x)
        # Weight bytes remain in this one image for every device/schedule.
        for window in args.windows:
            run("pim",f"pim_w{window}",layout,window_bytes=window)
            for batch in args.batch_tiles:
                run("npu",f"npu_w{window}_b{batch}",layout,window_bytes=window,batch_tiles=batch)
    if not args.keep_images:image.unlink()
    exp.save(base/"summary.json",rows)
    print(f"COMPLETE M={m} {name}: {len(rows)} exact runs",flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--tokens",type=int,nargs="+",default=[1,2,4,8,16])
    ap.add_argument("--cases",nargs="+",default=["npu_native",*candidates()])
    ap.add_argument("--batch-tiles",type=int,nargs="+",choices=[1,2,4],default=[2,4])
    ap.add_argument("--windows",type=int,nargs="+",default=[2048,8192])
    ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--timeout",type=int,default=1800)
    ap.add_argument("--model",type=Path,default=Path("/home/jongjip/models/llama31-8b"))
    ap.add_argument("--synthetic",action="store_true")
    ap.add_argument("--n",type=int,default=512)
    ap.add_argument("--k",type=int,default=2048)
    ap.add_argument("--keep-images",action="store_true")
    a=ap.parse_args()
    if a.workers<1 or not all(1<=m<=32 for m in a.tokens):ap.error("invalid worker or token count")
    if len(a.tokens)!=len(set(a.tokens)) or len(a.cases)!=len(set(a.cases)):ap.error("duplicate cases/tokens")
    if not set(a.cases)<={"npu_native",*candidates()}:ap.error("unknown layout candidate")
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/"binaries").mkdir();(out/"sources").mkdir()
    emulator=exp.SIM/"target/release/transactional_emulator"
    library=exp.LP6/"lp6-pim-main/libdramsim3.so"
    shutil.copy2(emulator,out/"binaries/emulator")
    shutil.copy2(library,out/"binaries/libdramsim3.so")
    for p in (Path(__file__),Path(exp.__file__)):
        shutil.copy2(p,out/"sources"/p.name)
    settings=re.sub(r"(?m)^num_cores\s*=\s*\d+","num_cores = 4",(exp.SIM.parent/"plena_settings.toml").read_text())
    (out/"settings.toml").write_text(settings)
    dram=exp.LP6/"lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini"
    provenance=dict(schema="lp6.experimental.dual_view.sweep.v1",argv=sys.argv,
        emulator_sha256=exp.sha(out/"binaries/emulator"),library_sha256=exp.sha(out/"binaries/libdramsim3.so"),
        settings_sha256=exp.sha(out/"settings.toml"),dram_config_sha256=exp.sha(dram),
        source_sha256={p.name:exp.sha(p) for p in (out/"sources").iterdir()},
        scope="INT8 input in LP6 to INT32 output in L2; one Llama layer-0 Up projection",
        tokens=a.tokens,candidates={name:candidates().get(name) for name in a.cases},
        npu_batch_tiles=a.batch_tiles,read_windows=a.windows,buffer_bytes=65536,
        scheduling="fair windows",selection="independent per M; minimize maximum normalized device latency")
    exp.save(out/"provenance.json",provenance)
    w=exp.load_weight(a.model,out,a.synthetic,a.n,a.k)
    n,k=w.shape
    inputs={m:np.random.default_rng(4096).integers(-127,128,(m,k),dtype=np.int8) for m in a.tokens}
    reference={m:x.astype(np.int32)@w.astype(np.int32).T for m,x in inputs.items()}
    start=time.monotonic();all_rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(run_case,out,name,m,w,inputs[m],reference[m],candidates().get(name),a):(m,name)
              for m in a.tokens for name in a.cases}
        for job in as_completed(jobs):
            m,name=jobs[job]
            try:all_rows+=job.result()
            except Exception as error:
                errors.append(dict(tokens=m,candidate=name,error=repr(error)))
                print(f"FAILED M={m} {name}: {error}",flush=True)
            exp.save(out/"summary.partial.json",dict(records=all_rows,errors=errors))
    assert exp.sha(dram)==provenance["dram_config_sha256"],"DRAM config changed during sweep"
    result=dict(provenance=provenance,records=sorted(all_rows,key=lambda r:(r["tokens"],r["candidate"],r["variant"])),
                errors=errors,wall_seconds=time.monotonic()-start)
    exp.save(out/"summary.json",result)
    print(json.dumps(dict(runs=len(all_rows),errors=errors,wall_seconds=result["wall_seconds"])),flush=True)
    if errors:raise SystemExit(1)


if __name__=="__main__":main()
