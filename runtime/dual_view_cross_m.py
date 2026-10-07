#!/usr/bin/env python3
"""Freeze the layout selected at each source M and replay it at every target M."""
import argparse, concurrent.futures, copy, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import dual_view_experiment as e

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=e.LP6/'lp6-pim-main/docs/evidence/2026-10-03/dual_view_per_m/comparison.json'
    prior=json.loads(src.read_text());shutil.copy2(src,out/'frozen_selections.json')
    base=e.ROOT/'experiments/dual_view_cross_full_20261003'
    settings=out/'settings.toml';shutil.copy2(base/'settings.toml',settings)
    emulator=base/'binaries/transactional_emulator';library=base/'binaries/libdramsim3.so'
    w=e.load_weight(Path('/home/jongjip/models/llama31-8b'),out);n,k=w.shape
    wi=w.astype(np.int32);expected={};inputs={}
    for m in (1,2,4,8,16):
        inputs[m]=np.random.default_rng(4096).integers(-127,128,(m,k),dtype=np.int8)
        expected[m]=inputs[m].astype(np.int32)@wi.T
    del wi
    images={};jobs=[];selection={}
    for c in prior['comparisons']:
        sm=c['tokens'];b=c['balanced'];frozen=copy.deepcopy(b['npu']['physical_layout']);frozen.pop('tokens')
        selection[sm]=dict(candidate=b['candidate'],weight_layout=frozen,npu_window=b['npu']['read_view']['window_bytes'],
                           pim_window=b['pim']['read_view']['window_bytes'],batch_tiles=b['npu']['batch_tiles'])
        signature=json.dumps(frozen,sort_keys=True)
        for tm in (1,2,4,8,16):
            l=e.Layout(**frozen,tokens=tm);key=(signature,tm)
            if key not in images:
                image=out/f'image_{len(images)}.bin';l.pack(image,w,inputs[tm]);images[key]=image
            for engine in ('npu','pim'):
                path=out/f'optimized_m{sm}'/f'evaluated_m{tm}'/engine;path.mkdir(parents=True)
                jobs.append((sm,tm,engine,l,images[key],path,selection[sm]))
    e.save(out/'provenance.json',dict(source_sha256=e.sha(src),emulator_sha256=e.sha(emulator),library_sha256=e.sha(library),
        codegen_sha256=e.sha(__file__),workers=args.workers,source_M=[1,2,4,8,16],target_M=[1,2,4,8,16],
        frozen_selections=selection,unique_physical_weight_layouts=len({json.dumps(s['weight_layout'],sort_keys=True) for s in selection.values()}),
        image_sha256={str(p.name):e.sha(p) for p in images.values()}))
    def run(job):
        sm,tm,engine,l,image,path,s=job
        if engine=='npu':e.npu_program(path,tm,n,k,batch_tiles=s['batch_tiles'])
        else:e.pim_program(path,l)
        r=e.execute(path,image,settings,emulator,expected[tm],l,engine=engine,k=k,timeout=1800,
                    window_bytes=s[engine+'_window'],fair_windows=True,library=library)
        baseline=next(c for c in prior['comparisons'] if c['tokens']==tm)['balanced'][engine]
        assert r['output_sha256']==baseline['output_sha256']
        # In this dataset all source-M selections are identical; replay must be cycle-identical too.
        assert r['latency_us']==baseline['latency_us'],(sm,tm,engine,r['latency_us'],baseline['latency_us'])
        r.update(source_M=sm,target_M=tm,candidate=s['candidate'],directory=str(path),cycle_identical_to_prior=True)
        e.save(path/'result.json',r)
        return r
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result());e.save(out/'summary.partial.json',results)
            print(f'PASS {len(results)}/50',flush=True)
    results.sort(key=lambda r:(r['source_M'],r['target_M'],r['engine']))
    e.save(out/'summary.json',results)
    for image in images.values():image.unlink()
if __name__=='__main__':main()
