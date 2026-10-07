#!/usr/bin/env python3
"""Frozen 50-case Up comparison: PA ABI, fixed-ISA F-only controls, legal plans."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import pa_ha_up_pa_abi as pa
import pa_ha_up_k_split as k

h=pa.h
ROOT=h.ROOT/'experiments/pa_ha_up_pa_abi_20261007_v3'
PREVIOUS=h.ROOT/'experiments/pa_ha_up_search_20261006_v2'
OLD=h.ROOT/'experiments/pa_ha_up_projection_20261006_v1'
IDS={'pim_priority':'s1_c5b8_k0_n0_sel','shared_s8':'s8_c5b8_k0_n0_sel012','shared_s16':'s16_c6_k0_n0_sel0123'}
MS=(1,2,4,8,16)


def prepare_job(item):
    label,engine,m,role=item
    dest=ROOT/'cases'/f'{label}_{engine}_m{m}_{role}'
    if (dest/'case.json').exists():return dest
    best=json.loads((PREVIOUS/'final_search.json').read_text())['results']
    if label=='npu_native':
        source=OLD/f'harness_static/npu_m{m}_{engine}'
        if engine=='npu':
            case=json.loads((source/'case.json').read_text())
            native=OLD/f'native_npu/m{m}'
            system=json.loads((native/'system.json').read_text())
            case['details']=dict(endpoint_event=system['command_symbols'][-1]['id'],weight_read_passes=1,kernel='native_nk32')
            case['mapping']['fields_lsb_first']=h.DEFAULT_BITS
            pa.convert(native,dest,fields=h.DEFAULT_BITS,image=OLD/'native.bin',
                reference=source/'reference.npz',settings=OLD/'settings.toml',config=source/'dram.ini',case_override=case)
        else:pa.convert(source,dest)
    else:
        candidate=IDS[label]
        own=next(r for r in best if r['candidate']==candidate and r['M']==m and r['engine']==engine)
        source=Path(own['case_path'])
        if role=='fixed_pa_order':
            fixed=next(r for r in best if r['candidate']==IDS['shared_s8'] and r['M']==m and r['engine']=='npu')
            template=Path(fixed['case_path'])
            case=json.loads((template/'case.json').read_text())
            case['layout']=candidate;case['mapping']=json.loads((source/'case.json').read_text())['mapping']
            pa.convert(template,dest,config=source/'dram.ini',case_override=case)
        else:pa.convert(source,dest)
    case=json.loads((dest/'case.json').read_text())
    case.update(comparison_label=label,comparison_role=role)
    h.save(dest/'case.json',case)
    return dest


def job(item):
    path=prepare_job(item)
    os.environ['LP6PIM_GEMV_STREAM_MODE']='output_increment'
    os.environ['LP6PIM_PA_ABI_TRACE']='1'
    if not (path/'result.json').exists():
        h.run(SimpleNamespace(case=path,emulator=ROOT/'emulator',library=ROOT/'libdramsim3.so',timeout=7200))
    result=json.loads((path/'result.json').read_text())
    audit=json.loads((path/'pa_lowering_audit.json').read_text())
    if item[1]=='pim':
        text=(path/'run.log').read_text()
        assert f"PIM_PA_ABI_V2 instructions={audit['primitive_count']} " in text
    result.update(label=item[0],engine=item[1],M=item[2],role=item[3],case_path=str(path))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,source in [('emulator',h.SIM/'target/release/transactional_emulator'),('libdramsim3.so',h.LP6/'lp6-pim-main/libdramsim3.so')]:
        if not (ROOT/name).exists():shutil.copy2(source,ROOT/name)
    tasks=[]
    for m in reversed(MS):
        tasks.extend([('npu_native',engine,m,'optimized') for engine in ('pim','npu')])
        for label in IDS:
            tasks.append((label,'pim',m,'optimized'))
            tasks.append((label,'npu',m,'fixed_pa_order'))
            if label!='shared_s8':tasks.append((label,'npu',m,'optimized'))
    if a.smoke:tasks=[('npu_native','pim',1,'optimized'),('pim_priority','pim',1,'optimized'),('shared_s16','pim',1,'optimized')]
    h.save(ROOT/'comparison_policy.json',dict(M=list(MS),cases=tasks,native_cases=len(tasks),workers=a.workers,
        hardware='4 NPU cores, 8 SC, 2 ranks, 8 MiB L2; unchanged clock/DRAM timings',
        shared_pa_image_labels=list(IDS),
        native_baseline_has_different_packing=True,
        fixed_control='identical complete NPU binary and identical PA byte image for S1/S8/S16 at each M; only dram.ini F differs',
        pim_control='PA instructions, legal mapping-specific partition/activation/reduction plans; not a pure F-only experiment',
        no_read_view=True,no_hidden_copy=True,scaled_GEMV_v2=False,
        emulator_sha256=h.sha(ROOT/'emulator'),library_sha256=h.sha(ROOT/'libdramsim3.so'),started_unix=time.time()))
    results=[];failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(job,t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            t=futures[f]
            try:
                r=f.result();results.append(r)
                print('PASS',len(results),'/',len(tasks),t,r['latency_us'],flush=True)
            except Exception as e:
                failures.append(dict(case=t,error=repr(e)));print('FAIL',t,repr(e),flush=True)
            h.save(ROOT/('smoke_summary.partial.json' if a.smoke else 'comparison_summary.partial.json'),dict(results=results,failures=failures))
    h.save(ROOT/('smoke_summary.json' if a.smoke else 'comparison_summary.json'),dict(required=len(tasks),completed=len(results),results=results,failures=failures))
    assert not failures and len(results)==len(tasks)


if __name__=='__main__':main()
