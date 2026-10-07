#!/usr/bin/env python3
"""Default F with the SAME n4 PA image/ISA as the three fixed-order controls."""
import concurrent.futures
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pa_ha_up_pa_abi as pa
import run_pa_abi_up_comparison as r
h=pa.h


def job(m):
    best=json.loads((r.PREVIOUS/'final_search.json').read_text())['results']
    source=Path(next(v['case_path'] for v in best if v['candidate']==r.IDS['shared_s8'] and v['engine']=='npu' and v['M']==m))
    dest=r.ROOT/'cases'/f'npu_decoder_same_image_npu_m{m}_fixed_pa_order'
    if not (dest/'case.json').exists():
        case=json.loads((source/'case.json').read_text());case['layout']='npu_decoder_same_image'
        case['mapping']=dict(fields_lsb_first=h.DEFAULT_BITS,read_view=False)
        pa.convert(source,dest,config=r.ROOT/'fixed_npu_decoder.ini',case_override=case)
        case=json.loads((dest/'case.json').read_text())
        case.update(comparison_label='npu_decoder_same_image',comparison_role='fixed_pa_order')
        h.save(dest/'case.json',case)
    os.environ['LP6PIM_GEMV_STREAM_MODE']='output_increment'
    if not (dest/'result.json').exists():
        h.run(SimpleNamespace(case=dest,emulator=r.ROOT/'emulator',library=r.ROOT/'libdramsim3.so',timeout=7200))
    result=json.loads((dest/'result.json').read_text())
    result.update(label='npu_decoder_same_image',role='fixed_pa_order',case_path=str(dest));return result


if __name__=='__main__':
    h.Mapping('npu').config(r.ROOT/'fixed_npu_decoder.ini')
    h.save(r.ROOT/'fixed_native_policy.json',dict(M=list(r.MS),required=5,
        objective='Add default F to identical NPU binary and n4 PA byte-image controls; not a different packing/kernel baseline.',
        prospective=True,not_search_or_posthoc_best_selection=True))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(job,r.MS):
            results.append(result);print('PASS',result['M'],result['latency_us'],flush=True)
            h.save(r.ROOT/'fixed_native_summary.partial.json',dict(results=results))
    h.save(r.ROOT/'fixed_native_summary.json',dict(completed=len(results),results=results))
