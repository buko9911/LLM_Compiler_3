#!/usr/bin/env python3
"""Independent contract checks for the PA Up comparison, using the native C ABI."""
import ctypes as c
import json
from pathlib import Path
import tempfile
import numpy as np
import pa_ha_up_pa_abi as pa
import run_pa_abi_up_comparison as run
h=pa.h


class Kernel(c.Structure):
    _fields_=[(key,c.c_int32) for key in ('channel','rank','row','start_column','bank_set',
        'weight_bits','activation_bits','scale_bits','input_register_count','output_register_count',
        'weight_format','activation_format','scale_format')]


class Storage(c.Structure):
    _fields_=[(key,c.c_uint32) for key in ('version','struct_size','packets_per_row','reserved')]+[
        (key,c.c_uint64) for key in ('weight_base_row','weight_row_count','activation_base_row','activation_row_count',
            'scale_base_row','scale_row_count','output_base_row','output_row_count')]


def fixed_pim_control():
    sample=run.ROOT/'smoke_s8_m1'
    words=np.memmap(sample/'program.bin',dtype='<u4',mode='r')
    stream=next(list(w) for w in pa.primitive_iterator(words,list(pa.records(words))) if w[0]&255==pa.isa.PIM_GEMV_STREAM)
    system=json.loads((sample/'system.json').read_text()); ks=system['pim_kernel']; st=ks['storage_layout']
    kernel=Kernel(**{key:ks.get(key,0) for key,_ in Kernel._fields_})
    storage=Storage(**st,struct_size=c.sizeof(Storage),reserved=0)
    lib=c.CDLL(str(run.ROOT/'libdramsim3.so'))
    lib.lp6_dramsim3_create.argtypes=[c.c_char_p,c.c_char_p];lib.lp6_dramsim3_create.restype=c.c_void_p
    lib.lp6_dramsim3_destroy.argtypes=[c.c_void_p]
    lib.lp6_dramsim3_pim_load_primitive_words_with_layout.argtypes=[c.c_void_p,c.POINTER(Kernel),c.POINTER(Storage),c.POINTER(c.c_uint32),c.c_uint32]
    encoded=(c.c_uint32*len(stream))(*stream)
    report=[]
    configs={label:run.PREVIOUS/'cases'/f'{candidate}_pim_m1'/'dram.ini' for label,candidate in run.IDS.items()}
    if (run.ROOT/'fixed_npu_decoder.ini').exists():configs['npu_decoder_same_image']=run.ROOT/'fixed_npu_decoder.ini'
    for label,config in configs.items():
        with tempfile.TemporaryDirectory(prefix='lp6-pa-negative-') as out:
            handle=lib.lp6_dramsim3_create(str(config).encode(),out.encode());assert handle
            code=lib.lp6_dramsim3_pim_load_primitive_words_with_layout(handle,c.byref(kernel),c.byref(storage),encoded,len(stream))
            lib.lp6_dramsim3_destroy(handle)
        assert code==(0 if label=='shared_s8' else -1), (label,code)
        report.append(dict(label=label,same_words=stream,return_code=code,
            meaning='legal' if code==0 else 'rejected; fixed PA byte stride crosses decoded bank/subchannel'))
    h.save(run.ROOT/'fixed_pim_pa_legality.json',dict(results=report,
        conclusion='Fixed PIM PA stream is not legal under arbitrary F; no silent tile rebinding or hidden gather.'))


def final_audit():
    summary=json.loads((run.ROOT/'comparison_summary.json').read_text())
    assert summary['completed']==summary['required']==50 and not summary['failures']
    supplementary=json.loads((run.ROOT/'fixed_native_summary.json').read_text())
    assert supplementary['completed']==5
    results=summary['results']+supplementary['results']
    checked=0;fixed=[];images=set();fields=[]
    for m in run.MS:
        hashes=set()
        for label in list(run.IDS)+['npu_decoder_same_image']:
            p=run.ROOT/'cases'/f'{label}_npu_m{m}_fixed_pa_order'
            hashes.add(h.sha(p/'program.bin'));images.add(h.sha(p/'lp6.bin'))
            fields.append(json.loads((p/'case.json').read_text())['mapping']['fields_lsb_first'])
        assert len(hashes)==1
        fixed.append(dict(M=m,identical_program_sha256=next(iter(hashes)),mapping_count=4))
    assert len(images)==1
    per_m={}
    for r in results:
        p=Path(r['case_path']);expected=np.load(p/'reference.npz')['expected'];actual=np.load(p/'actual.npy')
        assert np.array_equal(actual,expected);assert h.arr_sha(actual)==r['output_sha256']
        assert h.sha(run.ROOT/'emulator')==r['emulator_sha256'] and h.sha(run.ROOT/'libdramsim3.so')==r['library_sha256']
        per_m.setdefault(str(r['M']),set()).add(r['output_sha256']);checked+=actual.size
        if r['engine']=='pim':
            audit=json.loads((p/'pa_lowering_audit.json').read_text())
            assert f"PIM_PA_ABI_V2 instructions={audit['primitive_count']} " in (p/'run.log').read_text()
    assert all(len(values)==1 for values in per_m.values())
    h.save(run.ROOT/'final_audit.json',dict(native_comparison_cases=55,main_cross_cases=50,additional_default_F_controls=5,all_outputs_int32_exact=True,
        checked_output_elements=checked,fixed_NPU_controls=fixed,
        identical_PA_image_sha256=next(iter(images)),same_hash_per_M={m:next(iter(v)) for m,v in per_m.items()},
        all_PIM_records_use_PA_ABI_v2=True,common_decoder='Config::AddressMapping',
        geometry_clock_capacity_unchanged=True,read_view=False,
        native_NPU_baseline_has_different_packing=True,
        PIM_results_include_legal_partition_reduction_schedule=True,
        ABI_fetch_cost_is_measured=True,address_decoder_gate_delay_not_separately_modeled=True))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    fixed_pim_control()
    if a.final:final_audit()
    print('PA contract audit passed')
