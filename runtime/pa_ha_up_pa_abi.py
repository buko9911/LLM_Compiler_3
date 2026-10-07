#!/usr/bin/env python3
"""Recompile Up PIM records to PA ABI v2; preserve non-PIM ISA and paid costs.

This is a compiler lowering, not a runtime tile alias. The emitted binary has
real uint64 packet PAs and byte strides. The controller independently decodes
and validates them. Non-affine streams fail preparation instead of acquiring
an unmodeled scatter/read-view. Old artifacts are never overwritten.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
import struct
import numpy as np
import pa_ha_up_projection as h
import pa_ha_up_layout_search as layouts

isa = h.isa


def records(words):
    at = 5
    for ordinal in range(int(words[3])):
        start, head = at, int(words[at]); at += 1
        op = head & 63
        if op == isa.PIM:
            count = head >> 14
            if not count: count = int(words[at]); at += 1
            at += ((head >> 6 & 255) + 1) // 2
            begin = at
            for _ in range(count): at += isa.pim_instruction_words(int(words[at]))
            yield start, at, begin, count
        else:
            if op == isa.BARRIER: at += ((head >> 6 & 255) + 1) // 2
            elif op in (isa.GDMA_LOAD, isa.GDMA_STORE):
                at += 2 + (2 if head >> 6 & 1 else 0) + ((head >> 7 & 255) + 1) // 2
            elif op == isa.CORE_BEGIN:
                assert not head >> 6 & 1, 'extended core record needs explicit lowering support'
                extent = int(words[at]); at += 1 + ((extent >> 24) + 1) // 2 + (extent & 0xffffff)
            else: raise ValueError(f'unsupported system record {head:#x}')
            yield start, at, None, 0
    assert at == len(words) - 1 and int(words[at]) == isa.PROGRAM_END


def primitive_iterator(words, sections):
    for _, end, begin, count in sections:
        if begin is None: continue
        at = begin
        for _ in range(count):
            n = isa.pim_instruction_words(int(words[at]))
            yield tuple(int(x) for x in words[at:at+n])
            at += n
        assert at == end


def infer_storage(kernel, primitives):
    maximum = dict(weight=-1, activation=-1, scale=-1, output=-1)
    for words in primitives:
        p = isa.decode_pim_primitive(list(words))
        for key in maximum: maximum[key] = max(maximum[key], p[key])
        if p['opcode'] == isa.PIM_GEMV_STREAM:
            maximum['weight'] = max(maximum['weight'], p['weight'] + (p['inputs']-1)*p['weight_input_stride'] + (p['outputs']-1)*p['weight_output_stride'])
            maximum['activation'] = max(maximum['activation'], p['activation'] + (p['inputs']-1)*p['activation_input_stride'])
    # Match the historical inferred inventory exactly, including its reserved
    # activation rows (tile-count based). These gaps are not re-optimized here.
    counts = {key:(value+64)//64 for key,value in maximum.items()}
    base = kernel['row']; storage = dict(version=1, packets_per_row=64)
    for key in maximum:
        storage[key+'_base_row'] = base; storage[key+'_row_count'] = counts[key]
        base += counts[key]
    return storage


def convert(source: Path, destination: Path, *, fields=None, image=None,
            reference=None, settings=None, config=None, case_override=None):
    destination.mkdir(parents=True, exist_ok=True)
    system = json.loads((source/'system.json').read_text())
    case = dict(case_override) if case_override else json.loads((source/'case.json').read_text())
    fields = fields or case['mapping']['fields_lsb_first']
    words = np.memmap(source/'program.bin',dtype='<u4',mode='r')
    sections = list(records(words))
    has_pim = any(count for _,_,_,count in sections)
    kernel = system.get('pim_kernel')
    audit = dict(schema='lp6.pim_pa_lowering.v2', source=str(source.resolve()),
                 source_program_sha256=h.sha(source/'program.bin'), memory_address_operands='uint64 PA',
                 internal_route='PA -> Config::AddressMapping -> validated local lowering IR -> LP6 commands',
                 inferred_tile_abi_execution=False, primitive_count=0, affine_streams=0,
                 non_pim_records_byte_identical=True, pa_examples=[])
    if has_pim:
        assert kernel['weight_bits']==8 and kernel['activation_bits']==8 and kernel['scale_bits']==0
        storage = kernel.get('storage_layout') or infer_storage(kernel, primitive_iterator(words, sections))
        kernel['storage_layout'] = storage
        @lru_cache(None)
        def local_pa(base, tile):
            packet = kernel['start_column'] + tile
            return layouts.encode_pa(fields, ro=base+packet//64, co=packet%64)
        @lru_cache(None)
        def route_pa(partition, bank_set, bank):
            pu = max(bank,0); global_bank=pu*2+bank_set
            return layouts.encode_pa(fields, ch=kernel['channel']+partition%kernel['subchannels'],
                ra=kernel['rank']+partition//kernel['subchannels'], bg=global_bank//4, ba=global_bank%4)
        @lru_cache(None)
        def stream_strides(weight, inputs, outputs, istride, ostride):
            base = storage['weight_base_row']; first = local_pa(base,weight)
            ins = local_pa(base,weight+istride)-first if inputs>1 else 0
            outs = local_pa(base,weight+ostride)-first if outputs>1 else 0
            assert ins >= 0 and outs >= 0
            for o in range(outputs):
                for i in range(inputs):
                    assert local_pa(base,weight+i*istride+o*ostride)==first+i*ins+o*outs, 'non-affine PA stream; split required'
            return ins,outs
        @lru_cache(maxsize=1000000)
        def lower(old):
            p=isa.decode_pim_primitive(list(old)); op=p['opcode']
            route=route_pa(p['partition'],p['bank_set'],p['bank'])
            kw=dict(single_bank=p['bank']>=0)
            if op==isa.PIM_LOAD_ACTIVATION:
                assert p['activation']%4==0
                pa=route|local_pa(storage['activation_base_row'],p['activation']//4)
                kw['register']=(p['activation']//4)%kernel['input_register_count']
            elif op==isa.PIM_LOAD_SCALE: raise ValueError('scaled Up not in this W8A8 experiment')
            elif op==isa.PIM_CLEAR:
                pa=route|local_pa(storage['weight_base_row'],0);kw['accumulator']=p['accumulator']
            elif op==isa.PIM_STORE:
                pa=route|local_pa(storage['output_base_row'],p['output']);kw['accumulator']=p['accumulator']
            else:
                assert p['scale']==-1 and p['activation_input_stride'] in (0,1)
                pa=route|local_pa(storage['weight_base_row'],p['weight'])
                kw.update(activation_position=p['activation']%(kernel['input_register_count']*4),
                          accumulator=p['accumulator'])
                if op==isa.PIM_GEMV_STREAM:
                    ins,outs=stream_strides(p['weight'],p['inputs'],p['outputs'],p['weight_input_stride'],p['weight_output_stride'])
                    kw.update(inputs=p['inputs'],outputs=p['outputs'],weight_input_byte_stride=ins,
                              weight_output_byte_stride=outs,accumulator_output_stride=p['accumulator_output_stride'])
            return tuple(isa.pim_pa_primitive(op,pa,**kw))
    with (destination/'program.bin').open('wb') as out:
        out.write(words[:5].tobytes())
        for start,end,begin,count in sections:
            if begin is None: out.write(words[start:end].tobytes());continue
            out.write(words[start:begin].tobytes())
            cursor=begin
            buffer=bytearray()
            for _ in range(count):
                n=isa.pim_instruction_words(int(words[cursor]));old=tuple(int(x) for x in words[cursor:cursor+n]);cursor+=n
                new=lower(old)
                buffer.extend(struct.pack('<'+'I'*len(new),*new));audit['primitive_count']+=1
                if new[0]&255==isa.PIM_GEMV_STREAM:audit['affine_streams']+=1
                if len(audit['pa_examples'])<12:audit['pa_examples'].append(isa.decode_pim_pa_primitive(list(new)))
                if len(buffer)>1<<20:out.write(buffer);buffer.clear()
            out.write(buffer)
        out.write(struct.pack('<I',isa.PROGRAM_END))
        size=out.tell()//4;out.seek(8);out.write(struct.pack('<I',size))
    system['program_word_count']=size
    h.save(destination/'system.json',system)
    for name,override in [('lp6.bin',image),('reference.npz',reference),('settings.toml',settings),('dram.ini',config)]:
        src=Path(override) if override else source/name
        target=destination/name
        if name.endswith('.bin') or name.endswith('.npz'):
            if not target.exists():target.symlink_to(src.resolve())
        else:shutil.copy2(src,target)
    case.update(schema='lp6.pa_abi_up.v3',primitive_abi=2 if has_pim else None,
                source_case=str(source.resolve()),program_sha256=h.sha(destination/'program.bin'),
                config_sha256=h.sha(destination/'dram.ini'),source_sha256=h.sha(__file__))
    case['limitations']=[
        'PA primitive ABI v2 is new; NPU core ISA, PU datapath, clocks and geometry are unchanged.',
        'Static programmable common F is assumed; no runtime dual-view or byte permutation.',
        'Scaled GEMV v2 and automatic production F/S selection are not implemented.',
        'Finite legal consumers; no global-optimum or hardware MU-capacity claim.']
    case['details']=dict(case['details'],PA_based_PIM_ABI=has_pim)
    h.save(destination/'case.json',case);h.save(destination/'pa_lowering_audit.json',audit)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();print(convert(a.source,a.destination))
