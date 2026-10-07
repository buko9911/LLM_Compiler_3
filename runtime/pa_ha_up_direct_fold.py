#!/usr/bin/env python3
"""Paid NPU g4 consumer for one PA weight copy in [N/4,K/8,4,8].

Raw N128 weight bytes are already the compact matrix operand A[32,4*K].
No local weight transpose, gather, assembly, address alias, or read-view is
used. Dynamic INT8 X is expanded by real zero reads and existing COPY_STRIDED
instructions. M16 uses two M8 tiles with two resident B matrices; a raw weight
stripe is read from LP6/L2 once and reused for both matrix operations.

This bounded diagnostic only emits existing ISA. The caller supplies an image
and real PA->HA configuration and performs native execution separately.
Endpoint: INT8 X/W in LP6 -> canonical INT32 Y[M,N] in L2, no scales/quantization.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
from pathlib import Path

import numpy as np

import dual_view_experiment as e

isa = e.isa
STORAGE = 'n4_k8_n4_k8'
WEIGHT_BASE, ZERO_BASE = 128 << 20, 2 << 20
OUTPUT_L2, ZERO_L2, INPUT_L2, WEIGHT_L2 = 0, 1 << 20, e.X_L2, 2 << 20
X_L1, RAW_L1, B_L1, E_L1, C_L1 = 0, 0x40000, 0x100000, 0x200000, 0x240000
CORES, TILE_N, GROUP_LANES, MAX_M_TILE = 4, 128, 4, 8
B_TILE_STRIDE, LDMA_CHUNK = 512 << 10, 256 << 10


def _validate_shape(m, n, k, storage):
    if storage != STORAGE:
        raise ValueError(f'only checked storage {STORAGE!r} is supported')
    if m not in (1, 2, 4, 8, 16):
        raise ValueError('bounded direct fold supports M=1,2,4,8,16')
    if k != 4096 or n <= 0 or n % TILE_N:
        raise ValueError('bounded direct fold requires K=4096 and positive N divisible by128')


def _m_tiles(m):
    return [dict(m_begin=begin, m_size=min(MAX_M_TILE, m-begin),
                 b_l1=B_L1+(begin//MAX_M_TILE)*B_TILE_STRIDE)
            for begin in range(0, m, MAX_M_TILE)]


def _ranges(ranges, capacity):
    for name, start, size in ranges:
        if start < 0 or size <= 0 or start+size > capacity:
            raise ValueError(f'out-of-range allocation {name}: {start}+{size}')
    for i, (name, start, size) in enumerate(ranges):
        for other, other_start, other_size in ranges[i+1:]:
            if max(start,other_start) < min(start+size,other_start+other_size):
                raise ValueError(f'overlapping allocations: {name}, {other}')
    return [dict(name=name,offset=start,bytes=size) for name,start,size in ranges]


def _ldma(op, l1, l2, size):
    if size <= 0 or size % 32:
        raise ValueError('flat LDMA requires a positive 32-byte multiple')
    words = []
    for off in range(0,size,LDMA_CHUNK):
        words += isa.ldma_i(op,l1+off,l2+off,min(LDMA_CHUNK,size-off))
    return words


def _copy2d(op, first, second, row_bytes, rows, source_stride, destination_stride):
    # COPY: first=L1 destination, second=L1 source.
    # STORE: first=L1 source, second=L2 destination.
    if not (row_bytes > 0 and rows > 0 and source_stride >= row_bytes
            and destination_stride >= row_bytes):
        raise ValueError('COPY/STORE requires explicit nonoverlapping positive strides')
    words = [isa.c_set(isa.SET_BYTES,row_bytes),isa.c_set(isa.SET_ROWS,rows),
             isa.c_set(isa.SET_SRC_STRIDE,source_stride),
             isa.c_set(isa.SET_DST_STRIDE,destination_stride)]
    words += isa.load_constant(14,first)+isa.load_constant(15,second)
    words.append(isa.ldma_r(op,14,15))
    return words


def _prepare(m,k):
    words = _ldma(0,X_L1,INPUT_L2,m*k)
    for tile in _m_tiles(m):
        begin, mt, b0 = tile['m_begin'],tile['m_size'],tile['b_l1']
        cols = GROUP_LANES*mt
        b_bytes, e_bytes = GROUP_LANES*k*cols,k*cols
        # Both B and the reusable E scratch are dynamically zeroed with real
        # LDMA from the shared zero payload, which was actually read from LP6.
        words += _ldma(0,b0,ZERO_L2,b_bytes)
        words += _ldma(0,E_L1,ZERO_L2,e_bytes)
        for local_m in range(mt):
            words += _copy2d(9,E_L1+GROUP_LANES*local_m,
                             X_L1+(begin+local_m)*k,1,k,1,cols)
        for lane in range(GROUP_LANES):
            words += _copy2d(9,b0+8*lane*cols+lane,E_L1,
                             8*cols-lane,k//8,8*cols,8*GROUP_LANES*cols)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _compute(m,n,k,n0,slot):
    # The only weight movement before M_LD is these two 256 KiB transfers.
    words = _ldma(0,RAW_L1,slot,TILE_N*k)
    for tile in _m_tiles(m):
        begin,mt,b0 = tile['m_begin'],tile['m_size'],tile['b_l1']
        cols = GROUP_LANES*mt
        words += isa.load_constant(0,b0)+isa.m_ld(0,0,cols,GROUP_LANES*k)
        words += isa.load_constant(1,RAW_L1)+isa.m_ld(2,1,32,GROUP_LANES*k)
        words.append(isa.m_mma(False,False,GROUP_LANES*k//32))
        words += isa.load_constant(2,C_L1)+isa.m_wb(2,2,32,cols)
        words.append(isa.c_sync(isa.SYNC_MATRIX))
        for local_m in range(mt):
            words += _copy2d(5,C_L1+local_m*GROUP_LANES*4,
                             OUTPUT_L2+((begin+local_m)*n+n0)*4,
                             GROUP_LANES*4,32,cols*4,16)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _decoded_core_ops(words):
    """Resolve actual encoded GP/control operands, not the emitter arguments."""
    gp = [0]*16
    control = [0]*5
    result = []
    for name, instr in isa.core_instructions(words):
        w = instr[0]
        if name == 'C_LI':
            gp[(w>>6)&15] = w>>10
        elif name == 'C_LUI':
            gp[(w>>6)&15] = ((w>>10)&0xFFFFF)<<12
        elif name == 'C_ADDI':
            immediate = w>>14
            if immediate & (1<<17):
                immediate -= 1<<18
            gp[(w>>6)&15] = (gp[(w>>10)&15]+immediate)&0xFFFFFFFF
        elif name == 'C_SET':
            control[(w>>6)&7] = gp[(w>>10)&15] if (w>>9)&1 else w>>10
        elif name == 'LDMA.I':
            op = (w>>6)&15
            result.append(dict(kind='ldma',op=op,first=(w>>14)*32,
                               second=(instr[1]&0x3FFFF)*32,bytes=(instr[1]>>18)*32,
                               rows=1,source_stride=(instr[1]>>18)*32,
                               destination_stride=(instr[1]>>18)*32))
        elif name == 'LDMA.R':
            # The runtime replaces a zero stride by row_bytes. The checked
            # emitter never relies on stride=0 broadcast behavior.
            width = control[isa.SET_BYTES]
            result.append(dict(kind='ldma',op=(w>>6)&15,first=gp[(w>>10)&15],
                               second=gp[(w>>14)&15],bytes=width,
                               rows=control[isa.SET_ROWS],
                               source_stride=control[isa.SET_SRC_STRIDE] or width,
                               destination_stride=control[isa.SET_DST_STRIDE] or width))
        elif name == 'M_LD':
            result.append(dict(kind='m_ld',operand=(w>>6)&3,address=gp[(w>>8)&15],
                               mn=((w>>12)&31)+1,k=((w>>17)&0x3FFF)+1,
                               extended_stride=instr[1] if w>>31 else None))
        elif name == 'M_MMA':
            result.append(dict(kind='mma',count=(w>>8)&1023,fp16=bool((w>>6)&1),
                               accumulate=bool((w>>7)&1)))
        elif name == 'M_WB':
            result.append(dict(kind='m_wb',address=gp[(w>>8)&15],dtype=(w>>6)&3,
                               m=((w>>12)&31)+1,n=((w>>17)&31)+1))
        elif name != 'C_SYNC':
            raise AssertionError(f'unexpected instruction in checked direct consumer: {name}')
    return result


def _replay_preparation(m,k,x):
    """CPU execute encoded initializer's actual LDMA addresses/strides."""
    l1 = np.full(4<<20,-79,dtype=np.int8)
    defined = np.zeros(4<<20,dtype=bool)
    source_read_bytes = collections.Counter()
    for op in _decoded_core_ops(_prepare(m,k)):
        assert op['kind'] == 'ldma'
        dst,src,size = op['first'],op['second'],op['bytes']
        if op['op'] == 0:
            if INPUT_L2 <= src < INPUT_L2+m*k:
                data = x.reshape(-1)[src-INPUT_L2:src-INPUT_L2+size]
                assert len(data) == size
                source_read_bytes['X'] += size
            else:
                assert ZERO_L2 <= src and src+size <= ZERO_L2+4*k*4*min(m,8)
                data = np.zeros(size,dtype=np.int8)
                source_read_bytes['zero'] += size
            l1[dst:dst+size] = data
            defined[dst:dst+size] = True
        elif op['op'] == 9:
            for row in range(op['rows']):
                s = src+row*op['source_stride']
                d = dst+row*op['destination_stride']
                assert np.all(defined[s:s+size]) and len(l1[d:d+size]) == size
                l1[d:d+size] = l1[s:s+size].copy()
                defined[d:d+size] = True
        else:
            raise AssertionError('initializer has unsupported DMA operation')
    return l1,defined,dict(source_read_bytes)


def static_algebra(m,k=4096):
    """Full byte-layout / encoded-copy proof plus independent int64 dot."""
    _validate_shape(m,TILE_N,k,STORAGE)
    rng = np.random.default_rng(20261007+m)
    x = rng.integers(-128,128,size=(m,k),dtype=np.int16).astype(np.int8)
    w = rng.integers(-128,128,size=(TILE_N,k),dtype=np.int16).astype(np.int8)
    # Independent logical W[n,k] -> PA ordinal conversion.
    packed = w.reshape(32,4,k//8,8).transpose(0,2,1,3).copy().reshape(-1)
    a = packed.reshape(32,4*k)
    q,j,lane,element = np.indices((32,k//8,4,8))
    wanted_a = w[4*q+lane,8*j+element].reshape(32,4*k)
    assert np.array_equal(a,wanted_a)
    l1,defined,prep_reads = _replay_preparation(m,k,x)
    actual = np.zeros((m,TILE_N),dtype=np.int64)
    output_cover = np.zeros_like(actual,dtype=np.uint8)
    checked_b = 0
    for tile in _m_tiles(m):
        begin,mt,b0 = tile['m_begin'],tile['m_size'],tile['b_l1']
        cols=4*mt
        assert np.all(defined[b0:b0+4*k*cols])
        b=l1[b0:b0+4*k*cols].reshape(4*k,cols)
        expected_b=np.zeros_like(b)
        for l in range(4):
            row_indices=(np.arange(k//8)[:,None]*32+8*l+np.arange(8)).reshape(-1)
            for local_m in range(mt):
                expected_b[row_indices,4*local_m+l]=x[begin+local_m]
        assert np.array_equal(b,expected_b)
        checked_b += b.size
        c=a.astype(np.int64)@b.astype(np.int64)
        for local_m in range(mt):
            actual[begin+local_m]=c[:,4*local_m:4*local_m+4].reshape(-1)
            output_cover[begin+local_m]+=1
    expected=x.astype(np.int64)@w.astype(np.int64).T
    assert np.array_equal(actual,expected) and np.all(output_cover==1)
    return dict(M=m,K=k,group_lanes=4,m_tiles=_m_tiles(m),
                checked_output_elements=actual.size,checked_raw_weight_bytes=packed.size,
                checked_B_bytes=checked_b,A_is_raw_all_bytes_exact=True,
                encoded_initializer_all_B_bytes_exact=True,I64_dot_exact=True,
                output_element_cover_once=True,int32_absolute_bound=k*128*128,
                initializer_L2_bytes_per_core=prep_reads,
                expected_sha256=hashlib.sha256(expected.astype('<i4').tobytes()).hexdigest())


def _audit_compute(words,m,n,k,n0,slot):
    ops=_decoded_core_ops(words)
    moves=[x for x in ops if x['kind']=='ldma']
    loads=[x for x in moves if x['op']==0]
    assert loads==[dict(kind='ldma',op=0,first=RAW_L1+off,second=slot+off,
                        bytes=LDMA_CHUNK,rows=1,source_stride=LDMA_CHUNK,
                        destination_stride=LDMA_CHUNK) for off in [0,LDMA_CHUNK]]
    assert all(x['op'] in (0,5) for x in moves), 'weight copy/assembly is forbidden'
    ml=[x for x in ops if x['kind']=='m_ld']
    mma=[x for x in ops if x['kind']=='mma']
    stores=[x for x in moves if x['op']==5]
    expected_ml=[]
    expected_stores=[]
    for tile in _m_tiles(m):
        begin,mt,b0=tile['m_begin'],tile['m_size'],tile['b_l1']
        expected_ml += [dict(kind='m_ld',operand=0,address=b0,mn=4*mt,k=4*k,extended_stride=None),
                        dict(kind='m_ld',operand=2,address=RAW_L1,mn=32,k=4*k,extended_stride=None)]
        for local_m in range(mt):
            expected_stores.append(dict(kind='ldma',op=5,first=C_L1+local_m*16,
                second=OUTPUT_L2+((begin+local_m)*n+n0)*4,bytes=16,rows=32,
                source_stride=16*mt,destination_stride=16))
    assert ml==expected_ml and stores==expected_stores
    assert mma==[dict(kind='mma',count=4*k//32,fp16=False,accumulate=False)]*len(_m_tiles(m))
    # Each scatter is a contiguous logical Y token range even though its L1
    # source is strided. Coordinates are checked against decoded descriptors.
    coverage=[]
    for s in stores:
        addresses=s['second']+np.arange(32)[:,None]*s['destination_stride']+np.arange(16)
        assert np.all(np.diff(addresses.reshape(-1))==1)
        coverage.append((int(addresses.min()),int(addresses.max())+1))
    return coverage


def generate(path,m,n,k,storage=STORAGE):
    """Generate checked program/system/contract files; never run a simulator."""
    path=Path(path)
    _validate_shape(m,n,k,storage)
    if (path/'program.bin').exists() or (path/'system.json').exists():
        raise ValueError('refusing to overwrite a generated program')
    raw_bytes=TILE_N*k
    mt=min(m,MAX_M_TILE)
    zero_bytes=4*k*4*mt
    l1_ranges=_ranges([('X',X_L1,m*k),('raw_W_is_A',RAW_L1,raw_bytes),
                      *[(f"B_m{t['m_begin']}",t['b_l1'],4*k*4*t['m_size']) for t in _m_tiles(m)],
                      ('initializer_E',E_L1,k*4*mt),('matrix_C',C_L1,32*4*mt*4)],4<<20)
    l2_ranges=_ranges([('canonical_Y',OUTPUT_L2,m*n*4),('zero_constant',ZERO_L2,zero_bytes),
                      ('X',INPUT_L2,m*k),('raw_weight_slots',WEIGHT_L2,8*raw_bytes)],8<<20)
    lp6_ranges=_ranges([('X',e.X_BASE,m*k),('zero_constant',ZERO_BASE,zero_bytes),
                       ('packed_W',WEIGHT_BASE,n*k)],1<<34)
    # Numerical proof precedes filesystem writes.
    algebra=static_algebra(m,k)
    p=e.Program()
    x_ready=p.dma('input.load',[],e.X_BASE,INPUT_L2,m*k)
    zero_ready=p.dma('input.zero_load',[],ZERO_BASE,ZERO_L2,zero_bytes)
    previous=[p.core('input.direct_expand',[x_ready,zero_ready],core,_prepare(m,k))
              for core in range(CORES)]
    history=[[] for _ in range(CORES)]
    tiles=n//TILE_N
    assignments=[]
    for batch in range((tiles+CORES-1)//CORES):
        staged=[]
        for core in range(CORES):
            tile=batch*CORES+core
            if tile>=tiles:
                continue
            slot=WEIGHT_L2+(2*core+batch%2)*raw_bytes
            reuse=[history[core][batch-2]] if batch>=2 else []
            event=p.dma('weight.load',reuse,WEIGHT_BASE+tile*raw_bytes,slot,raw_bytes)
            staged.append((core,tile,slot,event,reuse))
        for core,tile,slot,weight_ready,reuse in staged:
            body=_compute(m,n,k,tile*TILE_N,slot)
            _audit_compute(body,m,n,k,tile*TILE_N,slot)
            event=p.core('npu.direct_fold',[previous[core],weight_ready],core,body)
            previous[core]=event
            history[core].append(event)
            assignments.append(dict(tile=tile,n_begin=tile*TILE_N,core=core,slot=slot,
                weight_event=weight_ready,consumer_event=event,reuse_dependency=reuse))
    final=p.barrier('done',previous)
    p.finish(path)
    d=isa.decode_program((path/'program.bin').read_bytes())
    records={r['id']:r for r in d['commands']}
    out_cover=np.zeros(m*n*4,dtype=np.uint8)
    external_weight_bytes=local_weight_bytes=0
    for a in assignments:
        r=records[a['weight_event']]
        assert r['kind']=='gdma_load' and r['rows']==1 and r['row_bytes']==raw_bytes
        assert r['lp6_address']==WEIGHT_BASE+a['tile']*raw_bytes and r['l2_address']==a['slot']
        assert r['dependencies']==a['reuse_dependency']
        external_weight_bytes+=r['row_bytes']
        c=records[a['consumer_event']]
        assert c['target_logical_core']==a['core'] and a['weight_event'] in c['dependencies']
        body=d['words'][c['isa_start_word']:c['isa_start_word']+c['isa_word_count']]
        cover=_audit_compute(body,m,n,k,a['n_begin'],a['slot'])
        for lo,hi in cover:
            assert OUTPUT_L2<=lo<hi<=OUTPUT_L2+m*n*4
            out_cover[lo-OUTPUT_L2:hi-OUTPUT_L2]+=1
        local_weight_bytes+=raw_bytes
    assert np.all(out_cover==1) and external_weight_bytes==local_weight_bytes==n*k
    assert all(r['kind'] in ('gdma_load','core_block','barrier') for r in records.values())
    assert records[final]['dependencies']==previous
    ancestors={}
    for event,r in records.items():
        ancestors[event]=set(r['dependencies'])
        for dep in r['dependencies']:
            assert dep<event
            ancestors[event]|=ancestors[dep]
    assert ancestors[final]==set(records)-{final}
    contract=dict(schema='lp6.pa_ha.direct_fold.v1',M=m,N=n,K=k,storage=storage,
        scope='INT8 X and one packed INT8 W in LP6 to canonical INT32 Y in L2',
        quantization_included=False,scale_included=False,new_hardware=False,read_view_alias=False,
        actual_dram_mapping_supplied_by_caller=True,group_lanes=4,m_tile_max=8,m_tiles=_m_tiles(m),
        weight_storage_order=['N/4','K/8',4,8],
        physical_weight_byte_formula='W_BASE + ((n//4)*(K//8)+k//8)*32 + (n%4)*8 + k%8',
        input_lp6=e.X_BASE,weight_lp6=WEIGHT_BASE,zero_lp6=ZERO_BASE,zero_lp6_bytes=zero_bytes,
        zero_source_must_be_all_zero=True,output_l2=OUTPUT_L2,
        l1_ranges=l1_ranges,l2_ranges=l2_ranges,lp6_ranges=lp6_ranges,
        l1_peak_bytes=max(r['offset']+r['bytes'] for r in l1_ranges),
        l2_peak_bytes=max(r['offset']+r['bytes'] for r in l2_ranges),
        raw_tile_bytes=raw_bytes,raw_slot_count=8,cores=4,tiles=tiles,
        weight_read_passes=1,LP6_weight_read_bytes=n*k,L2_weight_read_bytes=n*k,
        LP6_input_read_bytes=m*k,LP6_zero_read_bytes=zero_bytes,LP6_output_write_bytes=0,
        packet_assembly_bytes=0,compute_core_COPY_STRIDED_count=0,
        raw_W_direct_M_LD_activation_operand=True,raw_W_reused_across_M_tiles=True,
        initializer='paid_zero_LP6_then_encoded_E_transpose_and_shifted_bulk_copy',
        B_generated_once_per_core=True,B_reused_across_all_weight_tiles=True,
        B_generated_once_per_M_tile_per_core=True,expanded_B_total_bytes_per_core=4*k*4*m,
        initializer_E_live_bytes_per_core=k*4*mt,
        matrix_k=4*k,matrix_rows=32,matrix_columns_per_M_tile=4*mt,
        matrix_operations=tiles*len(_m_tiles(m)),
        matrix_compute_K32_slices=tiles*len(_m_tiles(m))*(4*k//32),
        raw_weight_MU_load_bytes=n*k*len(_m_tiles(m)),
        all_dynamic_operand_and_scatter_costs_included=True,
        LDMA_I_max_chunk_bytes=LDMA_CHUNK,assignments=assignments,
        final_join_event=final,all_commands_ancestors_of_final_join=True,
        all_output_bytes_written_once=True,command_count=len(records),
        program_sha256=e.sha(path/'program.bin'),generator_sha256=e.sha(Path(__file__)),
        dependency_source_sha256={str(Path(e.__file__)):e.sha(Path(e.__file__)),
                                  str(Path(isa.__file__)):e.sha(Path(isa.__file__))},
        static_algebra=algebra,native_validation_performed=False,
        limitations=['The shared memory mapping and image are supplied and checked by the caller.',
                     'M16 reloads the same L1 raw W into MU for the two M8 operations: current ISA releases both operands at final K.',
                     'Full-K operand sizes use existing ISA/simulator behavior; this adds no new hardware buffer capacity.',
                     'This is a diagnostic consumer, not automatic production compiler lowering.'])
    e.save(path/'direct_fold_contract.json',contract)
    return contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--m',required=True,type=int)
    parser.add_argument('--n',type=int,default=14336)
    parser.add_argument('--k',type=int,default=4096)
    parser.add_argument('--storage',default=STORAGE)
    args=parser.parse_args()
    result=generate(args.output,args.m,args.n,args.k,args.storage)
    print(f"Generated {result['command_count']} commands; raw weight assembly0; CPU algebra exact; no native run.")


if __name__=='__main__':
    main()
