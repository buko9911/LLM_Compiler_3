#!/usr/bin/env python3
"""Executable K-split Up experiments. All activation and reduction copies paid.

One n4/k8 weight copy and one static PA->HA function per case. This diagnostic
uses the existing ISA and simulator; it does not add an address read-view or a
global PIM reducer. Case execution independently checks every INT32 output.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pa_ha_up_projection as h
import pa_ha_up_direct_fold as direct
import pa_ha_dma_schedule as schedule
import pa_ha_up_layout_search as layouts

isa = h.isa
N, K = h.N, h.K
ROOT = h.ROOT / 'experiments/pa_ha_up_search_20261006_v2'
OLD = h.ROOT / 'experiments/pa_ha_up_projection_20261006_v1'
LIBRARY = h.LP6 / 'lp6-pim-main/build/research_pa_ha_20261006/address_bits_v1/source/libdramsim3.so'


def compact_gather(sources, destinations, width):
    """Exact COPY_STRIDED shapes, with repeated affine shapes in ISA loops.

    Looping reduces encoded size, not simulated work. A stride of zero is
    never interpreted as free broadcast. Check actual effective descriptors.
    """
    sources = np.asarray(sources, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)
    runs = list(h.affine_runs(sources, destinations, width))
    cursor = 0
    for src, dst, size, rows, ss, ds in runs:
        offsets = (np.arange(rows)[:, None] * ss + np.arange(size // width)[None, :] * width).ravel()
        wanted = sources[cursor:cursor+len(offsets)]
        assert np.array_equal(src + offsets, wanted)
        offsets = (np.arange(rows)[:, None] * ds + np.arange(size // width)[None, :] * width).ravel()
        assert np.array_equal(dst + offsets, destinations[cursor:cursor+len(offsets)])
        cursor += len(offsets)
    assert cursor == len(sources)
    body, i = [], 0
    while i < len(runs):
        src, dst, size, rows, ss, ds = runs[i]
        j = i + 1
        delta_src = delta_dst = 0
        if j < len(runs) and runs[j][2:] == runs[i][2:]:
            delta_src, delta_dst = runs[j][0]-src, runs[j][1]-dst
            # ISA C_ADDI has an unsigned immediate. Descending addresses must
            # start a new constant-based descriptor, never encode subtraction.
            if 0 <= delta_src < (1 << 17) and 0 <= delta_dst < (1 << 17):
                j += 1
                while j < len(runs) and j-i < 65535 and runs[j][2:] == runs[i][2:] and runs[j][0]-runs[j-1][0] == delta_src and runs[j][1]-runs[j-1][1] == delta_dst:
                    j += 1
        body += [isa.c_set(isa.SET_BYTES,size), isa.c_set(isa.SET_ROWS,rows),
                 isa.c_set(isa.SET_SRC_STRIDE,ss), isa.c_set(isa.SET_DST_STRIDE,ds)]
        body += h.const(14,dst) + h.const(15,src)
        if j-i > 1:
            body += [isa.c_loop_begin(j-i), isa.ldma_r(9,14,15),
                     isa.c_addi(14,14,delta_dst), isa.c_addi(15,15,delta_src), isa.c_loop_end()]
        else:
            body += [isa.ldma_r(9,14,15)]
        i = j
    return body


class PlanMapping:
    def __init__(self, plan):
        self.plan = plan
        self.bits = plan['fields_lsb_first']
        assert sorted(b for bs in self.bits.values() for b in bs) == list(range(5,34))
        assert all(len(self.bits[f]) == width for f,width in h.FIELDS.items())

    def encode(self, *, ch=0, bg=0, ba=0, co=0, ro=0, ra=0):
        result = 0
        for field, value in dict(ch=ch,bg=bg,ba=ba,co=co,ro=ro,ra=ra).items():
            for i,bit in enumerate(self.bits[field]):
                result = result | (((value >> i)&1) << bit)
        return result

    def packet(self, row, tile, pu, sc):
        return self.encode(ch=sc,bg=pu//2,ba=2*(pu%2),ro=row+tile//64,co=tile%64)

    def config(self, path):
        text = (h.LP6 / 'lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini').read_text()
        assert not re.search(r'(?m)^address_bits_',text)
        lines = '\n'.join('address_bits_'+f+' = '+','.join(map(str,self.bits[f])) for f in h.FIELDS)
        text,count = re.subn(r'(?m)^(address_mapping\s*=.*)$',r'\1\n'+lines,text)
        assert count == 1
        Path(path).write_text(text)


def tile(plan, output, k8):
    lk = plan['local_k8_count']
    order = plan['order']
    if order == 'c6':
        return output*lk+k8
    if order == 'c5b8':
        return (output//8)*(8*lk)+(k8//32)*256+(output%8)*32+k8%32
    raise ValueError(order)


def activation(p, plan, mapping, token, actrow, dependency):
    """K8 inputs are gathered at 8-byte resolution, including strided K."""
    lk = plan['local_k8_count']
    packets = []
    for unit in plan['units']:
        klist = plan['k_partitions'][unit['k_partition']]['k8_indices']
        for pk in range(lk//4):
            address = mapping.packet(actrow,pk,unit['pu'],unit['sc'])
            packets.append((int(address),[int(klist[4*pk+j])*8 for j in range(4)]))
    packets.sort()
    assert len({a for a,_ in packets}) == len(packets)
    chunk = len(packets)//4
    packed_events = []
    for core in range(4):
        entries = packets[core*chunk:(core+1)*chunk]
        sources = np.array([s for _,ss in entries for s in ss],dtype=np.int64)
        target = 0x40000 + np.arange(len(sources),dtype=np.int64)*8
        body = h.copy2d(0,0,h.X_L2+token*K,K)
        body += compact_gather(sources,target,8)
        body += h.copy2d(1,0x40000,h.RAW_L2+core*chunk*32,chunk*32)
        packed_events.append(p.core('activation.k_split_pack',[dependency],core,body))
    ready = p.barrier('activation.packed',packed_events)
    stores = h.packet_dmas(p,'activation.scatter',[ready],np.array([a for a,_ in packets]),h.RAW_L2,load=False)
    return p.barrier('activation.ready',stores),len(packets)*32


def reduce_outputs(p, plan, deps, packets, token):
    share, splits = N//4,plan['split_factor']
    by_core = [[] for _ in range(4)]
    for pa,values in packets:
        owners = {n//share for n,_,_ in values}
        # A packet may span logical ownership boundaries. Charge its full read
        # separately for each consuming core, then gather only owned values.
        for owner in owners:
            by_core[owner].append((pa,[v for v in values if v[0]//share == owner]))
    done,raw_next,reads_bytes = [],h.RAW_L2,0
    for core,entries in enumerate(by_core):
        entries.sort(key=lambda v:v[0])
        addresses = np.array([a for a,_ in entries],dtype=np.int64)
        assert len(np.unique(addresses)) == len(addresses)
        size = len(entries)*32
        assert size <= 2 << 20 and raw_next+size <= 8 << 20
        reads = h.packet_dmas(p,'partial.packet_gather',deps,addresses,raw_next,load=True)
        ready = p.barrier('partial.raw_ready',reads)
        pairs = sorted((split*share+n%share,pi*32+off)
                       for pi,(_,vs) in enumerate(entries) for n,off,split in vs)
        assert [t for t,_ in pairs] == list(range(splits*share))
        canonical = 2 << 20
        assert canonical+splits*share*4 <= 4 << 20
        body = h.copy2d(0,0,raw_next,size)
        body += compact_gather(np.array([s for _,s in pairs]),canonical+np.arange(len(pairs))*4,4)
        if splits > 1:
            body += [isa.c_set(isa.SET_VL,16)] + h.const(0,canonical)
            body += [isa.c_loop_begin(share//16),isa.v_mem(False,isa.I32,0,0,0)]
            for part in range(1,splits):
                body += [isa.v_mem(False,isa.I32,1,0,part*share*4),isa.v_op(0,isa.I32,0,0,1)]
            body += [isa.v_mem(True,isa.I32,0,0,0),isa.c_addi(0,0,64),isa.c_loop_end(),isa.c_sync(isa.SYNC_VPU)]
        body += h.copy2d(1,canonical,(token*N+core*share)*4,share*4)
        done.append(p.core('npu.partial_reduce' if splits > 1 else 'output.reorder',[ready],core,body))
        raw_next += size
        reads_bytes += size
    return p.barrier('token.done',done),reads_bytes,raw_next


def pim_program(path, m, plan):
    mapping, p = PlanMapping(plan),h.Program()
    lk,lo = plan['local_k8_count'],plan['local_output_groups']
    assert lk%4 == 0 and lo%8 == 0
    assert len(plan['units']) == 64
    actrow = h.WEIGHT_ROW + plan['weight_rows'] + 8
    outrow = actrow + (lk+63)//64 + 8
    previous = p.dma('input.load',[],h.X_BASE,h.X_L2,m*K)
    activation_bytes = partial_bytes = 0
    raw_peak = 0
    units = {(u['sc'],u['pu']):u for u in plan['units']}
    for token in range(m):
        adep,actbytes = activation(p,plan,mapping,token,actrow,previous)
        activation_bytes += actbytes
        commands,packets = [],[]
        for sc in range(8):
            def emit(op,**kw):
                return isa.pim_primitive(op,sc,**kw)
            for start in range(0,lo,8):
                commands += [emit(isa.PIM_CLEAR,accumulator=i) for i in range(8)]
                for k0 in range(0,lk,32):
                    count = min(32,lk-k0)
                    commands += [emit(isa.PIM_LOAD_ACTIVATION,activation=k0+i) for i in range(0,count,4)]
                    commands.append(emit(isa.PIM_GEMV_STREAM,weight=int(tile(plan,start,k0)),activation=k0,
                        accumulator=0,inputs=count,outputs=8,weight_input_stride=1,
                        weight_output_stride=lk if plan['order']=='c6' else 32,
                        activation_input_stride=1,accumulator_output_stride=1))
                for pair in range(4):
                    out = start//2+pair
                    commands.append(emit(isa.PIM_STORE,accumulator=pair*2,output=out))
                    for pu in range(8):
                        unit = units[(sc,pu)]
                        ng = plan['output_shards'][unit['output_shard']]['n4_groups']
                        vals = [(int(ng[start+2*pair+parity])*4+lane,lane*8+parity*4,unit['k_partition'])
                                for lane in range(4) for parity in range(2)]
                        packets.append((int(mapping.packet(outrow,out,pu,sc)),vals))
        pd = p.add('pim.compute',lambda event:isa.pim_record(event,[adep],commands))
        previous,readbytes,peak = reduce_outputs(p,plan,[pd],packets,token)
        partial_bytes += readbytes
        raw_peak = max(raw_peak,peak)
    endpoint = p.barrier('done',[previous])
    kernel = h.kernel_spec()
    kernel['storage_layout'] = dict(version=1,packets_per_row=64,
        weight_base_row=h.WEIGHT_ROW,weight_row_count=plan['weight_rows'],
        activation_base_row=actrow,activation_row_count=(lk//4+63)//64,
        scale_base_row=actrow+(lk//4+63)//64,scale_row_count=0,
        output_base_row=outrow,output_row_count=(lo//2+63)//64)
    p.finish(path,kernel)
    return dict(endpoint_event=endpoint,k_splits=plan['split_factor'],all_K_local=plan['split_factor']==1,
                activation_row=actrow,output_row=outrow,activation_scatter_payload_bytes=activation_bytes,
                partial_read_payload_bytes=partial_bytes,nominal_partial_bytes=m*N*plan['split_factor']*4,
                npu_reduction_additions=m*N*(plan['split_factor']-1),
                partials_streamed_one_token_at_a_time=True,l2_raw_peak_end=raw_peak,
                lp6_internal_weight_bytes=m*N*K,PU_count=64,global_PIM_reduction=False,
                all_reduction_and_permutation_paid=True,token_dependency_serial=True)


def make_common(root=ROOT):
    root.mkdir(parents=True,exist_ok=True)
    common = root/'common'
    common.mkdir(exist_ok=True)
    image = common/'lp6.bin'
    if image.exists():
        return common
    w = np.load(OLD/'weight.npy')
    x = np.random.default_rng(4096).integers(-127,128,(16,K),dtype=np.int8)
    packed = np.ascontiguousarray(w.reshape(N//4,4,K//8,8).transpose(0,2,1,3))
    assert np.array_equal(packed.transpose(0,2,1,3).reshape(N,K),w)
    with image.open('wb') as f:
        f.truncate(h.W_BASE+N*K)
        f.seek(h.W_BASE); f.write(packed.tobytes())
        f.seek(h.X_BASE); f.write(x.tobytes())
    expected = x.astype(np.int32) @ w.astype(np.int32).T
    for m in (1,2,4,8,16):
        np.savez(common/f'reference_m{m}.npz',x=x[:m],expected=expected[:m])
    h.save(common/'provenance.json',dict(weight_source=str(OLD/'weight.npy'),weight_sha256=h.arr_sha(w),
           storage_sha256=h.arr_sha(packed),storage='n4_k8_n4_k8',weight_bytes=N*K,weight_copies=1,
           all_storage_bytes_inverse_exact=True,X_seed=4096,source_sha256=h.sha(__file__)))
    return common


def prepare(plan,engine,m,root=ROOT,window=4096):
    path = root/'cases'/f"{plan['id']}_{engine}_m{m}"
    if (path/'case.json').exists():
        return path
    path.mkdir(parents=True,exist_ok=True)
    common = make_common(root)
    for target,source in [('lp6.bin','lp6.bin'),('reference.npz',f'reference_m{m}.npz')]:
        dest = path/target
        if not dest.exists():
            dest.symlink_to(common/source)
    mapping = PlanMapping(plan)
    mapping.config(path/'dram.ini')
    (path/'settings.toml').write_text(re.sub(r'(?m)^num_cores\s*=\s*\d+','num_cores = 4',(h.SIM.parent/'plena_settings.toml').read_text()))
    if engine == 'pim':
        details = pim_program(path,m,plan)
    else:
        staging = path/'unscheduled'
        contract = direct.generate(staging,m,N,K)
        audit = schedule.rewrite(staging,path,path/'dram.ini',h.W_BASE,N*K,window)
        decoded = isa.decode_program((path/'program.bin').read_bytes())
        endpoint = decoded['commands'][-1]['id']
        assert decoded['commands'][-1]['kind'] == 'barrier'
        details = dict(endpoint_event=endpoint,k_splits=0,weight_read_passes=1,
                       weight_assembly_bytes=0,dynamic_input_expand_paid=True,M_tiles=contract['m_tiles'],
                       raw_weight_MU_load_bytes=contract['raw_weight_MU_load_bytes'],
                       DMA_window_bytes=window,dma_schedule_source=str(staging))
    reference = np.load(common/f'reference_m{m}.npz')
    prov = json.loads((common/'provenance.json').read_text())
    h.save(path/'plan.json',plan)
    h.save(path/'case.json',dict(schema='lp6.pa_ha_up_search.v2',M=m,N=N,K=K,layout=plan['id'],engine=engine,
        mapping=dict(fields_lsb_first=plan['fields_lsb_first'],read_view=False),details=details,
        measured_boundary='I8 X in LP6 -> canonical I32 Y in L2',weight_copies=1,weight_base=h.W_BASE,
        weight_bytes=N*K,weight_sha256=prov['weight_sha256'],storage_sha256=prov['storage_sha256'],
        input_sha256=h.arr_sha(reference['x']),expected_sha256=h.arr_sha(reference['expected']),
        program_sha256=h.sha(path/'program.bin'),config_sha256=h.sha(path/'dram.ini'),
        source_sha256=h.sha(__file__),read_view=False,new_hardware=False,
        limitations=['Finite candidates; no global-optimum claim.','Fixed programmable PA->HA mapping is assumed as in v1.',
                     'Existing hardware and ISA; not an automatic production compiler pass.']))
    return path


def run(path,timeout=7200):
    # Pin GEMV mode: never inherit an alternative output-fixed semantics.
    os.environ['LP6PIM_GEMV_STREAM_MODE'] = 'output_increment'
    h.run(SimpleNamespace(case=Path(path),emulator=OLD/'emulator',library=LIBRARY,timeout=timeout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','common'))
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--engine',choices=('npu','pim'))
    parser.add_argument('--m',type=int,choices=(1,2,4,8,16))
    parser.add_argument('--case',type=Path)
    args=parser.parse_args()
    if args.command=='common':
        print(make_common())
    elif args.command=='prepare':
        print(prepare(json.loads(args.plan.read_text()),args.engine,args.m))
    else:
        run(args.case)


if __name__ == '__main__':
    main()
