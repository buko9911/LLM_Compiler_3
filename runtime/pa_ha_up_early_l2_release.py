#!/usr/bin/env python3
"""Release an L2 weight slot after its last prefetch reader completes.

All core instructions and memory addresses remain identical to direct_prefetch.
Only GDMA slot-reuse dependencies change. For tile t>=3, the prior occupant
t-2 was copied to L1 by core block t-3, whose final SYNC_ALL completed that
copy. Its later computation in block t-2 only reads the separate L1 buffer.
"""
from pathlib import Path
import hashlib
import struct
import pa_ha_up_direct_prefetch as pref

e,isa=pref.e,pref.isa


def generate(path,m,n,k):
    path=Path(path)
    original=path/'conservative'
    contract=pref.generate(original,m,n,k)
    decoded=isa.decode_program((original/'program.bin').read_bytes())
    core_events={(a['core'],a['local_tile']):a['consumer_event'] for a in contract['assignments']}
    weights={a['weight_event']:(a['core'],a['local_tile']) for a in contract['assignments']}
    p=e.Program()
    changes=[]
    before=bytearray();after=bytearray()
    for r in decoded['commands']:
        name=next(s['name'].rsplit('.',1)[0] for s in __import__('json').loads((original/'system.json').read_text())['command_symbols'] if s['id']==r['id'])
        deps=list(r['dependencies'])
        if r['id'] in weights:
            core,t=weights[r['id']]
            assert deps==([] if t<2 else [core_events[core,t-2]])
            if t>=3:
                deps=[core_events[core,t-3]]
                changes.append(dict(weight_tile=t,core=core,old_dependency=r['dependencies'][0],
                                    new_dependency=deps[0],last_L2_reader_tile=t-3))
        if r['kind'] in ('gdma_load','gdma_store'):
            event=p.dma(name,deps,r['lp6_address'],r['l2_address'],r['row_bytes'],
                load=r['kind']=='gdma_load',rows=r['rows'],lp6_stride=r['lp6_stride'],l2_stride=r['l2_stride'])
        elif r['kind']=='core_block':
            body=decoded['words'][r['isa_start_word']:r['isa_start_word']+r['isa_word_count']]
            blob=struct.pack(f'<{len(body)}I',*body)
            before.extend(blob);after.extend(blob)
            event=p.core(name,deps,r['target_logical_core'],body)
        elif r['kind']=='barrier':
            event=p.barrier(name,deps)
        else:raise ValueError(r['kind'])
        assert event==r['id'] and all(d<event for d in deps)
    assert before==after
    p.finish(path)
    e.save(path/'early_l2_release_contract.json',dict(M=m,N=n,K=k,
        core_ISA_bytes_unchanged=True,core_ISA_sha256=hashlib.sha256(before).hexdigest(),
        all_GDMA_addresses_sizes_and_record_order_unchanged=True,
        slot_lifetime='last L2 reader SYNC_ALL, rather than later L1 consumer completion',
        changed_dependencies=changes,changed_count=len(changes),
        source_program_sha256=e.sha(original/'program.bin'),program_sha256=e.sha(path/'program.bin'),
        original_generator_sha256=e.sha(pref.__file__),generator_sha256=e.sha(__file__),
        physical_MU_operand_capacity_validated=False,native_validation_performed=False))
    return contract
