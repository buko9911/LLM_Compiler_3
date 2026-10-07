#!/usr/bin/env python3
"""Schedule real weight PA windows with existing GDMA ISA; no read-view.

This is a bounded software scheduling control, not PFM's hardware FAS. Core
instructions, source bytes, L2 destinations and lifetime dependencies are kept.
"""
from __future__ import annotations
import argparse
from collections import deque
import configparser
import hashlib
import json
from pathlib import Path
import struct
import dual_view_experiment as e


def rewrite(source: Path, target: Path, config: Path, weight_base: int,
            weight_bytes: int, window_bytes: int = 4096):
    assert window_bytes > 0 and window_bytes % 64 == 0
    ini = configparser.ConfigParser(strict=False)
    ini.read(config)
    bits = [int(x) for x in ini.get('system', 'address_bits_ch',
                                    fallback='5,6,7').split(',')]
    assert len(bits) == 3
    def port(pa):
        sc = sum(((pa >> b) & 1) << i for i,b in enumerate(bits))
        return sc // 2
    binary = (source/'program.bin').read_bytes()
    decoded = e.isa.decode_program(binary)
    manifest = json.loads((source/'system.json').read_text())
    names = {x['id']: x['name'].rsplit('.',1)[0] for x in manifest['command_symbols']}
    records = decoded['commands']
    words = decoded['words']
    p = e.Program()
    remap = {}
    proofs = []
    before_core = bytearray()
    after_core = bytearray()
    def is_weight(r):
        return (r['kind'] == 'gdma_load' and r['rows'] == 1
                and weight_base <= r['lp6_address']
                and r['lp6_address']+r['row_bytes'] <= weight_base+weight_bytes)
    cursor = 0
    while cursor < len(records):
        r = records[cursor]
        if is_weight(r):
            group = []
            while cursor < len(records) and is_weight(records[cursor]):
                group.append(records[cursor]); cursor += 1
            queues = [deque() for _ in range(4)]
            children = {a['id']: [] for a in group}
            for a in group:
                for off in range(0,a['row_bytes'],window_bytes):
                    size = min(window_bytes,a['row_bytes']-off)
                    queues[port(a['lp6_address']+off)].append((a,off,size))
            while any(queues):
                for queue in queues:
                    if not queue: continue
                    a,off,size = queue.popleft()
                    event = p.dma('weight.window', [remap[d] for d in a['dependencies']],
                                  a['lp6_address']+off,a['l2_address']+off,size)
                    children[a['id']].append((off,size,event))
            for a in group:
                pieces = sorted(children[a['id']])
                end = 0
                for off,size,_ in pieces:
                    assert off == end
                    end += size
                assert end == a['row_bytes']
                remap[a['id']] = p.barrier('weight.ready',[v[2] for v in pieces])
                proofs.append(dict(original_event=a['id'],pa=a['lp6_address'],l2=a['l2_address'],
                                   bytes=end,windows=len(pieces)))
            continue
        deps = [remap[d] for d in r['dependencies']]
        name = names[r['id']]
        if r['kind'] in ('gdma_load','gdma_store'):
            event = p.dma(name,deps,r['lp6_address'],r['l2_address'],r['row_bytes'],
                          load=r['kind']=='gdma_load',rows=r['rows'],
                          lp6_stride=r['lp6_stride'],l2_stride=r['l2_stride'])
        elif r['kind'] == 'barrier':
            event = p.barrier(name,deps)
        elif r['kind'] == 'core_block':
            body = words[r['isa_start_word']:r['isa_start_word']+r['isa_word_count']]
            blob = struct.pack(f'<{len(body)}I',*body)
            before_core.extend(blob)
            event = p.core(name,deps,r['target_logical_core'],body)
            after_core.extend(blob)
        else:
            raise ValueError('This scheduling control only supports NPU programs')
        remap[r['id']] = event
        cursor += 1
    assert before_core == after_core
    p.finish(target)
    report = dict(schema='lp6.pa_ha.software_dma_schedule.v1',
                  physical_channel_count=4,window_bytes=window_bytes,
                  address_bits_ch=bits,source_program_sha256=e.sha(source/'program.bin'),
                  program_sha256=e.sha(target/'program.bin'),core_isa_unchanged=True,
                  core_isa_sha256=hashlib.sha256(before_core).hexdigest(),
                  every_source_byte_once_per_original_dma=True,
                  every_l2_destination_preserved=True,
                  original_weight_commands=len(proofs),
                  scheduled_weight_commands=sum(q['windows'] for q in proofs),
                  weight_transfer_bytes=sum(q['bytes'] for q in proofs),
                  descriptor_proofs=proofs)
    e.save(target/'dma_schedule_audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--weight-base',type=lambda s:int(s,0),required=True)
    parser.add_argument('--weight-bytes',type=int,required=True)
    parser.add_argument('--window-bytes',type=int,default=4096)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('output must be a new directory')
    report=rewrite(args.source,args.output,args.config,args.weight_base,args.weight_bytes,args.window_bytes)
    print(json.dumps({k:v for k,v in report.items() if k!='descriptor_proofs'},indent=2))


if __name__=='__main__': main()
