#!/usr/bin/env python3
"""Audit exact L2 access ranges and command dependencies in emitted ISA.

This is an independent static safety check, not a latency predictor. It decodes
all core local DMA instructions and global DMA records. At each conflicting
RAW/WAR/WAW access, the previous command must be a transitive dependency of the
new command. Block completion drains local DMA events in the existing ISA.

Run on a compiled build directory (image/simulation not required). The negative
control removes a required dependency and must be rejected by the same check.
"""
import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'core/simulator/npu/transactional_emulator/tools'))
from isa_v2 import decode_program
from isa_v2_trace import trace_v2


def row_ranges(base, size, rows, stride):
    return [(base + r * stride, base + r * stride + size) for r in range(rows)]


def accesses(program):
    """Return (begin,end,is_write) per system command; reject unknown DMA modes."""
    result = []
    for command in program['commands']:
        entries = []
        kind = command['kind']
        if kind in ('gdma_load', 'gdma_store'):
            entries = [(a,b,kind == 'gdma_load') for a,b in row_ranges(
                command['l2_address'], command['row_bytes'], command['rows'], command['l2_stride'])]
        elif kind == 'core_block':
            start = command['isa_start_word']; end = start + command['isa_word_count']
            for event in trace_v2(program['words'][start:end]):
                if event[0] != 'ldma': continue
                _, mode, l1, l2, size, shape, local_event = event
                if mode == 9: continue # L1-to-L1 copy has no L2 footprint.
                if mode not in (0,1,4,5,10,11,12,13):
                    raise ValueError(f'Unsupported local DMA mode {mode}; cannot claim coverage')
                write = mode in (1,5,11,13)
                rows, stride = (shape[0], shape[2 if write else 1]) if shape else (1,size)
                entries.extend((a,b,write) for a,b in row_ranges(l2,size,rows,stride))
        elif kind != 'barrier':
            raise ValueError(f'Unsupported system command {kind}; NPU-only checker')
        result.append(entries)
    return result


def check(commands, ranges, capacity):
    endpoints = sorted({value for entries in ranges for a,b,_ in entries for value in (a,b)})
    assert not endpoints or 0 <= endpoints[0] <= endpoints[-1] <= capacity, 'L2 capacity exceeded'
    ancestors = []
    for i, command in enumerate(commands):
        mask = 0
        for dependency in command['dependencies']:
            assert 0 <= dependency < i, 'Noncausal or invalid command dependency'
            mask |= ancestors[dependency] | (1 << dependency)
        ancestors.append(mask)
    writers = [-1] * max(0,len(endpoints)-1)
    readers = [set() for _ in writers]
    violations, checked = [], 0
    for i, entries in enumerate(ranges):
        for a,b,write in entries:
            for segment in range(bisect_left(endpoints,a), bisect_left(endpoints,b)):
                required = readers[segment] | {writers[segment]} if write else {writers[segment]}
                for previous in required:
                    if previous < 0 or previous == i: continue
                    checked += 1
                    if not (ancestors[i] >> previous) & 1:
                        violations.append(dict(command=i, previous=previous,
                            access='write' if write else 'read',
                            bytes=[endpoints[segment],endpoints[segment+1]]))
                        if len(violations) >= 20: return violations, checked
                if write:
                    writers[segment] = i
                    readers[segment].clear()
                else: readers[segment].add(i)
    return violations, checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('build', nargs='+', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--negative-control', action='store_true')
    args = ap.parse_args()
    results = {}
    for build in args.build:
        binary = (build/'program.bin').read_bytes()
        program = decode_program(binary)
        ranges = accesses(program)
        system = json.loads((build/'system.json').read_text())
        capacity = system['l2_bytes_required']
        violations, checked = check(program['commands'],ranges,capacity)
        assert not violations, (str(build), violations)
        item = dict(program_sha256=hashlib.sha256(binary).hexdigest(), commands=len(ranges),
            l2_access_ranges=sum(map(len,ranges)), dependency_obligations_checked=checked,
            l2_bytes_required=capacity, violations=violations)
        if args.negative_control:
            # Remove a ring slot's dependency on the core block that consumes
            # its old value. This is precisely the unsafe "refill earlier" edit.
            commands = program['commands']
            found = False
            for i,c in enumerate(commands):
                if c['kind'] != 'gdma_load': continue
                # A repeated equal-size load into the same slot, rather than
                # the unrelated initial constant-table/activation staging.
                if not any(old['kind'] == 'gdma_load' and
                           all(old[field] == c[field] for field in ('l2_address','row_bytes','rows','l2_stride'))
                           for old in commands[:i]): continue
                for dependency in c['dependencies']:
                    if commands[dependency]['kind'] != 'core_block': continue
                    copied = [dict(x) for x in commands]
                    copied[i]['dependencies'] = [x for x in c['dependencies'] if x != dependency]
                    failures, _ = check(copied,ranges,capacity)
                    if failures:
                        item['negative_control'] = dict(removed_edge=[dependency,i],detected=failures[0])
                        found = True; break
                if found: break
            assert found, 'Could not construct a missing-slot-release dependency negative control'
        results[str(build.resolve())] = item
        print(build.name, json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__': main()
