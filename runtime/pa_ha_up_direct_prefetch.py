#!/usr/bin/env python3
"""Existing direct N4 g4 math with paid, double-buffered L2->L1 prefetch.

The original direct consumer and its dynamic B initializer are unchanged.
Each core uses two raw L1 buffers; two existing asynchronous LDMA instructions
fetch the next N128 stripe before the current MMA. Every core block ends with
SYNC_ALL, so the following block never relies on a cross-block local event.
The caller supplies the same PA image/configuration and schedules native runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import pa_ha_up_direct_fold as direct

e, isa = direct.e, direct.isa
RAW_L1 = (direct.RAW_L1, 0x280000)
CORES, TILE_N = direct.CORES, direct.TILE_N
STORAGE = direct.STORAGE


def _tail(m, n, k, n0, raw):
    words = []
    for tile in direct._m_tiles(m):
        begin, mt, b0 = tile['m_begin'], tile['m_size'], tile['b_l1']
        cols = 4*mt
        words += isa.load_constant(0, b0)+isa.m_ld(0, 0, cols, 4*k)
        words += isa.load_constant(1, raw)+isa.m_ld(2, 1, 32, 4*k)
        words.append(isa.m_mma(False, False, 4*k//32))
        words += isa.load_constant(2, direct.C_L1)+isa.m_wb(2, 2, 32, cols)
        words.append(isa.c_sync(isa.SYNC_MATRIX))
        for local_m in range(mt):
            words += direct._copy2d(5, direct.C_L1+local_m*16,
                direct.OUTPUT_L2+((begin+local_m)*n+n0)*4,
                16, 32, cols*4, 16)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _compute(m, n, k, n0, local_tile, current_slot, next_slot):
    raw = RAW_L1[local_tile % 2]
    words = []
    if local_tile == 0:
        words += direct._ldma(0, raw, current_slot, TILE_N*k)
    if next_slot is not None:
        other = RAW_L1[(local_tile+1) % 2]
        for index, offset in enumerate((0, direct.LDMA_CHUNK)):
            words += isa.ldma_i(10, other+offset, next_slot+offset,
                                direct.LDMA_CHUNK, event=index+1)
    words += _tail(m, n, k, n0, raw)
    return words


def _audit_body(words, m, n, k, n0, local_tile, current_slot, next_slot):
    decoded = direct._decoded_core_ops(words)
    expected_prefix = []
    raw = RAW_L1[local_tile % 2]
    if local_tile == 0:
        expected_prefix += direct._decoded_core_ops(direct._ldma(0, raw, current_slot, TILE_N*k))
    if next_slot is not None:
        for offset in (0, direct.LDMA_CHUNK):
            expected_prefix.append(dict(kind='ldma', op=10,
                first=RAW_L1[(local_tile+1) % 2]+offset, second=next_slot+offset,
                bytes=direct.LDMA_CHUNK, rows=1, source_stride=direct.LDMA_CHUNK,
                destination_stride=direct.LDMA_CHUNK))
    assert decoded[:len(expected_prefix)] == expected_prefix
    tail = decoded[len(expected_prefix):]
    original = direct._decoded_core_ops(direct._compute(m, n, k, n0, current_slot))[2:]
    normalized = [dict(op, address=direct.RAW_L1) if op['kind'] == 'm_ld' and op['operand'] == 2
                  else op for op in tail]
    assert normalized == original
    assert all(op['address'] == raw for op in tail if op['kind'] == 'm_ld' and op['operand'] == 2)
    async_events = [(instr[0] >> 10) & 15 for name, instr in isa.core_instructions(words)
                    if name == 'LDMA.I' and ((instr[0] >> 6) & 15) == 10]
    assert async_events == ([1, 2] if next_slot is not None else [])
    assert words[-1] == isa.c_sync(isa.SYNC_ALL)
    assert all(op['op'] in (0, 5, 10) for op in decoded if op['kind'] == 'ldma')
    cover = direct._audit_compute(direct._compute(m, n, k, n0, current_slot),
                                 m, n, k, n0, current_slot)
    return cover, expected_prefix


def generate(path, m, n, k, storage=STORAGE):
    """Emit checked same-layout direct-g4 prefetch; no image/native creation."""
    path = Path(path)
    direct._validate_shape(m, n, k, storage)
    if (path/'program.bin').exists() or (path/'system.json').exists():
        raise ValueError('refusing to overwrite a generated program')
    raw_bytes, mt = TILE_N*k, min(m, 8)
    zero_bytes = 16*k*mt
    l1_ranges = direct._ranges([
        ('X', direct.X_L1, m*k),
        *[(f'raw_W_{i}', base, raw_bytes) for i, base in enumerate(RAW_L1)],
        *[(f"B_m{t['m_begin']}", t['b_l1'], 16*k*t['m_size']) for t in direct._m_tiles(m)],
        ('initializer_E', direct.E_L1, k*4*mt),
        ('matrix_C', direct.C_L1, 32*4*mt*4)], 4 << 20)
    l2_ranges = direct._ranges([
        ('canonical_Y', direct.OUTPUT_L2, m*n*4),
        ('zero_constant', direct.ZERO_L2, zero_bytes),
        ('X', direct.INPUT_L2, m*k),
        ('raw_weight_slots', direct.WEIGHT_L2, 8*raw_bytes)], 8 << 20)
    lp6_ranges = direct._ranges([
        ('X', e.X_BASE, m*k), ('zero_constant', direct.ZERO_BASE, zero_bytes),
        ('packed_W', direct.WEIGHT_BASE, n*k)], 1 << 34)
    algebra = direct.static_algebra(m, k)
    p = e.Program()
    x_ready = p.dma('input.load', [], e.X_BASE, direct.INPUT_L2, m*k)
    zero_ready = p.dma('input.zero_load', [], direct.ZERO_BASE, direct.ZERO_L2, zero_bytes)
    prepared = [p.core('input.direct_expand', [x_ready, zero_ready], core, direct._prepare(m, k))
                for core in range(CORES)]
    previous = list(prepared)
    total_tiles = n//TILE_N
    counts = [(total_tiles+CORES-1-core)//CORES for core in range(CORES)]
    loads = [{} for _ in range(CORES)]
    consumers = [[] for _ in range(CORES)]
    assignments = []

    def stage(core, local_tile, deps):
        if local_tile >= counts[core]:
            return
        tile = local_tile*CORES+core
        slot = direct.WEIGHT_L2+(2*core+local_tile%2)*raw_bytes
        event = p.dma('weight.load', deps, direct.WEIGHT_BASE+tile*raw_bytes, slot, raw_bytes)
        loads[core][local_tile] = dict(tile=tile, slot=slot, event=event, dependencies=deps)

    # The first two L2 slots of every core are filled before its first compute
    # can start. Later slot reuse waits for the prior consuming core event.
    for local_tile in (0, 1):
        for core in range(CORES):
            stage(core, local_tile, [])
    for local_tile in range(max(counts)):
        for core in range(CORES):
            if local_tile >= counts[core]:
                continue
            current = loads[core][local_tile]
            following = loads[core].get(local_tile+1)
            next_slot = following['slot'] if following else None
            deps = [previous[core], current['event']]
            if following:
                deps.append(following['event'])
            body = _compute(m, n, k, current['tile']*TILE_N,
                            local_tile, current['slot'], next_slot)
            _audit_body(body, m, n, k, current['tile']*TILE_N,
                        local_tile, current['slot'], next_slot)
            event = p.core('npu.direct_prefetch', deps, core, body)
            assignments.append(dict(core=core, local_tile=local_tile, tile=current['tile'],
                slot=current['slot'], raw_l1=RAW_L1[local_tile%2],
                next_slot=next_slot, next_weight_event=following['event'] if following else None,
                next_raw_l1=RAW_L1[(local_tile+1)%2] if following else None,
                weight_event=current['event'], previous_core_event=previous[core],
                consumer_event=event, dependencies=deps))
            previous[core] = event
            consumers[core].append(event)
        for core in range(CORES):
            if local_tile < counts[core]:
                stage(core, local_tile+2, [consumers[core][local_tile]])
    final = p.barrier('done', previous)
    p.finish(path)
    decoded = isa.decode_program((path/'program.bin').read_bytes())
    records = {r['id']: r for r in decoded['commands']}
    out_cover = np.zeros(m*n*4, dtype=np.uint8)
    raw_live = [{} for _ in range(CORES)]
    external_bytes = local_bytes = async_bytes = 0
    for core, schedule in enumerate(loads):
        for t, info in schedule.items():
            r = records[info['event']]
            assert r['kind'] == 'gdma_load' and r['rows'] == 1 and r['row_bytes'] == raw_bytes
            assert r['lp6_address'] == direct.WEIGHT_BASE+info['tile']*raw_bytes
            assert r['l2_address'] == info['slot']
            assert r['dependencies'] == ([] if t < 2 else [consumers[core][t-2]])
            external_bytes += raw_bytes
    for a in assignments:
        core, t = a['core'], a['local_tile']
        r = records[a['consumer_event']]
        assert r['target_logical_core'] == core and r['dependencies'] == a['dependencies']
        body = decoded['words'][r['isa_start_word']:r['isa_start_word']+r['isa_word_count']]
        cover, moves = _audit_body(body, m, n, k, a['tile']*TILE_N, t, a['slot'], a['next_slot'])
        if t == 0:
            raw_live[core][a['raw_l1']] = a['tile']
        assert raw_live[core][a['raw_l1']] == a['tile'], 'current raw bytes were not loaded by predecessor'
        if a['next_slot'] is not None:
            assert a['raw_l1'] != a['next_raw_l1']
            raw_live[core][a['next_raw_l1']] = a['tile']+CORES
        local_bytes += sum(op['bytes'] for op in moves)
        async_bytes += sum(op['bytes'] for op in moves if op['op'] == 10)
        for lo, hi in cover:
            assert direct.OUTPUT_L2 <= lo < hi <= direct.OUTPUT_L2+m*n*4
            out_cover[lo-direct.OUTPUT_L2:hi-direct.OUTPUT_L2] += 1
    assert np.all(out_cover == 1) and external_bytes == local_bytes == n*k
    assert records[final]['dependencies'] == previous
    ancestors = {}
    for event, r in records.items():
        ancestors[event] = set(r['dependencies'])
        for dep in r['dependencies']:
            assert dep < event
            ancestors[event] |= ancestors[dep]
    assert ancestors[final] == set(records)-{final}
    contract = dict(schema='lp6.pa_ha.direct_prefetch.v1', M=m, N=n, K=k, storage=storage,
        scope='INT8 X/W in LP6 to canonical INT32 Y in L2, with real L2-to-L1 prefetch',
        quantization_included=False, scale_included=False, read_view_alias=False,
        actual_dram_mapping_supplied_by_caller=True,
        input_lp6=e.X_BASE, weight_lp6=direct.WEIGHT_BASE, zero_lp6=direct.ZERO_BASE,
        zero_lp6_bytes=zero_bytes, zero_source_must_be_all_zero=True, output_l2=direct.OUTPUT_L2,
        group_lanes=4, m_tiles=direct._m_tiles(m), raw_tile_bytes=raw_bytes,
        raw_l1_buffers=list(RAW_L1), raw_l2_slots=8, cores=4,
        l1_ranges=l1_ranges, l2_ranges=l2_ranges, lp6_ranges=lp6_ranges,
        l1_peak_bytes=max(r['offset']+r['bytes'] for r in l1_ranges),
        l2_peak_bytes=max(r['offset']+r['bytes'] for r in l2_ranges),
        LP6_weight_read_bytes=external_bytes, L2_weight_read_bytes=local_bytes,
        L2_async_weight_read_bytes=async_bytes, L2_initial_blocking_weight_read_bytes=local_bytes-async_bytes,
        LP6_input_read_bytes=m*k, LP6_zero_read_bytes=zero_bytes, packet_assembly_bytes=0,
        first_two_weights_ready_before_first_compute=True,
        current_and_next_weight_GDMA_dependencies_checked=True,
        L2_slot_reuse_waits_current_consumer=True,
        async_local_events=[1, 2], every_core_ends_SYNC_ALL=True,
        no_cross_block_local_event_wait=True, raw_buffer_ownership_checked=True,
        all_compute_math_and_scatter_decoded_equivalent_to_original_direct=True,
        original_initializer_unchanged=True, B_generated_once_per_core=True,
        matrix_operations=total_tiles*len(direct._m_tiles(m)),
        matrix_compute_K32_slices=total_tiles*len(direct._m_tiles(m))*(4*k//32),
        final_join_event=final, all_commands_ancestors_of_final_join=True,
        all_output_bytes_written_once=True, command_count=len(records), assignments=assignments,
        program_sha256=e.sha(path/'program.bin'), generator_sha256=e.sha(Path(__file__)),
        dependency_source_sha256={str(Path(f.__file__)): e.sha(Path(f.__file__)) for f in (direct, e, isa)},
        static_algebra=algebra, native_validation_performed=False,
        physical_MU_operand_capacity_validated=False,
        limitations=['Current simulator snapshots full K operands but has no separately configured MU byte-capacity check.',
                     'L1/L2 capacity and ownership are checked; this does not prove physical MU capacity.',
                     'This diagnostic emits existing ISA and does not modify the production compiler.'])
    e.save(path/'direct_prefetch_contract.json', contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument('--n', type=int, default=14336)
    parser.add_argument('--k', type=int, default=4096)
    args = parser.parse_args()
    contract = generate(args.output, args.m, args.n, args.k)
    print(f"Generated {contract['command_count']} commands; all static checks passed; no native run.")


if __name__ == '__main__':
    main()
