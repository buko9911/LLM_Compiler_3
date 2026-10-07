#!/usr/bin/env python3
"""Paid M16 g2 NPU consumer of the unchanged [N/4,K/8,4,8] PA image.

Read a raw N128 stripe once, then assemble two N64 operands A[32,2*K]
using existing 16-byte COPY_STRIDED instructions. Dynamic X is expanded
once per core to B[2*K,2*M] by real zero reads and encoded copies. This
trades the direct g4 consumer's two M8 operations for two N64 operations;
it does not assume weight assembly, zeroing, or operand loading is free.

Endpoint: INT8 X and packed INT8 W in LP6 -> canonical INT32 Y[M,N] in L2.
The caller supplies the unchanged image and actual PA->HA configuration.
Only program generation and CPU/ISA validation are performed here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
from pathlib import Path

import numpy as np

import pa_ha_up_direct_fold as direct

e, isa = direct.e, direct.isa
STORAGE = direct.STORAGE
WEIGHT_BASE, ZERO_BASE = direct.WEIGHT_BASE, direct.ZERO_BASE
OUTPUT_L2, ZERO_L2 = direct.OUTPUT_L2, direct.ZERO_L2
INPUT_L2, WEIGHT_L2 = direct.INPUT_L2, direct.WEIGHT_L2
X_L1, RAW_L1, A_L1 = 0, 0x40000, 0x100000
B_L1, E_L1, C_L1 = 0x180000, 0x200000, 0x240000
CORES, TILE_N, GROUP_LANES, N_SUBTILE = 4, 128, 2, 64
LDMA_CHUNK = 256 << 10
_ldma, _copy2d = direct._ldma, direct._copy2d
_decoded_core_ops, _ranges = direct._decoded_core_ops, direct._ranges


def _validate_shape(m, n, k, storage):
    if storage != STORAGE:
        raise ValueError(f'only checked storage {STORAGE!r} is supported')
    if m != 16:
        raise ValueError('bounded N4 g2 consumer supports only M=16')
    if k != 4096 or n <= 0 or n % TILE_N:
        raise ValueError('bounded N4 g2 requires K=4096 and positive N divisible by128')


def _prepare(m, k):
    cols = GROUP_LANES*m
    words = _ldma(0, X_L1, INPUT_L2, m*k)
    words += _ldma(0, B_L1, ZERO_L2, GROUP_LANES*k*cols)
    words += _ldma(0, E_L1, ZERO_L2, k*cols)
    for token in range(m):
        words += _copy2d(9, E_L1+GROUP_LANES*token, X_L1+token*k,
                         1, k, 1, cols)
    for lane in range(GROUP_LANES):
        words += _copy2d(9, B_L1+8*lane*cols+lane, E_L1,
                         8*cols-lane, k//8, 8*cols, 8*GROUP_LANES*cols)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _assembly(k, subtile):
    words = []
    for q in range(32):
        # Each PA N4 group holds its complete K range in 4*K bytes. Select
        # either its first or second 16B half from every original 32B packet.
        source = RAW_L1 + (subtile*16+q//2)*4*k + (q%2)*16
        words += _copy2d(9, A_L1+q*GROUP_LANES*k, source,
                         16, k//8, 32, 16)
    return words


def _compute(m, n, k, n0, slot):
    words = _ldma(0, RAW_L1, slot, TILE_N*k)
    for subtile in range(2):
        words += _assembly(k, subtile)
        words += isa.load_constant(0, B_L1)+isa.m_ld(0, 0, 2*m, 2*k)
        words += isa.load_constant(1, A_L1)+isa.m_ld(2, 1, 32, 2*k)
        words.append(isa.m_mma(False, False, 2*k//32))
        words += isa.load_constant(2, C_L1)+isa.m_wb(2, 2, 32, 2*m)
        words.append(isa.c_sync(isa.SYNC_MATRIX))
        for token in range(m):
            words += _copy2d(5, C_L1+token*8,
                             OUTPUT_L2+(token*n+n0+subtile*N_SUBTILE)*4,
                             8, 32, 8*m, 8)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _replay_preparation(m, k, x):
    l1 = np.full(4 << 20, -79, dtype=np.int8)
    defined = np.zeros(4 << 20, dtype=bool)
    reads = collections.Counter()
    for op in _decoded_core_ops(_prepare(m, k)):
        assert op['kind'] == 'ldma'
        dst, src, width = op['first'], op['second'], op['bytes']
        if op['op'] == 0:
            if INPUT_L2 <= src < INPUT_L2+m*k:
                data = x.reshape(-1)[src-INPUT_L2:src-INPUT_L2+width]
                assert len(data) == width
                reads['X'] += width
            else:
                assert ZERO_L2 <= src and src+width <= ZERO_L2+2*k*2*m
                data = np.zeros(width, dtype=np.int8)
                reads['zero'] += width
            assert len(l1[dst:dst+width]) == width
            l1[dst:dst+width] = data
            defined[dst:dst+width] = True
        elif op['op'] == 9:
            for row in range(op['rows']):
                s = src+row*op['source_stride']
                d = dst+row*op['destination_stride']
                assert np.all(defined[s:s+width]) and len(l1[d:d+width]) == width
                l1[d:d+width] = l1[s:s+width].copy()
                defined[d:d+width] = True
        else:
            raise AssertionError('unsupported initializer DMA')
    return l1, defined, dict(reads)


def static_algebra(m=16, k=4096):
    """Replay the actual initializer/assembly/store ISA on one N128 stripe."""
    _validate_shape(m, TILE_N, k, STORAGE)
    rng = np.random.default_rng(20261008)
    x = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
    w = rng.integers(-128, 128, (TILE_N, k), dtype=np.int16).astype(np.int8)
    # The reference packs from logical W[n,k], independently of emitter PAs.
    raw = w.reshape(32, 4, k//8, 8).transpose(0, 2, 1, 3).copy().reshape(-1)
    l1, defined, reads = _replay_preparation(m, k, x)
    b = l1[B_L1:B_L1+2*k*2*m].reshape(2*k, 2*m)
    wanted_b = np.zeros_like(b)
    for lane in range(2):
        rows = (np.arange(k//8)[:, None]*16+8*lane+np.arange(8)).reshape(-1)
        for token in range(m):
            wanted_b[rows, 2*token+lane] = x[token]
    assert np.all(defined[B_L1:B_L1+b.size]) and np.array_equal(b, wanted_b)
    l1[RAW_L1:RAW_L1+raw.size] = raw
    defined[RAW_L1:RAW_L1+raw.size] = True
    source_coverage = np.zeros(raw.size, dtype=np.uint8)
    output = np.zeros(m*TILE_N*4, dtype=np.uint8)
    output_coverage = np.zeros_like(output)
    a_checked = 0
    for subtile in range(2):
        destination_coverage = np.zeros(32*2*k, dtype=np.uint8)
        for op in _decoded_core_ops(_assembly(k, subtile)):
            assert op['kind'] == 'ldma' and op['op'] == 9
            for row in range(op['rows']):
                src = op['second']+row*op['source_stride']
                dst = op['first']+row*op['destination_stride']
                width = op['bytes']
                assert RAW_L1 <= src and src+width <= RAW_L1+raw.size
                assert A_L1 <= dst and dst+width <= A_L1+32*2*k
                assert np.all(defined[src:src+width])
                l1[dst:dst+width] = l1[src:src+width].copy()
                source_coverage[src-RAW_L1:src-RAW_L1+width] += 1
                destination_coverage[dst-A_L1:dst-A_L1+width] += 1
        assert np.all(destination_coverage == 1)
        a = l1[A_L1:A_L1+32*2*k].reshape(32, 2*k)
        q, j, lane, element = np.indices((32, k//8, 2, 8))
        wanted_a = w[subtile*64+2*q+lane, 8*j+element].reshape(32, 2*k)
        assert np.array_equal(a, wanted_a)
        a_checked += a.size
        c = a.astype(np.int64) @ b.astype(np.int64)
        assert np.max(np.abs(c)) <= k*128*128 < 2**31
        c_bytes = c.astype('<i4').reshape(-1).view(np.uint8)
        store_ops = [op for op in _decoded_core_ops(_compute(m, TILE_N, k, 0, WEIGHT_L2))
                     if op['kind'] == 'ldma' and op['op'] == 5]
        for op in store_ops[subtile*m:(subtile+1)*m]:
            for row in range(op['rows']):
                src = op['first']-C_L1+row*op['source_stride']
                dst = op['second']-OUTPUT_L2+row*op['destination_stride']
                width = op['bytes']
                assert 0 <= src and src+width <= c_bytes.size
                assert 0 <= dst and dst+width <= output.size
                output[dst:dst+width] = c_bytes[src:src+width]
                output_coverage[dst:dst+width] += 1
    expected = x.astype(np.int64) @ w.astype(np.int64).T
    actual = output.view('<i4').reshape(m, TILE_N)
    assert np.array_equal(actual, expected)
    assert np.all(source_coverage == 1) and np.all(output_coverage == 1)
    return dict(M=m, K=k, group_lanes=2, N_subtiles=2,
                checked_raw_weight_bytes=raw.size, checked_A_bytes=a_checked,
                checked_B_bytes=b.size, checked_output_elements=expected.size,
                encoded_initializer_all_B_bytes_exact=True,
                encoded_assembly_all_A_bytes_exact=True,
                every_raw_weight_byte_assembled_once=True,
                encoded_output_scatter_all_bytes_exact=True,
                I64_dot_exact=True, int32_absolute_bound=k*128*128,
                initializer_L2_bytes_per_core=reads,
                expected_sha256=hashlib.sha256(expected.astype('<i4').tobytes()).hexdigest())


def _audit_compute(words, m, n, k, n0, slot):
    ops = _decoded_core_ops(words)
    expected_loads = [dict(kind='ldma', op=0, first=RAW_L1+off, second=slot+off,
                           bytes=LDMA_CHUNK, rows=1, source_stride=LDMA_CHUNK,
                           destination_stride=LDMA_CHUNK)
                      for off in (0, LDMA_CHUNK)]
    assert ops[:2] == expected_loads
    expected = list(expected_loads)
    coverage = []
    for subtile in range(2):
        for q in range(32):
            expected.append(dict(kind='ldma', op=9, first=A_L1+q*2*k,
                second=RAW_L1+(subtile*16+q//2)*4*k+(q%2)*16,
                bytes=16, rows=k//8, source_stride=32, destination_stride=16))
        expected += [dict(kind='m_ld', operand=0, address=B_L1, mn=2*m, k=2*k,
                          extended_stride=None),
                     dict(kind='m_ld', operand=2, address=A_L1, mn=32, k=2*k,
                          extended_stride=None),
                     dict(kind='mma', count=2*k//32, fp16=False, accumulate=False),
                     dict(kind='m_wb', address=C_L1, dtype=2, m=32, n=2*m)]
        for token in range(m):
            dst = OUTPUT_L2+(token*n+n0+subtile*64)*4
            expected.append(dict(kind='ldma', op=5, first=C_L1+token*8,
                second=dst, bytes=8, rows=32, source_stride=8*m, destination_stride=8))
            coverage.append((dst, dst+64*4))
    assert ops == expected, 'encoded instruction order or address contract changed'
    return coverage


def generate(path, m=16, n=14336, k=4096, storage=STORAGE):
    """Emit a bounded M16 g2 control; caller controls mapping and native run."""
    path = Path(path)
    _validate_shape(m, n, k, storage)
    if (path/'program.bin').exists() or (path/'system.json').exists():
        raise ValueError('refusing to overwrite a generated program')
    raw_bytes, b_bytes, e_bytes = 128*k, 2*k*2*m, k*2*m
    l1_ranges = _ranges([('X', X_L1, m*k), ('raw_W', RAW_L1, raw_bytes),
        ('assembled_A', A_L1, 32*2*k), ('expanded_B', B_L1, b_bytes),
        ('initializer_E', E_L1, e_bytes), ('matrix_C', C_L1, 32*2*m*4)], 4 << 20)
    l2_ranges = _ranges([('canonical_Y', OUTPUT_L2, m*n*4),
        ('zero_constant', ZERO_L2, b_bytes), ('X', INPUT_L2, m*k),
        ('raw_weight_slots', WEIGHT_L2, 8*raw_bytes)], 8 << 20)
    lp6_ranges = _ranges([('X', e.X_BASE, m*k), ('zero_constant', ZERO_BASE, b_bytes),
                         ('packed_W', WEIGHT_BASE, n*k)], 1 << 34)
    algebra = static_algebra(m, k)
    p = e.Program()
    x_ready = p.dma('input.load', [], e.X_BASE, INPUT_L2, m*k)
    zero_ready = p.dma('input.zero_load', [], ZERO_BASE, ZERO_L2, b_bytes)
    previous = [p.core('input.n4_g2_expand', [x_ready, zero_ready], core, _prepare(m, k))
                for core in range(CORES)]
    history, assignments = [[] for _ in range(CORES)], []
    tiles = n//TILE_N
    for batch in range((tiles+CORES-1)//CORES):
        staged = []
        for core in range(CORES):
            tile = batch*CORES+core
            if tile >= tiles:
                continue
            slot = WEIGHT_L2+(2*core+batch%2)*raw_bytes
            reuse = [history[core][batch-2]] if batch >= 2 else []
            event = p.dma('weight.load', reuse, WEIGHT_BASE+tile*raw_bytes, slot, raw_bytes)
            staged.append((core, tile, slot, event, reuse))
        for core, tile, slot, ready, reuse in staged:
            body = _compute(m, n, k, tile*TILE_N, slot)
            _audit_compute(body, m, n, k, tile*TILE_N, slot)
            event = p.core('npu.n4_g2_fold', [previous[core], ready], core, body)
            previous[core] = event
            history[core].append(event)
            assignments.append(dict(tile=tile, n_begin=tile*TILE_N, core=core, slot=slot,
                weight_event=ready, consumer_event=event, reuse_dependency=reuse))
    final = p.barrier('done', previous)
    p.finish(path)
    decoded = isa.decode_program((path/'program.bin').read_bytes())
    records = {r['id']: r for r in decoded['commands']}
    out_cover = np.zeros(m*n*4, dtype=np.uint8)
    external_bytes = local_bytes = 0
    for a in assignments:
        r = records[a['weight_event']]
        assert r['kind'] == 'gdma_load' and r['rows'] == 1 and r['row_bytes'] == raw_bytes
        assert r['lp6_address'] == WEIGHT_BASE+a['tile']*raw_bytes
        assert r['l2_address'] == a['slot'] and r['dependencies'] == a['reuse_dependency']
        external_bytes += r['row_bytes']
        c = records[a['consumer_event']]
        assert c['target_logical_core'] == a['core'] and a['weight_event'] in c['dependencies']
        body = decoded['words'][c['isa_start_word']:c['isa_start_word']+c['isa_word_count']]
        for lo, hi in _audit_compute(body, m, n, k, a['n_begin'], a['slot']):
            assert OUTPUT_L2 <= lo < hi <= OUTPUT_L2+m*n*4
            out_cover[lo-OUTPUT_L2:hi-OUTPUT_L2] += 1
        local_bytes += raw_bytes
    assert np.all(out_cover == 1) and external_bytes == local_bytes == n*k
    assert all(r['kind'] in ('gdma_load', 'core_block', 'barrier') for r in records.values())
    assert records[final]['dependencies'] == previous
    ancestors = {}
    for event, r in records.items():
        ancestors[event] = set(r['dependencies'])
        for dep in r['dependencies']:
            assert dep < event
            ancestors[event] |= ancestors[dep]
    assert ancestors[final] == set(records)-{final}
    contract = dict(schema='lp6.pa_ha.n4_g2.v1', M=m, N=n, K=k, storage=storage,
        scope='INT8 X and one packed INT8 W in LP6 to canonical INT32 Y in L2',
        quantization_included=False, scale_included=False, new_hardware=False,
        read_view_alias=False, actual_dram_mapping_supplied_by_caller=True,
        group_lanes=2, N_subtile=64, N_subtiles_per_raw_tile=2, m_tiles=1,
        weight_storage_order=['N/4', 'K/8', 4, 8],
        physical_weight_byte_formula='W_BASE + ((n//4)*(K//8)+k//8)*32 + (n%4)*8 + k%8',
        input_lp6=e.X_BASE, weight_lp6=WEIGHT_BASE, zero_lp6=ZERO_BASE,
        zero_lp6_bytes=b_bytes, zero_source_must_be_all_zero=True, output_l2=OUTPUT_L2,
        l1_ranges=l1_ranges, l2_ranges=l2_ranges, lp6_ranges=lp6_ranges,
        l1_peak_bytes=max(r['offset']+r['bytes'] for r in l1_ranges),
        l2_peak_bytes=max(r['offset']+r['bytes'] for r in l2_ranges),
        raw_tile_bytes=raw_bytes, raw_slot_count=8, cores=4, tiles=tiles,
        weight_read_passes=1, LP6_weight_read_bytes=n*k, L2_weight_read_bytes=n*k,
        LP6_input_read_bytes=m*k, LP6_zero_read_bytes=b_bytes, LP6_output_write_bytes=0,
        packet_assembly_bytes=n*k, compute_core_COPY_STRIDED_count=tiles*64,
        assembly_COPY_STRIDED_rows_each=k//8, assembly_COPY_STRIDED_bytes_each_row=16,
        every_raw_weight_byte_assembled_once=True,
        raw_W_direct_M_LD_activation_operand=False, raw_W_reused_across_N_subtiles=True,
        initializer='paid_zero_LP6_then_encoded_E_transpose_and_shifted_bulk_copy',
        B_generated_once_per_core=True, B_reused_across_all_weight_tiles=True,
        expanded_B_total_bytes_per_core=b_bytes, initializer_E_live_bytes_per_core=e_bytes,
        matrix_k=2*k, matrix_rows=32, matrix_columns=2*m,
        matrix_operations=tiles*2, matrix_compute_K32_slices=tiles*2*(2*k//32),
        raw_weight_MU_load_bytes=n*k, expanded_input_MU_load_bytes=tiles*2*b_bytes,
        all_dynamic_operand_and_scatter_costs_included=True,
        LDMA_I_max_chunk_bytes=LDMA_CHUNK, assignments=assignments,
        final_join_event=final, all_commands_ancestors_of_final_join=True,
        all_output_bytes_written_once=True, command_count=len(records),
        program_sha256=e.sha(path/'program.bin'), generator_sha256=e.sha(Path(__file__)),
        dependency_source_sha256={str(Path(f.__file__)): e.sha(Path(f.__file__))
                                  for f in (direct, e, isa)},
        static_algebra=algebra, native_validation_performed=False,
        limitations=[
            'The caller must check the image and actual PA-to-HA mapping; this generator creates no image.',
            'M16 g2 pays local packet assembly and twice loads B into MU per N128 stripe.',
            'Existing ISA releases both MU operands at final K; resident L1 B is not persistent MU state.',
            'This is a bounded diagnostic consumer, not automatic production compiler lowering.'])
    e.save(path/'n4_g2_contract.json', contract)
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--n', type=int, default=14336)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--storage', default=STORAGE)
    args = parser.parse_args()
    result = generate(args.output, args.m, args.n, args.k, args.storage)
    print(f"Generated {result['command_count']} commands; encoded CPU proof exact; no native run.")


if __name__ == '__main__':
    main()
