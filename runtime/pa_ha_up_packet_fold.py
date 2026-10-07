#!/usr/bin/env python3
"""Paid packet-fold NPU control for the actual PA-to-HA Up experiment.

Input: INT8 X[M,K] at LP6 4096, one INT8 W copy at LP6 128 MiB in
[N/32, 4, K/8, 8, 8] byte order. Output: canonical INT32 Y[M,N] at L2 0.
The caller supplies the image and actual DRAM mapping; this file creates no
read-view, image, host-computed B, native process, or PIM command.

M <= 8 uses four packet lanes together; M=16 uses two groups of two lanes.
Every raw N=128 stripe is read once and reused across the g2 lane groups.
All X preparation, zero traffic, local packet assembly and output scatter are
real existing ISA. This diagnostic is not automatic compiler lowering.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import dual_view_experiment as e

isa = e.isa
WEIGHT_BASE = 128 << 20
ZERO_BASE = 2 << 20
OUTPUT_L2 = 0
ZERO_L2 = 1 << 20
INPUT_L2 = e.X_L2
WEIGHT_L2 = 2 << 20
X_L1, RAW_L1, A_L1 = 0, 0x40000, 0x100000
B_L1, E_L1, C_L1 = 0x200000, 0x2A0000, 0x300000
CORES = 4
TILE_N = 128
LDMA_CHUNK = 256 << 10  # 512 KiB itself exceeds LDMA.I's 14-bit length.


def _group(m: int) -> int:
    if m not in (1, 2, 4, 8, 16):
        raise ValueError("bounded control supports M=1,2,4,8,16")
    return 4 if m <= 8 else 2


def _ranges(ranges, capacity):
    for name, start, size in ranges:
        if start < 0 or size <= 0 or start + size > capacity:
            raise ValueError(f"out-of-range allocation {name}: {start}+{size}")
    for i, (name, start, size) in enumerate(ranges):
        for other, other_start, other_size in ranges[i + 1:]:
            if max(start, other_start) < min(start + size, other_start + other_size):
                raise ValueError(f"overlapping allocations: {name}, {other}")
    return [dict(name=name, offset=start, bytes=size) for name, start, size in ranges]


def _ldma(op, l1, l2, size):
    if size <= 0 or size % 32:
        raise ValueError("flat LDMA requires a positive 32-byte multiple")
    words = []
    for off in range(0, size, LDMA_CHUNK):
        words += isa.ldma_i(op, l1 + off, l2 + off, min(LDMA_CHUNK, size - off))
    return words


def _copy2d(op, destination, source, row_bytes, rows, source_stride, destination_stride):
    # COPY_STRIDED: first GP is L1 destination, second GP is L1 source.
    # STORE_STRIDED: first GP is L1 source, second GP is L2 destination.
    words = [isa.c_set(isa.SET_BYTES, row_bytes), isa.c_set(isa.SET_ROWS, rows),
             isa.c_set(isa.SET_SRC_STRIDE, source_stride),
             isa.c_set(isa.SET_DST_STRIDE, destination_stride)]
    words += isa.load_constant(14, destination) + isa.load_constant(15, source)
    words.append(isa.ldma_r(op, 14, 15))
    return words


def _q_source(q, k):
    n0 = 4 * q
    return (n0 // 32) * (32 * k) + ((n0 % 32) // 8) * (8 * k) + (n0 % 8) * 8


def _prepare(m, k, g):
    cols = g * m
    b_bytes, e_bytes = g * k * cols, k * cols
    words = _ldma(0, X_L1, INPUT_L2, m * k)
    # One real shared LP6 zero read supplies both private zero-filled regions.
    words += _ldma(0, B_L1, ZERO_L2, b_bytes)
    words += _ldma(0, E_L1, ZERO_L2, e_bytes)
    # E[k,g*t] = X[t,k]; other columns remain explicitly loaded zero bytes.
    for token in range(m):
        words += _copy2d(9, E_L1 + g * token, X_L1 + token * k,
                         1, k, 1, cols)
    # Shift each E eight-row group to its diagonal lane of B. The trimmed
    # trailing zeros keep every write inside the B allocation.
    for lane in range(g):
        words += _copy2d(9, B_L1 + 8 * lane * cols + lane, E_L1,
                         8 * cols - lane, k // 8, 8 * cols, 8 * g * cols)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def _compute(m, n, k, g, n0, slot):
    words = _ldma(0, RAW_L1, slot, TILE_N * k)
    for lane0 in range(0, 4, g):
        for q in range(32):
            words += _copy2d(9, A_L1 + q * g * k,
                             RAW_L1 + _q_source(q, k) + lane0 * 8,
                             8 * g, k // 8, 64, 8 * g)
        words += isa.load_constant(0, B_L1) + isa.m_ld(0, 0, g * m, g * k)
        words += isa.load_constant(1, A_L1) + isa.m_ld(2, 1, 32, g * k)
        words.append(isa.m_mma(False, False, g * k // 32))
        words += isa.load_constant(2, C_L1) + isa.m_wb(2, 2, 32, g * m)
        words.append(isa.c_sync(isa.SYNC_MATRIX))
        for token in range(m):
            # C[q,g*t+l] -> Y[t,n0+4*q+lane0+l]. No arithmetic reduction.
            words += _copy2d(5, C_L1 + token * g * 4,
                             OUTPUT_L2 + (token * n + n0 + lane0) * 4,
                             g * 4, 32, g * m * 4, 16)
    words.append(isa.c_sync(isa.SYNC_ALL))
    return words


def static_algebra(m, k=4096):
    """CPU validation of actual copy descriptors, including all zero bytes.

    This validates the emitted layout algebra, not native timing or the native
    matrix implementation. The source W is independently indexed by [n,k].
    """
    g = _group(m)
    if k != 4096:
        raise ValueError("bounded full-K control requires K=4096")
    rng = np.random.default_rng(20261006 + m)
    x = rng.integers(-128, 128, size=(m, k), dtype=np.int16).astype(np.int8)
    w = rng.integers(-128, 128, size=(TILE_N, k), dtype=np.int16).astype(np.int8)
    raw = w.reshape(4, 4, 8, k // 8, 8).transpose(0, 1, 3, 2, 4).copy().reshape(-1)
    cols = g * m
    small_e = np.zeros((k, cols), dtype=np.int8)
    for token in range(m):
        small_e[:, g * token] = x[token]
    b = np.zeros(g * k * cols, dtype=np.int8)
    flat_e = small_e.reshape(-1)
    for lane in range(g):
        width = 8 * cols - lane
        for j in range(k // 8):
            start = 8 * lane * cols + lane + j * 8 * g * cols
            source = j * 8 * cols
            b[start:start + width] = flat_e[source:source + width]
    b = b.reshape(g * k, cols)
    expected_b = np.zeros_like(b)
    for j in range(k // 8):
        for lane in range(g):
            for token in range(m):
                expected_b[8 * g * j + 8 * lane:8 * g * j + 8 * lane + 8,
                           g * token + lane] = x[token, 8 * j:8 * j + 8]
    if not np.array_equal(b, expected_b):
        raise AssertionError("bulk initializer differs from independent block diagonal B")
    y = np.zeros((m, TILE_N), dtype=np.int64)
    cover = np.zeros(raw.size, dtype=np.uint8)
    output_cover = np.zeros(y.shape, dtype=np.uint8)
    for lane0 in range(0, 4, g):
        a = np.empty((32, g * k), dtype=np.int8)
        for q in range(32):
            indices = (_q_source(q, k) + lane0 * 8
                       + np.arange(k // 8)[:, None] * 64 + np.arange(8 * g))
            if int(indices.max()) >= raw.size:
                raise AssertionError("raw packet assembly exceeds the resident tile")
            a[q] = raw[indices].reshape(-1)
            cover[indices.reshape(-1)] += 1
            expected_a = w[4 * q + lane0:4 * q + lane0 + g].reshape(g, k // 8, 8)
            expected_a = expected_a.transpose(1, 0, 2).reshape(-1)
            if not np.array_equal(a[q], expected_a):
                raise AssertionError("A assembly source coordinates differ from W[n,k]")
        c = a.astype(np.int64) @ b.astype(np.int64)
        for token in range(m):
            for q in range(32):
                y[token, 4 * q + lane0:4 * q + lane0 + g] = c[q, g * token:g * token + g]
                output_cover[token, 4 * q + lane0:4 * q + lane0 + g] += 1
    expected = x.astype(np.int64) @ w.astype(np.int64).T
    if not (np.array_equal(y, expected) and np.all(cover == 1) and np.all(output_cover == 1)):
        raise AssertionError("packet fold, source byte cover, or output cover failed")
    return dict(M=m, K=k, group_lanes=g, checked_output_elements=int(y.size),
                checked_raw_weight_bytes=int(raw.size), checked_B_bytes=int(b.size),
                B_all_bytes_exact=True, A_all_bytes_exact=True, I64_dot_exact=True,
                raw_byte_copy_cover_once=True, output_element_cover_once=True,
                int32_absolute_bound=k * 128 * 128,
                expected_sha256=hashlib.sha256(expected.astype('<i4').tobytes()).hexdigest())


def generate(path, m, n, k):
    """Emit one checked unified NPU program, returning its memory/ISA contract.

    Caller image requirements: X at e.X_BASE, W at WEIGHT_BASE, and real zero
    bytes at ZERO_BASE. No image is read or changed by this function. Only
    program.bin, system.json and packet_fold_contract.json are written.
    """
    path = Path(path)
    g = _group(m)
    if k != 4096 or n <= 0 or n % TILE_N:
        raise ValueError("bounded control requires K=4096 and positive N divisible by128")
    if (path / 'program.bin').exists() or (path / 'system.json').exists():
        raise ValueError("refusing to overwrite a generated program")
    cols, raw_bytes = g * m, TILE_N * k
    b_bytes, e_bytes = g * k * cols, k * cols
    l1_ranges = _ranges([
        ('X', X_L1, m * k), ('raw_W', RAW_L1, raw_bytes),
        ('assembled_A', A_L1, 32 * g * k), ('expanded_B', B_L1, b_bytes),
        ('initializer_E', E_L1, e_bytes), ('matrix_C', C_L1, 32 * cols * 4)], 4 << 20)
    l2_ranges = _ranges([
        ('canonical_Y', OUTPUT_L2, m * n * 4), ('zero_constant', ZERO_L2, b_bytes),
        ('X', INPUT_L2, m * k), ('raw_weight_slots', WEIGHT_L2, CORES * 2 * raw_bytes)], 8 << 20)
    lp6_ranges = _ranges([
        ('X', e.X_BASE, m * k), ('zero_constant', ZERO_BASE, b_bytes),
        ('packed_W', WEIGHT_BASE, n * k)], 1 << 34)

    p = e.Program()
    x_ready = p.dma('input.load', [], e.X_BASE, INPUT_L2, m * k)
    zero_ready = p.dma('input.zero_load', [], ZERO_BASE, ZERO_L2, b_bytes)
    previous = [p.core('input.packet_expand', [x_ready, zero_ready], core, _prepare(m, k, g))
                for core in range(CORES)]
    history = [[] for _ in range(CORES)]
    tiles = n // TILE_N
    weight_events = []
    assignments = []
    for batch in range((tiles + CORES - 1) // CORES):
        staged = []
        for core in range(CORES):
            tile = batch * CORES + core
            if tile >= tiles:
                continue
            slot = WEIGHT_L2 + (core * 2 + batch % 2) * raw_bytes
            reuse = [history[core][batch - 2]] if batch >= 2 else []
            event = p.dma('weight.load', reuse, WEIGHT_BASE + tile * raw_bytes, slot, raw_bytes)
            weight_events.append(event)
            staged.append((core, tile, slot, event))
        for core, tile, slot, weight_ready in staged:
            event = p.core('npu.packet_fold', [previous[core], weight_ready], core,
                           _compute(m, n, k, g, tile * TILE_N, slot))
            previous[core] = event
            history[core].append(event)
            assignments.append(dict(tile=tile, n_begin=tile * TILE_N, core=core, slot=slot,
                                    weight_event=weight_ready, consumer_event=event))
    final = p.barrier('done', previous)
    p.finish(path)
    decoded = isa.decode_program((path / 'program.bin').read_bytes())
    records = {c['id']: c for c in decoded['commands']}
    weight_loads = [records[event] for event in weight_events]
    if sum(c['row_bytes'] * c['rows'] for c in weight_loads) != n * k:
        raise AssertionError('external W traffic is not exactly one copy')
    for tile, c in enumerate(weight_loads):
        if c['lp6_address'] != WEIGHT_BASE + tile * raw_bytes or c['rows'] != 1:
            raise AssertionError('raw W stripes are not a contiguous one-pass partition')
    if any(c['kind'] not in ('gdma_load', 'core_block', 'barrier') for c in records.values()):
        raise AssertionError('unexpected PIM or extra LP6 output command')
    contract = dict(
        schema='lp6.pa_ha.packet_fold.v1', M=m, N=n, K=k, group_lanes=g,
        scope='INT8 X and one packed INT8 W in LP6 to canonical INT32 Y in L2',
        quantization_included=False, scale_included=False, new_hardware=False,
        read_view_alias=False, actual_dram_mapping_supplied_by_caller=True,
        weight_storage_order=['N/32', 4, 'K/8', 8, 8],
        input_lp6=e.X_BASE, weight_lp6=WEIGHT_BASE, zero_lp6=ZERO_BASE,
        zero_lp6_bytes=b_bytes, zero_source_must_be_all_zero=True, output_l2=OUTPUT_L2,
        raw_tile_bytes=raw_bytes, raw_slot_count=8, l1_ranges=l1_ranges,
        l2_ranges=l2_ranges, lp6_ranges=lp6_ranges,
        l1_peak_bytes=max(r['offset'] + r['bytes'] for r in l1_ranges),
        l2_peak_bytes=max(r['offset'] + r['bytes'] for r in l2_ranges),
        initializer='paid_zero_LP6_then_E_transpose_and_shifted_bulk_copy',
        expanded_B_bytes_per_core=b_bytes, initializer_E_bytes_per_core=e_bytes,
        B_generated_once_per_core=True, B_reused_across_all_weight_tiles=True,
        raw_W_reused_across_g2_lane_groups=True,
        LP6_weight_read_bytes=n * k, LP6_input_read_bytes=m * k,
        LP6_zero_read_bytes=b_bytes, LP6_output_write_bytes=0,
        weight_read_passes=1, packet_assembly_bytes=n * k,
        matrix_k=g * k, matrix_rows=32, matrix_columns=cols,
        matrix_compute_slices=tiles * (4 // g) * (g * k // 32),
        matrix_operations=tiles * (4 // g),
        all_dynamic_operand_and_scatter_costs_included=True,
        LDMA_I_max_chunk_bytes=LDMA_CHUNK, cores=CORES,
        tiles=tiles, assignments=assignments, final_join_event=final,
        per_core_final_consumers=previous, command_count=len(records),
        program_sha256=e.sha(path / 'program.bin'), generator_sha256=e.sha(Path(__file__)),
        static_algebra=static_algebra(m, k),
        native_validation_performed=False,
        limitations=['Full-K operand sizes follow the existing simulator/ISA; no new operand-buffer capacity is introduced.',
                     'CPU layout/algebra and decoded records are checked; native numerical/timing validation is the caller\'s separate step.'])
    e.save(path / 'packet_fold_contract.json', contract)
    return contract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--m', required=True, type=int)
    ap.add_argument('--n', default=14336, type=int)
    ap.add_argument('--k', default=4096, type=int)
    args = ap.parse_args()
    result = generate(args.output, args.m, args.n, args.k)
    print(f"Generated {result['command_count']} commands; g{result['group_lanes']} CPU algebra exact. No native run.")


if __name__ == '__main__':
    main()
