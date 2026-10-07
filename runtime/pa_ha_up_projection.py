#!/usr/bin/env python3
"""Up-only, burst-preserving PA->HA permutations, with paid software layouts.

W8A8 boundary: dense INT8 X in LP6 -> canonical INT32 Y in L2. No read-view,
address alias, controller byte permutation, quant/dequant, or free broadcast.
`prepare` writes artifacts only. `run` is explicit and runs one prepared case.
Candidates are finite controls, not globally optimal mappings or kernels.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import numpy as np
import dual_view_experiment as e

isa = e.isa
ROOT, LP6, SIM = e.ROOT, e.LP6, e.SIM
N, K = 14336, 4096
W_BASE, X_BASE, INDEX_BASE, ZERO_BASE = 128 << 20, 4096, 1 << 20, 2 << 20
X_L2, INDEX_L2, RAW_L2 = 0x1E0000, 0x1D0000, 2 << 20
WEIGHT_ROW = 512
MAX_GDMA_BYTES = (1 << 20) - 64
FIELDS = {"ch": 3, "bg": 2, "ba": 2, "co": 6, "ro": 15, "ra": 1}
DEFAULT_BITS = {"ch": (5, 6, 7), "bg": (8, 9), "ba": (16, 17),
                "co": tuple(range(10, 16)), "ro": tuple(range(18, 33)), "ra": (33,)}
PIM_BITS = {"ch": (5, 15, 16), "bg": (18, 19), "ba": (26, 17),
            "co": (6, 7, 8, 9, 10, 20),
            "ro": (21, 22, 11, 12, 13, 14, 23, 24, 25, 27, 28, 29, 30, 31, 32), "ra": (33,)}
SHARED_BITS = dict(PIM_BITS, ch=(5, 17, 18), bg=(15, 16), ba=(26, 19))
MAPS = {"npu": DEFAULT_BITS, "pim": PIM_BITS, "shared": SHARED_BITS}


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def sha(path):
    return e.sha(path)


def arr_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class Mapping:
    name: str

    @property
    def bits(self):
        return MAPS[self.name]

    def validate(self):
        assert all(len(self.bits[f]) == width for f, width in FIELDS.items())
        assert sorted(b for seq in self.bits.values() for b in seq) == list(range(5, 34))

    def encode(self, *, ch=0, bg=0, ba=0, co=0, ro=0, ra=0):
        fields = dict(ch=ch, bg=bg, ba=ba, co=co, ro=ro, ra=ra)
        value = 0
        for field, bits in self.bits.items():
            v = fields[field]
            for j, bit in enumerate(bits):
                value = value | (((v >> j) & 1) << bit)
        return value

    def decode(self, pa):
        return {field: sum(((pa >> bit) & 1) << j for j, bit in enumerate(bits))
                for field, bits in self.bits.items()}

    def packet(self, row, tile, pu, sc, odd=0):
        return self.encode(ch=sc, bg=pu // 2, ba=2 * (pu % 2) + odd,
                           ro=row + tile // 64, co=tile % 64)

    def output_n(self, sc, pu, stream, acc, lane):
        a = stream * 8 + acc
        if self.name == "pim":
            return (a * 64 + pu * 8 + sc) * 4 + lane
        assert self.name == "shared"
        return a * 256 + (pu % 2) * 128 + (sc // 2) * 32 + (pu // 2) * 8 + (sc % 2) * 4 + lane

    def descriptor(self):
        return dict(schema="lp6.pa_ha_bits.v1", name=self.name, offset_bits=list(range(5)),
                    fields_lsb_first=self.bits, bijection_bits=list(range(34)),
                    controller_byte_permutation=False, read_view=False,
                    storage="npu_nk32" if self.name == "npu" else "n32_ng8_k8_n8_k8")

    def config(self, dest):
        src = LP6 / "lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini"
        text = src.read_text()
        assert not re.search(r"(?m)^address_bits_", text)
        lines = "\n".join(f"address_bits_{f} = " + ",".join(map(str, self.bits[f])) for f in FIELDS)
        text, count = re.subn(r"(?m)^(address_mapping\s*=.*)$", r"\1\n" + lines, text)
        assert count == 1
        Path(dest).write_text(text)


class Program(e.Program):
    def __init__(self):
        super().__init__()
        self.gdma_payload = {"load": 0, "store": 0}

    def dma(self, name, deps, lp6, l2, size, **kwargs):
        rows = kwargs.get("rows", 1)
        self.gdma_payload["load" if kwargs.get("load", True) else "store"] += size * rows
        return super().dma(name, deps, int(lp6), int(l2), int(size), **kwargs)

    def finish(self, path, kernel=None):
        # Each public ISA encoder checked its fields when the record was added.
        # Avoid materializing a second millions-of-primitives decoded object.
        self.words.append(isa.PROGRAM_END)
        self.words[:5] = [isa.PROGRAM_MAGIC, isa.PROGRAM_VERSION, len(self.words), len(self.symbols), self.core_words]
        path.mkdir(parents=True, exist_ok=True)
        (path / "program.bin").write_bytes(struct.pack(f"<{len(self.words)}I", *self.words))
        manifest = dict(schema="plena.v2.unified_program.isa_v2.0", program="program.bin",
                        program_word_count=len(self.words), command_count=len(self.symbols),
                        core_instruction_count=self.core_words, command_symbols=self.symbols,
                        scheduling_policy="compiler_static", logical_to_physical=[0, 1, 2, 3],
                        l2_bytes_required=8 << 20, l2_regions=[])
        if kernel is not None:
            manifest["pim_kernel"] = kernel
        save(path / "system.json", manifest)
        save(path / "traffic_static.json", dict(logical_GDMA_bytes=self.gdma_payload,
            note="Payload only; controller transactions and physical billing are measured separately."))


def const(reg, value):
    return isa.load_constant(reg, int(value))


def copy2d(op, dst, src, size, rows=1, src_stride=None, dst_stride=None):
    src_stride = size if src_stride is None else src_stride
    dst_stride = size if dst_stride is None else dst_stride
    return [isa.c_set(isa.SET_BYTES, int(size)), isa.c_set(isa.SET_ROWS, int(rows)),
            isa.c_set(isa.SET_SRC_STRIDE, int(src_stride)), isa.c_set(isa.SET_DST_STRIDE, int(dst_stride))] + \
        const(14, dst) + const(15, src) + [isa.ldma_r(op, 14, 15)]


def affine_runs(sources, destinations, row_bytes):
    """Exact, positive-stride grouping; does no read overfetch or byte packing."""
    assert len(sources) == len(destinations)
    i, count = 0, len(sources)
    while i < count:
        j, ss, ds = i + 1, 0, 0
        if j < count:
            ss, ds = int(sources[j] - sources[i]), int(destinations[j] - destinations[i])
            # LDMA's zero stride means the default row_bytes, not broadcast.
            # Its strided shapes also reject overlapping rows.  Repeated
            # sources must therefore be separate, fully charged copies.
            if ss < row_bytes or ds < row_bytes or ss >= 1 << 24 or ds >= 1 << 24:
                ss = ds = 0
            else:
                j += 1
                while j < count and j - i < 65536 and sources[j] - sources[j - 1] == ss and destinations[j] - destinations[j - 1] == ds:
                    j += 1
        rows = j - i
        # For invalid slope the run must contain only the first row.
        if rows == 1:
            ss = ds = row_bytes
        if rows > 1 and ss == ds == row_bytes:
            first = i
            maximum = MAX_GDMA_BYTES // row_bytes
            while first < j:
                take = min(j - first, maximum)
                yield int(sources[first]), int(destinations[first]), row_bytes * take, 1, row_bytes * take, row_bytes * take
                first += take
        else:
            yield int(sources[i]), int(destinations[i]), row_bytes, rows, ss, ds
        i = j


def local_gather(sources, destinations, row_bytes):
    words = []
    consumed = 0
    for src, dst, size, rows, ss, ds in affine_runs(sources, destinations, row_bytes):
        # Check the actual LDMA shape semantics, including the stride-0
        # default, against every requested logical row before emitting ISA.
        effective_ss, effective_ds = ss or size, ds or size
        assert effective_ss >= size and effective_ds >= size
        assert size % row_bytes == 0
        offsets = np.arange(size // row_bytes, dtype=np.int64) * row_bytes
        for row in range(rows):
            take = len(offsets)
            assert np.array_equal(np.asarray(sources[consumed:consumed + take]), src + row * effective_ss + offsets)
            assert np.array_equal(np.asarray(destinations[consumed:consumed + take]), dst + row * effective_ds + offsets)
            consumed += take
        words += copy2d(9, dst, src, size, rows, ss, ds)
    assert consumed == len(sources)
    return words


def packet_dmas(p, name, deps, addresses, l2_base, *, load):
    """Each listed 32B PA packet maps to exactly one contiguous raw L2 slot."""
    sources = np.asarray(addresses, dtype=np.int64)
    destinations = l2_base + np.arange(len(sources), dtype=np.int64) * 32
    events = []
    for src, dst, size, rows, ss, ds in affine_runs(sources, destinations, 32):
        events.append(p.dma(name, deps, src, dst, size, load=load, rows=rows, lp6_stride=ss, l2_stride=ds))
    return events


def kernel_spec():
    return dict(channel=0, rank=0, row=WEIGHT_ROW, start_column=0, bank_set=1,
                weight_bits=8, activation_bits=8, scale_bits=0, input_register_count=8,
                output_register_count=8, subchannels=8, weight_layout="even-v1", bankset_pingpong_lead=0)


def transpose_words(raw_l1, weight_l1):
    """4x actual 8-by-8 transposes per K8; RF index vectors 2..5 are I32[16]."""
    scratch = 0x60000
    body = []
    for ng in range(4):
        body += const(0, raw_l1 + ng * K * 8) + const(1, scratch)
        body += [isa.c_loop_begin(K // 8)]
        body += [isa.c_set(isa.SET_VL, 64), isa.v_mem(False, isa.I8, 0, 0, 0), isa.c_set(isa.SET_VL, 16)]
        for q in range(4):
            body += [isa.v_op(isa.V_OP_NAMES.index("GATHER"), isa.I8, 6 + q, 0, 2 + q)]
        for q in range(4):
            body += [isa.v_mem(True, isa.I8, 6 + q, 1, q * 16)]
        body += [isa.c_addi(0, 0, 64), isa.c_addi(1, 1, 64), isa.c_loop_end(), isa.c_sync(isa.SYNC_VPU)]
        body += copy2d(9, weight_l1 + ng * 8, scratch, 8, K, 8, 32)
        body += [isa.c_sync(isa.SYNC_LDMA)]
    return body


def npu_program(path, m, mapping, batch_tiles=4):
    p = Program()
    x = p.dma("input.load", [], X_BASE, X_L2, m * K)
    index = p.dma("transpose.index.load", [], INDEX_BASE, INDEX_L2, 256) if mapping.name != "npu" else None
    tile_bytes, count = K * 32, N // 128
    previous, history = [x] * 4, [[] for _ in range(4)]
    for batch in range((count + batch_tiles - 1) // batch_tiles):
        all_items = []
        for core in range(4):
            items = []
            for j in range(batch * batch_tiles, min(count, (batch + 1) * batch_tiles)):
                block = 4 * j + core
                slot = RAW_L2 + core * (3 << 19) + (j % 8) * tile_bytes
                reuse = 8 // batch_tiles
                deps = [history[core][batch - reuse]] if batch >= reuse else []
                event = p.dma("weight.load", deps, W_BASE + block * tile_bytes, slot, tile_bytes)
                items.append((block * 32, slot, event))
            all_items.append(items)
        for core, items in enumerate(all_items):
            body = isa.ldma_i(0, 0, X_L2, m * K)
            if mapping.name != "npu":
                body += isa.ldma_i(0, 0x20000, INDEX_L2, 256) + [isa.c_set(isa.SET_VL, 16)]
                for q in range(4):
                    body += const(0, 0x20000 + q * 64) + [isa.v_mem(False, isa.I32, 2 + q, 0, 0)]
            body += isa.ldma_i(0, 0x40000, items[0][1], tile_bytes)
            for t, (n0, slot, _) in enumerate(items):
                if t:
                    body += [isa.c_sync(isa.SYNC_EVENT, 1)]
                if mapping.name == "npu":
                    wp = 0x40000 + (t % 2) * tile_bytes
                else:
                    wp = 0x80000 + (t % 2) * tile_bytes
                    body += transpose_words(0x40000, wp)
                body += const(0, wp) + isa.m_ld(0, 0, 32, K)
                body += const(1, 0) + isa.m_ld(2, 1, m, K)
                body += [isa.m_mma(False, False, K // 32)]
                if t + 1 < len(items):
                    raw_dst = 0x40000 + ((t + 1) % 2) * tile_bytes if mapping.name == "npu" else 0x40000
                    body += isa.ldma_i(10, raw_dst, items[t + 1][1], tile_bytes, event=1)
                body += const(2, 0xC0000) + isa.m_wb(2, 2, m, 32)
                body += copy2d(5, 0xC0000, n0 * 4, 128, m, 128, N * 4)
            deps = [previous[core]] + [item[2] for item in items]
            if index is not None:
                deps.append(index)
            event = p.core("npu.compute", deps, core, body)
            history[core].append(event)
            previous[core] = event
    done = p.barrier("done", previous)
    p.finish(path)
    return dict(endpoint_event=done, weight_lp6_bytes=N * K, software_transpose=mapping.name != "npu",
                core_assignment="interleaved N32 tiles", batch_tiles=batch_tiles)


def feed_local_pim(p, m, mapping, actrow, xdep):
    previous = xdep
    for token in range(m):
        packets = [(mapping.packet(actrow, token * (K // 32) + pk, pu, sc), pk * 32)
                   for sc in range(8) for pu in range(8) for pk in range(K // 32)]
        packets.sort()
        addresses = np.array([a for a, _ in packets], dtype=np.int64)
        x_offsets = np.array([x for _, x in packets], dtype=np.int64)
        assert len(np.unique(addresses)) == len(addresses)
        ready = []
        share = len(packets) // 4
        for core in range(4):
            offsets = x_offsets[core * share:(core + 1) * share]
            body = isa.ldma_i(0, 0, X_L2 + token * K, K)
            body += local_gather(offsets, 0x40000 + np.arange(share) * 32, 32)
            body += copy2d(1, 0x40000, RAW_L2 + core * share * 32, share * 32)
            ready.append(p.core("activation.pack", [previous], core, body))
        packed = p.barrier("activation.packed", ready)
        stores = packet_dmas(p, "activation.scatter", [packed], addresses, RAW_L2, load=False)
        previous = p.barrier("activation.token_ready", stores)
    return p.barrier("activation.ready", [previous])


def output_gather(p, mapping, deps, packets, m, *, native=False, token=0):
    """Read complete packets, then paid local permutation (and reduction)."""
    share = N // 4
    by_core = [[] for _ in range(4)]
    for pa, values in packets:
        owners = {n % N // share for n, _, _ in values}
        assert len(owners) == 1
        by_core[next(iter(owners))].append((pa, values))
    done, l2_offset = [], RAW_L2
    for core, entries in enumerate(by_core):
        entries.sort(key=lambda x: x[0])
        addresses = np.array([pa for pa, _ in entries], dtype=np.int64)
        assert len(np.unique(addresses)) == len(addresses)
        size = len(entries) * 32
        assert l2_offset + size <= 8 << 20
        reads = packet_dmas(p, "output.packet_gather", deps, addresses, l2_offset, load=True)
        gathered = p.barrier("output.raw_ready", reads)
        pairs = []
        for packet_index, (_, values) in enumerate(entries):
            for n, offset, split in values:
                if native:
                    target = split * share + n % share
                else:
                    target = (n // N) * share + n % share
                pairs.append((target, packet_index * 32 + offset))
        pairs.sort()
        expected_count = (32 if native else m) * share
        assert [t for t, _ in pairs] == list(range(expected_count))
        canonical_l1 = 2 << 20
        assert size <= canonical_l1 and canonical_l1 + expected_count * 4 <= 4 << 20
        body = copy2d(0, 0, l2_offset, size)
        body += local_gather(np.array([s for _, s in pairs]), canonical_l1 + np.arange(expected_count) * 4, 4)
        if native:
            body += [isa.c_set(isa.SET_VL, 16)] + const(0, canonical_l1)
            body += [isa.c_loop_begin(share // 16)]
            body += [isa.v_mem(False, isa.I32, 0, 0, 0)]
            for split in range(1, 32):
                body += [isa.v_mem(False, isa.I32, 1, 0, split * share * 4), isa.v_op(0, isa.I32, 0, 0, 1)]
            body += [isa.v_mem(True, isa.I32, 0, 0, 0), isa.c_addi(0, 0, 64), isa.c_loop_end(), isa.c_sync(isa.SYNC_VPU)]
            body += copy2d(1, canonical_l1, (token * N + core * share) * 4, share * 4)
        else:
            body += copy2d(5, canonical_l1, core * share * 4, share * 4, m, share * 4, N * 4)
        done.append(p.core("npu.reduce" if native else "output.reorder", [gathered], core, body))
        l2_offset += size
    return p.barrier("token.done" if native else "done", done)


def pim_program(path, m, mapping):
    assert mapping.name in ("pim", "shared")
    p = Program()
    xdep = p.dma("input.load", [], X_BASE, X_L2, m * K)
    actrow, outrow = WEIGHT_ROW + 448, WEIGHT_ROW + 448 + m * 8
    adep = feed_local_pim(p, m, mapping, actrow, xdep)
    commands, packets = [], []
    for token in range(m):
        for sc in range(8):
            def emit(op, **kw):
                return isa.pim_primitive(op, sc, **kw)
            for stream in range(7):
                commands += [emit(isa.PIM_CLEAR, accumulator=i) for i in range(8)]
                for sweep in range(K // 256):
                    a = token * (K // 8) + sweep * 32
                    commands += [emit(isa.PIM_LOAD_ACTIVATION, activation=a + i) for i in range(0, 32, 4)]
                    w = stream * 8 * (K // 8) + sweep * 8 * 32
                    commands.append(emit(isa.PIM_GEMV_STREAM, weight=w, activation=a, accumulator=0, inputs=32, outputs=8,
                        weight_input_stride=1, weight_output_stride=32, activation_input_stride=1, accumulator_output_stride=1))
                for pair in range(4):
                    out = (token * 7 + stream) * 4 + pair
                    commands.append(emit(isa.PIM_STORE, accumulator=pair * 2, output=out))
                    for pu in range(8):
                        values = [(token * N + mapping.output_n(sc, pu, stream, pair * 2 + parity, lane), lane * 8 + parity * 4, 0)
                                  for lane in range(4) for parity in range(2)]
                        packets.append((mapping.packet(outrow, out, pu, sc), values))
    pd = p.add("pim.compute", lambda i: isa.pim_record(i, [adep], commands))
    done = output_gather(p, mapping, [pd], packets, m)
    p.finish(path, kernel_spec())
    return dict(endpoint_event=done, activation_row=actrow, output_row=outrow, k_splits=1,
                PIM_weight_packets=N * K // 32, output_packets=len(packets), all_K_local=True)


def native_pim_program(path, m, mapping):
    assert mapping.name == "npu"
    p = Program()
    actrow, outrow = WEIGHT_ROW + N // 64, WEIGHT_ROW + N // 64 + 16
    previous = p.dma("onehot.zero.load", [], ZERO_BASE, 6 << 20, 256 << 10)
    for token in range(m):
        xd = p.dma("input.load", [previous], X_BASE + token * K, X_L2, K)
        body = isa.ldma_i(0, 0, X_L2, K) + isa.ldma_i(0, 0x40000, 6 << 20, 256 << 10)
        for r in range(8):
            for bs in range(2):
                for bg in range(4):
                    for sc in range(8):
                        for c in range(4):
                            src = bs * 2048 + bg * 8 + sc + c * 32
                            dst = 0x40000 + (r * 2 + bs) * 16384 + bg * 256 + sc * 32 + c * 8 + r
                            body += copy2d(9, dst, src, 1, 16, 128, 1024)
        body += isa.ldma_i(1, 0x40000, 5 << 20, 256 << 10)
        packed = p.core("activation.onehot_pack", [xd], 0, body)
        writes = []
        for r in range(8):
            for bs in range(2):
                for parity in range(2):
                    writes.append(p.dma("activation.scatter", [packed], mapping.packet(actrow, r * 32 + bs * 16, parity, 0, bs),
                        (5 << 20) + (r * 2 + bs) * 16384, 16384, load=False))
        ready = p.barrier("activation.ready", writes)
        commands, packets = [], []
        for j in range(N // 64):
            for sc in range(8):
                def emit(op, **kw):
                    return isa.pim_primitive(op, sc, **kw)
                commands += [emit(isa.PIM_CLEAR, accumulator=r) for r in range(8)]
                for bs in range(2):
                    for r in range(8):
                        for chunk in (0, 32):
                            a = r * 128 + bs * 64 + chunk
                            commands += [emit(isa.PIM_LOAD_ACTIVATION, activation=a + i, bank_set=bs) for i in range(0, 32, 4)]
                            commands.append(emit(isa.PIM_GEMV_STREAM, weight=j * 64 + chunk, activation=a, accumulator=r, inputs=32, outputs=1,
                                weight_input_stride=1, weight_output_stride=0, activation_input_stride=1, accumulator_output_stride=1, bank_set=bs))
                for r in range(8):
                    out = j * 8 + r
                    commands.append(emit(isa.PIM_STORE, accumulator=r, output=out))
                    for pu in range(8):
                        values = [(j * 64 + (pu % 2) * 32 + lane * 8 + r, lane * 8, sc * 4 + pu // 2) for lane in range(4)]
                        packets.append((mapping.packet(outrow, out, pu, sc), values))
        pd = p.add("pim.compute", lambda i: isa.pim_record(i, [ready], commands))
        previous = output_gather(p, mapping, [pd], packets, 1, native=True, token=token)
    done = p.barrier("done", [previous])
    p.finish(path, kernel_spec())
    return dict(endpoint_event=done, activation_row=actrow, output_row=outrow,
                k_splits=32, onehot_passes=8, onehot_input_generation_paid=True,
                partial_packet_bytes_per_token=N * 256, all_K_local=False)


def weight_pack(w, mapping):
    if mapping.name == "npu":
        return np.ascontiguousarray(w.reshape(N // 32, 32, K).transpose(0, 2, 1))
    return np.ascontiguousarray(w.reshape(N // 32, 4, 8, K // 8, 8).transpose(0, 1, 3, 2, 4))


def weight_proof(w, packed, mapping):
    mapping.validate()
    assert packed.nbytes == w.nbytes == N * K
    if mapping.name == "npu":
        assert np.array_equal(packed.transpose(0, 2, 1).reshape(N, K), w)
        return dict(storage_inverse_exact=True, packet_MAC_shape="N32 at one K; needs onehot and 32-way reduction")
    assert np.array_equal(packed.transpose(0, 1, 3, 2, 4).reshape(N, K), w)
    flat = packed.view(np.uint8).ravel()
    coverage = np.zeros(N * K // 32, dtype=np.uint8)
    kk = np.arange(K // 8, dtype=np.int64)
    for sc in range(8):
        for pu in range(8):
            for stream in range(7):
                for acc in range(8):
                    tiles = stream * 8 * (K // 8) + (kk // 32) * 256 + acc * 32 + kk % 32
                    pa = mapping.packet(WEIGHT_ROW, tiles, pu, sc)
                    assert np.all((pa >= W_BASE) & (pa + 32 <= W_BASE + N * K))
                    packet_indices = (pa - W_BASE) // 32
                    coverage[packet_indices] += 1
                    ns = [mapping.output_n(sc, pu, stream, acc, lane) for lane in range(4)]
                    expected = w[ns].reshape(4, K // 8, 8).transpose(1, 0, 2).reshape(-1, 32).view(np.uint8)
                    actual = flat[(pa - W_BASE)[:, None] + np.arange(32)]
                    assert np.array_equal(actual, expected)
    assert np.all(coverage == 1)
    return dict(storage_inverse_exact=True, all_1835008_packets_payload_exact=True,
                every_weight_byte_once=True, all_K_local=True, output_coverage_per_PU=224,
                physical_row_count=448, packet_bytes=32, packet_axes="4N x 8K")


def transpose_proof(packed, w):
    if packed.ndim != 5:
        return None
    index = np.arange(64, dtype=np.int32).reshape(8, 8).T.ravel()
    raw = packed.reshape(N // 32, 4, K // 8, 64)
    converted = raw[..., index].reshape(N // 32, 4, K, 8).transpose(0, 2, 1, 3).reshape(N // 32, K, 32)
    expected = w.reshape(N // 32, 32, K).transpose(0, 2, 1)
    assert np.array_equal(converted, expected)
    return dict(all_weight_bytes_exact=True, source_I8_register_bytes=64,
                index_I32_register_elements=16, gathers_per_64B=4, registers_used=10,
                local_merge_per_N32=dict(row_bytes=8, rows=4096, src_stride=8, dst_stride=32, copies=4),
                software_only=True, no_controller_reassembly=True)


def prepare(args):
    path = args.output.resolve()
    path.mkdir(parents=True, exist_ok=False)
    mapping = Mapping(args.layout)
    if args.weight_cache:
        w = np.load(args.weight_cache)
        assert w.dtype == np.int8 and w.shape == (N, K)
        provenance = dict(source=str(args.weight_cache.resolve()), source_file_sha256=sha(args.weight_cache), int8_sha256=arr_sha(w))
        save(path / "weight_provenance.json", provenance)
    else:
        w = e.load_weight(args.model, path, False, N, K)
    m = args.m
    assert m in (1, 2, 4, 8, 16)
    x = np.random.default_rng(4096).integers(-127, 128, (m, K), dtype=np.int8)
    expected = x.astype(np.int32) @ w.astype(np.int32).T
    np.savez(path / "reference.npz", x=x, expected=expected)
    packed = weight_pack(w, mapping)
    proof = weight_proof(w, packed, mapping)
    trans = transpose_proof(packed, w)
    save(path / "static_weight_proof.json", dict(mapping=mapping.descriptor(), weight=proof, software_transpose=trans))
    image = path / "lp6.bin"
    index = np.arange(64, dtype=np.int32).reshape(8, 8).T.copy().ravel()
    with image.open("wb") as f:
        f.truncate(W_BASE + N * K)
        f.seek(W_BASE); f.write(packed.tobytes())
        f.seek(X_BASE); f.write(x.tobytes())
        f.seek(INDEX_BASE); f.write(index.tobytes())
    mapping.config(path / "dram.ini")
    settings = re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4", (SIM.parent / "plena_settings.toml").read_text())
    (path / "settings.toml").write_text(settings)
    if args.engine == "npu":
        details = npu_program(path, m, mapping, args.batch_tiles)
    elif mapping.name == "npu":
        details = native_pim_program(path, m, mapping)
    else:
        details = pim_program(path, m, mapping)
    case = dict(schema="lp6.pa_ha_up.v1", M=m, K=K, N=N, layout=args.layout, engine=args.engine,
                mapping=mapping.descriptor(), details=details, read_view=False,
                weight_base=W_BASE, weight_bytes=N * K, weight_copies=1,
                input_base=X_BASE, output_L2=0, measured_boundary="I8 X in LP6 -> canonical I32 Y in L2",
                input_sha256=arr_sha(x), expected_sha256=arr_sha(expected), weight_sha256=arr_sha(w),
                storage_sha256=arr_sha(packed), program_sha256=sha(path / "program.bin"),
                config_sha256=sha(path / "dram.ini"), source_sha256=sha(__file__),
                source_dependencies={str(Path(e.__file__)): sha(e.__file__), str(Path(isa.__file__)): sha(isa.__file__)},
                limitations=["Finite mappings and explicit kernels, not globally optimal.",
                    "PIM-format NPU uses paid V_GATHER; stronger prior g4/g2 kernels are not included.",
                    "Descriptor issue order may limit bandwidth independently of mapping.",
                    "No VA, TLB, page OS, FAS hardware, or hidden reorder buffer."])
    save(path / "case.json", case)
    print(json.dumps(dict(prepared=str(path), M=m, layout=args.layout, engine=args.engine,
                         program_bytes=(path / "program.bin").stat().st_size, native_runs=0)))


def run(args):
    path = args.case.resolve()
    assert not (path / "result.json").exists()
    case = json.loads((path / "case.json").read_text())
    assert sha(path / "program.bin") == case["program_sha256"]
    assert sha(path / "dram.ini") == case["config_sha256"]
    env = dict(os.environ, LD_PRELOAD=str(args.library.resolve()), LP6_DRAMSIM3_ROOT=str(LP6 / "lp6-pim-main"), LP6PIM_BANKSET_PINGPONG="0")
    command = [str(args.emulator.resolve()), "--settings", "settings.toml", "--system-program", "system.json",
               "--lp6-image", "lp6.bin", "--lp6-config", "dram.ini", "--lp6-size", "16GiB",
               "--command-report", "commands.json", "--sram-timing-out", "timing.json", "--log-level", "off"]
    assert not any("read-view" in x for x in command)
    provenance = dict(argv=command, environment={k: env[k] for k in ("LD_PRELOAD", "LP6_DRAMSIM3_ROOT", "LP6PIM_BANKSET_PINGPONG")},
                      emulator_sha256=sha(args.emulator), library_sha256=sha(args.library), source_sha256=sha(__file__),
                      program_sha256=sha(path / "program.bin"), config_sha256=sha(path / "dram.ini"))
    save(path / "invocation.json", provenance)
    started = time.monotonic()
    with (path / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    if result.returncode:
        raise RuntimeError((path / "run.log").read_text()[-8000:])
    expected = np.load(path / "reference.npz")["expected"]
    actual = np.fromfile(path / "l2_sram_dump.bin", dtype=np.int32, count=case["M"] * N).reshape(case["M"], N)
    np.save(path / "actual.npy", actual)
    assert np.array_equal(actual, expected), f"incorrect output: {np.count_nonzero(actual != expected)} elements"
    report = json.loads((path / "commands.json").read_text())
    events = {c["event_id"]: c for c in report["commands"]}
    endpoint = events[case["details"]["endpoint_event"]]["end_cycle"]
    assert endpoint == max(c["end_cycle"] for c in events.values())
    timing = json.loads((path / "timing.json").read_text())
    assert sha(args.emulator) == provenance["emulator_sha256"] and sha(args.library) == provenance["library_sha256"]
    output = dict(M=case["M"], N=N, K=K, layout=case["layout"], engine=case["engine"],
                  latency_cycles=endpoint, latency_us=endpoint / report["npu_clock_mhz"], all_outputs_int32_exact=True,
                  output_sha256=arr_sha(actual), lp6_bytes_read=timing["lp6_bytes_read"], lp6_bytes_written=timing["lp6_bytes_written"],
                  host_seconds=time.monotonic() - started, read_view=False, native_runs=1, **provenance)
    save(path / "result.json", output)
    for dump in path.glob("*sram_dump*.bin"):
        dump.unlink()
    print(json.dumps({k: output[k] for k in ("M", "layout", "engine", "latency_us", "all_outputs_int32_exact")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--layout", choices=MAPS, required=True)
    prep.add_argument("--engine", choices=("npu", "pim"), required=True)
    prep.add_argument("--m", type=int, choices=(1, 2, 4, 8, 16), required=True)
    prep.add_argument("--model", type=Path, default=Path("/home/jongjip/models/llama31-8b"))
    prep.add_argument("--weight-cache", type=Path)
    prep.add_argument("--batch-tiles", type=int, choices=(1, 2, 4), default=4)
    execute = sub.add_parser("run")
    execute.add_argument("--case", type=Path, required=True)
    execute.add_argument("--emulator", type=Path, required=True)
    execute.add_argument("--library", type=Path, required=True)
    execute.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    main()
