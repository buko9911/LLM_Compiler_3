#!/usr/bin/env python3
"""Finite, burst-preserving PA->HA plans for the fixed Up W8A8 experiment.

One unchanged weight image has layout [N/4,K/8,4,8].  All address descriptors
are permutations above the 32-byte packet offset.  A K split creates real
INT32 partial results; their NPU reduction is required, not supplied here.
This module generates plans and static proofs only, never runs a simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

N, K = 14336, 4096
W_BASE = 128 << 20
WEIGHT_ROW = 512
STORAGE = "n4_k8_n4_k8"
WIDTHS = {"ch": 3, "bg": 2, "ba": 2, "co": 6, "ro": 15, "ra": 1}
SLOTS = (("ch", 0), ("ch", 1), ("ch", 2), ("bg", 0), ("bg", 1), ("ba", 1))
ORDERS = ("c6", "c5b8", "c5_alln")


def deposit(value, positions):
    out = 0
    for i, bit in enumerate(positions):
        out = out | (((value >> i) & 1) << bit)
    return out


def extract(value, positions):
    out = 0
    for i, bit in enumerate(positions):
        out = out | (((value >> bit) & 1) << i)
    return out


def encode_pa(fields_lsb_first, *, ch=0, bg=0, ba=0, co=0, ro=0, ra=0):
    values = dict(ch=ch, bg=bg, ba=ba, co=co, ro=ro, ra=ra)
    out = 0
    for field, bits in fields_lsb_first.items():
        out = out | deposit(values[field], bits)
    return out


def decode_pa(fields_lsb_first, pa):
    return {field: extract(pa, bits) for field, bits in fields_lsb_first.items()}


def packet_address(plan, *, sc, pu, tile, row=None, bank_set=0):
    """Inverse HA mapping. tile is relative to the supplied physical row."""
    row = plan["weight_row"] if row is None else row
    return encode_pa(plan["fields_lsb_first"], ch=sc, bg=pu // 2,
                     ba=2 * (pu % 2) + bank_set, ro=row + tile // 64,
                     co=tile % 64, ra=0)


def raw_packet_pa(n4_group, k8_index, weight_base=W_BASE):
    return weight_base + (n4_group * (K // 8) + k8_index) * 32


def packet_coordinates(pa, weight_base=W_BASE):
    """Logical origin and in-packet coordinates; not an address alias."""
    rel = np.asarray(pa) - weight_base
    return dict(n4_group=rel // (K * 4), k8_index=(rel // 32) % (K // 8),
                n_lane=(rel % 32) // 8, k_lane=rel % 8)


def validate_fields(fields):
    assert all(len(fields[f]) == width for f, width in WIDTHS.items())
    assert sorted(bit for bits in fields.values() for bit in bits) == list(range(5, 34))
    assert fields["ba"][0] == 26 and fields["ra"] == [33]
    assert fields["ro"][9:] == [27, 28, 29, 30, 31, 32]
    assert encode_pa(fields, ro=WEIGHT_ROW) == W_BASE


def make_candidate(split_factor=1, order="c6", k_slots=None, k_window=0, n_window=0):
    assert split_factor in (1, 2, 4, 8, 16, 32)
    assert order in ORDERS
    s = split_factor.bit_length() - 1
    k_slots = tuple(range(s)) if k_slots is None else tuple(sorted(k_slots))
    assert len(k_slots) == s and len(set(k_slots)) == s
    assert all(0 <= slot < 6 for slot in k_slots)
    assert (k_window == 0 if s == 0 else 0 <= k_window <= 9 - s)
    n_selected_count = 6 - s
    assert 0 <= n_window <= 9 - n_selected_count
    k_selected = list(range(k_window, k_window + s))
    n_selected = list(range(n_window, n_window + n_selected_count))
    k_local = [b for b in range(9) if b not in k_selected]
    n_local = [b for b in range(12) if b not in n_selected]
    assert order == "c6" or len(k_local) >= 5
    if order == "c6":
        tile_bits = [5 + b for b in k_local] + [14 + b for b in n_local]
    elif order == "c5b8":
        tile_bits = ([5 + b for b in k_local[:5]] + [14 + b for b in n_local[:3]]
                     + [5 + b for b in k_local[5:]] + [14 + b for b in n_local[3:]])
    else:
        tile_bits = ([5 + b for b in k_local[:5]] + [14 + b for b in n_local]
                     + [5 + b for b in k_local[5:]])
    assert len(tile_bits) == 15
    fields = {"ch": [None] * 3, "bg": [None] * 2, "ba": [26, None],
              "co": tile_bits[:6], "ro": tile_bits[6:] + [27, 28, 29, 30, 31, 32], "ra": [33]}
    ki = ni = 0
    for slot, (field, bit) in enumerate(SLOTS):
        if slot in k_slots:
            fields[field][bit] = 5 + k_selected[ki]
            ki += 1
        else:
            fields[field][bit] = 14 + n_selected[ni]
            ni += 1
    validate_fields(fields)
    return dict(schema="lp6.pa_ha_up.candidate.v2", id=f"s{split_factor}_{order}_k{k_window}_n{n_window}_sel" + "".join(map(str, k_slots)),
                storage=STORAGE, split_factor=split_factor, order=order,
                k_selector_slots=list(k_slots), k_window=k_window, n_window=n_window,
                k_selected_bits=k_selected, n_selected_bits=n_selected,
                k_local_bits=k_local, n_local_bits=n_local, fields_lsb_first=fields,
                weight_base=W_BASE, weight_bytes=N * K, offset_bits=list(range(5)),
                read_view=False, controller_byte_permutation=False)


def generate_candidates(split_factors=(1, 2, 4, 8, 16, 32), orders=("c6", "c5b8"), window_policy="endpoints"):
    """A finite structural grid; `all` opens intermediate bit windows too."""
    assert window_policy in ("endpoints", "all")
    candidates, seen = [], set()
    for split in split_factors:
        s = split.bit_length() - 1
        ks = [0] if s == 0 else list(range(10 - s))
        ns = list(range(4 + s))
        if window_policy == "endpoints":
            ks, ns = sorted(set((ks[0], ks[-1]))), sorted(set((ns[0], ns[-1])))
        for order, slots, kw, nw in itertools.product(orders, itertools.combinations(range(6), s), ks, ns):
            if order != "c6" and split == 32:
                continue
            candidate = make_candidate(split, order, slots, kw, nw)
            signature = json.dumps(candidate["fields_lsb_first"], sort_keys=True)
            if signature not in seen:
                seen.add(signature)
                candidates.append(candidate)
    return candidates


def weight_tiles(plan, o, j):
    """Elementwise/broadcastable local output-group and K8 ordinals -> tile.

    Use o[:,None], j[None,:] for a Cartesian grid.  The arguments may be ints.
    """
    lk, lo, order = plan["local_k8_count"], plan["local_output_groups"], plan["order"]
    if order == "c6":
        return o * lk + j
    if order == "c5b8":
        return (o // 8) * (8 * lk) + (j // 32) * 256 + (o % 8) * 32 + j % 32
    padded_lo = 1 << (lo - 1).bit_length()
    return (j // 32) * (padded_lo * 32) + o * 32 + j % 32


def stream_batches(plan, accumulator_batch=8):
    """Common legal per-SC rectangles; activation id is token*Lk+input_begin.

    Every output batch must be cleared once, accumulated across all its input
    chunks, and stored only afterwards. c5b8 cannot cross its eight-output
    group boundary in one STREAM. This is a plan, not uncharged execution.
    """
    assert accumulator_batch in (8, 16)
    lo, lk = plan["local_output_groups"], plan["local_k8_count"]
    batch = min(accumulator_batch, 8) if plan["order"] == "c5b8" else accumulator_batch
    stride = lk if plan["order"] == "c6" else 32
    output = []
    for ob in range(0, lo, batch):
        no = min(batch, lo - ob)
        chunks = []
        for jb in range(0, lk, 32):
            ni = min(32, lk - jb)
            base = int(weight_tiles(plan, ob, jb))
            for oi in range(no):
                for ji in range(ni):
                    assert int(weight_tiles(plan, ob + oi, jb + ji)) == base + oi * stride + ji
            chunks.append(dict(input_begin=jb, inputs=ni, outputs=no, weight=base,
                               weight_input_stride=1, weight_output_stride=stride,
                               activation_input_stride=1, accumulator_output_stride=1))
        output.append(dict(output_begin=ob, output_count=no, chunks=chunks))
    return output


def build_plan(candidate):
    """Return compressed declarative arrays for all 64 (SC,PU) instances."""
    validate_fields(candidate["fields_lsb_first"])
    split = candidate["split_factor"]
    lk, lo = 512 // split, 56 * split
    k_parts = [dict(id=p, k8_indices=[int(deposit(j, candidate["k_local_bits"]) | deposit(p, candidate["k_selected_bits"]))
                                    for j in range(lk)]) for p in range(split)]
    shards = [dict(id=p, n4_groups=[int(deposit(o, candidate["n_local_bits"]) | deposit(p, candidate["n_selected_bits"]))
                                  for o in range(lo)]) for p in range(64 // split)]
    units = []
    for sc in range(8):
        for pu in range(8):
            values = dict(ch=sc, bg=pu // 2, ba=2 * (pu % 2))
            selector = sum(((values[field] >> bit) & 1) << i for i, (field, bit) in enumerate(SLOTS))
            kp = int(extract(selector, candidate["k_selector_slots"]))
            npart = int(extract(selector, [i for i in range(6) if i not in candidate["k_selector_slots"]]))
            units.append(dict(sc=sc, pu=pu, k_partition=kp, output_shard=npart))
    plan = dict(candidate, schema="lp6.pa_ha_up.plan.v2", N=N, K=K, weight_row=WEIGHT_ROW,
                local_k8_count=lk, local_output_groups=lo, k_partitions=k_parts,
                output_shards=shards, units=units, physical_units=64,
                activation_tile_contract="token * local_k8_count + local_K8_ordinal; pack four local K8 inputs into each IREG packet",
                all_K_local=split == 1, NPU_global_reduction_required=split > 1,
                partials_per_output=split, partial_I32_bytes_per_token=N * 4 * split,
                activation_replica_bytes_per_token=64 * K // split,
                NPU_I32_reduction_adds_per_token=N * (split - 1),
                PIM_internal_weight_packet_reads_per_token=N * K // 32,
                PIM_weight_MAC_commands_per_token=N * K // (32 * 8),
                storage_copies=1, weight_assembly_required_for_direct_NPU_g4=False)
    maximum = int(weight_tiles(plan, lo - 1, lk - 1))
    plan.update(weight_tile_max=maximum, weight_rows=(maximum + 64) // 64,
                weight_row_end_exclusive=WEIGHT_ROW + (maximum + 64) // 64,
                weight_HA_hole_packets=(maximum + 1) - lo * lk)
    plan["stream_batches_8"] = stream_batches(plan, 8)
    return plan


def validate_plan(plan, exhaustive_packets=True):
    """Coordinates, PA coverage and exact integer reduction; no timing claim."""
    kp, ns, units = plan["k_partitions"], plan["output_shards"], plan["units"]
    assert sorted(x for p in kp for x in p["k8_indices"]) == list(range(512))
    assert sorted(x for p in ns for x in p["n4_groups"]) == list(range(N // 4))
    assert len({(u["k_partition"], u["output_shard"]) for u in units}) == 64
    assert len({(u["sc"], u["pu"]) for u in units}) == 64
    assert plan["weight_row_end_exclusive"] <= 1024
    lo, lk = plan["local_output_groups"], plan["local_k8_count"]
    oo = np.arange(lo, dtype=np.int64)[:, None]
    jj = np.arange(lk, dtype=np.int64)[None, :]
    tiles = weight_tiles(plan, oo, jj)
    assert tiles.shape == (lo, lk) and np.unique(tiles).size == lo * lk
    if exhaustive_packets:
        coverage = np.zeros(N * K // 32, dtype=np.uint8)
        for u in units:
            ng = np.asarray(ns[u["output_shard"]]["n4_groups"], dtype=np.int64)[:, None]
            ki = np.asarray(kp[u["k_partition"]]["k8_indices"], dtype=np.int64)[None, :]
            pa = packet_address(plan, sc=u["sc"], pu=u["pu"], tile=tiles)
            assert np.array_equal(pa, raw_packet_pa(ng, ki))
            assert np.all((pa >= W_BASE) & (pa + 32 <= W_BASE + N * K))
            coverage[((pa - W_BASE) // 32).ravel()] += 1
        assert np.all(coverage == 1)
    rng = np.random.default_rng(261006)
    x = rng.integers(-128, 128, K, dtype=np.int32)
    w = rng.integers(-128, 128, (8, K), dtype=np.int32)
    partial = []
    for partition in kp:
        indices = (np.asarray(partition["k8_indices"])[:, None] * 8 + np.arange(8)).ravel()
        partial.append(w[:, indices] @ x[indices])
    reduced = np.stack(partial).sum(axis=0, dtype=np.int32)
    assert np.array_equal(reduced, w @ x)
    assert K * 128 * 128 < 1 << 31
    return dict(id=plan["id"], split_factor=plan["split_factor"], order=plan["order"],
                every_32B_weight_packet_once=bool(exhaustive_packets), packets=N * K // 32,
                offsets_preserved=True, rank0_even_banks=True, W_base_physical_row=512,
                weight_rows=plan["weight_rows"], weight_HA_hole_packets=plan["weight_HA_hole_packets"],
                partitions_exact_cover_K=True, output_shards_exact_cover_N=True,
                integer_partial_sum_reconstruction_exact=True,
                int32_full_signed8_worst_case_bound=K * 128 * 128,
                native_runs=0, timing_estimate=None)


def retained_candidates():
    """Bounded, premeasurement structural shortlist, not claimed optimal."""
    candidates, seen = [], set()
    def add(c, reason):
        sig = json.dumps(c["fields_lsb_first"], sort_keys=True)
        if sig in seen:
            return
        seen.add(sig)
        c["selection_reason"] = reason
        candidates.append(c)
    for s in (1, 2, 4, 8, 16, 32):
        for order in (("c6",) if s == 32 else ("c6", "c5b8")):
            add(make_candidate(s, order), "primary: Klow, SC-first, Nlow, local-order contrast")
    for split in (4, 8, 16):
        bits = split.bit_length() - 1
        for order in ("c6", "c5b8"):
            add(make_candidate(split, order, k_slots=sorted((3, 4, 5, 0, 1, 2)[:bits])),
                "resource contrast: Klow, BG/bank-first, Nlow")
            add(make_candidate(split, order, k_window=9 - bits),
                "input-contiguity contrast: Khigh, SC-first, Nlow")
    return candidates


def write_config(source_ini, destination, candidate):
    import re
    text = Path(source_ini).read_text()
    assert not re.search(r"(?m)^address_bits_", text)
    additions = "\n".join("address_bits_" + field + " = " + ",".join(map(str, bits))
                          for field, bits in candidate["fields_lsb_first"].items())
    text, count = re.subn(r"(?m)^(address_mapping\s*=.*)$", r"\1\n" + additions, text)
    assert count == 1
    Path(destination).write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grid = generate_candidates()
    retained = retained_candidates()
    assert len(retained) <= 24
    proofs = []
    for c in retained:
        plan = build_plan(c)
        proofs.append(validate_plan(plan))
        (args.output / (c["id"] + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "candidates.json").write_text(json.dumps(retained, indent=2) + "\n")
    summary = dict(schema="lp6.pa_ha_up.static_search.v2", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   storage=STORAGE, raw_weight_bytes=N * K, structural_grid_distinct_descriptors=len(grid),
                   grid_window_policy="low/high endpoints", retained=len(retained), proofs=proofs,
                   equivalent_c5b8_c6_at_S16_deduplicated=True, native_runs=0,
                   limitation="Finite premeasurement candidates; no global optimum, latency or bandwidth claim.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("retained", "structural_grid_distinct_descriptors", "native_runs", "source_sha256")}))


if __name__ == "__main__":
    main()
