#!/usr/bin/env python3
"""Bounded four-core PIM feed against an archived, exact one-core FP16 program.

--baseline is a completed pim_packet_npu.py result directory. Its source X/W,
manifest, scalar/INT8/INT32/FP16 references and default program remain immutable.
Only the new four-core variant executes; default-byte regression is optional.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

from pim_packet_npu import (ROOT, LP6, Device, EMU, SETTINGS, core_instructions,
                            decode_program, json_attributes, phase_times, save, sha)


def ancestors(commands, event):
    found = set()
    stack = list(commands[event]["dependencies"])
    while stack:
        at = stack.pop()
        if at in found:
            continue
        found.add(at)
        stack.extend(commands[at]["dependencies"])
    return found


def phase_program(build, phase):
    system = json.loads((build / "system.json").read_text())
    symbols = {s["id"]: s["name"] for s in system["command_symbols"]}
    program = decode_program((build / "program.bin").read_bytes())
    rows = [r for r in program["commands"] if symbols[r["id"]].split(".")[0] == phase]
    return program, rows


def check_plan(build, baseline_build=None):
    io = json.loads((build / "io.json").read_text())
    selected = [p for p in io["phases"] if p.get("pim_activation_cores") == 4]
    assert len(selected) == 1, selected
    phase = selected[0]
    plan = phase["pim_activation_plan"]
    assert plan["kernel"] == "pim_activation_scatter_4core_v1"
    assert plan["cores"] == phase["cores"] == 4 and plan["quant_cores"] == 1
    assert plan["isolated_quant_feed_phase_checked"] and plan["phase_exit_joins_all_sinks"]
    assert json_attributes((build / "compiled.mlir").read_text(), "hyb.pim_activation_plan") == [plan]
    m, k = plan["M"], plan["K"]
    ranks = [plan["rank"]] if "rank" in plan else plan["ranks"]
    assert ranks in ([0], [1], [0, 1]), ranks
    assert plan["q_bytes"] == plan["q_publish_bytes"] == m*k
    assert plan["q_worker_load_bytes"] == (m-(m+3)//4)*k
    assert plan["q_shared_l2"] + m*k <= plan["l2_window_bytes_per_core"]
    assert m*4 <= plan["q_shared_l2"]
    ranges = [(r["offset"], r["bytes"]) for r in plan["replica_l2_ranges"]]
    ranges.append((plan["q_shared_l2"], plan["q_bytes"]))
    for i, (start, size) in enumerate(ranges):
        assert start >= 0 and size > 0 and start + size <= plan["l2_capacity_bytes"]
        for begin, count in ranges[i+1:]:
            assert start + size <= begin or begin + count <= start, (ranges, i)
    assignments = [t for row in plan["token_assignment"] for t in row["tokens"]]
    assert sorted(assignments) == list(range(m))
    program, records = phase_program(build, phase["phase"])
    commands = program["commands"]
    stores = [commands[i] for i in plan["activation_store_events"]]
    assert stores and all(c["kind"] == "gdma_store" for c in stores)
    assert sum(c["row_bytes"]*c["rows"] for c in stores) == m*k*64*len(ranks) == plan["lp6_activation_write_bytes"]
    rank_bytes = Counter()
    for c in stores:
        begin = c["lp6_address"]
        end = begin + (c["rows"]-1)*c["lp6_stride"] + c["row_bytes"]
        rank = begin // (1 << 33)
        assert rank == (end-1) // (1 << 33) and rank in ranks
        rank_bytes[rank] += c["row_bytes"]*c["rows"]
    assert dict(rank_bytes) == {r: m*k*64 for r in ranks}, rank_bytes
    # Every rank reads the same paid replica; a later core overwrite must wait
    # for all earlier GDMA readers, including the other rank's readers.
    reuse_checks = 0
    for c in stores:
        producers = [i for i in c["dependencies"] if commands[i]["kind"] == "core_block"]
        assert len(producers) == 1, (c["id"], producers)
        writer = producers[0]
        start = c["l2_address"]
        end = start + (c["rows"]-1)*c["l2_stride"] + c["row_bytes"]
        upstream = ancestors(commands, writer)
        for previous in stores:
            if previous["id"] >= writer:
                continue
            pstart = previous["l2_address"]
            pend = pstart + (previous["rows"]-1)*previous["l2_stride"] + previous["row_bytes"]
            if start < pend and pstart < end:
                assert previous["id"] in upstream, (writer, previous["id"], "overwritten live GDMA source")
                reuse_checks += 1
    publish = plan["q_ready_event"]
    assert commands[publish]["kind"] == "core_block" and commands[publish]["target_logical_core"] == 0
    assert all(publish in ancestors(commands, c["id"]) for c in stores)
    pims = [c for c in commands if c["kind"] == "pim"]
    assert len(pims) == 1
    assert set(plan["activation_store_events"]) <= ancestors(commands, pims[0]["id"])
    math_ops = {"V_LD", "V_OP", "V_ST", "V_CVT", "S_OP", "S_LD", "S_ST"}
    counts = Counter()
    active = set()
    for record in records:
        if record["kind"] != "core_block":
            continue
        core = record["target_logical_core"]; active.add(core)
        start = record["isa_start_word"]
        names = Counter(name for name, _ in core_instructions(program["words"][start:start+record["isa_word_count"]]))
        if core != 0:
            assert not math_ops.intersection(names), (core, names)
        counts.update({name: count for name, count in names.items() if name in math_ops})
    assert active == set(range(min(4, m))), active
    if baseline_build:
        old_io = json.loads((baseline_build / "io.json").read_text())
        old_phase = old_io["phases"][0]["phase"]
        old, old_records = phase_program(baseline_build, old_phase)
        old_math = Counter()
        for record in old_records:
            if record["kind"] == "core_block":
                start = record["isa_start_word"]
                old_math.update(name for name, _ in core_instructions(old["words"][start:start+record["isa_word_count"]]) if name in math_ops)
        assert old_math == counts, (old_math, counts)
        old_pim = [c["pim_commands"] for c in old["commands"] if c["kind"] == "pim"]
        assert old_pim == [c["pim_commands"] for c in pims], "PIM primitive payload changed"
        def write_rows(rs):
            return sorted((c["lp6_address"]+r*c["lp6_stride"], c["row_bytes"])
                          for c in rs if c["kind"] == "gdma_store" for r in range(c["rows"]))
        assert write_rows(old_records) == write_rows(records), "activation/scale store PA or logical bytes changed"
    return dict(plan=plan, actual_quant_math_counts=dict(counts), quant_only_on_core0=True,
                actual_active_cores=sorted(active), all_activation_stores_depend_on_Q_publication=True,
                activation_write_bytes_by_rank=dict(rank_bytes), replica_reuse_dependency_checks=reuse_checks,
                actual_PIM_waits_for_all_activation_stores=True, default_PIM_primitives_and_store_PA_unchanged=bool(baseline_build))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiler", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--emulator", type=Path, default=EMU)
    ap.add_argument("--settings", type=Path, default=SETTINGS)
    ap.add_argument("--dramsim-library", type=Path, required=True)
    ap.add_argument("--skip-negatives", action="store_true")
    ap.add_argument("--check-default", action="store_true")
    ap.add_argument("--keep-dumps", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline.resolve()
    previous = json.loads((baseline / "summary.json").read_text())
    assert previous["all_requested_checks_passed"] and previous["provenance_unchanged"]
    archived = next(v for v in previous["native"] if v["variant"] == "pim")
    old = Path(archived["build"])
    reference = np.load(baseline / "reference.npz")
    m, n, k = previous["M"], previous["N"], previous["K"]
    compile_entry = next(c for c in previous["compile_cases"] if c["name"] == "pim_import")
    argv = compile_entry["argv"]
    source = Path(argv[argv.index("--pipeline=decode")+1])
    frontend = Path(argv[argv.index("--manifest")+1])
    target = baseline / "target.json"
    manifest = baseline / "weight_storage.json"
    library = args.dramsim_library.resolve()
    assert previous["provenance_before"][str(library)] == sha(library), "baseline used another DRAM simulator"
    logs = out / "logs"; logs.mkdir()
    tracked = [args.compiler.resolve(), args.emulator.resolve(), library, args.settings.resolve(), Path(__file__).resolve(),
               ROOT / "lib/Decode/Target/NPU/NpuCodegen.cpp", ROOT / "include/npu/Decode/Target/NpuCodegen.h",
               ROOT / "lib/Decode/Driver.cpp", manifest, source, frontend, target,
               old / "program.bin", old / "lp6.bin", baseline / "reference.npz"]
    before = {str(p): sha(p) for p in tracked}
    snapshots = out / "source_snapshot"; snapshots.mkdir()
    for p in tracked:
        if p.suffix in (".cpp", ".h", ".py", ".json", ".mlir", ".toml"):
            relative = p.relative_to(LP6) if p.is_relative_to(LP6) else Path("external") / p.name
            dest = snapshots / relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    result = dict(scope="Same FP16 LP6 input through unchanged row quant, same PIM primitive/PA and dequant; only bounded activation feed is parallelized",
                  M=m, N=n, K=k, baseline=str(baseline), provenance_before=before, negatives=[], compiles=[], native=[])

    def compile_case(name, extra=(), expect=None, source_path=source, target_path=target, use_import=True, force="pim"):
        build = out / name
        cmd = [str(args.compiler.resolve()), "--pipeline=decode", str(source_path), "--manifest", str(frontend),
               "--spec", str(target_path), f"--force={force}", "--debug-collect", "-o", str(build)]
        if use_import:
            cmd += ["--pim-weight-storage-import", str(manifest)]
        cmd += list(map(str, extra))
        if expect:
            build.mkdir(); save(build / "pim_weight_storage_verified.json", {"stale": True})
        run = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        text = run.stdout + run.stderr; (logs / f"{name}.log").write_text(text)
        item = dict(name=name, argv=cmd, exit_code=run.returncode)
        result["compiles"].append(item)
        save(out / "summary.json", result)
        if expect:
            assert run.returncode != 0 and expect in text, (name, run.returncode, text[-3500:])
            assert not (build / "pim_weight_storage_verified.json").exists()
            result["negatives"].append(dict(**item, diagnostic=expect, stale_sidecar_removed=True))
        else:
            assert run.returncode == 0, (name, run.returncode, text[-5000:])
        return build

    try:
        build = compile_case("feed4", ["--pim-activation-cores=4"])
        result["program_audit"] = check_plan(build, old)
        assert json.loads((build / "pim_weight_storage_verified.json").read_text()) == json.loads(manifest.read_text())
        os.environ["LD_PRELOAD"] = str(library)
        os.environ.pop("LP6PIM_BANKSET_PINGPONG", None)
        dev = Device(build, args.settings, LP6 / "lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini", args.emulator)
        dev.write("x", reference["x"])
        assert sha(build / "lp6.bin") == sha(old / "lp6.bin"), "complete LP6 image bytes changed"
        result["whole_LP6_image_after_X_identical"] = True
        if args.check_default:
            current = compile_case("default1")
            d = Device(current, args.settings, emulator=args.emulator); d.write("x", reference["x"])
            assert sha(current / "program.bin") == sha(old / "program.bin")
            assert sha(current / "lp6.bin") == sha(old / "lp6.bin")
            result["default_program_and_image_identical"] = True
        noimage = compile_case("no_manifest_no_image", ["--pim-activation-cores=4", "--no-image"], use_import=False)
        result["no_manifest_no_image_plan"] = check_plan(noimage)
        assert not (noimage / "lp6.bin").exists()

        if not args.skip_negatives:
            cli = "PIM activation cores must be 1 or 4"
            compile_case("negative_cores_value", ["--pim-activation-cores=2"], expect=cli)
            compile_case("negative_force", ["--pim-activation-cores=4"], force="npu", expect=cli)
            compile_case("negative_packet", ["--pim-activation-cores=4", "--pim-packet-npu=g4"], expect=cli)
            compile_case("negative_map", ["--pim-activation-cores=4", "--map=proj:pim"], expect=cli)
            for name, changes, diagnostic in [
                ("target_cores", {"cores": 2}, "requires a 4-core target"),
                ("shared_q_window", {"l2_staging_bytes_per_core": ((m*k+63)//64)*64+1024},
                 "L2 window cannot hold shared Q and one replicated column"),
                ("L1_reserve", {"l1_reserved_bytes": 32768}, "quant and feed scratch exceed the L1 reserve")]:
                spec = json.loads(target.read_text()); spec["npu"].update(changes)
                path = out / f"{name}_target.json"; save(path, spec)
                compile_case(f"negative_{name}", ["--pim-activation-cores=4"], target_path=path, expect=diagnostic)
            spec = json.loads(target.read_text()); spec["pim"].update(channels=2, partitions=2)
            path = out / "SC2_target.json"; save(path, spec)
            compile_case("negative_SC2", ["--pim-activation-cores=4"], target_path=path, use_import=False,
                         expect="requires a 4-core target and all 8 subchannels")
            sliced = out / "input_subview.mlir"
            text = source.read_text().replace(f"%x: tensor<{m}x{k}xf32>", f"%x: tensor<{m+1}x{k}xf32>", 1)
            text = text.replace("    %zero =", f"    %xview = tensor.extract_slice %x[1, 0] [{m}, {k}] [1, 1] : tensor<{m+1}x{k}xf32> to tensor<{m}x{k}xf32>\n    %zero =", 1)
            text = text.replace("linalg.matmul ins(%x,", "linalg.matmul ins(%xview,", 1)
            sliced.write_text(text)
            compile_case("negative_subview", ["--pim-activation-cores=4"], source_path=sliced,
                         expect="isolated quant/feed phase does not support subviews")
            mixed = out / "mixed_compute.mlir"
            text = source.read_text().replace("    %zero =", f'''    %xe = tensor.empty() : tensor<{m}x{k}xf32>
    %neg = linalg.generic {{indexing_maps = [affine_map<(i,j) -> (i,j)>, affine_map<(i,j) -> (i,j)>], iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<{m}x{k}xf32>) outs(%xe : tensor<{m}x{k}xf32>) {{
      ^bb0(%a: f32, %b: f32):
        %v = arith.negf %a : f32
        linalg.yield %v : f32
      }} -> tensor<{m}x{k}xf32>
    %zero =''', 1)
            mixed.write_text(text.replace("linalg.matmul ins(%x,", "linalg.matmul ins(%neg,", 1))
            compile_case("negative_mixed_compute", ["--pim-activation-cores=4"], source_path=mixed,
                         expect="isolated quant/feed phase does not support subviews, loops or other compute")

        outputs, report, wall = dev.run()
        def get(dtype, shape):
            entries = [outputs[name] for name, meta in dev.outputs.items() if meta["dtype"] == dtype and meta["shape"] == shape]
            assert len(entries) == 1
            return entries[0]
        q = get("i8", [m,k]); scales = get("f32", [m,1]); acc = get("i32", [m,n])
        assert q.tobytes() == reference["xq"].tobytes()
        assert scales.tobytes() == np.full((m,1),1/64,np.float32).tobytes()
        assert acc.tobytes() == reference["expected_i32"].tobytes()
        assert outputs["y"].tobytes() == reference["expected_f16"].tobytes()
        np.savez(build / "actual.npz", quantized=q, row_scales=scales, integer=acc, y=outputs["y"])
        times = phase_times(report); endpoint = sum(t for name, _, t in times if name != "collect")
        result["native"].append(dict(wall_seconds=wall, endpoint_us=endpoint, baseline_endpoint_us=archived["endpoint_us"],
            baseline_over_feed4=archived["endpoint_us"]/endpoint, phase_times=times,
            all_INT8_scale_INT32_FP16_byte_exact=True, int32_sha256=hashlib.sha256(acc.tobytes()).hexdigest(),
            fp16_sha256=hashlib.sha256(outputs["y"].tobytes()).hexdigest(), program_sha256=sha(build / "program.bin")))
        if not args.keep_dumps:
            for dump in build.glob("sram_dump*.bin"):
                dump.unlink()
        result["all_requested_checks_passed"] = True
    except BaseException:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        result["provenance_after"] = {str(p): sha(p) for p in tracked}
        result["provenance_unchanged"] = result["provenance_after"] == before
        save(out / "summary.json", result)
    assert result["provenance_unchanged"], "a source, binary or archived baseline changed"
    print(json.dumps(dict(native=result["native"], negatives=len(result["negatives"]), all_checks_passed=True)), flush=True)


if __name__ == "__main__":
    main()
