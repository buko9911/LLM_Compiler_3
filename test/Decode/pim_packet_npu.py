#!/usr/bin/env python3
"""Same PIM weight storage -> PIM and opt-in packet-NPU, including quant/dequant.

Creates a fresh --output directory, exports with M32, and imports with M7 by
default. No model download. The two native programs run sequentially. All
physical weight packets/scales, intermediate INT8/INT32, row scales and FP16
results are checked. Debug collect is excluded from reported endpoint time.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
LP6 = ROOT.parent
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(LP6 / "IESLAB_NPU_Simulator/transactional_emulator/tools"))
from decode import Device, EMU, SETTINGS, phase_times  # noqa: E402
from isa_v2 import core_instructions, decode_program  # noqa: E402
from pim_weight_storage import audit_images, save, seal, sha  # noqa: E402


def fixture(directory: Path, m: int, n: int, k: int, model: Path):
    """Only the frontend graph changes with M; W has the same source tensor."""
    directory.mkdir(parents=True)
    source = directory / "step.mlir"
    source.write_text(f'''module {{
  func.func @decode_step(%x: tensor<{m}x{k}xf32>, %w: tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32> {{
    %zero = arith.constant 0.0 : f32
    %wt = tensor.empty() : tensor<{k}x{n}xf32>
    %t = linalg.transpose ins(%w : tensor<{n}x{k}xf32>) outs(%wt : tensor<{k}x{n}xf32>) permutation = [1, 0]
    %empty = tensor.empty() : tensor<{m}x{n}xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    %y = linalg.matmul ins(%x, %t : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%init : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
    return %y : tensor<{m}x{n}xf32>
  }}
}}
''')
    frontend = directory / "frontend.json"
    save(frontend, dict(model=str(model), tokens=m, capacity=1, mask_min=-30000.0,
                       arguments=[dict(kind="input", name="x"), dict(kind="weight", name="proj.weight")],
                       results=[dict(kind="output", name="y")]))
    return source, frontend


def json_attributes(text: str, name: str):
    """Decode MLIR string attributes, whose escaped bytes use \\22 rather than \\"."""
    pattern = re.escape(name) + r'\s*=\s*"((?:\\[0-9A-Fa-f]{2}|\\.|[^"\\])*)"'
    values = []
    for match in re.finditer(pattern, text):
        raw = re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), match[1])
        values.append(json.loads(raw))
    return values


def check_ranges(plan, space: str, capacity: int):
    ranges = plan[f"{space}_ranges"]
    peak = plan[f"{space}_peak_bytes"]
    assert isinstance(peak, int) and 0 < peak <= capacity, (space, peak, capacity)
    assert ranges, (space, "missing memory ranges")
    names = set()
    for item in ranges:
        assert item["name"] and item["name"] not in names, (space, item)
        names.add(item["name"])
        begin, size = item["offset"], item["bytes"]
        assert isinstance(begin, int) and isinstance(size, int) and begin >= 0 and size > 0, item
        assert begin + size <= peak <= capacity, (space, item, peak, capacity)
    for i, first in enumerate(ranges):
        for second in ranges[i+1:]:
            assert (first["offset"] + first["bytes"] <= second["offset"] or
                    second["offset"] + second["bytes"] <= first["offset"]), (space, first, second)
    assert max(item["offset"] + item["bytes"] for item in ranges) == peak
    return dict(ranges=ranges, reported_peak_bytes=peak, physical_capacity_bytes=capacity,
                all_ranges_within_reported_peak_and_capacity=True, all_declared_ranges_disjoint=True)


def audit_packet_metadata(build: Path, storage, m: int, n: int, k: int, spec):
    io = json.loads((build / "io.json").read_text())
    system = json.loads((build / "system.json").read_text())
    ir = (build / "compiled.mlir").read_text()
    decoded = decode_program((build / "program.bin").read_bytes())
    assert io["pim_commands"] == 0
    assert not any(c["kind"] == "pim" for c in decoded["commands"])
    assert len(system["logical_to_physical"]) == 4
    assert all(p["kind"] == "npu" for p in io["phases"])
    packet = [p for p in io["phases"] if p.get("kernel") == "pim_packet_g4"]
    assert len(packet) == 1, io["phases"]
    index = io["phases"].index(packet[0])
    assert index > 0 and index < len(io["phases"]) - 1, "quant/dequant phases must surround packet phase"
    assert packet[0]["cores"] == 4
    plan = packet[0]["packet_plan"]
    expected = dict(kernel="pim_packet_g4", group_lanes=4, M=m, N=n, K=k, Kseg=256,
                    Kblock=min(k, 2048), cores=4, raw_slots_per_core=2,
                    raw_slot_bytes=128 * min(k, 2048), input_dtype="i8", output_dtype="i32",
                    output_lp6_store_included=True, partial_dtype="i32",
                    packet_location_api="Program::forEachWeightPacketLocation",
                    all_packet_contiguity_checked=True)
    for key, value in expected.items():
        assert plan.get(key) == value, (key, plan.get(key), value)
    for key, size in [("input_lp6", m*k), ("output_lp6", 4*m*n)]:
        address = plan[key]
        assert isinstance(address, int) and address >= 0x10000
        assert address + size <= storage["weights"][0]["scale_address"], (key, address, size)
    a, b = plan["input_lp6"], plan["output_lp6"]
    assert a + m*k <= b or b + 4*m*n <= a, "packet input/output overlap"
    ranges = {s: check_ranges(plan, s, spec["npu"][f"{s}_bytes"]) for s in ["l1", "l2"]}
    pim_plans = json_attributes(ir, "hyb.pim_plan")
    assert len(pim_plans) == 1
    p = pim_plans[0]
    assert p["acts"] == p["phases"] == [] and p["next_a"] == p["next_o"] == 0
    assert len(p["projections"]) == 1
    q, w = p["projections"][0], storage["weights"][0]
    assert q["acts"] == [] and q["compute_count"] == q["compute_first"] == 0
    for qkey, wkey in [("weight", "name"), ("n", "n"), ("k", "k"), ("n_pad", "n_pad"),
                       ("stream_part", "stream_part"), ("stream_weight_base", "stream_weight_base")]:
        assert q[qkey] == w[wkey], (qkey, q[qkey], w[wkey])
    assert p["kernel_row"] == storage["kernel_row"]
    assert p["channels"] == 8 and p["bankset_pingpong_lead"] == 0
    assert 'hyb.npu_kernel = "pim_packet_g4"' in ir and "hyb.packet_projection" in ir
    assert "npu_nk32" not in ir, "unexpected dedicated NPU weight copy"
    assert all(o["layout"] not in ("pim_act", "pim_out") for o in io["outputs"])
    ir_plans = json_attributes(ir, "hyb.packet_plan")
    assert ir_plans == [plan], "compiled.mlir must contain the final checked packet plan"
    symbols = {s["id"]: s["name"] for s in system["command_symbols"]}
    records = [c for c in decoded["commands"] if symbols[c["id"]].split(".")[0] == packet[0]["phase"]]
    counts = Counter()
    reload_dtypes = []
    writes = []
    for record in records:
        if record["kind"] == "core_block":
            start = record["isa_start_word"]
            for name, words in core_instructions(decoded["words"][start:start+record["isa_word_count"]]):
                counts[name] += 1
                if name == "M_LD_ACC":
                    reload_dtypes.append("i32" if words[0] >> 6 & 1 else "f32")
        if record["kind"] == "gdma_store":
            writes += [(record["lp6_address"] + row*record["lp6_stride"], record["row_bytes"])
                       for row in range(record["rows"])]
    assert counts["M_MMA"] > 0 and counts["M_WB"] > 0
    assert (counts["M_LD_ACC"] > 0) == (k > 2048), (k, counts)
    assert all(dtype == "i32" for dtype in reload_dtypes), reload_dtypes
    cursor = plan["output_lp6"]
    for begin, size in sorted(writes):
        assert begin == cursor, ("output GDMA gap/duplicate", begin, cursor)
        cursor += size
    assert cursor == plan["output_lp6"] + 4*m*n, "canonical INT32 LP6 output must be fully stored"
    return dict(packet_phase=packet[0], storage_only_pim_plan=p, memory=ranges,
                actual_program_pim_records=0, dedicated_npu_weight_copy=False,
                packet_plan_in_compiled_mlir=True, system_command_count=decoded["command_count"],
                packet_core_instruction_counts=dict(counts), accumulator_reload_dtypes=reload_dtypes,
                actual_output_gdma_bytes=4*m*n, actual_output_gdma_covers_every_byte_exactly_once=True)


def provenance_files(args):
    paths = [args.compiler, args.emulator, args.dramsim_library, args.settings, args.lp6_config,
             Path(__file__), ROOT / "test/Decode/pim_weight_storage.py", ROOT / "runtime/decode.py",
             ROOT / "configs/target.json", LP6 / "IESLAB_NPU_Simulator/transactional_emulator/tools/isa_v2.py"]
    paths += sorted((ROOT / "lib/Decode").rglob("*.cpp"))
    paths += sorted((ROOT / "include/npu/Decode").rglob("*.h"))
    paths += [ROOT / rel for rel in ["include/npu/Target/PimLayout.h", "lib/Target/PimLayout.cpp",
                                   "include/npu/Target/Program.h", "lib/Target/Program.cpp",
                                   "lib/Target/PimPrimitive.cpp", "lib/Target/IsaV2.cpp"]]
    if args.old_compiler:
        paths.append(args.old_compiler)
    return sorted(set(p.resolve() for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/bin/npu-compile")
    parser.add_argument("--old-compiler", type=Path, help="optional immutable pre-change compiler for default PIM bytes")
    parser.add_argument("--emulator", type=Path, default=EMU)
    parser.add_argument("--dramsim-library", type=Path, required=True)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--lp6-config", type=Path, default=LP6 / "lp6-pim-main/configs/LPDDR6_PIM_Mobile_4CH.ini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--export-m", type=int, default=32)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--skip-negatives", action="store_true")
    parser.add_argument("--keep-dumps", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.m <= 8 and args.n > 0 and args.n % 4096 == 0 and args.k > 0 and args.k % 256 == 0
    assert args.export_m > args.m
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"; logs.mkdir()
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    paths = provenance_files(args)
    before = {str(p): sha(p) for p in paths}
    for path in paths:
        if path.suffix in (".cpp", ".h", ".py", ".json", ".toml", ".ini"):
            relative = path.relative_to(LP6) if path.is_relative_to(LP6) else Path("external") / path.name
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    result = dict(scope="FP16 LP6 input -> real row quant -> PIM or packet-NPU -> real dequant -> FP16 LP6 output; collect excluded",
                  dynamic_runtime_switching=False, compile_only=args.compile_only,
                  M=args.m, N=args.n, K=args.k, export_M=args.export_m,
                  argv=sys.argv, provenance_before=before, compile_cases=[], negatives=[], native=[])
    save(out / "summary.json", result)
    spec = json.loads((ROOT / "configs/target.json").read_text())
    spec["pim"]["partitions"] = 8
    spec_path = out / "target.json"; save(spec_path, spec)
    rng = np.random.default_rng(195121)
    wq = rng.integers(-127, 128, (args.n, args.k), dtype=np.int8); wq[:, 0] = 127
    source_values = wq.astype(np.float16) / np.float16(64)
    model = out / "model"; model.mkdir()
    save_file({"proj.weight": source_values}, model / "model.safetensors")
    sources = {}

    def source_for(m, n=None, k=None):
        key = (m, n or args.n, k or args.k)
        if key not in sources:
            sources[key] = fixture(out / f"input_m{key[0]}_n{key[1]}_k{key[2]}", *key, model)
        return sources[key]

    def compile_case(name, m, options=(), force="npu", expect=None, target=None, shape=None, compiler=None,
                     source_override=None, frontend_override=None):
        build = out / name
        source, frontend = source_for(m, *(shape or (args.n, args.k)))
        if source_override:
            source = source_override
        if frontend_override:
            frontend = frontend_override
        argv = [str((compiler or args.compiler).resolve()), "--pipeline=decode", str(source),
                "--manifest", str(frontend), "--spec", str(target or spec_path), "--debug-collect", "-o", str(build)]
        if force is not None:
            argv.append(f"--force={force}")
        argv += list(map(str, options))
        if expect:
            build.mkdir(parents=True, exist_ok=True)
            save(build / "pim_weight_storage_verified.json", {"stale_test_marker": True})
        run = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
        log = run.stdout + run.stderr
        (logs / f"{name}.log").write_text(log)
        entry = dict(name=name, argv=argv, exit_code=run.returncode)
        result["compile_cases"].append(entry)
        save(out / "summary.json", result)
        if expect:
            needles = [expect] if isinstance(expect, str) else expect
            assert run.returncode != 0 and all(s in log for s in needles), (name, run.returncode, log[-3000:])
            assert not (build / "pim_weight_storage_verified.json").exists(), (name, "stale verified sidecar")
            result["negatives"].append(dict(**entry, diagnostic=needles, stale_verified_sidecar_removed=True))
        else:
            assert run.returncode == 0, (name, run.returncode, log[-5000:])
        return build

    try:
        exported = out / "weight_storage.json"
        export_build = compile_case("pim_export", args.export_m,
                                    ["--pim-weight-storage-export", exported], force="pim")
        storage = json.loads(exported.read_text())
        sealed = copy.deepcopy(storage); seal(sealed)
        assert sealed == storage, "independent manifest digests"
        pim = compile_case("pim_import", args.m, ["--pim-weight-storage-import", exported], force="pim")
        packet_options = ["--pim-weight-storage-import", exported, "--pim-packet-npu=g4"]
        npu = compile_case("packet_npu_import", args.m, packet_options)
        for build in [pim, npu]:
            assert json.loads((build / "pim_weight_storage_verified.json").read_text()) == storage
        result["storage_manifest"] = storage
        result["image_audits_before_input"] = {
            "export_vs_pim": audit_images(storage, [export_build / "lp6.bin", pim / "lp6.bin"], wq, source_values),
            "export_vs_npu": audit_images(storage, [export_build / "lp6.bin", npu / "lp6.bin"], wq, source_values)}
        result["packet_metadata"] = audit_packet_metadata(npu, storage, args.m, args.n, args.k, spec)
        assert json.loads((pim / "io.json").read_text())["pim_commands"] > 0
        if args.old_compiler:
            old = compile_case("pim_old", args.m, ["--pim-weight-storage-import", exported],
                               force="pim", compiler=args.old_compiler)
            result["default_pim_before_after"] = {f: dict(before=sha(old / f), after=sha(pim / f))
                                                   for f in ["program.bin", "lp6.bin"]}
            assert all(e["before"] == e["after"] for e in result["default_pim_before_after"].values())

        if not args.skip_negatives:
            prefix = "PIM packet NPU requires g4, --force=npu, --pim-weight-storage-import"
            compile_case("negative_missing_import", args.m, ["--pim-packet-npu=g4"], expect=prefix)
            compile_case("negative_force_pim", args.m, packet_options, force="pim", expect=prefix)
            compile_case("negative_force_unspecified", args.m, packet_options, force=None, expect=prefix)
            compile_case("negative_g2", args.m, ["--pim-weight-storage-import", exported, "--pim-packet-npu=g2"], expect=prefix)
            compile_case("negative_map", args.m, [*packet_options, "--map=proj:npu"], expect=prefix)
            compile_case("negative_no_image", args.m, [*packet_options, "--no-image"], expect="incompatible with --no-image")
            compile_case("negative_serial_prefetch", args.m, [*packet_options, "--npu-weight-prefetch=false"],
                         expect="requires --npu-weight-prefetch=true")
            for name, m, shape in [("M9", 9, None), ("N_tail", args.m, (3584, args.k)),
                                    ("K_tail", args.m, (args.n, args.k+1))]:
                compile_case(f"negative_{name}", m, packet_options, shape=shape,
                             expect="requires M in 1..8, N % 4096 == 0 and K % 256 == 0")
            for name, modify in [("cores2", lambda s: s["npu"].update(cores=2)),
                                  ("SC2", lambda s: s["pim"].update(channels=2, partitions=2))]:
                changed = copy.deepcopy(spec); modify(changed)
                target = out / f"negative_{name}_target.json"; save(target, changed)
                compile_case(f"negative_{name}", args.m, packet_options, target=target,
                             expect="requires imported PIM storage, 4 cores, a 32x32 SA, 8 SC and even bankset")
            ping = copy.deepcopy(storage); ping["bankset_pingpong_lead"] = 12; seal(ping)
            ping_path = out / "negative_ping_manifest.json"; save(ping_path, ping)
            compile_case("negative_ping", args.m, ["--pim-weight-storage-import", ping_path, "--pim-packet-npu=g4"],
                         expect="requires imported PIM storage, 4 cores, a 32x32 SA, 8 SC and even bankset")
            rank = copy.deepcopy(storage); rank["partitions"] = 16
            rank["weights"][0]["stream_part"] = [p+8 for p in rank["weights"][0]["stream_part"]]
            seal(rank); rank_path = out / "negative_rank1_manifest.json"; save(rank_path, rank)
            rank_spec = copy.deepcopy(spec); rank_spec["pim"]["partitions"] = 16
            rank_target = out / "negative_rank1_target.json"; save(rank_target, rank_spec)
            compile_case("negative_rank1", args.m, ["--pim-weight-storage-import", rank_path, "--pim-packet-npu=g4"],
                         target=rank_target, expect="only supports imported rank 0 storage")
            bad = copy.deepcopy(storage); bad["weights"][0]["packet_sha256"] = "0"*64; seal(bad)
            bad_path = out / "negative_packet_hash_manifest.json"; save(bad_path, bad)
            compile_case("negative_packet_content", args.m,
                         ["--pim-weight-storage-import", bad_path, "--pim-packet-npu=g4"], expect="packet SHA256 mismatch")
            injected = out / "injected_packet_attribute.mlir"
            injected.write_text(source_for(args.m)[0].read_text().replace("module {", 'module attributes {hyb.npu_kernel = "pim_packet_g4"} {', 1))
            compile_case("negative_injected_attribute", args.m, packet_options, source_override=injected,
                         expect="reserved PIM weight storage IR attributes")
            nested = out / "injected_nested_packet_attribute.mlir"
            nested.write_text(source_for(args.m)[0].read_text().replace(
                "arith.constant 0.0 : f32", 'arith.constant {hyb.npu_kernel = "pim_packet_g4"} 0.0 : f32', 1))
            compile_case("negative_nested_attribute", args.m, packet_options, source_override=nested,
                         expect="reserved PIM weight storage IR attributes")
            zero = out / "zero_projection.mlir"
            zero.write_text(f'''module {{
  func.func @decode_step(%x: tensor<{args.m}x{args.k}xf32>, %w: tensor<{args.n}x{args.k}xf32>) -> tensor<{args.m}x{args.k}xf32> {{
    return %x : tensor<{args.m}x{args.k}xf32>
  }}
}}
''')
            compile_case("negative_zero_projection", args.m, packet_options, source_override=zero,
                         expect="first version needs one storage-only projection")
            two = out / "two_projection.mlir"
            text = source_for(args.m)[0].read_text().replace(
                f"%w: tensor<{args.n}x{args.k}xf32>)", f"%w: tensor<{args.n}x{args.k}xf32>, %w2: tensor<{args.n}x{args.n}xf32>)", 1)
            text = text.replace(f"    return %y : tensor<{args.m}x{args.n}xf32>", f'''    %wt2 = tensor.empty() : tensor<{args.n}x{args.n}xf32>
    %t2 = linalg.transpose ins(%w2 : tensor<{args.n}x{args.n}xf32>) outs(%wt2 : tensor<{args.n}x{args.n}xf32>) permutation = [1, 0]
    %empty2 = tensor.empty() : tensor<{args.m}x{args.n}xf32>
    %init2 = linalg.fill ins(%zero : f32) outs(%empty2 : tensor<{args.m}x{args.n}xf32>) -> tensor<{args.m}x{args.n}xf32>
    %y2 = linalg.matmul ins(%y, %t2 : tensor<{args.m}x{args.n}xf32>, tensor<{args.n}x{args.n}xf32>) outs(%init2 : tensor<{args.m}x{args.n}xf32>) -> tensor<{args.m}x{args.n}xf32>
    return %y2 : tensor<{args.m}x{args.n}xf32>''')
            two.write_text(text)
            two_front = out / "two_projection_frontend.json"
            frontend = json.loads(source_for(args.m)[1].read_text())
            frontend["arguments"].append(dict(kind="weight", name="proj2.weight"))
            save(two_front, frontend)
            compile_case("negative_two_projection", args.m, packet_options, source_override=two, frontend_override=two_front,
                         expect="supports exactly one driver-requested pim_packet_g4 NPU phase")

        if not args.compile_only:
            os.environ["LD_PRELOAD"] = str(args.dramsim_library.resolve())
            os.environ.pop("LP6PIM_BANKSET_PINGPONG", None)
            xq = np.random.default_rng(99181 + args.k).integers(-127, 128, (args.m, args.k), dtype=np.int8)
            xq[:, 0] = 127
            x = xq.astype(np.float16) / np.float16(64)
            expected_i32 = xq.astype(np.int32) @ wq.astype(np.int32).T
            expected_f16 = (expected_i32.astype(np.float32) / 4096).astype(np.float16)
            np.savez(out / "reference.npz", x=x, xq=xq, expected_i32=expected_i32, expected_f16=expected_f16)
            result["source_x_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()
            for label, build in [("pim", pim), ("packet_npu", npu)]:
                dev = Device(build, settings=args.settings, emulator=args.emulator, lp6_config=args.lp6_config)
                dev.write("x", x)
                result.setdefault("image_audits_after_input", {})[label] = audit_images(
                    storage, [export_build / "lp6.bin", build / "lp6.bin"], wq, source_values)
                outputs, report, wall = dev.run()
                def select(dtype, shape):
                    matches = [(name, outputs[name], meta) for name, meta in dev.outputs.items()
                               if meta["dtype"] == dtype and meta["shape"] == shape]
                    assert len(matches) == 1, (label, dtype, shape, [v[0] for v in matches])
                    return matches[0]
                qname, qvalue, qmeta = select("i8", [args.m, args.k])
                sname, scales, _ = select("f32", [args.m, 1])
                iname, integer, imeta = select("i32", [args.m, args.n])
                assert qvalue.tobytes() == xq.tobytes(), (label, "INT8 row quant")
                assert scales.tobytes() == np.full((args.m, 1), 1/64, dtype=np.float32).tobytes(), (label, "row scales")
                assert integer.tobytes() == expected_i32.tobytes(), (label, "INT32 matmul")
                assert outputs["y"].tobytes() == expected_f16.tobytes(), (label, "FP16 dequant")
                assert imeta["layout"] == ("pim_out" if label == "pim" else "dense")
                assert qmeta["layout"] == ("pim_act" if label == "pim" else "dense")
                times = phase_times(report)
                assert any(p[0] == "collect" for p in times)
                if label == "packet_npu":
                    assert all(c["kind"] != "pim" for c in report["commands"])
                np.savez(build / "actual.npz", quantized=qvalue, row_scales=scales, integer=integer, y=outputs["y"])
                result["native"].append(dict(variant=label, build=str(build), wall_seconds=wall, phase_times=times,
                    endpoint_us=sum(t for name, _, t in times if name != "collect"),
                    collect_us=sum(t for name, _, t in times if name == "collect"),
                    all_INT8_row_quant_byte_exact=True, all_FP32_row_scales_byte_exact=True,
                    all_INT32_matmul_byte_exact=True, all_FP16_output_byte_exact=True,
                    collected_names=dict(quant=qname, scale=sname, integer=iname),
                    int32_sha256=hashlib.sha256(integer.tobytes()).hexdigest(),
                    fp16_sha256=hashlib.sha256(outputs["y"].tobytes()).hexdigest(),
                    program_sha256=sha(build / "program.bin"), image_after_input_sha256=sha(build / "lp6.bin")))
                save(out / "summary.json", result)
                print(f"{label}: all INT8/scales/INT32/FP16 exact; endpoint {result['native'][-1]['endpoint_us']:.5f} us", flush=True)
                if not args.keep_dumps:
                    for dump in build.glob("sram_dump*.bin"):
                        dump.unlink()
                    result["native"][-1]["sram_dumps_removed_after_exact_check"] = True
            assert result["native"][0]["int32_sha256"] == result["native"][1]["int32_sha256"]
            assert result["native"][0]["fp16_sha256"] == result["native"][1]["fp16_sha256"]
        result["all_requested_checks_passed"] = True
    except BaseException:
        result["failure"] = traceback.format_exc()
        raise
    finally:
        result["provenance_after"] = {str(p): sha(p) for p in paths}
        result["provenance_unchanged"] = result["provenance_after"] == before
        save(out / "summary.json", result)
    assert result["provenance_unchanged"], "source or executable changed during regression"
    print(json.dumps(dict(native_cases=len(result["native"]), negative_cases=len(result["negatives"]),
                          all_requested_checks_passed=True, provenance_unchanged=True)), flush=True)


if __name__ == "__main__":
    main()
