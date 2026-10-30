#!/usr/bin/env python3
"""Native regressions for common staging lifetime and local DMA dependencies.

These are compiler/emulator microtests, not model throughput measurements.
All modes use the existing SRAM/DRAM settings, identical input bytes and an
independent NumPy oracle. The timeline observations are execution evidence;
an absent overlap is reported rather than turned into an assumed speedup.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from safetensors.numpy import save_file

from npu_attention_fusion_runtime import digest, run
from npu_gemv_optimizations_runtime import source as gemv_source


def reduction_source(rows, columns, stride=1):
    original = f"tensor<{rows}x{columns * stride}xf16>"
    tensor = f"tensor<{rows}x{columns}xf16>"
    scalar = f"tensor<{rows}xf32>"
    vector = f"tensor<{columns}xf16>"
    sliced = (f"%x = tensor.extract_slice %input[0,0][{rows},{columns}][1,{stride}] "
              f": {original} to {tensor}" if stride != 1 else "")
    x = "%x" if stride != 1 else "%input"
    return f'''module {{func.func @decode_step(%input:{original},%gamma:{vector})->{tensor} {{
      {sliced}
      %zero = arith.constant 0.0 : f32
      %empty = tensor.empty() : {scalar}
      %init = linalg.fill ins(%zero:f32) outs(%empty:{scalar})->{scalar}
      %sum = linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(r)>],
        iterator_types=["parallel","reduction"]}} ins({x}:{tensor}) outs(%init:{scalar}) {{
        ^bb0(%value:f16,%acc:f32):
          %wide = arith.extf %value : f16 to f32
          %next = arith.addf %wide,%acc : f32
          linalg.yield %next : f32
      }}->{scalar}
      %output = tensor.empty() : {tensor}
      %y = linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(c)>,
        affine_map<(r,c)->(r)>,affine_map<(r,c)->(r,c)>],iterator_types=["parallel","parallel"]}}
        ins({x},%gamma,%sum:{tensor},{vector},{scalar}) outs(%output:{tensor}) {{
        ^bb0(%value:f16,%scale:f16,%total_arg:f32,%unused:f16):
          %addition = arith.addf %value,%scale : f16
          %rounded = arith.truncf %total_arg : f32 to f16
          %product = arith.mulf %addition,%rounded : f16
          linalg.yield %product : f16
      }}->{tensor}
      return %y : {tensor}
    }} }}'''


def matrix_source(m, k, n):
    a, b, c = (f"tensor<{x}x{y}xf16>" for x, y in ((m, k), (k, n), (m, n)))
    gamma = f"tensor<{n}xf16>"
    return f'''module {{func.func @decode_step(%a:{a},%b:{b},%gamma:{gamma})->{c} {{
      %zero = arith.constant 0.0 : f16
      %empty = tensor.empty() : {c}
      %init = linalg.fill ins(%zero:f16) outs(%empty:{c})->{c}
      %product = linalg.matmul ins(%a,%b:{a},{b}) outs(%init:{c})->{c}
      %output = tensor.empty() : {c}
      %y = linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(c)>,
        affine_map<(r,c)->(r,c)>],iterator_types=["parallel","parallel"]}}
        ins(%product,%gamma:{c},{gamma}) outs(%output:{c}) {{
        ^bb0(%value:f16,%scale:f16,%unused:f16):
          %sum = arith.addf %value,%scale : f16
          linalg.yield %sum : f16
      }}->{c}
      return %y : {c}
    }} }}'''


def intervals(events):
    result = []
    for event in sorted(events, key=lambda item: item["begin_cycle"]):
        begin, end = event["begin_cycle"], event["end_cycle"]
        if end <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def intersection(left, right):
    a, b = intervals(left), intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def observe(build, io):
    timeline = json.loads((build / "timeline.json").read_text())
    assert timeline["dropped_events"] == 0, "Timeline overflow invalidates overlap evidence"
    events = timeline["events"]
    matrix = [e for e in events if e["kind"] == "matrix_compute"]
    # Async output is a true transfer interval, not an instruction issue or wait.
    output = [e for e in events if e["kind"] == "ldma_write"]
    reductions = [e for e in events if e["kind"] == "vpu_stream" and
                  ("reduce" in e["label"].lower() or "sumf16" in e["label"].lower())]
    gamma = next((a for a in io["args"] if a["name"] == "gamma"), None)
    gamma_reads = []
    if gamma:
        for event in events:
            if event["kind"] != "gdma_load":
                continue
            address = re.search(r"LP6 (0x[0-9a-fA-F]+)", event["detail"])
            if address and gamma["addr"] <= int(address.group(1), 16) < gamma["addr"] + gamma["bytes"]:
                gamma_reads.append(event)
    def core(event):
        match = re.match(r"Core (\d+) /", event["lane"])
        return int(match.group(1)) if match else 0
    matrix_overlap = {str(c): intersection([e for e in output if core(e) == c],
                                           [e for e in matrix if core(e) == c])
                      for c in sorted({core(e) for e in output + matrix})}
    return dict(
        gamma_gdma_events=len(gamma_reads), reduction_events=len(reductions),
        gamma_reduction_overlap_cycles=intersection(gamma_reads, reductions),
        gamma_first_begin=min((e["begin_cycle"] for e in gamma_reads), default=None),
        gamma_last_end=max((e["end_cycle"] for e in gamma_reads), default=None),
        reduction_first_begin=min((e["begin_cycle"] for e in reductions), default=None),
        reduction_last_end=max((e["end_cycle"] for e in reductions), default=None),
        async_output_events=len(output), matrix_compute_events=len(matrix),
        output_dma_matrix_overlap_cycles_by_core=matrix_overlap,
        local_event_waits=sum(e["kind"] == "event_wait" for e in events),
        core_blocks=sum(e["kind"] == "core_block" for e in events),
    )


def fixtures():
    rng = np.random.default_rng(20261105)
    cases = []
    for name, rows, columns, stride, window in [
        ("reduction_gamma", 1, 8192, 1, 0),
        ("strided_reduction_gamma", 2, 513, 2, 1024),
        ("multichunk_reduction_gamma", 1, 8192, 1, 1024),
    ]:
        inputs = {"input": (rng.integers(-4, 5, (rows, columns * stride)) / 16).astype(np.float16),
                  "gamma": (rng.integers(-4, 5, columns) / 16).astype(np.float16)}
        values = inputs["input"][:, ::stride]
        sums = values.astype(np.float32).sum(axis=-1, keepdims=True).astype(np.float16)
        reference = ((values + inputs["gamma"]).astype(np.float16) * sums).astype(np.float16)
        cases.append(dict(name=name, source=reduction_source(rows, columns, stride), inputs=inputs,
                          reference=reference, tokens=rows, staging=window, weights={}, options={}))
    for name, m, k, n in [("predecessor_matrix", 8, 32, 256), ("matrix_tail", 5, 35, 37)]:
        inputs = {"a": (rng.integers(-4, 5, (m, k)) / 16).astype(np.float16),
                  "b": (rng.integers(-4, 5, (k, n)) / 16).astype(np.float16),
                  "gamma": (rng.integers(-4, 5, n) / 16).astype(np.float16)}
        reference = (np.matmul(inputs["a"].astype(np.float32), inputs["b"].astype(np.float32))
                     .astype(np.float16) + inputs["gamma"]).astype(np.float16)
        cases.append(dict(name=name, source=matrix_source(m, k, n), inputs=inputs,
                          reference=reference, tokens=m, staging=1024, weights={}, options={}))
    # These use the real quantized GEMV streaming loop. Preserving the event-1
    # weight prefetch path while output event 2 crosses matrix work is essential.
    for name, m, k, n, tile in [("stream_output_matrix", 32, 32, 1024, 4096),
                               ("stream_output_tail", 33, 64, 1152, 6144)]:
        xq = rng.integers(-127, 128, (m, k), dtype=np.int8)
        wq = rng.integers(-127, 128, (n, k), dtype=np.int8)
        xq[:, 0] = wq[:, 0] = 127
        x = (xq.astype(np.float16) / np.float16(64)).astype(np.float16)
        weight = (wq.astype(np.float16) / np.float16(64)).astype(np.float16)
        reference = ((xq.astype(np.int32) @ wq.astype(np.int32).T).astype(np.float32) / 4096).astype(np.float16)
        cases.append(dict(name=name, source=gemv_source(m, k, n), inputs={"x": x},
                          reference=reference, tokens=m, staging=0, weights={"proj.weight": weight},
                          options=dict(gemv_tile_bytes=tile, matrix_init_elision=True,
                                       gemv_epilogue_fusion=False)))
    return cases


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    parser.add_argument("--emulator", type=Path, default=workspace / "core/simulator/npu/transactional_emulator/target/release/transactional_emulator")
    parser.add_argument("--settings", type=Path, default=workspace / "research/checkpoint_race/hardware/kernel_probe/settings_4core.toml")
    parser.add_argument("--lp6-library", type=Path, default=workspace / "core/ramulator2/liblp6_ramulator.so")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="Optional exact fixture names")
    parser.add_argument("--modes", nargs="+", choices=("off", "release", "memory", "both", "blocking"),
                        default=("off", "release", "memory", "both"))
    args = parser.parse_args()
    if args.modes[0] not in ("off", "blocking"):
        parser.error("First mode must be a numerical control: off or blocking")
    sys.path.insert(0, str(root / "runtime"))
    from decode import Device
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ini = workspace / "core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini"
    paths = [Path(__file__).resolve(), Path(__file__).with_name("npu_gemv_optimizations_runtime.py"),
             Path(__file__).with_name("npu_attention_fusion_runtime.py"), root / "runtime/decode.py",
             args.compiler.resolve(), args.emulator.resolve(), args.settings.resolve(), ini,
             args.lp6_library.resolve(), args.lp6_library.resolve().parent / "libramulator.so"]
    provenance = {str(p): digest(p) for p in paths}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {"kind": "native_cycle_simulator_functional_microtests", "cases": {}}
    cases = fixtures()
    if args.cases:
        assert set(args.cases) <= {c["name"] for c in cases}, "Unknown fixture name"
        cases = [c for c in cases if c["name"] in args.cases]
    for fixture in cases:
        name = fixture["name"]
        case = out / name
        case.mkdir()
        model = case / "model"
        model.mkdir()
        weights = fixture["weights"] or {"unused": np.zeros(1, dtype=np.float16)}
        save_file(weights, model / "model.safetensors")
        source = case / "step.mlir"
        source.write_text(fixture["source"])
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model), tokens=fixture["tokens"], capacity=0,
            mask_min=-30000., arguments=[dict(kind="input", name=n) for n in fixture["inputs"]] +
            [dict(kind="weight", name=n) for n in fixture["weights"]],
            results=[dict(kind="output", name="y")]), indent=2))
        np.savez(case / "inputs.npz", **fixture["inputs"])
        np.save(case / "reference.npy", fixture["reference"])
        control = None
        modes = {}
        for mode in args.modes:
            spec = json.loads((root / "configs/target.json").read_text())
            spec["npu"].update(staging_early_release=mode in ("release", "both"),
                memory_dependency_schedule=mode in ("memory", "both"),
                async_input_loads=mode != "blocking",
                async_output_stores=mode != "blocking", async_output_store_vector_overlap=True,
                attention_fusion=False, projection_fusion=False, activation_residency=True,
                input_tile_pipeline=False, matmul_rhs_packing=False, matmul_transpose_output=False,
                l2_staging_bytes_per_core=fixture["staging"], **fixture["options"])
            target = case / f"{mode}.json"
            target.write_text(json.dumps(spec, indent=2))
            build = case / mode
            command = [args.compiler.resolve(), "--pipeline=decode", source, "--manifest", manifest,
                       "--spec", target, "--force=npu", "-o", build]
            run(command, workspace, case / f"{mode}.compile.log", dict(os.environ, HYBRID_PROFILE_OPERATIONS="1"))
            io = json.loads((build / "io.json").read_text())
            device = Device(build, settings=args.settings, emulator=args.emulator,
                            lp6_config=ini, lp6_backend="ramulator", lp6_library=args.lp6_library)
            for key, array in fixture["inputs"].items():
                device.write(key, array)
            (build / "invocation.json").write_text(json.dumps(dict(command=list(map(str, command)),
                program_sha256=digest(build / "program.bin"), image_sha256=digest(build / "lp6.bin"),
                settings_sha256=digest(device.settings)), indent=2))
            output, _, wall = device.run(profile=True, timeline_max_events=2000000)
            actual = output["y"]
            np.save(build / "output.npy", actual)
            reference = fixture["reference"]
            assert actual.shape == reference.shape, (name, mode, actual.shape, reference.shape)
            assert np.isfinite(actual).all(), (name, mode, "nonfinite output")
            error = float(np.max(np.abs(actual.astype(np.float32) - reference.astype(np.float32))))
            assert error == 0., (name, mode, "independent oracle mismatch", error)
            if control is None:
                control = actual.copy()
            assert actual.tobytes() == control.tobytes(), (name, mode, "control byte mismatch")
            timing = json.loads((build / "timing.json").read_text())
            cuts = sum(p.get("dependency_schedule", {}).get("staging_release_cuts", 0) for p in io["phases"])
            if mode in ("release", "both") and not name.startswith("stream_"):
                assert cuts > 0, (name, mode, "release option was a noop")
            observation = observe(build, io)
            modes[mode] = dict(total_us=timing["total_simulated_cycles"] / timing["npu_clock_mhz"],
                wall_seconds=wall, byte_exact_with_control=True, max_abs_numpy_error=error,
                staging_release_cuts=cuts, read_bytes=timing["lp6_bytes_read"],
                write_bytes=timing["lp6_bytes_written"], observations=observation,
                phases=io["phases"], program_sha256=digest(build / "program.bin"))
            print(name, mode, json.dumps({k: v for k, v in modes[mode].items() if k != "phases"}), flush=True)
        if name == "reduction_gamma" and "off" in modes:
            before = modes["off"]["observations"]
            assert before["reduction_events"] > 0 and before["gamma_gdma_events"] > 0
            for mode in ("release", "both"):
                if mode in modes:
                    after = modes[mode]["observations"]
                    assert after["gamma_reduction_overlap_cycles"] > 0, (mode, "no actual reuse of freed staging")
                    assert after["gamma_first_begin"] < before["gamma_first_begin"], (mode, "gamma was not released earlier")
        if name.startswith("stream_output_") and "off" in modes:
            before = modes["off"]["observations"]["output_dma_matrix_overlap_cycles_by_core"]
            for mode in ("memory", "both"):
                if mode in modes:
                    after = modes[mode]["observations"]["output_dma_matrix_overlap_cycles_by_core"]
                    assert any(cycles > before.get(core, 0) for core, cycles in after.items()), (name, mode, "matrix DMA scheduling was a noop")
        summary["cases"][name] = modes
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert provenance == {str(p): digest(p) for p in paths}, "Build/source changed during run"


if __name__ == "__main__":
    main()
