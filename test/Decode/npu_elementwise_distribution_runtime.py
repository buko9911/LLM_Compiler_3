#!/usr/bin/env python3
"""Native numerical and actual-core-use checks for generic tensor distribution.

OFF/ON bytes must match. Arithmetic microtests also use an independent NumPy
oracle; nonlinear checks explicitly follow the existing FP16 instruction
rounding contract, not the FP32 spelling of imported tensor IR.
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
from npu_elementwise_distribution import source


def native_source(**kwargs):
    return source(**kwargs).replace("@phase(", "@decode_step(").replace('attributes {hyb.phase="npu"}', "")


def silu_source(rows, columns):
    tensor = f"tensor<{rows}x{columns}xf16>"
    return f"""module {{func.func @decode_step(%gate:{tensor},%up:{tensor})->{tensor} {{
      %one=arith.constant 1.0:f16
      %empty=tensor.empty():{tensor}
      %y=linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(r,c)>,
        affine_map<(r,c)->(r,c)>],iterator_types=[\"parallel\",\"parallel\"]}}
        ins(%gate,%up:{tensor},{tensor}) outs(%empty:{tensor}) {{
      ^bb0(%x:f16,%u:f16,%unused:f16):
        %n=arith.negf %x:f16
        %e=math.exp %n:f16
        %d=arith.addf %e,%one:f16
        %r=arith.divf %one,%d:f16
        %s=arith.mulf %x,%r:f16
        %v=arith.mulf %s,%u:f16
        linalg.yield %v:f16
      }}->{tensor}
      return %y:{tensor}
    }} }}"""


def assembly_source():
    half, full = "tensor<8x32x64xf16>", "tensor<8x32x128xf16>"
    pieces = []
    for name, operation in (("sum", "addf"), ("difference", "subf")):
        pieces.append(f"""%{name}_empty=tensor.empty():{half}
      %{name}=linalg.generic {{indexing_maps=[#id,#id,#id],
        iterator_types=[\"parallel\",\"parallel\",\"parallel\"]}}
        ins(%a,%b:{half},{half}) outs(%{name}_empty:{half}) {{
      ^bb0(%x:f16,%y:f16,%unused:f16):
        %z=arith.{operation} %x,%y:f16
        linalg.yield %z:f16
      }}->{half}""")
    return f"""#id=affine_map<(m,h,d)->(m,h,d)>
    module {{func.func @decode_step(%a:{half},%b:{half})->{full} {{
      {' '.join(pieces)}
      %empty=tensor.empty():{full}
      %left=tensor.insert_slice %sum into %empty[0,0,0][8,32,64][1,1,1]:{half} into {full}
      %joined=tensor.insert_slice %difference into %left[0,0,64][8,32,64][1,1,1]:{half} into {full}
      %last_empty=tensor.empty():{full}
      %last=linalg.generic {{indexing_maps=[#id,#id],
        iterator_types=[\"parallel\",\"parallel\",\"parallel\"]}}
        ins(%joined:{full}) outs(%last_empty:{full}) {{
      ^bb0(%x:f16,%unused:f16):
        %neg=arith.negf %x:f16
        linalg.yield %neg:f16
      }}->{full}
      return %last:{full}
    }} }}"""


def fixtures():
    rng = np.random.default_rng(20261106)
    result = []
    for name, rows, columns, options in [
        ("row_tail", 6, 65, {}),
        ("inner_partition", 1, 9728, {}),
        ("broadcast", 8, 257, {"broadcast": True}),
        ("strided_input", 8, 129, {"strided_input": True}),
        ("destination_read", 8, 128, {"read_init": True}),
        ("producer_consumer", 8, 256, {"chain": True, "share_intermediate": True}),
        ("signed_zero_subnormal", 4, 65, {}),
    ]:
        a_shape = (rows, columns * 2) if options.get("strided_input") else (rows, columns)
        b_shape = (columns,) if options.get("broadcast") else (rows, columns)
        inputs = {"a": (rng.integers(-16, 17, a_shape) / 32).astype(np.float16),
                  "b": (rng.integers(-16, 17, b_shape) / 32).astype(np.float16)}
        if options.get("read_init"):
            inputs["initial"] = (rng.integers(-16, 17, (rows, columns)) / 32).astype(np.float16)
        if name == "signed_zero_subnormal":
            for array in inputs.values():
                array.flat[::4] = np.float16(-0.)
                array.flat[1::4] = np.float16(0.)
                array.flat[2::4] = np.nextafter(np.float16(0), np.float16(1))
                array.flat[3::4] = -np.nextafter(np.float16(0), np.float16(1))
        a = inputs["a"][:, ::2] if options.get("strided_input") else inputs["a"]
        reference = (a.astype(np.float32) + inputs["b"].astype(np.float32)).astype(np.float16)
        if options.get("read_init"):
            reference = (reference.astype(np.float32) + inputs["initial"].astype(np.float32)).astype(np.float16)
        extra_reference = {}
        if options.get("chain"):
            extra_reference["intermediate"] = reference.copy()
            reference = -reference
        result.append(dict(name=name, source=native_source(rows=rows, columns=columns, **options),
            inputs=inputs, reference=reference, extra_reference=extra_reference, tokens=rows, tolerance=0.,
            expected_distributions=0 if options.get("read_init") else 2 if options.get("chain") else 1))
    inputs = {key: (rng.integers(-16, 17, (8, 32, 64)) / 32).astype(np.float16) for key in ("a", "b")}
    reference = -np.concatenate(((inputs["a"] + inputs["b"]).astype(np.float16),
                                  (inputs["a"] - inputs["b"]).astype(np.float16)), axis=-1)
    result.append(dict(name="shared_strided_assembly", source=assembly_source(), inputs=inputs,
        reference=reference, tokens=8, tolerance=0., expected_distributions=3, shared_staging=True))
    for name, rows, columns in (("silu_m1", 1, 9728), ("silu_m8", 8, 9728),
                                ("silu_half_exp_overflow", 1, 256)):
        shape = (rows, columns)
        inputs = {"gate": (rng.integers(-64, 65, shape) / 16).astype(np.float16),
                  "up": (rng.integers(-16, 17, shape) / 16).astype(np.float16)}
        if name == "silu_half_exp_overflow":
            inputs["gate"].fill(-12.)
            inputs["up"].fill(1.)
        with np.errstate(over="ignore"):
            exponent = np.exp(-inputs["gate"].astype(np.float32)).astype(np.float16)
        denominator = (exponent.astype(np.float32) + 1.).astype(np.float16)
        reciprocal = (1. / denominator.astype(np.float32)).astype(np.float16)
        silu = (inputs["gate"].astype(np.float32) * reciprocal.astype(np.float32)).astype(np.float16)
        reference = (silu.astype(np.float32) * inputs["up"].astype(np.float32)).astype(np.float16)
        fp32_reference = (inputs["gate"].astype(np.float32) /
            (1. + np.exp(-inputs["gate"].astype(np.float32))) * inputs["up"].astype(np.float32))
        result.append(dict(name=name, source=silu_source(*shape), inputs=inputs,
            reference=reference, fp32_reference=fp32_reference,
            tokens=rows, tolerance=0., expected_distributions=1))
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    parser.add_argument("--emulator", type=Path, default=workspace / "core/simulator/npu/transactional_emulator/target/release/transactional_emulator")
    parser.add_argument("--settings", type=Path, default=workspace / "research/checkpoint_race/hardware/kernel_probe/settings_4core.toml")
    parser.add_argument("--lp6-library", type=Path, default=workspace / "core/ramulator2/liblp6_ramulator.so")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--command-schedule", action="store_true",
                        help="Also exercise the system command scheduler on both numerical paths")
    args = parser.parse_args()
    sys.path.insert(0, str(root / "runtime"))
    from decode import Device
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ini = workspace / "core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini"
    paths = [Path(__file__).resolve(), Path(__file__).with_name("npu_elementwise_distribution.py"),
        Path(__file__).with_name("npu_attention_fusion_runtime.py"), root / "runtime/decode.py",
        args.compiler.resolve(), args.emulator.resolve(), args.settings.resolve(), ini,
        args.lp6_library.resolve(), args.lp6_library.resolve().parent / "libramulator.so"]
    provenance = {str(p): digest(p) for p in paths}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    model = out / "model"
    model.mkdir()
    save_file({"unused": np.zeros(1, dtype=np.float16)}, model / "model.safetensors")
    cases = fixtures()
    if args.cases:
        assert set(args.cases) <= {c["name"] for c in cases}, "Unknown fixture"
        cases = [c for c in cases if c["name"] in args.cases]
    summary = {"kind": "native_cycle_simulator_functional_microtests", "cases": {}}
    for fixture in cases:
        name = fixture["name"]
        case = out / name
        case.mkdir()
        src = case / "step.mlir"
        src.write_text(fixture["source"])
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model), tokens=fixture["tokens"], capacity=0,
            mask_min=-30000., arguments=[dict(kind="input", name=k) for k in fixture["inputs"]],
            results=[dict(kind="output", name=k) for k in ["y", *fixture.get("extra_reference", {})]]), indent=2))
        np.savez(case / "inputs.npz", **fixture["inputs"])
        np.save(case / "reference.npy", fixture["reference"])
        control = None
        modes = {}
        for mode, enabled in (("off", False), ("distributed", True)):
            spec = json.loads((root / "configs/target.json").read_text())
            spec["npu"].update(elementwise_distribution=enabled, distribute_min_elements=1,
                vector_scalar_reuse=False, vector_chunk_pipeline=False,
                activation_residency=True, attention_fusion=False, flash_attention=False,
                flash_decoding=False, projection_fusion=False, overlap_dequant_gemv=False,
                l2_staging_bytes_per_core=0)
            if args.command_schedule:
                spec["npu"]["command_schedule"] = True
            target = case / f"{mode}.json"
            target.write_text(json.dumps(spec, indent=2))
            build = case / mode
            command = [args.compiler.resolve(), "--pipeline=decode", src, "--manifest", manifest,
                       "--spec", target, "--force=npu", "-o", build]
            run(command, workspace, case / f"{mode}.compile.log", dict(os.environ, HYBRID_PROFILE_OPERATIONS="1"))
            compiled = (build / "compiled.mlir").read_text()
            distributions = len(re.findall(r"hyb\.elementwise_axis =", compiled))
            assert distributions == (fixture["expected_distributions"] if enabled else 0), (name, mode, distributions)
            staging = compiled.count("hyb.shared_copy_staging")
            if enabled and fixture.get("shared_staging"):
                assert staging >= 3, (name, "shared copy legalization was a noop")
            device = Device(build, settings=args.settings, emulator=args.emulator,
                            lp6_config=ini, lp6_backend="ramulator", lp6_library=args.lp6_library)
            for key, array in fixture["inputs"].items():
                device.write(key, array)
            actual_outputs, _, wall = device.run(profile=True, timeline_max_events=2000000)
            actual = actual_outputs["y"]
            np.save(build / "output.npy", actual)
            assert actual.shape == fixture["reference"].shape and np.isfinite(actual).all(), (name, mode)
            error = float(np.max(np.abs(actual.astype(np.float32) - fixture["reference"].astype(np.float32))))
            assert error <= fixture["tolerance"], (name, mode, "independent oracle mismatch", error)
            for key, expected in fixture.get("extra_reference", {}).items():
                assert actual_outputs[key].tobytes() == expected.tobytes(), (name, mode, key, "oracle mismatch")
            if control is None:
                control = {key: array.copy() for key, array in actual_outputs.items()}
            for key, array in actual_outputs.items():
                assert array.tobytes() == control[key].tobytes(), (name, mode, key, "OFF/ON byte mismatch")
            timeline = json.loads((build / "timeline.json").read_text())
            assert timeline["dropped_events"] == 0
            by_core = {}
            for event in timeline["events"]:
                if event["kind"] != "vpu_stream":
                    continue
                match = re.match(r"Core (\d+) /", event["lane"])
                core = match.group(1) if match else "0"
                by_core[core] = by_core.get(core, 0) + 1
            if enabled and fixture["expected_distributions"]:
                assert len(by_core) >= 2, (name, "no actual multicore VPU execution", by_core)
            timing = json.loads((build / "timing.json").read_text())
            modes[mode] = dict(total_us=timing["total_simulated_cycles"] / timing["npu_clock_mhz"],
                max_abs_numpy_error=error, numpy_tolerance=fixture["tolerance"],
                byte_exact_with_control=True, distributions=distributions,
                shared_copy_staging_markers=staging,
                vpu_instruction_records_by_core=by_core, wall_seconds=wall,
                read_bytes=timing["lp6_bytes_read"], write_bytes=timing["lp6_bytes_written"],
                program_sha256=digest(build / "program.bin"))
            if "fp32_reference" in fixture:
                modes[mode]["numpy_contract"] = "existing_instruction_fp16_step_rounding"
                modes[mode]["max_abs_full_fp32_difference"] = float(np.max(np.abs(
                    actual.astype(np.float32) - fixture["fp32_reference"])))
            print(name, mode, json.dumps(modes[mode]), flush=True)
        summary["cases"][name] = modes
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert provenance == {str(p): digest(p) for p in paths}, "Compiler or test inputs changed during run"


if __name__ == "__main__":
    main()
