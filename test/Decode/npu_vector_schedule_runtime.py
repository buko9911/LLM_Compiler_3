#!/usr/bin/env python3
"""Native tests of generic vector scheduling, preserving legacy FP16 rounding.

The independent oracle deliberately models the documented legacy primitive
rounding. A separate FP32 SiLU oracle records its known semantic discrepancy;
it is never hidden by a tolerance or attributed to the scheduling options.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import save_file

from npu_attention_fusion_runtime import digest, run


def rounded(value):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.asarray(value, dtype=np.float16).astype(np.float32)


def source(rows, columns, body, inputs, reduction=False):
    tensor = f"tensor<{rows}x{columns}xf16>"
    scalar = f"tensor<{rows}xf32>"
    types = {name: scalar if kind == "scalar" else tensor for name, kind in inputs}
    output = scalar if reduction else tensor
    # Storage import uses FP16 tensors. Produce row FP32 values through an
    # actual reduction, as the model does for softmax denominators.
    declarations = ",".join(f"%input_{name}:{tensor}" for name, _ in inputs)
    scalar_setup = []
    for name, kind in inputs:
        if kind != "scalar": continue
        scalar_setup.append(f'''%zero_{name} = arith.constant 0.0 : f32
          %empty_{name} = tensor.empty() : {scalar}
          %init_{name} = linalg.fill ins(%zero_{name}:f32) outs(%empty_{name}:{scalar})->{scalar}
          %row_{name} = linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(r)>],
            iterator_types=["parallel","reduction"]}} ins(%input_{name}:{tensor}) outs(%init_{name}:{scalar}) {{
            ^bb0(%value:f16,%accumulator:f32):
              %wide = arith.extf %value : f16 to f32
              %total = arith.addf %wide,%accumulator : f32
              linalg.yield %total : f32
          }}->{scalar}''')
    maps = ["affine_map<(r,c)->(r)>" if kind == "scalar" else "affine_map<(r,c)->(r,c)>"
            for _, kind in inputs] + ["affine_map<(r,c)->(r)>" if reduction else "affine_map<(r,c)->(r,c)>"]
    region = ",".join(f"%{name}_arg:{'f32' if kind == 'scalar' else 'f16'}" for name, kind in inputs)
    init = (f"%zero = arith.constant 0.0 : f32\n%init = linalg.fill ins(%zero:f32) outs(%empty:{output})->{output}"
            if reduction else "")
    return f'''module {{func.func @decode_step({declarations})->{output} {{
      %one = arith.constant 1.0 : f32
      %empty = tensor.empty() : {output}
      {init}
      {''.join(scalar_setup)}
      %y = linalg.generic {{indexing_maps=[{','.join(maps)}],
        iterator_types=["parallel","{'reduction' if reduction else 'parallel'}"]}}
        ins({','.join(('%row_' if kind == 'scalar' else '%input_')+name for name,kind in inputs)}:{','.join(types[name] for name,_ in inputs)})
        outs({'%init' if reduction else '%empty'}:{output}) {{
        ^bb0({region},%out:{'f32' if reduction else 'f16'}):
          {body}
      }}->{output}
      return %y : {output}
    }} }}'''


def fixtures():
    cases = []
    rng = np.random.default_rng(41)
    for rows, columns in [(1, 32), (3, 33), (2, 97), (3, 257)]:
        x = (rng.integers(-16, 17, size=(rows, columns)) / 8).astype(np.float16)
        u = (rng.integers(-8, 9, size=x.shape) / 4).astype(np.float16)
        x.flat[:4] = [-12., -1., -0., 0.]
        u.flat[:4] = 1.
        body = '''%x = arith.extf %x_arg : f16 to f32
          %u = arith.extf %u_arg : f16 to f32
          %negative = arith.negf %x : f32
          %exponent = math.exp %negative : f32
          %denominator = arith.addf %exponent,%one : f32
          %inverse = arith.divf %one,%denominator : f32
          %activation = arith.mulf %x,%inverse : f32
          %product = arith.mulf %activation,%u : f32
          %result = arith.truncf %product : f32 to f16
          linalg.yield %result : f16'''
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denominator = rounded(rounded(np.exp(-x.astype(np.float32))) + 1.)
            reference = rounded(rounded(x.astype(np.float32) * rounded(1. / denominator)) * u).astype(np.float16)
            fp32 = (x.astype(np.float32) / (1. + np.exp(-x.astype(np.float32))) * u).astype(np.float16)
        cases.append(dict(name=f"nonlinear_r{rows}_n{columns}",
            source=source(rows, columns, body, [("x", "vector"), ("u", "vector")]),
            inputs=dict(x=x, u=u), reference=reference, fp32_reference=fp32,
            pipeline=columns > 32, scalar_reuse=False))
    for columns in [33, 97]:
        rows = 3
        x = (rng.integers(-32, 33, size=(rows, columns)) / 8).astype(np.float16)
        d = np.array([2., 4., 8.], dtype=np.float32)
        body = '''%x = arith.extf %x_arg : f16 to f32
          %first = arith.divf %x,%d_arg : f32
          %second = arith.divf %first,%d_arg : f32
          %result = arith.truncf %second : f32 to f16
          linalg.yield %result : f16'''
        reference = rounded(rounded(x.astype(np.float32) / d[:, None]) / d[:, None]).astype(np.float16)
        stored_d = np.zeros_like(x);stored_d[:, 0] = d
        cases.append(dict(name=f"row_divisor_n{columns}",
            source=source(rows, columns, body, [("x", "vector"), ("d", "scalar")]),
            inputs=dict(x=x, d=stored_d), reference=reference, pipeline=True, scalar_reuse=True))
    rows, columns = 3, 65
    x = rng.integers(-8, 9, size=(rows, columns)).astype(np.float16)
    body = '''%wide = arith.extf %x_arg : f16 to f32
      %sum = arith.addf %wide,%out : f32
      linalg.yield %sum : f32'''
    cases.append(dict(name="reduction_excluded", source=source(rows, columns, body, [("x", "vector")], True),
        inputs=dict(x=x), reference=x.astype(np.float32).sum(axis=1), pipeline=False, scalar_reuse=False))
    inputs = {f"x{i}": (rng.integers(-4, 5, size=(2, 97))/4).astype(np.float16) for i in range(9)}
    lines = ["%v1 = arith.addf %x0_arg,%x1_arg : f16"]
    value = rounded(inputs["x0"] + inputs["x1"])
    for i in range(2, 9):
        lines.append(f"%v{i} = arith.addf %v{i-1},%x{i}_arg : f16")
        value = rounded(value + inputs[f"x{i}"])
    lines.append("linalg.yield %v8 : f16")
    cases.append(dict(name="register_pressure_fallback", source=source(2, 97, "\n".join(lines), [(n,"vector") for n in inputs]),
        inputs=inputs, reference=value.astype(np.float16), pipeline=False, scalar_reuse=False))
    return cases


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=root/"build/bin/npu-compile")
    parser.add_argument("--emulator", type=Path, default=workspace/"core/simulator/npu/transactional_emulator/target/release/transactional_emulator")
    parser.add_argument("--settings", type=Path, default=workspace/"research/checkpoint_race/hardware/kernel_probe/settings_4core.toml")
    parser.add_argument("--lp6-library", type=Path, default=workspace/"core/ramulator2/liblp6_ramulator.so")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    sys.path.insert(0, str(root/"runtime"))
    from decode import Device
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), args.compiler, args.emulator, args.settings, args.lp6_library]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    (out/"provenance.json").write_text(json.dumps(hashes, indent=2)+"\n")
    report = dict(kind="native_vector_scheduling_microtests", numerical_contract="legacy FP16 primitives, FP32 scalar/reduction",
                  fp32_reference="Diagnostic only; known legacy mismatch is recorded, not accepted as FP32 correctness", cases={})
    cases = fixtures()
    if args.cases:
        assert set(args.cases) <= {f["name"] for f in cases}
        cases = [f for f in cases if f["name"] in args.cases]
    for fixture in cases:
        case = out/fixture["name"];case.mkdir()
        model = case/"model";model.mkdir()
        save_file({"unused": np.zeros(1,dtype=np.float16)}, model/"model.safetensors")
        source_path = case/"step.mlir";source_path.write_text(fixture["source"])
        manifest = case/"manifest.json"
        manifest.write_text(json.dumps(dict(model=str(model), tokens=next(iter(fixture["inputs"].values())).shape[0], capacity=0,
            mask_min=-30000., arguments=[dict(kind="input",name=n) for n in fixture["inputs"]],
            results=[dict(kind="output",name="y")])))
        np.savez(case/"inputs.npz", **fixture["inputs"])
        np.save(case/"legacy_reference.npy", fixture["reference"])
        control = None
        records = {}
        for mode in ["off", "reuse", "pipeline", "both"]:
            spec = json.loads((root/"configs/target.json").read_text())
            reuse, pipeline = mode in ("reuse","both"), mode in ("pipeline","both")
            spec["npu"].update(vector_scalar_reuse=reuse, vector_chunk_pipeline=pipeline,
                elementwise_distribution=False, staging_early_release=False,
                memory_dependency_schedule=False, attention_fusion=False, projection_fusion=False,
                activation_residency=True,
                input_tile_pipeline=False, matmul_rhs_packing=False, matmul_transpose_output=False)
            target = case/f"{mode}.json";target.write_text(json.dumps(spec,indent=2))
            build = case/mode
            command = [args.compiler, "--pipeline=decode", source_path, "--manifest", manifest,
                       "--spec",target,"--force=npu","-o",build]
            run(command,workspace,case/f"{mode}.compile.log",dict(os.environ,HYBRID_PROFILE_OPERATIONS="1"))
            device = Device(build, settings=args.settings, emulator=args.emulator,
                lp6_config=workspace/"core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini",
                lp6_backend="ramulator",lp6_library=args.lp6_library)
            for name,value in fixture["inputs"].items(): device.write(name,value)
            output,_,wall=device.run(profile=True,timeline_max_events=500000)
            actual=output["y"];np.save(build/"output.npy",actual)
            reference=fixture["reference"]
            assert actual.dtype==reference.dtype and actual.shape==reference.shape,(fixture["name"],actual.shape,reference.shape)
            assert actual.tobytes()==reference.tobytes(),(fixture["name"],mode,"legacy oracle mismatch",float(np.max(np.abs(actual.astype(np.float32)-reference))))
            if control is None: control=actual.copy()
            assert actual.tobytes()==control.tobytes(),(fixture["name"],mode,"optimization changed output")
            io=json.loads((build/"io.json").read_text())
            fields=[p.get("vector_schedule",{}) for p in io["phases"]]
            stats={key:sum(p.get(key,0) for p in fields) for key in
                ["scalar_reciprocals_hoisted","scalar_reciprocal_reuses","pipeline_groups","pipelined_chunks"]}
            expect_reuse=reuse and fixture["scalar_reuse"]
            expect_pipeline=pipeline and fixture["pipeline"] and (not fixture["scalar_reuse"] or reuse)
            assert (stats["scalar_reciprocals_hoisted"]>0)==expect_reuse,(fixture["name"],mode,stats)
            assert (stats["pipeline_groups"]>0)==expect_pipeline,(fixture["name"],mode,stats)
            timing=json.loads((build/"timing.json").read_text())
            record=dict(total_us=timing["total_simulated_cycles"]/timing["npu_clock_mhz"], wall_seconds=wall,
                byte_exact_with_off=True,byte_exact_with_legacy_oracle=True,actual_applied=stats,
                program_sha256=digest(build/"program.bin"))
            if "fp32_reference" in fixture:
                fp32=fixture["fp32_reference"]
                record.update(fp32_different_elements=int(np.count_nonzero(actual.view(np.uint16)!=fp32.view(np.uint16))),
                    fp32_max_absolute_error=float(np.max(np.abs(actual.astype(np.float32)-fp32))))
                assert record["fp32_different_elements"]>0,"known precision discrepancy disappeared; review numerical contract"
            records[mode]=record
            print(fixture["name"],mode,json.dumps(record),flush=True)
        report["cases"][fixture["name"]]=records
        (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    assert hashes=={str(p.resolve()):digest(p) for p in paths},"Compiler/source changed during native validation"


if __name__=="__main__": main()
