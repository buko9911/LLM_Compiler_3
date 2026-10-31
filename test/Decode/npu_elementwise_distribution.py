#!/usr/bin/env python3
"""Admission and storage-independent structure checks for generic distribution."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def source(rows=8, columns=9728, phase="npu", dtype="f16", broadcast=False,
           indexed=False, chain=False, share_intermediate=False, read_init=False, reduction=False,
           transposed_input=False, strided_input=False, dynamic=False):
    row = "?" if dynamic else str(rows)
    tensor = f"tensor<{row}x{columns}x{dtype}>"
    input_tensor = f"tensor<{columns}x{row}x{dtype}>" if transposed_input else tensor
    if strided_input:
        input_tensor = f"tensor<{row}x{columns * 2}x{dtype}>"
    other = f"tensor<{columns}x{dtype}>" if broadcast else tensor
    output = f"tensor<{row}x{dtype}>" if reduction else tensor
    amap = "c,r" if transposed_input else "r,c"
    bmap = "c" if broadcast else "r,c"
    cmap = "r" if reduction else "r,c"
    iterators = '"parallel","reduction"' if reduction else '"parallel","parallel"'
    body = "%v = arith.addf %a0,%b0 : " + dtype
    if indexed:
        body = f"""%index = linalg.index 0 : index
        %integer = arith.index_cast %index : index to i64
        %offset = arith.sitofp %integer : i64 to {dtype}
        %v = arith.addf %a0,%offset : {dtype}"""
    if read_init or reduction:
        body += f"\n%w = arith.addf %v,%old : {dtype}"
    yielded = "%w" if read_init or reduction else "%v"
    initializer = "%initial" if read_init else "%empty"
    extra = f",%initial:{output}" if read_init else ""
    shape = "%rows" if dynamic else ""
    dimensions = f"%zero=arith.constant 0:index\n%rows=tensor.dim %a,%zero:{input_tensor}" if dynamic else ""
    if strided_input:
        dimensions += (f"\n%a_view=tensor.extract_slice %a[0,0][{rows},{columns}][1,2]"
                       f":{input_tensor} to {tensor}")
    input_value = "%a_view" if strided_input else "%a"
    generic_input = tensor if strided_input else input_tensor
    init = f"%empty=tensor.empty({shape}):{output}" if not read_init else ""
    if reduction:
        init += f"\n%z=arith.constant 0.0:{dtype}\n%initialized=linalg.fill ins(%z:{dtype}) outs(%empty:{output})->{output}"
        initializer = "%initialized"
    second = ""
    if chain:
        second = f"""%last_empty=tensor.empty():{tensor}
      %last=linalg.generic {{indexing_maps=[affine_map<(r,c)->(r,c)>,affine_map<(r,c)->(r,c)>],
        iterator_types=[\"parallel\",\"parallel\"]}} ins(%result:{tensor}) outs(%last_empty:{tensor}) {{
      ^bb0(%x:{dtype},%unused:{dtype}):
        %neg=arith.negf %x:{dtype}
        linalg.yield %neg:{dtype}
      }}->{tensor}"""
    returned = "%last" if chain else "%result"
    signature = f"({output},{output})" if share_intermediate else output
    returned_values = f"{returned},%result:{output},{output}" if share_intermediate else f"{returned}:{output}"
    return f"""module {{func.func @phase(%a:{input_tensor},%b:{other}{extra})->{signature}
      attributes {{hyb.phase=\"{phase}\"}} {{
      {dimensions}
      {init}
      %result=linalg.generic {{indexing_maps=[affine_map<(r,c)->({amap})>,affine_map<(r,c)->({bmap})>,
        affine_map<(r,c)->({cmap})>],iterator_types=[{iterators}]}}
        ins({input_value},%b:{generic_input},{other}) outs({initializer}:{output}) {{
      ^bb0(%a0:{dtype},%b0:{dtype},%old:{dtype}):
        {body}
        linalg.yield {yielded}:{dtype}
      }}->{output}
      {second}
      return {returned_values}
    }} }}"""


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    # (source args, spec overrides, expected distributed ops, partition axis/cores).
    cases = {
        "m8_rows": ({}, {}, 1, (0, 4)),
        "m1_columns": ({"rows": 1}, {"distribute_min_elements": 4096, "core_split_us": 0}, 1, (1, 4)),
        "m1_default_threshold": ({"rows": 1}, {}, 0, None),
        "broadcast_columns": ({"broadcast": True}, {}, 1, (0, 4)),
        "transpose_read_rejected": ({"transposed_input": True}, {}, 0, None),
        "strided_view": ({"strided_input": True}, {}, 1, (0, 4)),
        "read_destination": ({"read_init": True}, {}, 0, None),
        "producer_consumer": ({"chain": True}, {}, 2, (0, 4)),
        "row_divisor_tail": ({"rows": 6, "columns": 65}, {"distribute_min_elements": 1, "core_split_us": 0}, 1, (0, 3)),
        "inner_divisor": ({"rows": 1, "columns": 96}, {"distribute_min_elements": 1, "core_split_us": 0}, 1, (1, 3)),
        # The cost rule (NpuCost.h): the same op stays on one core when splitting saves less than the
        # fixed cost of a split, and is split when the cost is lowered.
        "cost_one_core": ({"rows": 1, "columns": 96}, {"distribute_min_elements": 1, "core_split_us": 0.5}, 0, None),
        "cost_split": ({"rows": 1, "columns": 96}, {"distribute_min_elements": 1, "core_split_us": 0.001}, 1, (1, 3)),
        "inner_unaligned": ({"rows": 1, "columns": 100}, {"distribute_min_elements": 1, "core_split_us": 0}, 0, None),
        "disabled": ({}, {"elementwise_distribution": False}, 0, None),
        "single_core": ({}, {"cores": 1}, 0, None),
        "small": ({"rows": 1, "columns": 32}, {}, 0, None),
        "capacity": ({}, {"l1_bytes": 262400}, 0, None),
        "pim": ({"phase": "pim"}, {}, 0, None),
        "f32": ({"dtype": "f32"}, {}, 0, None),
        "dynamic": ({"dynamic": True}, {}, 0, None),
        "index_dependent": ({"indexed": True}, {}, 0, None),
        "reduction": ({"reduction": True}, {}, 0, None),
        "mixed_phases": ({}, {}, 0, None),
    }
    results = {}
    for name, (kwargs, options, expected, partition) in cases.items():
        text = source(**kwargs)
        if name == "mixed_phases":
            text = text[:-1] + 'func.func private @pim() attributes {hyb.phase="pim"}\n}'
        src = out / f"{name}.mlir"
        src.write_text(text)
        spec = json.loads((root / "configs/target.json").read_text())
        spec["npu"].update(elementwise_distribution=True)
        spec["npu"].update(options)
        target = out / f"{name}.json"
        target.write_text(json.dumps(spec))
        flag = f"--hyb-npu-elementwise-distribute=spec={target.resolve()}"
        command = [str(args.optimizer.resolve()), str(src), flag]
        run = subprocess.run(command, text=True, capture_output=True)
        (out / f"{name}.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 0, (name, run.stderr)
        actual = len(re.findall(r"hyb\.elementwise_axis =", run.stdout))
        assert actual == expected, (name, actual, expected)
        if name.startswith("cost_"):     # the choice and its numbers are recorded on the phase
            choice = "split" if expected else "one core"
            assert re.search(r'choice = "%s".*kind = "split"' % choice, run.stdout), (name, "decision record")
        if expected:
            axis, cores = partition
            assert run.stdout.count(f"hyb.elementwise_axis = {axis} : i64") == expected
            assert run.stdout.count(f"hyb.cores = {cores} : i64") == expected
            assert f"hyb.elementwise_distributions = {expected} : i64" in run.stdout
            assert run.stdout.count("tensor.parallel_insert_slice") == expected
            assert run.stdout.count("arith.addf") == text.count("arith.addf")
            again = subprocess.run([command[0], "-", flag], input=run.stdout, text=True, capture_output=True)
            assert again.returncode == 0, (name, again.stderr)
            assert again.stdout.count("scf.forall (") == expected, (name, "double distribution")
            assert f"hyb.elementwise_distributions = {expected} : i64" in again.stdout
        results[name] = dict(passed=True, distributions=actual)
    for option in ("elementwise_distribution", "vector_scalar_reuse", "vector_chunk_pipeline", "command_schedule"):
        for invalid in (1, "true", None):
            spec = json.loads((root / "configs/target.json").read_text())
            spec["npu"][option] = invalid
            target = out / "invalid.json"
            target.write_text(json.dumps(spec))
            run = subprocess.run([str(args.optimizer.resolve()), str(out / "m8_rows.mlir"),
                f"--hyb-npu-elementwise-distribute=spec={target.resolve()}"], text=True, capture_output=True)
            assert run.returncode != 0 and option in run.stderr, (option, invalid, run.stderr)
            results[f"invalid_{option}_{repr(invalid)}"] = dict(passed=True)
    shared = out / "shared_copy.mlir"
    shared.write_text('''module {func.func @shared(%src:memref<8x32x64xf16,2>,
      %dst:memref<8x32x64xf16,strided<[4096,128,1]>,2>) attributes {hyb.phase="npu"} {
      memref.copy %src,%dst:memref<8x32x64xf16,2> to memref<8x32x64xf16,strided<[4096,128,1]>,2>
      return
    }}''')
    run = subprocess.run([str(args.optimizer.resolve()), str(shared), "--hyb-npu-promote"], text=True, capture_output=True)
    (out / "shared_copy.log").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stderr
    assert run.stdout.count("hyb.shared_copy_staging") == 3, run.stdout
    assert run.stdout.count("memref.copy") == 2, run.stdout
    assert "memref<8x32x64xf16, 1>" in run.stdout and "strided<[4096, 128, 1]>" in run.stdout
    results["shared_strided_copy_legalization"] = dict(passed=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
