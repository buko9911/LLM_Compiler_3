#!/usr/bin/env python3
"""Four-core PIM activation feed on one and on two full ranks: plan checks and a native run.

Uses the main compiler's image and program. The M5 case reuses core0's staging slot for token4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file

from pim_packet_npu import ROOT, LP6, Device, fixture, json_attributes, save, sha
from pim_activation_cores import check_plan, ancestors
import workspace

EVIDENCE = workspace.EVIDENCE / "2026-10-06/research"


def compile_case(compiler, out, name, source, manifest, target, cores=1,
                 image=False, expect=None):
    build = out / name
    assert not build.exists(), build
    argv = [str(compiler), "--pipeline=decode", str(source), "--manifest", str(manifest),
            "--spec", str(target), "--force=pim", "--pim-bankset-pingpong=0",
            "--debug-collect", f"--pim-activation-cores={cores}", "-o", str(build)]
    if not image:
        argv.append("--no-image")
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    text = result.stdout + result.stderr
    save(out / f"{name}_compile.json", dict(argv=argv, exit_code=result.returncode,
                                            expected_diagnostic=expect))
    (out / f"{name}_compile.log").write_text(text)
    if expect:
        assert result.returncode != 0 and expect in text, text[-6000:]
    else:
        assert result.returncode == 0, text[-6000:]
    return build


def prepare(args):
    out = args.output.resolve()
    out.mkdir(exist_ok=True)
    assert not (out / "fixture.json").exists()
    model = out / "model"; model.mkdir()
    rng = np.random.default_rng(812607)
    wq = rng.integers(-127, 128, (8192, 512), dtype=np.int8); wq[:, 0] = 127
    qx = rng.integers(-127, 128, (5, 512), dtype=np.int8); qx[:, 0] = 127
    save_file({"proj.weight": (wq.astype(np.float32)/np.float32(64)).astype(np.float16)},
              str(model / "model.safetensors"))
    x = (qx.astype(np.float32)/np.float32(64)).astype(np.float16)
    acc = qx.astype(np.int32) @ wq.astype(np.int32).T
    y = ((acc.astype(np.float32)*np.float32(1/64))*np.float32(1/64)).astype(np.float16)
    np.savez(out / "reference.npz", x=x, qx=qx, sx=np.full((5, 1), 1/64, np.float32), acc=acc, y=y)
    source, manifest = fixture(out / "frontend", 5, 8192, 512, model)
    base = json.loads(args.target.read_text())
    paths = {}
    for name, parts, concurrent in [("single", 8, False), ("dual", 16, True), ("partial", 12, True)]:
        target = json.loads(json.dumps(base))
        target["pim"].update(channels=8, ranks=2, partitions=parts, concurrent_ranks=concurrent)
        paths[name] = out / f"target_{name}.json"; save(paths[name], target)
    save(out / "fixture.json", dict(source=str(source), manifest=str(manifest),
         targets={k: str(v) for k, v in paths.items()}, small=dict(M=5, N=8192, K=512)))


def check(args):
    out = args.output.resolve(); cfg = json.loads((out / "fixture.json").read_text())
    source, manifest = Path(cfg["source"]), Path(cfg["manifest"])
    targets = {k: Path(v) for k, v in cfg["targets"].items()}
    checks = {}
    for name, target, cores in [("single1", targets["single"], 1),
                                ("single4", targets["single"], 4),
                                ("dual1", targets["dual"], 1)]:
        compile_case(args.compiler, out, f"after_{name}", source, manifest, target, cores)
    checks["single4_plan"] = check_plan(out / "after_single4", out / "after_single1")
    dual = compile_case(args.compiler, out, "after_dual4", source, manifest, targets["dual"], 4)
    checks["dual4_plan"] = check_plan(dual, out / "after_dual1")
    assert checks["dual4_plan"]["plan"]["ranks"] == [0, 1]
    assert checks["dual4_plan"]["replica_reuse_dependency_checks"] > 0
    compile_case(args.compiler, out, "negative_partial_rank", source, manifest, targets["partial"], 4,
                 expect="one or two full ranks")
    # The changed guard also revalidates uniqueness/rank geometry/equal rank
    # sets for every activation row. Current PlanLP6 always gives the rows of
    # one phase the same set; malformed internal metadata is not fabricated
    # through a new production hook just to make a CLI negative test.
    checks["negative_partial_rank_rejected"] = True
    checks["internal_equal_row_rank_set_guard"] = "source review; current frontend planner produces equal sets"
    checks["compiler_sha256"] = sha(args.compiler)
    save(out / "compile_checks.json", checks)
    print("Compile, preservation, transfer and dependency checks passed", flush=True)


def native(args):
    out = args.output.resolve(); cfg = json.loads((out / "fixture.json").read_text())
    source, manifest = Path(cfg["source"]), Path(cfg["manifest"])
    target = Path(cfg["targets"]["dual"]); reference = out / "reference.npz"
    tracked = [args.compiler, args.emulator, args.library, args.settings, args.dram_config,
               Path(__file__), ROOT / "test/Decode/pim_activation_cores.py",
               ROOT / "lib/Decode/Target/NPU/NpuCodegen.cpp", ROOT / "lib/Decode/Target/NPU/CodegenPim.cpp",
               source, manifest, target, reference]
    model = Path(json.loads(manifest.read_text())["model"])
    tracked += sorted(model.glob("*.safetensors")) if model.is_dir() else [model]
    before = {str(p.resolve()): sha(p) for p in tracked}
    build = compile_case(args.compiler, out, "native_small", source, manifest, target, 4, image=True)
    audit = check_plan(build, out / "after_dual1")
    assert audit["plan"]["ranks"] == [0, 1]
    io = json.loads((build / "io.json").read_text())
    plan = json_attributes((build / "compiled.mlir").read_text(), "hyb.pim_plan")[0]
    assert plan["channels"] == 8 and plan["partitions"] == 16
    assert len(plan["projections"]) == 1
    # The driver uses ftruncate+mmap and writes actual regions, so rank1 does
    # not require an 8GiB dense host buffer or a physically allocated hole.
    st = (build / "lp6.bin").stat()
    assert st.st_size > 1 << 33 and st.st_blocks*512 < 512 << 20
    ref = np.load(reference)
    from pim_activation_cores import phase_program
    decoded, _ = phase_program(build, "unused")
    m, k = ref["x"].shape; n = ref["y"].shape[1]
    os.environ.pop("LP6PIM_BANKSET_PINGPONG", None)
    os.environ.pop("LP6PIM_INLINE_SCALE_GROUP", None)
    dev = Device(build, args.settings, args.dram_config, args.emulator, lp6_library=args.library)
    dev.write("x", ref["x"])
    # Every rank must be written by the generated program, including
    # zero-valued quantized inputs. No host Q/scale injection is used.
    with (build / "lp6.bin").open("r+b") as f:
        for i in audit["plan"]["activation_store_events"]:
            c = decoded["commands"][i]
            for row in range(c["rows"]):
                f.seek(c["lp6_address"] + row*c["lp6_stride"])
                f.write(bytes([0xA5])*c["row_bytes"])
    outputs, report, wall = dev.run()
    def get(dtype, shape):
        values = [outputs[n] for n, meta in dev.outputs.items() if meta["dtype"] == dtype and meta["shape"] == shape]
        assert len(values) == 1
        return values[0]
    q, scale, acc = get("i8", [m, k]), get("f32", [m, 1]), get("i32", [m, n])
    actual = dict(qx=q, sx=scale.reshape(ref["sx"].shape), acc=acc, y=outputs["y"])
    assert all(actual[key].tobytes() == ref[key].tobytes() for key in actual)
    # Debug collection is excluded; all earlier records must have completed.
    system = json.loads((build / "system.json").read_text())
    symbols = {r["id"]: r["name"] for r in system["command_symbols"]}
    precollect = [r for r in report["commands"] if not symbols[r["event_id"]].startswith("collect.")]
    endpoint = max(r["end_cycle"] for r in precollect)
    collected = [r for r in report["commands"] if symbols[r["event_id"]].startswith("collect.")]
    assert collected and min(r["issue_cycle"] for r in collected) >= endpoint
    timing = {r["event_id"]: r for r in report["commands"]}
    for c in decoded["commands"]:
        assert all(timing[c["id"]]["issue_cycle"] >= timing[d]["end_cycle"] for d in c["dependencies"])
    after = {str(p.resolve()): sha(p) for p in tracked}
    assert before == after
    result = dict(M=m, N=n, K=k, all_Qx_Sx_INT32_FP16_byte_exact=True, endpoint_cycles=endpoint, endpoint_us=endpoint/800,
                  wall_seconds=wall, program_sha256=sha(build / "program.bin"), feed_audit=audit, projection=plan["projections"][0])
    save(out / "native_small_result.json", result)
    print(json.dumps({k: result[k] for k in ["M", "N", "K", "endpoint_us", "all_Qx_Sx_INT32_FP16_byte_exact"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=Path, default=ROOT / "configs/target.json")
    p.add_argument("--emulator", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--settings", type=Path, default=workspace.SETTINGS)
    p.add_argument("--dram-config", type=Path, default=workspace.LP6_CONFIG)
    a = p.parse_args()
    prepare(a)
    check(a)
    native(a)


if __name__ == "__main__":
    main()
