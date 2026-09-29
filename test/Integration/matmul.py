"""Independent PyTorch oracle and real Unified Program v5 simulator execution.

Shapes come from the package manifest, not from this file, so one checker serves
every entry point and size. A 32x32 case has no tile loop at all; larger cases do,
and several MLIR interfaces are only consulted once a loop nest exists.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess

import numpy as np

import harness
import torch


def grid(count, seed):
    """|x| <= 1/2 on the 1/16 grid: products land on 1/256 and FP32 sums stay exact."""
    return ((torch.arange(count) * seed % 17 - 8).float() / 16).half()


def oracle(rows, inner, columns):
    a = grid(rows * inner, 7).reshape(rows, inner)
    b = grid(inner * columns, 3).reshape(inner, columns)
    return a, b, (a.float() @ b.float()).half().numpy().astype("<f2")


# The hand-written target-stage source declares its own addresses and emits no
# manifest, so the test states that layout once instead of guessing it.
TARGET_LAYOUT = {
    "arguments": [{"index": 0, "address": 0, "rows": 32, "cols": 32},
                  {"index": 1, "address": 2048, "rows": 32, "cols": 32}],
    # matmul32.mlir ends with a GDMA that copies C back into L2 at 6144.
    "outputs": [{"address": 4096, "rows": 32, "cols": 32, "readback_l2_address": 6144}],
    "dram_bytes": 6144,
}


def run(package, emulator, settings, fallback=None):
    """Execute one package and return its shape, output tensor and the oracle's."""
    manifest_path = package / "metadata.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else fallback
    assert manifest is not None, f"{package.name}: no package manifest and no declared layout"
    arguments = sorted(manifest["arguments"], key=lambda d: d["index"])
    assert len(arguments) == 2 and len(manifest["outputs"]) == 1, "expected one matmul ABI"
    result = manifest["outputs"][0]
    rows, inner, columns = arguments[0]["rows"], arguments[0]["cols"], arguments[1]["cols"]
    assert arguments[1]["rows"] == inner and result["rows"] == rows and result["cols"] == columns, \
        f"{package.name}: manifest shapes are not one matmul"
    a, b, expected = oracle(rows, inner, columns)

    image = pack(manifest, arguments, (a, b))
    (package / "hbm.bin").write_bytes(bytes(image))
    harness.run(package, emulator, settings)
    actual = harness.read(package, result["readback_l2_address"], rows, columns,
                          row_stride=result.get("readback_row_stride_bytes"))
    return (rows, inner, columns), actual, expected



def pack(manifest, arguments, tensors):
    """Build the DRAM image an entry point asks for.

    A weight the compiler folded a transpose into is written transposed; the
    package names those arguments, because nothing in the declared shape alone
    says which orientation the bytes are in.
    """
    flipped = set(manifest.get("pretransposed", []))
    image = bytearray(manifest["dram_bytes"])
    for descriptor, tensor in zip(arguments, tensors):
        if descriptor["index"] in flipped:
            tensor = tensor.T.contiguous()
        # A weight the graph reads panel by panel is stored that way, so one
        # panel is one transfer instead of one short row per K.
        if panel := descriptor.get("packed"):
            rows, columns = descriptor["rows"], descriptor["cols"]
            tensor = tensor.reshape(rows, columns // panel, panel)
            tensor = tensor.permute(1, 0, 2).contiguous().reshape(rows, columns)
        raw = tensor.numpy().astype("<f2").tobytes()
        assert len(raw) == descriptor["rows"] * descriptor["cols"] * 2, \
            f"argument {descriptor['index']} is not the declared shape"
        image[descriptor["address"]:descriptor["address"] + len(raw)] = raw
    return bytes(image)


def main():
    parser = argparse.ArgumentParser()
    for name in ("compiler", "emulator", "settings", "source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True, nargs="+")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    emulator, settings = args.emulator.resolve(), args.settings.resolve()

    # The target path enters at Stage 4 with hand-written IR; the graph path runs
    # Stage 1..4. Each is also re-read from its saved stage files.
    jobs = [("memory", args.source, ["--from=target"], TARGET_LAYOUT),
            ("reparse", args.source, ["--from=target", "--reparse-stages"], TARGET_LAYOUT)]
    for source in args.graph:
        jobs.append((f"graph-{source.stem}", source, ["--from=graph", "--fp16", "--readback"], None))
        jobs.append((f"graph-{source.stem}-reparse", source,
                     ["--from=graph", "--fp16", "--readback", "--reparse-stages"], None))
    packages = []
    for mode, source, flags, layout in jobs:
        package = out / mode
        subprocess.run([str(args.compiler), *flags, str(source),
                        "--settings", str(args.settings), "-o", str(package)], check=True)
        packages.append((package, layout))

    # Re-reading a saved stage must not change the program. Entry points select
    # different instructions, so only same-source pairs are compared byte for byte.
    for (left, _), (right, _) in zip(packages[0::2], packages[1::2]):
        for filename in ("program.bin", "system.json"):
            assert (left / filename).read_bytes() == (right / filename).read_bytes(), \
                f"{filename} differs after reparse: {left.name} vs {right.name}"

    # Four cores. The array-tile loop hands one tile at a time to the next core,
    # so the same package has to come back with the same bytes: only the block
    # boundaries and the core each block runs on have changed.
    four = out / "hardware4.toml"
    four.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", "num_cores = 4",
                           args.settings.read_text()))
    for source in args.graph:
        package = out / f"cores4-{source.stem}"
        subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(source),
                        "--settings", str(four), "-o", str(package)], check=True)
        manifest = json.loads((package / "system.json").read_text())
        assert manifest["logical_to_physical"] == [0, 1, 2, 3], \
            f"{package.name}: package does not claim four cores"
        packages.append((package, None))

    checked = []
    for package, layout in packages:
        shape, actual, expected = run(package, emulator,
                                      four.resolve() if package.name.startswith("cores4") else settings,
                                      layout)
        np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                      err_msg=f"{package.name} {shape} is not bit-exact")
        checked.append({"package": package.name, "m": shape[0], "k": shape[1], "n": shape[2],
                        "elements_bit_exact": shape[0] * shape[2]})
    # Exercise async DMA with repeated L1 reuse and partial-K accumulation.
    # 6 KiB fits exactly two 32x32 inputs and one 32x32 output.
    tight = out / "hardware-tight.toml"
    tight.write_text(re.sub(r"(?m)^l1_sram_size_bytes\s*=\s*\d+",
                            "l1_sram_size_bytes = 6144", four.read_text()))
    for source in args.graph:
        package = out / f"tight-{source.stem}"
        subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback", str(source),
                        "--settings", str(tight), "--reparse-stages", "-o", str(package)], check=True)
        shape, actual, expected = run(package, emulator, tight.resolve())
        np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"),
                                      err_msg=f"{package.name}: async DMA with K chunks")
        checked.append({"package": package.name, "m": shape[0], "k": shape[1], "n": shape[2],
                        "elements_bit_exact": shape[0] * shape[2]})
    pipeline_settings = out / "hardware-pipeline.toml"
    pipeline_settings.write_text(four.read_text().replace("size_bytes = 8388608", "size_bytes = 524288"))
    package = out / "pipeline"
    subprocess.run([str(args.compiler), "--from=graph", "--fp16", "--readback",
                    str(Path(__file__).with_name("pipeline-graph.mlir")),
                    "--settings", str(pipeline_settings), "--reparse-stages", "-o", str(package)], check=True)
    stats = json.loads((package / "metadata.json").read_text())["local_optimizations"]
    assert stats["activation_cache_hits"] > 0, stats
    assert stats["weight_prefetches"] > 0, stats
    shape, actual, expected = run(package, emulator, pipeline_settings.resolve())
    np.testing.assert_array_equal(actual.view("<u2"), expected.view("<u2"))
    checked.append({"package": package.name, "m": shape[0], "k": shape[1], "n": shape[2],
                    "elements_bit_exact": shape[0] * shape[2], "optimizations": stats})
    # Same placement, no cache/prefetch: a numerical control and stage-restart check.
    plain = out / "pipeline-plain.mlir"
    plain.write_text((package / "03-placed.mlir").read_text()
                     .replace("npu.cache_activation", "test.disabled_cache")
                     .replace("npu.prefetch_weight", "test.disabled_prefetch"))
    control = out / "pipeline-control"
    subprocess.run([str(args.compiler), "--from=placed", str(plain), "--settings",
                    str(pipeline_settings), "-o", str(control)], check=True)
    _, baseline, _ = run(control, emulator, pipeline_settings.resolve())
    np.testing.assert_array_equal(actual.view("<u2"), baseline.view("<u2"))
    (out / "results.json").write_text(json.dumps(
        {"oracle": "PyTorch float32 matmul -> float16", "checked": checked,
         "in_memory_reparse_identical": True}, indent=2))
    print(f"{len(checked)} packages bit-exact against PyTorch: " +
          ", ".join(f"{c['package']}({c['m']}x{c['n']}x{c['k']})" for c in checked))


if __name__ == "__main__":
    main()
