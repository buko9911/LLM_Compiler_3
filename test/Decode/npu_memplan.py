#!/usr/bin/env python3
"""Check L2 live-buffer separation and exhaustion under automatic staging."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", type=Path, default=root / "build/bin/npu-opt")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.output = Path(tempfile.mkdtemp(prefix="run-", dir=a.output))
    results = {}
    cases = [
        ("two_live_buffers", 512, 4, 0, None),
        ("one_core", 512, 1, 0, None),
        ("no_shared_buffers", 0, 4, 0, None),
        ("exhausted_auto", 2048, 4, 0, "no room left"),
        ("fixed_overflow", 512, 4, 1024, "L2 overflow"),
        ("invalid_window", 512, 4, 2048, "invalid NPU staging policy"),
    ]
    for name, size, cores, staging, error in cases:
        spec = json.loads((root / "configs/target.json").read_text())
        spec["npu"].update(cores=cores, l2_bytes=4096, l2_staging_bytes_per_core=staging)
        path = a.output / (name + ".json")
        path.write_text(json.dumps(spec, indent=2) + "\n")
        body = ""
        if size:
            body = f"""%a = memref.alloc() : memref<{size}xi8, 2>
    %b = memref.alloc() : memref<{size}xi8, 2>
    memref.copy %a, %b : memref<{size}xi8, 2> to memref<{size}xi8, 2>
"""
        source = a.output / (name + ".mlir")
        source.write_text('module {\n  func.func @phase() attributes {hyb.phase = "npu"} {\n    ' +
                          body + '    return\n  }\n}\n')
        p = subprocess.run([str(a.optimizer.resolve()), str(source),
                            f"--hyb-npu-memplan=spec={path.resolve()}"], capture_output=True, text=True)
        (a.output / (name + ".log")).write_text(p.stdout + p.stderr)
        if error:
            assert p.returncode != 0 and error in p.stderr, (name, p.stderr)
            results[name] = {"rejected": True, "diagnostic": error}
            continue
        assert p.returncode == 0, (name, p.stderr)
        window = int(re.search(r"hyb.l2_staging_per_core = (\d+)", p.stdout)[1])
        assert window * cores + 2 * size == 4096
        offsets = [int(v) for v in re.findall(r"hyb.offset = (\d+)", p.stdout)]
        assert len(offsets) == (2 if size else 0)
        if size:
            assert all(window * cores <= start and start + size <= 4096 for start in offsets)
            assert abs(offsets[0] - offsets[1]) >= size
        results[name] = {"window": window, "live_buffer_offsets": offsets,
                         "within_capacity_and_disjoint": True}
    (a.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
