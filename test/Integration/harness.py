"""Run a package on NPU_Simulator main and read one output tensor back.

NPU_Simulator main (ISA ver 1.0) dumps each core's L1 and the shared L2 but not
DRAM, so an output is read from ``l2_sram_dump.bin``:

* a compiled graph asks for this with ``plena-compile --readback``; the
  compiler appends GDMA loads that copy every output from DRAM into an L2
  window and records it as ``outputs[].readback_l2_address`` in metadata.json;
* hand-written target IR ends with the same kind of ``plena.dma`` load itself
  and the caller passes the L2 address it chose.
"""
import json
import subprocess
from pathlib import Path

import numpy as np


def run(package: Path, emulator: Path, settings: Path, timeout: int = 900,
        log_level: str = "warn") -> str:
    """Execute package/system.json with package/hbm.bin; return the simulator log.

    ``log_level="info"`` makes the simulator print its summary (total_cycles=...).
    """
    completed = subprocess.run([str(emulator), "--system-program", "system.json",
                                "--lp6-image", "hbm.bin", "--settings", str(settings),
                                "--log-level", log_level],
                               cwd=package, capture_output=True, text=True, timeout=timeout)
    log = completed.stdout + completed.stderr
    (package / "simulator.log").write_text(log)
    if completed.returncode:
        raise RuntimeError(log)
    return log


def readback_address(package: Path, dram_address: int) -> tuple[int, int | None]:
    """L2 window and row stride of the output the program stored at dram_address."""
    manifest = json.loads((package / "metadata.json").read_text())
    for output in manifest["outputs"]:
        if output["address"] == dram_address and "readback_l2_address" in output:
            return output["readback_l2_address"], output.get("readback_row_stride_bytes")
    raise KeyError(f"{package.name}: no readback window for DRAM {dram_address}; "
                   "compile with --readback")


def read(package: Path, l2_address: int, rows: int, columns: int, dtype: str = "<f2",
         row_stride: int | None = None) -> np.ndarray:
    """rows x columns elements from l2_sram_dump.bin starting at l2_address."""
    width = np.dtype(dtype).itemsize
    stride = row_stride or columns * width
    raw = (package / "l2_sram_dump.bin").read_bytes()
    end = l2_address + (rows - 1) * stride + columns * width
    assert len(raw) >= end, f"{package.name}: L2 dump ends at {len(raw)}, output needs {end}"
    out = np.empty((rows, columns), dtype=dtype)
    for r in range(rows):
        start = l2_address + r * stride
        out[r] = np.frombuffer(raw[start:start + columns * width], dtype=dtype)
    return out


def simulate(package: Path, emulator: Path, settings: Path, address: int, rows: int,
             columns: int, dtype: str = "<f2", l2_address: int | None = None) -> np.ndarray:
    """Run, then return the output stored at DRAM ``address``.

    ``l2_address`` is for hand-written target IR, which carries its own readback.
    """
    run(package, emulator, settings)
    stride = None
    if l2_address is None:
        l2_address, stride = readback_address(package, address)
    return read(package, l2_address, rows, columns, dtype, stride)
