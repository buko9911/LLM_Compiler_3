"""Host side of a Qwen3 decode step: inputs, simulator runs, outputs.

prepare() decodes the last token of a prompt: it builds the KV windows by
running the PyTorch module over the earlier prompt tokens one step at a time
(FP32), writes the checked step's inputs as arg<N>.bin beside the captured
weights, and returns the FP32 reference outputs of that step.

run_npu() runs an npu-compile package, run_hybrid() an graph-hybrid package (one
program whose NPU and PIM phases follow each other). Both return the outputs
read back from the simulator's L2 dump and the cycle counts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qwen3 import Checkpoint, Config, DecodeStep, load, rope_tables, step_mask, w8a8

LP6 = Path("/home/jongjip/LP6")
EMULATOR = LP6 / "IESLAB_NPU_Simulator/transactional_emulator/target/release/transactional_emulator"
SETTINGS = LP6 / "IESLAB_NPU_Simulator/plena_settings.toml"
BACKEND = LP6 / "lp6-pim-main"
# LP6 configuration per subchannel count of a package's PIM layout (npu-compile --pipeline=graph-hybrid pim.channels):
# four LPDDR6 channels (the simulator's default) or one. The NPU-only run of a comparison uses
# the hybrid package's so both sides share one memory system.
LP6_CONFIGS = {8: BACKEND / "configs/LPDDR6_PIM_Mobile_4CH.ini", 2: BACKEND / "configs/LPDDR6_PIM_Mobile_2CH.ini"}
NPU_HZ = 800e6
DTYPES = {"float16": "<f2", "float32": "<f4", "int8": "i1", "int32": "<i4"}


@dataclass
class Step:
    manifest: dict
    inputs: Path                      # arg<N>.bin of the step (weights linked from the capture)
    expected: list[np.ndarray]        # FP32 reference, one per result
    names: list[str] = field(default_factory=list)


def prepare(capture: Path, out: Path, prompt: str, quantize: bool) -> Step:
    manifest = json.loads((capture / "manifest.json").read_text())
    c = Config(**manifest["config"])
    L, M, P = manifest["layers"], manifest["tokens"], manifest["capacity"]
    model = Path(manifest["model"])
    from transformers import AutoTokenizer
    tokens = AutoTokenizer.from_pretrained(model)(prompt)["input_ids"]
    assert M <= len(tokens) <= P + M, "prompt shorter than one step or longer than the cache"
    checkpoint = Checkpoint(model)
    embed = checkpoint.get("model.embed_tokens.weight")
    reference = DecodeStep(c, L, manifest["head"]).float()
    load(reference, checkpoint, manifest["first_layer"])
    if quantize:
        w8a8(reference)

    # The module takes M and P from its inputs, so the prefix goes through it one
    # token at a time in the same window (P + M - 1 cached slots, 1 new).
    def run(x, past, windows):
        steps = x.shape[0]
        cos, sin = rope_tables(c, list(range(past, past + steps)))
        mask = step_mask(steps, P + M - steps, past)
        with torch.no_grad():
            return reference(x, cos, sin, mask, *windows), cos, sin, mask

    windows = [torch.zeros(c.kv_heads, P + M, c.head_dim) for _ in range(2 * L)]
    past = len(tokens) - M
    for t, token in enumerate(tokens[:past]):
        outputs, *_ = run(embed[token][None].float(), t, windows)
        for i in range(2 * L):                  # the host moves the new row into slot t
            windows[i][:, t] = outputs[1 + i][:, 0]
    x = embed[tokens[past:]].to(torch.float16)
    outputs, cos, sin, mask = run(x.float(), past, [w.clone() for w in windows])
    step_inputs = [x, cos.half(), sin.half(), mask.half()] + [w.half() for w in windows]

    source = out / "inputs"
    source.mkdir(parents=True, exist_ok=True)
    for path in list(capture.glob("arg*.bin")) + [capture / "arguments.txt"]:
        target = source / path.name
        if not target.exists():
            target.symlink_to(path.resolve())
    for index, tensor in enumerate(step_inputs):
        target = source / f"arg{index}.bin"
        target.unlink(missing_ok=True)
        target.write_bytes(tensor.contiguous().numpy().tobytes())
    expected = [o.float().numpy() for o in outputs]
    np.savez(out / "expected.npz", *expected)
    return Step(manifest, source, expected, [r["name"] for r in manifest["results"]])


def reuse(capture: Path, previous: Path) -> Step:
    """The step prepare() wrote into `previous`: its inputs and reference outputs."""
    manifest = json.loads((capture / "manifest.json").read_text())
    saved = np.load(previous / "expected.npz")
    expected = [saved[f"arr_{i}"] for i in range(len(saved.files))]
    return Step(manifest, previous / "inputs", expected, [r["name"] for r in manifest["results"]])


def simulator_settings(hardware: Path, out: Path, settings: Path = SETTINGS) -> Path:
    """The simulator settings with the core count the compile used."""
    cores = re.search(r"(?m)^num_cores\s*=\s*(\d+)", hardware.read_text())
    path = out / "settings.toml"
    path.write_text(re.sub(r"(?m)^num_cores\s*=\s*\d+", f"num_cores = {cores.group(1) if cores else 1}",
                           settings.read_text()))
    return path


def subchannels_of(package: Path) -> int:
    """LP6 subchannels a package's PIM layout targets; 8 (the default LP6) without PIM."""
    kernel = json.loads((package / "system.json").read_text()).get("pim_kernel")
    return int(kernel.get("subchannels", 2)) if kernel else 8


def _simulate(package: Path, args: list[str], timeout: int, subchannels: int) -> str:
    env = dict(os.environ, LP6_DRAMSIM3_ROOT=str(BACKEND))
    done = subprocess.run([str(EMULATOR), "--lp6-config", str(LP6_CONFIGS[subchannels])] + args, cwd=package,
                          capture_output=True, text=True, timeout=timeout, env=env)
    log = re.sub(r"\x1b\[[0-9;]*m", "", done.stdout + done.stderr)
    (package / "simulator.log").write_text(log)
    if done.returncode:
        raise RuntimeError(log[-6000:])
    return log


def _read_l2(package: Path, outputs: list[dict]) -> list[np.ndarray]:
    raw = (package / "l2_sram_dump.bin").read_bytes()
    result = []
    for o in outputs:
        dtype = np.dtype(DTYPES[o["dtype"]])
        at = o["readback_l2_address"]
        count = o["bytes"] // dtype.itemsize
        assert len(raw) >= at + o["bytes"], f"L2 dump ends at {len(raw)}, output needs {at + o['bytes']}"
        result.append(np.frombuffer(raw[at:at + o["bytes"]], dtype=dtype, count=count).copy())
    return result


def run_npu(package: Path, settings: Path, timeout: int = 14400, subchannels: int = 8):
    """One npu-compile package (compiled with --readback): outputs and total cycles."""
    log = _simulate(package, ["--system-program", "system.json", "--lp6-image", "hbm.bin",
                              "--settings", str(settings), "--log-level", "info"], timeout, subchannels)
    meta = json.loads((package / "metadata.json").read_text())
    outputs = [dict(o, bytes=o["rows"] * o["cols"] * (4 if o["dtype"] in ("float32", "int32") else
                                                      1 if o["dtype"] == "int8" else 2),
                    readback_l2_address=o["readback_l2_address"]) for o in meta["outputs"]]
    cycles = re.search(r"total_cycles=(\d+)", log)
    return _read_l2(package, outputs), int(cycles.group(1)) if cycles else None


def run_hybrid(package: Path, settings: Path, timeout: int = 14400, subchannels: int | None = None):
    """One graph-hybrid package: outputs and the per-phase report (phase_report)."""
    _simulate(package, ["--settings", str(settings), "--system-program", "system.json",
                        "--command-report", "commands.json", "--lp6-image", "lp6.bin",
                        "--lp6-size", "16GiB", "--log-level", "info"], timeout,
              subchannels or subchannels_of(package))
    meta = json.loads((package / "metadata.json").read_text())
    report = phase_report(json.loads((package / "commands.json").read_text()))
    (package / "phases.json").write_text(json.dumps(report, indent=1) + "\n")
    return _read_l2(package, meta["outputs"]), report


def phase_report(commands: dict) -> dict:
    """Phases of one npu-compile --pipeline=graph-hybrid program from the simulator's command report.

    Records are named <phase>.<kind>_<ordinal> (npu<i>), a PIM record after its
    phase (pim<g>). A phase runs from the end of the previous one to the end of
    its last record."""
    phases = []
    for c in sorted(commands["commands"], key=lambda c: c["event_id"]):
        name = c["name"].split(".")[0]
        if not phases or phases[-1]["name"] != name:
            start = phases[-1]["end_npu_cycle"] if phases else 0
            phases.append({"kind": "npu", "name": name, "start_npu_cycle": start, "end_npu_cycle": start})
        phase = phases[-1]
        phase["end_npu_cycle"] = max(phase["end_npu_cycle"], c["end_cycle"])
        if c["kind"] == "pim":
            phase["kind"] = "pim"
    return {"phases": phases}


def phase_summary(report: dict) -> dict:
    """NPU cycles per kind of phase, and the total, from a phase report."""
    total = {"npu": 0, "pim": 0}
    for phase in report["phases"]:
        total[phase["kind"]] += phase["end_npu_cycle"] - phase["start_npu_cycle"]
    total["all"] = total["npu"] + total["pim"]
    return total


def compare(name: str, got: np.ndarray, want: np.ndarray) -> float:
    got = got.astype(np.float32).reshape(-1)
    want = np.asarray(want, dtype=np.float32).reshape(-1)
    error = float(np.abs(got - want).max())
    peak = float(np.abs(want).max())
    relative = error / peak if peak else error
    print(f"  {name:10s} max|err| {error:.3e}  max|ref| {peak:.3e}  rel {relative:.3e}"
          f"{'' if np.isfinite(got).all() else '  NON-FINITE'}")
    return relative
