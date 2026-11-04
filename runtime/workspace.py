"""Locations of the simulators next to this compiler in the LP6 workspace.

The workspace keeps the three code repositories under core/:

    core/compiler            this repository (LLM_Compiler_3)
    core/simulator/npu       Rust NPU simulator (IESLAB_NPU_Simulator)
    core/ramulator2          LPDDR6-PIM DRAM/PIM backend: Ramulator2 + LP6 PIM, liblp6_ramulator.so

The DRAMSim3-based core/simulator/dram-pim backend was retired on 2026-11-04 (archive/dram-pim-dramsim3).

Measurement data that is no longer part of the code lives in archive/.
"""
from __future__ import annotations

from pathlib import Path

COMPILER = Path(__file__).resolve().parents[1]
# Worktrees may live in workspace/reviews rather than workspace/core.
WORKSPACE = next((p for p in COMPILER.parents if (p / "core/simulator/npu").is_dir()), None)
if WORKSPACE is None:
    raise RuntimeError("cannot locate the LP6 workspace containing core/simulator/npu")
CORE = WORKSPACE / "core"

NPU = CORE / "simulator/npu"
EMULATOR = NPU / "transactional_emulator/target/release/transactional_emulator"
SETTINGS = NPU / "plena_settings.toml"
NPU_TOOLS = NPU / "transactional_emulator/tools"

RAMULATOR = CORE / "ramulator2"
RAMULATOR_LIBRARY = RAMULATOR / "liblp6_ramulator.so"
LP6_CONFIGS = RAMULATOR / "configs/lp6"
LP6_CONFIG = LP6_CONFIGS / "LPDDR6_PIM_Mobile_4CH.ini"      # four channels, the default
LP6_CONFIG_1CH = LP6_CONFIGS / "LPDDR6_PIM_Mobile_2CH.ini"  # one channel

EVIDENCE = WORKSPACE / "archive/evidence"
