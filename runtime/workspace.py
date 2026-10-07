"""Locations of the simulators next to this compiler in the LP6 workspace.

The workspace keeps the three code repositories under core/:

    core/compiler            this repository (LLM_Compiler_3)
    core/simulator/npu       Rust NPU simulator (IESLAB_NPU_Simulator)
    core/simulator/dram-pim  C++ LPDDR6 DRAM/PIM backend (lp6-pim-simulator)

Measurement data that is no longer part of the code lives in archive/.
"""
from __future__ import annotations

from pathlib import Path

COMPILER = Path(__file__).resolve().parents[1]
CORE = COMPILER.parent
WORKSPACE = CORE.parent

NPU = CORE / "simulator/npu"
EMULATOR = NPU / "transactional_emulator/target/release/transactional_emulator"
SETTINGS = NPU / "plena_settings.toml"
NPU_TOOLS = NPU / "transactional_emulator/tools"

DRAM_PIM = CORE / "simulator/dram-pim"
LP6_CONFIG = DRAM_PIM / "configs/LPDDR6_PIM_Mobile_4CH.ini"      # four channels, the default
LP6_CONFIG_1CH = DRAM_PIM / "configs/LPDDR6_PIM_Mobile_2CH.ini"  # one channel

EVIDENCE = WORKSPACE / "archive/evidence"
