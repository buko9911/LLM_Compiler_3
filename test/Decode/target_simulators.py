#!/usr/bin/env python3
"""configs/target.json, the NPU simulator settings and the LP6 INI must describe the same hardware."""
import configparser
import json
from pathlib import Path
import tomllib

COMPILER = Path(__file__).resolve().parents[2]
CORE = COMPILER.parent
target = json.loads((COMPILER / "configs/target.json").read_text())
npu, pim, lp6 = target["npu"], target["pim"], target["lp6"]
toml = tomllib.loads((CORE / "simulator/npu/plena_settings.toml").read_text())["TRANSACTIONAL"]
ini = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
ini.read(CORE / "ramulator2/configs/lp6" / toml["ARCHITECTURE"]["lp6_config"])

arch = toml["ARCHITECTURE"]
checks = {
    "npu.clock_mhz": (npu["clock_mhz"], arch["npu_clock_mhz"]),
    "npu.sa_rows": (npu["sa_rows"], arch["array_rows"]),
    "npu.sa_cols": (npu["sa_cols"], arch["array_columns"]),
    "npu.cores": (npu["cores"], toml["SYSTEM"]["num_cores"]),
    "npu.l1_bytes": (npu["l1_bytes"], arch["l1_sram_size_bytes"]),
    "npu.l2_bytes": (npu["l2_bytes"], toml["L2"]["size_bytes"]),
    "npu.command_fetch_words_per_cycle": (npu["command_fetch_words_per_cycle"], toml["COMMAND_PROCESSOR"]["fetch_words_per_cycle"]),
    "lp6.capacity_bytes": (lp6["capacity_bytes"], arch["lp6_capacity_bytes"]),
    "pim.channels": (pim["channels"], ini.getint("system", "channels")),
    "pim.ranks": (pim["ranks"], ini.getint("system", "ranks")),
    "pim.pus": (pim["pus"], ini.getint("pim", "pim_units_per_subchannel")),
    "pim.lanes": (pim["lanes"], ini.getint("pim", "pim_lanes")),
    "pim.nlld_ck": (pim["nlld_ck"], ini.getint("pim_timing", "nLLD")),
    "pim.clock_mhz": (round(pim["clock_mhz"], 1), round(1000 / ini.getfloat("timing", "tCK"), 1)),
}
wrong = {k: v for k, v in checks.items() if v[0] != v[1]}
assert not wrong, wrong
print("target, simulator settings and INI agree:", ", ".join(checks))
