# Qwen3-4B NPU-only presets

Target specs chosen by simulation for Qwen3-4B decode on the NPU alone. The fixture was
Qwen3-4B layer 0 + final norm + LM head, W8A8 / FP16, KV capacity 512 with 32 valid past
tokens, 4 NPU cores at 800 MHz, L1 4 MiB per core, L2 8 MiB, and the LPDDR6 Ramulator backend.

| File | Tokens per step | Compile with | Fixture time (simulated) |
|---|---|---|---|
| `qwen3_4b_m1_c512.json` | 1 | `--force=npu --npu-phase-split=none` | 5426.089 µs |
| `qwen3_4b_m8_c512.json` | 8 | `--force=npu --npu-phase-split=layer` | 6239.010 µs |

The times are for that one-layer fixture, not for a full decode step. The presets are not
general defaults for other models, precisions or context lengths.

For the full 36-layer model, the 1-token preset needs `--npu-phase-split=layer`. With `none`,
one phase exceeds the ISA 2.0 dependency record: command dependencies must lie within 65,535
earlier commands.

Both presets keep the hardware capacities and the arithmetic of the default target. They differ
only in which optional passes and backend schedules are on (see the `npu.*` table in the compiler
`README.md`). Candidate sweeps, exactness checks and reproduction scripts are in the workspace:

- [command scheduling audit](../../../../research/npu_command_schedule_audit/README.md)
  (current selection, [preset identity checks](../../../../research/npu_command_schedule_audit/validation/preset_identity.json))
- [VPU schedule sweep](../../../../research/npu_vpu_schedule_audit/README.md)
- [dependency schedule sweep](../../../../research/npu_dependency_schedule_audit/README.md)
- [layout / tile / slot sweep](../../../../research/npu_layer_baseline_audit/README.md)
