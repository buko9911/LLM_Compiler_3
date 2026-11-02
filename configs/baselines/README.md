# NPU-only target

`npu_only_lp6_4ch.json` is the NPU-only target for every Qwen3-4B step: 1-token decode, 8-token
verify and 64-token prefill. The step comes only from the export (`frontend/export.py --tokens M
--capacity P`, P = 0 for prefill); the file never names a step. Hardware: four LPDDR6 channels
(8 subchannels x 2 ranks), 4 NPU cores at 800 MHz, L1 4 MiB per core, L2 8 MiB, run with the 4-core
simulator settings and the LPDDR6 Ramulator backend. Compile with `--force=npu --npu-phase-split=layer`
(with `none`, one 36-layer phase exceeds the ISA 2.0 dependency record: command dependencies must lie
within 65,535 earlier commands).

The `npu.*` switches enable the optional passes. Whether a pass applies to an op is a cost comparison
of the tensor shapes against the hardware numbers in the file (`include/npu/Decode/Transforms/NpuCost.h`):
splitting an op over the cores, running SiLU*up tile by tile inside the up GEMV loop, fusing projections
that read the same input, cutting a GEMV partition that does not fit L1 into column chunks. Every choice
and its numbers are listed in the output `io.json` under `phases[].decisions`, so equal shapes get equal
programs whatever the step.

| Hardware number | Value | Source |
|---|---|---|
| `l1_read_bytes_per_cycle`, `l1_write_bytes_per_cycle` | 128, 128 | 32 banks x 4 B (simulator settings) |
| `mma_k_per_cycle`, `mma_pipeline_cycles` | 2, 2 | INT8 matrix timing (simulator settings) |
| `vector_lanes`, `vector_exp_per_cycle`, `vector_rcp_per_cycle`, `vector_sqrt_per_cycle` | 16, 4, 2, 2 | VPU lanes and shared SFUs (simulator settings) |
| `gdma_read_gbps` | 92.9 | a Qwen3-4B program's GDMA reads replayed without compute (2026-10-31) |
| `core_split_us` | 0.5 | fixed cost of running one op on 4 cores instead of 1, Qwen3-4B one-layer fixtures, every op split vs none at M = 1, 4, 8 (2026-10-31) |

The `*_min_elements` keys are floors kept for experiments (0: none). Measurements, the comparison with
the earlier per-step presets and what remains: workspace `docs/research/NPU_ONLY_BASELINE.md`.

Until 2026-11-02 there were three per-step presets with fixed size thresholds
(`qwen3_4b_m1_c512.json`, `qwen3_4b_m8_c512.json`, `qwen3_4b_prefill64.json`, the last two differing
only in `pair_consumer_pipeline`); they are in git history (commit `a19bf01`). Candidate sweeps, exactness
checks and reproduction scripts of the earlier selections are in the workspace:

- [command scheduling audit](../../../../research/npu_command_schedule_audit/README.md)
  (selection, [preset identity checks](../../../../research/npu_command_schedule_audit/validation/preset_identity.json))
- [VPU schedule sweep](../../../../research/npu_vpu_schedule_audit/README.md)
- [dependency schedule sweep](../../../../research/npu_dependency_schedule_audit/README.md)
- [layout / tile / slot sweep](../../../../research/npu_layer_baseline_audit/README.md)
