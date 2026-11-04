# NPU-only target

`npu_only_lp6_4ch.json` is the NPU-only target for every Qwen3-4B step: 1-token decode, 8-token
verify and 64-token prefill. The step comes only from the export (`frontend/export.py --tokens M
--capacity P`, P = 0 for prefill); the file never names a step. Hardware: four LPDDR6 channels
(8 subchannels x 2 ranks), 4 NPU cores at 800 MHz, L1 4 MiB per core, L2 8 MiB, run with the 4-core
simulator settings and the LPDDR6 Ramulator backend. Compile with `--force=npu` (NPU phases end at layer
boundaries: one 36-layer phase would exceed the ISA 2.0 dependency record, whose command dependencies
must lie within 65,535 earlier commands).

The file holds hardware numbers only; the optimizations have no switches (2026-11-02), and a key that
is not a target parameter is an error. Whether a rewrite applies to an op is a cost comparison of the
tensor shapes against these numbers (`include/npu/Decode/Transforms/NpuCost.h`): splitting an op over
the cores, full-row versus online softmax (does the full-row attention fit L2 and L1), running SiLU*up
tile by tile inside the up GEMV loop, fusing projections that read the same input, cutting a GEMV
partition that does not fit L1 into column chunks. Every choice and its numbers are listed in the output
`io.json` under `phases[].decisions`, so equal shapes get equal programs whatever the step.

| Hardware number | Value | Source |
|---|---|---|
| `l1_read_bytes_per_cycle`, `l1_write_bytes_per_cycle` | 128, 128 | 32 banks x 4 B (simulator settings) |
| `mma_k_per_cycle`, `mma_pipeline_cycles` | 2, 2 | INT8 matrix timing (simulator settings) |
| `vector_lanes`, `vector_exp_per_cycle`, `vector_rcp_per_cycle`, `vector_sqrt_per_cycle` | 16, 4, 2, 2 | VPU lanes and shared SFUs (simulator settings) |
| `gdma_read_gbps` | 92.9 | a Qwen3-4B program's GDMA reads replayed without compute (2026-10-31) |
| `core_split_us` | 0.5 | fixed cost of running one op on 4 cores instead of 1, Qwen3-4B one-layer fixtures, every op split vs none at M = 1, 4, 8 (2026-10-31) |

The same file is the NPU+PIM target (compile without `--force`; `hyb-partition` picks the device per GEMV
group). Its PIM stream numbers were fitted on the Ramulator LPDDR6-PIM model (2026-11-03, Qwen3-4B 2-layer
decode fixtures, every projection at A = 4..14 accumulators per stream plus mixed layouts; 35 PIM phases
within 1.4% except one at 6%):

`clock_mhz` 2666.667 is the DRAM CK of the model (tCK 0.375 ns). Until 2026-11-04 it was 2400 and the three
stream costs were in 1/2400 us; they were rescaled by 2666.667 / 2400. Every estimate of this file is unchanged: 23 builds (Qwen3-4B 2-layer
M = 1, 2, 3, 4, 8, 64 with this file, target.json and target_1ch.json; 36-layer M = 1, 3, 4, 8 and prefill 64 with this
file) gave byte-identical program.bin and the same NPU/PIM placement, PIM estimates within 0.0004%. target_1ch.json has
no fitted stream costs (LdMAC = nLLD 10 CK), so its PIM estimates are 10% lower than before; its placement and
program.bin did not change on those six fixtures.

| PIM number | Value | Meaning |
|---|---|---|
| `bankset_pingpong_lead` | 12 | weight rows alternate bank sets, the other set's row opens 12 LdMACs ahead (leads 12..48 measured within 0.1%; 0 is 14% slower per layer) |
| `ldmac_ck` | 11.6111 | DRAM CK per LdMAC in a long stream, 4.35 ns (nLLD 10 CK = 3.75 ns, so 86% of the nLLD rate) |
| `sweep_ck` | 871.111 | fixed DRAM CK per 256-input sweep of a stream (activation register loads, row changes) |
| `stream_ck` | 1130 | fixed DRAM CK per stream (clear and store of its accumulators) |
| `readback_ns_per_output` | 0.33 | NPU time per PIM output and token after the PIM phase beyond the gather bytes (dequant, gather setup): Qwen3-4B LM head 151,936 outputs in 69 us (hyb-partition only) |
| `npu_run_split_us` | 45 | hyb-partition: NPU time lost when a PIM group splits a layer's NPU weight stream (O or gate/up on PIM, the rest NPU): Qwen3-4B 2-layer, 4 and 8 tokens, measured 2026-11-03 (`npu_run_split_us_per_token` 0) |
| `concurrent_ranks` | false | two ranks of one subchannel computing together gave no gain on this model (2026-11-03) |

Measurements, the comparison with the earlier per-step presets and what remains: workspace
`docs/research/NPU_ONLY_BASELINE.md`. The contribution of each former switch (measured 2026-11-02, before
they were removed): workspace `docs/architecture/COMPILER.md` appendix A.

Until 2026-11-02 there were three per-step presets with fixed size thresholds
(`qwen3_4b_m1_c512.json`, `qwen3_4b_m8_c512.json`, `qwen3_4b_prefill64.json`, the last two differing
only in `pair_consumer_pipeline`); they are in git history (commit `a19bf01`). Candidate sweeps, exactness
checks and reproduction scripts of the earlier selections are in the workspace:

- [command scheduling audit](../../../../research/npu_command_schedule_audit/README.md)
  (selection, [preset identity checks](../../../../research/npu_command_schedule_audit/validation/preset_identity.json))
- [VPU schedule sweep](../../../../research/npu_vpu_schedule_audit/README.md)
- [dependency schedule sweep](../../../../research/npu_dependency_schedule_audit/README.md)
- [layout / tile / slot sweep](../../../../research/npu_layer_baseline_audit/README.md)
