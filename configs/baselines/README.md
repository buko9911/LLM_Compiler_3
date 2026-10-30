# Validated fixed-shape NPU-only presets

These presets were selected on Qwen3-4B layer0 + final norm + LM head,
W8A8/FP16, capacity512, valid past32, 4 cores at800MHz, L1 4MiB/core,
L2 8MiB, with the existing LPDDR6/Ramulator native simulator.

Use `--force=npu`. M8 uses `--npu-phase-split=layer`; M1 uses
`--npu-phase-split=none`. These are not universal defaults for other models,
precision contracts or context lengths. Actual valid-length attention and
full-model decoding are outside this validation scope.

See [current results, exactness checks, candidate regressions and reproduction](../../../../research/npu_command_schedule_audit/README.md)
and [pass architecture](../../../../docs/architecture/NPU_LAYER_SCHEDULING.md).
The presets minimize the complete measured fixture and keep transfer motion
disabled: moving a safe copy earlier did not improve these executions.
Both enable generic staging release; M1 additionally enables range-checked
matrix/output-DMA overlap. Hardware capacities and arithmetic are unchanged.
Both enable two-chunk VPU scheduling. M8 also reuses row-invariant scalar
reciprocals and distributes independent elementwise work with a minimum of
65536 elements; dequant's existing 16384-element threshold is pinned separately.
M1 leaves scalar reuse and elementwise distribution disabled: their measured
combinations did not beat chunk scheduling alone. The presets were recompiled
and checked against the measured programs, system descriptions, lowered IR,
and operation maps in [preset identity checks](../../../../research/npu_command_schedule_audit/validation/preset_identity.json).
M8 now enables `command_schedule`: precise strided GDMA effects, dependency-safe
system-command ordering, and proven transitive-edge reduction. It reduces this
fixture from 6257.43625 to 6239.01 microseconds. M1 retains its previous preset
because the measured alternatives were slower. Future-prefetch admission gaps
remain; this selection does not claim a stall-free or globally optimal schedule.
The common first-MMA weight prefetch also applies to larger eligible tiles;
those measured larger-tile configurations did not win the full fixture.
The [previous VPU sweep](../../../../research/npu_vpu_schedule_audit/README.md)
is preserved with its original programs and results.
The [previous dependency sweep](../../../../research/npu_dependency_schedule_audit/README.md)
and its selected configurations remain available for comparison.
The [earlier layout/tile/slot sweep](../../../../research/npu_layer_baseline_audit/README.md)
remains available as an immutable comparison.
