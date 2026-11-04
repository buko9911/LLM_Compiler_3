# Target description

`target.json` is everything the compiler knows about the hardware: LPDDR6-PIM 4 channels = 8 subchannels x 2 ranks at
10667 Mbps, NPU 4 cores at 800 MHz. It is the one target for every step and for both NPU-only and NPU+PIM programs. The
step comes only from the export (`frontend/export.py --tokens M --capacity P`, P = 0 for prefill), NPU-only is
`--force=npu`, and NPU+PIM is no `--force` (`hyb-partition` picks the device per GEMV group).

The file names every key. A missing key or a key the compiler does not know is an error, so nothing is decided by a hidden
default and a removed switch cannot linger. There are no optimization switches: whether a rewrite applies to an op is a cost
comparison of the tensor shapes against these numbers (`include/npu/Decode/Transforms/NpuCost.h`), and every choice with its
numbers is listed in the output `io.json` under `phases[].decisions`.

Run with the matching simulators: `core/simulator/npu/plena_settings.toml` and
`core/ramulator2/configs/lp6/LPDDR6_PIM_Mobile_4CH.ini`. The test `decode-target-simulators` fails when this file and those
two disagree.

## NPU

| Key | Value | Kind | Meaning |
|---|---|---|---|
| `clock_mhz`, `cores`, `sa_rows`, `sa_cols` | 800, 4, 32, 32 | hardware | NPU clock, cores, systolic array of a core |
| `l1_bytes`, `l2_bytes` | 4 MiB, 8 MiB | hardware | L1 per core, shared L2 |
| `l1_read_bytes_per_cycle`, `l1_write_bytes_per_cycle` | 128, 128 | hardware | 32 banks x 4 B |
| `mma_k_per_cycle`, `mma_pipeline_cycles` | 2, 2 | hardware | INT8 matrix timing |
| `vector_lanes`, `vector_exp_per_cycle`, `vector_rcp_per_cycle`, `vector_sqrt_per_cycle` | 16, 4, 2, 2 | hardware | VPU lanes and shared SFUs |
| `completion_event_slots` | 65536 | hardware | Commands per program |
| `gdma_read_gbps` | 92.9 | measured | Sustained LP6 read of a Qwen3-4B program replayed without compute (2026-10-31) |
| `core_split_us` | 0.5 | measured | Fixed cost of running one op on 4 cores instead of 1, Qwen3-4B 1-layer, M = 1, 4, 8 (2026-10-31) |
| `gemv_fixed_us` | 2.0 | estimate | Fixed cost per NPU GEMV in `hyb-partition` |
| `l1_reserved_bytes`, `l2_staging_bytes_per_core` | 256 KiB, 0 | plan | L1 constant pool and scratch; 0 sizes the L2 staging window per phase |
| `gemv_tile_bytes`, `input_tile_columns`, `weight_stream_max_slots`, `gdma_lookahead`, `staging_misc_bytes` | 128 KiB, 128, 0, 2, 64 KiB | plan | Tile sizes and weight stream limits |

## PIM

| Key | Value | Kind | Meaning |
|---|---|---|---|
| `clock_mhz` | 2666.667 | hardware | DRAM CK (tCK 0.375 ns, Ramulator LPDDR6 10667). The `*_ck` numbers are in these cycles |
| `pus`, `lanes`, `accumulators`, `k_per_sweep` | 8, 4, 16, 256 | draft | PUs per subchannel and rank, INT32 lanes, accumulators, inputs per sweep. `npu/Target/PimLayout.h` is written for them; other values are refused |
| `channels`, `ranks`, `partitions` | 8, 2, 16 | hardware | Subchannels, ranks, partitions = subchannels x ranks |
| `concurrent_ranks` | true | setting | true: one PIM phase computes on both ranks. false: on one rank; the other serves normal reads meanwhile |
| `nlld_ck` | 10 | draft | Minimum LdMAC spacing, draft Table 51 at 9600-10667 Mbps |
| `bankset_pingpong_lead` | 12 | assumption | Weight rows alternate bank sets, the other set's row opens 12 LdMACs ahead (leads 12..48 measured within 0.1%; 0 is 14% slower per layer). The draft does not spell this operation out |
| `ldmac_ck`, `sweep_ck`, `stream_ck` | 11.6111, 871.111, 1130 | fitted | DRAM CK per LdMAC in a long stream, per 256-input sweep, per stream. Fit on Qwen3-4B 2-layer decode fixtures (`research/npu_pim_baseline/fit_pim.py`) |
| `readback_gbps`, `act_write_gbps` | 32, 96 | estimate | NPU gather of INT32 outputs and scatter of INT8 activations |
| `readback_ns_per_output` | 0.33 | measured | NPU time per PIM output and token beyond the gather bytes: LM head 151,936 outputs in 69 us |
| `npu_run_split_us` | 45 | measured | NPU time lost when a PIM group splits a layer's NPU weight stream, 2-layer fixture, 4 and 8 tokens |
| `stream_accs`, `max_split_k` | 2..16, 8 | search | Accumulator counts and split-K tried by the stream layout search |

## LP6

`capacity_bytes` is 16 GiB, the address window the compiler plans in. The Ramulator device behind it has 32 GiB and uses the
refresh timing of 32 Gb dies (tRFC 380 ns).

## Reproducing older results

The per-step presets before 2026-11-02 are at compiler commit `a19bf01`. The NPU-only baseline is tag
`baseline-qwen3-4b-final`, whose target was `configs/baselines/npu_only_lp6_4ch.json`.
