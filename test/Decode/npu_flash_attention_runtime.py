#!/usr/bin/env python3
"""Numerical and native simulator checks for decode FlashAttention/Flash-Decoding.

The independent reference materializes attention in float64. It is deliberately
not the online recurrence under test. Legacy FP16 storage is a second diagnostic,
not the correctness oracle. Simulator times describe tiny attention kernels,
never hardware measurements or complete language-model throughput.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from safetensors.numpy import save_file

from npu_flash_attention import flash_source


@dataclass(frozen=True)
class Case:
    name: str
    heads: int
    groups: int
    tokens: int
    keys: int
    depth: int
    mask: str = "causal"
    scale: float = 0.25
    key_tile: int = 16
    query_tile: int = 8
    score_scale: float = 1.0
    kv_prefix: int = 0
    key_sequence_major: bool = False


# Fixed tolerances are set before running native kernels. FP16 matrix inputs and
# probability operands permit rounding; online state must remain FP32. Report
# errors and argmax separately even when a value tolerance passes.
ATOL = 0.01
RTOL = 0.02
ONLINE_ATOL = 0.002
ONLINE_RTOL = 0.005
CASES = (
    Case("m1_causal", 4, 2, 1, 65, 16),
    Case("m8_gqa_causal_tail", 4, 2, 8, 69, 16, query_tile=3),
    Case("scaled_gqa_causal", 2, 2, 8, 41, 8, score_scale=0.3535533905932738),
    Case("m8_masked_first_block", 2, 2, 8, 49, 16, "late_valid"),
    Case("m3_odd_heads_keys", 3, 2, 3, 23, 8, query_tile=2),
    Case("all_masked_negative_infinity", 2, 2, 3, 35, 8, "negative_infinity"),
    Case("finite_masked_row", 2, 2, 3, 35, 8, "finite_row"),
    Case("extreme_logits", 2, 2, 3, 35, 8, "causal", 8.0),
    Case("prefill_q32", 2, 2, 32, 32, 16, query_tile=8),
    Case("prefill_q64_tail", 2, 2, 64, 67, 16, query_tile=16),
    Case("concat_transposed_key", 2, 2, 3, 67, 8, kv_prefix=64),
    Case("concat_sequence_major_key", 2, 2, 8, 72, 8, kv_prefix=64, key_sequence_major=True),
)
PERFORMANCE_CASES = (
    Case("long_decode_h1", 1, 4, 1, 4096, 32, key_tile=128, query_tile=32),
    Case("long_decode_h8", 8, 4, 1, 4096, 32, key_tile=128, query_tile=32),
)
ALL_CASES = CASES + PERFORMANCE_CASES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_softmax_product(score, values, case):
    maximum = score.max(axis=-1, keepdims=True)
    valid = np.isfinite(maximum)
    with np.errstate(invalid="ignore", over="raise"):
        exp = np.where(valid, np.exp(score - np.where(valid, maximum, 0)), 0)
    total = exp.sum(axis=-1, keepdims=True)
    probability = np.divide(exp, total, out=np.zeros_like(exp), where=total > 0)
    return probability.reshape(case.heads, case.groups * case.tokens, case.keys) @ values


def full_reference(inputs, case):
    """Conventional dense attention in FP64, with zero all-masked rows.

    Q already contains any desired model scale. A finite negative additive mask
    remains finite: it does not mean a structurally excluded key. Only -inf
    excludes a key. This distinction matters for an entirely finite-mask row.
    """
    q, k, v = (inputs[name].astype(np.float64) for name in ("q", "k", "v"))
    score = (q @ k).reshape(case.heads, case.groups, case.tokens, case.keys)
    score *= np.float32(case.score_scale)
    score += inputs["mask"].astype(np.float64)[None, None, :, :]
    return dense_softmax_product(score, v, case)


def rounded_score(inputs, case):
    """Preserve the old graph's QK and additive-mask FP16 storage boundaries."""
    q, k = (inputs[name].astype(np.float32) for name in ("q", "k"))
    score = (q @ k).astype(np.float16).reshape(case.heads, case.groups, case.tokens, case.keys)
    if case.score_scale != 1.0:
        score = (score.astype(np.float32) * np.float32(case.score_scale)).astype(np.float16)
    return (score.astype(np.float32) + inputs["mask"].astype(np.float32)[None, None]).astype(np.float16)


def score_contract_reference(inputs, case):
    """Independent dense oracle for the preserved rounded-logit contract.

    This does not use key tiles, recurrence, partial maxima or split merging.
    It isolates online correctness from pre-existing FP16 score rounding.
    """
    return dense_softmax_product(rounded_score(inputs, case).astype(np.float64),
                                 inputs["v"].astype(np.float64), case)


def online_contract_reference(inputs, case, splits=1):
    """FP32-state / FP16-P algorithm oracle, separate from the dense oracle.

    Matrix products execute in NumPy, not the NPU simulator. This routine only
    diagnoses expected mixed-precision error; its wall time is never reported
    as NPU performance. Partitions have floor(keys/splits) keys. Each includes
    its own short tile; fewer than `splits` leftover keys follow the merge.
    """
    score = rounded_score(inputs, case).astype(np.float32).reshape(
        case.heads, case.groups * case.tokens, case.keys)
    v = inputs["v"].astype(np.float32)
    shape = score.shape[:-1]

    def empty():
        return (np.full(shape, -np.inf, dtype=np.float32), np.zeros(shape, dtype=np.float32),
                np.zeros((*shape, case.depth), dtype=np.float32))

    def rescale(old, new):
        # Guard the initial -inf and a fully masked key chunk explicitly.
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(new), np.exp(old - new), 0).astype(np.float32)

    def update(state, begin, end):
        maximum, total, numerator = state
        block = score[..., begin:end]
        next_max = np.maximum(maximum, block.max(axis=-1))
        alpha = rescale(maximum, next_max)
        with np.errstate(invalid="ignore"):
            probability = np.where(np.isfinite(next_max)[..., None],
                                   np.exp(block - next_max[..., None]), 0).astype(np.float32)
        next_total = total * alpha + probability.sum(axis=-1, dtype=np.float32)
        contribution = probability.astype(np.float16).astype(np.float32) @ v[:, begin:end, :]
        next_numerator = numerator * alpha[..., None] + contribution
        return next_max, next_total.astype(np.float32), next_numerator.astype(np.float32)

    def combine(left, right):
        lm, ll, lo = left
        rm, rl, ro = right
        maximum = np.maximum(lm, rm)
        a, b = rescale(lm, maximum), rescale(rm, maximum)
        return maximum, ll * a + rl * b, lo * a[..., None] + ro * b[..., None]

    prefix = case.kv_prefix or case.keys
    uniform = prefix // splits
    state = empty()
    if uniform:
        for split in range(splits):
            partial = empty()
            for start in range(split * uniform, (split + 1) * uniform, case.key_tile):
                partial = update(partial, start, min(start + case.key_tile, (split + 1) * uniform))
            state = combine(state, partial)
    for start in range(uniform * splits, prefix, case.key_tile):
        state = update(state, start, min(start + case.key_tile, prefix))
    if case.kv_prefix:
        for start in range(prefix, case.keys, case.key_tile):
            state = update(state, start, min(start + case.key_tile, case.keys))
    _, total, numerator = state
    return np.divide(numerator, total[..., None], out=np.zeros_like(numerator),
                     where=total[..., None] > 0).astype(np.float16)


def legacy_reference(inputs, case):
    """Old graph's FP16 score/exp/probability boundaries, for diagnostics only."""
    q, k, v = (inputs[name].astype(np.float32) for name in ("q", "k", "v"))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        score = (q @ k).astype(np.float16).reshape(
            case.heads, case.groups, case.tokens, case.keys)
        if case.score_scale != 1.0:
            score = (score.astype(np.float32) * np.float32(case.score_scale)).astype(np.float16)
        score = (score.astype(np.float32) + inputs["mask"].astype(np.float32)[None, None]).astype(np.float16)
        exp = np.exp(score.astype(np.float32) - score.astype(np.float32).max(-1, keepdims=True)).astype(np.float16)
        probability = (exp.astype(np.float32) / exp.astype(np.float32).sum(-1, keepdims=True)).astype(np.float16)
        return (probability.reshape(case.heads, case.groups * case.tokens, case.keys).astype(np.float32) @ v).astype(np.float16)


def make_inputs(case):
    rng = np.random.default_rng(20261030)
    h, g, m, s, d = case.heads, case.groups, case.tokens, case.keys, case.depth
    inputs = {
        "q": (rng.normal(size=(h, g * m, d)) * case.scale).astype(np.float16),
        "k": (rng.normal(size=(h, d, s)) * case.scale).astype(np.float16),
        "v": (rng.normal(size=(h, s, d)) * 0.5).astype(np.float16),
    }
    mask = np.where(np.arange(s)[None, :] <= s - m + np.arange(m)[:, None], 0, -np.inf).astype(np.float16)
    if case.mask == "late_valid":
        # The first online block contributes no valid keys; later blocks do.
        mask[:, :case.key_tile] = -np.inf
    elif case.mask == "negative_infinity":
        mask[-1, :] = -np.inf
    elif case.mask == "finite_row":
        mask[np.isneginf(mask)] = -30000
        mask[-1, :] = -30000
    inputs["mask"] = mask
    if case.kv_prefix:
        inputs["kp"], inputs["kn"] = inputs["k"][..., :case.kv_prefix], inputs["k"][..., case.kv_prefix:]
        inputs["vp"], inputs["vn"] = inputs["v"][:, :case.kv_prefix], inputs["v"][:, case.kv_prefix:]
        if case.key_sequence_major:
            inputs["kp"], inputs["kn"] = inputs["kp"].transpose(0, 2, 1), inputs["kn"].transpose(0, 2, 1)
    return inputs


def errors(actual, expected, atol=ATOL, rtol=RTOL):
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(e)
    both_finite = bool(finite.all())
    absolute = np.abs(a[finite] - e[finite])
    maximum = float(absolute.max(initial=0))
    rmse = float(np.sqrt(np.mean(absolute * absolute))) if absolute.size else None
    return {
        "all_finite": both_finite,
        "actual_nonfinite_elements": int((~np.isfinite(a)).sum()),
        "max_abs": maximum if absolute.size else None,
        "rmse": rmse,
        "tolerance_pass": bool(np.allclose(a, e, atol=atol, rtol=rtol, equal_nan=False)),
        "tolerance": {"atol": atol, "rtol": rtol},
        "argmax_equal": both_finite and bool(np.array_equal(a.argmax(-1), e.argmax(-1))),
        "argmax_mismatch_rows": int(np.count_nonzero(a.argmax(-1) != e.argmax(-1))) if both_finite else None,
        "argmax_scope": "attention output feature index, not final language-model token",
    }


def make_spec(root, mode, case, splits):
    target = json.loads((root / "configs/target.json").read_text())
    target["npu"].update(
        flash_attention=mode != "off",
        flash_decoding=mode == "decoding",
        flash_kv_slicing=mode != "off" and bool(case.kv_prefix),
        attention_query_tile=case.query_tile,
        attention_key_tile=case.key_tile,
        attention_splits=splits if mode == "decoding" else 1,
        attention_fusion=True,
        activation_residency=False,
        overlap_dequant_gemv=False,
        l2_staging_bytes_per_core=0,
    )
    return target


def run(command, cwd, log, env=None):
    with log.open("w") as stream:
        status = subprocess.run(list(map(str, command)), cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    if status.returncode:
        raise RuntimeError(f"Command failed: {command}\n{log.read_text()[-7000:]}")


def reference_checks(output):
    """Test the independent oracle's mask and causal conventions, without timing."""
    results = {}
    for case in CASES:
        inputs = make_inputs(case)
        expected = full_reference(inputs, case)
        assert np.isfinite(expected).all(), case.name
        # Independent causal check: perturb future V, which must not affect an
        # earlier query. Use row 0 of each GQA group and retain all past keys.
        if case.tokens > 1 and case.mask not in ("finite_row", "negative_infinity"):
            changed = dict(inputs)
            changed["v"] = inputs["v"].copy()
            changed["v"][:, case.keys - case.tokens + 1:, :] += 3
            alternate = full_reference(changed, case)
            np.testing.assert_array_equal(expected[:, ::case.tokens, :], alternate[:, ::case.tokens, :])
        if case.mask == "negative_infinity":
            np.testing.assert_array_equal(expected[:, case.tokens - 1::case.tokens, :], 0)
        contracted = score_contract_reference(inputs, case)
        online = online_contract_reference(inputs, case)
        split = online_contract_reference(inputs, case, 2)
        assert errors(online, contracted)["tolerance_pass"], case.name
        assert errors(split, contracted)["tolerance_pass"], case.name
        results[case.name] = {
            "finite_reference": True,
            "legacy_vs_fp64": errors(legacy_reference(inputs, case), expected),
            "rounded_score_vs_fp64": errors(contracted, expected),
            "online_vs_dense_score_contract": errors(online, contracted),
            "split2_vs_dense_score_contract": errors(split, contracted),
            "split2_vs_online": errors(split, online),
        }
    (output / "reference_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=root / "build/bin/npu-compile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", default="", help="Comma-separated case names")
    parser.add_argument("--modes", default="off,attention,decoding")
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--emulator", type=Path, default=workspace / "core/simulator/npu/transactional_emulator/target/release/transactional_emulator")
    parser.add_argument("--settings", type=Path, default=workspace / "research/checkpoint_race/hardware/kernel_probe/settings_4core.toml")
    parser.add_argument("--lp6-library", type=Path, default=workspace / "core/ramulator2/liblp6_ramulator.so")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if not modes or any(mode not in ("off", "attention", "decoding") for mode in modes):
        parser.error("--modes accepts off,attention,decoding")
    selected = set(args.only.split(",")) if args.only else {case.name for case in CASES}
    unknown = selected - {case.name for case in ALL_CASES}
    if unknown:
        parser.error(f"Unknown cases: {sorted(unknown)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference_checks(output)
    if args.reference_only:
        print(f"Independent reference contracts passed: {output / 'reference_checks.json'}")
        return
    model = output / "model"
    model.mkdir()
    save_file({"unused": np.zeros((1,), dtype=np.float16)}, model / "model.safetensors")
    paths = [args.compiler.resolve(), args.emulator.resolve(), args.settings.resolve(),
             args.lp6_library.resolve(), args.lp6_library.resolve().parent / "libramulator.so",
             Path(__file__).resolve(), Path(__file__).with_name("npu_attention_fusion.py"),
             Path(__file__).with_name("npu_flash_attention.py")]
    provenance = {str(path): digest(path) for path in paths if path.exists()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {"kind": "compile_only" if args.compile_only else "native_cycle_simulator_attention_microtests",
               "off_baseline": "materialized_head_fused",
               "tolerance": {"dense_rounded_score": {"atol": ATOL, "rtol": RTOL},
                             "online_contract": {"atol": ONLINE_ATOL, "rtol": ONLINE_RTOL}}, "cases": {}}
    failures = []
    for case in ALL_CASES:
        if case.name not in selected:
            continue
        directory = output / case.name
        directory.mkdir()
        source = directory / "step.mlir"
        ir = flash_source(heads=case.heads, groups=case.groups, tokens=case.tokens,
                          keys=case.keys, depth=case.depth, with_mask=True, score_scale=case.score_scale,
                          kv_prefix=case.kv_prefix, key_sequence_major=case.key_sequence_major)
        ir = ir.replace("@phase(", "@decode_step(").replace('attributes {hyb.phase = "npu"}', "")
        source.write_text(ir)
        manifest = directory / "manifest.json"
        argument_names = ("q", "kp", "kn", "vp", "vn", "mask") if case.kv_prefix else ("q", "k", "v", "mask")
        manifest.write_text(json.dumps(dict(model=str(model), tokens=case.tokens, capacity=case.keys-case.tokens,
            mask_min=-30000., arguments=[dict(kind="input", name=name) for name in argument_names],
            results=[dict(kind="output", name="y")]), indent=2))
        inputs = make_inputs(case)
        np.savez(directory / "inputs.npz", **inputs)
        expected, legacy = full_reference(inputs, case), legacy_reference(inputs, case)
        contracted = score_contract_reference(inputs, case)
        np.save(directory / "reference_fp64.npy", expected)
        np.save(directory / "reference_legacy.npy", legacy)
        np.save(directory / "reference_rounded_score.npy", contracted)
        case_results = {"shape": case.__dict__, "modes": {}}
        summary["cases"][case.name] = case_results
        outputs = {}
        for mode in modes:
            build = directory / mode
            target = directory / f"{mode}.json"
            target.write_text(json.dumps(make_spec(root, mode, case, args.splits), indent=2))
            compile_command = [args.compiler.resolve(), "--pipeline=decode", source, "--manifest", manifest,
                               "--spec", target, "--force=npu", "-o", build]
            try:
                run(compile_command, workspace, directory / f"{mode}.compile.log")
                io = json.loads((build / "io.json").read_text())
                if mode != "off":
                    reports = [phase.get("flash_attention", {}) for phase in io["phases"]]
                    count = sum(report.get("transformed_chains", 0) for report in reports)
                    assert count == 1, (case.name, mode, "requested online kernel was not transformed", reports)
                    algorithm = "flash_decoding" if mode == "decoding" else "flash_attention"
                    assert any(report.get("algorithm") == algorithm for report in reports), reports
                    if case.kv_prefix:
                        assert sum(report.get("segmented_kv_chains", 0) for report in reports) == 1, ("KV slicing was not applied", reports)
                if args.compile_only:
                    case_results["modes"][mode] = {"compiled": True, "phase_metadata": io["phases"]}
                    continue
                with (build / "lp6.bin").open("r+b") as image:
                    for argument in io["args"]:
                        array = np.asarray(inputs[argument["name"]], dtype=np.float16).reshape(argument["shape"])
                        image.seek(argument["addr"])
                        image.write(array.tobytes())
                simulation = [args.emulator.resolve(), "--settings", args.settings.resolve(),
                    "--lp6-config", workspace / "core/simulator/dram-pim/configs/LPDDR6_PIM_Mobile_4CH.ini",
                    "--lp6-backend", "ramulator", "--lp6-library", args.lp6_library.resolve(),
                    "--system-program", "system.json", "--lp6-image", "lp6.bin", "--lp6-size", "16GiB",
                    "--command-report", "commands.json", "--sram-timing-out", "timing.json",
                    "--timeline-out", "timeline.json", "--timeline-merge-gap", "0",
                    "--timeline-max-events", "1000000", "--log-level", "off"]
                hashes = {str(path): digest(path) for path in (source, manifest, target, directory / "inputs.npz", build / "program.bin", build / "lp6.bin")}
                (build / "invocation.json").write_text(json.dumps(dict(command=list(map(str, simulation)), source_sha256=hashes), indent=2))
                begin = time.monotonic()
                run(simulation, build, build / "run.log", dict(os.environ, LP6_DRAMSIM3_ROOT=str(workspace / "core/simulator/dram-pim")))
                wall = time.monotonic() - begin
                descriptor = io["outputs"][0]
                raw = (build / "sram_dump.core0.bin").read_bytes()
                actual = np.frombuffer(raw[descriptor["l1"]:descriptor["l1"] + descriptor["bytes"]], dtype=np.float16).reshape(descriptor["shape"])
                np.save(build / "output.npy", actual)
                outputs[mode] = actual.copy()
                timing = json.loads((build / "timing.json").read_text())
                timeline = json.loads((build / "timeline.json").read_text())
                result = {"compiled": True, "executed": True,
                    "total_us": timing["total_simulated_cycles"] / timing["npu_clock_mhz"],
                    "wall_seconds": wall, "fp64_reference": errors(actual, expected),
                    "dense_rounded_score_reference": errors(actual, contracted),
                    "online_contract_reference": errors(actual, online_contract_reference(inputs, case, args.splits if mode == "decoding" else 1),
                                                        ONLINE_ATOL, ONLINE_RTOL),
                    "legacy_reference": errors(actual, legacy), "elements": int(actual.size),
                    "lp6_read_bytes": timing["lp6_bytes_read"], "lp6_write_bytes": timing["lp6_bytes_written"],
                    "l1_peak_bytes": max(phase.get("l1_peak", 0) for phase in io["phases"]),
                    "l2_required_bytes": timing["l2_bytes_required"],
                    "system_commands": timing["system_command_count"],
                    "trace": {"dropped_events": timeline["dropped_events"],
                              "merge_gap_cycles": timeline["merge_gap_cycles"],
                              "events": len(timeline["events"])},
                    "phase_metadata": io["phases"]}
                if "off" in outputs and mode != "off":
                    result["versus_native_off"] = errors(actual, outputs["off"])
                    result["byte_exact_with_off"] = actual.tobytes() == outputs["off"].tobytes()
                case_results["modes"][mode] = result
                # The old rounded graph is diagnostic, including known NaN
                # all-masked behavior. The new kernels must satisfy the oracle.
                if mode != "off" and not result["dense_rounded_score_reference"]["tolerance_pass"]:
                    failures.append(f"{case.name}/{mode}: dense rounded-score reference mismatch")
                if mode != "off" and not result["online_contract_reference"]["tolerance_pass"]:
                    failures.append(f"{case.name}/{mode}: online mixed-precision reference mismatch")
                print(case.name, mode, json.dumps({key: value for key, value in result.items() if key != "phase_metadata"}), flush=True)
            except Exception as error:
                case_results["modes"][mode] = {"error": str(error)}
                failures.append(f"{case.name}/{mode}: {error}")
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["failures"] = failures
    for path in paths:
        if path.exists() and digest(path) != provenance[str(path)]:
            failures.append(f"Input tool changed during execution: {path}")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
