#!/usr/bin/env python3
"""Structural and configuration regressions for actual online attention.

Numerical tests live in npu_flash_attention_runtime.py. These checks establish
that online tiling/split merging is present, unsupported requests are rejected,
and compiler options cannot silently request an invalid hardware schedule.
"""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

from npu_attention_fusion import attention_source


def _scaled_source(*, score_scale=1.0, **graph):
    """Add an explicit score scale without changing the existing test builder."""
    source = attention_source(**graph)
    if score_scale == 1.0:
        return source
    heads = graph.get("heads", 4)
    groups = graph.get("groups", 2)
    tokens = graph.get("tokens", 3)
    keys = graph.get("keys", 7)
    shape = f"{heads}x{groups*tokens}x{keys}xf16"
    bits = struct.unpack("<I", struct.pack("<f", score_scale))[0]
    scale = f'''    %scaled = linalg.generic {{indexing_maps=[#scale_id,#scale_id],iterator_types=["parallel","parallel","parallel"]}}
        ins(%scores:tensor<{shape}>) outs(%score_empty:tensor<{shape}>) {{
    ^bb0(%x:f16,%o:f16):
      %xf = arith.extf %x : f16 to f32
      %factor = arith.constant 0x{bits:08X} : f32
      %product = arith.mulf %xf,%factor : f32
      %rounded = arith.truncf %product : f32 to f16
      linalg.yield %rounded : f16
    }} -> tensor<{shape}>
'''
    source = source.replace("tensor.expand_shape %scores", "tensor.expand_shape %scaled")
    source = source.replace("ins(%scores:tensor", "ins(%scaled:tensor")
    lines = source.splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if "%scores = linalg.batch_matmul" in line)
    lines.insert(index + 1, scale)
    return "#scale_id = affine_map<(h,m,k)->(h,m,k)>\n" + "".join(lines)


def flash_source(*, score_scale=1.0, kv_prefix=0, key_sequence_major=False,
                 shared_kv=False, kv_value_prefix=0, **graph):
    source = _scaled_source(score_scale=score_scale, **graph)
    heads, keys, depth = (graph.get(name, default) for name, default in
                          (("heads", 4), ("keys", 7), ("depth", 4)))
    old_key = f"{heads}x{depth}x{keys}xf16"
    key_type = f"{heads}x{keys}x{depth}xf16" if key_sequence_major else old_key
    if key_sequence_major:
        source = source.replace(f"%k:tensor<{old_key}>", f"%k:tensor<{key_type}>")
        lines = source.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines) if "%scores = linalg.batch_matmul" in line)
        maps = "indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>, affine_map<(b,m,n,k)->(b,n,k)>, affine_map<(b,m,n,k)->(b,m,n)>]"
        lines[index] = lines[index].replace("linalg.batch_matmul ins", f"linalg.batch_matmul {maps} ins").replace(f"tensor<{old_key}>", f"tensor<{key_type}>")
        source = "".join(lines)
    if kv_prefix:
        assert 0 < kv_prefix < keys
        suffix = keys - kv_prefix
        if key_sequence_major:
            prefix_key, suffix_key, axis = f"{heads}x{kv_prefix}x{depth}xf16", f"{heads}x{suffix}x{depth}xf16", 1
        else:
            prefix_key, suffix_key, axis = f"{heads}x{depth}x{kv_prefix}xf16", f"{heads}x{depth}x{suffix}xf16", 2
        value_type = f"{heads}x{keys}x{depth}xf16"
        vp = kv_value_prefix or kv_prefix
        prefix_value, suffix_value = f"{heads}x{vp}x{depth}xf16", f"{heads}x{keys-vp}x{depth}xf16"
        source = source.replace(f"%k:tensor<{key_type}>", f"%kp:tensor<{prefix_key}>, %kn:tensor<{suffix_key}>")
        source = source.replace(f"%v:tensor<{value_type}>", f"%vp:tensor<{prefix_value}>, %vn:tensor<{suffix_value}>")
        concat = f'''%k = tensor.concat dim({axis}) %kp, %kn : (tensor<{prefix_key}>, tensor<{suffix_key}>) -> tensor<{key_type}>
    %v = tensor.concat dim(1) %vp, %vn : (tensor<{prefix_value}>, tensor<{suffix_value}>) -> tensor<{value_type}>
    '''
        source = source.replace("%zero = arith.constant", concat + "%zero = arith.constant", 1)
        if shared_kv:
            out = f"{heads}x{graph.get('groups', 2)*graph.get('tokens', 3)}x{depth}xf16"
            lines = source.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if "func.func @phase" in line)
            lines[index] = lines[index].replace(f") -> tensor<{out}>", f") -> (tensor<{out}>, tensor<{key_type}>, tensor<{value_type}>)")
            source = "".join(lines)
            source = source.replace(f"return %y : tensor<{out}>", f"return %y,%k,%v : tensor<{out}>,tensor<{key_type}>,tensor<{value_type}>")
    return source


def invoke(optimizer, source, target=None, canonicalize=False):
    command = [str(optimizer), str(source)]
    if canonicalize:
        command.extend(["--hyb-simplify", "--canonicalize"])
    if target is not None:
        command.append(f"--hyb-npu-flash-attention=spec={target}")
    return subprocess.run(command, text=True, capture_output=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, default=root / "build/bin/npu-opt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output)).resolve()
    base = dict(heads=4, groups=2, tokens=3, keys=23, depth=8, with_mask=True)
    # name, graph changes, target changes, transform expected.
    cases = (
        ("online_gqa_mask_tail", {}, {}, True),
        ("split_key_decode", {"heads": 1, "tokens": 1, "keys": 67},
         {"flash_decoding": True, "attention_splits": 4}, True),
        ("prefill_query_tiles", {"tokens": 33, "keys": 65}, {}, True),
        ("direct_no_mask", {"reshaped": False, "with_mask": False}, {}, True),
        ("gqa_no_mask", {"with_mask": False}, {}, True),
        ("non_power_two_score_scale", {"score_scale": 0.3535533905932738}, {}, True),
        ("concat_transposed_key", {"keys": 67, "kv_prefix": 64}, {"flash_kv_slicing": True}, True),
        ("concat_sequence_major_key", {"tokens": 8, "keys": 72, "kv_prefix": 64, "key_sequence_major": True},
         {"flash_kv_slicing": True, "flash_decoding": True, "attention_splits": 2}, True),
        ("concat_slicing_off", {"keys": 67, "kv_prefix": 64}, {"flash_kv_slicing": False}, True),
        ("concat_mismatched_boundaries", {"keys": 67, "kv_prefix": 64, "kv_value_prefix": 63},
         {"flash_kv_slicing": True}, True),
        ("concat_shared_results", {"keys": 67, "kv_prefix": 64, "shared_kv": True},
         {"flash_kv_slicing": True}, True),
        ("canonical_single_head_m1", {"heads": 1, "groups": 4, "tokens": 1, "keys": 67, "depth": 32}, {}, True),
        ("canonical_single_head_m8", {"heads": 1, "tokens": 8, "keys": 72, "depth": 16}, {}, True),
        ("disabled", {}, {"flash_attention": False}, False),
        ("shared_score_result", {"shared": True}, {}, False),
        ("index_dependent_softmax", {"indexed": True}, {}, False),
        ("pim_phase", {"phase": "pim"}, {}, False),
        ("mixed_module", {}, {}, False),
        ("wrong_softmax_math", {}, {}, False),
        ("fixed_nonzero_mask_row", {}, {}, False),
        ("nonzero_qk_init", {}, {}, False),
        ("nonzero_pv_init", {}, {}, False),
        ("nonzero_sum_init", {}, {}, False),
        ("wrong_max_init", {}, {}, False),
        ("empty_requested_split", {"tokens": 1, "keys": 3},
         {"flash_decoding": True, "attention_splits": 4}, False),
    )
    summary = {"kind": "compiler_structure_not_runtime_performance", "cases": {}}
    for name, graph_options, config_options, transformed in cases:
        graph = base | graph_options
        text = flash_source(**graph)
        if name == "mixed_module":
            text = text.removesuffix("}\n") + '  func.func private @pim() attributes {hyb.phase = "pim"}\n}\n'
        if name == "wrong_softmax_math":
            text = text.replace("math.exp %sub", "math.log %sub")
        if name == "fixed_nonzero_mask_row":
            text = text.replace("#mask = affine_map<(b,g,m,s)->(m,s)>",
                                "#mask = affine_map<(b,g,m,s)->(1,s)>")
        if name.startswith("nonzero_"):
            text = text.replace("%zero32 = arith.constant 0.0 : f32", "%zero32 = arith.constant 0.0 : f32\n    %one16 = arith.constant 1.0 : f16\n    %one32 = arith.constant 1.0 : f32")
            if name == "nonzero_qk_init":
                text = text.replace("%score_init = linalg.fill ins(%zero : f16)", "%score_init = linalg.fill ins(%one16 : f16)")
            elif name == "nonzero_pv_init":
                text = text.replace("%out_init = linalg.fill ins(%zero : f16)", "%out_init = linalg.fill ins(%one16 : f16)")
            else:
                text = text.replace("%sum_init = linalg.fill ins(%zero32 : f32)", "%sum_init = linalg.fill ins(%one32 : f32)")
        if name == "wrong_max_init":
            text = text.replace("%max_init = linalg.fill ins(%neg : f32)", "%max_init = linalg.fill ins(%zero32 : f32)")
        source = output / f"{name}.mlir"
        source.write_text(text)
        target = json.loads((root / "configs/target.json").read_text())
        target["npu"].update(flash_attention=True, flash_decoding=False,
            attention_query_tile=2, attention_key_tile=8, attention_splits=1)
        target["npu"].update(config_options)
        target_file = output / f"{name}.json"
        target_file.write_text(json.dumps(target, indent=2))
        result = invoke(args.optimizer.resolve(), source, target_file,
                        canonicalize=name.startswith("canonical_single_head"))
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        if not transformed and name != "disabled":
            # Explicitly requesting online attention on an unsupported graph
            # must fail rather than silently claiming the feature was enabled.
            assert result.returncode != 0, (name, "unsupported request accepted")
            assert "error:" in result.stderr, (name, result.stderr)
            summary["cases"][name] = {"rejected": True, "passed": True}
            continue
        assert result.returncode == 0, (name, result.stderr)
        counts = re.findall(r"transformed_chains = (\d+) : i64", result.stdout)
        transformed_count = sum(map(int, counts))
        assert transformed_count == int(transformed), (name, transformed_count, result.stdout)
        if transformed:
            algorithm = "flash_decoding" if config_options.get("flash_decoding") else "flash_attention"
            assert f'algorithm = "{algorithm}"' in result.stdout, result.stdout
            assert "hyb.flash_math" in result.stdout, result.stdout
            assert "hyb.flash_tile" in result.stdout, result.stdout
            assert "math.exp" in result.stdout, result.stdout
            assert "f32" in result.stdout, result.stdout
            # The original full score/probability tensor must disappear. The
            # inputs contain full K/V, but score storage is bounded by tiles.
            h, g, m, s = (graph[key] for key in ("heads", "groups", "tokens", "keys"))
            for shape in (f"{h}x{g*m}x{s}xf16", f"{h}x{g}x{m}x{s}xf16"):
                assert not re.search(r"tensor\.empty\(\)[^\n]*tensor<" + re.escape(shape) + r">", result.stdout), (name, "full score workspace remains")
            if graph.get("kv_prefix"):
                sliced = config_options.get("flash_kv_slicing") and not graph.get("shared_kv") and not graph.get("kv_value_prefix")
                assert ("segmented_kv_chains = 1" in result.stdout) == bool(sliced), (name, result.stdout)
                assert ("tensor.concat" not in result.stdout) == bool(sliced), (name, "unexpected concat materialization")
            if algorithm == "flash_decoding":
                assert "hyb.flash_decoding" in result.stdout, result.stdout
                assert "scf.forall" in result.stdout, result.stdout
            once = output / f"{name}.once.mlir"
            once.write_text(result.stdout)
            twice = invoke(args.optimizer.resolve(), once, target_file)
            assert twice.returncode == 0, (name, twice.stderr)
            assert twice.stdout == result.stdout, (name, "online pass is not idempotent")
        else:
            plain = invoke(args.optimizer.resolve(), source)
            assert plain.returncode == 0, plain.stderr
            assert result.stdout == plain.stdout, (name, "inactive path changed")
            assert "hyb.flash_tile" not in result.stdout, name
        summary["cases"][name] = {"transformed_chains": transformed_count, "passed": True}

    invalid = (
        ("bool_type", {"flash_attention": 1}),
        ("decode_without_flash", {"flash_attention": False, "flash_decoding": True, "attention_splits": 2}),
        ("query_tile_zero", {"attention_query_tile": 0}),
        ("key_tile_zero", {"attention_key_tile": 0}),
        ("too_many_splits", {"flash_decoding": True, "attention_splits": 8}),
        ("single_split_decode", {"flash_decoding": True, "attention_splits": 1}),
        ("splits_without_decoding", {"attention_splits": 2}),
        ("fractional_tile", {"attention_key_tile": 8.5}),
        ("overflow_tile", {"attention_query_tile": 2**63}),
    )
    source = output / "online_gqa_mask_tail.mlir"
    for name, overrides in invalid:
        target = json.loads((root / "configs/target.json").read_text())
        target["npu"].update(flash_attention=True, flash_decoding=False,
            attention_query_tile=2, attention_key_tile=8, attention_splits=1)
        target["npu"].update(overrides)
        target_file = output / f"invalid_{name}.json"
        target_file.write_text(json.dumps(target, indent=2))
        result = invoke(args.optimizer.resolve(), source, target_file)
        (output / f"invalid_{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode != 0, (name, "invalid configuration accepted")
        summary["cases"][f"invalid_{name}"] = {"rejected": True, "passed": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
