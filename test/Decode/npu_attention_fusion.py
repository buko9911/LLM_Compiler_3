#!/usr/bin/env python3
"""Regression for head-local attention fusion across GQA reshapes.

The pass changes only distribution/locality, not softmax/key reductions.
Native numerical A/B is exercised separately by the decode replay harness.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def attention_source(heads=4, groups=2, tokens=3, keys=7, depth=4,
                     reshaped=True, shared=False, indexed=False, phase="npu", with_mask=False):
    rows = groups * tokens
    score = f"{heads}x{rows}x{keys}xf16"
    out = f"{heads}x{rows}x{depth}xf16"
    query = f"{heads}x{rows}x{depth}xf16"
    key = f"{heads}x{depth}x{keys}xf16"
    value = f"{heads}x{keys}x{depth}xf16"
    if reshaped:
        prob = f"{heads}x{groups}x{tokens}x{keys}xf16"
        stats = f"{heads}x{groups}x{tokens}xf32"
        ids = "b,g,m,s"
        output_map = "b,g,m"
        reassoc = "[[0],[1,2],[3]]"
        expand = f"%expanded = tensor.expand_shape %scores {reassoc} output_shape [{heads},{groups},{tokens},{keys}] : tensor<{score}> into tensor<{prob}>"
        collapse = f"%collapsed = tensor.collapse_shape %p {reassoc} : tensor<{prob}> into tensor<{score}>"
        score_value, probability_value = "%expanded", "%collapsed"
    else:
        prob = score
        stats = f"{heads}x{rows}xf32"
        ids, output_map = "b,m,s", "b,m"
        expand = collapse = ""
        score_value, probability_value = "%scores", "%p"
    iters = ','.join('"parallel"' for _ in ids.split(','))
    reductions = ','.join(['"parallel"'] * (len(ids.split(','))-1) + ['"reduction"'])
    result_type = f"(tensor<{out}>,tensor<{score}>)" if shared else f"tensor<{out}>"
    returned = f"%y,%scores : tensor<{out}>,tensor<{score}>" if shared else f"%y : tensor<{out}>"
    index_op = "%index = linalg.index 0 : index" if indexed else ""
    mask_arg = f", %mask:tensor<{tokens}x{keys}xf16>" if with_mask else ""
    mask_op = ""
    mask_map = ""
    if with_mask:
        assert reshaped
        mask_map = f"#mask = affine_map<({ids})->(m,s)>"
        mask_op = f'''%masked = linalg.generic {{indexing_maps=[#id,#mask,#id],iterator_types=[{iters}]}}
        ins({score_value},%mask:tensor<{prob}>,tensor<{tokens}x{keys}xf16>) outs(%prob_empty:tensor<{prob}>) {{
    ^bb0(%x:f16,%mask_value:f16,%o:f16):
      %xf = arith.extf %x : f16 to f32
      %mf = arith.extf %mask_value : f16 to f32
      %add = arith.addf %xf,%mf : f32
      %r = arith.truncf %add : f32 to f16
      linalg.yield %r : f16
    }} -> tensor<{prob}>'''
        score_value = "%masked"
    return f'''#id = affine_map<({ids})->({ids})>
#row = affine_map<({ids})->({output_map})>
{mask_map}
module {{
  func.func @phase(%q:tensor<{query}>, %k:tensor<{key}>, %v:tensor<{value}>{mask_arg}) -> {result_type}
      attributes {{hyb.phase = "{phase}"}} {{
    %zero = arith.constant 0.0 : f16
    %zero32 = arith.constant 0.0 : f32
    %neg = arith.constant 0xFF800000 : f32
    %score_empty = tensor.empty() : tensor<{score}>
    %score_init = linalg.fill ins(%zero : f16) outs(%score_empty : tensor<{score}>) -> tensor<{score}>
    %scores = linalg.batch_matmul ins(%q,%k : tensor<{query}>,tensor<{key}>) outs(%score_init : tensor<{score}>) -> tensor<{score}>
    {expand}
    %prob_empty = tensor.empty() : tensor<{prob}>
    {mask_op}
    %row_empty = tensor.empty() : tensor<{stats}>
    %max_init = linalg.fill ins(%neg : f32) outs(%row_empty : tensor<{stats}>) -> tensor<{stats}>
    %max = linalg.generic {{indexing_maps=[#id,#row],iterator_types=[{reductions}]}}
        ins({score_value}:tensor<{prob}>) outs(%max_init:tensor<{stats}>) {{
    ^bb0(%x:f16,%acc:f32):
      %xf = arith.extf %x : f16 to f32
      %r = arith.maximumf %xf,%acc : f32
      linalg.yield %r : f32
    }} -> tensor<{stats}>
    %exp = linalg.generic {{indexing_maps=[#id,#row,#id],iterator_types=[{iters}]}}
        ins({score_value},%max:tensor<{prob}>,tensor<{stats}>) outs(%prob_empty:tensor<{prob}>) {{
    ^bb0(%x:f16,%max_arg:f32,%o:f16):
      {index_op}
      %xf = arith.extf %x : f16 to f32
      %sub = arith.subf %xf,%max_arg : f32
      %e = math.exp %sub : f32
      %r = arith.truncf %e : f32 to f16
      linalg.yield %r : f16
    }} -> tensor<{prob}>
    %sum_init = linalg.fill ins(%zero32 : f32) outs(%row_empty : tensor<{stats}>) -> tensor<{stats}>
    %sum = linalg.generic {{indexing_maps=[#id,#row],iterator_types=[{reductions}]}}
        ins(%exp:tensor<{prob}>) outs(%sum_init:tensor<{stats}>) {{
    ^bb0(%x:f16,%acc:f32):
      %xf = arith.extf %x : f16 to f32
      %r = arith.addf %xf,%acc : f32
      linalg.yield %r : f32
    }} -> tensor<{stats}>
    %p = linalg.generic {{indexing_maps=[#id,#row,#id],iterator_types=[{iters}]}}
        ins(%exp,%sum:tensor<{prob}>,tensor<{stats}>) outs(%prob_empty:tensor<{prob}>) {{
    ^bb0(%x:f16,%sum_arg:f32,%o:f16):
      %xf = arith.extf %x : f16 to f32
      %d = arith.divf %xf,%sum_arg : f32
      %r = arith.truncf %d : f32 to f16
      linalg.yield %r : f16
    }} -> tensor<{prob}>
    {collapse}
    %out_empty = tensor.empty() : tensor<{out}>
    %out_init = linalg.fill ins(%zero : f16) outs(%out_empty : tensor<{out}>) -> tensor<{out}>
    %y = linalg.batch_matmul ins({probability_value},%v:tensor<{score}>,tensor<{value}>) outs(%out_init:tensor<{out}>) -> tensor<{out}>
    return {returned}
  }}
}}
'''


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", type=Path, default=root/"build/bin/npu-opt")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    cases = {
        "gqa_reshape": (dict(), True, 4, 1),
        "disabled": (dict(), False, 4, 0),
        "already_direct": (dict(reshaped=False), True, 4, 0),
        "shared_scores": (dict(shared=True), True, 4, 0),
        "index_dependent_body": (dict(indexed=True), True, 4, 0),
        "one_core": (dict(), True, 1, 0),
        "head_divisor": (dict(heads=6), True, 4, 1),
        "pim_phase": (dict(phase="pim"), True, 4, 0),
        "mixed_module": (dict(), True, 4, 0),
        "per_token_mask": (dict(with_mask=True), True, 4, 1),
    }
    results = {}
    for name, (kwargs, enabled, cores, expected) in cases.items():
        source = output/f"{name}.mlir"
        text = attention_source(**kwargs)
        if name == "mixed_module":
            text = text.removesuffix("}\n") + '  func.func private @pim() attributes {hyb.phase = "pim"}\n}\n'
        source.write_text(text)
        spec = json.loads((root/"configs/target.json").read_text())
        spec["npu"].update(attention_fusion=enabled, cores=cores)
        target = output/f"{name}.json"; target.write_text(json.dumps(spec))
        result = subprocess.run([str(args.optimizer.resolve()), str(source),
                                f"--hyb-npu-attention-fusion=spec={target.resolve()}"],
                               text=True, capture_output=True)
        (output/f"{name}.log").write_text(result.stdout+result.stderr)
        assert result.returncode == 0, (name,result.stderr)
        assert result.stdout.count("hyb.attention_fused") == expected, name
        if expected:
            assert result.stdout.count("linalg.batch_matmul") == 2, result.stdout
            assert result.stdout.count("math.exp") == 1, result.stdout
            assert result.stdout.count("arith.maximumf") == 1, result.stdout
            assert result.stdout.count("arith.divf") == 1, result.stdout
            # Two full-key reductions remain; neither was transformed into an
            # online recurrence or tiled over the seven-key dimension.
            assert result.stdout.count('"reduction"') == 2, result.stdout
            actual_cores = 3 if name == "head_divisor" else cores
            assert f"hyb.cores = {actual_cores} : i64" in result.stdout, result.stdout
            assert "tensor<1x2x3x7xf16>" in result.stdout or "tensor<2x2x3x7xf16>" in result.stdout
            loop_pos = result.stdout.index("scf.forall")
            assert result.stdout.index("math.exp") > loop_pos
            assert result.stdout.index("linalg.batch_matmul") > loop_pos
            assert "affine.linearize_index" not in result.stdout
            assert "affine.delinearize_index" not in result.stdout
            once = output/f"{name}.once.mlir"
            once.write_text(result.stdout)
            repeated = subprocess.run([str(args.optimizer.resolve()), str(once),
                                       f"--hyb-npu-attention-fusion=spec={target.resolve()}"],
                                      text=True,capture_output=True)
            assert repeated.returncode == 0, repeated.stderr
            assert repeated.stdout == result.stdout, (name,"pass must be idempotent")
        if name == "already_direct":
            assert "already_direct = 1" in result.stdout, result.stdout
        if not enabled or name in ("pim_phase", "mixed_module"):
            assert "hyb.attention_fusion" not in result.stdout, result.stdout
        results[name] = dict(fused_chains=expected)
    (output/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
