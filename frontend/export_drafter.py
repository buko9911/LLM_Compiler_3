#!/usr/bin/env python3
"""Export one drafter step to linalg-on-tensors MLIR (drafters.py). Writes <out>/step.mlir and <out>/manifest.json in the format of export.py.

  export_drafter.py --kind block --model DIR --block 7 --context 5 --capacity 512 --head-rows 7 --out OUT
  export_drafter.py --kind markov --model DIR --rows 1 --out OUT
  export_drafter.py --kind eagle_extend|eagle_layer --model DIR --context C --capacity 512 --out OUT
  export_drafter.py --kind eagle_lm --model DIR --out OUT
--lm-model/--lm-name: read the LM head from the target checkpoint (DFlash-16 has none of its own)."""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch_mlir import fx
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drafters as DR
from decoder import K_CACHE_LAYOUT

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["block", "markov", "eagle_extend", "eagle_layer", "eagle_lm"], required=True)
ap.add_argument("--model", type=Path, required=True)
ap.add_argument("--block", type=int, default=7)
ap.add_argument("--context", type=int, default=5)
ap.add_argument("--capacity", type=int, default=512)
ap.add_argument("--head-rows", type=int, default=0)
ap.add_argument("--rows", type=int, default=1)
ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--lm-name", default="lm_head.weight")
ap.add_argument("--target-hidden", type=int, default=2560)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
d = DR.dims_of(a.model) if a.kind not in ("eagle_extend", "eagle_layer", "eagle_lm") else None
if d is None:
    c = json.loads((a.model / "config.json").read_text())
    from decoder import DecoderDims
    d = DecoderDims(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"], c.get("head_dim", 128),
                    c["intermediate_size"], c["vocab_size"], c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], False)
if a.layers:
    d.layers = a.layers
meta = lambda *s: torch.empty(*s, device="meta")
H, D, KVH = d.hidden, d.head_dim, d.kv_heads
P = a.capacity
args, results = [], []
inp = lambda name, *s: (args.append(dict(kind="input", name=name, shape=list(s))), meta(*s))[1]
t0 = time.time()
if a.kind == "block":
    B, C = a.block, a.context
    head = a.head_rows or B
    x, ctx = inp("x", B, H), inp("ctx", C, 5 * H)
    cc, sc = inp("cos_ctx", C, D), inp("sin_ctx", C, D)
    cos, sin, mask = inp("cos", B, D), inp("sin", B, D), inp("mask", B, P + C + B)
    pk = [meta(KVH, D, P) for _ in range(d.layers)]
    pv = [meta(KVH, P, D) for _ in range(d.layers)]
    args += [dict(kind="state", name=f"past_k.{i}", layer=i, shape=[KVH, D, P], layout=K_CACHE_LAYOUT) for i in range(d.layers)]
    args += [dict(kind="state", name=f"past_v.{i}", layer=i, shape=[KVH, P, D]) for i in range(d.layers)]
    wn = DR.block_weight_names(d, head, lm_name=a.lm_name)
    W = [meta(*s) for _, s in wn]
    module = fx.export_and_import(DR.BlockStep(d, B, C, P, head), x, ctx, cc, sc, cos, sin, mask, pk, pv, W, output_type="linalg-on-tensors", func_name="decode_step")
    results = [dict(kind="output", name="logits", shape=[head, d.vocab])]
    results += [dict(kind="output", name=f"new_k.{i}", layer=i, shape=[KVH, C, D]) for i in range(d.layers)]
    results += [dict(kind="output", name=f"new_v.{i}", layer=i, shape=[KVH, C, D]) for i in range(d.layers)]
    M = B
elif a.kind == "markov":
    R = a.rows
    prev, base = inp("prev", R, 256), inp("base", R, d.vocab)
    wn = [("markov_head.markov_w2.weight", (d.vocab, 256))]
    module = fx.export_and_import(DR.MarkovStep(), prev, base, meta(d.vocab, 256), output_type="linalg-on-tensors", func_name="decode_step")
    results = [dict(kind="output", name="logits", shape=[R, d.vocab])]
    M, P = R, 0
elif a.kind in ("eagle_extend", "eagle_layer"):
    M = a.context if a.kind == "eagle_extend" else 1
    ext = a.kind == "eagle_extend"
    hid = inp("hidden", M, (5 if ext else 1) * H)
    emb = inp("embed", M, H)
    cos, sin, mask = inp("cos", M, D), inp("sin", M, D), inp("mask", M, P + M)
    pk, pv = [meta(KVH, D, P)], [meta(KVH, P, D)]
    args += [dict(kind="state", name="past_k.0", layer=0, shape=[KVH, D, P], layout=K_CACHE_LAYOUT),
             dict(kind="state", name="past_v.0", layer=0, shape=[KVH, P, D])]
    wn = ([("fc.weight", (H, 5 * H))] if ext else []) + DR.eagle_weight_names(d)
    W = [meta(*s) for _, s in wn]
    module = fx.export_and_import(DR.EagleLayer(d, M, P, ext), hid, emb, cos, sin, mask, pk, pv, W, output_type="linalg-on-tensors", func_name="decode_step")
    results = [dict(kind="output", name="hidden", shape=[M, H]), dict(kind="output", name="new_k.0", layer=0, shape=[KVH, M, D]),
               dict(kind="output", name="new_v.0", layer=0, shape=[KVH, M, D])]
else:
    hid = inp("hidden", 1, H)
    wn = [("norm.weight", (H,)), (a.lm_name, (d.vocab, H))]
    module = fx.export_and_import(DR.EagleLm(d), hid, meta(H), meta(d.vocab, H), output_type="linalg-on-tensors", func_name="decode_step")
    results = [dict(kind="output", name="logits", shape=[1, d.vocab])]
    M, P = 1, 0
t1 = time.time()
args += [dict(kind="weight", name=n, shape=list(s)) for n, s in wn]
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "step.mlir").write_text(module.operation.get_asm(print_generic_op_form=False, large_elements_limit=16))
(a.out / "manifest.json").write_text(json.dumps(dict(model=str(a.model), family=d.model_type, step=a.kind, tokens=M, capacity=P, dims=d.__dict__, mask_min=-30000.0,
                                                     arguments=args, results=results), indent=1))
print(f"exported {a.kind}: M={M} P={P}: {len(args)} args, {len(results)} results, {(a.out / 'step.mlir').stat().st_size / 1e6:.1f} MB IR [{t1 - t0:.0f}s]")
