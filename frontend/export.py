#!/usr/bin/env python3
"""Export a Qwen3, Llama or Qwen3.5 (hybrid Gated DeltaNet) decode step to linalg-on-tensors MLIR.

Writes <out>/step.mlir (the func @decode_step on tensors) and <out>/manifest.json that names
every function argument and result (kind: input / state / weight / output, checkpoint tensor
name, shape). No weight data enters the IR: the export runs on meta tensors and the compiler
reads the checkpoint itself."""
import argparse, json, sys, time
from pathlib import Path
import torch
from torch_mlir import fx
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder import DecoderDims, DecodeStep, weight_names, K_CACHE_LAYOUT
import qwen35

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=Path("/home/jongjip/models/qwen-3-4b"))
ap.add_argument("--tokens", type=int, default=1, help="M: tokens per decode step")
ap.add_argument("--capacity", type=int, default=256,
                help="P: KV cache capacity; 0 exports a prefill step with no cache input")
ap.add_argument("--layers", type=int, default=0, help="truncate to N layers (0 = all)")
ap.add_argument("--logits", choices=["all", "last"], default=None,
                help="LM head rows: all positions (default for decode/verify) or the last (default for prefill)")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--debug-outputs", action="store_true",
                help="Qwen3.5: also return the first GDN layer's readout, gated norm and out_proj (debugging)")
a = ap.parse_args()

hybrid = qwen35.HybridDims.is_hybrid(a.model)
d = qwen35.HybridDims.from_config(a.model) if hybrid else DecoderDims.from_config(a.model)
if a.layers: d.layers = a.layers
M, P = a.tokens, a.capacity
H, D, KVH = d.hidden, d.head_dim, d.kv_heads
meta = lambda *s: torch.empty(*s, device="meta")
R = d.rope_dim if hybrid else D                                  # rotary dims (partial RoPE)
x, cos, sin, mask = meta(M, H), meta(M, R), meta(M, R), meta(M, P + M)
kv_layers = d.attention_layers() if hybrid else list(range(d.layers))
gdn_layers = d.gdn_layers() if hybrid else []
# Qwen3/Llama store K as [KVH, D, P] (decoder.K_CACHE_LAYOUT); the Qwen3.5 step keeps [KVH, P, D].
pk = [meta(KVH, P, D) if hybrid else meta(KVH, D, P) for _ in kv_layers] if P else []
pv = [meta(KVH, P, D) for _ in kv_layers] if P else []
conv = [meta(d.conv_kernel - 1, d.conv_dim) for _ in gdn_layers]
state = [meta(d.v_heads, d.v_dim, d.k_dim) for _ in gdn_layers]
step = "prefill" if P == 0 else "decode" if M == 1 else "verify"
wn = qwen35.weight_names(d) if hybrid else weight_names(d)
W = [meta(*s) for _, s in wn]
t0 = time.time()
last = (a.logits or ("last" if step == "prefill" else "all")) == "last"
if hybrid:
    module = fx.export_and_import(qwen35.HybridStep(d, M, P, last_logits=last, debug=a.debug_outputs), x, cos, sin, mask, pk, pv, conv, state, W,
                                  output_type="linalg-on-tensors", func_name="decode_step")
else:
    module = fx.export_and_import(DecodeStep(d, M, P, last_logits=last), x, cos, sin, mask, pk, pv, W,
                                  output_type="linalg-on-tensors", func_name="decode_step")
t1 = time.time()
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "step.mlir").write_text(module.operation.get_asm(print_generic_op_form=False, large_elements_limit=16))
args = [dict(kind="input", name=n, shape=list(t.shape)) for n, t in (("x", x), ("cos", cos), ("sin", sin), ("mask", mask))]
if P:
    args += [dict(kind="state", name=f"past_k.{i}", layer=i, shape=[KVH, P, D]) if hybrid else
             dict(kind="state", name=f"past_k.{i}", layer=i, shape=[KVH, D, P], layout=K_CACHE_LAYOUT)
             for i in kv_layers]
    args += [dict(kind="state", name=f"past_v.{i}", layer=i, shape=[KVH, P, D]) for i in kv_layers]
# GDN layers: the last K-1 conv inputs (FP16 like the activations) and the FP32 recurrent state.
args += [dict(kind="state", name=f"conv.{i}", layer=i, shape=[d.conv_kernel - 1, d.conv_dim]) for i in gdn_layers]
args += [dict(kind="state", name=f"state.{i}", layer=i, shape=[d.v_heads, d.v_dim, d.k_dim], dtype="f32")
         for i in gdn_layers]
args += [dict(kind="weight", name=n, shape=list(s)) for n, s in wn]
results = [dict(kind="output", name="logits", shape=[1 if last else M, d.vocab])]
results += [dict(kind="output", name=f"new_k.{i}", layer=i, shape=[KVH, M, D]) for i in kv_layers]
results += [dict(kind="output", name=f"new_v.{i}", layer=i, shape=[KVH, M, D]) for i in kv_layers]
results += [dict(kind="output", name=f"new_conv.{i}", layer=i, shape=[d.conv_kernel - 1, d.conv_dim]) for i in gdn_layers]
results += [dict(kind="output", name=f"new_state.{i}", layer=i, shape=[d.v_heads, d.v_dim, d.k_dim], dtype="f32")
            for i in gdn_layers]
if hybrid and a.debug_outputs:
    results += [dict(kind="output", name="debug.readout", shape=[M, d.v_heads, d.v_dim]),
                dict(kind="output", name="debug.gated_norm", shape=[M, d.v_heads, d.v_dim]),
                dict(kind="output", name="debug.out_proj", shape=[M, d.hidden])]
(a.out / "manifest.json").write_text(json.dumps(dict(
    model=str(a.model), family=d.model_type, step=step, tokens=M, capacity=P, dims=d.__dict__, mask_min=-30000.0,
    arguments=args, results=results), indent=1))
print(f"exported {d.layers} layers, M={M}, P={P}: {len(args)} args, {len(results)} results, "
      f"{(a.out / 'step.mlir').stat().st_size / 1e6:.1f} MB IR [{t1 - t0:.0f}s]")
