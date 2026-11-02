#!/usr/bin/env python3
"""Check the explicit-state Qwen3.5 step against transformers (FP32, CPU).

Runs the HF text model on the prompt without its last M tokens to build the caches (KV of the
attention layers, conv and recurrent state of the GDN layers), then compares one step of M tokens
(logits of every row, new K/V rows, new conv and recurrent states) between HF and the explicit-state
step. Use --layers N to truncate both models."""
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder import step_mask
from qwen35 import HybridDims, HybridStep, weight_names, rope_tables
from weights import SafetensorStore

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=Path("/data/jongjip/models/qwen35-4b"))
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--tokens", type=int, default=1, help="M: tokens of the checked step")
ap.add_argument("--capacity", type=int, default=32)
ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Germany is")
ap.add_argument("--output", type=Path, help="write validation metrics as JSON")
a = ap.parse_args()

d = HybridDims.from_config(a.model); d.layers = a.layers
cfg = AutoConfig.from_pretrained(a.model).text_config
cfg.num_hidden_layers = a.layers; cfg.layer_types = cfg.layer_types[:a.layers]
hf = Qwen3_5ForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(a.prompt, return_tensors="pt").input_ids
T, M = ids.shape[1], a.tokens
prefix = T - M
assert prefix >= 1 and a.capacity >= prefix
with torch.no_grad():
    ref = hf(ids, use_cache=True)
    pre = hf(ids[:, :prefix], use_cache=True)
cache, full = pre.past_key_values, ref.past_key_values
store = SafetensorStore(a.model)
W = [store.get(n).float() for n, _ in weight_names(d)]
P = a.capacity
attn, gdn = d.attention_layers(), d.gdn_layers()
pk = [torch.zeros(d.kv_heads, P, d.head_dim) for _ in attn]
pv = [torch.zeros(d.kv_heads, P, d.head_dim) for _ in attn]
for j, i in enumerate(attn):
    pk[j][:, :prefix] = cache.layers[i].keys[0]; pv[j][:, :prefix] = cache.layers[i].values[0]
K1 = d.conv_kernel - 1
conv = [cache.layers[i].conv_states[0][0, :, -K1:].t().contiguous() for i in gdn]  # [K-1, C]
state = [cache.layers[i].recurrent_states[0][0].transpose(-1, -2).contiguous() for i in gdn]   # [VH, dv, dk]
cos, sin = rope_tables(d, list(range(prefix, T)))
x = W[-1][ids[0, prefix:]] if d.tied else store.get("lm_head.weight").float()[ids[0, prefix:]]
step = HybridStep(d, M, P)
with torch.no_grad():
    logits, nk, nv, nconv, nstate = step(x, cos, sin, step_mask(M, P, prefix), pk, pv, conv, state, W)
want = ref.logits[0, prefix:]
err = ((logits - want).abs().max() / want.abs().max()).item()
torch.testing.assert_close(logits, want, rtol=2e-4, atol=2e-4)
kerr = 0.0
for j, i in enumerate(attn):
    torch.testing.assert_close(nk[j], full.layers[i].keys[0, :, prefix:], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(nv[j], full.layers[i].values[0, :, prefix:], rtol=2e-4, atol=2e-4)
    kerr = max(kerr, (nk[j] - full.layers[i].keys[0, :, prefix:]).abs().max().item())
serr = cerr = 0.0
for j, i in enumerate(gdn):
    want_state = full.layers[i].recurrent_states[0][0].transpose(-1, -2)
    want_conv = full.layers[i].conv_states[0][0, :, -K1:].t()
    torch.testing.assert_close(nstate[j], want_state, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(nconv[j], want_conv, rtol=2e-4, atol=2e-4)
    serr = max(serr, (nstate[j] - want_state).abs().max().item())
    cerr = max(cerr, (nconv[j] - want_conv).abs().max().item())
result = dict(model=str(a.model), family=d.model_type, layers=d.layers, layer_types=d.kinds(), prompt=a.prompt,
              prompt_tokens=T, tokens=M, capacity=P, logits_max_relative_error=err, new_k_max_absolute_error=kerr,
              recurrent_state_max_absolute_error=serr, conv_state_max_absolute_error=cerr,
              hf_argmax=want.argmax(-1).tolist(), frontend_argmax=logits.argmax(-1).tolist(), passed=True)
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
print(f"layers {d.layers} ({d.kinds().count('linear_attention')} GDN), M={M}, prompt tokens {T}: logits max rel err "
      f"{err:.2e}, new K max abs err {kerr:.2e}, state max abs err {serr:.2e}, conv max abs err {cerr:.2e}, "
      f"top-1 HF {result['hf_argmax']} ours {result['frontend_argmax']}")
