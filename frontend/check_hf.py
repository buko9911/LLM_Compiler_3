#!/usr/bin/env python3
"""Check the explicit-state decoder against transformers (FP32, CPU).

Runs the HF model on prompt[:-1] to build the KV cache, then compares one decode step for the
last prompt token (logits and its K/V rows) between HF and the explicit-state step.
Use --layers N to truncate the model (both sides) for a quick check."""
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decoder import DecoderDims, DecodeStep, weight_names, rope_tables, step_mask
from weights import SafetensorStore

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=Path("/home/jongjip/models/qwen-3-4b"))
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--capacity", type=int, default=32)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--output", type=Path, help="write validation metrics as JSON")
a = ap.parse_args()

d = DecoderDims.from_config(a.model); d.layers = a.layers
cfg = AutoConfig.from_pretrained(a.model); cfg.num_hidden_layers = a.layers
hf = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok(a.prompt, return_tensors="pt").input_ids
T = ids.shape[1]
with torch.no_grad():
    ref = hf(ids, use_cache=True)
    pre = hf(ids[:, :-1], use_cache=True)
cache = pre.past_key_values
store = SafetensorStore(a.model)
W = [store.get(n).float() for n, _ in weight_names(d)]
P = a.capacity; assert P >= T - 1
pk = torch.zeros(d.layers, d.kv_heads, P, d.head_dim); pv = torch.zeros_like(pk)
for i in range(d.layers):
    pk[i, :, :T - 1] = cache.layers[i].keys[0]; pv[i, :, :T - 1] = cache.layers[i].values[0]
cos, sin = rope_tables(d, [T - 1])
x = W[-1][ids[0, -1]][None] if d.tied else store.get("model.embed_tokens.weight").float()[ids[0, -1]][None]
step = DecodeStep(d, 1, P)
with torch.no_grad():
    logits, nk, nv = step(x, cos, sin, step_mask(1, P, T - 1), list(pk), list(pv), W)
full = ref.past_key_values
err = (logits[0] - ref.logits[0, -1]).abs().max().item() / ref.logits[0, -1].abs().max().item()
kerr = max((nk[i][:, 0] - full.layers[i].keys[0, :, -1]).abs().max().item() for i in range(d.layers))
torch.testing.assert_close(logits[0], ref.logits[0, -1], rtol=2e-4, atol=2e-4)
for i in range(d.layers):
    torch.testing.assert_close(nk[i][:, 0], full.layers[i].keys[0, :, -1], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(nv[i][:, 0], full.layers[i].values[0, :, -1], rtol=2e-4, atol=2e-4)
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(model=str(a.model), family=d.model_type, layers=d.layers,
        prompt=a.prompt, prompt_tokens=T, capacity=P, logits_max_relative_error=err,
        new_k_max_absolute_error=kerr, hf_argmax=int(ref.logits[0, -1].argmax()),
        frontend_argmax=int(logits[0].argmax()), passed=True), indent=2) + "\n")
print(f"layers {d.layers}, prompt tokens {T}: logits max rel err {err:.2e}, new K max abs err {kerr:.2e}, "
      f"top-1 HF {ref.logits[0, -1].argmax().item()} ours {logits[0].argmax().item()}")
