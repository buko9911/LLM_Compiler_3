#!/usr/bin/env python3
"""Compare decoder math, causal KV handling and Llama3 RoPE with Hugging Face."""
import json
from pathlib import Path
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))
from decoder import DecoderDims, DecodeStep, rope_tables, step_mask, weight_names
from prefill import prefill_kv


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    rows = []
    for family, Config, Model in (("llama", LlamaConfig, LlamaForCausalLM),
                                   ("qwen3", Qwen3Config, Qwen3ForCausalLM)):
        kw = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=96,
                  rms_norm_eps=1e-5, tie_word_embeddings=family == "qwen3",
                  attention_dropout=0.0, attention_bias=False, mlp_bias=False)
        if family == "llama":
            kw.update(rope_theta=500000.0, max_position_embeddings=131072,
                      rope_scaling=dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                                        high_freq_factor=4.0, original_max_position_embeddings=8192))
        else:
            # Qwen3 may have a query width different from the hidden width.
            kw.update(head_dim=32, rope_theta=1000000.0)
        config = Config(**kw)
        config._attn_implementation = "eager"
        hf = Model(config).eval()
        with tempfile.TemporaryDirectory(prefix=f"frontend-{family}-") as tmp:
            path = Path(tmp)
            hf.save_pretrained(path)
            d = DecoderDims.from_config(path)
            state = hf.state_dict()
            weights = [state[name] for name, _ in weight_names(d)]
            prefix = torch.tensor([[2, 7, 13, 17, 19]])
            expected = hf(prefix, use_cache=True).past_key_values
            pk, pv = prefill_kv(path, d, prefix[0].tolist(), 7)
            for i in range(d.layers):
                torch.testing.assert_close(pk[i, :, :5], expected.layers[i].keys[0], rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(pv[i, :, :5], expected.layers[i].values[0], rtol=2e-5, atol=2e-5)
                assert not pk[i, :, 5:].count_nonzero() and not pv[i, :, 5:].count_nonzero()
            for m in (1, 3):
                new = torch.tensor([[23, 29, 31]])[:, :m]
                ref = hf(torch.cat((prefix, new), dim=1), use_cache=True)
                x = hf.model.embed_tokens(new)[0]
                cos, sin = rope_tables(d, list(range(5, 5 + m)))
                got, nk, nv = DecodeStep(d, m, 7)(x, cos, sin, step_mask(m, 7, 5),
                                                list(pk), list(pv), weights)
                torch.testing.assert_close(got, ref.logits[0, -m:], rtol=2e-5, atol=2e-5)
                for i in range(d.layers):
                    torch.testing.assert_close(nk[i], ref.past_key_values.layers[i].keys[0, :, -m:],
                                               rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(nv[i], ref.past_key_values.layers[i].values[0, :, -m:],
                                               rtol=2e-5, atol=2e-5)
                rows.append(dict(family=family, tokens=m,
                                 max_logit_error=float((got-ref.logits[0, -m:]).abs().max())))
            if family == "llama":
                positions = torch.tensor([[0, 1, 255, 8191, 8192, 65535, 131071]])
                cos, sin = rope_tables(d, positions[0].tolist())
                hc, hs = hf.model.rotary_emb(torch.zeros(1, positions.numel(), d.hidden), positions)
                torch.testing.assert_close(cos, hc[0], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(sin, hs[0], rtol=1e-6, atol=1e-6)
                rows.append(dict(family=family, long_position_rope="pass"))
    print(json.dumps(dict(passed=True, cases=rows), indent=2))


if __name__ == "__main__":
    main()
