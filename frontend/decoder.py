"""Qwen3 / Llama decode step with explicit state, for torch.export.

A decode step processes M new tokens against a KV cache of fixed capacity. All shapes are
static (the NPU program has no indirect addressing):

  inputs   x        [M, H]              token embeddings (the host gathers the rows)
           cos, sin [M, D]              rotary tables of the M positions (host)
           mask     [M, P + M]          additive attention mask (0 or MASK_MIN; host)
           past_k/v [NL][KVH, P, D]     KV cache, P = cache capacity (read only in the step)
           weights  as named tensors    (see weight_names)
  outputs  logits   [M, V]
           new_k/v  [NL][KVH, M, D]     K/V rows of the M tokens; the host writes them into
                                        the cache (as NPU LLM runtimes do)

Both families use RMSNorm, rotate-half RoPE, grouped-query attention and SwiGLU.
Qwen3 also normalizes Q/K per head; Llama does not. Llama 3.1 uses wavelength-scaled
RoPE. Embedding tying follows the checkpoint config. check_hf.py checks the math
against the installed transformers model.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import torch

MASK_MIN = -30000.0      # additive mask for invalid positions (finite in FP16)


@dataclass
class DecoderDims:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    eps: float
    rope_theta: float
    tied: bool
    model_type: str = "qwen3"
    rope_scaling: dict | None = None

    @property
    def qk_norm(self):
        return self.model_type == "qwen3"

    @staticmethod
    def from_config(path: Path) -> "DecoderDims":
        c = json.loads((Path(path) / "config.json").read_text())
        family = c["model_type"]
        if family not in ("qwen3", "llama"):
            raise ValueError(f"unsupported decoder family: {family}")
        if c.get("attention_bias", False) or c.get("mlp_bias", False) or c.get("hidden_act", "silu") != "silu":
            raise ValueError("only bias-free attention/MLP with SiLU is supported")
        rp = c.get("rope_parameters") or c.get("rope_scaling") or {}
        rt = rp.get("rope_type", rp.get("type", "default"))
        if rt not in ("default", "llama3") or rp.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError(f"unsupported RoPE parameters: {rp}")
        if c["num_attention_heads"] % c["num_key_value_heads"]:
            raise ValueError("query heads must be divisible by KV heads")
        return DecoderDims(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"],
                         c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]), c["intermediate_size"],
                         c["vocab_size"], c["rms_norm_eps"], rp.get("rope_theta", c.get("rope_theta", 10000.0)),
                         c.get("tie_word_embeddings", False), family, rp if rt != "default" else None)


def weight_names(d: DecoderDims) -> list[tuple[str, tuple[int, ...]]]:
    """Checkpoint tensor names and shapes the step reads, in argument order."""
    H, D, F = d.hidden, d.head_dim, d.ffn
    out = []
    for i in range(d.layers):
        p = f"model.layers.{i}."
        out += [(p + "input_layernorm.weight", (H,)),
                (p + "self_attn.q_proj.weight", (d.heads * D, H)),
                (p + "self_attn.k_proj.weight", (d.kv_heads * D, H)),
                (p + "self_attn.v_proj.weight", (d.kv_heads * D, H))]
        if d.qk_norm:
            out += [(p + "self_attn.q_norm.weight", (D,)), (p + "self_attn.k_norm.weight", (D,))]
        out += [(p + "self_attn.o_proj.weight", (H, d.heads * D)),
                (p + "post_attention_layernorm.weight", (H,)),
                (p + "mlp.gate_proj.weight", (F, H)),
                (p + "mlp.up_proj.weight", (F, H)),
                (p + "mlp.down_proj.weight", (H, F))]
    out += [("model.norm.weight", (H,)),
            ("model.embed_tokens.weight" if d.tied else "lm_head.weight", (d.vocab, H))]
    return out


def rmsnorm(x, w, eps):
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rope(x, cos, sin):
    """x [..., M, n, D] with cos/sin [M, 1, D] broadcast over the n heads: rotate-half RoPE."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def fake_quant_rows(x):
    """Token-wise symmetric INT8 (the hyb.quant contract), returned dequantized."""
    s = x.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.full_like(s, 1e-12))
    return torch.clamp(torch.round(x / s), -127, 127) * s


class DecodeStep(torch.nn.Module):
    """act_quant: quantize the input of every linear token-wise to INT8 (reference for W8A8;
    pass dequantized per-row INT8 weights). Off for export."""
    def __init__(self, d: DecoderDims, tokens: int, capacity: int, act_quant: bool = False):
        super().__init__()
        self.d, self.M, self.P, self.act_quant = d, tokens, capacity, act_quant

    def lin(self, x, w):
        if isinstance(w, tuple):            # (INT8 weight, per-row scale): dequantize on use
            w = w[0].float() * w[1][:, None]
        elif w.dtype != torch.float32:
            w = w.float()
        return (fake_quant_rows(x) if self.act_quant else x) @ w.t()

    def forward(self, x, cos, sin, mask, past_k, past_v, weights):
        d, M = self.d, self.M
        H, D, QH, KVH = d.hidden, d.head_dim, d.heads, d.kv_heads
        G = QH // KVH
        w = iter(weights)
        c = cos[:, None, :]; s = sin[:, None, :]          # [M, 1, D]
        new_k, new_v = [], []
        for i in range(d.layers):
            ln1, wq, wk, wv = (next(w) for _ in range(4))
            if d.qk_norm:
                qn, kn = next(w), next(w)
            wo, ln2, wg, wu, wd = (next(w) for _ in range(5))
            h = rmsnorm(x, ln1, d.eps)
            q = self.lin(h, wq).reshape(M, QH, D)
            if d.qk_norm:
                q = rmsnorm(q, qn, d.eps)
            k = self.lin(h, wk).reshape(M, KVH, D)
            if d.qk_norm:
                k = rmsnorm(k, kn, d.eps)
            v = self.lin(h, wv).reshape(M, KVH, D)
            q = rope(q, c, s)
            k = rope(k, c, s)
            k = k.transpose(0, 1)                          # [KVH, M, D]
            v = v.transpose(0, 1)
            new_k.append(k); new_v.append(v)
            K = torch.cat((past_k[i], k), dim=1)           # [KVH, P + M, D]
            V = torch.cat((past_v[i], v), dim=1)
            qg = q.transpose(0, 1).reshape(KVH, G * M, D)  # query rows of a KV group: head-major
            sc = (qg @ K.transpose(1, 2)) * (1.0 / D ** 0.5)
            sc = sc.reshape(KVH, G, M, -1) + mask          # mask [M, P + M] broadcast
            p = torch.softmax(sc, dim=-1).reshape(KVH, G * M, -1)
            o = (p @ V).reshape(QH, M, D).transpose(0, 1).reshape(M, QH * D)
            x = x + self.lin(o, wo)
            h = rmsnorm(x, ln2, d.eps)
            x = x + self.lin(torch.nn.functional.silu(self.lin(h, wg)) * self.lin(h, wu), wd)
        fn, lm = next(w), next(w)
        logits = self.lin(rmsnorm(x, fn, d.eps), lm)
        return logits, new_k, new_v


def rope_tables(d: DecoderDims, positions):
    # Preserve the original Qwen table calculation. Llama3 follows HF's FP32
    # frequency scaling and position arithmetic, including long positions.
    dtype = torch.float32 if d.rope_scaling else torch.float64
    inv = 1.0 / (d.rope_theta ** (torch.arange(0, d.head_dim, 2, dtype=dtype) / d.head_dim))
    if d.rope_scaling:
        rp = d.rope_scaling
        factor, low, high = rp["factor"], rp["low_freq_factor"], rp["high_freq_factor"]
        context = rp["original_max_position_embeddings"]
        wavelength = 2 * math.pi / inv
        inv = torch.where(wavelength > context / low, inv / factor, inv)
        smooth = (context / wavelength - low) / (high - low)
        medium = (wavelength >= context / high) & (wavelength <= context / low)
        inv = torch.where(medium, (1 - smooth) * inv / factor + smooth * inv, inv)
    ang = torch.tensor(positions, dtype=dtype)[:, None] * inv[None, :]
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos().float(), emb.sin().float()


def step_mask(M: int, capacity: int, past_len: int):
    """Additive mask: past slots < past_len valid, new tokens causal among themselves."""
    m = torch.full((M, capacity + M), MASK_MIN)
    m[:, :past_len] = 0.0
    for t in range(M):
        m[t, capacity:capacity + t + 1] = 0.0
    return m
