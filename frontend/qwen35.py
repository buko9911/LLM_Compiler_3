"""Qwen3.5 (hybrid: Gated DeltaNet and gated attention layers) decode step with explicit state, for
torch.export. The step has the same contract as decoder.DecodeStep: M new tokens against fixed-size
state, all shapes static.

Every fourth layer is full attention, the others are Gated DeltaNet (GDN) layers.

  inputs   x, cos, sin, mask    as decoder.DecodeStep; cos/sin cover only the rotary dims (partial RoPE)
           past_k/v [NA][KVH, P, D]    KV cache of the NA attention layers
           conv     [NG][K-1, C]       the last K-1 inputs of each GDN layer's causal conv (C channels),
                                       token-major so the taps vectorize along the channels
           state    [NG][VH, dv, dk]   each GDN layer's recurrent state, FP32, value-major: the reads
                                       S^T k and S^T q reduce the innermost (key) axis
           weights  as named tensors (weight_names)
  outputs  logits, new_k/v [NA][KVH, M, D], conv [NG][K-1, C], state [NG][VH, dv, dk]

GDN layer, token t (as transformers' torch_recurrent_gated_delta_rule):
  qkv = causal_conv4(in_proj_qkv(h)) -> SiLU; q, k = l2norm per head (q / sqrt(dk)), shared by
  VH/KH value heads; beta = sigmoid(in_proj_b(h)); decay = exp(-exp(A_log) * softplus(in_proj_a(h) + dt_bias))
  S = decay S;  S += (beta (v - S^T k)) k^T;  o = S^T q;  out_proj(rmsnorm(o) * w * silu(in_proj_z(h)))
softplus is computed from exp only (the NPU vector unit has no log or compare): softplus(u) =
max(u, 0) + log1p(exp(-|u|)) with max(u, 0) = (u + |u|) / 2 and log1p on [0, 1] by two Newton steps on
exp(z) = 1 + y (relative error < 1e-7).

Attention layer: q_proj gives the query and an output gate per head; q/k RMSNorm per head; RoPE on
the first rope_dim dims of each head; softmax attention (GQA); output * sigmoid(gate); o_proj.
Norms are zero-centered (x * (1 + w)) except the GDN output norm (x * w).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import torch

from decoder import MASK_MIN, fake_quant_rows, rope

FAMILY = "qwen3_5"


@dataclass
class HybridDims:
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
    rope_dim: int
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_kernel: int
    layer_types: list = field(default_factory=list)
    model_type: str = FAMILY
    prefix: str = "model.language_model."

    @property
    def conv_dim(self):
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim

    def kinds(self):
        return self.layer_types[:self.layers]

    def attention_layers(self):
        return [i for i, t in enumerate(self.kinds()) if t == "full_attention"]

    def gdn_layers(self):
        return [i for i, t in enumerate(self.kinds()) if t == "linear_attention"]

    @staticmethod
    def is_hybrid(path: Path) -> bool:
        c = json.loads((Path(path) / "config.json").read_text())
        return c.get("model_type") in ("qwen3_5", "qwen3_5_text")

    @staticmethod
    def from_config(path: Path) -> "HybridDims":
        c = json.loads((Path(path) / "config.json").read_text())
        t = c.get("text_config", c)
        rp = t.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default" or not t.get("attn_output_gate", False):
            raise ValueError(f"unsupported Qwen3.5 attention: {rp}, attn_output_gate={t.get('attn_output_gate')}")
        if t.get("attention_bias", False) or t.get("hidden_act", "silu") != "silu":
            raise ValueError("only bias-free attention/MLP with SiLU is supported")
        # Text tokens have equal temporal/height/width positions, so the interleaved multimodal RoPE
        # reduces to standard RoPE over the rotary dims.
        head_dim = t.get("head_dim", t["hidden_size"] // t["num_attention_heads"])
        return HybridDims(t["hidden_size"], t["num_hidden_layers"], t["num_attention_heads"],
                          t["num_key_value_heads"], head_dim, t["intermediate_size"], t["vocab_size"],
                          t["rms_norm_eps"], rp["rope_theta"], c.get("tie_word_embeddings", t.get("tie_word_embeddings", False)),
                          int(head_dim * rp.get("partial_rotary_factor", 1.0)), t["linear_num_key_heads"],
                          t["linear_num_value_heads"], t["linear_key_head_dim"], t["linear_value_head_dim"],
                          t["linear_conv_kernel_dim"], list(t["layer_types"]))


def weight_names(d: HybridDims) -> list[tuple[str, tuple[int, ...]]]:
    """Checkpoint tensor names and shapes the step reads, in argument order."""
    H, D, F = d.hidden, d.head_dim, d.ffn
    kd, vd = d.k_heads * d.k_dim, d.v_heads * d.v_dim
    out = []
    for i, kind in enumerate(d.kinds()):
        p = f"{d.prefix}layers.{i}."
        out.append((p + "input_layernorm.weight", (H,)))
        if kind == "linear_attention":
            a = p + "linear_attn."
            out += [(a + "in_proj_qkv.weight", (2 * kd + vd, H)), (a + "in_proj_z.weight", (vd, H)),
                    (a + "in_proj_b.weight", (d.v_heads, H)), (a + "in_proj_a.weight", (d.v_heads, H)),
                    (a + "conv1d.weight", (d.conv_dim, 1, d.conv_kernel)), (a + "A_log", (d.v_heads,)),
                    (a + "dt_bias", (d.v_heads,)), (a + "norm.weight", (d.v_dim,)), (a + "out_proj.weight", (H, vd))]
        else:
            a = p + "self_attn."
            out += [(a + "q_proj.weight", (2 * d.heads * D, H)), (a + "k_proj.weight", (d.kv_heads * D, H)),
                    (a + "v_proj.weight", (d.kv_heads * D, H)), (a + "q_norm.weight", (D,)),
                    (a + "k_norm.weight", (D,)), (a + "o_proj.weight", (H, d.heads * D))]
        out += [(p + "post_attention_layernorm.weight", (H,)), (p + "mlp.gate_proj.weight", (F, H)),
                (p + "mlp.up_proj.weight", (F, H)), (p + "mlp.down_proj.weight", (H, F))]
    out += [(d.prefix + "norm.weight", (H,)),
            (d.prefix + "embed_tokens.weight" if d.tied else "lm_head.weight", (d.vocab, H))]
    return out


def rmsnorm(x, w, eps, offset=0.0):
    """x * rsqrt(mean(x^2) + eps) * (offset + w); offset 1 is the zero-centered norm."""
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * (w + offset if offset else w)


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def softplus(u):
    """max(u, 0) + log1p(exp(-|u|)) with log1p(y), y in (0, 1], from exp only: z0 = y / (1 + y/2), then
    two Newton steps z <- z - 1 + (1 + y) exp(-z) on exp(z) = 1 + y."""
    y = torch.exp(-torch.abs(u))
    z = y / (1.0 + 0.5 * y)
    for _ in range(2):
        z = z - 1.0 + (1.0 + y) * torch.exp(-z)
    return 0.5 * (u + torch.abs(u)) + z            # max(u, 0) exactly, without a compare


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class HybridStep(torch.nn.Module):
    """act_quant: quantize the input of every linear token-wise to INT8 (the W8A8 reference)."""
    def __init__(self, d: HybridDims, tokens: int, capacity: int, act_quant: bool = False,
                 last_logits: bool = False, debug: bool = False):
        super().__init__()
        self.d, self.M, self.P, self.act_quant, self.last_logits = d, tokens, capacity, act_quant, last_logits
        self.debug = debug      # also return the first GDN layer's readout, gated norm and out_proj
        self.trace = []

    def lin(self, x, w):
        if isinstance(w, tuple):            # (INT8 weight, per-row scale): dequantize on use
            w = w[0].float() * w[1][:, None]
        elif w.dtype != torch.float32:
            w = w.float()
        return (fake_quant_rows(x) if self.act_quant else x) @ w.t()

    def gdn(self, h, conv, state, w):
        d, M = self.d, self.M
        wqkv, wz, wb, wa, wconv, a_log, dt_bias, wnorm, wout = w
        KH, VH, dk, dv, K = d.k_heads, d.v_heads, d.k_dim, d.v_dim, d.conv_kernel
        qkv = self.lin(h, wqkv)                                   # [M, C]
        z = self.lin(h, wz).reshape(M, VH, dv)
        b = self.lin(h, wb)                                       # [M, VH]
        a = self.lin(h, wa)
        # Causal depthwise conv over [the last K-1 inputs; the M new ones], token-major.
        xs = torch.cat((conv, qkv), dim=0)                        # [K-1+M, C]
        taps = wconv.reshape(d.conv_dim, K).t()                   # [K, C]
        y = xs[0:M] * taps[0]
        for j in range(1, K):
            y = y + xs[j:j + M] * taps[j]
        new_conv = xs[M:]                                         # [K-1, C]
        y = torch.nn.functional.silu(y)                           # [M, C]
        q, k, v = torch.split(y, [KH * dk, KH * dk, VH * dv], dim=-1)
        q = l2norm(q.reshape(M, KH, dk)) * (1.0 / math.sqrt(dk))
        k = l2norm(k.reshape(M, KH, dk))
        v = v.reshape(M, VH, dv)
        r = VH // KH                                              # value heads per key head
        q = q[:, :, None, :].expand(M, KH, r, dk).reshape(M, VH, dk)
        k = k[:, :, None, :].expand(M, KH, r, dk).reshape(M, VH, dk)
        beta = sigmoid(b)
        decay = torch.exp(-torch.exp(a_log.float()) * softplus(a + dt_bias.float()))   # [M, VH]
        S = state                                                 # [VH, dv, dk]
        out = []
        for t in range(M):
            S = S * decay[t][:, None, None]
            kv = (S * k[t][:, None, :]).sum(-1)                   # [VH, dv]
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + delta[:, :, None] * k[t][:, None, :]
            out.append((S * q[t][:, None, :]).sum(-1))           # [VH, dv]
        o = torch.stack(out)                                      # [M, VH, dv]
        normed = rmsnorm(o, wnorm, d.eps) * torch.nn.functional.silu(z)
        y = self.lin(normed.reshape(M, VH * dv), wout)
        if self.debug and not self.trace:
            self.trace += [o, normed, y]
        return y, new_conv, S

    def attention(self, h, c, s, mask, past_k, past_v, w):
        d, M = self.d, self.M
        D, QH, KVH, R = d.head_dim, d.heads, d.kv_heads, d.rope_dim
        G = QH // KVH
        wq, wk, wv, qn, kn, wo = w
        qg = self.lin(h, wq).reshape(M, QH, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(M, QH * D)
        q = rmsnorm(q, qn, d.eps, 1.0)
        k = rmsnorm(self.lin(h, wk).reshape(M, KVH, D), kn, d.eps, 1.0)
        v = self.lin(h, wv).reshape(M, KVH, D)
        q = torch.cat((rope(q[..., :R], c, s), q[..., R:]), dim=-1)
        k = torch.cat((rope(k[..., :R], c, s), k[..., R:]), dim=-1)
        k = k.transpose(0, 1)                                     # [KVH, M, D]
        v = v.transpose(0, 1)
        if self.P:
            K, V = torch.cat((past_k, k), dim=1), torch.cat((past_v, v), dim=1)
        else:
            K, V = k, v
        qg = q.transpose(0, 1).reshape(KVH, G * M, D)             # query rows of a KV group: head-major
        sc = (qg @ K.transpose(1, 2)) * (1.0 / D ** 0.5)
        sc = sc.reshape(KVH, G, M, -1) + mask
        p = torch.softmax(sc, dim=-1).reshape(KVH, G * M, -1)
        o = (p @ V).reshape(QH, M, D).transpose(0, 1).reshape(M, QH * D)
        return self.lin(o * sigmoid(gate), wo), k, v

    def forward(self, x, cos, sin, mask, past_k, past_v, conv, state, weights):
        d = self.d
        w = iter(weights)
        c = cos[:, None, :]; s = sin[:, None, :]                  # [M, 1, R]
        new_k, new_v, new_conv, new_state = [], [], [], []
        ia = ig = 0
        for kind in d.kinds():
            ln1 = next(w)
            h = rmsnorm(x, ln1, d.eps, 1.0)
            if kind == "linear_attention":
                y, cs, st = self.gdn(h, conv[ig], state[ig], [next(w) for _ in range(9)])
                new_conv.append(cs); new_state.append(st); ig += 1
            else:
                y, k, v = self.attention(h, c, s, mask, past_k[ia] if self.P else None,
                                         past_v[ia] if self.P else None, [next(w) for _ in range(6)])
                new_k.append(k); new_v.append(v); ia += 1
            x = x + y
            ln2, wg, wu, wd = (next(w) for _ in range(4))
            h = rmsnorm(x, ln2, d.eps, 1.0)
            x = x + self.lin(torch.nn.functional.silu(self.lin(h, wg)) * self.lin(h, wu), wd)
        fn, lm = next(w), next(w)
        if self.last_logits:
            x = x[-1:]
        logits = self.lin(rmsnorm(x, fn, d.eps, 1.0), lm)
        if self.debug:
            return logits, new_k, new_v, new_conv, new_state, list(self.trace)
        return logits, new_k, new_v, new_conv, new_state


def rope_tables(d: HybridDims, positions):
    """cos/sin [M, rope_dim] as transformers' Qwen3_5TextRotaryEmbedding (FP32 frequencies)."""
    inv = 1.0 / (d.rope_theta ** (torch.arange(0, d.rope_dim, 2, dtype=torch.float) / d.rope_dim))
    ang = torch.tensor(positions, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos(), emb.sin()
