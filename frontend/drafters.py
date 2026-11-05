"""Drafter steps for speculative decoding, with explicit state, for torch.export (research: drafter + PIM measurement, 2026-11-05).

One round of each drafter as the NPU runs it. Shapes are static: C = target rows of the last verify that the drafter takes in (the rows of
the accepted tokens, tau of the last round), B = block rows, P = drafter KV cache capacity. Sampling and token lookups stay on the host, as
in every decode step of this compiler (the host gathers embedding rows and samples from the logits).

  block   DSpark / DFlash backbone: fc + hidden norm on the C context rows, then per layer Q on the B block rows, K and V on the C context and
          B block rows, attention over cache + C + B with the host's additive mask, O, MLP; final norm; LM head on `head_rows` rows.
          (deepspec/modeling/dspark/qwen3/modeling.py; models/zlab_qwen3_4b_dflash_b16/modeling_dflash.py)
          Only the K/V of the C context rows are kept: the round crops the cache to `start`.
  markov  DSpark Markov head, one step: prev [R, 256] x w2^T + the block's base logits row. A drafted token depends on the one before, so a round
          runs this step once per drafted token, one after the other (DSpark: R = 1; PCTree: R = 1 at depth 0, then R = 4 parents).
          The confidence head is left out: at threshold 0 (all recorded traces) it does not change the proposal.
  eagle_extend  EAGLE-3 after verify: fc on C rows, one layer on [embed; norm(hidden)], K/V rows of C tokens.
  eagle_lm      EAGLE-3 one draft token: norm + LM head on one row.
  eagle_layer   EAGLE-3 one draft token: the layer on one row (deepspec/modeling/eagle3/qwen3/modeling.py).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import torch

from decoder import DecoderDims, MASK_MIN, rmsnorm, rope, fake_quant_rows, K_CACHE_LAYOUT, step_cache  # noqa: F401
from weights import SafetensorStore

TAPS = [1, 9, 17, 25, 33]            # target layers the drafters read (all four checkpoints' config.json)


def dims_of(model: Path) -> DecoderDims:
    return DecoderDims.from_config(model)


def lin(x, w, act_quant=False):
    if isinstance(w, tuple):                 # (INT8 weight, per-row scale): reference path
        w = w[0].float() * w[1][:, None]
    elif w.dtype != torch.float32:
        w = w.float()
    return (fake_quant_rows(x) if act_quant else x) @ w.t()


def attention(d, q, k, v, past_k, past_v, mask, P, act_quant=False):
    """q [M, QH, D] (RoPE done), k and v [KVH, R, D] with R >= M rows (RoPE done): grouped-query attention over cache + R rows, as decoder.py."""
    M = q.shape[0]
    QH, KVH, D = d.heads, d.kv_heads, d.head_dim
    G = QH // KVH
    if P:
        KT = torch.cat((past_k, k.transpose(1, 2)), dim=2)
        V = torch.cat((past_v, v), dim=1)
    else:
        KT, V = k.transpose(1, 2), v
    qg = q.transpose(0, 1).reshape(KVH, G * M, D)
    sc = (qg @ KT) * (1.0 / D ** 0.5)
    sc = sc.reshape(KVH, G, M, -1) + mask
    p = torch.softmax(sc, dim=-1).reshape(KVH, G * M, -1)
    return (p @ V).reshape(QH, M, D).transpose(0, 1).reshape(M, QH * D)


def block_weight_names(d, head_rows, prefix="", lm_name="lm_head.weight"):
    H, D, F = d.hidden, d.head_dim, d.ffn
    out = [("fc.weight", (H, 5 * H)), ("hidden_norm.weight", (H,))]
    for i in range(d.layers):
        p = f"layers.{i}."
        out += [(p + "input_layernorm.weight", (H,)),
                (p + "self_attn.q_proj.weight", (d.heads * D, H)), (p + "self_attn.k_proj.weight", (d.kv_heads * D, H)),
                (p + "self_attn.v_proj.weight", (d.kv_heads * D, H)),
                (p + "self_attn.q_norm.weight", (D,)), (p + "self_attn.k_norm.weight", (D,)),
                (p + "self_attn.o_proj.weight", (H, d.heads * D)), (p + "post_attention_layernorm.weight", (H,)),
                (p + "mlp.gate_proj.weight", (F, H)), (p + "mlp.up_proj.weight", (F, H)), (p + "mlp.down_proj.weight", (H, F))]
    out += [("norm.weight", (H,)), (lm_name, (d.vocab, H))]
    return out


class BlockStep(torch.nn.Module):
    """DSpark / DFlash drafter backbone and LM head for one round."""
    def __init__(self, d, B, C, P, head_rows, act_quant=False):
        super().__init__()
        self.d, self.B, self.C, self.P, self.head_rows, self.act_quant = d, B, C, P, head_rows, act_quant

    def forward(self, x, ctx, cos_c, sin_c, cos, sin, mask, past_k, past_v, weights):
        d, B, C = self.d, self.B, self.C
        H, D, QH, KVH = d.hidden, d.head_dim, d.heads, d.kv_heads
        aq = self.act_quant
        w = iter(weights)
        wfc, hn = next(w), next(w)
        t = rmsnorm(lin(ctx, wfc, aq), hn, d.eps)
        cc, sc = cos_c[:, None, :], sin_c[:, None, :]        # [C, 1, D], the host's tables of the context rows
        cq, sq = cos[:, None, :], sin[:, None, :]            # [B, 1, D], of the block rows
        new_k, new_v = [], []
        for i in range(d.layers):
            ln1, wq, wk, wv, qn, kn, wo, ln2, wg, wu, wd = (next(w) for _ in range(11))
            h = rmsnorm(x, ln1, d.eps)
            q = rmsnorm(lin(h, wq, aq).reshape(B, QH, D), qn, d.eps)
            # K and V of the context rows and of the block rows are separate tensors: RMSNorm and RoPE act row by row, so this is the checkpoint's
            # concat-then-normalize, and the kept context rows are a result of their own (no copy out of a larger buffer).
            kc = rope(rmsnorm(lin(t, wk, aq).reshape(C, KVH, D), kn, d.eps), cc, sc).transpose(0, 1)
            kb = rope(rmsnorm(lin(h, wk, aq).reshape(B, KVH, D), kn, d.eps), cq, sq).transpose(0, 1)
            vc = lin(t, wv, aq).reshape(C, KVH, D).transpose(0, 1)
            vb = lin(h, wv, aq).reshape(B, KVH, D).transpose(0, 1)
            q = rope(q, cq, sq)
            k = torch.cat((kc, kb), 1)                       # [KVH, C + B, D]
            v = torch.cat((vc, vb), 1)
            new_k.append(kc); new_v.append(vc)
            o = attention(d, q, k, v, past_k[i] if self.P else None, past_v[i] if self.P else None, mask, self.P)
            x = x + lin(o, wo, aq)
            h = rmsnorm(x, ln2, d.eps)
            x = x + lin(torch.nn.functional.silu(lin(h, wg, aq)) * lin(h, wu, aq), wd, aq)
        fn, lm = next(w), next(w)
        h = rmsnorm(x, fn, d.eps)
        if self.head_rows != B:
            h = h[-self.head_rows:]
        return lin(h, lm, aq), new_k, new_v


class MarkovStep(torch.nn.Module):
    """One Markov step: base logits row(s) + prev @ w2^T."""
    def __init__(self, act_quant=False):
        super().__init__()
        self.act_quant = act_quant

    def forward(self, prev, base, w2):
        return base + lin(prev, w2, self.act_quant)


def eagle_weight_names(d):
    H, D, F = d.hidden, d.head_dim, d.ffn
    p = "layers.0."
    return [(p + "hidden_norm.weight", (H,)), (p + "input_layernorm.weight", (H,)),
            (p + "self_attn.q_proj.weight", (d.heads * D, 2 * H)), (p + "self_attn.k_proj.weight", (d.kv_heads * D, 2 * H)),
            (p + "self_attn.v_proj.weight", (d.kv_heads * D, 2 * H)),
            (p + "self_attn.q_norm.weight", (D,)), (p + "self_attn.k_norm.weight", (D,)),
            (p + "self_attn.o_proj.weight", (H, d.heads * D)), (p + "post_attention_layernorm.weight", (H,)),
            (p + "mlp.gate_proj.weight", (F, H)), (p + "mlp.up_proj.weight", (F, H)), (p + "mlp.down_proj.weight", (H, F))]


class EagleLayer(torch.nn.Module):
    """The EAGLE-3 layer on M rows; fc first when the input is the 5 x H target feature (extend)."""
    def __init__(self, d, M, P, extend, act_quant=False):
        super().__init__()
        self.d, self.M, self.P, self.extend, self.act_quant = d, M, P, extend, act_quant

    def forward(self, hidden, embed, cos, sin, mask, past_k, past_v, weights):
        d, M = self.d, self.M
        H, D, QH, KVH = d.hidden, d.head_dim, d.heads, d.kv_heads
        aq = self.act_quant
        w = iter(weights)
        if self.extend:
            hidden = lin(hidden, next(w), aq)
        hn, iln, wq, wk, wv, qn, kn, wo, ln2, wg, wu, wd = (next(w) for _ in range(12))
        c = cos[:, None, :]; s = sin[:, None, :]
        x = hidden
        z = torch.cat((rmsnorm(embed, iln, d.eps), rmsnorm(x, hn, d.eps)), dim=-1)
        q = rope(rmsnorm(lin(z, wq, aq).reshape(M, QH, D), qn, d.eps), c, s)
        k = rope(rmsnorm(lin(z, wk, aq).reshape(M, KVH, D), kn, d.eps), c, s).transpose(0, 1)
        v = lin(z, wv, aq).reshape(M, KVH, D).transpose(0, 1)
        o = attention(d, q, k, v, past_k[0], past_v[0], mask, self.P)
        x = x + lin(o, wo, aq)
        h = rmsnorm(x, ln2, d.eps)
        x = x + lin(torch.nn.functional.silu(lin(h, wg, aq)) * lin(h, wu, aq), wd, aq)
        return x, [k], [v]


class EagleLm(torch.nn.Module):
    def __init__(self, d, act_quant=False):
        super().__init__()
        self.d, self.act_quant = d, act_quant

    def forward(self, hidden, fn, lm):
        return lin(rmsnorm(hidden, fn, self.d.eps), lm, self.act_quant)
