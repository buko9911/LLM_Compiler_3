"""Qwen3 decode step with its KV cache as plain tensors, written for torch.export.

One step takes M new tokens against a cache of fixed capacity P. Every shape is
static, so the program has no indirect addressing:

  inputs   x         [M, H]          token embeddings (the host gathers the rows)
           cos, sin  [M, D]          rotary tables of the M positions (host)
           mask      [M, P + M]      additive attention mask, 0 or MASK_MIN (host)
           cache_k/v [KVH, P + M, D] one attention window per layer: P cached slots,
                                     then M slots the step overwrites in place with
                                     the new rows (StaticCache-style)
  outputs  x or logits               hidden state, or logits when the head is on
           new_k/v   [KVH, M, D]     one pair per layer, the same rows; the host
                                     moves them into its cached slots

Attention runs over the whole window, so P + M is what the attention tiles see.
Choose P + M as a multiple of 32. Writing the new rows into the window instead
of concatenating keeps a step from copying the whole cache.

The math is transformers' Qwen3 written out: RMSNorm in FP32, q/k/v projections,
per-head q/k RMSNorm, rotate-half RoPE, grouped-query attention (the G query
heads of one KV head attend together, no K/V repetition), softmax in FP32,
SwiGLU MLP and a tied lm_head. Parameter names follow the checkpoint with the
"model." prefix dropped.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

MASK_MIN = -30000.0  # finite in FP16, and far below any real score


@dataclass
class Config:
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

    @staticmethod
    def load(model: Path) -> "Config":
        c = json.loads((Path(model) / "config.json").read_text())
        assert c["model_type"] == "qwen3", c["model_type"]
        return Config(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                      c["num_key_value_heads"], c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                      c["intermediate_size"], c["vocab_size"], c["rms_norm_eps"], c["rope_theta"],
                      c.get("tie_word_embeddings", False))


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        t = x.to(torch.float32)
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * t.to(x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.hidden, c.heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.heads * c.head_dim, c.hidden, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)

    def forward(self, h, cos, sin, mask, cache_k, cache_v):
        c, M = self.c, h.shape[0]
        QH, KVH, D = c.heads, c.kv_heads, c.head_dim
        G = QH // KVH
        q = self.q_norm(self.q_proj(h).view(M, QH, D)).transpose(0, 1)   # [QH, M, D]
        k = self.k_norm(self.k_proj(h).view(M, KVH, D)).transpose(0, 1)  # [KVH, M, D]
        v = self.v_proj(h).view(M, KVH, D).transpose(0, 1)
        q = q * cos + rotate_half(q) * sin                                # cos/sin [M, D]
        k = k * cos + rotate_half(k) * sin
        P = cache_k.shape[1] - M
        keys = torch.slice_scatter(cache_k, k, dim=1, start=P)            # [KVH, P + M, D]
        values = torch.slice_scatter(cache_v, v, dim=1, start=P)
        # The G query heads of one KV head are consecutive, so a KV group's
        # query rows are one [G * M, D] block: head-major, then token.
        grouped = q.reshape(KVH, G * M, D)
        scores = torch.matmul(grouped, keys.transpose(1, 2)) * (D ** -0.5)
        scores = (scores.view(KVH, G, M, -1) + mask).view(KVH, G * M, -1)
        p = torch.softmax(scores.to(torch.float32), dim=-1).to(h.dtype)
        o = torch.matmul(p, values)                                       # [KVH, G * M, D]
        o = o.reshape(QH, M, D).transpose(0, 1).reshape(M, QH * D)
        return self.o_proj(o), k, v


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.up_proj = nn.Linear(c.hidden, c.ffn, bias=False)
        self.down_proj = nn.Linear(c.ffn, c.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden, c.eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, mask, cache_k, cache_v):
        a, k, v = self.self_attn(self.input_layernorm(x), cos, sin, mask, cache_k, cache_v)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), k, v


class DecodeStep(nn.Module):
    """`layers` decoder layers, then (head=True) the final norm and the lm_head."""

    def __init__(self, c: Config, layers: int, head: bool):
        super().__init__()
        self.c, self.head = c, head
        self.layers = nn.ModuleList(Layer(c) for _ in range(layers))
        if head:
            self.norm = RMSNorm(c.hidden, c.eps)
            self.lm_head = nn.Linear(c.hidden, c.vocab, bias=False)

    def forward(self, x, cos, sin, mask, *cache):
        new = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cos, sin, mask, cache[2 * i], cache[2 * i + 1])
            new += [k, v]
        if self.head:
            x = self.lm_head(self.norm(x))
        return (x, *new)


class Checkpoint:
    """Tensors of a sharded safetensors checkpoint, read one at a time."""

    _TYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def __init__(self, model: Path):
        self.model = Path(model)
        index = self.model / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() \
            else ["model.safetensors"]
        self.where = {}
        for name in files:
            with open(self.model / name, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
            for key, meta in header.items():
                if key != "__metadata__":
                    begin, end = meta["data_offsets"]
                    self.where[key] = (self.model / name, 8 + size + begin, end - begin,
                                       meta["dtype"], tuple(meta["shape"]))

    def get(self, key: str) -> torch.Tensor:
        path, offset, size, dtype, shape = self.where[key]
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(size)
        return torch.frombuffer(bytearray(raw), dtype=self._TYPES[dtype]).reshape(shape)


def load(step: DecodeStep, checkpoint: Checkpoint, first_layer: int = 0) -> None:
    """Copy checkpoint layers first_layer.. (and the head) into step, in its dtype."""
    for name, parameter in step.named_parameters():
        if name.startswith("layers."):
            index, rest = name[len("layers."):].split(".", 1)
            key = f"model.layers.{first_layer + int(index)}.{rest}"
        elif name == "lm_head.weight":
            key = "model.embed_tokens.weight" if step.c.tied else "lm_head.weight"
        else:
            key = "model." + name
        with torch.no_grad():
            parameter.copy_(checkpoint.get(key).to(parameter.dtype))


class W8A8Linear(nn.Module):
    """A linear layer under the compiler's W8A8 contract (npu-compile --quantize=w8a8).

    Weight, per output channel n:  s[n] = fl32(max_k |W[n,k]|) / 127,
                                   q = clamp(rne(W / s[n]), -127, 127), 0 where s[n] = 0.
    Activation, per row m:         alpha[m] = fl32(max_k |x[m,k]|) * fl32(1/127),
                                   qx = clamp(rne(x / alpha[m]), -127, 127), 0 where alpha[m] = 0.
    Output:                        y = fp16((fp32(qx @ q^T) * alpha[m]) * s[n]).

    x is taken at FP16, as the NPU holds it. Given the same FP16 x this is the
    compiled program's result bit for bit: the integer product is exact, and
    every FP32 step is one IEEE operation in the same order.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach().to(torch.float16).float()
        s = w.abs().amax(1) / torch.tensor(127.0)
        q = torch.where(s[:, None] > 0, torch.clamp(torch.round(w / torch.where(s > 0, s, 1.0)[:, None]), -127, 127),
                        torch.zeros_like(w))
        self.register_buffer("q", q.to(torch.int8))
        self.register_buffer("s", s)

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float16).float()
        alpha = x.abs().amax(-1, keepdim=True) * torch.tensor(1 / 127, dtype=torch.float32)
        qx = torch.where(alpha > 0, torch.clamp(torch.round(x / torch.where(alpha > 0, alpha, 1.0)), -127, 127),
                         torch.zeros_like(x))
        acc = (qx.double() @ self.q.double().t()).float()
        return ((acc * alpha) * self.s).to(torch.float16).to(dtype)


def w8a8(module: nn.Module) -> nn.Module:
    """Every nn.Linear of module replaced by its W8A8Linear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, W8A8Linear(child))
        else:
            w8a8(child)
    return module


def rope_tables(c: Config, positions) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float64) / c.head_dim))
    angles = torch.tensor(positions, dtype=torch.float64)[:, None] * inverse[None, :]
    both = torch.cat((angles, angles), dim=-1)
    return both.cos().float(), both.sin().float()


def step_mask(M: int, capacity: int, past: int) -> torch.Tensor:
    """0 for the first `past` cache slots and causally among the new tokens."""
    mask = torch.full((M, capacity + M), MASK_MIN)
    mask[:, :past] = 0.0
    for t in range(M):
        mask[t, capacity:capacity + t + 1] = 0.0
    return mask
