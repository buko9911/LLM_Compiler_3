#!/usr/bin/env python3
"""Per-phase check of one decoder layer from a --debug-collect replay (M = 1).

Every NPU phase is recomputed in FP32 from the device's own phase inputs (the INT32 PIM outputs,
activation scales and residuals it read), so the error of one phase does not propagate into the
next. INT8 outputs are compared as the number of +-1 rounding flips, FP16 outputs as relative
L2 error. PIM phases must be bit exact against the device's INT8 inputs.

Phase names follow hyb-outline-phases for a Qwen3 step: layer l uses npu_{8l} (norm + quant of
its input), pim_{8l+1} (q/k/v), npu_{8l+2} (attention), pim_{8l+3} (o), npu_{8l+4} (residual +
norm + quant), pim_{8l+5} (gate/up), npu_{8l+6} (SiLU), pim_{8l+7} (down), npu_{8l+8} (residual +
next norm + quant); then pim_{8L+1} (lm_head) and npu_{8L+2} (logits). Buffers are named after
their producer phase; their role is taken from layout / dtype."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
from weights import SafetensorStore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("build", type=Path)
ap.add_argument("inputs", type=Path)
ap.add_argument("--layer", type=int, default=0)
a = ap.parse_args()
io = json.loads((a.build / "io.json").read_text())
r = np.load(a.build / "replay.npz"); z = np.load(a.inputs)
st = SafetensorStore(io["model"])
dims = io["dims"]; H, D, KVH, QH, NL = dims["hidden"], dims["head_dim"], dims["kv_heads"], dims["heads"], dims["layers"]
eps, l = dims["eps"], a.layer


def buffers(phase):
    items = [o for o in io["outputs"] if o["name"].startswith(phase + ".arg")]
    return sorted(items, key=lambda o: int(o["name"].split(".arg")[1]))


def get(o):
    return r[o["name"].replace(".", "_")]


def role(phase, what):
    for o in buffers(phase):
        if what == "qx" and o["layout"] == "pim_act":
            return get(o)[0]
        if what == "alpha" and o["layout"] == "dense" and o["dtype"] == "f32" and o["bytes"] == 4:
            return float(get(o).ravel()[0])
        if what == "resid" and o["layout"] == "dense" and o["dtype"] == "f16" and o["bytes"] == 2 * H:
            return get(o).ravel().astype(np.float64)
    raise KeyError(f"{phase}: no {what} buffer collected")


def raws(phase):
    return [get(o)[0] for o in buffers(phase) if o["layout"] == "pim_out"]


def W(n): return st.get(n).float().numpy()
def q8(w):
    # Keep the independent NumPy quantizer bounded even for a 128K-vocabulary head.
    q = np.empty(tuple(w.shape), np.int8)
    s = np.empty(w.shape[0], np.float32)
    for first in range(0, w.shape[0], 2048):
        part = w[first:first + 2048].float().numpy()
        scale = np.abs(part).max(1) / np.float32(127)
        scale = np.where(scale > 0, scale, np.float32(1e-12))
        q[first:first + len(part)] = np.clip(np.rint(part / scale[:, None]), -127, 127).astype(np.int8)
        s[first:first + len(part)] = scale
    return q, s
def rms(x, w): return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps) * w
def quant(h):
    s = np.abs(h).max() / 127.0
    return np.clip(np.rint(h / s), -127, 127), s
def rel(got, ref): return float(np.linalg.norm(np.asarray(got, np.float64) - ref) / np.linalg.norm(ref))


P = f"model.layers.{l}."
Q = {}
def qw(name):
    if name not in Q:
        Q[name] = q8(st.get(name))
    return Q[name]
def deq(name, raw, alpha): return raw[:qw(name)[0].shape[0]].astype(np.float64) * alpha * qw(name)[1]

rows = []
def check_pim(phase, names, qx):
    for raw, name in zip(raws(phase), names):
        q = qw(name)[0]
        got = raw[:q.shape[0]].astype(np.int64)
        diff = 0
        for first in range(0, q.shape[0], 2048):
            ref = q[first:first + 2048].astype(np.int64) @ qx.astype(np.int64)
            diff = max(diff, int(np.abs(got[first:first + len(ref)] - ref).max()))
        rows.append((f"{phase} {name.split('.')[-2]}", "int32 max|diff|", diff, 0))
def check_quant(phase, h):
    qx, s = quant(h)
    rows.append((f"{phase} qx", "flips", int((role(phase, "qx").astype(np.int32) != qx).sum()), h.size))
    rows.append((f"{phase} alpha", "rel", rel([role(phase, "alpha")], [s]), 0))

n = [f"{'npu' if k % 2 == 0 else 'pim'}_{8 * l + k}" for k in range(9)]
if l == 0:
    x = z["x"][0].astype(np.float64)
    check_quant(n[0], rms(x, W(P + "input_layernorm.weight")))
else:
    x = role(n[0], "resid")
qx0, a0 = role(n[0], "qx"), role(n[0], "alpha")
check_pim(n[1], [P + "self_attn.q_proj.weight", P + "self_attn.k_proj.weight", P + "self_attn.v_proj.weight"], qx0)

rq, rk, rv = raws(n[1])
q = deq(P + "self_attn.q_proj.weight", rq, a0).reshape(QH, D)
k = deq(P + "self_attn.k_proj.weight", rk, a0).reshape(KVH, D)
if dims.get("model_type", "qwen3") == "qwen3":
    q = rms(q, W(P + "self_attn.q_norm.weight"))
    k = rms(k, W(P + "self_attn.k_norm.weight"))
v = deq(P + "self_attn.v_proj.weight", rv, a0).reshape(KVH, D)
cos, sin = z["cos"][0].astype(np.float64), z["sin"][0].astype(np.float64)
def rope(t): return t * cos + np.concatenate((-t[:, D // 2:], t[:, :D // 2]), -1) * sin
q, k = rope(q), rope(k)
rows.append((f"{n[2]} new_k.{l}", "rel", rel(r[f"new_k_{l}"][:, 0], k), 0))
rows.append((f"{n[2]} new_v.{l}", "rel", rel(r[f"new_v_{l}"][:, 0], v), 0))
Kc = np.concatenate((z["K"][l], k[:, None]), 1); Vc = np.concatenate((z["V"][l], v[:, None]), 1)
mask = z["mask"][0].astype(np.float64)
o = np.zeros((QH, D))
for h in range(QH):
    g = h // (QH // KVH)
    sc = Kc[g] @ q[h] / np.sqrt(D) + mask
    p = np.exp(sc - sc.max()); p /= p.sum(); o[h] = p @ Vc[g]
check_quant(n[2], o.reshape(-1))
check_pim(n[3], [P + "self_attn.o_proj.weight"], role(n[2], "qx"))

res = x + deq(P + "self_attn.o_proj.weight", raws(n[3])[0], role(n[2], "alpha"))
rows.append((f"{n[4]} residual", "rel", rel(role(n[4], "resid"), res), 0))
resd = role(n[4], "resid")
check_quant(n[4], rms(resd, W(P + "post_attention_layernorm.weight")))
check_pim(n[5], [P + "mlp.gate_proj.weight", P + "mlp.up_proj.weight"], role(n[4], "qx"))

gt, up = (deq(P + f"mlp.{w}_proj.weight", raw, role(n[4], "alpha")) for w, raw in zip(("gate", "up"), raws(n[5])))
check_quant(n[6], gt / (1 + np.exp(-gt)) * up)
check_pim(n[7], [P + "mlp.down_proj.weight"], role(n[6], "qx"))

res2 = resd + deq(P + "mlp.down_proj.weight", raws(n[7])[0], role(n[6], "alpha"))
if l < NL - 1:
    rows.append((f"{n[8]} residual", "rel", rel(role(n[8], "resid"), res2), 0))
    check_quant(n[8], rms(role(n[8], "resid"), W(f"model.layers.{l + 1}.input_layernorm.weight")))
else:
    check_quant(n[8], rms(res2, W("model.norm.weight")))
    lm = "model.embed_tokens.weight" if dims.get("tied", True) else "lm_head.weight"
    pl, nl = f"pim_{8 * NL + 1}", f"npu_{8 * NL + 2}"
    check_pim(pl, [lm], role(n[8], "qx"))
    lg = deq(lm, raws(pl)[0], role(n[8], "alpha"))
    rows.append((f"{nl} logits", "rel", rel(r["logits"][0], lg), 0))
    rows.append((f"{nl} logits", "top-1 equal", int(np.argmax(r["logits"][0])) == int(np.argmax(lg)), 0))

bad = 0
for name, kind, val, size in rows:
    ok = {"int32 max|diff|": lambda: val == 0, "rel": lambda: val < 2e-3, "top-1 equal": lambda: bool(val),
          "flips": lambda: val <= 0.02 * size}[kind]()
    bad += not ok
    print(f"{name:34s} {kind:16s} {val}{'' if ok else '   <-- FAIL'}")
print(f"layer {l}: " + ("PASS" if not bad else f"FAIL ({bad})"))
sys.exit(1 if bad else 0)
