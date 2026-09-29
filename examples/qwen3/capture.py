"""Stage 0 for a Qwen3 decode step: 00-imported.mlir and the weights as arg<N>.bin.

The step's inputs (x, cos, sin, mask and the caches) come first, in that order,
then the parameters sorted by name. manifest.json names every argument and
result so the host knows where each tensor goes.

    python examples/qwen3/capture.py --out /tmp/qwen3-1 --layers 1
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE))
import torch
from npu.capture import capture
from qwen3 import Checkpoint, Config, DecodeStep, load

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=Path("/home/jongjip/models/qwen-3-4b"))
ap.add_argument("--layers", type=int, default=1, help="decoder layers in the step")
ap.add_argument("--first-layer", type=int, default=0, help="checkpoint layer of the first one")
ap.add_argument("--head", action="store_true", help="end with the final norm and the lm_head")
ap.add_argument("--tokens", type=int, default=1, help="M: new tokens per step")
ap.add_argument("--capacity", type=int, default=255, help="P: cache slots; P + M should be a multiple of 32")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

c = Config.load(a.model)
M, P = a.tokens, a.capacity
step = DecodeStep(c, a.layers, a.head).to(torch.float16)
load(step, Checkpoint(a.model), a.first_layer)

f16 = lambda *shape: torch.zeros(*shape, dtype=torch.float16)
inputs = [("x", f16(M, c.hidden)), ("cos", f16(M, c.head_dim)), ("sin", f16(M, c.head_dim)),
          ("mask", f16(M, P + M))]
for i in range(a.layers):
    inputs += [(f"cache_k.{i}", f16(c.kv_heads, P + M, c.head_dim)),
               (f"cache_v.{i}", f16(c.kv_heads, P + M, c.head_dim))]

result = capture(step, [t for _, t in inputs], name="decode")
result.write(a.out)
arguments = [dict(index=i, name=name, kind="input" if i < len(inputs) else "weight",
                  shape=list(shape), dtype=dtype)
             for i, (name, shape, dtype) in enumerate(
                 [(n, tuple(t.shape), "float16") for n, t in inputs] + result.arguments[len(inputs):])]
results = [dict(name="logits" if a.head else "x", shape=[M, c.vocab if a.head else c.hidden])]
for i in range(a.layers):
    results += [dict(name=f"new_k.{i}", shape=[c.kv_heads, M, c.head_dim]),
                dict(name=f"new_v.{i}", shape=[c.kv_heads, M, c.head_dim])]
(a.out / "manifest.json").write_text(json.dumps(dict(
    model=str(a.model), config=c.__dict__, layers=a.layers, first_layer=a.first_layer, head=a.head,
    tokens=M, capacity=P, arguments=arguments, results=results), indent=1))
print(f"captured {a.layers} layer(s), M={M}, P={P}: {len(result.mlir.splitlines())} lines, "
      f"{len(arguments)} arguments", file=sys.stderr)
