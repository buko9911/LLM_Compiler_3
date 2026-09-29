import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from plena.capture import capture

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
ap.add_argument("--seq", type=int, default=32)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(a.model)
cfg.num_hidden_layers = a.layers
cfg.torch_dtype = torch.float16
cfg._attn_implementation = "eager"
print(f"hidden={cfg.hidden_size} ffn={cfg.intermediate_size} heads={cfg.num_attention_heads}/"
      f"{cfg.num_key_value_heads} layers={cfg.num_hidden_layers}", file=sys.stderr)

model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
layer = model.model.layers[0].eval()

hidden = torch.randn(1, a.seq, cfg.hidden_size, dtype=torch.float16)
position = torch.arange(a.seq).unsqueeze(0)
rotary = model.model.rotary_emb
cos, sin = rotary(hidden, position)
def causal(seq, dtype):
    """Additive prefill mask: 0 where a token may look, the type's floor above."""
    floor = torch.finfo(dtype).min
    return torch.full((seq, seq), floor, dtype=dtype).triu(1)[None, None]


mask = causal(a.seq, torch.float16)

class One(torch.nn.Module):
    """One decoder layer with its prefill mask and RoPE tables fixed.

    The mask is a buffer, so the capture lifts it into an argument like any other
    constant: the compiler cannot write one at run time.
    """

    def __init__(self, block, cos, sin, mask):
        super().__init__()
        self.block = block
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("mask", mask)

    def forward(self, x):
        out = self.block(x, attention_mask=self.mask,
                         position_embeddings=(self.cos, self.sin))
        return out[0] if isinstance(out, tuple) else out

res = capture(One(layer, cos, sin, mask), (hidden,))
res.write(a.out)
print(f"captured: {len(res.mlir.splitlines())} lines, {len(res.arguments)} arguments", file=sys.stderr)
