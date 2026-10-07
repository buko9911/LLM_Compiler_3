#!/usr/bin/env python3
"""Save common decode inputs for NPU/PIM comparisons, using streamed HF prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
from decoder import DecoderDims, rope_tables, step_mask
from prefill import prefill_kv
from weights import SafetensorStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--capacity", type=int, default=255)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        ap.error("output exists; use a new file")
    d = DecoderDims.from_config(a.model)
    if a.layers:
        d.layers = a.layers
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist()
    if not ids or len(ids) - 1 > a.capacity:
        ap.error("empty prompt or KV capacity too small")
    start = time.time()
    k, v = prefill_kv(a.model, d, ids[:-1], a.capacity,
                      progress=lambda i, n: print(f"host prefill layer {i}/{n}", flush=True))
    store = SafetensorStore(a.model)
    x = store.get("model.embed_tokens.weight")[ids[-1]].float()[None]
    cos, sin = rope_tables(d, [len(ids) - 1])
    mask = step_mask(1, a.capacity, len(ids) - 1)
    arrays = {name: t.numpy() for name, t in dict(x=x, cos=cos, sin=sin, mask=mask, K=k, V=v).items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **arrays)
    meta = dict(model=str(a.model.resolve()), dims=d.__dict__, tokens=1, capacity=a.capacity,
                prompt=a.prompt, prompt_token_ids=ids, valid_attention_positions=len(ids),
                attention_shape=list(mask.shape), prefill="FP32 Hugging Face decoder layers, streamed from checkpoint",
                prefill_in_device_timing=False, preparation_wall_seconds=time.time()-start,
                config_sha256=hashlib.sha256((a.model/'config.json').read_bytes()).hexdigest(),
                inputs_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                arrays={n:dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.tobytes()).hexdigest())
                        for n,t in arrays.items()})
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f"saved {a.output}: {d.layers} layers, {len(ids)} valid attention positions",flush=True)


if __name__ == "__main__":
    main()
