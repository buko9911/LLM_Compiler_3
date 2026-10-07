"""Build an FP32 host KV cache while holding only one checkpoint layer at a time.

Uses the installed Hugging Face layer implementation, independently of the
explicit-state frontend. Prefill is outside the modeled device decode time.
"""
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from decoder import DecoderDims
from weights import SafetensorStore


@torch.no_grad()
def prefill_kv(model: Path, dims: DecoderDims, token_ids, capacity: int, progress=None):
    if len(token_ids) > capacity:
        raise ValueError("prefill exceeds KV capacity")
    keys = torch.zeros(dims.layers, dims.kv_heads, capacity, dims.head_dim)
    values = torch.zeros_like(keys)
    if not len(token_ids):
        return keys, values
    if dims.model_type == "llama":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer as Layer
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Rotary
    elif dims.model_type == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as Layer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding as Rotary
    else:
        raise ValueError(f"unsupported prefill family: {dims.model_type}")
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    config.num_hidden_layers = dims.layers
    config._attn_implementation = "eager"
    store = SafetensorStore(model)
    hidden = store.get("model.embed_tokens.weight")[token_ids].float()[None]
    positions = torch.arange(len(token_ids))[None]
    rotary = Rotary(config)
    embeddings = rotary(hidden, positions)
    mask = torch.full((len(token_ids), len(token_ids)), torch.finfo(torch.float32).min).triu(1)[None, None]
    cache = DynamicCache(config=config)
    for i in range(dims.layers):
        with torch.device("meta"):
            layer = Layer(config, layer_idx=i)
        state = {name: store.get(f"model.layers.{i}.{name}").float() for name in layer.state_dict()}
        layer.load_state_dict(state, assign=True)
        layer.eval()
        hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                       position_embeddings=embeddings, past_key_values=cache, use_cache=True)
        keys[i, :, :len(token_ids)] = cache.layers[i].keys[0]
        values[i, :, :len(token_ids)] = cache.layers[i].values[0]
        del layer, state
        if progress:
            progress(i + 1, dims.layers)
    return keys, values
