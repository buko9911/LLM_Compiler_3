"""Read tensors from a (sharded) safetensors checkpoint without loading the whole model."""
from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np
import torch

_DT = {"BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4)}


class SafetensorStore:
    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        files = json.loads(idx.read_text())["weight_map"] if idx.exists() else None
        self.where = {}
        for f in sorted(set(files.values())) if files else ["model.safetensors"]:
            with open(self.dir / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name == "__metadata__": continue
                self.where[name] = (self.dir / f, 8 + n + meta["data_offsets"][0], meta["dtype"], tuple(meta["shape"]),
                                    meta["data_offsets"][1] - meta["data_offsets"][0])

    def shape(self, name): return self.where[name][3]

    def get(self, name) -> torch.Tensor:
        path, off, dt, shape, nbytes = self.where[name]
        tdt, _ = _DT[dt]
        with open(path, "rb") as fh:
            fh.seek(off); buf = fh.read(nbytes)
        return torch.frombuffer(bytearray(buf), dtype=tdt).reshape(shape)
