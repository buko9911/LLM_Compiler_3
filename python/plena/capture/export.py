"""torch.export -> torch-mlir -> linalg, with the parameters lifted out.

Decomposition is ON: the pipeline lowers general kernels rather than recognising
patterns, so linalg is the right thing to receive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.func import functional_call


@dataclass
class CaptureResult:
    """One captured entry point: its linalg IR and the bytes its arguments need."""

    mlir: str
    #: One entry per function argument, in order: (name, shape, dtype string).
    arguments: list[tuple[str, tuple[int, ...], str]] = field(default_factory=list)
    #: Argument index -> the tensor that belongs in it. Activations are absent.
    contents: dict[int, torch.Tensor] = field(default_factory=dict)

    def write(self, directory: Path) -> None:
        """Save the IR and every known argument as arg<N>.bin, little-endian."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "00-imported.mlir").write_text(self.mlir)
        for index, tensor in sorted(self.contents.items()):
            raw = tensor.detach().contiguous().cpu().numpy()
            (directory / f"arg{index}.bin").write_bytes(raw.tobytes())
        lines = [f"{i} {name} {'x'.join(map(str, shape))} {dtype}"
                 for i, (name, shape, dtype) in enumerate(self.arguments)]
        (directory / "arguments.txt").write_text("\n".join(lines) + "\n")


def _strip_non_persistent(module: torch.nn.Module) -> None:
    """RoPE keeps inv_freq as a non-persistent buffer, which export cannot map."""
    for submodule in module.modules():
        buffers = getattr(submodule, "_non_persistent_buffers_set", None)
        if buffers:
            buffers.clear()


def capture_callable(function: Callable[..., Any], example: Sequence[torch.Tensor],
                     *, name: str = "main") -> str:
    """Import one plain callable as linalg IR."""
    from torch_mlir import fx

    exported = torch.export.export(_Wrapped(function), tuple(example), strict=False)
    module = fx.export_and_import(exported, output_type="linalg-on-tensors",
                                  func_name=name, allow_non_finites=True)
    return str(module)


class _Wrapped(torch.nn.Module):
    def __init__(self, function: Callable[..., Any]):
        super().__init__()
        self._function = function

    def forward(self, *args):  # noqa: D102 - trivial passthrough
        return self._function(*args)


def capture(model: torch.nn.Module, example: Sequence[torch.Tensor],
            *, name: str = "main", dtype: torch.dtype = torch.float16) -> CaptureResult:
    """Capture a model with its parameters lifted into function arguments.

    Registering the parameters instead would inline them: an 8B model's IR reaches
    16 GB. They become ordinary arguments here and travel as bytes beside the IR.
    """
    model = model.to(dtype).eval()
    _strip_non_persistent(model)
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))
    order = sorted(named)

    class Functional(torch.nn.Module):
        def forward(self, *values):
            activations = values[:len(example)]
            parameters = dict(zip(order, values[len(example):]))
            return functional_call(model, parameters, tuple(activations))

    weights = [named[key].detach() for key in order]
    inputs = tuple(list(example) + weights)
    exported = torch.export.export(Functional(), inputs, strict=False)

    from torch_mlir import fx
    module = fx.export_and_import(exported, output_type="linalg-on-tensors",
                                  func_name=name, allow_non_finites=True)

    arguments: list[tuple[str, tuple[int, ...], str]] = []
    for index, tensor in enumerate(example):
        arguments.append((f"input{index}", tuple(tensor.shape), str(tensor.dtype).removeprefix("torch.")))
    for key, tensor in zip(order, weights):
        arguments.append((key, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch.")))
    contents = {len(example) + i: tensor for i, tensor in enumerate(weights)}
    return CaptureResult(mlir=str(module), arguments=arguments, contents=contents)
