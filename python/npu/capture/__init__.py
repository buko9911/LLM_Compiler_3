"""Stage 0: pull a computation graph out of a model. No IR text is written by hand.

The compiler is value-free -- the ISA has no instruction that writes a register
into memory -- so every weight, table and initial value leaves here as bytes and
is named by the package, never baked into the program.
"""
from .export import CaptureResult, capture, capture_callable

__all__ = ["CaptureResult", "capture", "capture_callable"]
