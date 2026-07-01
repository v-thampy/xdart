"""Headless scan/frame reduction API.

This package is the public boundary intended for applications such as xdart:
the GUI chooses a :class:`ReductionPlan`, supplies a :class:`Scan`, and lets
``xrd_tools`` own the reduction work.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CancelToken",
    "CompositeSink",
    "FlushPolicy",
    "Frame",
    "FrameSource",
    "FrameReduction",
    "GI1DMode",
    "GI2DMode",
    "GIFreezeError",
    "GIMode",
    "Integration1DPlan",
    "Integration2DPlan",
    "MaskSpec",
    "MemorySink",
    "NexusSink",
    "PrepareDiagnostics",
    "ReductionPlan",
    "ReductionProgress",
    "ReductionResult",
    "ReductionSession",
    "ReductionSink",
    "Scan",
    "XYESink",
    "prepare_gi_freeze",
    "run_reduction",
    "StrictPolicy",
    "StrictnessError",
    "MissingNormalizationError",
    "GIAllDummyError",
]

_CORE_EXPORTS = {
    "CancelToken",
    "CompositeSink",
    "Frame",
    "FrameSource",
    "FrameReduction",
    "GI1DMode",
    "GI2DMode",
    "GIFreezeError",
    "GIMode",
    "Integration1DPlan",
    "Integration2DPlan",
    "MaskSpec",
    "MemorySink",
    "NexusSink",
    "PrepareDiagnostics",
    "ReductionPlan",
    "ReductionProgress",
    "ReductionResult",
    "ReductionSession",
    "ReductionSink",
    "Scan",
    "XYESink",
    "prepare_gi_freeze",
    "run_reduction",
}

_STRICTNESS_EXPORTS = {
    "StrictPolicy",
    "StrictnessError",
    "MissingNormalizationError",
    "GIAllDummyError",
}


def __getattr__(name: str) -> Any:
    if name == "FlushPolicy":
        value = getattr(import_module("xrd_tools.reduction.cadence"), name)
    elif name in _CORE_EXPORTS:
        value = getattr(import_module("xrd_tools.reduction.core"), name)
    elif name in _STRICTNESS_EXPORTS:
        value = getattr(import_module("xrd_tools.core.strictness"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
