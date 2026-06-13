"""Headless scan/frame reduction API.

This package is the public boundary intended for applications such as xdart:
the GUI chooses a :class:`ReductionPlan`, supplies a :class:`Scan`, and lets
``xrd_tools`` own the reduction work.
"""

from xrd_tools.reduction.cadence import FlushPolicy
from xrd_tools.reduction.core import (
    CancelToken,
    CompositeSink,
    Frame,
    FrameSource,
    FrameReduction,
    GI1DMode,
    GI2DMode,
    GIFreezeError,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    MemorySink,
    NexusSink,
    PrepareDiagnostics,
    ReductionPlan,
    ReductionProgress,
    ReductionResult,
    ReductionSession,
    ReductionSink,
    Scan,
    XYESink,
    prepare_gi_freeze,
    run_reduction,
)

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
]
