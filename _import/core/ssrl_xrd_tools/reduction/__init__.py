"""Headless scan/frame reduction API.

This package is the public boundary intended for applications such as xdart:
the GUI chooses a :class:`ReductionPlan`, supplies a :class:`Scan`, and lets
``ssrl_xrd_tools`` own the reduction work.
"""

from ssrl_xrd_tools.reduction.core import (
    CancelToken,
    Frame,
    FrameReduction,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    MemorySink,
    NexusSink,
    ReductionPlan,
    ReductionProgress,
    ReductionResult,
    ReductionSink,
    Scan,
    run_reduction,
)

__all__ = [
    "CancelToken",
    "Frame",
    "FrameReduction",
    "GIMode",
    "Integration1DPlan",
    "Integration2DPlan",
    "MaskSpec",
    "MemorySink",
    "NexusSink",
    "ReductionPlan",
    "ReductionProgress",
    "ReductionResult",
    "ReductionSink",
    "Scan",
    "run_reduction",
]
