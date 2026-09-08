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
    "BoundOutputSinkGraph",
    "CompositeSink",
    "DisplayBackgroundPlan",
    "DisplayBackgroundResult",
    "FrameBackgroundPlan",
    "FrameBackgroundResult",
    "FlushPolicy",
    "Frame",
    "FrameSource",
    "FrameReduction",
    "FrameOutcome",
    "FrameOutcomeReceipt",
    "GI1DMode",
    "GI2DMode",
    "GIFreezeError",
    "GIMode",
    "Integration1DPlan",
    "Integration2DPlan",
    "MaskSpec",
    "MemorySink",
    "NexusSink",
    "NexusTerminalDisposition",
    "NexusTerminalResult",
    "OutputSinkKind",
    "OutputSinkChildrenProvider",
    "OutputSinkKindProvider",
    "PrepareDiagnostics",
    "ReductionPlan",
    "ReductionProgress",
    "ReductionResult",
    "ReductionSession",
    "ReductionSink",
    "Scan",
    "XYESink",
    "prepare_gi_freeze",
    "bind_dynamic_output_sink",
    "run_reduction",
    "run_display_background",
    "resolve_frame_background",
    "requires_active_xye_output",
    "classify_output_sink_graph",
    "supports_durable_xye_receipts",
    "StrictPolicy",
    "TransactionalXYESink",
    "StrictnessError",
    "MissingNormalizationError",
    "GIAllDummyError",
]
__all__ += ["ReintegratePlan"]
__all__ += [
    "ReintegrateRecipeMigrationRequired",
    "ReintegrateSuccessorPlan", "ReintegrateSuccessorProgress",
    "ReintegrateSuccessorResult", "run_reintegrate_successor",
]
__all__ += [
    "AdmissionCommitment", "LegacyRouteReason", "PreparedCapsuleMiss",
    "PreparedCapsuleMissCode", "PreparedDimensionAdmission",
    "PreparedDimensionPayload", "PreparedReintegrateBundle",
    "PreparedReintegrateExecution", "PreparedReintegrateOffer",
    "PreparedRouteChanged", "PreparedRouteRejected",
    "prepare_reintegrate_bundle", "unprepared_reintegrate_offer",
]
__all__ += [
    "AverageCommand", "AverageContributor",
    "AverageFiniteCounts", "AverageFiniteCountsEvidence",
    "AveragePendingPhase", "AverageRunnerPhase", "AverageScanPending",
    "AverageScanPlan", "AverageScanProgress", "AverageScanRecipe",
    "AverageScanResult", "AverageScanRunner", "iter_average_contributors",
]

_CORE_EXPORTS = {
    "CancelToken",
    "BoundOutputSinkGraph",
    "CompositeSink",
    "Frame",
    "FrameSource",
    "FrameReduction",
    "FrameOutcome",
    "FrameOutcomeReceipt",
    "GI1DMode",
    "GI2DMode",
    "GIFreezeError",
    "GIMode",
    "Integration1DPlan",
    "Integration2DPlan",
    "MaskSpec",
    "MemorySink",
    "NexusSink",
    "NexusTerminalDisposition",
    "NexusTerminalResult",
    "OutputSinkKind",
    "OutputSinkChildrenProvider",
    "OutputSinkKindProvider",
    "PrepareDiagnostics",
    "ReductionPlan",
    "ReductionProgress",
    "ReductionResult",
    "ReductionSession",
    "ReductionSink",
    "Scan",
    "TransactionalXYESink",
    "XYESink",
    "prepare_gi_freeze",
    "bind_dynamic_output_sink",
    "run_reduction",
    "requires_active_xye_output",
    "classify_output_sink_graph",
    "supports_durable_xye_receipts",
}

_STRICTNESS_EXPORTS = {
    "StrictPolicy",
    "StrictnessError",
    "MissingNormalizationError",
    "GIAllDummyError",
}

_BACKGROUND_EXPORTS = {"DisplayBackgroundPlan", "DisplayBackgroundResult", "run_display_background"}
_BACKGROUND_EXPORTS.update({"FrameBackgroundPlan", "FrameBackgroundResult", "resolve_frame_background"})
_REINTEGRATE_EXPORTS = {"ReintegratePlan"}
_REINTEGRATE_SUCCESSOR_EXPORTS = {
    "ReintegrateRecipeMigrationRequired",
    "ReintegrateSuccessorPlan", "ReintegrateSuccessorProgress",
    "ReintegrateSuccessorResult", "run_reintegrate_successor",
}
_REINTEGRATE_PREPARED_EXPORTS = {
    "AdmissionCommitment", "LegacyRouteReason", "PreparedCapsuleMiss",
    "PreparedCapsuleMissCode", "PreparedDimensionAdmission",
    "PreparedDimensionPayload", "PreparedReintegrateBundle",
    "PreparedReintegrateExecution", "PreparedReintegrateOffer",
    "PreparedRouteChanged", "PreparedRouteRejected",
    "prepare_reintegrate_bundle", "unprepared_reintegrate_offer",
}
_AVERAGE_EXPORTS = {
    "AverageCommand", "AverageContributor",
    "AverageFiniteCounts", "AverageFiniteCountsEvidence",
    "AveragePendingPhase", "AverageRunnerPhase", "AverageScanPending",
    "AverageScanPlan", "AverageScanProgress", "AverageScanRecipe",
    "AverageScanResult", "AverageScanRunner", "iter_average_contributors",
}


def __getattr__(name: str) -> Any:
    if name == "FlushPolicy":
        value = getattr(import_module("xrd_tools.session.policy"), name)
    elif name in _BACKGROUND_EXPORTS:
        value = getattr(import_module("xrd_tools.reduction.background"), name)
    elif name in _REINTEGRATE_EXPORTS:
        value = getattr(import_module("xrd_tools.reduction.reintegrate"), name)
    elif name in _REINTEGRATE_SUCCESSOR_EXPORTS:
        value = getattr(
            import_module("xrd_tools.reduction.reintegrate_successor"), name,
        )
    elif name in _REINTEGRATE_PREPARED_EXPORTS:
        value = getattr(
            import_module("xrd_tools.reduction.reintegrate_prepared"), name,
        )
    elif name in _AVERAGE_EXPORTS:
        value = getattr(import_module("xrd_tools.reduction.average"), name)
    elif name in _CORE_EXPORTS:
        value = getattr(import_module("xrd_tools.reduction.core"), name)
    elif name in _STRICTNESS_EXPORTS:
        value = getattr(import_module("xrd_tools.core.strictness"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
