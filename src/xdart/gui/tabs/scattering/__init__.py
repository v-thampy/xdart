"""Qt-free lifecycle kernel for the scattering workspace vNext experiment."""

from importlib import import_module


_LAZY_EXPORTS = {
    "BrowseLoadOutcome": ".browse_values",
    "BrowseLoadRequest": ".browse_values",
    "BrowseLoadStatus": ".browse_values",
    "CleanupStatus": ".events",
    "ContextController": ".context_controller",
    "ContextProjection": ".context_projection",
    "DurableFinal": ".events",
    "DurablePaused": ".events",
    "ExceptionDetail": ".start_outcomes",
    "ExecutionEnded": ".events",
    "ExecutorAccepted": ".events",
    "ExecutorStartFailed": ".events",
    "FatalExecution": ".events",
    "IllegalTransition": ".state_machine",
    "LifecycleError": ".events",
    "LifecycleResult": ".events",
    "LifecycleStatus": ".events",
    "OwnersClosed": ".events",
    "PauseRequested": ".events",
    "PreflightAccepted": ".events",
    "PreflightRefused": ".events",
    "ProjectionRequest": ".context_projection",
    "RecoveryFailure": ".start_outcomes",
    "RequestId": ".events",
    "ResumeRequested": ".events",
    "Resumed": ".events",
    "RunExecutorPort": ".contracts",
    "RunIdentity": ".events",
    "RunPhase": ".state_machine",
    "RunSignal": ".state_machine",
    "ScatteringCoordinator": ".coordinator",
    "SourceCapture": ".contracts",
    "SourcePort": ".contracts",
    "SourceSelection": ".contracts",
    "StartCapture": ".start_outcomes",
    "StartClosed": ".start_outcomes",
    "StartFailed": ".start_outcomes",
    "StartFailureKind": ".start_outcomes",
    "StartLaunched": ".start_outcomes",
    "StartPipeline": ".start_pipeline",
    "StartRecaptureCause": ".start_outcomes",
    "StartRecaptureRequired": ".start_outcomes",
    "StartRefusal": ".start_outcomes",
    "StartRefused": ".start_outcomes",
    "StartRejected": ".start_outcomes",
    "StartRejection": ".start_outcomes",
    "StopRequested": ".events",
    "transition": ".state_machine",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "CleanupStatus",
    "BrowseLoadOutcome",
    "BrowseLoadRequest",
    "BrowseLoadStatus",
    "ContextController",
    "ContextProjection",
    "DurablePaused",
    "DurableFinal",
    "ExecutorAccepted",
    "ExecutorStartFailed",
    "ExecutionEnded",
    "ExceptionDetail",
    "FatalExecution",
    "IllegalTransition",
    "LifecycleError",
    "LifecycleResult",
    "LifecycleStatus",
    "OwnersClosed",
    "PauseRequested",
    "PreflightAccepted",
    "PreflightRefused",
    "RequestId",
    "ResumeRequested",
    "Resumed",
    "RecoveryFailure",
    "RunExecutorPort",
    "RunIdentity",
    "RunPhase",
    "RunSignal",
    "SourceCapture",
    "SourcePort",
    "SourceSelection",
    "ScatteringCoordinator",
    "StartCapture",
    "StartClosed",
    "StartFailed",
    "StartFailureKind",
    "StartLaunched",
    "StartPipeline",
    "StartRecaptureCause",
    "StartRecaptureRequired",
    "StartRefusal",
    "StartRefused",
    "StartRejected",
    "StartRejection",
    "StopRequested",
    "ProjectionRequest",
    "transition",
]
