# -*- coding: utf-8 -*-
"""``xrd_tools.session`` - the headless scan-session layer.

The public session classes are loaded lazily so importing a lightweight
submodule such as ``xrd_tools.session.readiness`` does not pull in reduction
writers or image-reader dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ScanSession",
    "FrameEvent",
    "FrameRecordStore",
    "ProgressEvent",
    "StateChangeEvent",
    # X1 store/provider read-authority projections (headless):
    "MetadataRow",
    "Capability",
    "CapabilityState",
    "CapabilityDisposition",
    "DisplayCapabilities",
    "WavelengthEvidence",
    "WavelengthStatus",
    "FrameProjection",
    "MetadataConflictError",
    "metadata_row_from_view",
    "metadata_row_from_provider",
    "metadata_row_from_record",
    "normalization_channels",
    "normalization_value",
    "wavelength_evidence",
    "display_capabilities",
    "project_frame",
    # Immutable run-boundary configuration values (headless):
    "GIIntent",
    "ThresholdIntent",
    "RunIntent",
    "FrozenGIConfiguration",
    "FrozenThresholdPolicy",
    "FrozenSourceSpec",
    "FrozenRunConfiguration",
    "RunConfigurationRefused",
    "admit_run_configuration",
    "require_run_configuration",
    "jsonable_run_value",
    # Canonical revisioned RunIntent owner (headless):
    "RunIntentSnapshot",
    "IntentCommitAccepted",
    "IntentFreezeAccepted",
    "IntentRecaptureRequired",
    "RunIntentStore",
    # GI theta-motor policy (Qt-free, single source of truth).  `resolve_gi_motor`
    # is THE motor-resolution decision (raw selection + choice knowledge ->
    # effective motor); `pick_default_gi_motor` is the default-pick it delegates
    # to.  GUI owners import the resolver rather than re-deriving the rule.
    "GI_MOTOR_PREFERENCE",
    "pick_default_gi_motor",
    "resolve_gi_motor",
    # Slice-5 whole-scan normalization aggregate (headless value):
    "ScanNormAggregate",
    "accepts_norm_aggregate",
    "channel_is_partial",
    "empty_norm_aggregate",
    "fold_norm_metadata",
    "next_norm_revision",
    # Revision-qualified hydration request/completion values (headless):
    "HydrationPurpose",
    "HydrationOutcome",
    "HydrationScope",
    "HydrationReadKey",
    "HydrationToken",
    "HydrationCompletion",
    "normalize_hydration_purpose",
]

_SCAN_SESSION_EXPORTS = {
    "ScanSession",
    "FrameEvent",
    "ProgressEvent",
    "StateChangeEvent",
}

_FRAME_PROJECTION_EXPORTS = {
    "MetadataRow",
    "Capability",
    "CapabilityState",
    "CapabilityDisposition",
    "DisplayCapabilities",
    "WavelengthEvidence",
    "WavelengthStatus",
    "FrameProjection",
    "MetadataConflictError",
    "metadata_row_from_view",
    "metadata_row_from_provider",
    "metadata_row_from_record",
    "normalization_channels",
    "normalization_value",
    "wavelength_evidence",
    "display_capabilities",
    "project_frame",
}

_RUN_CONFIGURATION_EXPORTS = {
    "GIIntent",
    "ThresholdIntent",
    "RunIntent",
    "FrozenGIConfiguration",
    "FrozenThresholdPolicy",
    "FrozenSourceSpec",
    "FrozenRunConfiguration",
    "RunConfigurationRefused",
    "admit_run_configuration",
    "require_run_configuration",
    "jsonable_run_value",
    "resolve_gi_motor",
}


_INTENT_STORE_EXPORTS = {
    "RunIntentSnapshot",
    "IntentCommitAccepted",
    "IntentFreezeAccepted",
    "IntentRecaptureRequired",
    "RunIntentStore",
}


_GI_MOTOR_EXPORTS = {
    "GI_MOTOR_PREFERENCE",
    "pick_default_gi_motor",
}


_SCAN_NORM_EXPORTS = {
    "ScanNormAggregate",
    "accepts_norm_aggregate",
    "channel_is_partial",
    "empty_norm_aggregate",
    "fold_norm_metadata",
    "next_norm_revision",
}


_HYDRATION_EXPORTS = {
    "HydrationPurpose",
    "HydrationOutcome",
    "HydrationScope",
    "HydrationReadKey",
    "HydrationToken",
    "HydrationCompletion",
    "normalize_hydration_purpose",
}


def __getattr__(name: str) -> Any:
    if name == "FrameRecordStore":
        value = getattr(import_module("xrd_tools.session.frame_record_store"), name)
    elif name in _GI_MOTOR_EXPORTS:
        value = getattr(import_module("xrd_tools.session.gi_motor"), name)
    elif name in _SCAN_SESSION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.scan_session"), name)
    elif name in _FRAME_PROJECTION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.frame_projection"), name)
    elif name in _RUN_CONFIGURATION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.run_configuration"), name)
    elif name in _INTENT_STORE_EXPORTS:
        value = getattr(import_module("xrd_tools.session.intent_store"), name)
    elif name in _HYDRATION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.hydration"), name)
    elif name in _SCAN_NORM_EXPORTS:
        value = getattr(import_module("xrd_tools.session.scan_norm"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
