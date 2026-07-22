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


def __getattr__(name: str) -> Any:
    if name == "FrameRecordStore":
        value = getattr(import_module("xrd_tools.session.frame_record_store"), name)
    elif name in _SCAN_SESSION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.scan_session"), name)
    elif name in _FRAME_PROJECTION_EXPORTS:
        value = getattr(import_module("xrd_tools.session.frame_projection"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
