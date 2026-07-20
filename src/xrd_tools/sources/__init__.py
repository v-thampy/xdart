"""Headless frame-source adapters.

Sources are the input seam for reduction, RSM, stitching, and notebooks.  They
wrap existing readers while presenting one small protocol: frame labels, lazy
frame loading, chunk iteration, metadata, and capabilities.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from xrd_tools.core.scan import (
    FrameSource,
    SourceCapabilities,
    SourceKind,
    SourceSpec,
)

if TYPE_CHECKING:
    from xrd_tools.sources.adapters import (
        SourceFormatAdapter,
        adapter_for_kind,
        all_adapters,
        get_adapter,
        register_adapter,
    )
    from xrd_tools.sources.base import BaseFrameSource, ensure_frame_source
    from xrd_tools.sources.composite import CompositeFrameSource, concat_sources
    from xrd_tools.sources.cursor import (
        ContainerCursor,
        ContainerNotReadyError,
        CursorClosedError,
        ReadBlock,
    )
    from xrd_tools.sources.descriptor import (
        ContainerDescriptor,
        describe_container,
        describe_container_from_open,
    )
    from xrd_tools.sources.metadata_provider import (
        BlueskyMetadataProvider,
        EmptyMetadataProvider,
        MetadataProvider,
    )
    from xrd_tools.sources.directory_index import (
        DEFAULT_RETRY_DEADLINE,
        DirectoryIndex,
        IndexDelta,
        RetryState,
        Snapshot,
        StaleCandidateError,
    )
    from xrd_tools.sources.directory_session import (
        CandidateObservation,
        DirectoryIndexSession,
        DirectoryObservation,
        DirectorySessionConfig,
    )
    from xrd_tools.sources.discover import Candidate, discover_scans, enumerate_candidates
    from xrd_tools.sources.run_plan import (
        RunCandidatePlan,
        RunReconcile,
        SupersededPlanError,
    )
    from xrd_tools.sources.read_plan import ReadPlan, plan_reads
    from xrd_tools.sources.grouping import flatten_scan_groups, parse_scan_groups
    from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
    from xrd_tools.sources.memory import LiveFrameSource, MemoryFrameSource
    from xrd_tools.sources.nexus import NexusStackSource, ProcessedNexusSource
    from xrd_tools.sources.probe import (
        ProbeResult,
        ProbeState,
        probe_first_frame,
        raw_is_reachable,
    )
    from xrd_tools.sources.readiness import (
        SourceReadinessObservation,
        capabilities_for_processed,
        describe_source_readiness,
        nxwriter_finalization_policy,
        observe_source_readiness,
    )
    from xrd_tools.sources.registry import (
        guess_source_kind,
        open_source,
        register_source,
    )
    from xrd_tools.sources.spec import SpecSource


_LAZY_EXPORTS = {
    "BaseFrameSource": "xrd_tools.sources.base",
    "BlueskyMetadataProvider": "xrd_tools.sources.metadata_provider",
    "Candidate": "xrd_tools.sources.discover",
    "CandidateObservation": "xrd_tools.sources.directory_session",
    "CompositeFrameSource": "xrd_tools.sources.composite",
    "ContainerCursor": "xrd_tools.sources.cursor",
    "ContainerDescriptor": "xrd_tools.sources.descriptor",
    "ContainerNotReadyError": "xrd_tools.sources.cursor",
    "CursorClosedError": "xrd_tools.sources.cursor",
    "DEFAULT_RETRY_DEADLINE": "xrd_tools.sources.directory_index",
    "DirectoryIndex": "xrd_tools.sources.directory_index",
    "DirectoryIndexSession": "xrd_tools.sources.directory_session",
    "DirectoryObservation": "xrd_tools.sources.directory_session",
    "DirectorySessionConfig": "xrd_tools.sources.directory_session",
    "EmptyMetadataProvider": "xrd_tools.sources.metadata_provider",
    "ImageFileSource": "xrd_tools.sources.image",
    "IndexDelta": "xrd_tools.sources.directory_index",
    "MetadataProvider": "xrd_tools.sources.metadata_provider",
    "ReadBlock": "xrd_tools.sources.cursor",
    "ReadPlan": "xrd_tools.sources.read_plan",
    "LiveFrameSource": "xrd_tools.sources.memory",
    "MemoryFrameSource": "xrd_tools.sources.memory",
    "NexusStackSource": "xrd_tools.sources.nexus",
    "ProbeResult": "xrd_tools.sources.probe",
    "ProbeState": "xrd_tools.sources.probe",
    "ProcessedNexusSource": "xrd_tools.sources.nexus",
    "RetryState": "xrd_tools.sources.directory_index",
    "RunCandidatePlan": "xrd_tools.sources.run_plan",
    "RunReconcile": "xrd_tools.sources.run_plan",
    "SupersededPlanError": "xrd_tools.sources.run_plan",
    "Snapshot": "xrd_tools.sources.directory_index",
    "StaleCandidateError": "xrd_tools.sources.directory_index",
    "SourceFormatAdapter": "xrd_tools.sources.adapters",
    "SourceReadinessObservation": "xrd_tools.sources.readiness",
    "SpecSource": "xrd_tools.sources.spec",
    "TiffSeriesSource": "xrd_tools.sources.image",
    "adapter_for_kind": "xrd_tools.sources.adapters",
    "all_adapters": "xrd_tools.sources.adapters",
    "concat_sources": "xrd_tools.sources.composite",
    "capabilities_for_processed": "xrd_tools.sources.readiness",
    "describe_container": "xrd_tools.sources.descriptor",
    "describe_container_from_open": "xrd_tools.sources.descriptor",
    "describe_source_readiness": "xrd_tools.sources.readiness",
    "observe_source_readiness": "xrd_tools.sources.readiness",
    "discover_scans": "xrd_tools.sources.discover",
    "enumerate_candidates": "xrd_tools.sources.discover",
    "ensure_frame_source": "xrd_tools.sources.base",
    "plan_reads": "xrd_tools.sources.read_plan",
    "flatten_scan_groups": "xrd_tools.sources.grouping",
    "get_adapter": "xrd_tools.sources.adapters",
    "guess_source_kind": "xrd_tools.sources.registry",
    "nxwriter_finalization_policy": "xrd_tools.sources.readiness",
    "open_source": "xrd_tools.sources.registry",
    "parse_scan_groups": "xrd_tools.sources.grouping",
    "probe_first_frame": "xrd_tools.sources.probe",
    "raw_is_reachable": "xrd_tools.sources.probe",
    "register_adapter": "xrd_tools.sources.adapters",
    "register_source": "xrd_tools.sources.registry",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "BaseFrameSource",
    "BlueskyMetadataProvider",
    "Candidate",
    "CandidateObservation",
    "CompositeFrameSource",
    "ContainerCursor",
    "ContainerDescriptor",
    "ContainerNotReadyError",
    "CursorClosedError",
    "DEFAULT_RETRY_DEADLINE",
    "DirectoryIndex",
    "DirectoryIndexSession",
    "DirectoryObservation",
    "DirectorySessionConfig",
    "EmptyMetadataProvider",
    "FrameSource",
    "ImageFileSource",
    "IndexDelta",
    "MetadataProvider",
    "ReadBlock",
    "ReadPlan",
    "LiveFrameSource",
    "MemoryFrameSource",
    "NexusStackSource",
    "ProbeResult",
    "ProbeState",
    "ProcessedNexusSource",
    "RetryState",
    "RunCandidatePlan",
    "RunReconcile",
    "SupersededPlanError",
    "Snapshot",
    "StaleCandidateError",
    "SourceCapabilities",
    "SourceFormatAdapter",
    "SourceReadinessObservation",
    "SourceKind",
    "SourceSpec",
    "SpecSource",
    "TiffSeriesSource",
    "adapter_for_kind",
    "all_adapters",
    "capabilities_for_processed",
    "concat_sources",
    "describe_container",
    "describe_container_from_open",
    "describe_source_readiness",
    "observe_source_readiness",
    "discover_scans",
    "enumerate_candidates",
    "ensure_frame_source",
    "plan_reads",
    "flatten_scan_groups",
    "get_adapter",
    "guess_source_kind",
    "nxwriter_finalization_policy",
    "open_source",
    "parse_scan_groups",
    "probe_first_frame",
    "raw_is_reachable",
    "register_adapter",
    "register_source",
]
