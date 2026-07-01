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
    from xrd_tools.sources.base import BaseFrameSource, ensure_frame_source
    from xrd_tools.sources.composite import CompositeFrameSource, concat_sources
    from xrd_tools.sources.discover import discover_scans
    from xrd_tools.sources.grouping import flatten_scan_groups, parse_scan_groups
    from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
    from xrd_tools.sources.memory import LiveFrameSource, MemoryFrameSource
    from xrd_tools.sources.nexus import NexusStackSource, ProcessedNexusSource
    from xrd_tools.sources.probe import probe_first_frame, raw_is_reachable
    from xrd_tools.sources.readiness import (
        capabilities_for_processed,
        describe_source_readiness,
    )
    from xrd_tools.sources.registry import (
        guess_source_kind,
        open_source,
        register_source,
    )
    from xrd_tools.sources.spec import SpecSource


_LAZY_EXPORTS = {
    "BaseFrameSource": "xrd_tools.sources.base",
    "CompositeFrameSource": "xrd_tools.sources.composite",
    "ImageFileSource": "xrd_tools.sources.image",
    "LiveFrameSource": "xrd_tools.sources.memory",
    "MemoryFrameSource": "xrd_tools.sources.memory",
    "NexusStackSource": "xrd_tools.sources.nexus",
    "ProcessedNexusSource": "xrd_tools.sources.nexus",
    "SpecSource": "xrd_tools.sources.spec",
    "TiffSeriesSource": "xrd_tools.sources.image",
    "concat_sources": "xrd_tools.sources.composite",
    "capabilities_for_processed": "xrd_tools.sources.readiness",
    "describe_source_readiness": "xrd_tools.sources.readiness",
    "discover_scans": "xrd_tools.sources.discover",
    "ensure_frame_source": "xrd_tools.sources.base",
    "flatten_scan_groups": "xrd_tools.sources.grouping",
    "guess_source_kind": "xrd_tools.sources.registry",
    "open_source": "xrd_tools.sources.registry",
    "parse_scan_groups": "xrd_tools.sources.grouping",
    "probe_first_frame": "xrd_tools.sources.probe",
    "raw_is_reachable": "xrd_tools.sources.probe",
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
    "CompositeFrameSource",
    "FrameSource",
    "ImageFileSource",
    "LiveFrameSource",
    "MemoryFrameSource",
    "NexusStackSource",
    "ProcessedNexusSource",
    "SourceCapabilities",
    "SourceKind",
    "SourceSpec",
    "SpecSource",
    "TiffSeriesSource",
    "capabilities_for_processed",
    "concat_sources",
    "describe_source_readiness",
    "discover_scans",
    "ensure_frame_source",
    "flatten_scan_groups",
    "guess_source_kind",
    "open_source",
    "parse_scan_groups",
    "probe_first_frame",
    "raw_is_reachable",
    "register_source",
]
