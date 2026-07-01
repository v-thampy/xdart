"""Headless frame-source adapters.

Sources are the input seam for reduction, RSM, stitching, and notebooks.  They
wrap existing readers while presenting one small protocol: frame labels, lazy
frame loading, chunk iteration, metadata, and capabilities.
"""

from xrd_tools.core.scan import (
    FrameSource,
    SourceCapabilities,
    SourceKind,
    SourceSpec,
)
from xrd_tools.sources.base import BaseFrameSource, ensure_frame_source
from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
from xrd_tools.sources.memory import LiveFrameSource, MemoryFrameSource
from xrd_tools.sources.nexus import NexusStackSource, ProcessedNexusSource
from xrd_tools.sources.registry import (
    guess_source_kind,
    open_source,
    register_source,
)
from xrd_tools.sources.composite import CompositeFrameSource, concat_sources
from xrd_tools.sources.discover import discover_scans
from xrd_tools.sources.grouping import flatten_scan_groups, parse_scan_groups
from xrd_tools.sources.spec import SpecSource

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
    "concat_sources",
    "discover_scans",
    "ensure_frame_source",
    "flatten_scan_groups",
    "guess_source_kind",
    "open_source",
    "parse_scan_groups",
    "register_source",
]
