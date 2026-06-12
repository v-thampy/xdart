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

__all__ = [
    "BaseFrameSource",
    "FrameSource",
    "ImageFileSource",
    "LiveFrameSource",
    "MemoryFrameSource",
    "NexusStackSource",
    "ProcessedNexusSource",
    "SourceCapabilities",
    "SourceKind",
    "SourceSpec",
    "TiffSeriesSource",
    "ensure_frame_source",
    "guess_source_kind",
    "open_source",
    "register_source",
]
