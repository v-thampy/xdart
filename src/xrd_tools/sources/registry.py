"""Source factory and lightweight registry.

Extension policy (how a new detector/acquisition format is added — the seam the
post-v1.1 plug-and-play source registry generalises):

1. Add a :class:`~xrd_tools.core.scan.SourceKind` member if the format is a new
   *kind* of source (``NEXUS_STACK`` already covers Bluesky/NXWriter + Eiger
   masters; ``TILED`` is reserved for a Tiled client).
2. Teach :func:`guess_source_kind` to map the URI (directory / extension /
   sniffed content) to that kind — extension-family first, content-sniff only
   for the ambiguous cases (SPEC files are extensionless, so they are sniffed).
3. Provide the opener EITHER as a built-in arm in :func:`open_source` (in-tree
   formats) OR via :func:`register_source(kind, factory)` (out-of-tree / plugin
   formats).  The registry is consulted BEFORE the built-in dispatch, so a
   registered factory OVERRIDES the built-in opener for that kind.

The seam is pinned by ``tests/core/test_source_registry_seam.py`` (H17): adding a
format is a registration + one classification arm, never a rewrite of
``open_source``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import FrameSource, SourceKind, SourceSpec, coerce_source_kind
from xrd_tools.io.image_source import ImageSourceKind, classify_image_source
from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
from xrd_tools.sources.memory import LiveFrameSource, MemoryFrameSource
from xrd_tools.sources.nexus import NexusStackSource, ProcessedNexusSource
from xrd_tools.sources.spec import SpecSource


SourceFactory = Callable[[SourceSpec], FrameSource]
_REGISTRY: dict[SourceKind, SourceFactory] = {}


def register_source(kind: SourceKind | str, factory: SourceFactory) -> None:
    _REGISTRY[coerce_source_kind(kind)] = factory


def guess_source_kind(uri: str | Path) -> SourceKind:
    path = Path(uri)
    if path.is_dir():
        return SourceKind.TIFF_SERIES
    # SSRL SPEC files are extensionless; detect by content, not suffix.  Probe
    # only no-extension (or .spec/.dat) files to avoid I/O on every image.
    if path.suffix == "" or path.suffix.lower() in {".spec", ".dat"}:
        from xrd_tools.io.spec import is_spec_file
        if is_spec_file(path):
            return SourceKind.SPEC
    info = classify_image_source(path)
    if info.kind is ImageSourceKind.PROCESSED_XDART or info.kind is ImageSourceKind.THUMBNAIL_ONLY:
        return SourceKind.PROCESSED_NEXUS
    if path.suffix.lower() in {".h5", ".hdf5", ".nxs", ".cxi"}:
        return SourceKind.NEXUS_STACK
    if path.suffix.lower() in {".tif", ".tiff"}:
        return SourceKind.IMAGE_FILE
    return SourceKind.IMAGE_FILE if info.kind is ImageSourceKind.RAW_MASTER else SourceKind.UNKNOWN


def open_source(uri_or_spec: str | Path | SourceSpec | FrameSource, **opts: Any) -> FrameSource:
    """Open a source from a URI/spec or return an existing FrameSource."""

    if hasattr(uri_or_spec, "frame_indices") and hasattr(uri_or_spec, "load_frame"):
        return uri_or_spec  # type: ignore[return-value]

    if isinstance(uri_or_spec, SourceSpec):
        spec = uri_or_spec
    else:
        kind = opts.pop("kind", SourceKind.UNKNOWN)
        if kind == SourceKind.UNKNOWN or str(kind) == SourceKind.UNKNOWN.value:
            kind = guess_source_kind(uri_or_spec)
        spec = SourceSpec(uri_or_spec, kind, options=opts)

    factory = _REGISTRY.get(coerce_source_kind(spec.kind))
    if factory is not None:
        return factory(spec)

    kind = coerce_source_kind(spec.kind)
    if kind is SourceKind.TIFF_SERIES:
        path = Path(spec.uri)
        if path.is_dir():
            return TiffSeriesSource.from_directory(path, **dict(spec.options))
        return TiffSeriesSource([path], **dict(spec.options))
    if kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
        return NexusStackSource(spec.uri, entry=spec.entry or "entry")
    if kind is SourceKind.PROCESSED_NEXUS:
        # N1: open_source(nxs, source_root=...) repoints a moved raw tree.
        return ProcessedNexusSource(
            spec.uri, entry=spec.entry or "entry",
            source_root=dict(spec.options).get("source_root"))
    if kind is SourceKind.IMAGE_FILE:
        return ImageFileSource(spec.uri, **dict(spec.options))
    if kind is SourceKind.SPEC:
        opts = dict(spec.options)
        return SpecSource(
            spec.uri, scan=opts.get("scan"), image_dir=opts.get("image_dir"),
            image_stem=opts.get("image_stem"),
            read_image_kwargs=opts.get("read_image_kwargs"))
    if kind is SourceKind.LIVE:
        return LiveFrameSource(name=str(spec.uri))
    raise ValueError(f"cannot open source {spec.uri!r} with kind {kind.value!r}")


__all__ = [
    "MemoryFrameSource",
    "LiveFrameSource",
    "ImageFileSource",
    "NexusStackSource",
    "ProcessedNexusSource",
    "SpecSource",
    "TiffSeriesSource",
    "guess_source_kind",
    "open_source",
    "register_source",
]
