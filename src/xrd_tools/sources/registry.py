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

R1 generalizes step 3 with :mod:`xrd_tools.sources.adapters`: a
:class:`~xrd_tools.sources.adapters.SourceFormatAdapter` bundles the opener
with name-only candidate rules, a canonical scan-name function, and a content
probe, so a new format is ONE :func:`~xrd_tools.sources.adapters.register_adapter`
call that also plugs into name-only directory discovery
(:mod:`xrd_tools.sources.discover`) — not just ``open_source``.  The legacy
two-argument :func:`register_source` keeps working unchanged and is still
consulted first (see :func:`open_source`); built-in formats are themselves
registered through the adapter seam (:func:`_register_builtin_adapters`) so
one seam serves both in-tree and out-of-tree formats.  Pinned by
``tests/core/test_source_format_adapters.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import FrameSource, SourceKind, SourceSpec, coerce_source_kind
from xrd_tools.io.image_source import ImageSourceKind, classify_image_source
from xrd_tools.sources.adapters import adapter_for_kind
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
    if path.suffix.lower() == ".nexus":
        # Reserved future xdart output extension (O/H23, not yet written by any
        # producer): structurally a processed-record container, so it opens the
        # same way a processed .nxs does.  Directory discovery excludes it from
        # RAW candidates (xrd_tools.sources.discover) — this only governs
        # explicit open_source()/guess_source_kind() routing.
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

    # R1 adapter seam: consulted after the legacy register_source() override
    # (above, preserved exactly) and before the built-in if-chain below, so a
    # registered adapter opens its kind the same way a built-in one does —
    # built-ins are themselves registered through this seam (see
    # _register_builtin_adapters), so for every kind covered today this branch
    # reproduces the if-chain's own construction and the chain below becomes a
    # dead-but-harmless fallback for any kind nothing has adapted yet.
    adapter = adapter_for_kind(spec.kind)
    if adapter is not None:
        return adapter.open(spec)

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


# ---------------------------------------------------------------------------
# R1 built-in format adapters — the seam a new format (in-tree or plugin)
# declares itself through instead of editing the if-chain above (see
# xrd_tools.sources.adapters).  Each callable below reproduces EXACTLY the
# construction the if-chain already performs for that kind, so registering
# them changes nothing observable: open_source() consults the adapter
# registry before ever reaching the if-chain, which becomes a dead-but-
# harmless fallback for any kind nothing has adapted (heavy imports stay
# lazy, inside each callable, matching the rest of this package).
# ---------------------------------------------------------------------------

def _nexus_is_candidate(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() not in {".nxs", ".h5", ".hdf5"}:
        return False
    from xrd_tools.io.image import _is_eiger_master
    # An Eiger _data_NNNNNN.h5 sidecar is not its own candidate — its frames
    # are only reachable through the sibling _master.h5 (mirrors
    # xrd_tools.sources.discover.discover_scans's EIGER_MASTER branch).
    if "_data_" in path.stem and not _is_eiger_master(path):
        return False
    return True


def _nexus_scan_name(path: Path) -> str:
    """Container full-stem naming rule (the numeric suffix is part of scan
    identity; an Eiger ``_master`` tag is stripped).  Duplicates the
    container branch of xdart's ``scan_name_from_source`` (GUI layer, out of
    R1 scope to import from) so this headless adapter has no dependency on
    xdart; keep the two in agreement by inspection if that rule ever
    changes."""
    stem = Path(path).stem
    if stem.lower().endswith("_master"):
        return stem[: -len("_master")]
    return stem


def _nexus_probe(path: Path) -> Any:
    """Raw, stateless, single-shot content classification — "what does this
    file look like right now."  A young file that is unreadable or not yet
    finalized reads as IN_PROGRESS; a readable file with no detector dataset
    reads as IMAGELESS.  Whether an IMAGELESS/IN_PROGRESS verdict should still
    be SURFACED as provisional (not yet trusted as final) is the
    DirectoryIndex retry policy's job, not this function's — see
    ``directory_index.py``'s bounded-readiness handling."""
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    from xrd_tools.io.bluesky_nexus import is_unfinalized_nxwriter
    from xrd_tools.io.processed_scan_id import (
        ProcessedXdartInputError,
        is_processed_xdart_path,
    )
    from xrd_tools.io.image import _find_hdf5_image_dataset, _is_eiger_master
    import h5py

    path = Path(path)
    if is_unfinalized_nxwriter(path):
        return ProbeResult(
            ProbeState.IN_PROGRESS,
            reason="NXWriter run not yet finalized (no end_time), or unreadable",
        )
    if is_processed_xdart_path(path):
        return ProbeResult(
            ProbeState.PROCESSED_OUTPUT,
            reason="processed xdart record (integrated_1d/2d or schema stamp)",
            kind=SourceKind.PROCESSED_NEXUS,
        )
    try:
        with h5py.File(path, "r") as f:
            try:
                ds = _find_hdf5_image_dataset(f)
            except ProcessedXdartInputError:
                return ProbeResult(
                    ProbeState.PROCESSED_OUTPUT,
                    reason="processed xdart record",
                    kind=SourceKind.PROCESSED_NEXUS,
                )
            except ValueError:
                return ProbeResult(
                    ProbeState.IMAGELESS, reason="no 2-D+ detector dataset found")
            n = int(ds.shape[0]) if ds.ndim >= 3 else 1
    except OSError as exc:
        return ProbeResult(
            ProbeState.IN_PROGRESS,
            reason=f"unreadable HDF5 (treated as still-writing): {exc}",
        )
    if n <= 0:
        return ProbeResult(ProbeState.IMAGELESS, reason="zero-frame detector dataset")
    kind = SourceKind.EIGER_MASTER if _is_eiger_master(path) else SourceKind.NEXUS_STACK
    return ProbeResult(ProbeState.READY, reason="detector dataset present", kind=kind)


def _nexus_open(spec: SourceSpec) -> FrameSource:
    if coerce_source_kind(spec.kind) is SourceKind.PROCESSED_NEXUS:
        # N1: open_source(nxs, source_root=...) repoints a moved raw tree.
        return ProcessedNexusSource(
            spec.uri, entry=spec.entry or "entry",
            source_root=dict(spec.options).get("source_root"))
    return NexusStackSource(spec.uri, entry=spec.entry or "entry")


def _image_is_candidate(path: Path) -> bool:
    from xrd_tools.io.image import SUPPORTED_EXTS
    ext = Path(path).suffix.lower()
    # .h5/.hdf5/.nxs are owned by the nexus-family adapter (above); excluded
    # here even though SUPPORTED_EXTS lists them, so the two never both claim
    # the same path.
    return ext in SUPPORTED_EXTS and ext not in {".h5", ".hdf5", ".nxs"}


def _image_scan_name(path: Path) -> str:
    import re
    stem = Path(path).stem
    match = re.match(r"^(.*?)[_-](\d+)$", stem)
    return match.group(1) if match else stem


def _image_probe(path: Path) -> Any:
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    import fabio
    try:
        with fabio.open(str(path)) as f:
            n = int(getattr(f, "nframes", 1) or 0)
    except Exception as exc:
        return ProbeResult(ProbeState.INVALID, reason=f"unreadable image file: {exc}")
    if n <= 0:
        return ProbeResult(ProbeState.IMAGELESS, reason="zero frames")
    return ProbeResult(
        ProbeState.READY, reason="image file readable", kind=SourceKind.IMAGE_FILE)


def _image_open(spec: SourceSpec) -> FrameSource:
    return ImageFileSource(spec.uri, **dict(spec.options))


def _tiff_series_is_candidate(_path: Path) -> bool:
    # A TIFF series is a directory of files, not itself a file candidate;
    # directory-level grouping stays owned by discover.discover_scans /
    # sources.grouping — R1's DirectoryIndex is per-file candidate identity
    # only (see directory_index.py), so this format never claims a file.
    return False


def _tiff_series_scan_name(path: Path) -> str:
    return Path(path).name


def _tiff_series_probe(path: Path) -> Any:
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    return ProbeResult(
        ProbeState.READY, reason="tiff series directory", kind=SourceKind.TIFF_SERIES)


def _tiff_series_open(spec: SourceSpec) -> FrameSource:
    path = Path(spec.uri)
    if path.is_dir():
        return TiffSeriesSource.from_directory(path, **dict(spec.options))
    return TiffSeriesSource([path], **dict(spec.options))


def _spec_is_candidate(path: Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix == "" or suffix in {".spec", ".dat"}


def _spec_scan_name(path: Path) -> str:
    return Path(path).stem


def _spec_probe(path: Path) -> Any:
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    from xrd_tools.io.spec import is_spec_file
    if is_spec_file(Path(path)):
        return ProbeResult(
            ProbeState.READY, reason="SPEC file content detected", kind=SourceKind.SPEC)
    return ProbeResult(ProbeState.INVALID, reason="not a SPEC file")


def _spec_open(spec: SourceSpec) -> FrameSource:
    opts = dict(spec.options)
    return SpecSource(
        spec.uri, scan=opts.get("scan"), image_dir=opts.get("image_dir"),
        image_stem=opts.get("image_stem"),
        read_image_kwargs=opts.get("read_image_kwargs"))


def _live_is_candidate(_path: Path) -> bool:
    # LIVE is an in-memory acquisition-side source, never directory-discovered.
    return False


def _live_scan_name(path: Path) -> str:
    return str(path)


def _live_probe(path: Path) -> Any:
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    return ProbeResult(ProbeState.READY, reason="live source", kind=SourceKind.LIVE)


def _live_open(spec: SourceSpec) -> FrameSource:
    return LiveFrameSource(name=str(spec.uri))


def _register_builtin_adapters() -> None:
    from xrd_tools.sources.adapters import SourceFormatAdapter, register_adapter

    from xrd_tools.sources.readiness import nxwriter_finalization_policy

    register_adapter(SourceFormatAdapter(
        id="nexus_hdf5",
        kinds=(SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER, SourceKind.PROCESSED_NEXUS),
        is_candidate=_nexus_is_candidate,
        scan_name=_nexus_scan_name,
        probe=_nexus_probe,
        open=_nexus_open,
        finalization_policy=nxwriter_finalization_policy,
        is_output_format=True,
    ))
    register_adapter(SourceFormatAdapter(
        id="image_file",
        kinds=(SourceKind.IMAGE_FILE,),
        is_candidate=_image_is_candidate,
        scan_name=_image_scan_name,
        probe=_image_probe,
        open=_image_open,
    ))
    register_adapter(SourceFormatAdapter(
        id="tiff_series",
        kinds=(SourceKind.TIFF_SERIES,),
        is_candidate=_tiff_series_is_candidate,
        scan_name=_tiff_series_scan_name,
        probe=_tiff_series_probe,
        open=_tiff_series_open,
    ))
    register_adapter(SourceFormatAdapter(
        id="spec",
        kinds=(SourceKind.SPEC,),
        is_candidate=_spec_is_candidate,
        scan_name=_spec_scan_name,
        probe=_spec_probe,
        open=_spec_open,
    ))
    register_adapter(SourceFormatAdapter(
        id="live",
        kinds=(SourceKind.LIVE,),
        is_candidate=_live_is_candidate,
        scan_name=_live_scan_name,
        probe=_live_probe,
        open=_live_open,
    ))


_register_builtin_adapters()


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
