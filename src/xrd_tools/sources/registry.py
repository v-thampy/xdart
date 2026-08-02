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
from xrd_tools.sources.adapters import adapter_for_kind, candidate_owner
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


def _path_candidate_owner(uri: str | Path) -> "Any | None":
    """The adapter that claims *uri* as a name-only candidate, or ``None``.

    Name-only (suffix/stem inspection via each adapter's ``is_candidate``) — no
    file is opened and existence is not required.  Returns ``None`` for a
    virtual/non-path uri (no adapter's predicate matches) and for a directory
    or output-only ``.nexus`` (also unclaimed), so :func:`open_source` cleanly
    falls through to the kind lookup for those."""
    try:
        return candidate_owner(Path(uri))
    except Exception:
        return None


def open_source(uri_or_spec: str | Path | SourceSpec | FrameSource, **opts: Any) -> FrameSource:
    """Open a source from a URI/spec or return an existing FrameSource.

    Dispatch order: (1) a legacy :func:`register_source` factory for the kind
    (first global override); (2) R1-R8 — the adapter that CLAIMS the path, when
    compatible with the requested kind, so a discovered candidate opens through
    the same adapter that discovered and probed it; (3) the kind-owning adapter
    (:func:`~xrd_tools.sources.adapters.adapter_for_kind`) for virtual/non-path
    sources and output-only ``.nexus``; (4) the built-in if-chain fallback."""

    if hasattr(uri_or_spec, "frame_indices") and hasattr(uri_or_spec, "load_frame"):
        return uri_or_spec  # type: ignore[return-value]

    if isinstance(uri_or_spec, SourceSpec):
        spec = uri_or_spec
    else:
        kind = opts.pop("kind", SourceKind.UNKNOWN)
        if kind == SourceKind.UNKNOWN or str(kind) == SourceKind.UNKNOWN.value:
            kind = guess_source_kind(uri_or_spec)
        spec = SourceSpec(uri_or_spec, kind, options=opts)

    # 1) Legacy register_source(kind, factory): the FIRST global kind override,
    #    preserved exactly (a site can still swap an implementation by kind).
    factory = _REGISTRY.get(coerce_source_kind(spec.kind))
    if factory is not None:
        return factory(spec)

    # 2) R1-R8 path-based ownership: prefer the adapter that CLAIMS this path
    #    (its is_candidate) when that adapter is compatible with the requested
    #    kind, so discovery/probe/open all use the one adapter that owns the
    #    file.  This stops an out-of-tree adapter that declares an existing kind
    #    for an UNRELATED predicate (e.g. IMAGE_FILE for *.xyz) from hijacking
    #    the open of a .tif that the built-in image_file adapter claimed.
    #    Skipped for virtual/non-path uris (no adapter claims them -> owner None)
    #    and for output-only .nexus (excluded from candidates -> owner None),
    #    both of which fall through to the kind lookup below.
    kind = coerce_source_kind(spec.kind)
    claim_owner = _path_candidate_owner(spec.uri)
    if claim_owner is not None and kind in claim_owner.kinds:
        return claim_owner.open(spec)

    # 3) Kind lookup: the adapter that owns the requested kind (external > built-
    #    in, last-registered wins).  Built-ins are themselves registered through
    #    this seam, so for every in-tree kind this reproduces the if-chain's own
    #    construction; the chain below is a dead-but-harmless fallback for any
    #    kind nothing has adapted yet, and for virtual sources whose owner
    #    declares the kind without claiming a path.
    adapter = adapter_for_kind(spec.kind)
    if adapter is not None:
        return adapter.open(spec)

    if kind is SourceKind.TIFF_SERIES:
        path = Path(spec.uri)
        opts = dict(spec.options)
        files = opts.pop("files", ())
        opts.pop("selected_file", None)
        opts.pop("selection_mode", None)
        scan_name = opts.pop("scan_name", None)
        if files:
            opts.pop("pattern", None)
            return TiffSeriesSource(files, name=scan_name or None, **opts)
        if path.is_dir():
            return TiffSeriesSource.from_directory(path, **opts)
        return TiffSeriesSource([path], **opts)
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

#: Raw-readable NeXus-family container extensions.  ``.cxi`` is included for
#: parity with the existing guess_source_kind()/discover_scans() rules and
#: discover._NEXUS_EXTS (R1-R4); ``.nexus`` is deliberately ABSENT — it is a
#: reserved output-only extension, openable explicitly but excluded from raw
#: directory discovery (see guess_source_kind and discover.enumerate_candidates).
_NEXUS_CANDIDATE_EXTS = {".nxs", ".h5", ".hdf5", ".cxi"}


def _nexus_is_candidate(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() not in _NEXUS_CANDIDATE_EXTS:
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
    """Classify one NeXus/HDF5 candidate through the handle-aware descriptor.

    ``describe_container`` opens one short-lived handle and delegates every
    finality, processed-output, detector-layout, and frame-count decision to
    ``describe_container_from_open``.  The resulting descriptor is retained on
    the probe value instead of throwing those already-resolved facts away.
    """
    from xrd_tools.sources.descriptor import describe_container
    from xrd_tools.sources.probe import ProbeResult

    descriptor = describe_container(Path(path))
    return ProbeResult(
        descriptor.state,
        reason=descriptor.reason,
        kind=descriptor.kind,
        descriptor=descriptor,
    )


def _nexus_open(spec: SourceSpec) -> FrameSource:
    if coerce_source_kind(spec.kind) is SourceKind.PROCESSED_NEXUS:
        # N1: open_source(nxs, source_root=...) repoints a moved raw tree.
        return ProcessedNexusSource(
            spec.uri, entry=spec.entry or "entry",
            source_root=dict(spec.options).get("source_root"))
    return NexusStackSource(spec.uri, entry=spec.entry or "entry")


def _nexus_metadata_provider(path: Path) -> Any:
    """R2 fill of the nexus-family adapter's reserved metadata-provider seam.

    Opens a short-lived :class:`~xrd_tools.sources.cursor.ContainerCursor`, gets
    its lazy provider, and MATERIALIZES it (so the returned provider holds plain
    numpy/py values and reopens no master on later reads).  A Bluesky/NXWriter
    container yields a per-frame provider; every other raw stack yields the
    empty (sidecar-preserving) provider."""
    from xrd_tools.sources.cursor import ContainerCursor

    with ContainerCursor(Path(path)) as cursor:
        provider = cursor.metadata_provider()
        provider.scan_table()  # materialize while the handle is open
        return provider


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
    opts = dict(spec.options)
    files = opts.pop("files", ())
    opts.pop("selected_file", None)
    opts.pop("selection_mode", None)
    scan_name = opts.pop("scan_name", None)
    if files:
        opts.pop("pattern", None)
        return TiffSeriesSource(files, name=scan_name or None, **opts)
    if path.is_dir():
        return TiffSeriesSource.from_directory(path, **opts)
    return TiffSeriesSource([path], **opts)


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
        metadata_provider=_nexus_metadata_provider,
        finalization_policy=nxwriter_finalization_policy,
        is_output_format=True,
    ), builtin=True)
    register_adapter(SourceFormatAdapter(
        id="image_file",
        kinds=(SourceKind.IMAGE_FILE,),
        is_candidate=_image_is_candidate,
        scan_name=_image_scan_name,
        probe=_image_probe,
        open=_image_open,
    ), builtin=True)
    register_adapter(SourceFormatAdapter(
        id="tiff_series",
        kinds=(SourceKind.TIFF_SERIES,),
        is_candidate=_tiff_series_is_candidate,
        scan_name=_tiff_series_scan_name,
        probe=_tiff_series_probe,
        open=_tiff_series_open,
    ), builtin=True)
    register_adapter(SourceFormatAdapter(
        id="spec",
        kinds=(SourceKind.SPEC,),
        is_candidate=_spec_is_candidate,
        scan_name=_spec_scan_name,
        probe=_spec_probe,
        open=_spec_open,
    ), builtin=True)
    register_adapter(SourceFormatAdapter(
        id="live",
        kinds=(SourceKind.LIVE,),
        is_candidate=_live_is_candidate,
        scan_name=_live_scan_name,
        probe=_live_probe,
        open=_live_open,
    ), builtin=True)


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
