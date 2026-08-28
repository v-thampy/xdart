"""Source-kind inference and adapter-backed source opening.

The :class:`~xrd_tools.sources.adapters.SourceFormatAdapter` registry is the
only source-opening registry.  Built-in and out-of-tree formats both declare
their discovery, probing, naming, and opening behavior through one
:func:`~xrd_tools.sources.adapters.register_adapter` call.  Add a
:class:`~xrd_tools.core.scan.SourceKind` and extend :func:`guess_source_kind`
only when a new format also needs URI-based kind inference.

``open_source`` preserves explicitly typed :class:`~xrd_tools.core.scan.SourceSpec`
values and existing :class:`~xrd_tools.core.scan.FrameSource` objects.  A
processed kind first requires exact current-schema qualification.  Other paths
first honor the compatible candidate owner, keeping discovery, probing, and
opening on the same adapter; virtual sources fall through to the kind owner.
There is no separate opener registry or built-in dispatch table.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import FrameSource, SourceKind, SourceSpec, coerce_source_kind
from xrd_tools.io.image_source import ImageSourceKind, classify_image_source
from xrd_tools.sources.adapters import (
    adapter_for_kind,
    candidate_owner,
    explicit_source_owner,
)
from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
from xrd_tools.sources.memory import LiveFrameSource, MemoryFrameSource
from xrd_tools.sources.nexus import NexusStackSource, ProcessedNexusSource
from xrd_tools.sources.spec import SpecSource


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
    if path.suffix.lower() in {".h5", ".hdf5", ".nxs", ".nexus", ".cxi"}:
        return SourceKind.NEXUS_STACK
    if path.suffix.lower() in {".tif", ".tiff"}:
        return SourceKind.IMAGE_FILE
    return SourceKind.IMAGE_FILE if info.kind is ImageSourceKind.RAW_MASTER else SourceKind.UNKNOWN


def _path_candidate_owner(uri: str | Path) -> "Any | None":
    """The adapter that claims *uri* as a name-only candidate, or ``None``.

    Name-only (suffix/stem inspection via each adapter's ``is_candidate``) — no
    file is opened and existence is not required.  Returns ``None`` for a
    virtual/non-path URI, a directory, or any path no adapter claims."""
    try:
        return candidate_owner(Path(uri))
    except Exception:
        return None


def open_source(uri_or_spec: str | Path | SourceSpec | FrameSource, **opts: Any) -> FrameSource:
    """Open a source from a URI/spec or return an existing FrameSource.

    Dispatch order: current processed outputs first require strict explicit
    qualification; otherwise (1) the adapter that claims the path, when
    compatible with the requested kind, so a discovered candidate opens through
    the same owner that discovered and probed it; (2) the kind-owning adapter
    for virtual/non-path sources and explicitly typed incompatible candidates.
    An unadapted kind raises a clean :class:`ValueError`."""

    if hasattr(uri_or_spec, "frame_indices") and hasattr(uri_or_spec, "load_frame"):
        return uri_or_spec  # type: ignore[return-value]

    if isinstance(uri_or_spec, SourceSpec):
        spec = uri_or_spec
    else:
        kind = opts.pop("kind", SourceKind.UNKNOWN)
        if kind == SourceKind.UNKNOWN or str(kind) == SourceKind.UNKNOWN.value:
            kind = guess_source_kind(uri_or_spec)
        spec = SourceSpec(uri_or_spec, kind, options=opts)

    kind = coerce_source_kind(spec.kind)
    if kind is SourceKind.PROCESSED_NEXUS:
        # A caller-supplied kind and a .nexus suffix are not admission.  Current
        # processed records must pass exact schema/structure qualification
        # before any global registry or name-only adapter can construct one.
        owner = explicit_source_owner(Path(spec.uri), kind)
        if owner is None:
            raise ValueError("processed source is not a current xdart .nexus record")
        return owner.open(spec)

    # Path-based ownership: prefer the adapter that CLAIMS this path
    #    (its is_candidate) when that adapter is compatible with the requested
    #    kind, so discovery/probe/open all use the one adapter that owns the
    #    file.  This stops an out-of-tree adapter that declares an existing kind
    #    for an UNRELATED predicate (e.g. IMAGE_FILE for *.xyz) from hijacking
    #    the open of a .tif that the built-in image_file adapter claimed.
    #    Skipped for virtual/non-path URIs (no adapter claims them -> owner
    #    None); those fall through to the kind lookup below.
    claim_owner = _path_candidate_owner(spec.uri)
    if claim_owner is not None and kind in claim_owner.kinds:
        return claim_owner.open(spec)

    # Kind lookup: the adapter that owns the requested kind (external > built-in,
    # last-registered wins).  Built-ins use this same seam; there is no parallel
    # built-in dispatch table.
    adapter = adapter_for_kind(kind)
    if adapter is not None:
        return adapter.open(spec)
    raise ValueError(f"cannot open source {spec.uri!r} with kind {kind.value!r}")


# ---------------------------------------------------------------------------
# R1 built-in format adapters — the one seam a new format (in-tree or plugin)
# declares itself through (see xrd_tools.sources.adapters).  Heavy imports stay
# lazy inside the probe/provider callables.
# ---------------------------------------------------------------------------

#: Raw-readable NeXus-family container extensions.  Content probing, not the
#: suffix, separates raw ``.nexus`` from current processed output.
_NEXUS_CANDIDATE_EXTS = {".nxs", ".h5", ".hdf5", ".nexus", ".cxi"}


def _nexus_is_candidate(path: Path) -> bool:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _NEXUS_CANDIDATE_EXTS:
        return False
    prefix, marker, ordinal = path.stem.rpartition("data_")
    # Only an exact Eiger HDF5 data-sidecar name is excluded.  Legitimate
    # NeXus-family scan names may contain ``_data_`` without being sidecars.
    if (suffix == ".h5" and marker
            and (not prefix or prefix.endswith("_"))
            and len(ordinal) in {5, 6} and ordinal.isascii()
            and ordinal.isdecimal()):
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


_CBF_BINARY_STARTER = b"\x0c\x1a\x04\xd5"

_TIFF_PAYLOAD_TAG_PAIRS = ((273, 279), (324, 325))
_TIFF_INTEGER_FIELD_FORMATS = {
    1: "B",   # BYTE
    3: "H",   # SHORT
    4: "I",   # LONG
    13: "I",  # IFD
    16: "Q",  # LONG8
    18: "Q",  # IFD8
}
_TIFF_MAX_IFDS = 4096
_TIFF_MAX_TOTAL_IFD_ENTRIES = 65536
_TIFF_MAX_PAYLOAD_PARTS = 1_000_000
_TIFF_MAX_TOTAL_TAG_BYTES = 8 << 20


def _cbf_incomplete_reason(path: Path) -> str | None:
    try:
        before = path.stat()
        captured = (int(before.st_size), int(before.st_mtime_ns))
        with path.open("rb") as stream:
            header = stream.read(min(captured[0], 1 << 20))
        after = path.stat()
    except OSError as exc:
        return f"CBF is not yet stable: {exc}"
    if captured != (int(after.st_size), int(after.st_mtime_ns)):
        return "CBF changed during structural completeness probing"
    starter = header.find(_CBF_BINARY_STARTER)
    if starter < 0:
        return "CBF binary STARTER is absent within the 1 MiB header ceiling"
    size_line = next((line for line in reversed(header[:starter].splitlines())
                      if line.partition(b":")[0].strip() == b"X-Binary-Size"), None)
    value = b"" if size_line is None else size_line.partition(b":")[2].strip()
    if not value.isdigit() or len(value) > 20:
        return "CBF X-Binary-Size is absent or invalid"
    if captured[0] < starter + len(_CBF_BINARY_STARTER) + int(value):
        return "CBF binary payload is still incomplete"
    return None


def _tiff_header_incomplete_reason(stream: Any, captured_size: int) -> str | None:
    """Prove a recognized TIFF's declared pixel payload extends beyond EOF.

    This is deliberately a small, bounded classic/BigTIFF IFD reader, not an
    image decoder.  It reads only directory entries and the integer arrays
    named by StripOffsets/StripByteCounts or TileOffsets/TileByteCounts.  Any
    unknown field type, missing/duplicate tag, implausible count, or other
    ambiguity falls through to Fabio unchanged; only structural truncation is
    classified here.
    """
    stream.seek(0)
    header = stream.read(min(16, captured_size))
    if len(header) < 2:
        if not header or header in {b"I", b"M"}:
            return "TIFF header is still incomplete"
        return None
    if header[:2] not in {b"II", b"MM"}:
        return None
    endian = "<" if header[:2] == b"II" else ">"
    if len(header) < 4:
        return "TIFF header is still incomplete"
    magic = struct.unpack(endian + "H", header[2:4])[0]
    if magic == 42:
        if len(header) < 8:
            return "TIFF header is still incomplete"
        count_format, offset_format = "H", "I"
        entry_format, entry_size, inline_size = "HHI4s", 12, 4
        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
    elif magic == 43:
        if len(header) < 16:
            return "BigTIFF header is still incomplete"
        offset_size, reserved = struct.unpack(endian + "HH", header[4:8])
        if offset_size != 8 or reserved != 0:
            return None
        count_format, offset_format = "Q", "Q"
        entry_format, entry_size, inline_size = "HHQ8s", 20, 8
        ifd_offset = struct.unpack(endian + "Q", header[8:16])[0]
    else:
        return None

    count_size = struct.calcsize(count_format)
    next_size = struct.calcsize(offset_format)
    seen_ifds: set[int] = set()
    relevant_tags = {
        tag for pair in _TIFF_PAYLOAD_TAG_PAIRS for tag in pair
    }
    total_entries = 0
    total_tag_bytes = 0

    for _page in range(_TIFF_MAX_IFDS):
        if ifd_offset == 0:
            return None
        if ifd_offset in seen_ifds:
            return None
        seen_ifds.add(ifd_offset)
        if ifd_offset > captured_size - count_size:
            return "TIFF image directory is still incomplete"
        stream.seek(ifd_offset)
        raw_count = stream.read(count_size)
        if len(raw_count) != count_size:
            return "TIFF image directory is still incomplete"
        entry_count = struct.unpack(endian + count_format, raw_count)[0]
        if (entry_count < 1
                or entry_count > _TIFF_MAX_TOTAL_IFD_ENTRIES - total_entries):
            return None
        total_entries += entry_count
        entries_size = entry_count * entry_size
        directory_end = ifd_offset + count_size + entries_size + next_size
        if directory_end > captured_size:
            return "TIFF image directory is still incomplete"
        raw_entries = stream.read(entries_size)
        raw_next = stream.read(next_size)
        if len(raw_entries) != entries_size or len(raw_next) != next_size:
            return "TIFF image directory is still incomplete"

        values_by_tag: dict[int, tuple[int, ...]] = {}
        for index in range(entry_count):
            start = index * entry_size
            raw_entry = raw_entries[start:start + entry_size]
            tag, field_type, value_count, value_or_offset = struct.unpack(
                endian + entry_format, raw_entry)
            if tag not in relevant_tags:
                continue
            if tag in values_by_tag:
                return None
            value_format = _TIFF_INTEGER_FIELD_FORMATS.get(field_type)
            if value_format is None:
                return None
            if value_count < 1 or value_count > _TIFF_MAX_PAYLOAD_PARTS:
                return None
            field_size = struct.calcsize(value_format)
            value_bytes = value_count * field_size
            if value_bytes > _TIFF_MAX_TOTAL_TAG_BYTES - total_tag_bytes:
                return None
            total_tag_bytes += value_bytes
            if value_bytes <= inline_size:
                raw_values = value_or_offset[:value_bytes]
            else:
                value_offset = struct.unpack(
                    endian + offset_format, value_or_offset)[0]
                if value_offset > captured_size - value_bytes:
                    return "TIFF strip/tile metadata is still incomplete"
                position = stream.tell()
                stream.seek(value_offset)
                raw_values = stream.read(value_bytes)
                stream.seek(position)
                if len(raw_values) != value_bytes:
                    return "TIFF strip/tile metadata is still incomplete"
            values_by_tag[tag] = tuple(struct.unpack(
                endian + f"{value_count}{value_format}", raw_values))

        pairs = []
        for offset_tag, count_tag in _TIFF_PAYLOAD_TAG_PAIRS:
            offsets = values_by_tag.get(offset_tag)
            byte_counts = values_by_tag.get(count_tag)
            if (offsets is None) != (byte_counts is None):
                return None
            if offsets is not None:
                pairs.append((offsets, byte_counts))
        if len(pairs) != 1:
            return None
        offsets, byte_counts = pairs[0]
        if (not offsets or len(offsets) != len(byte_counts)
                or len(offsets) > _TIFF_MAX_PAYLOAD_PARTS):
            return None
        for offset, byte_count in zip(offsets, byte_counts):
            if byte_count < 1:
                return None
            if offset > captured_size or byte_count > captured_size - offset:
                return "TIFF pixel payload is still incomplete"

        ifd_offset = struct.unpack(endian + offset_format, raw_next)[0]

    # More IFDs than the explicit ceiling is outside this fast structural
    # proof; preserve the decoder fallback instead of guessing.
    return None


def _tiff_incomplete_reason(path: Path) -> str | None:
    try:
        before = path.stat()
        captured = (
            int(before.st_dev), int(before.st_ino),
            int(before.st_size), int(before.st_mtime_ns),
        )
        with path.open("rb") as stream:
            reason = _tiff_header_incomplete_reason(stream, captured[2])
        after = path.stat()
    except OSError as exc:
        return f"TIFF is not yet stable: {exc}"
    except (KeyError, OverflowError, struct.error, ValueError):
        # A malformed or unsupported header is not proof of truncation.  Let
        # the existing decoder retain authority over its verdict.
        return None
    current = (
        int(after.st_dev), int(after.st_ino),
        int(after.st_size), int(after.st_mtime_ns),
    )
    if current != captured:
        return "TIFF changed during structural completeness probing"
    return reason


def _image_probe(path: Path) -> Any:
    from xrd_tools.sources.probe import ProbeResult, ProbeState
    if path.suffix.lower() == ".raw":
        from xrd_tools.io.image import (
            infer_raw_detector_shape,
            read_image,
        )

        shape = infer_raw_detector_shape(path)
        if shape is None:
            # A headerless RAW file may still be landing.  Keep an unknown
            # payload size provisional so the DirectoryIndex retry window can
            # observe a later exact known-detector size; never guess geometry.
            return ProbeResult(
                ProbeState.IN_PROGRESS,
                reason="RAW payload does not yet match one known detector",
                kind=SourceKind.IMAGE_FILE,
            )
        try:
            image = read_image(
                path,
                detector_shape=shape,
                preserve_dtype=True,
            )
        except Exception as exc:
            return ProbeResult(
                ProbeState.IN_PROGRESS,
                reason=f"RAW image is not yet readable: {exc}",
                kind=SourceKind.IMAGE_FILE,
            )
        if image.ndim != 2 or tuple(image.shape) != tuple(shape):
            return ProbeResult(
                ProbeState.INVALID,
                reason="RAW image shape did not match inferred detector",
                kind=SourceKind.IMAGE_FILE,
            )
        return ProbeResult(
            ProbeState.READY,
            reason="RAW image file readable",
            kind=SourceKind.IMAGE_FILE,
        )
    if path.suffix.lower() == ".cbf":
        reason = _cbf_incomplete_reason(path)
        if reason is not None:
            return ProbeResult(
                ProbeState.IN_PROGRESS, reason=reason, kind=SourceKind.IMAGE_FILE)
    if path.suffix.lower() in {".tif", ".tiff"}:
        reason = _tiff_incomplete_reason(path)
        if reason is not None:
            return ProbeResult(
                ProbeState.IN_PROGRESS, reason=reason, kind=SourceKind.IMAGE_FILE)
    import fabio
    try:
        with fabio.open(str(path)) as f:
            n = int(getattr(f, "nframes", 1) or 0)
            import numpy as np
            pixels = np.asarray(f.data)
    except Exception as exc:
        return ProbeResult(
            ProbeState.IN_PROGRESS,
            reason=f"image decoder has no complete payload yet: {exc}",
            kind=SourceKind.IMAGE_FILE,
        )
    if n <= 0:
        return ProbeResult(ProbeState.IMAGELESS, reason="zero frames")
    if pixels.ndim != 2 or pixels.size == 0:
        return ProbeResult(
            ProbeState.IN_PROGRESS,
            reason="image file has no complete 2-D pixel payload",
            kind=SourceKind.IMAGE_FILE,
        )
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
]
