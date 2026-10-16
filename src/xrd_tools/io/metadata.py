"""Metadata parsing: read_txt_metadata, read_pdi_metadata, read_image_metadata."""
from __future__ import annotations

import configparser
import hashlib
import logging
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from xrd_tools.io.stat_identity import identity_ctime_ns

logger = logging.getLogger(__name__)

__all__ = [
    "ImageMetadataRead",
    "MetadataDiscoveryChanged",
    "MetadataInputTooLarge",
    "MetadataScratchCleanupFailed",
    "read_txt_metadata",
    "read_pdi_metadata",
    "read_image_metadata",
    "read_image_metadata_observed",
]

MetadataValue = int | float | str
MetadataSourceRevision = tuple[int, int, int, int, int, int]


class MetadataInputTooLarge(ValueError):
    """A selected metadata input exceeded its admitted byte bound."""


class MetadataScratchCleanupFailed(RuntimeError):
    """A bounded SPEC snapshot could not be closed or removed."""


class MetadataSourceChanged(OSError):
    """A metadata file changed while it was being read.

    A recorded candidate or the selected source changed between the reader's
    record of it and its check, or its bytes moved under a bounded read.  The
    message is the reader's own refusal text, and ``metadata_path`` names the
    file or directory that changed.  Source qualification raises it as a typed
    source-revision change, so a live Run can read again at its next poll.
    """

    def __init__(self, message: str, *, path: Path | str | None = None) -> None:
        super().__init__(message)
        self.metadata_path = None if path is None else Path(path)


class MetadataDiscoveryChanged(MetadataSourceChanged):
    """Only directories that discovery lists changed during a read.

    A file was added, removed or renamed in one, as a live acquisition does at
    any time.  No candidate file and no selected source changed, so the read
    can simply be repeated; :func:`read_image_metadata_observed` does, a
    bounded number of times.
    """


@dataclass(frozen=True, slots=True)
class ImageMetadataRead:
    """One metadata result plus the exact file that supplied it.

    ``source_path`` is ``None`` when discovery consumed no accepted metadata
    file.  In particular, Auto reports the candidate whose parser succeeded,
    not merely the first companion name that existed.
    """

    values: Mapping[str, MetadataValue]
    source_path: Path | None
    source_revision: MetadataSourceRevision | None = None
    discovery_revisions: tuple[
        tuple[Path, MetadataSourceRevision | None], ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(dict(self.values)),
        )
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))
        discovery_revisions = tuple(
            (Path(path), revision)
            for path, revision in self.discovery_revisions
        )
        if len({path for path, _revision in discovery_revisions}) != len(
            discovery_revisions
        ):
            raise ValueError("discovery revisions must have unique paths")
        for _path, observed_revision in discovery_revisions:
            if observed_revision is not None:
                _validate_metadata_revision(observed_revision)
        object.__setattr__(
            self, "discovery_revisions", discovery_revisions,
        )
        revision = self.source_revision
        if revision is not None:
            if self.source_path is None:
                raise ValueError("source_revision requires source_path")
            _validate_metadata_revision(revision)


def _validate_metadata_revision(revision: object) -> None:
    if (
        type(revision) is not tuple
        or len(revision) != 6
        or any(type(value) is not int or value < 0 for value in revision)
    ):
        raise TypeError(
            "metadata revision must be six nonnegative exact integers"
        )


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _metadata_discovery_paths(
    image_path: Path, meta_format: str, meta_dir: Path | str | None,
) -> tuple[Path, ...]:
    if meta_format == "spec":
        candidates = (
            *((_lexical_absolute(meta_dir),) if meta_dir not in (None, "") else ()),
            image_path.parent,
            image_path.parent.parent,
            image_path.parent.parent.parent,
        )
    else:
        candidates = (image_path.parent,)
    ordered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        lexical = _lexical_absolute(candidate)
        if lexical not in seen:
            seen.add(lexical)
            ordered.append(lexical)
    return tuple(ordered)


def _metadata_path_revision(
    path: Path,
) -> MetadataSourceRevision | None:
    try:
        state = path.stat()
    except OSError:
        return None
    return (
        int(state.st_mode), int(state.st_dev), int(state.st_ino),
        int(state.st_size), int(state.st_mtime_ns),
        identity_ctime_ns(state.st_ctime_ns),
    )


def _metadata_discovery_revisions(
    image_path: Path, meta_format: str, meta_dir: Path | str | None,
) -> tuple[tuple[Path, MetadataSourceRevision | None], ...]:
    paths = list(_metadata_discovery_paths(
        image_path, meta_format, meta_dir,
    ))
    if meta_format == "auto":
        paths.extend(
            candidate
            for candidate, _convention, _suffix
            in _iter_auto_sidecar_candidates(image_path)
        )
    ordered: list[tuple[Path, MetadataSourceRevision | None]] = []
    seen: set[Path] = set()
    for candidate in paths:
        lexical = _lexical_absolute(candidate)
        if lexical in seen:
            continue
        seen.add(lexical)
        ordered.append((lexical, _metadata_path_revision(lexical)))
    return tuple(ordered)


def _is_directory_revision(revision: MetadataSourceRevision | None) -> bool:
    return revision is not None and stat.S_ISDIR(revision[0])


def _with_discovery_revisions(
    result: ImageMetadataRead,
    discovery_revisions: tuple[
        tuple[Path, MetadataSourceRevision | None], ...
    ],
) -> ImageMetadataRead:
    if result.source_path is not None:
        if (
            result.source_revision is None
            or _metadata_path_revision(result.source_path)
            != result.source_revision
        ):
            raise MetadataSourceChanged(
                f"metadata source changed during read: {result.source_path}",
                path=result.source_path,
            )
    combined = (*discovery_revisions, *result.discovery_revisions)
    ordered: list[tuple[Path, MetadataSourceRevision | None]] = []
    seen: dict[Path, MetadataSourceRevision | None] = {}
    changed: list[tuple[
        Path, MetadataSourceRevision | None, MetadataSourceRevision | None,
    ]] = []
    for path, revision in combined:
        if path in seen:
            if seen[path] != revision:
                changed.append((path, seen[path], revision))
            continue
        seen[path] = revision
        ordered.append((path, revision))
    unique = tuple(ordered)
    for path, revision in unique:
        current = _metadata_path_revision(path)
        if current != revision:
            changed.append((path, revision, current))
    if changed:
        message = f"metadata discovery changed during read: {changed[0][0]}"
        # A directory that stayed a directory only gained, lost or renamed an
        # entry; any other change is refused at once.
        if all(
            _is_directory_revision(before) and _is_directory_revision(after)
            for _path, before, after in changed
        ):
            raise MetadataDiscoveryChanged(message, path=changed[0][0])
        raise MetadataSourceChanged(message, path=changed[0][0])
    return ImageMetadataRead(
        result.values,
        result.source_path,
        result.source_revision,
        unique,
    )

_STRUCTURED_SIDECAR_MIN_PAIRS = 3
_AUTO_SIDECAR_CACHE: dict[tuple[Path, str], tuple[str, str]] = {}
# BL-3: auto sidecar discovery considers ONLY these extensions, in this
# priority order.  Before this, ANY non-image companion (a per-frame ``.poni``
# with colon pairs, an ``img.tif.log``, pretty-printed JSON) could latch as the
# metadata sidecar and poison scan_data — and it sorted alphabetically, so junk
# beat the real ``.txt``.  Explicit ``meta_format`` bypasses this allow-list.
_AUTO_SIDECAR_EXTENSIONS = ("txt", "pdi", "metadata", "inf")
# A metadata sidecar is small; anything larger is not one (and reading it wastes
# I/O + risks parsing binary garbage into ≥3 spurious pairs).
_AUTO_SIDECAR_MAX_BYTES = 1 << 20  # 1 MiB
_IMAGE_EXTENSIONS = {
    ".bmp",
    ".cbf",
    ".edf",
    ".gif",
    ".h5",
    ".hdf5",
    ".img",
    ".jpeg",
    ".jpg",
    ".nexus",
    ".npy",
    ".nxs",
    ".png",
    ".raw",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_kv_pairs(text: str, delimiters: str = ",|=") -> dict[str, float]:
    """Split *text* on *delimiters* and return alternating key/value pairs.

    Parameters
    ----------
    text : str
        Raw text containing interleaved key and value tokens.
    delimiters : str
        Regex alternation of delimiter characters (default ``",|="``).

    Returns
    -------
    dict[str, float]
        Mapping of key (first word) to float value.  Pairs where the value
        cannot be converted to float are silently skipped.
    """
    tokens = re.split(delimiters, text)
    result: dict[str, float] = {}
    for key_tok, val_tok in zip(tokens[::2], tokens[1::2]):
        key = key_tok.split()[0] if key_tok.split() else None
        if key is None:
            continue
        try:
            result[key] = float(val_tok)
        except (ValueError, TypeError):
            pass
    return result


def _coerce_structured_value(value: str | None) -> MetadataValue:
    """Coerce a structured sidecar value to int, then float, then string."""
    if value is None:
        return ""

    value = value.strip()
    if value == "":
        return ""

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _is_plausible_metadata_key(key: str) -> bool:
    """Return True for non-empty key tokens that look like metadata names."""
    key = key.strip()
    return (
        bool(key)
        and len(key) <= 128
        and not key.startswith(("#", ";", "["))
        and any(ch.isalnum() or ch == "_" for ch in key)
    )


def _parse_configparser_metadata(text: str) -> tuple[dict[str, MetadataValue], int]:
    """Parse a ``[metadata]`` section using configparser."""
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str

    try:
        parser.read_string(text)
    except configparser.Error:
        return {}, 0

    if not parser.has_section("metadata"):
        return {}, 0

    result: dict[str, MetadataValue] = {}
    pair_count = 0
    for key, value in parser.items("metadata", raw=True):
        key = key.strip()
        if not _is_plausible_metadata_key(key):
            continue
        result[key] = _coerce_structured_value(value)
        pair_count += 1
    return result, pair_count


def _parse_linewise_metadata(text: str) -> tuple[dict[str, MetadataValue], int]:
    """Parse ``name=value`` / ``name: value`` lines with last-key-wins."""
    result: dict[str, MetadataValue] = {}
    pair_count = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "[")):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        key = key.strip()
        if not _is_plausible_metadata_key(key):
            continue

        result[key] = _coerce_structured_value(value)
        pair_count += 1

    return result, pair_count


def _parse_structured_metadata(text: str) -> tuple[dict[str, MetadataValue], int]:
    """Parse generic structured sidecar text.

    A ``[metadata]`` INI section takes precedence.  Files without such a
    section fall back to line-wise ``name=value`` / ``name: value`` parsing.
    """
    metadata, pair_count = _parse_configparser_metadata(text)
    if pair_count:
        return metadata, pair_count
    return _parse_linewise_metadata(text)


def _looks_like_text(text: str) -> bool:
    """Reject binary content read with ``errors='replace'`` (BL-3 plausibility).

    Binary bytes decode to U+FFFD and non-printable control chars pile up; a
    real metadata sidecar has at most a stray one (e.g. a latin-1 µ in a value).
    So this rejects on a HIGH RATIO of replacement/control chars, not on a single
    bad byte — a plausible text file with one undecodable value still reads."""
    if not text:
        return False
    n = len(text)
    bad = text.count("�") + sum(
        1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    return bad <= max(4, n // 20)   # tolerate ~5% + a small fixed slack


def _read_structured_metadata(
    path: Path | str, *, _snapshot_bytes: bytes | None = None,
) -> tuple[dict[str, MetadataValue], int]:
    """Read a generic structured sidecar with replacement decoding."""
    path = Path(path)
    try:
        # BL-3: cap size (a metadata sidecar is small) before reading.
        if _snapshot_bytes is None and path.stat().st_size > _AUTO_SIDECAR_MAX_BYTES:
            logger.debug("_read_structured_metadata: %s over %d bytes; skipping",
                         path, _AUTO_SIDECAR_MAX_BYTES)
            return {}, 0
        text = (path.read_text(encoding="utf-8", errors="replace")
                if _snapshot_bytes is None else
                _snapshot_bytes.decode("utf-8", errors="replace"))
    except Exception:
        logger.warning("_read_structured_metadata: failed to read %s", path, exc_info=True)
        return {}, 0

    # BL-3: reject binary content that replacement-decoding would otherwise let
    # parse into spurious pairs.
    if not _looks_like_text(text):
        logger.debug("_read_structured_metadata: %s is not text; skipping", path)
        return {}, 0
    return _parse_structured_metadata(text)


# S-20: per-directory listing cache for the case-insensitive FALLBACK only,
# keyed by the directory's mtime so it invalidates when files are added.  With
# the exact-case is_file() fast path below, a directory is listed at most ONCE
# per (dir, mtime) -- not ~16x/frame as the first BL-3 rewrite did (worse than
# the single iterdir it replaced).
_DIR_LISTING_CACHE: dict[Path, tuple[float, dict[str, Path]]] = {}


def _dir_listing_lower(directory: Path) -> dict[str, Path]:
    """``{lowercased-name: path}`` for *directory*, cached until its mtime bumps."""
    try:
        mtime = directory.stat().st_mtime
    except OSError:
        return {}
    cached = _DIR_LISTING_CACHE.get(directory)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        listing = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    except OSError:
        listing = {}
    _DIR_LISTING_CACHE[directory] = (mtime, listing)
    return listing


def _existing_path_case_insensitive(candidate: Path) -> Path | None:
    """Return an existing sibling matching *candidate.name*, exact case first."""
    # S-20: exact-case fast path -- ONE stat, no directory listing (the common
    # case: sidecars are correctly cased).
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    # Case-insensitive fallback via the per-directory listing cache.
    return _dir_listing_lower(candidate.parent).get(candidate.name.lower())


def _find_sidecar(image_path: Path, ext: str) -> Path | None:
    """Locate a sidecar file for *image_path* with extension *ext*.

    Tries two strategies:
    1. Replace the image suffix: ``image_0001.txt``
    2. Append the extension:     ``image_0001.tif.txt``

    Parameters
    ----------
    image_path : Path
        Path to the primary image file.
    ext : str
        Target extension **without** leading dot (e.g. ``"txt"``).

    Returns
    -------
    Path | None
        First existing candidate, or ``None`` if neither exists.
    """
    ext = ext[1:] if ext.startswith(".") else ext
    ext = ext.lower()
    if not ext:
        return None

    candidate_replace = _existing_path_case_insensitive(
        image_path.with_suffix(f".{ext}")
    )
    if candidate_replace is not None:
        return candidate_replace

    candidate_append = _existing_path_case_insensitive(
        Path(str(image_path) + f".{ext}")
    )
    if candidate_append is not None:
        return candidate_append

    return None


def _parse_structured_sidecar_if_plausible(
    path: Path, *, min_pairs: int = _STRUCTURED_SIDECAR_MIN_PAIRS,
    _snapshot_bytes: bytes | None = None,
) -> dict[str, MetadataValue] | None:
    """Parse a structured ``name=value`` sidecar if it yields ``>= min_pairs``.

    ``min_pairs`` defaults to the AUTO plausibility threshold (3, so a random
    companion isn't mistaken for metadata).  An EXPLICIT ``meta_format`` passes
    ``min_pairs=1`` — a 1–2 field sidecar the user asked for must not be silently
    dropped (BL-3)."""
    metadata, pair_count = _read_structured_metadata(
        path, _snapshot_bytes=_snapshot_bytes,
    )
    if pair_count >= min_pairs:
        return metadata
    return None


def _bounded_metadata_snapshot(
    path: Path,
    limit: int | None,
    *,
    _with_revision: bool = False,
) -> bytes | tuple[bytes, MetadataSourceRevision] | None:
    if limit is None:
        return None
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_input_bytes must be a positive integer")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fence_revision = lambda value: (
        int(value.st_mode), int(value.st_dev), int(value.st_ino),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )
    if fence_revision(before) != fence_revision(after):
        raise MetadataSourceChanged(
            f"metadata input changed during bounded read: {path}", path=path,
        )
    if len(payload) > limit:
        if os.environ.get("XDART_METADATA_DEBUG") == "1":
            logger.warning("metadata_snapshot_refused path=%s limit_bytes=%d",
                           path, limit, stack_info=True)
        raise MetadataInputTooLarge(f"metadata input exceeds {limit} bytes: {path}")
    if _with_revision:
        # A descriptor view: comparable with the pathname revisions above
        # only through the win32 ctime seam.
        return payload, (
            int(after.st_mode), int(after.st_dev), int(after.st_ino),
            int(after.st_size), int(after.st_mtime_ns),
            identity_ctime_ns(after.st_ctime_ns),
        )
    return payload


_SPEC_POINT_FILE_MAX_BYTES = 1 << 25


def _bounded_spec_point_snapshot(path: Path, scan_number: int, image_number: int,
                                 limit: int) -> tuple[bytes, MetadataSourceRevision]:
    """Copy one SPEC point for silx, retaining the original descriptor revision.

    SPEC is a multi-scan input, unlike a per-image sidecar. The parser payload
    and each scanned line retain the caller's bound; physical scanning has its
    own 32 MiB bound. Motor-name updates preceding the first matching scan are
    inherited. Keeping the last requested-or-earlier row preserves clamping.
    Every data row of the selected scan, the discarded ones too, is checked
    against its ``#L`` labels as it passes (``_spec_row_problem``): one
    malformed row raises ValueError, never a shifted or substituted point.
    """
    from xrd_tools.io.spec import _spec_row_problem
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_input_bytes must be a positive integer")
    headers: dict[bytes, bytes] = {}
    selected = bytearray()
    point = b""
    active = False
    file_header = True
    row_index = 0
    labels = None
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > _SPEC_POINT_FILE_MAX_BYTES:
            raise MetadataInputTooLarge(
                f"SPEC file exceeds {_SPEC_POINT_FILE_MAX_BYTES} bytes: {path}")
        scanned = 0
        mca_continuation = False
        while raw := stream.readline(limit + 1):
            scanned += len(raw)
            if len(raw) > limit:
                raise MetadataInputTooLarge(f"SPEC line exceeds {limit} bytes: {path}")
            if scanned > _SPEC_POINT_FILE_MAX_BYTES:
                raise MetadataInputTooLarge(
                    f"SPEC file exceeds {_SPEC_POINT_FILE_MAX_BYTES} bytes: {path}")
            line = raw.strip()
            marker = line.split(None, 1)[0] if line else b""
            if active and marker == b"#F":
                break
            if marker == b"#S":
                if active:
                    break
                fields = line.split(None, 2)
                file_header = False
                active = len(fields) > 1 and fields[1] == str(scan_number).encode("ascii")
                if active:
                    selected.extend(raw)
            elif active:
                if line.startswith(b"@") or mca_continuation:
                    mca_continuation = line.endswith(b"\\")
                    continue
                if line.startswith(b"#"):
                    if marker == b"#L":  # silx's labels: separated by two or more spaces
                        labels = tuple(re.split(r"\s{2,}", line[2:].decode("utf-8", "replace").strip()))
                    selected.extend(raw)
                elif line:
                    if line.endswith(b"\\"):
                        raise MetadataInputTooLarge("bounded SPEC point requires an unwrapped data row")
                    problem = None if labels is None else _spec_row_problem(
                        line.decode("utf-8", "replace").split(), labels)
                    if problem is not None:
                        column, label, token = problem
                        raise ValueError(
                            f"SPEC {path}, scan {scan_number}, data row {row_index + 1}, column {column + 1} "
                            f"({label!r}): expected a complete decimal/scientific numeric token, got {token!r}")
                    if row_index <= image_number:
                        point = raw if raw.endswith(b"\n") else raw + b"\n"
                    row_index += 1
            elif marker == b"#F":
                headers = {marker: raw}
                file_header = True
            elif file_header and (marker == b"#E" or re.fullmatch(rb"#[Oo]\d+", marker)):
                headers[marker] = raw
            if sum(map(len, headers.values())) + len(selected) + len(point) > limit:
                raise MetadataInputTooLarge(f"SPEC point snapshot exceeds {limit} bytes: {path}")
        after = os.fstat(stream.fileno())
    descriptor_fence = lambda value: (
        value.st_mode, value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if descriptor_fence(before) != descriptor_fence(after):
        raise MetadataSourceChanged(
            f"metadata input changed during bounded read: {path}", path=path,
        )
    revision = (int(after.st_mode), int(after.st_dev), int(after.st_ino),
                int(after.st_size), int(after.st_mtime_ns), identity_ctime_ns(after.st_ctime_ns))
    return b"".join(headers.values()) + bytes(selected) + point, revision


def _read_auto_candidate_metadata(
    path: Path, *, max_input_bytes: int | None = None,
) -> tuple[dict[str, MetadataValue], MetadataSourceRevision | None] | None:
    """Try the appropriate parser for an auto-discovered sidecar."""
    ext = path.suffix.lower().lstrip(".")

    # Defensive cap (review follow-up): the .txt / .pdi primary parsers read the
    # WHOLE file (path.read_text()) on EVERY frame; the 1 MiB cap previously
    # guarded only the generic-structured route.  Bound the two first-class auto
    # routes the same way so a pathological/accidentally-huge companion cannot pin
    # memory or stall per-frame discovery (a real .pdi/.txt sidecar is a few KB).
    try:
        if (max_input_bytes is None
                and path.stat().st_size > _AUTO_SIDECAR_MAX_BYTES):
            logger.warning(
                "auto metadata: skipping oversize sidecar %s (> %d-byte cap)",
                path, _AUTO_SIDECAR_MAX_BYTES)
            return None
    except OSError:
        return None

    observed = _bounded_metadata_snapshot(
        path, max_input_bytes, _with_revision=True,
    )
    snapshot, revision = (None, None) if observed is None else observed
    if revision is None:
        revision = _metadata_path_revision(path)
    if ext == "txt":
        metadata = read_txt_metadata(
            path, _snapshot_bytes=snapshot,
        )       # SSRL # Counters / # Motors
        if metadata:
            return metadata, revision
        # BL-3: the SSRL .txt parser returns {} for a generic name=value .txt,
        # which would let a WORSE candidate latch — fall back to the structured
        # parser so a plausible generic .txt still wins.
        metadata = _parse_structured_sidecar_if_plausible(
            path, _snapshot_bytes=snapshot,
        )
        return None if metadata is None else (metadata, revision)

    if ext == "pdi":
        metadata = read_pdi_metadata(path, _snapshot_bytes=snapshot)
        return None if not metadata else (metadata, revision)

    metadata = _parse_structured_sidecar_if_plausible(
        path, _snapshot_bytes=snapshot,
    )
    return None if metadata is None else (metadata, revision)


def _auto_cache_key(image_path: Path) -> tuple[Path, str]:
    return image_path.parent, image_path.suffix.lower()


def _is_non_image_companion(path: Path, image_path: Path) -> bool:
    return (
        path.is_file()
        and path != image_path
        and path.suffix.lower() not in _IMAGE_EXTENSIONS
    )


def _iter_auto_sidecar_candidates(image_path: Path):
    """Yield auto sidecar candidates: allow-listed extensions ONLY, ranked by
    :data:`_AUTO_SIDECAR_EXTENSIONS`, append-form (``image.tif.<ext>``) before
    replace-form (``image.<ext>``).

    BL-3: this replaces the old alphabetical scan of EVERY non-image companion
    (where a per-frame ``.poni`` / ``img.tif.log`` / JSON could sort first and
    latch as the metadata sidecar).  It also stats the specific candidate paths
    rather than listing the whole directory (S-20 stat-first)."""
    seen: set[Path] = set()
    for ext in _AUTO_SIDECAR_EXTENSIONS:
        append = _existing_path_case_insensitive(Path(str(image_path) + f".{ext}"))
        replace = _existing_path_case_insensitive(image_path.with_suffix(f".{ext}"))
        # Preserve the on-disk case in the cached suffix (reconstruction is
        # case-insensitive either way, but keep it faithful).
        for candidate, convention, start in (
            (append, "append", len(image_path.name)),
            (replace, "replace", len(image_path.stem)),
        ):
            if (candidate is not None and candidate not in seen
                    and _is_non_image_companion(candidate, image_path)):
                seen.add(candidate)
                yield candidate, convention, candidate.name[start:]


def _auto_sidecar_from_cache(image_path: Path) -> Path | None:
    cached = _AUTO_SIDECAR_CACHE.get(_auto_cache_key(image_path))
    if cached is None:
        return None

    convention, suffix = cached
    if convention == "append":
        candidate = _existing_path_case_insensitive(Path(str(image_path) + suffix))
    elif convention == "replace":
        candidate = _existing_path_case_insensitive(
            image_path.with_name(image_path.stem + suffix)
        )
    else:
        _AUTO_SIDECAR_CACHE.pop(_auto_cache_key(image_path), None)
        return None

    if candidate is not None and _is_non_image_companion(candidate, image_path):
        return candidate

    _AUTO_SIDECAR_CACHE.pop(_auto_cache_key(image_path), None)
    return None


def _discover_auto_sidecar(
    image_path: Path, *, max_input_bytes: int | None = None,
) -> tuple[
    Path, str, str, dict[str, MetadataValue], MetadataSourceRevision | None
] | None:
    for candidate, convention, suffix in _iter_auto_sidecar_candidates(image_path):
        observed = _read_auto_candidate_metadata(
            candidate, max_input_bytes=max_input_bytes,
        )
        if observed is not None:
            metadata, revision = observed
            return candidate, convention, suffix, metadata, revision
    return None


def _read_auto_metadata_observed(
    image_path: Path, *, max_input_bytes: int | None = None,
) -> ImageMetadataRead:
    cached_sidecar = _auto_sidecar_from_cache(image_path)
    if cached_sidecar is not None:
        observed = _read_auto_candidate_metadata(
            cached_sidecar, max_input_bytes=max_input_bytes,
        )
        if observed is not None:
            metadata, revision = observed
            return ImageMetadataRead(metadata, cached_sidecar, revision)
        _AUTO_SIDECAR_CACHE.pop(_auto_cache_key(image_path), None)

    discovered = _discover_auto_sidecar(
        image_path, max_input_bytes=max_input_bytes,
    )
    if discovered is None:
        logger.debug("read_image_metadata: no auto sidecar found for %s", image_path)
        return ImageMetadataRead({}, None)

    candidate, convention, suffix, metadata, revision = discovered
    _AUTO_SIDECAR_CACHE[_auto_cache_key(image_path)] = (convention, suffix)
    # BL-3: log the convention auto locked onto so a wrong latch is visible in
    # the run log (it is then applied to EVERY frame via the cache).
    logger.info("metadata: auto locked onto %s-form '*%s' (%d fields, e.g. %s)",
                convention, suffix, len(metadata), candidate.name)
    return ImageMetadataRead(metadata, candidate, revision)


def _read_auto_metadata(image_path: Path) -> dict[str, MetadataValue]:
    return dict(_read_auto_metadata_observed(image_path).values)


def _extract_scan_info(image_path: Path) -> tuple[str | None, int | None, int]:
    """Parse SSRL filename conventions to extract scan metadata.

    Recognised image-filename layouts:

    1. ``b_{username}_{specname}_scan{N}_{M}.{ext}`` — the SSRL
       Pilatus convention where the SPEC username is prefixed with
       ``b_``.  Strip everything through the second underscore
       (i.e. ``b_thampy_``) to recover ``{specname}_scan{N}_{M}``.
    2. ``checkout_{specname}_scan{N}_{M}.{ext}`` — the one
       non-``b_`` username currently in use at the beamline; strip
       the literal ``checkout_`` prefix.
    3. ``{specname}_scan{N}_{M}.{ext}`` — generic; no prefix strip.
       Also covers Eiger HDF5 masters whose filenames end in
       ``_scan{N}_master`` (no per-frame index in the filename
       itself; ``image_number`` returns 0).

    For the trailing ``_scan{N}_{M}`` part, ``{M}`` is either an
    integer frame index or the literal ``master`` (Eiger).  ``{M}``
    as ``master`` returns ``image_number=0``.

    Parameters
    ----------
    image_path : Path
        Path to an SSRL detector image file.

    Returns
    -------
    tuple[str | None, int | None, int]
        ``(spec_filename, scan_number, image_number)`` where *spec_filename*
        is the bare SPEC file name (no directory), *scan_number* is the
        integer scan index, and *image_number* is the zero-based image index
        within the scan.  *spec_filename* and *scan_number* are ``None`` when
        the pattern is not found.
    """
    stem = image_path.stem

    # Rule 1: ``b_<username>_...`` → strip through second underscore.
    if stem.startswith("b_"):
        first = stem.find("_")
        second = stem.find("_", first + 1) if first >= 0 else -1
        if second >= 0:
            stem = stem[second + 1:]
    # Rule 2: ``checkout_...`` → strip literal prefix.
    elif stem.startswith("checkout_"):
        stem = stem[len("checkout_"):]
    # Rule 3: no prefix strip — generic case.

    # ``_scan<N>_<M>`` where <M> is digits (TIF/RAW/CBF/EDF per-frame
    # file) or the literal "master" (Eiger HDF5 master file).
    match = re.search(r"_scan(\d+)_(\d+|master)$", stem)
    if match is None:
        return None, None, 0

    scan_number = int(match.group(1))
    suffix = match.group(2)
    image_number = int(suffix) if suffix.isdigit() else 0
    spec_fname = stem[: match.start()]

    return spec_fname, scan_number, image_number


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------


def read_txt_metadata(
    path: Path | str, *, _snapshot_bytes: bytes | None = None,
) -> dict[str, float]:
    """Read an SSRL ``.txt`` sidecar metadata file.

    The file contains two single-line sections introduced by ``# Counters``
    and ``# Motors``, each with comma-or-equals-separated ``key = value``
    pairs.  An ``epoch`` timestamp is extracted from the ``time: <datetime>``
    field.

    Parameters
    ----------
    path : Path | str
        Path to the ``.txt`` metadata file.

    Returns
    -------
    dict[str, float]
        Flat mapping of counter/motor names and ``epoch`` to float values.
        Returns ``{}`` on any parse failure.
    """
    path = Path(path)
    try:
        data = (path.read_text() if _snapshot_bytes is None else
                _snapshot_bytes.decode("utf-8", errors="replace"))

        counters_match = re.search(r"# Counters\n(.*)\n", data)
        motors_match = re.search(r"# Motors\n(.*)\n", data)
        if counters_match is None or motors_match is None:
            logger.warning("read_txt_metadata: missing Counters/Motors section in %s", path)
            return {}

        counters = _parse_kv_pairs(counters_match.group(1), r",|=")
        motors = _parse_kv_pairs(motors_match.group(1), r",|=")

        time_str = data[data.index("time: ") + 5 : data.index("# Temp") - 2].strip()
        d = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        extras: dict[str, float] = {"epoch": time.mktime(d.timetuple())}

        return {**counters, **motors, **extras}

    except Exception:
        logger.warning("read_txt_metadata: failed to parse %s", path, exc_info=True)
        return {}


def read_pdi_metadata(
    path: Path | str, *, _snapshot_bytes: bytes | None = None,
) -> dict[str, float]:
    """Read a Pilatus Detector Image (``.pdi``) sidecar metadata file.

    Two format variants are tried in order:

    * **Primary** — sections ``All Counters;...;;# All Motors`` and
      ``All Motors;...;#``.
    * **Fallback** — section between
      ``# Diffractometer Motor Positions for image`` and
      ``# Calculated Detector Calibration Parameters for image:``.
      If a ``2Theta`` key is present a ``TwoTheta`` alias is added.
    * **Last resort** — ``{'TwoTheta': 0.0, 'Theta': 0.0}`` for motors.

    An ``epoch`` value is extracted from the last non-empty segment after the
    final ``;``.

    Parameters
    ----------
    path : Path | str
        Path to the ``.pdi`` metadata file.

    Returns
    -------
    dict[str, float]
        Flat mapping of counter/motor names and ``epoch`` to float values.
        Returns ``{}`` on any parse failure.
    """
    path = Path(path)
    try:
        data = (path.read_text() if _snapshot_bytes is None else
                _snapshot_bytes.decode("utf-8", errors="replace")).replace("\n", ";")

        # --- counters & motors ---
        try:
            counters_match = re.search(r"All Counters;(.*);;# All Motors", data)
            motors_match = re.search(r"All Motors;(.*);#", data)
            if counters_match is None or motors_match is None:
                raise AttributeError
            counters = _parse_kv_pairs(counters_match.group(1), r";|=")
            motors = _parse_kv_pairs(motors_match.group(1), r";|=")
        except AttributeError:
            ss1 = r"# Diffractometer Motor Positions for image;# "
            ss2 = r";# Calculated Detector Calibration Parameters for image:"
            try:
                motors_match = re.search(f"{ss1}(.*){ss2}", data)
                if motors_match is None:
                    raise AttributeError
                motors = _parse_kv_pairs(motors_match.group(1), r";|=")
                if "2Theta" in motors:
                    motors["TwoTheta"] = motors["2Theta"]
            except (AttributeError, KeyError):
                # BW / .pdi junk-latch: do NOT fabricate motor values from
                # unparseable content.  The old last-resort returned
                # {'TwoTheta': 0.0, 'Theta': 0.0} for ANY garbage, which is
                # non-empty -> a junk `.pdi` latched as valid metadata and
                # bypassed every BL-3 gate.  Reject: no recognizable section.
                logger.warning(
                    "read_pdi_metadata: no recognizable Counters/Motors or "
                    "Diffractometer section in %s", path)
                return {}
            counters = {}

        # --- epoch ---
        extras: dict[str, float] = {}
        tail = data[data.rindex(";") + 1 :]
        if tail:
            try:
                extras["epoch"] = float(tail)
            except ValueError:
                pass

        # BW junk-latch (primary path, review follow-up): the section markers can
        # match while their captured groups hold only non-float tokens, leaving
        # counters AND motors empty.  A float epoch tail then made the result
        # non-empty ({'epoch': ...}) -> a junk .pdi still latched as valid metadata
        # (the earlier fix only closed the Diffractometer FALLBACK branch).  An
        # epoch alone is not metadata: require at least one real counter/motor.
        if not counters and not motors:
            logger.warning(
                "read_pdi_metadata: markers matched but no parseable "
                "counters/motors (epoch-only) in %s -- rejecting", path)
            return {}

        return {**counters, **motors, **extras}

    except Exception:
        logger.warning("read_pdi_metadata: failed to parse %s", path, exc_info=True)
        return {}


def read_image_metadata(
    image_path: Path | str,
    meta_format: str | None = "txt",
    meta_dir: Path | str | None = None,
    *,
    max_input_bytes: int | None = None,
) -> dict[str, MetadataValue]:
    """Unified metadata reader for SSRL detector image sidecar files.

    Parameters
    ----------
    image_path : Path | str
        Path to the detector image file.
    meta_format : str | None
        One of ``"txt"``, ``"pdi"``, ``"spec"``, ``"auto"``, or a generic
        sidecar extension such as ``"metadata"``.  Case-insensitive.
        ``None`` is treated as ``"auto"``.  Generic extensions are parsed as
        structured ``name=value`` / ``name: value`` sidecars when they yield
        at least three plausible pairs.
    meta_dir : Path | str | None
        Optional explicit directory to search for the metadata file.
        Currently used only for ``meta_format="spec"`` — overrides the
        default (image folder + immediate parent) when set.  Useful when
        the SPEC file is stored separately from the image files.  Pass
        ``None`` (default) or the empty string to use the default
        location heuristic.

    Returns
    -------
    dict[str, int | float | str]
        Flat mapping of metadata key → value.  Existing SSRL readers return
        floats; structured sidecars may also return strings.  Returns ``{}``
        when no sidecar is found or parsing fails.
    """
    return dict(
        read_image_metadata_observed(
            image_path,
            meta_format=meta_format,
            meta_dir=meta_dir,
            max_input_bytes=max_input_bytes,
        ).values
    )


def live_sidecar_observation(
    image_path: Path | str, meta_format: str,
) -> tuple[tuple[str, MetadataSourceRevision | None], ...]:
    """The sidecar files a live Run waits on for one image, with their revisions.

    An explicit sidecar format names the one file :func:`_find_sidecar`
    selects; Auto names every allow-listed candidate that exists now.  ``()``
    means none exists.  Nothing is opened or parsed: the live owner compares two
    observations a poll apart before it admits the frame.
    """

    image = _lexical_absolute(image_path)
    meta_format_norm = str(meta_format).strip().lower()
    if meta_format_norm == "auto":
        found = tuple(
            candidate for candidate, _convention, _suffix
            in _iter_auto_sidecar_candidates(image)
        )
    else:
        sidecar = _find_sidecar(image, meta_format_norm)
        found = () if sidecar is None else (sidecar,)
    return tuple((str(path), _metadata_path_revision(path)) for path in found)


#: Pauses in seconds before reading again when only a listed directory
#: changed during a read (MetadataDiscoveryChanged): a live acquisition
#: writes beside the image at any time.  Five reads in all.
_DISCOVERY_PAUSES_S = (0.005, 0.01, 0.02, 0.04)


def read_image_metadata_observed(
    image_path: Path | str,
    meta_format: str | None = "txt",
    meta_dir: Path | str | None = None,
    *,
    max_input_bytes: int | None = None,
    spec_point_snapshot: bool = False,
) -> ImageMetadataRead:
    """Read image metadata and report the exact accepted metadata file.

    The parsing and discovery policy is identical to :func:`read_image_metadata`.
    The additional path is intended for admission, collision protection, and
    provenance owners that must retain the file actually consumed.
    ``spec_point_snapshot`` selects Average's bounded single-point SPEC policy;
    it requires ``max_input_bytes`` and leaves other formats unchanged.

    A read during which only listed directories changed is read again, at most
    five reads in all, and the first read whose inputs held still is returned
    with its own recorded revisions; the last refusal is raised. A changed
    candidate or selected source is refused at once.
    """

    for pause in _DISCOVERY_PAUSES_S:
        try:
            return _read_image_metadata_observed_once(
                image_path, meta_format, meta_dir,
                max_input_bytes=max_input_bytes,
                spec_point_snapshot=spec_point_snapshot,
            )
        except MetadataDiscoveryChanged:
            time.sleep(pause)
    return _read_image_metadata_observed_once(
        image_path, meta_format, meta_dir,
        max_input_bytes=max_input_bytes,
        spec_point_snapshot=spec_point_snapshot,
    )


def _read_image_metadata_observed_once(
    image_path: Path | str,
    meta_format: str | None,
    meta_dir: Path | str | None,
    *,
    max_input_bytes: int | None,
    spec_point_snapshot: bool,
) -> ImageMetadataRead:
    """One read for :func:`read_image_metadata_observed`."""

    image_path = _lexical_absolute(image_path)

    # Normalise case so callers can pass "SPEC" / "Spec" / "spec"
    # interchangeably — xdart's UI surfaces the value as "SPEC" in
    # the metadata format combobox, but the comparison below expects
    # lowercase.  Pre-fix this silently returned ``{}`` and logged
    # "unknown meta_format" for every SPEC frame, so positioners
    # never made it into the scan.
    meta_format_norm = (
        "auto" if meta_format is None else str(meta_format).strip().lower()
    )
    discovery_revisions = _metadata_discovery_revisions(
        image_path, meta_format_norm, meta_dir,
    )

    if meta_format_norm == "auto":
        return _with_discovery_revisions(
            _read_auto_metadata_observed(
                image_path, max_input_bytes=max_input_bytes,
            ),
            discovery_revisions,
        )

    if meta_format_norm in ("txt", "pdi"):
        sidecar = _find_sidecar(image_path, meta_format_norm)
        if sidecar is None:
            logger.debug(
                "read_image_metadata: no %s sidecar found for %s",
                meta_format_norm,
                image_path,
            )
            return _with_discovery_revisions(
                ImageMetadataRead({}, None), discovery_revisions,
            )
        observed = _bounded_metadata_snapshot(
            sidecar, max_input_bytes, _with_revision=True,
        )
        snapshot, revision = (None, None) if observed is None else observed
        if revision is None:
            revision = _metadata_path_revision(sidecar)
        if meta_format_norm == "txt":
            return _with_discovery_revisions(
                ImageMetadataRead(
                    read_txt_metadata(sidecar, _snapshot_bytes=snapshot),
                    sidecar,
                    revision,
                ),
                discovery_revisions,
            )
        return _with_discovery_revisions(
            ImageMetadataRead(
                read_pdi_metadata(sidecar, _snapshot_bytes=snapshot),
                sidecar,
                revision,
            ),
            discovery_revisions,
        )

    if meta_format_norm == "spec":
        # Normalise empty-string to None for downstream "use default
        # heuristic" branch.
        search_dir = (
            Path(meta_dir) if meta_dir not in (None, "") else None
        )
        return _with_discovery_revisions(
            _read_spec_metadata_observed(
                image_path,
                search_dir=search_dir,
                max_input_bytes=max_input_bytes,
                spec_point_snapshot=spec_point_snapshot,
            ),
            discovery_revisions,
        )

    # Any other value is treated as a generic structured-sidecar extension.
    sidecar = _find_sidecar(image_path, meta_format_norm)
    if sidecar is None:
        logger.debug("read_image_metadata: no %r sidecar found for %s",
                     meta_format, image_path)
        return _with_discovery_revisions(
            ImageMetadataRead({}, None), discovery_revisions,
        )
    # BL-3: an EXPLICIT format is a deliberate user choice — accept a 1-2 field
    # sidecar (the AUTO min-pairs=3 plausibility gate does NOT apply here; before
    # this, 1-2-pair explicit sidecars were silently dropped with a misleading
    # "unknown meta_format" warning).
    observed = _bounded_metadata_snapshot(
        sidecar, max_input_bytes, _with_revision=True,
    )
    snapshot, revision = (None, None) if observed is None else observed
    if revision is None:
        revision = _metadata_path_revision(sidecar)
    metadata = _parse_structured_sidecar_if_plausible(
        sidecar, min_pairs=1, _snapshot_bytes=snapshot,
    )
    if metadata is not None:
        return _with_discovery_revisions(
            ImageMetadataRead(metadata, sidecar, revision),
            discovery_revisions,
        )
    logger.warning("read_image_metadata: %s (format %r) had no readable "
                   "key=value fields", sidecar, meta_format)
    return _with_discovery_revisions(
        ImageMetadataRead({}, sidecar, revision),
        discovery_revisions,
    )


# ---------------------------------------------------------------------------
# SPEC metadata (private — called from read_image_metadata)
# ---------------------------------------------------------------------------


def _spec_search_dirs(image_path: Path, search_dir: Path | str | None) -> list[Path]:
    """Where an image's SPEC file is looked for, in order.

    The explicit directory first (when given), then the image's own folder and
    up to TWO levels of parent directories above it (SSRL convention: the SPEC
    file may sit a couple levels up).  Duplicates are harmless.
    """
    candidate_dirs: list[Path] = []
    if search_dir is not None:
        candidate_dirs.append(Path(search_dir))
    candidate_dirs.extend([
        image_path.parent,
        image_path.parent.parent,
        image_path.parent.parent.parent,
    ])
    return candidate_dirs


def _read_spec_metadata_observed(
    image_path: Path,
    search_dir: Path | None = None,
    *,
    max_input_bytes: int | None = None,
    spec_point_snapshot: bool = False,
) -> ImageMetadataRead:
    """Extract per-image counters and motor positions from a SPEC file.

    The SPEC file (a SPEC data file, named with NO extension — the bare
    ``spec_fname`` from the image stem) is located by, in order:

    1. ``search_dir`` (when provided) — the explicit directory the
       caller passed in via ``read_image_metadata(meta_dir=...)``.
       Useful when the SPEC file lives apart from the image files.
    2. The image's own directory (the same folder as the image).
    3. Up to two levels of parent directories above that folder.

    The scan and image numbers are inferred from the filename stem
    using SSRL naming conventions.

    Parameters
    ----------
    image_path : Path
        Path to the detector image file.
    search_dir : Path | None
        Explicit directory to look in first.  ``None`` falls back to
        the image-dir + parent-dir search.

    Returns
    -------
    dict[str, float]
        Counters for the image point merged with motor positions.
        Returns ``{}`` on any error. A selected scan holding a malformed row
        (a non-number, a missing or an extra column) is one: silx would skip
        that row and shift every later one onto the wrong image
        (FINITE-SPEC-INVALID-LINE-SHIFT), so no read of that scan returns a
        row, with one warning per read.
    """
    try:
        from silx.io.specfile import SpecFile  # noqa: PLC0415
    except ImportError:
        logger.warning("read_image_metadata (spec): silx is not installed")
        return ImageMetadataRead({}, None)

    spec_file: Path | None = None
    try:
        spec_fname, scan_number, image_number = _extract_scan_info(image_path)
        if spec_fname is None or scan_number is None:
            logger.warning(
                "_read_spec_metadata: cannot parse scan info from filename %s",
                image_path.name,
            )
            return ImageMetadataRead({}, None)

        candidate_dirs = _spec_search_dirs(image_path, search_dir)
        for sd in candidate_dirs:
            candidate = sd / spec_fname
            if candidate.is_file():
                spec_file = candidate
                break

        if spec_file is None:
            logger.warning(
                "_read_spec_metadata: SPEC file %r not found in %s",
                spec_fname,
                [str(d) for d in candidate_dirs],
            )
            return ImageMetadataRead({}, None)

        observed = (_bounded_spec_point_snapshot(spec_file, scan_number, image_number,
                                                max_input_bytes)
                    if spec_point_snapshot else _bounded_metadata_snapshot(
                        spec_file, max_input_bytes, _with_revision=True))
        from xrd_tools.io.spec import _validate_spec_numeric_tokens
        if observed is None:
            revision = _metadata_path_revision(spec_file)
            sf = SpecFile(str(spec_file))
            scan = sf[f"{scan_number}.1"]
            # the strict scan validator reads the selected scan's physical rows itself (no second parser)
            _validate_spec_numeric_tokens(spec_file, scan, tuple(str(label) for label in scan.labels))
            npts = scan.data.shape[1]
            img_idx = min(image_number, npts - 1)
            (counters,) = _spec_counters(scan, (img_idx,))
            motors = {name: float(scan.motor_position_by_name(name))
                      for name in scan.motor_names}
            return ImageMetadataRead(
                {**counters, **motors}, spec_file, revision,
            )
        snapshot, revision = observed

        scratch_handle = None
        scratch = None
        parser = None
        primary: BaseException | None = None
        result: ImageMetadataRead | None = None
        cleanup_error: BaseException | None = None
        try:
            scratch_handle = tempfile.NamedTemporaryFile(
                prefix="xdart-spec-", delete=False,
            )
            scratch = Path(scratch_handle.name)
            scratch_handle.write(snapshot)
            scratch_handle.close()
            parser = SpecFile(str(scratch))
            scan = parser[f"{scan_number}.1"]
            if not spec_point_snapshot:  # the bytes silx parsed; a point's whole scan was checked as it was copied
                _validate_spec_numeric_tokens(scratch, scan, tuple(str(label) for label in scan.labels))
            npts = scan.data.shape[1]
            img_idx = min(0 if spec_point_snapshot else image_number, npts - 1)
            (counters,) = _spec_counters(scan, (img_idx,))
            motors = {name: float(scan.motor_position_by_name(name))
                      for name in scan.motor_names}
            result = ImageMetadataRead(
                {**counters, **motors}, spec_file, revision,
            )
        except BaseException as exc:
            primary = exc
        finally:
            if parser is not None:
                try:
                    parser.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if scratch_handle is not None and not scratch_handle.closed:
                try:
                    scratch_handle.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if scratch is not None:
                try:
                    scratch.unlink()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        if primary is not None:
            if cleanup_error is not None:
                failure = MetadataScratchCleanupFailed(
                    f"SPEC snapshot close/unlink failed: {cleanup_error}"
                )
                failure.__cause__ = cleanup_error
                raise primary from failure
            if not isinstance(primary, Exception):
                raise primary
            logger.warning(
                "_read_spec_metadata: failed to parse bounded snapshot for %s",
                image_path, exc_info=primary,
            )
            return ImageMetadataRead({}, spec_file, revision)
        if cleanup_error is not None:
            raise MetadataScratchCleanupFailed(
                f"SPEC snapshot close/unlink failed: {cleanup_error}"
            ) from cleanup_error
        assert result is not None
        return result

    except (MetadataInputTooLarge, MetadataScratchCleanupFailed):
        raise
    except Exception as error:
        if spec_point_snapshot and isinstance(error, OSError):
            raise
        if isinstance(error.__cause__, MetadataScratchCleanupFailed):
            raise
        logger.warning(
            "_read_spec_metadata: failed to read SPEC metadata for %s",
            image_path,
            exc_info=True,
        )
        return ImageMetadataRead(
            {}, spec_file,
            None if spec_file is None else _metadata_path_revision(spec_file),
        )


def _read_spec_metadata(
    image_path: Path,
    search_dir: Path | None = None,
) -> dict[str, float]:
    return dict(
        _read_spec_metadata_observed(
            image_path,
            search_dir=search_dir,
        ).values
    )


# ---------------------------------------------------------------------------
# SPEC rows of an image series: one read per scan, and Live row readiness
# ---------------------------------------------------------------------------

_SPEC_SCAN_LINE = re.compile(rb"#S[ \t]+(\d+)(?=\s|$)")


class SpecRowPending(Exception):
    """A live image's own SPEC row is not there yet: its SPEC file, scan or row is missing."""


class SpecRowInvalid(ValueError):
    """A complete SPEC line at or before a live image's row is not a row the scan's parser reads as such."""


class _SpecParseDisagreement(ValueError):
    """silx read a scan's rows differently from the scan's own lines."""


@dataclass(frozen=True, slots=True)
class SpecScanExtent:
    """The bytes of one SPEC scan that an image series' rows were read from.

    A SPEC file only grows while it is acquired: rows are appended and later
    scans follow.  The extent keeps the file-header lines in effect for the
    scan (offset, length; one SHA-256 over them) and the scan's own bytes from
    its ``#S`` line: ``marks`` holds, for the end of the scan header and then
    for each row, the end offset and the SHA-256 of the scan's bytes up to it.
    Appending keeps an extent; rewriting a byte it covers breaks it.  It lives
    in memory only: nothing persisted records it.
    """

    path: str
    header: tuple[tuple[int, int], ...]
    header_sha256: str
    scan_offset: int
    marks: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        ends = [end for end, _digest in self.marks]
        if (
            type(self.path) is not str or not os.path.isabs(self.path)
            or type(self.scan_offset) is not int or self.scan_offset < 0
            or not self.marks or ends != sorted(ends) or ends[0] < self.scan_offset
            or any(type(offset) is not int or type(length) is not int
                   or offset < 0 or length <= 0 for offset, length in self.header)
        ):
            raise TypeError("SPEC scan extent is invalid")

    def through(self, rows: int) -> SpecScanExtent:
        """This extent up to the end of its first ``rows`` rows."""
        return replace(self, marks=self.marks[:rows + 1])

    def extends(self, earlier: SpecScanExtent) -> bool:
        """Whether this extent covers ``earlier``'s bytes unchanged, in the same file."""
        count = len(earlier.marks)
        return (
            os.path.normcase(self.path) == os.path.normcase(earlier.path)
            and (self.header, self.header_sha256, self.scan_offset)
            == (earlier.header, earlier.header_sha256, earlier.scan_offset)
            and len(self.marks) >= count
            and self.marks[count - 1] == earlier.marks[count - 1]
        )


def spec_extent_holds(extent: SpecScanExtent) -> bool:
    """Whether the SPEC file still holds ``extent``'s bytes; it may have grown."""
    try:
        with open(extent.path, "rb") as stream:
            digest = hashlib.sha256()
            for offset, length in extent.header:
                stream.seek(offset)
                chunk = stream.read(length)
                if len(chunk) != length:
                    return False
                digest.update(chunk)
            if digest.hexdigest() != extent.header_sha256:
                return False
            end, expected = extent.marks[-1]
            stream.seek(extent.scan_offset)
            chunk = stream.read(end - extent.scan_offset)
    except OSError:
        return False
    return (len(chunk) == end - extent.scan_offset
            and hashlib.sha256(chunk).hexdigest() == expected)


@dataclass(frozen=True, slots=True)
class _SpecScan:
    """One read of a SPEC scan: its header lines in effect, its lines and its complete rows."""

    header: tuple[bytes, ...]
    lines: tuple[bytes, ...]  # the scan's own lines in order, MCA data left out
    rows: tuple[int, ...]  # the position in ``lines`` of each complete row
    columns: int | None  # the declared column count: ``#N``, else the ``#L`` labels
    extent: SpecScanExtent  # through every complete row

    def valid_rows(self, limit: int) -> int:
        """How many leading rows, of the first ``limit``, hold ``columns`` numbers each."""
        from xrd_tools.io.spec import _SPEC_NUMBER

        for index, position in enumerate(self.rows[:limit]):
            tokens = self.lines[position].split()
            if not tokens or self.columns is not None and len(tokens) != self.columns:
                return index
            try:
                for token in tokens:
                    if not _SPEC_NUMBER.fullmatch(token.decode("ascii")):
                        return index
            except ValueError:
                return index
        return min(limit, len(self.rows))


def _line_start(data: bytes, position: int) -> bool:
    return position == 0 or data[position - 1:position] == b"\n"


def _find_spec_scan(data: bytes, scan_number: int) -> int | None:
    """The offset of the first ``#S scan_number`` line; ``None`` when there is none."""
    position = data.find(b"#S")
    while position >= 0:
        match = _SPEC_SCAN_LINE.match(data, position) if _line_start(data, position) else None
        if match is not None and int(match[1]) == scan_number:
            return position
        position = data.find(b"#S", position + 2)
    return None


def _last_spec_file_header(data: bytes, end: int) -> int:
    """The offset of the last ``#F`` line before ``end``; 0 when there is none."""
    position = data.rfind(b"#F", 0, end)
    while position >= 0:
        if _line_start(data, position) and data[position + 2:position + 3] in (b"", b" ", b"\t", b"\r", b"\n"):
            return position
        position = data.rfind(b"#F", 0, position)
    return 0


def _spec_count(text: bytes) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def _spec_cancelled(cancelled) -> None:
    """Stop a SPEC read for a Stop: the error qualification's callers already translate."""
    if cancelled is not None and cancelled():
        raise InterruptedError("source qualification cancelled")


def _read_spec_scan(path: Path, scan_number: int, *, cancelled=None) -> _SpecScan | None:
    """Read the first ``#S scan_number`` block of a SPEC file, up to its last complete line.

    Like the bounded point snapshot, the file-header lines in effect are the
    ``#F``/``#E``/``#O`` lines of the last ``#F`` block before the scan, the
    scan ends at the next ``#S`` or ``#F`` line, and MCA (``@``) data is left
    out.  A line without its newline is still being written and ends the read.
    The file is copied once, up to the size it had when opened: bytes rewritten
    or truncated meanwhile cannot fault the read, as they could under a mapped
    view.  ``None`` when the file holds no such scan.  ``cancelled`` is asked
    after the copy and every few thousand lines.
    """
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        data = stream.read(size) if size > 0 else b""
    _spec_cancelled(cancelled)
    start = _find_spec_scan(data, scan_number)
    if start is None:
        return None
    position = _last_spec_file_header(data, start)
    headers: dict[bytes, tuple[int, int]] = {}
    while position < start:
        end = data.find(b"\n", position, start)
        end = start if end < 0 else end + 1
        line = data[position:end].strip()
        marker = line.split(None, 1)[0] if line else b""
        if marker == b"#S":
            break
        if marker == b"#F":
            headers = {marker: (position, end - position)}
        elif marker == b"#E" or re.fullmatch(rb"#[Oo]\d+", marker):
            headers[marker] = (position, end - position)
        position = end
    header = tuple(headers.values())
    header_bytes = tuple(data[offset:offset + length] for offset, length in header)
    digest = hashlib.sha256()
    lines: list[bytes] = []
    rows: list[int] = []
    marks: list[tuple[int, str]] = []
    declared = labels = None
    position, mca = start, False
    while position < len(data):
        if not len(lines) % 4096:
            _spec_cancelled(cancelled)
        end = data.find(b"\n", position)
        if end < 0:
            break
        end += 1
        raw = data[position:end]
        line = raw.strip()
        marker = line.split(None, 1)[0] if line else b""
        if position != start and marker in (b"#S", b"#F"):
            break
        if mca or line.startswith(b"@"):
            mca = line.endswith(b"\\")
        elif line.startswith(b"#"):
            lines.append(raw)
            if marker == b"#N" and declared is None:
                declared = _spec_count(line[2:])
            elif marker == b"#L" and labels is None:
                labels = len(re.split(rb"\s{2,}", line[2:].strip()))
        elif line:
            if not marks:
                marks.append((position, digest.hexdigest()))
            lines.append(raw)
            rows.append(len(lines) - 1)
            digest.update(raw)
            marks.append((end, digest.hexdigest()))
            position = end
            continue
        digest.update(raw)
        position = end
    if not marks:
        marks.append((position, digest.hexdigest()))
    header_digest = hashlib.sha256(b"".join(header_bytes)).hexdigest()
    extent = SpecScanExtent(str(_lexical_absolute(path)), header, header_digest, start, tuple(marks))
    return _SpecScan(header_bytes, tuple(lines), tuple(rows),
                     declared if declared is not None else labels, extent)


def _resolve_spec_image(image_path: Path, meta_dir: Path | str | None) -> tuple[Path, int, int] | None:
    """An image's SPEC file (the reader's own folder search), scan and row; ``None`` when it names none."""
    spec_fname, scan_number, image_number = _extract_scan_info(image_path)
    if spec_fname is None or scan_number is None:
        return None
    for folder in _spec_search_dirs(image_path, meta_dir):
        candidate = folder / spec_fname
        if candidate.is_file():
            return _lexical_absolute(candidate), scan_number, image_number
    return None


def _spec_counters(scan, rows: Sequence[int]) -> list[dict[str, float]]:
    """Each row's counters, one value per label from the label's first column.

    silx's ``data_column_by_name`` re-reads its column on every call, so a
    per-row, per-label lookup costs rows x labels column reads.  This takes the
    scan's table once and reads each label's first column, the column that
    lookup returns.  A table that does not have one row per label keeps the
    per-label lookup, so its failure stays the one silx raises.
    """
    labels = tuple(scan.labels)
    table = scan.data
    if getattr(table, "ndim", 0) != 2 or table.shape[0] < len(labels):
        return [{label: float(scan.data_column_by_name(label)[row]) for label in labels} for row in rows]
    first: dict[str, int] = {}
    for index, label in enumerate(labels):
        first.setdefault(label, index)
    return [{label: float(table[index, row]) for label, index in first.items()} for row in rows]


def _same_number(left: float, right: float) -> bool:
    return left == right or left != left and right != right


def _spec_rows_values(payload: bytes, scan_number: int, rows: Sequence[int], *,
                      own_rows: Sequence[tuple[float, ...]] | None = None,
                      row_count: int | None = None, cancelled=None) -> list[dict[str, float]]:
    """Each requested row's counters, with the scan's motor positions, parsed by silx.

    silx reads files only, so ``payload`` (one scan) goes through a scratch
    file.  With ``own_rows`` (each requested row's numbers as its own line
    holds them) and ``row_count`` (the payload's rows), silx must read the
    same table: that many rows, and each requested row equal to its line,
    else _SpecParseDisagreement.  A parse failure is raised; a failure to
    close or remove the scratch file raises MetadataScratchCleanupFailed,
    chained to any parse failure.
    """
    from silx.io.specfile import SpecFile  # noqa: PLC0415

    scratch_handle = scratch = parser = None
    primary: BaseException | None = None
    cleanup_error: BaseException | None = None
    values: list[dict[str, float]] = []
    try:
        scratch_handle = tempfile.NamedTemporaryFile(prefix="xdart-spec-", delete=False)
        scratch = Path(scratch_handle.name)
        scratch_handle.write(payload)
        scratch_handle.close()
        parser = SpecFile(str(scratch))
        scan = parser[f"{scan_number}.1"]
        _spec_cancelled(cancelled)
        if own_rows is not None:
            table = scan.data
            if table.ndim != 2 or table.shape[1] != row_count or any(
                    table.shape[0] != len(own) or not all(map(_same_number, map(float, table[:, row]), own))
                    for row, own in zip(rows, own_rows)):
                raise _SpecParseDisagreement(f"silx read scan {scan_number} differently from its lines")
        motors = {name: float(scan.motor_position_by_name(name)) for name in scan.motor_names}
        values.extend({**counters, **motors} for counters in _spec_counters(scan, rows))
    except BaseException as exc:
        primary = exc
    finally:
        for close in (None if parser is None else parser.close,
                      None if scratch_handle is None or scratch_handle.closed else scratch_handle.close,
                      None if scratch is None else scratch.unlink):
            if close is not None:
                try:
                    close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
    if primary is not None:
        if cleanup_error is not None:
            failure = MetadataScratchCleanupFailed(f"SPEC snapshot close/unlink failed: {cleanup_error}")
            failure.__cause__ = cleanup_error
            raise primary from failure
        raise primary
    if cleanup_error is not None:
        raise MetadataScratchCleanupFailed(
            f"SPEC snapshot close/unlink failed: {cleanup_error}"
        ) from cleanup_error
    return values


@dataclass(frozen=True, slots=True)
class SpecPoints:
    """The SPEC rows of an image series, read once: one result per image, and their extent.

    ``extent`` is ``None`` when a live read found the row of an image the Run
    already published gone or not valid: every image then has no metadata.
    """

    path: Path | None
    extent: SpecScanExtent | None
    reads: tuple[ImageMetadataRead, ...]


def _not_a_row(spec_path: Path, scan_number: int, scan: _SpecScan, index: int) -> str:
    line = scan.lines[scan.rows[index]].strip().decode("utf-8", "replace")
    shape = "numbers" if scan.columns is None else f"{scan.columns} numbers"
    return f"{spec_path.name} scan {scan_number} row {index + 1} is not a row of {shape}: {line!r}"


def read_spec_points(image_paths: Sequence[Path | str], meta_dir: Path | str | None = None, *,
                     published: int | None = None, cancelled=None) -> SpecPoints | None:
    """Every image's own SPEC row, from one read of their scan.

    Each image's name gives its SPEC file (the reader's own folder search),
    scan and row, as for :func:`read_image_metadata`.  A row is valid when its
    line is complete, holds the scan's declared column count (``#N``, else the
    ``#L`` labels) of numbers, and silx reads it as that same row.  The values
    are the ones that reader returns for an image whose row is valid; the
    extent covers the file-header lines and the scan through the last of these
    rows.

    Finite (``published`` is ``None``): ``None`` unless every image names the
    same scan of the same SPEC file and that scan holds a valid row for each;
    callers then read each image as before.

    Live (``published`` is how many leading images the Run already published):
    never another image's row.  If a published image's row is gone or not
    valid, every image comes back without metadata and ``extent`` is ``None``,
    for the caller's published-metadata fence; otherwise an image whose row is
    not there yet raises SpecRowPending, and one at or after a complete line
    that is not a valid row raises SpecRowInvalid naming that line.
    """
    paths = tuple(_lexical_absolute(path) for path in image_paths)
    live = published is not None
    if not paths:
        if live:
            raise ValueError("SPEC points need at least one image")
        return None
    resolved = [_resolve_spec_image(path, meta_dir) for path in paths]
    scans = {value[:2] for value in resolved if value is not None}
    if len(scans) > 1:
        if live:
            raise SpecRowInvalid(f"{paths[0].name} and the images after it name different SPEC scans")
        return None
    spec_path = scan_number = scan = None
    if scans:
        ((spec_path, scan_number),) = scans
        _spec_cancelled(cancelled)
        try:
            scan = _read_spec_scan(spec_path, scan_number, cancelled=cancelled)
        except InterruptedError:  # a Stop, not a file that went away
            raise
        except OSError:  # the file went away under the read
            scan = None
    numbers = tuple(None if value is None else value[2] for value in resolved)
    needed = max((number for number in numbers if number is not None), default=-1) + 1
    rows = 0 if scan is None else len(scan.rows)
    valid = 0 if scan is None else scan.valid_rows(needed)
    if not live:
        if scan is None or None in numbers or needed > valid:
            return None
    else:
        published = min(published, len(paths))
        if any(number is None or number >= valid for number in numbers[:published]):
            return SpecPoints(spec_path, None, tuple(ImageMetadataRead({}, None) for _path in paths))
        for path, number in zip(paths[published:], numbers[published:]):
            if number is not None and number >= valid and valid < rows:
                raise SpecRowInvalid(_not_a_row(spec_path, scan_number, scan, valid))
            if number is None or number >= valid:
                raise SpecRowPending(f"waiting for the SPEC row of {path.name}")
    own = [tuple(float(token) for token in scan.lines[scan.rows[number]].split()) for number in numbers]
    _spec_cancelled(cancelled)
    try:
        values = _spec_rows_values(
            b"".join(scan.header) + b"".join(scan.lines[:scan.rows[needed - 1] + 1]),
            scan_number, numbers, own_rows=own, row_count=needed, cancelled=cancelled)
    except (MetadataScratchCleanupFailed, InterruptedError):
        raise
    except Exception as error:
        if isinstance(error.__cause__, MetadataScratchCleanupFailed):
            raise
        if live:
            raise SpecRowInvalid(f"{spec_path.name} scan {scan_number} rows 1 to {needed} cannot be read "
                                 f"as their lines hold them: {error}") from error
        logger.warning("read_spec_points: failed to parse scan %s of %s",
                       scan_number, spec_path, exc_info=True)
        return None
    return SpecPoints(spec_path, scan.extent.through(needed),
                      tuple(ImageMetadataRead(value, spec_path) for value in values))


def spec_rows_ready(image_paths: Sequence[Path | str], meta_dir: Path | str | None = None) -> int:
    """How many leading images already have their own complete SPEC row.

    A live Run admits an image under SPEC metadata only then: no other point's
    row is borrowed while its own is missing.  A row is final once its newline
    is written, so one look is enough.  An image whose name gives no SPEC
    file, or whose file or scan does not exist yet, is not ready.
    """
    rows: dict[tuple[Path, int], int] = {}
    for index, path in enumerate(image_paths):
        resolved = _resolve_spec_image(_lexical_absolute(path), meta_dir)
        if resolved is None:
            return index
        spec_path, scan_number, number = resolved
        if (spec_path, scan_number) not in rows:
            try:
                scan = _read_spec_scan(spec_path, scan_number)
            except (OSError, ValueError):
                return index
            rows[spec_path, scan_number] = 0 if scan is None else len(scan.rows)
        if number >= rows[spec_path, scan_number]:
            return index
    return len(image_paths)
