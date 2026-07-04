"""Metadata parsing: read_txt_metadata, read_pdi_metadata, read_image_metadata."""
from __future__ import annotations

import configparser
import logging
import re
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["read_txt_metadata", "read_pdi_metadata", "read_image_metadata"]

MetadataValue = int | float | str

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


def _read_structured_metadata(path: Path | str) -> tuple[dict[str, MetadataValue], int]:
    """Read a generic structured sidecar with replacement decoding."""
    path = Path(path)
    try:
        # BL-3: cap size (a metadata sidecar is small) before reading.
        if path.stat().st_size > _AUTO_SIDECAR_MAX_BYTES:
            logger.debug("_read_structured_metadata: %s over %d bytes; skipping",
                         path, _AUTO_SIDECAR_MAX_BYTES)
            return {}, 0
        text = path.read_text(encoding="utf-8", errors="replace")
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
) -> dict[str, MetadataValue] | None:
    """Parse a structured ``name=value`` sidecar if it yields ``>= min_pairs``.

    ``min_pairs`` defaults to the AUTO plausibility threshold (3, so a random
    companion isn't mistaken for metadata).  An EXPLICIT ``meta_format`` passes
    ``min_pairs=1`` — a 1–2 field sidecar the user asked for must not be silently
    dropped (BL-3)."""
    metadata, pair_count = _read_structured_metadata(path)
    if pair_count >= min_pairs:
        return metadata
    return None


def _read_auto_candidate_metadata(path: Path) -> dict[str, MetadataValue] | None:
    """Try the appropriate parser for an auto-discovered sidecar."""
    ext = path.suffix.lower().lstrip(".")

    # Defensive cap (review follow-up): the .txt / .pdi primary parsers read the
    # WHOLE file (path.read_text()) on EVERY frame; the 1 MiB cap previously
    # guarded only the generic-structured route.  Bound the two first-class auto
    # routes the same way so a pathological/accidentally-huge companion cannot pin
    # memory or stall per-frame discovery (a real .pdi/.txt sidecar is a few KB).
    try:
        if path.stat().st_size > _AUTO_SIDECAR_MAX_BYTES:
            logger.warning(
                "auto metadata: skipping oversize sidecar %s (> %d-byte cap)",
                path, _AUTO_SIDECAR_MAX_BYTES)
            return None
    except OSError:
        return None

    if ext == "txt":
        metadata = read_txt_metadata(path)       # SSRL # Counters / # Motors
        if metadata:
            return metadata
        # BL-3: the SSRL .txt parser returns {} for a generic name=value .txt,
        # which would let a WORSE candidate latch — fall back to the structured
        # parser so a plausible generic .txt still wins.
        return _parse_structured_sidecar_if_plausible(path)

    if ext == "pdi":
        metadata = read_pdi_metadata(path)
        return metadata or None

    return _parse_structured_sidecar_if_plausible(path)


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
    image_path: Path,
) -> tuple[Path, str, str, dict[str, MetadataValue]] | None:
    for candidate, convention, suffix in _iter_auto_sidecar_candidates(image_path):
        metadata = _read_auto_candidate_metadata(candidate)
        if metadata is not None:
            return candidate, convention, suffix, metadata
    return None


def _read_auto_metadata(image_path: Path) -> dict[str, MetadataValue]:
    cached_sidecar = _auto_sidecar_from_cache(image_path)
    if cached_sidecar is not None:
        metadata = _read_auto_candidate_metadata(cached_sidecar)
        if metadata is not None:
            return metadata
        _AUTO_SIDECAR_CACHE.pop(_auto_cache_key(image_path), None)

    discovered = _discover_auto_sidecar(image_path)
    if discovered is None:
        logger.debug("read_image_metadata: no auto sidecar found for %s", image_path)
        return {}

    candidate, convention, suffix, metadata = discovered
    _AUTO_SIDECAR_CACHE[_auto_cache_key(image_path)] = (convention, suffix)
    # BL-3: log the convention auto locked onto so a wrong latch is visible in
    # the run log (it is then applied to EVERY frame via the cache).
    logger.info("metadata: auto locked onto %s-form '*%s' (%d fields, e.g. %s)",
                convention, suffix, len(metadata), candidate.name)
    return metadata


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


def read_txt_metadata(path: Path | str) -> dict[str, float]:
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
        data = path.read_text()

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


def read_pdi_metadata(path: Path | str) -> dict[str, float]:
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
        data = path.read_text().replace("\n", ";")

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
    image_path = Path(image_path)

    # Normalise case so callers can pass "SPEC" / "Spec" / "spec"
    # interchangeably — xdart's UI surfaces the value as "SPEC" in
    # the metadata format combobox, but the comparison below expects
    # lowercase.  Pre-fix this silently returned ``{}`` and logged
    # "unknown meta_format" for every SPEC frame, so positioners
    # never made it into the scan.
    meta_format_norm = (
        "auto" if meta_format is None else str(meta_format).strip().lower()
    )

    if meta_format_norm == "auto":
        return _read_auto_metadata(image_path)

    if meta_format_norm in ("txt", "pdi"):
        sidecar = _find_sidecar(image_path, meta_format_norm)
        if sidecar is None:
            logger.debug(
                "read_image_metadata: no %s sidecar found for %s",
                meta_format_norm,
                image_path,
            )
            return {}
        if meta_format_norm == "txt":
            return read_txt_metadata(sidecar)
        return read_pdi_metadata(sidecar)

    if meta_format_norm == "spec":
        # Normalise empty-string to None for downstream "use default
        # heuristic" branch.
        search_dir = (
            Path(meta_dir) if meta_dir not in (None, "") else None
        )
        return _read_spec_metadata(image_path, search_dir=search_dir)

    # Any other value is treated as a generic structured-sidecar extension.
    sidecar = _find_sidecar(image_path, meta_format_norm)
    if sidecar is None:
        logger.debug("read_image_metadata: no %r sidecar found for %s",
                     meta_format, image_path)
        return {}
    # BL-3: an EXPLICIT format is a deliberate user choice — accept a 1-2 field
    # sidecar (the AUTO min-pairs=3 plausibility gate does NOT apply here; before
    # this, 1-2-pair explicit sidecars were silently dropped with a misleading
    # "unknown meta_format" warning).
    metadata = _parse_structured_sidecar_if_plausible(sidecar, min_pairs=1)
    if metadata is not None:
        return metadata
    logger.warning("read_image_metadata: %s (format %r) had no readable "
                   "key=value fields", sidecar, meta_format)
    return {}


# ---------------------------------------------------------------------------
# SPEC metadata (private — called from read_image_metadata)
# ---------------------------------------------------------------------------


def _read_spec_metadata(
    image_path: Path,
    search_dir: Path | None = None,
) -> dict[str, float]:
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
        Returns ``{}`` on any error.
    """
    try:
        from silx.io.specfile import SpecFile  # noqa: PLC0415
    except ImportError:
        logger.warning("read_image_metadata (spec): silx is not installed")
        return {}

    try:
        spec_fname, scan_number, image_number = _extract_scan_info(image_path)
        if spec_fname is None or scan_number is None:
            logger.warning(
                "_read_spec_metadata: cannot parse scan info from filename %s",
                image_path.name,
            )
            return {}

        # Build the ordered search list — explicit dir first (when
        # given), then the default fallbacks.  Duplicates are harmless
        # but we skip None entries.
        candidate_dirs: list[Path] = []
        if search_dir is not None:
            candidate_dirs.append(Path(search_dir))
        # The image's own folder, then up to TWO levels of parent directories
        # above it (SSRL convention: the SPEC file may sit a couple levels up).
        candidate_dirs.extend([
            image_path.parent,
            image_path.parent.parent,
            image_path.parent.parent.parent,
        ])

        spec_file: Path | None = None
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
            return {}

        sf = SpecFile(str(spec_file))
        scan = sf[f"{scan_number}.1"]
        npts = scan.data.shape[1]
        img_idx = min(image_number, npts - 1)

        counters: dict[str, float] = {
            label: float(scan.data_column_by_name(label)[img_idx])
            for label in scan.labels
        }
        motors: dict[str, float] = {
            name: float(scan.motor_position_by_name(name))
            for name in scan.motor_names
        }

        return {**counters, **motors}

    except Exception:
        logger.warning(
            "_read_spec_metadata: failed to read SPEC metadata for %s",
            image_path,
            exc_info=True,
        )
        return {}
