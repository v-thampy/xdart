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


def _read_structured_metadata(path: Path | str) -> tuple[dict[str, MetadataValue], int]:
    """Read a generic structured sidecar with replacement decoding."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning("_read_structured_metadata: failed to read %s", path, exc_info=True)
        return {}, 0

    return _parse_structured_metadata(text)


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
    if not ext:
        return None

    candidate_replace = image_path.with_suffix(f".{ext}")
    if candidate_replace.exists():
        return candidate_replace

    candidate_append = Path(str(image_path) + f".{ext}")
    if candidate_append.exists():
        return candidate_append

    return None


def _parse_structured_sidecar_if_plausible(path: Path) -> dict[str, MetadataValue] | None:
    metadata, pair_count = _read_structured_metadata(path)
    if pair_count >= _STRUCTURED_SIDECAR_MIN_PAIRS:
        return metadata
    return None


def _read_auto_candidate_metadata(path: Path) -> dict[str, MetadataValue] | None:
    """Try the appropriate parser for an auto-discovered sidecar."""
    ext = path.suffix.lower().lstrip(".")

    if ext == "txt":
        metadata = read_txt_metadata(path)
        return metadata or None

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
    """Yield auto sidecar candidates as appended-name, then replaced-stem."""
    try:
        entries = sorted(image_path.parent.iterdir(), key=lambda p: p.name)
    except OSError:
        return

    seen: set[Path] = set()
    appended_prefix = image_path.name + "."
    for candidate in entries:
        if (
            candidate.name.startswith(appended_prefix)
            and _is_non_image_companion(candidate, image_path)
        ):
            seen.add(candidate)
            yield candidate, "append", candidate.name[len(image_path.name):]

    replaced_prefix = image_path.stem + "."
    for candidate in entries:
        if candidate in seen:
            continue
        if (
            candidate.name.startswith(replaced_prefix)
            and _is_non_image_companion(candidate, image_path)
        ):
            yield candidate, "replace", candidate.name[len(image_path.stem):]


def _auto_sidecar_from_cache(image_path: Path) -> Path | None:
    cached = _AUTO_SIDECAR_CACHE.get(_auto_cache_key(image_path))
    if cached is None:
        return None

    convention, suffix = cached
    if convention == "append":
        candidate = Path(str(image_path) + suffix)
    elif convention == "replace":
        candidate = image_path.with_name(image_path.stem + suffix)
    else:
        _AUTO_SIDECAR_CACHE.pop(_auto_cache_key(image_path), None)
        return None

    if _is_non_image_companion(candidate, image_path):
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

    _, convention, suffix, metadata = discovered
    _AUTO_SIDECAR_CACHE[_auto_cache_key(image_path)] = (convention, suffix)
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
                motors = {"TwoTheta": 0.0, "Theta": 0.0}
            counters = {}

        # --- epoch ---
        extras: dict[str, float] = {}
        tail = data[data.rindex(";") + 1 :]
        if tail:
            try:
                extras["epoch"] = float(tail)
            except ValueError:
                pass

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

    sidecar = _find_sidecar(image_path, meta_format_norm)
    if sidecar is not None:
        metadata = _parse_structured_sidecar_if_plausible(sidecar)
        if metadata is not None:
            return metadata
        logger.debug(
            "read_image_metadata: %s did not look like structured metadata",
            sidecar,
        )

    logger.warning("read_image_metadata: unknown meta_format %r", meta_format)
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
