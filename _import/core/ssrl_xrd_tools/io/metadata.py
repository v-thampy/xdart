"""Metadata parsing: read_txt_metadata, read_pdi_metadata, read_image_metadata."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["read_txt_metadata", "read_pdi_metadata", "read_image_metadata"]


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
    candidate_replace = image_path.with_suffix(f".{ext}")
    if candidate_replace.exists():
        return candidate_replace

    candidate_append = Path(str(image_path) + f".{ext}")
    if candidate_append.exists():
        return candidate_append

    return None


def _extract_scan_info(image_path: Path) -> tuple[str | None, int | None, int]:
    """Parse SSRL filename conventions to extract scan metadata.

    Expected filename pattern::

        {prefix}_{specname}_scan{N}_{M}.{ext}

    where ``N`` is the scan number and ``M`` is the image number within that
    scan.  A leading ``b_`` prefix (background files) is stripped before
    parsing.

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
    if stem.startswith("b_"):
        stem = stem[2:]

    match = re.search(r"_scan(\d+)_(\d+)$", stem)
    if match is None:
        return None, None, 0

    scan_number = int(match.group(1))
    image_number = int(match.group(2))
    spec_fname = stem[stem.find("_") + 1 : match.start()]

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
    meta_format: str = "txt",
) -> dict[str, float]:
    """Unified metadata reader for SSRL detector image sidecar files.

    Parameters
    ----------
    image_path : Path | str
        Path to the detector image file.
    meta_format : str
        One of ``"txt"``, ``"pdi"``, or ``"spec"``.

    Returns
    -------
    dict[str, float]
        Flat mapping of metadata key → value.  Returns ``{}`` when no sidecar
        is found or parsing fails.
    """
    image_path = Path(image_path)

    if meta_format in ("txt", "pdi"):
        sidecar = _find_sidecar(image_path, meta_format)
        if sidecar is None:
            logger.debug(
                "read_image_metadata: no %s sidecar found for %s",
                meta_format,
                image_path,
            )
            return {}
        if meta_format == "txt":
            return read_txt_metadata(sidecar)
        return read_pdi_metadata(sidecar)

    if meta_format == "spec":
        return _read_spec_metadata(image_path)

    logger.warning("read_image_metadata: unknown meta_format %r", meta_format)
    return {}


# ---------------------------------------------------------------------------
# SPEC metadata (private — called from read_image_metadata)
# ---------------------------------------------------------------------------


def _read_spec_metadata(image_path: Path) -> dict[str, float]:
    """Extract per-image counters and motor positions from a SPEC file.

    The SPEC file is located by searching the image's directory and its
    immediate parent.  The scan and image numbers are inferred from the
    filename stem using SSRL naming conventions.

    Parameters
    ----------
    image_path : Path
        Path to the detector image file.

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

        spec_file: Path | None = None
        for search_dir in (image_path.parent, image_path.parent.parent):
            candidate = search_dir / spec_fname
            if candidate.is_file():
                spec_file = candidate
                break

        if spec_file is None:
            logger.warning(
                "_read_spec_metadata: SPEC file %r not found near %s",
                spec_fname,
                image_path,
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
