"""
Tiled client reader for Bluesky scan data.

Tiled is an optional dependency. All ``tiled.client`` imports are guarded —
this module is safe to import even when tiled is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ssrl_xrd_tools.core.metadata import ScanMetadata
from ssrl_xrd_tools.transforms import energy_to_wavelength

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tiled import
# ---------------------------------------------------------------------------

_HAS_TILED = False
try:
    from tiled.client import from_uri as _tiled_from_uri  # type: ignore[import-untyped]
    _HAS_TILED = True
except ImportError:
    pass


def _require_tiled() -> None:
    if not _HAS_TILED:
        raise ImportError(
            "tiled is required: pip install tiled[client]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def connect_tiled(
    uri: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create and return a Tiled catalog client.

    Thin wrapper around ``tiled.client.from_uri`` so callers don't need to
    import tiled directly.

    Parameters
    ----------
    uri : str
        Tiled server URI (e.g., ``"https://tiled.ssrl.slac.stanford.edu"``).
    api_key : str, optional
        API key for authentication.
    **kwargs
        Additional keyword arguments passed to ``tiled.client.from_uri``.

    Returns
    -------
    tiled.client.CatalogClient

    Raises
    ------
    ImportError
        If tiled is not installed.
    """
    _require_tiled()
    if api_key is not None:
        kwargs["api_key"] = api_key
    return _tiled_from_uri(uri, **kwargs)  # type: ignore[name-defined]


def read_tiled_run(
    client: Any,
    scan_id: str | int,
    motor_names: list[str] | None = None,
    counter_names: list[str] | None = None,
    stream: str = "primary",
) -> ScanMetadata:
    """
    Read a Bluesky run from a Tiled catalog and return ``ScanMetadata``.

    Parameters
    ----------
    client : tiled.client.CatalogClient or similar
        Connected Tiled catalog client.  The caller is responsible for
        authentication and connection setup (see :func:`connect_tiled`).
    scan_id : str or int
        Scan identifier — a uid string or integer scan_id.
    motor_names : list of str, optional
        Motor column names to extract from the data stream.  If *None*, the
        run's ``start.motors`` metadata field is used.
    counter_names : list of str, optional
        Counter/detector names to extract.  If *None*, the run's
        ``start.detectors`` metadata field is used.
    stream : str, optional
        Data stream name (default ``"primary"``).

    Returns
    -------
    ScanMetadata

    Raises
    ------
    ImportError
        If tiled is not installed.
    KeyError
        If ``scan_id`` is not found in the catalog.
    """
    _require_tiled()

    try:
        run = client[scan_id]
    except KeyError:
        raise KeyError(f"Scan {scan_id!r} not found in Tiled catalog")

    start: dict[str, Any] = run.metadata.get("start", {})

    # --- energy & wavelength ------------------------------------------------
    energy = _extract_energy(run, start, stream)
    wavelength = _extract_wavelength(start, energy)

    # --- UB matrix ----------------------------------------------------------
    ub_matrix = _extract_ub(start)

    # --- sample name --------------------------------------------------------
    sample_name = str(start.get("sample_name", ""))

    # --- per-point arrays from the data stream ------------------------------
    if motor_names is None:
        motor_names = list(start.get("motors", []))
    if counter_names is None:
        counter_names = list(start.get("detectors", []))

    angles, counters = _read_stream(run, stream, motor_names, counter_names)

    return ScanMetadata(
        scan_id=str(scan_id),
        energy=energy,
        wavelength=wavelength,
        angles=angles,
        counters=counters,
        ub_matrix=ub_matrix,
        sample_name=sample_name,
        scan_type=str(start.get("plan_name", "")),
        source="tiled",
        h5_path=None,
    )


def list_scans(
    client: Any,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    List recent scans in a Tiled catalog.

    Returns a list of dicts with keys ``scan_id``, ``uid``,
    ``plan_name``, ``sample_name``, and ``num_points``.

    Parameters
    ----------
    client : tiled.client.CatalogClient
        Connected Tiled catalog.
    limit : int, optional
        Maximum number of scans to return (default 50).

    Returns
    -------
    list of dict
        Each dict has keys: ``scan_id``, ``uid``, ``plan_name``,
        ``sample_name``, ``num_points``.

    Raises
    ------
    ImportError
        If tiled is not installed.
    """
    _require_tiled()

    results: list[dict[str, Any]] = []
    for _, run in _iter_client(client, limit):
        start: dict[str, Any] = {}
        try:
            start = run.metadata.get("start", {})
        except Exception:
            logger.debug("Could not read metadata for run", exc_info=True)

        results.append(
            {
                "scan_id": start.get("scan_id", ""),
                "uid": start.get("uid", ""),
                "plan_name": start.get("plan_name", ""),
                "sample_name": start.get("sample_name", ""),
                "num_points": start.get("num_points", None),
            }
        )
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _iter_client(client: Any, limit: int):
    """Yield (key, run) pairs from a Tiled catalog, up to *limit* items."""
    try:
        items = client.items()
    except AttributeError:
        # Some client versions expose __iter__ directly
        items = enumerate(client)
    count = 0
    for key, run in items:
        if count >= limit:
            break
        yield key, run
        count += 1


def _extract_energy(run: Any, start: dict[str, Any], stream: str) -> float:
    """
    Attempt to read beam energy (keV).

    Checks (in order):
    1. ``start["energy"]``
    2. ``run[stream]["energy"]`` scalar column (first value)
    """
    raw = start.get("energy")
    if raw is not None:
        try:
            val = float(np.asarray(raw).ravel()[0])
            if np.isfinite(val):
                return val
        except Exception:
            pass

    # Try to read from the data stream
    try:
        ds = run[stream].read()
        if "energy" in ds:
            arr = np.asarray(ds["energy"].values, dtype=float).ravel()
            if len(arr) > 0 and np.isfinite(arr[0]):
                return float(arr[0])
    except Exception:
        logger.debug("Could not read energy from stream %r", stream, exc_info=True)

    logger.warning("Energy not found in Tiled run; using NaN")
    return float(np.nan)


def _extract_wavelength(start: dict[str, Any], energy: float) -> float:
    """
    Attempt to read wavelength (Å) or derive from energy.

    Checks (in order):
    1. ``start["wavelength"]``
    2. Compute via ``energy_to_wavelength(energy)``
    """
    raw = start.get("wavelength")
    if raw is not None:
        try:
            val = float(np.asarray(raw).ravel()[0])
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    if np.isfinite(energy) and energy > 0:
        return float(energy_to_wavelength(energy))

    logger.warning("Wavelength not derivable; using NaN")
    return float(np.nan)


def _extract_ub(start: dict[str, Any]) -> np.ndarray | None:
    """
    Extract the 3×3 UB matrix from the start document.

    Accepts flat (9-element) or shaped (3×3) arrays.
    """
    raw = start.get("ub_matrix")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float)
        return arr.reshape(3, 3)
    except Exception:
        logger.warning("Could not parse ub_matrix from start document", exc_info=True)
        return None


def _read_stream(
    run: Any,
    stream: str,
    motor_names: list[str],
    counter_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract per-point arrays from a Tiled data stream.

    Parameters
    ----------
    run : tiled run object
    stream : str
        Stream name (e.g. ``"primary"``).
    motor_names : list of str
    counter_names : list of str

    Returns
    -------
    angles : dict[str, np.ndarray]
    counters : dict[str, np.ndarray]
    """
    angles: dict[str, np.ndarray] = {}
    counters: dict[str, np.ndarray] = {}

    if not motor_names and not counter_names:
        return angles, counters

    try:
        ds = run[stream].read()
    except Exception:
        logger.warning("Could not read stream %r", stream, exc_info=True)
        return angles, counters

    for name in motor_names:
        if name in ds:
            try:
                angles[name] = np.asarray(ds[name].values, dtype=float).ravel()
            except Exception:
                logger.debug("Could not read motor %r from stream", name, exc_info=True)
        else:
            logger.debug("Motor %r not found in stream %r", name, stream)

    for name in counter_names:
        if name in ds:
            try:
                counters[name] = np.asarray(ds[name].values, dtype=float).ravel()
            except Exception:
                logger.debug("Could not read counter %r from stream", name, exc_info=True)
        else:
            logger.debug("Counter %r not found in stream %r", name, stream)

    return angles, counters
