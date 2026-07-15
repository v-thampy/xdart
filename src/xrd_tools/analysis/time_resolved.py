"""Headless tools for time-resolved 1-D XRD series.

The module keeps full 1-D stacks in an analysis-friendly xarray Dataset while
leaving raw detector frames and 2-D cakes on disk until explicitly requested.
All preprocessing returns a new Dataset, preserving the acquisition data and a
small provenance trail in attrs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import xarray as xr

from xrd_tools.analysis.axis_units import canonical_q_unit, require_inverse_angstrom
from xrd_tools.io import (
    get_1d,
    get_2d,
    get_metadata,
    get_raw_frame,
    get_thumbnail,
    read_scan_data,
)

__all__ = [
    "FrameLocator",
    "TimeResolvedSeries",
    "LinearThermalExpansion",
    "TabulatedThermalExpansion",
    "discover_processed_scans",
    "load_time_resolved_series",
    "normalize_monitor",
    "normalize_reference_band",
    "flag_normalization_outliers",
    "bin_time_resolved",
    "select_time_zero",
    "fit_peak_series",
    "flag_fit_quality",
    "lattice_from_q",
    "add_lattice_results",
    "temperature_rate",
    "add_temperature_results",
    "export_time_resolved_results",
]


def _natural_key(path: Path) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def discover_processed_scans(
    directory: str | Path,
    *,
    pattern: str = "*.nxs",
    recursive: bool = False,
) -> list[Path]:
    """Return naturally sorted processed 1-D NeXus files under ``directory``.

    Acquisition NeXus files and unrelated ``.nxs`` files can live beside
    processed results.  Inspect only their lightweight metadata here; the 1-D
    stacks remain untouched until :func:`load_time_resolved_series` is called.
    """
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = root.rglob(pattern) if recursive else root.glob(pattern)
    processed: list[Path] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            if get_metadata(path).get("has_1d", False):
                processed.append(path)
        except (OSError, KeyError, ValueError):
            # A non-xdart or partially written NeXus file is not a candidate.
            continue
    return sorted(processed, key=_natural_key)


@dataclass(frozen=True, slots=True)
class FrameLocator:
    """On-disk identity for one row of a :class:`TimeResolvedSeries`."""

    scan_file: Path
    frame_label: int


@dataclass(slots=True)
class TimeResolvedSeries:
    """Eager 1-D Dataset plus lazy raw/cake accessors.

    ``pattern`` arguments are zero-based row positions in ``dataset``. The
    original NeXus frame label remains available as ``dataset.frame_label``.
    """

    dataset: xr.Dataset
    locators: tuple[FrameLocator, ...]
    source_root: Path | None = None

    def _locator(self, pattern: int) -> FrameLocator:
        pos = int(pattern)
        if pos < 0:
            pos += len(self.locators)
        if pos < 0 or pos >= len(self.locators):
            raise IndexError(f"pattern {pattern} outside 0..{len(self.locators) - 1}")
        return self.locators[pos]

    def get_pattern(self, pattern: int, *, variable: str = "intensity") -> xr.DataArray:
        """Return one eager 1-D pattern as a labeled DataArray."""
        self._locator(pattern)
        return self.dataset[variable].isel(pattern=int(pattern))

    def get_cake(self, pattern: int):
        """Read one 2-D cake lazily from its processed file."""
        loc = self._locator(pattern)
        return get_2d(loc.scan_file, frame=loc.frame_label)

    def get_raw(self, pattern: int) -> np.ndarray:
        """Read one full raw detector frame, never a thumbnail substitute.

        :meth:`get_thumbnail` is intentionally the separate access path.  A
        missing or relocated raw source raises ``KeyError`` rather than quietly
        returning a small display image with different resolution and range.
        """
        loc = self._locator(pattern)
        return get_raw_frame(
            loc.scan_file,
            loc.frame_label,
            allow_thumbnail=False,
            source_root=self.source_root,
        )

    def get_thumbnail(self, pattern: int) -> np.ndarray:
        """Read the stored thumbnail without touching the raw source."""
        loc = self._locator(pattern)
        return get_thumbnail(loc.scan_file, loc.frame_label)


def _coerce_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths).expanduser()
        out = discover_processed_scans(path) if path.is_dir() else [path]
    else:
        out = [Path(path).expanduser() for path in paths]
    if not out:
        raise ValueError("no processed NeXus files selected")
    missing = [path for path in out if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    for path in out:
        try:
            is_processed = bool(get_metadata(path).get("has_1d", False))
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"{path} is not a readable processed 1-D NeXus file") from exc
        if not is_processed:
            raise ValueError(f"{path} is not a processed 1-D NeXus file")
    return out


def _period_for_path(
    frame_period_s: float | Mapping[str, float] | None,
    path: Path,
) -> float | None:
    if frame_period_s is None:
        return None
    if isinstance(frame_period_s, Mapping):
        candidates = (str(path), str(path.resolve()), path.name, path.stem)
        value = next((frame_period_s[key] for key in candidates if key in frame_period_s), None)
        if value is None:
            return None
    else:
        value = frame_period_s
    period = float(value)
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"frame_period_s must be finite and positive, got {value!r}")
    return period


_SECONDS_PER_TIME_UNIT = {
    "s": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
}


def _per_path_option(
    value: str | Mapping[str, str] | None,
    path: Path,
    *,
    name: str,
) -> str | None:
    """Resolve a scalar or path-keyed explicit loading option."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        candidates = (str(path), str(path.resolve()), path.name, path.stem)
        resolved = next((value[key] for key in candidates if key in value), None)
        return None if resolved is None else str(resolved)
    return str(value)


def _time_from_scan_data(
    scan_data: Mapping[str, Any],
    n_frames: int,
    *,
    time_key: str | None,
    time_unit: str | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return explicit persisted physical time in seconds, never guessed time."""
    if time_key is None:
        return None, None
    if time_unit is None:
        raise ValueError(
            f"time_key={time_key!r} requires an explicit time_unit; "
            "numeric scan-data columns carry no trusted unit metadata")
    unit_key = time_unit.strip().lower()
    factor = _SECONDS_PER_TIME_UNIT.get(unit_key)
    if factor is None:
        raise ValueError(
            f"time_unit={time_unit!r} is not a documented physical-time unit; "
            f"choose one of {sorted(_SECONDS_PER_TIME_UNIT)}")
    if time_key not in scan_data:
        raise KeyError(f"selected time_key {time_key!r} is absent from scan_data")
    values = np.asarray(scan_data[time_key], dtype=float).reshape(-1)
    if len(values) != n_frames or not np.all(np.isfinite(values)):
        raise ValueError(f"time_key {time_key!r} is not finite and frame-aligned")
    values = (values - values[0]) * factor
    if n_frames > 1 and np.any(np.diff(values) < 0):
        raise ValueError(f"time_key {time_key!r} must be monotonic within each scan")
    return values, f"scan_data:{time_key}[{unit_key}]"


def _interpolate_rows(rows: np.ndarray, old_q: np.ndarray, new_q: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.interp(new_q, old_q, row, left=np.nan, right=np.nan)
        for row in np.asarray(rows, dtype=float)
    ])


def _validate_q_grid(q: np.ndarray, *, path: Path) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)):
        raise ValueError(f"{path}: q grid must be a finite one-dimensional array")
    if np.any(np.diff(q) <= 0):
        raise ValueError(f"{path}: q grid must be strictly increasing")
    return q


def load_time_resolved_series(
    paths: str | Path | Sequence[str | Path],
    *,
    frame_period_s: float | Mapping[str, float] | None = None,
    time_key: str | Mapping[str, str] | None = None,
    time_unit: str | Mapping[str, str] | None = None,
    q_policy: str = "strict",
    reference_q: np.ndarray | None = None,
    metadata_keys: Sequence[str] = (),
    source_root: str | Path | None = None,
) -> TimeResolvedSeries:
    """Load one or more processed scans into a common 1-D xarray Dataset.

    Parameters
    ----------
    paths
        A processed file, a directory of ``*.nxs`` files, or explicit files.
    frame_period_s
        Scalar period for every file, a mapping keyed by full path/name/stem,
        or ``None``. It is used only when explicit persisted timing is not
        selected with ``time_key`` and ``time_unit``.
    time_key, time_unit
        Explicit persisted scan-data column and its proven physical unit. Both
        accept a scalar or a mapping keyed by full path/name/stem. Numeric
        columns are never treated as seconds based on their names. If neither
        explicit timing nor ``frame_period_s`` is supplied, time is expressed
        in frame-index units and rates are unavailable.
    q_policy
        ``"strict"`` requires identical q grids. ``"interpolate"`` maps each
        scan onto ``reference_q`` or the first scan's grid.
    reference_q
        Optional target q grid for interpolation.
    metadata_keys
        Persisted numeric per-frame ``scan_data`` columns to retain as
        pattern-aligned coordinates (for example ``("i0",)`` for monitor
        normalization). Missing values are represented as ``nan`` so a later
        normalization can make its own validity decision.
    source_root
        Optional moved-data root used only by lazy raw-frame access.
    """
    scan_paths = _coerce_paths(paths)
    if q_policy not in {"strict", "interpolate"}:
        raise ValueError("q_policy must be 'strict' or 'interpolate'")

    target_q = None
    if reference_q is not None:
        target_q = _validate_q_grid(np.asarray(reference_q, dtype=float), path=Path("reference_q"))
    metadata_keys = tuple(dict.fromkeys(str(key) for key in metadata_keys))
    reserved_metadata = {
        "pattern", "q", "scan_index", "scan_name", "scan_file",
        "frame_label", "frame_in_scan", "time", "sequence_time",
        "time_seconds", "time_source",
    }
    collisions = set(metadata_keys) & reserved_metadata
    if collisions:
        raise ValueError(f"metadata_keys collide with structural coordinates: {sorted(collisions)}")

    intensities: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    has_sigma = False
    locators: list[FrameLocator] = []
    scan_indices: list[int] = []
    scan_names: list[str] = []
    scan_files: list[str] = []
    frame_labels: list[int] = []
    frame_in_scan: list[int] = []
    physical_times: list[float] = []
    time_sources: list[str] = []
    metadata_values: dict[str, list[float]] = {key: [] for key in metadata_keys}
    scan_interpolated: list[bool] = []
    q_unit = None
    all_seconds = True

    for scan_index, path in enumerate(scan_paths):
        result = get_1d(path)
        q = _validate_q_grid(result.q, path=path)
        current_q_unit = canonical_q_unit(result.q_unit)
        if q_unit is None:
            q_unit = current_q_unit
        elif not q_unit and not current_q_unit:
            pass
        elif not q_unit or not current_q_unit:
            raise ValueError(
                f"{path}: every stacked scan must declare a compatible q unit; "
                f"reference={q_unit or 'unspecified'!r}, current={current_q_unit or 'unspecified'!r}"
            )
        elif current_q_unit != q_unit:
            raise ValueError(
                f"{path}: q unit {current_q_unit!r} differs from reference unit {q_unit!r}"
            )
        rows = np.asarray(result.intensity, dtype=float)
        if rows.ndim == 1:
            rows = rows[None, :]
        frames = np.asarray(result.frames, dtype=int).reshape(-1)
        if len(rows) != len(frames):
            raise ValueError(f"{path}: intensity rows do not match frame labels")

        if rows.ndim != 2 or rows.shape[1] != q.size:
            raise ValueError(f"{path}: intensity must have shape (frame, q)")
        if target_q is None:
            target_q = q.copy()
        compatible = q.shape == target_q.shape and np.allclose(
            q, target_q, rtol=1e-7, atol=1e-9, equal_nan=False)
        if not compatible:
            if q_policy == "strict":
                raise ValueError(
                    f"q grid in {path.name} differs from the reference grid; "
                    "use q_policy='interpolate' to align it explicitly")
            rows = _interpolate_rows(rows, q, target_q)

        sigma = result.sigma
        if sigma is None:
            sigma_rows = np.full((len(rows), len(target_q)), np.nan, dtype=float)
        else:
            has_sigma = True
            sigma_rows = np.asarray(sigma, dtype=float)
            if sigma_rows.ndim == 1:
                sigma_rows = sigma_rows[None, :]
            if sigma_rows.shape != (len(frames), len(q)):
                raise ValueError(f"{path}: sigma rows do not match intensity rows and q grid")
            if not compatible:
                sigma_rows = _interpolate_rows(sigma_rows, q, target_q)

        scan_data = read_scan_data(path, frames)
        selected_time_key = _per_path_option(time_key, path, name="time_key")
        selected_time_unit = _per_path_option(time_unit, path, name="time_unit")
        scan_time, time_source = _time_from_scan_data(
            scan_data,
            len(frames),
            time_key=selected_time_key,
            time_unit=selected_time_unit,
        )
        if scan_time is None:
            period = _period_for_path(frame_period_s, path)
            if period is None:
                scan_time = np.arange(len(frames), dtype=float)
                time_source = "frame_index"
                all_seconds = False
            else:
                scan_time = np.arange(len(frames), dtype=float) * period
                time_source = "frame_period_s"

        intensities.append(rows)
        sigmas.append(sigma_rows)
        scan_interpolated.extend([not compatible] * len(frames))
        for key in metadata_keys:
            value = scan_data.get(key)
            if value is None:
                metadata_values[key].extend([np.nan] * len(frames))
                continue
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size != len(frames):
                raise ValueError(f"{path}: metadata key {key!r} is not frame-aligned")
            metadata_values[key].extend(value.tolist())
        for row, frame in enumerate(frames):
            locators.append(FrameLocator(path, int(frame)))
            scan_indices.append(scan_index)
            scan_names.append(path.stem)
            scan_files.append(str(path))
            frame_labels.append(int(frame))
            frame_in_scan.append(row)
            physical_times.append(float(scan_time[row]) if time_source != "frame_index" else np.nan)
            time_sources.append(str(time_source))
    assert target_q is not None
    intensity_stack = np.concatenate(intensities, axis=0)
    sigma_stack = np.concatenate(sigmas, axis=0)
    n_patterns = len(intensity_stack)
    if all_seconds:
        time_values = np.asarray(physical_times, dtype=float)
        sequence_values = np.full(n_patterns, np.nan, dtype=float)
        offset = 0.0
        sequence_is_known = True
        for scan_index in range(len(scan_paths)):
            positions = np.flatnonzero(np.asarray(scan_indices) == scan_index)
            values = time_values[positions]
            if not sequence_is_known:
                continue
            if values.size > 1:
                step = float(np.nanmedian(np.diff(values)))
                step = step if np.isfinite(step) and step > 0 else None
            else:
                period = _period_for_path(frame_period_s, scan_paths[scan_index])
                step = period
            sequence_values[positions] = values + offset
            if step is None:
                sequence_is_known = False
            elif values.size:
                offset = float(values[-1] + offset + step)
        time_coordinate_unit = "s"
    else:
        # Retain physical values separately while exposing only unambiguous
        # frame-based coordinates when even one scan lacks physical timing.
        time_values = np.asarray(frame_in_scan, dtype=float)
        sequence_values = np.arange(n_patterns, dtype=float).tolist()
        time_coordinate_unit = "frame"

    coords: dict[str, Any] = {
        "pattern": np.arange(n_patterns, dtype=np.int64),
        "q": target_q,
        "scan_index": ("pattern", np.asarray(scan_indices, dtype=np.int64)),
        "scan_name": ("pattern", np.asarray(scan_names, dtype=str)),
        "scan_file": ("pattern", np.asarray(scan_files, dtype=str)),
        "frame_label": ("pattern", np.asarray(frame_labels, dtype=np.int64)),
        "frame_in_scan": ("pattern", np.asarray(frame_in_scan, dtype=np.int64)),
        "time": ("pattern", time_values),
        "sequence_time": ("pattern", sequence_values),
        "time_seconds": ("pattern", np.asarray(physical_times, dtype=float)),
        "time_source": ("pattern", np.asarray(time_sources, dtype=str)),
        "q_interpolated": ("pattern", np.asarray(scan_interpolated, dtype=bool)),
    }
    for key, values in metadata_values.items():
        coords[key] = ("pattern", np.asarray(values, dtype=float))
    data_vars: dict[str, Any] = {
        "intensity": (("pattern", "q"), intensity_stack),
    }
    if has_sigma:
        data_vars["sigma"] = (("pattern", "q"), sigma_stack)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.coords["q"].attrs["units"] = q_unit or ""
    ds.coords["time"].attrs["units"] = time_coordinate_unit
    ds.coords["sequence_time"].attrs["units"] = time_coordinate_unit
    ds.coords["sequence_time"].attrs["availability"] = (
        "nan after a scan boundary whose physical cadence is unknown")
    ds.coords["time_seconds"].attrs.update({
        "units": "s",
        "availability": "nan for scans without physical timing",
    })
    ds.attrs.update({
        "source_files": [str(path) for path in scan_paths],
        "q_policy": q_policy,
        "q_unit": q_unit or "",
        "time_units": time_coordinate_unit,
        "time_has_mixed_sources": not all_seconds,
        "time_contract": {
            "time_key": time_key,
            "time_unit": time_unit,
            "frame_period_s": frame_period_s,
            "numeric_column_units_inferred": False,
        },
        "metadata_keys": list(metadata_keys),
        "q_interpolation": {
            "target": "q coordinate",
            "outside_source_coverage": "nan",
            "interpolated_scan_files": [
                str(path) for index, path in enumerate(scan_paths)
                if any(
                    scan_interpolated[pos]
                    for pos, scan_index in enumerate(scan_indices)
                    if scan_index == index
                )
            ],
        },
    })
    return TimeResolvedSeries(
        dataset=ds,
        locators=tuple(locators),
        source_root=None if source_root is None else Path(source_root).expanduser(),
    )


def normalize_monitor(
    dataset: xr.Dataset,
    monitor: str | Sequence[float] | np.ndarray,
    *,
    intensity_var: str = "intensity",
    output_var: str = "intensity_normalized",
    target: str | float = "median",
) -> xr.Dataset:
    """Normalize each pattern by a persisted monitor or aligned numeric array.

    A string resolves a pattern-aligned coordinate or data variable, typically
    a key loaded through ``metadata_keys=("i0",)``.  Zero, missing, and
    non-finite monitor readings leave a ``nan`` row and are recorded in the
    validity mask; no diffraction profile is pointwise altered or clipped.
    """
    values = np.asarray(dataset[intensity_var].values, dtype=float)
    if values.ndim != 2 or values.shape[0] != dataset.sizes.get("pattern", 0):
        raise ValueError(f"{intensity_var!r} must have dimensions ('pattern', 'q')")
    if isinstance(monitor, str):
        if monitor in dataset.coords:
            monitor_values = dataset.coords[monitor].values
        elif monitor in dataset.data_vars:
            monitor_values = dataset[monitor].values
        else:
            raise KeyError(f"monitor {monitor!r} is not present in the dataset")
        monitor_name = monitor
    else:
        monitor_values = monitor
        monitor_name = "provided_array"
    factors = np.asarray(monitor_values, dtype=float).reshape(-1)
    if factors.size != values.shape[0]:
        raise ValueError("monitor must have one value per pattern")
    valid = np.isfinite(factors) & (np.abs(factors) > np.finfo(float).eps)
    if not np.any(valid):
        raise ValueError("monitor contains no finite, non-zero values")
    if target == "median":
        target_value = float(np.nanmedian(factors[valid]))
    elif isinstance(target, (int, float)):
        target_value = float(target)
    else:
        raise ValueError("target must be 'median' or a numeric value")
    normalized = np.full_like(values, np.nan, dtype=float)
    normalized[valid] = values[valid] / factors[valid, None] * target_value

    out = dataset.copy()
    out["monitor_normalization_factor"] = ("pattern", factors)
    out["monitor_normalization_valid"] = ("pattern", valid)
    out[output_var] = (("pattern", "q"), normalized)
    out["monitor_normalization_factor"].attrs.update({
        "monitor": monitor_name,
        "target": target_value,
    })
    out[output_var].attrs.update({
        "method": "monitor",
        "factor_variable": "monitor_normalization_factor",
        "source_variable": intensity_var,
    })
    return out


def normalize_reference_band(
    dataset: xr.Dataset,
    *,
    q_range: tuple[float, float],
    intensity_var: str = "intensity",
    output_var: str = "intensity_normalized",
    statistic: str = "median",
    target: str | float = "median",
) -> xr.Dataset:
    """Normalize each pattern by a peak-free q-band statistic.

    ``target='median'`` preserves the typical absolute intensity scale. Use a
    numeric target (commonly 1.0) for unit normalization.
    """
    q_lo, q_hi = map(float, q_range)
    if not np.isfinite([q_lo, q_hi]).all() or q_hi <= q_lo:
        raise ValueError("q_range must be finite and increasing")
    q = np.asarray(dataset.coords["q"].values, dtype=float)
    mask = (q >= q_lo) & (q <= q_hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"reference band {q_range} contains fewer than two q points")
    values = np.asarray(dataset[intensity_var].values, dtype=float)
    band = values[:, mask]
    if statistic == "median":
        factors = np.nanmedian(band, axis=1)
    elif statistic == "mean":
        factors = np.nanmean(band, axis=1)
    elif statistic == "integral":
        factors = np.trapezoid(band, x=q[mask], axis=1)
    else:
        raise ValueError("statistic must be 'median', 'mean', or 'integral'")
    valid = np.isfinite(factors) & (np.abs(factors) > np.finfo(float).eps)
    if not np.any(valid):
        raise ValueError("reference band produced no finite, non-zero factors")
    if target == "median":
        target_value = float(np.nanmedian(factors[valid]))
    elif isinstance(target, (int, float)):
        target_value = float(target)
    else:
        raise ValueError("target must be 'median' or a numeric value")
    normalized = np.full_like(values, np.nan, dtype=float)
    normalized[valid] = values[valid] / factors[valid, None] * target_value

    out = dataset.copy()
    out["normalization_factor"] = ("pattern", factors)
    out["normalization_valid"] = ("pattern", valid)
    out[output_var] = (("pattern", "q"), normalized)
    out["normalization_factor"].attrs.update({
        "q_range": (q_lo, q_hi),
        "statistic": statistic,
        "target": target_value,
    })
    out[output_var].attrs.update({
        "method": "reference_band",
        "factor_variable": "normalization_factor",
        "source_variable": intensity_var,
    })
    return out


def flag_normalization_outliers(
    dataset: xr.Dataset,
    *,
    factor_var: str = "normalization_factor",
    zmax: float = 6.0,
) -> xr.Dataset:
    """Flag common-mode intensity outliers using a robust median/MAD score."""
    factors = np.asarray(dataset[factor_var].values, dtype=float)
    valid = np.isfinite(factors)
    median = float(np.nanmedian(factors[valid])) if np.any(valid) else np.nan
    mad = float(np.nanmedian(np.abs(factors[valid] - median))) if np.any(valid) else np.nan
    score = np.zeros_like(factors, dtype=float)
    if np.isfinite(mad) and mad > 0:
        score[valid] = 0.67448975 * (factors[valid] - median) / mad
    score[~valid] = np.nan
    outlier = valid & (np.abs(score) > float(zmax))
    norm_valid = (
        np.asarray(dataset["normalization_valid"].values, dtype=bool)
        if "normalization_valid" in dataset else valid)
    out = dataset.copy()
    out["normalization_robust_z"] = ("pattern", score)
    out["normalization_outlier"] = ("pattern", outlier)
    out["pattern_valid"] = ("pattern", norm_valid & ~outlier)
    out["normalization_robust_z"].attrs["zmax"] = float(zmax)
    return out


def _contiguous_scan_chunks(dataset: xr.Dataset, bin_size: int) -> list[np.ndarray]:
    if "scan_index" in dataset.coords:
        scans = np.asarray(dataset.coords["scan_index"].values)
    else:
        scans = np.zeros(dataset.sizes["pattern"], dtype=int)
    chunks: list[np.ndarray] = []
    start = 0
    while start < len(scans):
        stop = start + 1
        while stop < len(scans) and scans[stop] == scans[start]:
            stop += 1
        for chunk_start in range(start, stop, bin_size):
            chunks.append(np.arange(chunk_start, min(chunk_start + bin_size, stop)))
        start = stop
    return chunks


def _reduce_chunk(values, indices, axis, *, reducer: str, sigma: bool, boolean: bool):
    block = np.take(values, indices, axis=axis)
    if boolean:
        return np.all(block, axis=axis)
    if sigma and reducer == "mean":
        count = np.sum(np.isfinite(block), axis=axis)
        result = np.sqrt(np.nansum(np.square(block), axis=axis))
        return np.divide(result, count, out=np.full_like(result, np.nan), where=count > 0)
    func = np.nanmean if reducer == "mean" else np.nanmedian
    return func(block, axis=axis)


def bin_time_resolved(
    dataset: xr.Dataset,
    *,
    bin_size: int,
    reducer: str = "mean",
) -> xr.Dataset:
    """Temporally bin patterns without crossing scan boundaries."""
    size = int(bin_size)
    if size < 1:
        raise ValueError("bin_size must be >= 1")
    if reducer not in {"mean", "median"}:
        raise ValueError("reducer must be 'mean' or 'median'")
    if size == 1:
        return dataset.copy()
    chunks = _contiguous_scan_chunks(dataset, size)

    data_vars: dict[str, Any] = {}
    for name, var in dataset.data_vars.items():
        if "pattern" not in var.dims:
            data_vars[name] = (var.dims, np.asarray(var.values))
            continue
        axis = var.get_axis_num("pattern")
        values = np.asarray(var.values)
        reduced = np.stack([
            _reduce_chunk(
                values,
                indices,
                axis,
                reducer=reducer,
                sigma=name.startswith("sigma"),
                boolean=values.dtype.kind == "b",
            )
            for indices in chunks
        ], axis=axis)
        data_vars[name] = (var.dims, reduced)

    coords: dict[str, Any] = {"pattern": np.arange(len(chunks), dtype=np.int64)}
    for name, coord in dataset.coords.items():
        if name == "pattern":
            continue
        if "pattern" not in coord.dims:
            coords[name] = (coord.dims, np.asarray(coord.values))
            continue
        values = np.asarray(coord.values)
        if values.dtype.kind in "iufc" and name not in {"scan_index", "frame_label", "frame_in_scan"}:
            reduced = np.asarray([np.nanmean(values[idx]) for idx in chunks])
        else:
            reduced = np.asarray([values[idx[0]] for idx in chunks])
        coords[name] = ("pattern", reduced)
    frame_labels = np.asarray(dataset.coords["frame_label"].values)
    coords["frame_label_start"] = (
        "pattern", np.asarray([frame_labels[idx[0]] for idx in chunks]))
    coords["frame_label_end"] = (
        "pattern", np.asarray([frame_labels[idx[-1]] for idx in chunks]))
    coords["frame_count"] = (
        "pattern", np.asarray([len(idx) for idx in chunks], dtype=np.int64))

    out = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(dataset.attrs))
    for name, var in dataset.data_vars.items():
        if name in out:
            out[name].attrs.update(var.attrs)
    for name, coord in dataset.coords.items():
        if name in out.coords:
            out.coords[name].attrs.update(coord.attrs)
    out.attrs["temporal_bin"] = {"size": size, "reducer": reducer}
    return out


def select_time_zero(
    dataset: xr.Dataset,
    *,
    zero_pattern: int,
    source_coord: str = "time",
    output_coord: str = "time_zeroed",
) -> xr.Dataset:
    """Add a time coordinate shifted so ``zero_pattern`` is zero."""
    pos = int(zero_pattern)
    values = np.asarray(dataset.coords[source_coord].values, dtype=float)
    if pos < 0:
        pos += len(values)
    if pos < 0 or pos >= len(values):
        raise IndexError(zero_pattern)
    out = dataset.copy()
    out = out.assign_coords({output_coord: ("pattern", values - values[pos])})
    out.coords[output_coord].attrs.update(dataset.coords[source_coord].attrs)
    out.coords[output_coord].attrs["zero_pattern"] = int(pos)
    return out


def fit_peak_series(
    dataset: xr.Dataset,
    plan,
    *,
    intensity_var: str | None = None,
    q_range: tuple[float, float] | None = None,
    pattern_indices: Sequence[int] | None = None,
    valid_only: bool = True,
    progress_callback: Callable[[int, int, Any], None] | None = None,
) -> xr.Dataset:
    """Fit peaks frame-by-frame and return compact xarray results.

    Fit curves and residuals are retained for interactive review. Heavy lmfit
    result objects are released after each pattern.
    """
    from xrd_tools.analysis.runner import AnalysisInput, PeakFitAnalyzer

    if intensity_var is None:
        intensity_var = (
            "intensity_normalized"
            if "intensity_normalized" in dataset else "intensity")
    q = np.asarray(dataset.coords["q"].values, dtype=float)
    if q_range is None:
        q_mask = np.ones_like(q, dtype=bool)
    else:
        q_mask = (q >= float(q_range[0])) & (q <= float(q_range[1]))
    if np.count_nonzero(q_mask) < 5:
        raise ValueError("fit q_range contains fewer than five points")
    q_fit = q[q_mask]

    if pattern_indices is None:
        indices = np.arange(dataset.sizes["pattern"], dtype=int)
    else:
        indices = np.asarray(pattern_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("pattern_indices must be one-dimensional")
        indices = np.where(indices < 0, indices + dataset.sizes["pattern"], indices)
        if np.any(indices < 0) or np.any(indices >= dataset.sizes["pattern"]):
            raise IndexError("pattern_indices contains an out-of-range position")
    if valid_only and "pattern_valid" in dataset:
        valid = np.asarray(dataset["pattern_valid"].values, dtype=bool)
        indices = indices[valid[indices]]
    if len(indices) == 0:
        raise ValueError("no patterns selected for fitting")
    analyzer = PeakFitAnalyzer(plan=plan)
    rows = np.asarray(dataset[intensity_var].values)
    params: list[dict[str, float]] = []
    success: list[bool] = []
    messages: list[str] = []
    fits: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    backgrounds: list[np.ndarray] = []

    for done, pos in enumerate(indices, start=1):
        label = (
            str(dataset.coords["frame_label"].values[pos])
            if "frame_label" in dataset.coords else str(pos))
        outcome = analyzer.analyze(AnalysisInput(
            label=label,
            x=q_fit,
            y=np.asarray(rows[pos, q_mask], dtype=float),
            x_unit=str(dataset.coords["q"].attrs.get("units", "")),
        ))
        params.append(dict(outcome.params))
        success.append(bool(outcome.ok))
        messages.append(str(outcome.message or ""))
        traces = outcome.overlay.traces if outcome.overlay is not None else {}
        fits.append(np.asarray(traces.get("fit", np.full_like(q_fit, np.nan)), dtype=float))
        residuals.append(np.asarray(
            traces.get("residual", np.full_like(q_fit, np.nan)), dtype=float))
        backgrounds.append(np.asarray(
            traces.get("background", np.full_like(q_fit, np.nan)), dtype=float))
        if progress_callback is not None:
            progress_callback(done, len(indices), outcome)

    keys = sorted({key for row in params for key in row})
    data_vars: dict[str, Any] = {
        "fit_success": ("fit_pattern", np.asarray(success, dtype=bool)),
        "fit_message": ("fit_pattern", np.asarray(messages, dtype=str)),
        "fit": (("fit_pattern", "q_fit"), np.asarray(fits, dtype=np.float32)),
        "residual": (
            ("fit_pattern", "q_fit"), np.asarray(residuals, dtype=np.float32)),
        "background": (
            ("fit_pattern", "q_fit"), np.asarray(backgrounds, dtype=np.float32)),
    }
    for key in keys:
        data_vars[key] = (
            "fit_pattern",
            np.asarray([row.get(key, np.nan) for row in params], dtype=float),
        )
    coords: dict[str, Any] = {
        "fit_pattern": np.arange(len(indices), dtype=np.int64),
        "q_fit": q_fit,
        "pattern": ("fit_pattern", np.asarray(dataset.coords["pattern"].values)[indices]),
    }
    for name, coord in dataset.coords.items():
        if name in {"pattern", "q"} or coord.dims != ("pattern",):
            continue
        coords[name] = ("fit_pattern", np.asarray(coord.values)[indices])
    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.coords["q_fit"].attrs.update(dataset.coords["q"].attrs)
    for name, coord in dataset.coords.items():
        if name in out.coords:
            out.coords[name].attrs.update(coord.attrs)
    try:
        plan_dict = asdict(plan)
    except TypeError:
        plan_dict = {"repr": repr(plan)}
    out.attrs.update({
        "analysis": "peak_fit_series",
        "intensity_variable": intensity_var,
        "q_range": None if q_range is None else tuple(map(float, q_range)),
        "plan": json.loads(json.dumps(plan_dict, default=str)),
    })
    return out


def flag_fit_quality(
    fit_dataset: xr.Dataset,
    *,
    max_redchi: float | None = None,
    max_center_error: float | None = None,
) -> xr.Dataset:
    """Add ``fit_valid`` without dropping failed or poorly constrained fits."""
    valid = np.asarray(fit_dataset["fit_success"].values, dtype=bool).copy()
    center_names = sorted(
        name for name in fit_dataset.data_vars
        if re.fullmatch(r"center_\d+", name))
    for name in center_names:
        valid &= np.isfinite(np.asarray(fit_dataset[name].values, dtype=float))
    if max_redchi is not None:
        if "redchi" not in fit_dataset:
            raise KeyError("fit dataset has no 'redchi' quality metric")
        redchi = np.asarray(fit_dataset["redchi"].values, dtype=float)
        valid &= np.isfinite(redchi) & (redchi <= float(max_redchi))
    if max_center_error is not None:
        error_names = sorted(
            name for name in fit_dataset.data_vars
            if re.fullmatch(r"center_err_\d+", name))
        if not error_names:
            raise KeyError("fit dataset has no center-error metrics")
        for name in error_names:
            error = np.asarray(fit_dataset[name].values, dtype=float)
            valid &= np.isfinite(error) & (error <= float(max_center_error))
    out = fit_dataset.copy()
    out["fit_valid"] = ("fit_pattern", valid)
    out["fit_valid"].attrs.update({
        "max_redchi": max_redchi,
        "max_center_error": max_center_error,
    })
    return out


def lattice_from_q(
    q_peak: Any,
    hkl: Sequence[int],
    *,
    q_unit: str | None,
) -> np.ndarray:
    """Return cubic lattice parameter in angstrom from inverse-angstrom q."""
    require_inverse_angstrom(q_unit, operation="q-to-lattice conversion")
    if len(hkl) != 3:
        raise ValueError("hkl must contain exactly three indices")
    norm = float(np.sqrt(sum(float(v) ** 2 for v in hkl)))
    if norm <= 0:
        raise ValueError("hkl cannot be (0, 0, 0)")
    q = np.asarray(q_peak, dtype=float)
    return 2.0 * np.pi * norm / q


def add_lattice_results(
    fit_dataset: xr.Dataset,
    *,
    hkls: Sequence[Sequence[int]],
) -> xr.Dataset:
    """Convert fitted inverse-angstrom q centers to cubic lattice parameters."""
    hkls = tuple(tuple(int(v) for v in hkl) for hkl in hkls)
    if not hkls:
        raise ValueError("at least one reflection is required")
    q_unit = (
        fit_dataset.coords["q_fit"].attrs.get("units")
        if "q_fit" in fit_dataset.coords else None
    )
    require_inverse_angstrom(q_unit, operation="q-to-lattice conversion")
    lattice_rows = []
    error_rows = []
    for i, hkl in enumerate(hkls):
        center_name = f"center_{i}"
        if center_name not in fit_dataset:
            raise KeyError(f"fit dataset has no {center_name!r}")
        q = np.asarray(fit_dataset[center_name].values, dtype=float)
        a = lattice_from_q(q, hkl, q_unit=q_unit)
        error_name = f"center_err_{i}"
        if error_name in fit_dataset:
            q_err = np.asarray(fit_dataset[error_name].values, dtype=float)
            a_err = np.abs(a * q_err / q)
        else:
            a_err = np.full_like(a, np.nan)
        lattice_rows.append(a)
        error_rows.append(a_err)
    lattice = np.stack(lattice_rows, axis=1)
    errors = np.stack(error_rows, axis=1)
    if "fit_valid" in fit_dataset:
        fit_valid = np.asarray(fit_dataset["fit_valid"].values, dtype=bool)
        lattice[~fit_valid] = np.nan
        errors[~fit_valid] = np.nan
    mean = np.full(lattice.shape[0], np.nan)
    mean_err = np.full(lattice.shape[0], np.nan)
    for row in range(lattice.shape[0]):
        valid = np.isfinite(lattice[row])
        weighted = valid & np.isfinite(errors[row]) & (errors[row] > 0)
        if np.any(weighted):
            weights = 1.0 / np.square(errors[row, weighted])
            mean[row] = np.sum(lattice[row, weighted] * weights) / np.sum(weights)
            mean_err[row] = np.sqrt(1.0 / np.sum(weights))
        elif np.any(valid):
            mean[row] = np.nanmedian(lattice[row, valid])

    labels = ["(" + "".join(str(v) for v in hkl) + ")" for hkl in hkls]
    out = fit_dataset.copy()
    out = out.assign_coords(reflection=np.asarray(labels, dtype=str))
    out["lattice_A"] = (("fit_pattern", "reflection"), lattice)
    out["lattice_error_A"] = (("fit_pattern", "reflection"), errors)
    out["lattice_mean_A"] = ("fit_pattern", mean)
    out["lattice_mean_error_A"] = ("fit_pattern", mean_err)
    out.coords["reflection"].attrs["hkls"] = hkls
    out["lattice_A"].attrs["units"] = "angstrom"
    out["lattice_mean_A"].attrs["units"] = "angstrom"
    out.attrs["lattice_q_unit"] = q_unit
    return out


@dataclass(frozen=True, slots=True)
class LinearThermalExpansion:
    """Explicit linear lattice-temperature calibration.

    This is a model assumption, not an intrinsic conversion. Thin-film stress
    and temperature-dependent expansion can require a richer calibration.
    """

    reference_lattice_A: float
    reference_temperature_K: float
    alpha_per_K: float

    def __post_init__(self):
        values = (
            self.reference_lattice_A,
            self.reference_temperature_K,
            self.alpha_per_K,
        )
        if not np.isfinite(values).all() or self.reference_lattice_A <= 0:
            raise ValueError("thermal calibration values must be finite")
        if self.alpha_per_K == 0:
            raise ValueError("alpha_per_K cannot be zero")

    def temperature(self, lattice_A: Any) -> np.ndarray:
        lattice = np.asarray(lattice_A, dtype=float)
        strain = lattice / self.reference_lattice_A - 1.0
        return self.reference_temperature_K + strain / self.alpha_per_K

    def lattice(self, temperature_K: Any) -> np.ndarray:
        temperature = np.asarray(temperature_K, dtype=float)
        return self.reference_lattice_A * (
            1.0 + self.alpha_per_K * (temperature - self.reference_temperature_K))

    def temperature_derivative_per_A(self, lattice_A: Any) -> np.ndarray:
        """Return dT/da for uncertainty propagation."""
        lattice = np.asarray(lattice_A, dtype=float)
        return np.full_like(
            lattice,
            1.0 / (self.reference_lattice_A * self.alpha_per_K),
            dtype=float,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TabulatedThermalExpansion:
    """Monotonic lattice-temperature calibration from reference data."""

    temperature_K: tuple[float, ...]
    lattice_A: tuple[float, ...]

    def __post_init__(self):
        temperature = np.asarray(self.temperature_K, dtype=float)
        lattice = np.asarray(self.lattice_A, dtype=float)
        if temperature.ndim != 1 or lattice.shape != temperature.shape or len(lattice) < 2:
            raise ValueError("temperature_K and lattice_A need matching 1-D tables")
        if not np.isfinite(temperature).all() or not np.isfinite(lattice).all():
            raise ValueError("thermal calibration table must be finite")
        if np.any(np.diff(temperature) <= 0):
            raise ValueError("temperature_K must be strictly increasing")
        if not (np.all(np.diff(lattice) > 0) or np.all(np.diff(lattice) < 0)):
            raise ValueError("lattice_A must be monotonic for inversion")

    def temperature(self, lattice_A: Any) -> np.ndarray:
        lattice = np.asarray(self.lattice_A, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        if lattice[0] > lattice[-1]:
            lattice = lattice[::-1]
            temperature = temperature[::-1]
        return np.interp(np.asarray(lattice_A, dtype=float), lattice, temperature,
                         left=np.nan, right=np.nan)

    def lattice(self, temperature_K: Any) -> np.ndarray:
        return np.interp(
            np.asarray(temperature_K, dtype=float),
            np.asarray(self.temperature_K, dtype=float),
            np.asarray(self.lattice_A, dtype=float),
            left=np.nan,
            right=np.nan,
        )

    def temperature_derivative_per_A(self, lattice_A: Any) -> np.ndarray:
        """Piecewise dT/da, ``nan`` outside the calibrated lattice range."""
        lattice = np.asarray(self.lattice_A, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        if lattice[0] > lattice[-1]:
            lattice = lattice[::-1]
            temperature = temperature[::-1]
        slope = np.diff(temperature) / np.diff(lattice)
        values = np.asarray(lattice_A, dtype=float)
        indices = np.searchsorted(lattice, values, side="right") - 1
        valid = (indices >= 0) & (indices < len(slope))
        # The upper endpoint belongs to the final finite interval.
        valid |= values == lattice[-1]
        indices = np.clip(indices, 0, len(slope) - 1)
        result = np.full_like(values, np.nan, dtype=float)
        result[valid] = slope[indices[valid]]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_K": list(self.temperature_K),
            "lattice_A": list(self.lattice_A),
        }


def temperature_rate(
    time_s: Any,
    temperature_K: Any,
    *,
    smooth_window: int | None = None,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed temperature and ``dT/dt`` in K/s."""
    time = np.asarray(time_s, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    if time.shape != temperature.shape or time.ndim != 1:
        raise ValueError("time and temperature must be matching 1-D arrays")
    valid = np.isfinite(time) & np.isfinite(temperature)
    if np.count_nonzero(valid) < 2:
        return temperature.copy(), np.full_like(temperature, np.nan)
    valid_time = time[valid]
    if np.any(np.diff(valid_time) <= 0):
        raise ValueError("time must be strictly increasing; select one scan or use sequence_time")
    work = temperature.copy()
    if not np.all(valid):
        work[~valid] = np.interp(time[~valid], valid_time, temperature[valid])
    if smooth_window is not None and int(smooth_window) >= 3:
        from scipy.signal import savgol_filter

        window = int(smooth_window)
        if window % 2 == 0:
            window += 1
        window = min(window, len(work) if len(work) % 2 else len(work) - 1)
        if window >= 3:
            work = savgol_filter(work, window_length=window,
                                 polyorder=min(int(polyorder), window - 1))
    rate = np.gradient(work, time)
    work[~valid] = np.nan
    rate[~valid] = np.nan
    return work, rate


def add_temperature_results(
    lattice_dataset: xr.Dataset,
    calibration: Any,
    *,
    lattice_var: str = "lattice_mean_A",
    time_coord: str = "time",
    smooth_window: int | None = None,
    polyorder: int = 2,
) -> xr.Dataset:
    """Add temperature, optional uncertainty, and rate from a calibration.

    ``time_coord`` must explicitly be in seconds.  The routine does not infer
    K/s from frame indices, and it records whether a calibration exposed a
    derivative suitable for propagating lattice uncertainty.
    """
    if time_coord not in lattice_dataset.coords:
        raise KeyError(f"dataset has no {time_coord!r} coordinate")
    time_units = str(lattice_dataset.coords[time_coord].attrs.get("units", ""))
    if time_units not in {"s", "second", "seconds"}:
        raise ValueError(
            f"{time_coord!r} must have physical seconds units before computing K/s; "
            f"got {time_units or 'unspecified'!r}"
        )
    lattice = np.asarray(lattice_dataset[lattice_var].values, dtype=float)
    temperature = calibration.temperature(lattice)
    time = np.asarray(lattice_dataset.coords[time_coord].values, dtype=float)
    smooth, rate = temperature_rate(
        time, temperature, smooth_window=smooth_window, polyorder=polyorder)
    dim = lattice_dataset[lattice_var].dims[0]
    out = lattice_dataset.copy()
    out["temperature_K"] = (dim, temperature)
    out["temperature_C"] = (dim, temperature - 273.15)
    out["temperature_smoothed_K"] = (dim, smooth)
    out["temperature_rate_K_per_s"] = (dim, rate)
    lattice_error_name = (
        "lattice_mean_error_A"
        if lattice_var == "lattice_mean_A" else lattice_var.replace("_A", "_error_A")
    )
    derivative = getattr(calibration, "temperature_derivative_per_A", None)
    if lattice_error_name in lattice_dataset and callable(derivative):
        lattice_error = np.asarray(lattice_dataset[lattice_error_name].values, dtype=float)
        temp_error = np.abs(np.asarray(derivative(lattice), dtype=float) * lattice_error)
        out["temperature_error_K"] = (dim, temp_error)
        out["temperature_error_K"].attrs.update({
            "units": "K",
            "method": "first_order_lattice_uncertainty_propagation",
            "lattice_error_variable": lattice_error_name,
        })
    out["temperature_K"].attrs.update({
        "units": "K",
        "calibration": calibration.to_dict(),
        "warning": (
            "includes thermal and mechanical/anisotropic lattice strain; "
            "texture and disagreement among reflections are diagnostics, not noise"
        ),
    })
    out["temperature_rate_K_per_s"].attrs["units"] = "K/s"
    out["temperature_rate_K_per_s"].attrs.update({
        "time_coordinate": time_coord,
        "smoothing_window": smooth_window,
        "polyorder": polyorder if smooth_window is not None else None,
    })
    out.attrs["thermal_calibration"] = calibration.to_dict()
    out.attrs["temperature_uncertainty"] = (
        "propagated from lattice uncertainty"
        if "temperature_error_K" in out else "not available from this calibration/result"
    )
    return out


def _json_attr(value: Any) -> Any:
    """Convert an xarray attribute to a NetCDF-safe scalar or JSON string."""
    if isinstance(value, (str, int, float, np.number, bool)):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _netcdf_safe_copy(dataset: xr.Dataset) -> xr.Dataset:
    """Return a shallow result copy whose attrs can be persisted by xarray."""
    out = dataset.copy(deep=False)
    out.attrs = {key: _json_attr(value) for key, value in out.attrs.items()}
    for variable in out.variables.values():
        variable.attrs = {key: _json_attr(value) for key, value in variable.attrs.items()}
    return out


def export_time_resolved_results(
    dataset: xr.Dataset,
    *,
    netcdf_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    scalar_vars: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Explicitly export compact time-resolved results as NetCDF and/or CSV.

    The caller chooses every output path.  This function never reads or copies
    raw detector frames or 2-D cakes; it only persists the xarray result that
    the analysis has already constructed.  CSV contains scalar variables on a
    single ``pattern`` or ``fit_pattern`` axis, while NetCDF retains compact
    fit/background/residual curves and serializable provenance attrs.
    """
    if netcdf_path is None and csv_path is None:
        raise ValueError("choose netcdf_path and/or csv_path")
    paths: dict[str, Path] = {}
    if netcdf_path is not None:
        path = Path(netcdf_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        export = _netcdf_safe_copy(dataset)
        export.attrs.setdefault("export_kind", "time_resolved_compact_results")
        export.to_netcdf(path)
        paths["netcdf"] = path

    if csv_path is not None:
        import pandas as pd

        path = Path(csv_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        dimensions = [dim for dim in ("fit_pattern", "pattern") if dim in dataset.sizes]
        if not dimensions:
            raise ValueError("dataset needs a pattern or fit_pattern dimension for CSV export")
        dim = dimensions[0]
        names = list(scalar_vars) if scalar_vars is not None else [
            name for name, var in dataset.data_vars.items() if var.dims == (dim,)
        ]
        missing = [name for name in names if name not in dataset]
        if missing:
            raise KeyError(f"dataset has no requested scalar variables: {missing}")
        columns: dict[str, Any] = {}
        for name, coord in dataset.coords.items():
            if coord.dims == (dim,):
                columns[name] = np.asarray(coord.values)
        for name in names:
            if dataset[name].dims != (dim,):
                raise ValueError(f"{name!r} is not scalar on {dim!r}")
            columns[name] = np.asarray(dataset[name].values)
        pd.DataFrame(columns).to_csv(path, index=False)
        paths["csv"] = path
    return paths
