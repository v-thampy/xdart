"""FrameView readers for processed xdart/ssrl NeXus scans."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from ssrl_xrd_tools.core.frame_view import (
    FrameGeometry,
    FrameView,
    TwoDKind,
    axis_from_unit,
    numeric_metadata,
    two_d_kind_from_units,
)
from ssrl_xrd_tools.io.read import (
    _decode,
    _dequantize_thumbnail,
    _entry,
    _frame_index,
    _resolve_positions,
    _scan_data_for_frames,
    _slice_stack,
    get_1d,
    get_2d,
    get_frames,
)


def _decode_kind(value, x_unit: str | None, y_unit: str | None) -> TwoDKind:
    if value is not None:
        try:
            return TwoDKind(str(_decode(value)))
        except ValueError:
            pass
    return two_d_kind_from_units(x_unit, y_unit)


def _read_2d_sigma_and_kind(
    scan_file: Path,
    frame: int,
    *,
    entry: str,
) -> tuple[np.ndarray | None, TwoDKind]:
    with h5py.File(scan_file, "r") as f:
        e = _entry(f, entry)
        if "integrated_2d" not in e:
            return None, TwoDKind.Q_CHI
        g = e["integrated_2d"]
        q_unit = _decode(g["q"].attrs.get("units")) if "q" in g and "units" in g["q"].attrs else None
        chi_unit = _decode(g["chi"].attrs.get("units")) if "chi" in g and "units" in g["chi"].attrs else None
        kind = _decode_kind(g.attrs.get("two_d_kind"), q_unit, chi_unit)
        if "sigma" not in g:
            return None, kind
        positions, _, single = _resolve_positions(_frame_index(e, prefer="integrated_2d"), frame)
        return _slice_stack(g["sigma"], positions, single), kind


def _read_thumbnail(
    scan_file: Path,
    frame: int,
    *,
    entry: str,
) -> tuple[np.ndarray | None, bool]:
    with h5py.File(scan_file, "r") as f:
        e = _entry(f, entry)
        fg = e.get(f"frames/frame_{int(frame):04d}")
        if fg is None or "thumbnail" not in fg:
            return None, False
        return _dequantize_thumbnail(fg["thumbnail"]), True


def _read_source_ref(
    scan_file: Path,
    frame: int,
    *,
    entry: str,
) -> tuple[str | None, int | None]:
    with h5py.File(scan_file, "r") as f:
        e = _entry(f, entry)
        fg = e.get(f"frames/frame_{int(frame):04d}")
        if fg is None or "source" not in fg:
            return None, None
        src = fg["source"]
        path = None
        source_idx = None
        if "path" in src:
            path = str(_decode(src["path"][()]))
        if "frame_index" in src:
            source_idx = int(np.asarray(src["frame_index"][()]).ravel()[0])
        return path, source_idx


def _read_geometry(
    scan_file: Path,
    frame: int,
    *,
    entry: str,
) -> FrameGeometry | None:
    with h5py.File(scan_file, "r") as f:
        e = _entry(f, entry)
        if "per_frame_geometry" not in e:
            return None
        g = e["per_frame_geometry"]
        positions, _, single = _resolve_positions(_frame_index(e, prefer="per_frame_geometry"), frame)

        def _read_scalar(name: str) -> float | None:
            if name not in g:
                return None
            value = _slice_stack(g[name], positions, single)
            try:
                return float(np.asarray(value).ravel()[0])
            except (TypeError, ValueError, IndexError):
                return None

        return FrameGeometry(
            rot1=_read_scalar("rot1"),
            rot2=_read_scalar("rot2"),
            rot3=_read_scalar("rot3"),
            incident_angle=_read_scalar("incident_angle"),
        )


def read_frame_view(
    scan_file: str | Path,
    frame: int,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
) -> FrameView:
    """Read one processed frame as a canonical :class:`FrameView`.

    This slices individual datasets lazily; it does not materialise a full
    ``(n_frames, chi, q)`` stack.
    """

    path = Path(scan_file)
    result_1d = None
    try:
        result_1d = get_1d(path, frame, entry=entry)
    except KeyError:
        pass

    result_2d = None
    try:
        result_2d = get_2d(path, frame, entry=entry)
    except KeyError:
        pass

    sigma_2d = None
    two_d_kind = TwoDKind.Q_CHI
    if result_2d is not None:
        sigma_2d, two_d_kind = _read_2d_sigma_and_kind(path, int(frame), entry=entry)

    thumbnail = None
    mask_baked = False
    if include_thumbnail:
        thumbnail, mask_baked = _read_thumbnail(path, int(frame), entry=entry)

    source_path, source_frame_index = _read_source_ref(path, int(frame), entry=entry)
    metadata_arrays = _scan_data_for_frames(path, [int(frame)], entry=entry)
    metadata_raw = {
        key: np.asarray(value)[0].item()
        for key, value in metadata_arrays.items()
        if np.asarray(value).shape[:1] == (1,)
    }
    geometry = _read_geometry(path, int(frame), entry=entry)
    incident_angle = None if geometry is None else geometry.incident_angle

    return FrameView(
        label=int(frame),
        axis_1d=(
            None
            if result_1d is None
            else axis_from_unit(result_1d.q_unit, result_1d.q)
        ),
        intensity_1d=None if result_1d is None else result_1d.intensity,
        sigma_1d=None if result_1d is None else result_1d.sigma,
        axis_2d_x=(
            None
            if result_2d is None
            else axis_from_unit(result_2d.q_unit, result_2d.q)
        ),
        axis_2d_y=(
            None
            if result_2d is None
            else axis_from_unit(result_2d.chi_unit, result_2d.chi)
        ),
        intensity_2d=None if result_2d is None else result_2d.intensity,
        sigma_2d=sigma_2d,
        two_d_kind=two_d_kind,
        thumbnail=thumbnail,
        mask_baked=mask_baked,
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
        incident_angle=incident_angle,
        geometry=geometry,
        source_path=source_path,
        source_frame_index=source_frame_index,
    )


def iter_frame_views(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
):
    """Yield :class:`FrameView` objects for selected frame labels."""

    path = Path(scan_file)
    labels = get_frames(path, entry=entry, union=True) if frames is None else frames
    for frame in labels:
        yield read_frame_view(
            path,
            int(frame),
            entry=entry,
            include_thumbnail=include_thumbnail,
        )
