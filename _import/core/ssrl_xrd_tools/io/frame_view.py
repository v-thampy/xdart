"""FrameView readers for processed xdart/ssrl NeXus scans."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from ssrl_xrd_tools.core.frame_view import (
    Axis,
    FrameGeometry,
    FrameView,
    TwoDKind,
    axis_from_unit,
    numeric_metadata,
    two_d_kind_from_units,
)
from ssrl_xrd_tools.io.read import _decode, _dequantize_thumbnail, _entry


def _decode_kind(value, x_unit: str | None, y_unit: str | None) -> TwoDKind:
    if value is not None:
        try:
            return TwoDKind(str(_decode(value)))
        except ValueError:
            pass
    return two_d_kind_from_units(x_unit, y_unit)


def _frame_map(group: h5py.Group | None) -> dict[int, int]:
    if group is None or "frame_index" not in group:
        return {}
    labels = [int(v) for v in np.asarray(group["frame_index"][()]).ravel()]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{group.name}/frame_index contains duplicate labels")
    return {label: row for row, label in enumerate(labels)}


def _dataset_unit(group: h5py.Group | None, name: str) -> str | None:
    if group is None or name not in group or "units" not in group[name].attrs:
        return None
    return _decode(group[name].attrs.get("units"))


def _dataset_values(ds: h5py.Dataset) -> np.ndarray:
    if h5py.check_string_dtype(ds.dtype) is not None:
        return np.asarray(ds.asstr()[()])
    arr = np.asarray(ds[()])
    if arr.dtype.kind == "S":
        return arr.astype(str)
    return arr


class FrameViewReader:
    """Reusable reader for many :class:`FrameView` records from one scan.

    The one-shot :func:`read_frame_view` API is convenient for notebooks and
    selected-frame GUI reads.  Long scans, RSM, stitching, and batch
    validation need the same contract without reopening the HDF5 file and
    rereading axes for every frame.  This context manager opens once, caches
    frame-label maps and axes, and slices only the requested rows.
    """

    def __init__(
        self,
        scan_file: str | Path,
        *,
        entry: str = "entry",
        include_thumbnail: bool = True,
    ) -> None:
        self.path = Path(scan_file)
        self.entry_name = entry
        self.include_thumbnail = bool(include_thumbnail)
        self._h5: h5py.File | None = None
        self._entry: h5py.Group | None = None
        self._g1: h5py.Group | None = None
        self._g2: h5py.Group | None = None
        self._geom: h5py.Group | None = None
        self._scan_data: h5py.Group | None = None
        self._map_1d: dict[int, int] = {}
        self._map_2d: dict[int, int] = {}
        self._map_geom: dict[int, int] = {}
        self._map_scan_data: dict[int, int] = {}
        self._axis_1d: Axis | None = None
        self._axis_2d_x: Axis | None = None
        self._axis_2d_y: Axis | None = None
        self._two_d_kind = TwoDKind.Q_CHI
        # Lazily-filled cache of the scan_data columns for THIS open, so a
        # full-scan read slices each column once instead of re-reading every
        # column for every frame (was O(N^2)).  Reset on open/close.
        self._scan_data_columns: dict[str, np.ndarray] | None = None

    def __enter__(self) -> "FrameViewReader":
        self._h5 = h5py.File(self.path, "r")
        self._entry = _entry(self._h5, self.entry_name)
        self._g1 = self._entry.get("integrated_1d")
        self._g2 = self._entry.get("integrated_2d")
        self._geom = self._entry.get("per_frame_geometry")
        self._scan_data = self._entry.get("scan_data")
        self._map_1d = _frame_map(self._g1)
        self._map_2d = _frame_map(self._g2)
        self._map_geom = _frame_map(self._geom)
        self._map_scan_data = _frame_map(self._scan_data)
        self._scan_data_columns = None  # rebuild lazily for this open

        if self._g1 is not None and "q" in self._g1:
            self._axis_1d = axis_from_unit(
                _dataset_unit(self._g1, "q"),
                np.asarray(self._g1["q"][()]),
            )
        if self._g2 is not None and "q" in self._g2 and "chi" in self._g2:
            q_unit = _dataset_unit(self._g2, "q")
            chi_unit = _dataset_unit(self._g2, "chi")
            self._axis_2d_x = axis_from_unit(q_unit, np.asarray(self._g2["q"][()]))
            self._axis_2d_y = axis_from_unit(chi_unit, np.asarray(self._g2["chi"][()]))
            self._two_d_kind = _decode_kind(
                self._g2.attrs.get("two_d_kind"), q_unit, chi_unit,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._entry = None
        self._scan_data_columns = None

    def _row(self, mapping: dict[int, int], frame: int) -> int | None:
        return mapping.get(int(frame))

    def labels(self) -> tuple[int, ...]:
        """Frame labels known to this scan without reopening the file."""

        labels = set(self._map_1d) | set(self._map_2d) | set(self._map_geom) | set(self._map_scan_data)
        entry = self._entry
        if entry is not None and "frames" in entry:
            for name in entry["frames"]:
                if not str(name).startswith("frame_"):
                    continue
                try:
                    labels.add(int(str(name).removeprefix("frame_")))
                except ValueError:
                    continue
        return tuple(sorted(labels))

    def _metadata_for_frame(self, frame: int) -> dict[str, object]:
        row = self._row(self._map_scan_data, frame)
        group = self._scan_data
        if row is None or group is None:
            return {}
        cols = self._scan_data_columns
        if cols is None:
            # Read each scan_data column ONCE per open, then slice by row —
            # not once per (frame, column).  read_frame_views() loops every
            # frame, so the old per-frame ``item[()]`` full-column read was
            # O(N^2) in the column reads on long scans.
            cols = {}
            for key, item in group.items():
                if key == "frame_index" or not isinstance(item, h5py.Dataset):
                    continue
                try:
                    cols[str(key)] = _dataset_values(item)
                except (TypeError, ValueError):
                    continue
            self._scan_data_columns = cols
        out: dict[str, object] = {}
        for key, arr in cols.items():
            try:
                value = arr[row]
            except (IndexError, TypeError):
                continue
            if np.asarray(value).shape == ():
                scalar = np.asarray(value).item()
                out[key] = _decode(scalar)
        return out

    def _geometry_for_frame(self, frame: int) -> FrameGeometry | None:
        row = self._row(self._map_geom, frame)
        group = self._geom
        if row is None or group is None:
            return None

        def read_scalar(name: str) -> float | None:
            if name not in group:
                return None
            try:
                return float(np.asarray(group[name][row]).ravel()[0])
            except (TypeError, ValueError, IndexError):
                return None

        return FrameGeometry(
            rot1=read_scalar("rot1"),
            rot2=read_scalar("rot2"),
            rot3=read_scalar("rot3"),
            incident_angle=read_scalar("incident_angle"),
        )

    def _thumbnail_for_frame(self, frame: int) -> tuple[np.ndarray | None, bool]:
        entry = self._entry
        if not self.include_thumbnail or entry is None:
            return None, False
        fg = entry.get(f"frames/frame_{int(frame):04d}")
        if fg is None or "thumbnail" not in fg:
            return None, False
        return _dequantize_thumbnail(fg["thumbnail"]), True

    def _source_for_frame(self, frame: int) -> tuple[str | None, int | None]:
        entry = self._entry
        if entry is None:
            return None, None
        fg = entry.get(f"frames/frame_{int(frame):04d}")
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

    def read(self, frame: int) -> FrameView:
        frame = int(frame)
        row_1d = self._row(self._map_1d, frame)
        row_2d = self._row(self._map_2d, frame)

        intensity_1d = sigma_1d = None
        if row_1d is not None and self._g1 is not None:
            intensity_1d = np.asarray(self._g1["intensity"][row_1d])
            if "sigma" in self._g1:
                sigma_1d = np.asarray(self._g1["sigma"][row_1d])

        intensity_2d = sigma_2d = None
        if row_2d is not None and self._g2 is not None:
            intensity_2d = np.asarray(self._g2["intensity"][row_2d])
            if "sigma" in self._g2:
                sigma_2d = np.asarray(self._g2["sigma"][row_2d])

        thumbnail, mask_baked = self._thumbnail_for_frame(frame)
        source_path, source_frame_index = self._source_for_frame(frame)
        metadata_raw = self._metadata_for_frame(frame)
        geometry = self._geometry_for_frame(frame)
        incident_angle = None if geometry is None else geometry.incident_angle

        return FrameView(
            label=frame,
            axis_1d=self._axis_1d if intensity_1d is not None else None,
            intensity_1d=intensity_1d,
            sigma_1d=sigma_1d,
            axis_2d_x=self._axis_2d_x if intensity_2d is not None else None,
            axis_2d_y=self._axis_2d_y if intensity_2d is not None else None,
            intensity_2d=intensity_2d,
            sigma_2d=sigma_2d,
            two_d_kind=self._two_d_kind,
            thumbnail=thumbnail,
            mask_baked=mask_baked,
            metadata_raw=metadata_raw,
            metadata_numeric=numeric_metadata(metadata_raw),
            incident_angle=incident_angle,
            geometry=geometry,
            source_path=source_path,
            source_frame_index=source_frame_index,
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

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
    ) as reader:
        return reader.read(int(frame))


def iter_frame_views(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
):
    """Yield :class:`FrameView` objects one at a time from a single open reader.

    Streams frame-by-frame so RSM / stitching / fitting can consume a long
    scan without materialising every view first.  The HDF5 file stays open
    for the life of the generator and is closed when it is exhausted (or
    closed early via ``GeneratorExit``).
    """

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
    ) as reader:
        labels = reader.labels() if frames is None else frames
        for frame in labels:
            yield reader.read(int(frame))


def read_frame_views(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
) -> tuple[FrameView, ...]:
    """Read selected frame labels using one HDF5 open (eager).

    The preferred headless API for callers that want the whole list at once;
    a thin ``tuple(iter_frame_views(...))`` over the streaming generator.
    """

    return tuple(
        iter_frame_views(
            scan_file, frames, entry=entry, include_thumbnail=include_thumbnail,
        )
    )
