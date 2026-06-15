"""FrameView readers for processed xdart/ssrl NeXus scans."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from xrd_tools.core.frame_view import (
    DEFAULT_MODE_KEY,
    Axis,
    FrameGeometry,
    FrameRecord,
    FrameView,
    TwoDKind,
    axis_from_unit,
    numeric_metadata,
    two_d_kind_from_units,
)
from xrd_tools.io.read import _decode, _dequantize_thumbnail, _entry
from xrd_tools.io.schema import (
    GI_MODE_KEYS_1D,
    GI_MODE_KEYS_2D,
    MULTI_RESULT_MODES_ATTR,
    PRIMARY_MODE_ATTR,
    mode_subgroup_name,
)


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
        source_root: str | Path | None = None,
    ) -> None:
        self.path = Path(scan_file)
        self.entry_name = entry
        self.include_thumbnail = bool(include_thumbnail)
        # N1: repoint a moved raw tree (overrides the stored @source_base); the
        # project root the relative source paths were written against is read
        # from the file in __enter__.
        self.source_root = source_root
        self._source_base: str | None = None
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
        # Multi-result (ADR-0003): per-mode groups/maps/axes.  The scalar fields
        # above remain bound to the PRIMARY mode (back-compat for labels()/read).
        # Insertion order puts the primary first in each dict ⇒ modes_*()[0] == primary.
        self._g1_modes: dict[str, h5py.Group] = {}
        self._g2_modes: dict[str, h5py.Group] = {}
        self._map_1d_modes: dict[str, dict[int, int]] = {}
        self._map_2d_modes: dict[str, dict[int, int]] = {}
        self._axis_1d_modes: dict[str, Axis] = {}
        self._axis_2d_x_modes: dict[str, Axis] = {}
        self._axis_2d_y_modes: dict[str, Axis] = {}
        self._two_d_kind_modes: dict[str, TwoDKind] = {}
        self._primary_mode_1d: str = DEFAULT_MODE_KEY
        self._primary_mode_2d: str = DEFAULT_MODE_KEY
        self._multi_result_modes: bool = False
        # Lazily-filled cache of the scan_data columns for THIS open, so a
        # full-scan read slices each column once instead of re-reading every
        # column for every frame (was O(N^2)).  Reset on open/close.
        self._scan_data_columns: dict[str, np.ndarray] | None = None

    def __enter__(self) -> "FrameViewReader":
        self._h5 = h5py.File(self.path, "r")
        # Anything raising past this point (missing entry group, duplicate
        # frame labels, malformed datasets) happens BEFORE the caller's
        # with-block exists, so __exit__ never runs — close the handle
        # ourselves or it leaks (and locks the file on Windows).
        try:
            return self._enter_inner()
        except BaseException:
            self._h5.close()
            self._h5 = None
            raise

    def _enter_inner(self) -> "FrameViewReader":
        self._entry = _entry(self._h5, self.entry_name)
        # C1: surface a newer-than-supported schema before any dataset access
        # fails with an opaque KeyError.
        from xrd_tools.io.nexus import warn_if_newer_schema
        warn_if_newer_schema(self._entry, str(self.path))
        # N1: the project root the relative source paths point under (None on old
        # absolute-path files; harmless there).
        self._source_base = (
            _decode(self._entry.attrs["source_base"])
            if "source_base" in self._entry.attrs else None)
        self._g1 = self._entry.get("integrated_1d")
        self._g2 = self._entry.get("integrated_2d")
        self._geom = self._entry.get("per_frame_geometry")
        self._scan_data = self._entry.get("scan_data")
        self._map_1d = _frame_map(self._g1)
        self._map_2d = _frame_map(self._g2)
        self._map_geom = _frame_map(self._geom)
        self._map_scan_data = _frame_map(self._scan_data)
        self._scan_data_columns = None  # rebuild lazily for this open

        # Multi-result discovery (ADR-0003).  Read the per-scan primary + the
        # mode-aware capability marker, then register the primary (top-level)
        # FIRST and probe the SCHEMA-known GI subgroup names (never blind child
        # enumeration — an unrelated child must not become a phantom mode).
        def _primary_attr(grp):
            if grp is None or PRIMARY_MODE_ATTR not in grp.attrs:
                return DEFAULT_MODE_KEY
            return str(_decode(grp.attrs[PRIMARY_MODE_ATTR]))

        self._primary_mode_1d = _primary_attr(self._g1)
        self._primary_mode_2d = _primary_attr(self._g2)
        self._multi_result_modes = bool(
            (self._g1 is not None and MULTI_RESULT_MODES_ATTR in self._g1.attrs)
            or (self._g2 is not None and MULTI_RESULT_MODES_ATTR in self._g2.attrs)
        )

        def _register_1d(mode, g):
            # Register only a READABLE mode (intensity + its q axis present) so
            # modes_1d() never advertises a mode that read()/read_record cannot
            # load — foreign/partially-written-file robustness.
            if "intensity" not in g or "q" not in g:
                return
            self._g1_modes[mode] = g
            self._map_1d_modes[mode] = _frame_map(g)
            self._axis_1d_modes[mode] = axis_from_unit(
                _dataset_unit(g, "q"), np.asarray(g["q"][()]))

        def _register_2d(mode, g):
            if "intensity" not in g or "q" not in g or "chi" not in g:
                return
            self._g2_modes[mode] = g
            self._map_2d_modes[mode] = _frame_map(g)
            qu, cu = _dataset_unit(g, "q"), _dataset_unit(g, "chi")
            self._axis_2d_x_modes[mode] = axis_from_unit(qu, np.asarray(g["q"][()]))
            self._axis_2d_y_modes[mode] = axis_from_unit(cu, np.asarray(g["chi"][()]))
            self._two_d_kind_modes[mode] = _decode_kind(
                g.attrs.get("two_d_kind"), qu, cu)

        if self._g1 is not None:
            _register_1d(self._primary_mode_1d, self._g1)
            for k in GI_MODE_KEYS_1D:
                if k == self._primary_mode_1d:
                    continue
                child = self._g1.get(mode_subgroup_name(k))
                if isinstance(child, h5py.Group):
                    _register_1d(k, child)
        if self._g2 is not None:
            _register_2d(self._primary_mode_2d, self._g2)
            for k in GI_MODE_KEYS_2D:
                if k == self._primary_mode_2d:
                    continue
                child = self._g2.get(mode_subgroup_name(k))
                if isinstance(child, h5py.Group):
                    _register_2d(k, child)

        # Scalar aliases = the PRIMARY mode's entries (back-compat: labels(),
        # read() without mode args, and external _axis_* consumers are unchanged
        # for both single-mode and mode-aware files).
        self._axis_1d = self._axis_1d_modes.get(self._primary_mode_1d)
        self._axis_2d_x = self._axis_2d_x_modes.get(self._primary_mode_2d)
        self._axis_2d_y = self._axis_2d_y_modes.get(self._primary_mode_2d)
        self._two_d_kind = self._two_d_kind_modes.get(
            self._primary_mode_2d, TwoDKind.Q_CHI)
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

        labels = set(self._map_geom) | set(self._map_scan_data)
        for mode_map in (*self._map_1d_modes.values(), *self._map_2d_modes.values()):
            labels |= set(mode_map)
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
            stored = str(_decode(src["path"][()]))
            # N1: resolve the (relative) stored path to an absolute master so
            # FrameView consumers (FrameSource/Scan/notebooks/RSM/stitch) can
            # locate the raw after the data moves.  Precedence source_root >
            # @source_base > scan dir; absolute paths used as-is (back-compat).
            # Fall back to the stored string when nothing resolves, so the
            # field is never silently blanked (provenance preserved).
            from xrd_tools.io.read import resolve_source_master
            resolved = resolve_source_master(
                stored, scan_file=self.path,
                source_base=self._source_base, source_root=self.source_root)
            path = str(resolved) if resolved is not None else stored
        if "frame_index" in src:
            source_idx = int(np.asarray(src["frame_index"][()]).ravel()[0])
        return path, source_idx

    def _common_fields(self, frame: int) -> dict:
        """Shared per-frame fields (thumbnail/source/metadata/geometry) — the
        same for every mode of one frame."""
        thumbnail, mask_baked = self._thumbnail_for_frame(frame)
        source_path, source_frame_index = self._source_for_frame(frame)
        metadata_raw = self._metadata_for_frame(frame)
        geometry = self._geometry_for_frame(frame)
        incident = None if geometry is None else geometry.incident_angle
        return dict(
            thumbnail=thumbnail, mask_baked=mask_baked,
            metadata_raw=metadata_raw,
            metadata_numeric=numeric_metadata(metadata_raw),
            incident_angle=incident, geometry=geometry,
            source_path=source_path, source_frame_index=source_frame_index,
        )

    def read(self, frame: int, *, mode_1d=None, mode_2d=None) -> FrameView:
        """One combined :class:`FrameView` for ``frame``.

        ``mode_1d`` / ``mode_2d`` select GI sub-modes (default: the per-scan
        primary).  An unknown/absent mode leaves that dimension empty (same as
        a frame with no row).  On a single-mode/old file both default to the
        ``DEFAULT_MODE_KEY`` top-level slot ⇒ behaviour is unchanged."""
        frame = int(frame)
        m1 = mode_1d if mode_1d is not None else self._primary_mode_1d
        m2 = mode_2d if mode_2d is not None else self._primary_mode_2d
        g1 = self._g1_modes.get(m1)
        g2 = self._g2_modes.get(m2)
        row_1d = self._row(self._map_1d_modes.get(m1, {}), frame)
        row_2d = self._row(self._map_2d_modes.get(m2, {}), frame)

        intensity_1d = sigma_1d = None
        if row_1d is not None and g1 is not None:
            intensity_1d = np.asarray(g1["intensity"][row_1d])
            if "sigma" in g1:
                sigma_1d = np.asarray(g1["sigma"][row_1d])

        intensity_2d = sigma_2d = None
        if row_2d is not None and g2 is not None:
            intensity_2d = np.asarray(g2["intensity"][row_2d])
            if "sigma" in g2:
                sigma_2d = np.asarray(g2["sigma"][row_2d])

        return FrameView(
            label=frame,
            axis_1d=self._axis_1d_modes.get(m1) if intensity_1d is not None else None,
            intensity_1d=intensity_1d,
            sigma_1d=sigma_1d,
            axis_2d_x=self._axis_2d_x_modes.get(m2) if intensity_2d is not None else None,
            axis_2d_y=self._axis_2d_y_modes.get(m2) if intensity_2d is not None else None,
            intensity_2d=intensity_2d,
            sigma_2d=sigma_2d,
            two_d_kind=self._two_d_kind_modes.get(m2, TwoDKind.Q_CHI),
            **self._common_fields(frame),
        )

    def _view_for(self, frame: int, dim: str, mode: str) -> "FrameView | None":
        """Dimension-pure :class:`FrameView` for one ``(dim, mode)``, or ``None``
        if absent.  Shared per-frame fields are included so a record's per-dim
        views carry them; :class:`FrameRecord` re-projects to dimension-pure."""
        frame = int(frame)
        if dim == "1d":
            g = self._g1_modes.get(mode)
            row = self._row(self._map_1d_modes.get(mode, {}), frame)
            if g is None or row is None:
                return None
            sig = np.asarray(g["sigma"][row]) if "sigma" in g else None
            return FrameView(
                label=frame, axis_1d=self._axis_1d_modes.get(mode),
                intensity_1d=np.asarray(g["intensity"][row]), sigma_1d=sig,
                **self._common_fields(frame),
            )
        g = self._g2_modes.get(mode)
        row = self._row(self._map_2d_modes.get(mode, {}), frame)
        if g is None or row is None:
            return None
        sig = np.asarray(g["sigma"][row]) if "sigma" in g else None
        return FrameView(
            label=frame, axis_2d_x=self._axis_2d_x_modes.get(mode),
            axis_2d_y=self._axis_2d_y_modes.get(mode),
            intensity_2d=np.asarray(g["intensity"][row]), sigma_2d=sig,
            two_d_kind=self._two_d_kind_modes.get(mode, TwoDKind.Q_CHI),
            **self._common_fields(frame),
        )

    def modes_1d(self) -> tuple:
        """GI 1D mode_keys present (primary first)."""
        return tuple(self._g1_modes)

    def modes_2d(self) -> tuple:
        """GI 2D mode_keys present (primary first)."""
        return tuple(self._g2_modes)

    def primary_mode_1d(self) -> str:
        return self._primary_mode_1d

    def primary_mode_2d(self) -> str:
        return self._primary_mode_2d

    def is_multi_mode(self) -> bool:
        """True if the file carries the per-mode capability marker."""
        return self._multi_result_modes

    def read_record(self, frame: int) -> FrameRecord:
        """Read every mode of ``frame`` into a multi-result :class:`FrameRecord`.

        On a single-mode/old file this is exactly
        ``FrameRecord.from_view(self.read(frame))`` (one ``DEFAULT_MODE_KEY``
        entry per dimension)."""
        frame = int(frame)
        r1d: dict = {}
        r2d: dict = {}
        for m in self.modes_1d():
            v = self._view_for(frame, "1d", m)
            if v is not None and v.has_1d:
                r1d[m] = v
        for m in self.modes_2d():
            v = self._view_for(frame, "2d", m)
            if v is not None and v.has_2d:
                r2d[m] = v
        a1 = self._primary_mode_1d if self._primary_mode_1d in r1d else next(
            iter(r1d), DEFAULT_MODE_KEY)
        a2 = self._primary_mode_2d if self._primary_mode_2d in r2d else next(
            iter(r2d), DEFAULT_MODE_KEY)
        return FrameRecord(
            label=frame, results_1d=r1d, results_2d=r2d,
            active_mode_1d=a1, active_mode_2d=a2,
        )


def read_frame_view(
    scan_file: str | Path,
    frame: int,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
    source_root: str | Path | None = None,
    mode_1d=None,
    mode_2d=None,
) -> FrameView:
    """Read one processed frame as a canonical :class:`FrameView`.

    This slices individual datasets lazily; it does not materialise a full
    ``(n_frames, chi, q)`` stack.  ``source_root`` (N1) repoints a moved raw
    tree so ``FrameView.source_path`` resolves to the relocated master.
    ``mode_1d`` / ``mode_2d`` select GI sub-modes (default: the per-scan
    primary).
    """

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
        source_root=source_root,
    ) as reader:
        return reader.read(int(frame), mode_1d=mode_1d, mode_2d=mode_2d)


def read_frame_record(
    scan_file: str | Path,
    frame: int,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
    source_root: str | Path | None = None,
) -> FrameRecord:
    """Read one processed frame as a multi-result :class:`FrameRecord`
    (every persisted GI mode)."""

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
        source_root=source_root,
    ) as reader:
        return reader.read_record(int(frame))


def iter_frame_records(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
    source_root: str | Path | None = None,
):
    """Yield :class:`FrameRecord` objects one at a time from one open reader."""

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
        source_root=source_root,
    ) as reader:
        labels = reader.labels() if frames is None else frames
        for frame in labels:
            yield reader.read_record(int(frame))


def read_frame_records(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
    source_root: str | Path | None = None,
) -> tuple[FrameRecord, ...]:
    """Read selected frame labels as :class:`FrameRecord`s using one open
    (eager; a thin ``tuple(iter_frame_records(...))``)."""

    return tuple(
        iter_frame_records(
            scan_file, frames, entry=entry, include_thumbnail=include_thumbnail,
            source_root=source_root,
        )
    )


def iter_frame_views(
    scan_file: str | Path,
    frames: Iterable[int] | None = None,
    *,
    entry: str = "entry",
    include_thumbnail: bool = True,
    source_root: str | Path | None = None,
):
    """Yield :class:`FrameView` objects one at a time from a single open reader.

    Streams frame-by-frame so RSM / stitching / fitting can consume a long
    scan without materialising every view first.  The HDF5 file stays open
    for the life of the generator and is closed when it is exhausted (or
    closed early via ``GeneratorExit``).  ``source_root`` (N1) repoints a moved
    raw tree for the resolved ``FrameView.source_path``.
    """

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
        source_root=source_root,
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
    source_root: str | Path | None = None,
) -> tuple[FrameView, ...]:
    """Read selected frame labels using one HDF5 open (eager).

    The preferred headless API for callers that want the whole list at once;
    a thin ``tuple(iter_frame_views(...))`` over the streaming generator.
    """

    return tuple(
        iter_frame_views(
            scan_file, frames, entry=entry, include_thumbnail=include_thumbnail,
            source_root=source_root,
        )
    )
