"""FrameView readers for processed xdart/ssrl NeXus scans."""

from __future__ import annotations

from pathlib import Path
import math
import sys
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
from xrd_tools.io.read import _decode
from xrd_tools.io.schema import (
    GI_MODE_KEYS_1D,
    GI_MODE_KEYS_2D,
    MONOTONIC_ATTR,
    MULTI_RESULT_MODES_ATTR,
    PRIMARY_MODE_ATTR,
    REINTEGRATE_SHADOW_COMPLETE_ATTR,
    REINTEGRATE_SHADOW_SUFFIX,
    THUMBNAIL_LUT_ATTRS,
    mode_subgroup_name,
)


# Headless processed-artifact admission limits.  One million rows/points is
# the established vNext persisted-frame and viewer ceiling; 64 MiB is the
# established bounded analysis payload ceiling.  These checks happen from HDF
# metadata before Dataset.__getitem__ so a foreign sparse dataset cannot pick
# the Browse worker's allocation size.
_MAX_PERSISTED_FRAME_ROWS = 1_000_000
_MAX_AXIS_POINTS = 1_000_000
_MAX_SCAN_DATA_COLUMNS = 256
_MAX_SCAN_DATA_BYTES = 64 * 1024 * 1024
_MAX_SCAN_DATA_SCALAR_BYTES = 64 * 1024
_MAX_SOURCE_PATH_BYTES = 4096
_MAX_1D_ROW_BYTES = 64 * 1024 * 1024
_MAX_2D_ROW_BYTES = 64 * 1024 * 1024
_MAX_ATTRIBUTE_BYTES = 256


def _direct_dataset(group: h5py.Group, name: str) -> h5py.Dataset | None:
    try:
        link = group.get(name, getlink=True)
        node = group.get(name) if type(link) is h5py.HardLink else None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(node, h5py.Dataset):
        return None
    try:
        external = tuple(node.external or ())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return None if node.is_virtual or external else node


def _direct_group(group: h5py.Group, name: str) -> h5py.Group | None:
    try:
        link = group.get(name, getlink=True)
        node = group.get(name) if type(link) is h5py.HardLink else None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return node if isinstance(node, h5py.Group) else None


def _required_direct_group(
    group: h5py.Group, name: str, *, role: str,
) -> h5py.Group | None:
    try:
        link = group.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} link is unreadable") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"{role} is not a local hard-linked group")
    value = _direct_group(group, name)
    if value is None:
        raise ValueError(f"{role} is not a local hard-linked group")
    return value


def _required_direct_dataset(
    group: h5py.Group, name: str, *, role: str,
) -> h5py.Dataset | None:
    try:
        link = group.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} link is unreadable") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"{role} is not a bounded local hard-linked dataset")
    value = _direct_dataset(group, name)
    if value is None:
        raise ValueError(f"{role} is not a bounded local hard-linked dataset")
    return value


def _integrated_group(
    entry: h5py.Group, name: str,
) -> h5py.Group | None:
    """Local-hard equivalent of schema orphan-shadow recovery."""

    canonical = _required_direct_group(entry, name, role=f"/{name}")
    if canonical is not None:
        return canonical
    shadow_name = f"{name}{REINTEGRATE_SHADOW_SUFFIX}"
    shadow = _required_direct_group(
        entry, shadow_name, role=f"/{shadow_name}",
    )
    if shadow is None:
        return None
    complete = _bounded_bool_attr(
        shadow,
        REINTEGRATE_SHADOW_COMPLETE_ATTR,
        role=f"{shadow.name} completion marker",
    )
    return shadow if complete else None


def _bounded_text_attr(
    owner: object,
    name: str,
    *,
    role: str,
    max_bytes: int = _MAX_ATTRIBUTE_BYTES,
) -> str | None:
    attrs = getattr(owner, "attrs", None)
    if attrs is None or name not in attrs:
        return None
    try:
        attr = attrs.get_id(name)
        info = h5py.check_string_dtype(attr.dtype)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} is unreadable") from error
    if (
        attr.shape != ()
        or info is None
        or info.encoding not in {"ascii", "utf-8"}
        or (info.length is not None and info.length > max_bytes)
    ):
        raise ValueError(f"{role} is not a bounded text attribute")
    destination = np.empty((), dtype=f"S{max_bytes + 1}")
    try:
        attr.read(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} could not be read as bounded text") from error
    raw = bytes(destination[()])
    if len(raw) > max_bytes:
        raise ValueError(f"{role} exceeds the text byte ceiling")
    try:
        return raw.decode(info.encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not valid text") from error


def _bounded_bool_attr(owner: object, name: str, *, role: str) -> bool | None:
    attrs = getattr(owner, "attrs", None)
    if attrs is None or name not in attrs:
        return None
    try:
        attr = attrs.get_id(name)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} is unreadable") from error
    if attr.shape != () or attr.dtype.kind not in "biu" or attr.dtype.itemsize > 8:
        raise ValueError(f"{role} is not a bounded boolean attribute")
    return bool(attrs[name])


def _bounded_number_attr(
    owner: object, name: str, *, role: str, default: float,
) -> float:
    attrs = getattr(owner, "attrs", None)
    if attrs is None or name not in attrs:
        return float(default)
    try:
        attr = attrs.get_id(name)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} is unreadable") from error
    if attr.shape != () or attr.dtype.kind not in "iuf" or attr.dtype.itemsize > 8:
        raise ValueError(f"{role} is not a bounded numeric attribute")
    value = float(attrs[name])
    if not math.isfinite(value):
        raise ValueError(f"{role} is not finite")
    return value


def _bounded_integral_scalar(dataset: object, *, role: str) -> int:
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.shape != ()
        or dataset.dtype.kind not in "iu"
        or dataset.dtype.itemsize > 8
        or dataset.is_virtual
    ):
        raise ValueError(f"{role} is not a bounded integer scalar")
    return int(dataset[()])


def _dataset_bytes(dataset: h5py.Dataset) -> int | None:
    """Fixed-width materialized byte count, or ``None`` for a vlen dtype."""

    if h5py.check_string_dtype(dataset.dtype) is not None and dataset.dtype.kind == "O":
        return None
    return int(dataset.size) * int(dataset.dtype.itemsize)


def _qualify_vector(
    dataset: object,
    *,
    role: str,
    max_items: int,
    max_bytes: int,
    numeric: bool,
) -> h5py.Dataset:
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
        raise ValueError(f"{role} is not a one-dimensional dataset")
    count = int(dataset.shape[0])
    byte_count = _dataset_bytes(dataset)
    if not 0 <= count <= max_items:
        raise ValueError(f"{role} item count exceeds limit")
    if numeric and dataset.dtype.kind not in "iuf":
        raise ValueError(f"{role} is not a fixed-width numeric dataset")
    if byte_count is not None and byte_count > max_bytes:
        raise ValueError(f"{role} materialized bytes exceed limit")
    return dataset


def _read_numeric_vector(dataset: object, *, role: str) -> np.ndarray:
    qualified = _qualify_vector(
        dataset,
        role=role,
        max_items=_MAX_AXIS_POINTS,
        max_bytes=_MAX_1D_ROW_BYTES,
        numeric=True,
    )
    return np.asarray(qualified[()])


def _qualify_frame_index(group: h5py.Group, dataset: object) -> h5py.Dataset:
    qualified = _qualify_vector(
        dataset,
        role=f"{group.name}/frame_index",
        max_items=_MAX_PERSISTED_FRAME_ROWS,
        max_bytes=_MAX_PERSISTED_FRAME_ROWS * 8,
        numeric=True,
    )
    if qualified.dtype.kind not in "iu" or qualified.dtype.itemsize > 8:
        raise ValueError(f"{group.name}/frame_index is not an integer inventory")
    return qualified


def _qualify_1d_stack(
    group: h5py.Group, dataset: object, *, points: int, role: str,
) -> h5py.Dataset:
    rows_node = _required_direct_dataset(
        group, "frame_index", role=f"{group.name}/frame_index",
    )
    rows = int(rows_node.shape[0]) if isinstance(rows_node, h5py.Dataset) else 0
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.ndim != 2
        or dataset.dtype.kind not in "iuf"
        or dataset.shape != (rows, points)
        or int(points) * int(dataset.dtype.itemsize) > _MAX_1D_ROW_BYTES
    ):
        raise ValueError(f"{group.name}/{role} is not a bounded 1-D row stack")
    return dataset


def _read_1d_row(
    group: h5py.Group, name: str, row: int, *, points: int,
) -> np.ndarray:
    dataset = _qualify_1d_stack(
        group,
        _required_direct_dataset(group, name, role=f"{group.name}/{name}"),
        points=points,
        role=name,
    )
    if not 0 <= int(row) < int(dataset.shape[0]):
        raise ValueError(f"{group.name}/{name} row is out of range")
    return np.asarray(dataset[int(row)])


def _qualify_2d_stack(
    group: h5py.Group,
    dataset: object,
    *,
    q_points: int,
    chi_points: int,
    role: str,
) -> h5py.Dataset:
    rows_node = _required_direct_dataset(
        group, "frame_index", role=f"{group.name}/frame_index",
    )
    rows = int(rows_node.shape[0]) if isinstance(rows_node, h5py.Dataset) else 0
    row_bytes = int(q_points) * int(chi_points) * int(
        dataset.dtype.itemsize if isinstance(dataset, h5py.Dataset) else 0
    )
    if (
        not isinstance(dataset, h5py.Dataset)
        or dataset.ndim != 3
        or dataset.dtype.kind not in "iuf"
        or dataset.shape != (rows, chi_points, q_points)
        or row_bytes > _MAX_2D_ROW_BYTES
    ):
        raise ValueError(f"{group.name}/{role} is not a bounded 2-D row stack")
    return dataset


def _read_2d_row(
    group: h5py.Group,
    name: str,
    row: int,
    *,
    q_points: int,
    chi_points: int,
) -> np.ndarray:
    dataset = _qualify_2d_stack(
        group,
        _required_direct_dataset(group, name, role=f"{group.name}/{name}"),
        q_points=q_points,
        chi_points=chi_points,
        role=name,
    )
    if not 0 <= int(row) < int(dataset.shape[0]):
        raise ValueError(f"{group.name}/{name} row is out of range")
    return np.asarray(dataset[int(row)])


def _bounded_utf8_item(
    dataset: h5py.Dataset, row: int, *, role: str,
) -> str:
    """Read one vlen/fixed UTF-8 cell through a finite conversion buffer."""

    info = h5py.check_string_dtype(dataset.dtype)
    if (
        info is None
        or info.encoding not in {"ascii", "utf-8"}
        or (info.length is not None and info.length > _MAX_SCAN_DATA_SCALAR_BYTES)
    ):
        raise ValueError(f"{role} is not bounded UTF-8")
    destination = np.empty(
        (1,), dtype=f"S{_MAX_SCAN_DATA_SCALAR_BYTES + 1}",
    )
    try:
        dataset.read_direct(
            destination,
            source_sel=np.s_[int(row) : int(row) + 1],
            dest_sel=np.s_[:],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} could not be read as bounded UTF-8") from error
    raw = bytes(destination[0])
    if len(raw) > _MAX_SCAN_DATA_SCALAR_BYTES:
        raise ValueError(f"{role} exceeds the UTF-8 byte ceiling")
    try:
        return raw.decode(info.encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not UTF-8") from error


def _scan_data_items(
    group: h5py.Group | None,
) -> tuple[tuple[str, h5py.Dataset], ...]:
    if group is None:
        return ()
    if len(group) > _MAX_SCAN_DATA_COLUMNS + 1:
        raise ValueError(f"{group.name} column inventory exceeds limit")
    frame_index = _direct_dataset(group, "frame_index")
    rows = int(frame_index.shape[0]) if isinstance(frame_index, h5py.Dataset) else 0
    total_fixed_bytes = 0
    items: list[tuple[str, h5py.Dataset]] = []
    for key in group:
        if key == "frame_index":
            continue
        item = _direct_dataset(group, key)
        if item is None or item.shape != (rows,):
            raise ValueError(f"{group.name}/{key} is not one scalar per frame")
        string = h5py.check_string_dtype(item.dtype)
        if string is None and item.dtype.kind not in "iufbS":
            raise ValueError(f"{group.name}/{key} has an unsupported dtype")
        byte_count = _dataset_bytes(item)
        if byte_count is not None:
            total_fixed_bytes += byte_count
            if total_fixed_bytes > _MAX_SCAN_DATA_BYTES:
                raise ValueError(f"{group.name} materialized columns exceed limit")
        items.append((str(key), item))
    if len(items) > _MAX_SCAN_DATA_COLUMNS:
        raise ValueError(f"{group.name} column inventory exceeds limit")
    return tuple(items)


def _scan_data_scalar(dataset: h5py.Dataset, row: int, *, role: str):
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return _bounded_utf8_item(dataset, row, role=role)
    return dataset[int(row)]


def _scan_data_column(
    dataset: h5py.Dataset, *, role: str, allowance_bytes: int,
) -> tuple[np.ndarray, int]:
    if type(allowance_bytes) is not int or allowance_bytes < 0:
        raise ValueError(f"{role} has no remaining materialization allowance")
    if h5py.check_string_dtype(dataset.dtype) is None:
        logical_bytes = int(dataset.size) * int(dataset.dtype.itemsize)
        if logical_bytes > allowance_bytes:
            raise ValueError(f"{role} materialized bytes exceed allowance")
        value = np.asarray(dataset[()])
        return value, logical_bytes
    rows = int(dataset.shape[0])
    pointer_bytes = rows * int(np.dtype(object).itemsize)
    minimum = pointer_bytes + rows * 64
    if minimum > allowance_bytes:
        raise ValueError(f"{role} retained object bytes exceed limit")
    values = np.empty((rows,), dtype=object)
    total = pointer_bytes
    for row in range(int(dataset.shape[0])):
        if total + 64 > allowance_bytes:
            raise ValueError(f"{role} retained object bytes exceed limit")
        value = _bounded_utf8_item(dataset, row, role=role)
        retained_bytes = max(64, int(sys.getsizeof(value)))
        if total + retained_bytes > allowance_bytes:
            raise ValueError(f"{role} retained object bytes exceed limit")
        total += retained_bytes
        values[row] = value
    return values, total


def _bounded_utf8_scalar(
    dataset: object, *, role: str, max_bytes: int,
) -> str:
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
        raise ValueError(f"{role} is not a UTF-8 scalar")
    info = h5py.check_string_dtype(dataset.dtype)
    if (
        info is None
        or info.encoding not in {"ascii", "utf-8"}
        or (info.length is not None and info.length > max_bytes)
    ):
        raise ValueError(f"{role} is not bounded UTF-8")
    destination = np.empty(
        (), dtype=f"S{max_bytes + 1}",
    )
    try:
        dataset.read_direct(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} could not be read as bounded UTF-8") from error
    raw = bytes(destination[()])
    if len(raw) > max_bytes:
        raise ValueError(f"{role} exceeds the UTF-8 byte ceiling")
    try:
        return raw.decode(info.encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not UTF-8") from error


def _read_thumbnail(
    frame_group: h5py.Group, thumbnail: h5py.Dataset,
) -> np.ndarray:
    """Qualify and hydrate one bounded local thumbnail and optional mask."""

    if (
        thumbnail.ndim != 2
        or not all(1 <= int(part) <= 256 for part in thumbnail.shape)
        or thumbnail.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}
        or int(thumbnail.size) * int(thumbnail.dtype.itemsize) > 256 * 256 * 2
    ):
        raise ValueError(f"{thumbnail.name} is not a bounded thumbnail")
    vmin_key, vmax_key, dtype_key = THUMBNAIL_LUT_ATTRS
    vmin = _bounded_number_attr(
        thumbnail, vmin_key, role=f"{thumbnail.name} {vmin_key}", default=0.0,
    )
    vmax = _bounded_number_attr(
        thumbnail, vmax_key, role=f"{thumbnail.name} {vmax_key}", default=1.0,
    )
    if vmax < vmin:
        raise ValueError(f"{thumbnail.name} LUT range is reversed")
    persisted_dtype = _bounded_text_attr(
        thumbnail,
        dtype_key,
        role=f"{thumbnail.name} {dtype_key}",
        max_bytes=16,
    )
    if persisted_dtype is not None and persisted_dtype != thumbnail.dtype.name:
        raise ValueError(f"{thumbnail.name} LUT dtype differs from its dataset")
    mask = None
    mask = _required_direct_dataset(
        frame_group,
        "thumbnail_mask",
        role=f"{frame_group.name}/thumbnail_mask",
    )
    if mask is not None:
        if (
            mask.shape != thumbnail.shape
            or mask.dtype != np.dtype(bool)
            or mask.ndim != 2
        ):
            raise ValueError(f"{frame_group.name}/thumbnail_mask is not bounded")

    values = np.asarray(thumbnail[()], dtype=float)
    scale = 65535.0 if thumbnail.dtype == np.dtype(np.uint16) else 255.0
    result = vmin + (values / scale) * (vmax - vmin)
    if mask is not None:
        invalid = np.asarray(mask[()], dtype=bool)
        result[invalid] = np.nan
    return result


def _decode_kind(value, x_unit: str | None, y_unit: str | None) -> TwoDKind:
    if value is not None:
        try:
            return TwoDKind(str(_decode(value)))
        except ValueError:
            pass
    return two_d_kind_from_units(x_unit, y_unit)


def _frame_map(group: h5py.Group | None, target_frame: int | None = None) -> dict[int, int]:
    if group is None:
        return {}
    inventory = _required_direct_dataset(
        group, "frame_index", role=f"{group.name}/frame_index",
    )
    if inventory is None:
        return {}
    dataset = _qualify_frame_index(
        group, inventory,
    )
    if target_frame is not None:
        target = int(target_frame)
        size = int(dataset.size)
        marker = _bounded_bool_attr(
            group, MONOTONIC_ATTR, role=f"{group.name} monotonic marker",
        )
        if marker is not None and bool(marker):
            low, high = 0, size
            while low < high:
                middle = (low + high) // 2
                value = int(np.asarray(dataset[middle]).ravel()[0])
                low, high = (middle + 1, high) if value < target else (low, middle)
            found = low < size and int(np.asarray(dataset[low]).ravel()[0]) == target
            return {target: low} if found else {}
        rows = []
        step = dataset.chunks[0] if dataset.chunks else 4096
        for start in range(0, size, step):
            labels = np.asarray(dataset[start : start + step]).ravel()
            rows.extend(start + int(offset) for offset in np.flatnonzero(labels == target))
        if len(rows) > 1:
            raise ValueError(f"{group.name}/frame_index contains duplicate labels")
        return {} if not rows else {target: rows[0]}
    labels = [int(v) for v in np.asarray(dataset[()]).ravel()]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{group.name}/frame_index contains duplicate labels")
    return {label: row for row, label in enumerate(labels)}


def _dataset_unit(group: h5py.Group | None, name: str) -> str | None:
    dataset = None if group is None else _direct_dataset(group, name)
    if dataset is None or "units" not in dataset.attrs:
        return None
    return _bounded_text_attr(
        dataset, "units", role=f"{dataset.name} units",
    )


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
        resolve_source: bool = True,
        target_frame: int | None = None,
    ) -> None:
        self.path = Path(scan_file)
        self.entry_name = entry
        self.include_thumbnail = bool(include_thumbnail)
        # N1: repoint a moved raw tree (overrides the stored @source_base); the
        # project root the relative source paths were written against is read
        # from the file in __enter__.
        self.source_root = source_root
        self.resolve_source = bool(resolve_source)
        if target_frame is not None and (type(target_frame) is not int or target_frame < 0):
            raise TypeError("target_frame must be an exact nonnegative integer")
        self.target_frame = target_frame
        self._source_base: str | None = None
        self._h5: h5py.File | None = None
        self._entry: h5py.Group | None = None
        self._g1: h5py.Group | None = None
        self._g2: h5py.Group | None = None
        self._geom: h5py.Group | None = None
        self._scan_data: h5py.Group | None = None
        self._frames: h5py.Group | None = None
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
        self._scan_data_items: tuple[tuple[str, h5py.Dataset], ...] = ()

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
        self._entry = _required_direct_group(
            self._h5, self.entry_name, role=f"/{self.entry_name}",
        )
        if self._entry is None:
            raise KeyError(f"No {self.entry_name!r} group in {self.path}")
        # C1: surface a newer-than-supported schema before any dataset access
        # fails with an opaque KeyError.
        from xrd_tools.io.nexus import warn_if_newer_schema
        warn_if_newer_schema(self._entry, str(self.path))
        # N1: the project root the relative source paths point under (None on old
        # absolute-path files; harmless there).
        self._source_base = _bounded_text_attr(
            self._entry,
            "source_base",
            role=f"{self._entry.name} source base",
            max_bytes=_MAX_SOURCE_PATH_BYTES,
        )
        # Recover an orphan __reint shadow left by a crash mid-swap (read-only
        # adoption) so a reintegrate interrupted between del-canonical and
        # move-shadow still opens on its complete result.
        self._g1 = _integrated_group(self._entry, "integrated_1d")
        self._g2 = _integrated_group(self._entry, "integrated_2d")
        self._geom = _required_direct_group(
            self._entry,
            "per_frame_geometry",
            role=f"{self._entry.name}/per_frame_geometry",
        )
        self._scan_data = _required_direct_group(
            self._entry, "scan_data", role=f"{self._entry.name}/scan_data",
        )
        self._frames = _required_direct_group(
            self._entry, "frames", role=f"{self._entry.name}/frames",
        )
        self._map_1d = _frame_map(self._g1, self.target_frame)
        self._map_2d = _frame_map(self._g2, self.target_frame)
        self._map_geom = _frame_map(self._geom, self.target_frame)
        self._map_scan_data = _frame_map(self._scan_data, self.target_frame)
        self._scan_data_items = _scan_data_items(self._scan_data)
        self._scan_data_columns = None  # rebuild lazily for this open

        # Multi-result discovery (ADR-0003).  Read the per-scan primary + the
        # mode-aware capability marker, then register the primary (top-level)
        # FIRST and probe the SCHEMA-known GI subgroup names (never blind child
        # enumeration — an unrelated child must not become a phantom mode).
        def _primary_attr(grp):
            if grp is None or PRIMARY_MODE_ATTR not in grp.attrs:
                return DEFAULT_MODE_KEY
            return str(_bounded_text_attr(
                grp, PRIMARY_MODE_ATTR, role=f"{grp.name} primary mode",
            ))

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
            intensity_node = _required_direct_dataset(
                g, "intensity", role=f"{g.name}/intensity",
            )
            q_node = _required_direct_dataset(
                g, "q", role=f"{g.name}/q",
            )
            if intensity_node is None or q_node is None:
                return
            q = _read_numeric_vector(
                q_node,
                role=f"{g.name}/q",
            )
            _qualify_1d_stack(
                g,
                intensity_node,
                points=int(q.size),
                role="intensity",
            )
            sigma_node = _required_direct_dataset(
                g, "sigma", role=f"{g.name}/sigma",
            )
            if sigma_node is not None:
                _qualify_1d_stack(
                    g, sigma_node,
                    points=int(q.size),
                    role="sigma",
                )
            self._g1_modes[mode] = g
            self._map_1d_modes[mode] = _frame_map(g, self.target_frame)
            self._axis_1d_modes[mode] = axis_from_unit(
                _dataset_unit(g, "q"), q)

        def _register_2d(mode, g):
            intensity_node = _required_direct_dataset(
                g, "intensity", role=f"{g.name}/intensity",
            )
            q_node = _required_direct_dataset(
                g, "q", role=f"{g.name}/q",
            )
            chi_node = _required_direct_dataset(
                g, "chi", role=f"{g.name}/chi",
            )
            if intensity_node is None or q_node is None or chi_node is None:
                return
            q = _read_numeric_vector(
                q_node,
                role=f"{g.name}/q",
            )
            chi = _read_numeric_vector(
                chi_node,
                role=f"{g.name}/chi",
            )
            _qualify_2d_stack(
                g, intensity_node,
                q_points=int(q.size),
                chi_points=int(chi.size),
                role="intensity",
            )
            sigma_node = _required_direct_dataset(
                g, "sigma", role=f"{g.name}/sigma",
            )
            if sigma_node is not None:
                _qualify_2d_stack(
                    g, sigma_node,
                    q_points=int(q.size),
                    chi_points=int(chi.size),
                    role="sigma",
                )
            self._g2_modes[mode] = g
            self._map_2d_modes[mode] = _frame_map(g, self.target_frame)
            qu, cu = _dataset_unit(g, "q"), _dataset_unit(g, "chi")
            self._axis_2d_x_modes[mode] = axis_from_unit(
                qu, q,
            )
            self._axis_2d_y_modes[mode] = axis_from_unit(
                cu, chi,
            )
            self._two_d_kind_modes[mode] = _decode_kind(
                _bounded_text_attr(
                    g, "two_d_kind", role=f"{g.name} 2-D kind",
                ),
                qu,
                cu,
            )

        if self._g1 is not None:
            _register_1d(self._primary_mode_1d, self._g1)
            for k in GI_MODE_KEYS_1D:
                if k == self._primary_mode_1d:
                    continue
                child_name = mode_subgroup_name(k)
                child = _required_direct_group(
                    self._g1, child_name, role=f"{self._g1.name}/{child_name}",
                )
                if child is not None:
                    _register_1d(k, child)
        if self._g2 is not None:
            _register_2d(self._primary_mode_2d, self._g2)
            for k in GI_MODE_KEYS_2D:
                if k == self._primary_mode_2d:
                    continue
                child_name = mode_subgroup_name(k)
                child = _required_direct_group(
                    self._g2, child_name, role=f"{self._g2.name}/{child_name}",
                )
                if child is not None:
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
        self._geom = None
        self._scan_data = None
        self._frames = None
        self._scan_data_columns = None
        self._scan_data_items = ()

    def _row(self, mapping: dict[int, int], frame: int) -> int | None:
        return mapping.get(int(frame))

    def labels(self) -> tuple[int, ...]:
        """Frame labels known to this scan without reopening the file."""

        labels = set(self._map_geom) | set(self._map_scan_data)
        for mode_map in (*self._map_1d_modes.values(), *self._map_2d_modes.values()):
            labels |= set(mode_map)
        frames = self._frames
        if frames is not None:
            if len(frames) > _MAX_PERSISTED_FRAME_ROWS:
                raise ValueError(f"{frames.name} inventory exceeds limit")
            for name in frames:
                if not str(name).startswith("frame_"):
                    continue
                try:
                    labels.add(int(str(name).removeprefix("frame_")))
                except ValueError:
                    continue
        return tuple(sorted(labels))

    def has_frame(self, frame: int) -> bool:
        """Whether one exact frame has any persisted processed record."""
        frame = int(frame)
        maps = (self._map_1d, self._map_2d, self._map_geom, self._map_scan_data,
                *self._map_1d_modes.values(), *self._map_2d_modes.values())
        if any(frame in mapping for mapping in maps):
            return True
        record = (
            None
            if self._frames is None
            else _direct_group(self._frames, f"frame_{frame:04d}")
        )
        return bool(record is not None and len(record))

    def _metadata_for_frame(self, frame: int) -> dict[str, object]:
        row = self._row(self._map_scan_data, frame)
        group = self._scan_data
        if row is None or group is None:
            return {}
        if self.target_frame is not None:
            out: dict[str, object] = {}
            for key, item in self._scan_data_items:
                value = _scan_data_scalar(
                    item, row, role=f"{group.name}/{key}",
                )
                if np.asarray(value).shape == ():
                    out[str(key)] = _decode(np.asarray(value).item())
            return out
        cols = self._scan_data_columns
        if cols is None:
            # Read each scan_data column ONCE per open, then slice by row —
            # not once per (frame, column).  read_frame_views() loops every
            # frame, so the old per-frame ``item[()]`` full-column read was
            # O(N^2) in the column reads on long scans.
            cols = {}
            total = 0
            for key, item in self._scan_data_items:
                column, logical_bytes = _scan_data_column(
                    item,
                    role=f"{group.name}/{key}",
                    allowance_bytes=_MAX_SCAN_DATA_BYTES - total,
                )
                total += logical_bytes
                if total > _MAX_SCAN_DATA_BYTES:  # defensive contract check
                    raise AssertionError("scan-data reader exceeded its allowance")
                cols[str(key)] = column
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
            item = _required_direct_dataset(
                group, name, role=f"{group.name}/{name}",
            )
            rows = _direct_dataset(group, "frame_index")
            if item is None:
                return None
            if (
                rows is None
                or item.shape != rows.shape
                or item.ndim != 1
                or item.dtype.kind not in "iuf"
                or item.dtype.itemsize > 8
            ):
                raise ValueError(f"{group.name}/{name} is not a bounded frame column")
            try:
                return float(item[row])
            except (TypeError, ValueError, IndexError):
                return None

        return FrameGeometry(
            rot1=read_scalar("rot1"),
            rot2=read_scalar("rot2"),
            rot3=read_scalar("rot3"),
            incident_angle=read_scalar("incident_angle"),
        )

    def _thumbnail_for_frame(
        self, frame: int, *, include_data: bool = True,
    ) -> tuple[np.ndarray | None, bool]:
        frames = self._frames
        if not self.include_thumbnail or frames is None:
            return None, False
        frame_name = f"frame_{int(frame):04d}"
        fg = _required_direct_group(
            frames, frame_name, role=f"{frames.name}/{frame_name}",
        )
        if fg is None:
            return None, False
        thumbnail = _required_direct_dataset(
            fg, "thumbnail", role=f"{fg.name}/thumbnail",
        )
        if thumbnail is None:
            return None, False
        mask_baked = _bounded_bool_attr(
            thumbnail,
            "mask_baked",
            role=f"{thumbnail.name} mask marker",
        )
        return (
            _read_thumbnail(fg, thumbnail) if include_data else None,
            True if mask_baked is None else mask_baked,
        )

    def _source_for_frame(self, frame: int) -> tuple[str | None, int | None]:
        frames = self._frames
        if frames is None:
            return None, None
        frame_name = f"frame_{int(frame):04d}"
        fg = _required_direct_group(
            frames, frame_name, role=f"{frames.name}/{frame_name}",
        )
        if fg is None:
            return None, None
        src = _required_direct_group(
            fg, "source", role=f"{fg.name}/source",
        )
        if src is None:
            return None, None
        path = None
        source_idx = None
        path_node = _required_direct_dataset(
            src, "path", role=f"{src.name}/path",
        )
        if path_node is not None:
            stored = _bounded_utf8_scalar(
                path_node,
                role=f"{src.name}/path",
                max_bytes=_MAX_SOURCE_PATH_BYTES,
            )
            if not self.resolve_source:
                path = stored
            else:
                # Resolve for normal FrameView consumers; FramePreview asks for
                # the exact persisted locator and performs its own bounded read.
                from xrd_tools.io.read import resolve_source_master
                resolved = resolve_source_master(
                    stored, scan_file=self.path,
                    source_base=self._source_base, source_root=self.source_root)
                path = str(resolved) if resolved is not None else stored
        source_index_node = _required_direct_dataset(
            src, "frame_index", role=f"{src.name}/frame_index",
        )
        if source_index_node is not None:
            source_idx = _bounded_integral_scalar(
                source_index_node,
                role=f"{src.name}/frame_index",
            )
        return path, source_idx

    def _common_fields(
        self, frame: int, *, include_heavy: bool = True,
    ) -> dict:
        """Shared per-frame fields (thumbnail/source/metadata/geometry) — the
        same for every mode of one frame."""
        thumbnail, mask_baked = self._thumbnail_for_frame(
            frame, include_data=include_heavy,
        )
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
            points = int(self._axis_1d_modes[m1].values.size)
            intensity_1d = _read_1d_row(
                g1, "intensity", row_1d, points=points,
            )
            if _required_direct_dataset(
                g1, "sigma", role=f"{g1.name}/sigma",
            ) is not None:
                sigma_1d = _read_1d_row(
                    g1, "sigma", row_1d, points=points,
                )

        intensity_2d = sigma_2d = None
        if row_2d is not None and g2 is not None:
            q_points = int(self._axis_2d_x_modes[m2].values.size)
            chi_points = int(self._axis_2d_y_modes[m2].values.size)
            intensity_2d = _read_2d_row(
                g2,
                "intensity",
                row_2d,
                q_points=q_points,
                chi_points=chi_points,
            )
            if _required_direct_dataset(
                g2, "sigma", role=f"{g2.name}/sigma",
            ) is not None:
                sigma_2d = _read_2d_row(
                    g2,
                    "sigma",
                    row_2d,
                    q_points=q_points,
                    chi_points=chi_points,
                )

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

    def _view_for(
        self,
        frame: int,
        dim: str,
        mode: str,
        *,
        include_heavy: bool = True,
        common: dict | None = None,
    ) -> "FrameView | None":
        """Dimension-pure :class:`FrameView` for one ``(dim, mode)``, or ``None``
        if absent.  Shared per-frame fields are included so a record's per-dim
        views carry them; :class:`FrameRecord` re-projects to dimension-pure."""
        frame = int(frame)
        if dim == "1d":
            g = self._g1_modes.get(mode)
            row = self._row(self._map_1d_modes.get(mode, {}), frame)
            if g is None or row is None:
                return None
            points = int(self._axis_1d_modes[mode].values.size)
            sigma_node = _required_direct_dataset(
                g, "sigma", role=f"{g.name}/sigma",
            )
            sig = (
                _read_1d_row(g, "sigma", row, points=points)
                if sigma_node is not None else None
            )
            return FrameView(
                label=frame, axis_1d=self._axis_1d_modes.get(mode),
                intensity_1d=_read_1d_row(
                    g, "intensity", row, points=points,
                ), sigma_1d=sig,
                **(common if common is not None else self._common_fields(frame)),
            )
        g = self._g2_modes.get(mode)
        row = self._row(self._map_2d_modes.get(mode, {}), frame)
        if g is None or row is None:
            return None
        q_points = int(self._axis_2d_x_modes[mode].values.size)
        chi_points = int(self._axis_2d_y_modes[mode].values.size)
        sigma_node = _required_direct_dataset(
            g, "sigma", role=f"{g.name}/sigma",
        )
        sig = (
            _read_2d_row(
                g,
                "sigma",
                row,
                q_points=q_points,
                chi_points=chi_points,
            )
            if include_heavy and sigma_node is not None else None
        )
        return FrameView(
            label=frame, axis_2d_x=self._axis_2d_x_modes.get(mode),
            axis_2d_y=self._axis_2d_y_modes.get(mode),
            intensity_2d=(
                _read_2d_row(
                    g,
                    "intensity",
                    row,
                    q_points=q_points,
                    chi_points=chi_points,
                )
                if include_heavy else None
            ),
            sigma_2d=sig,
            two_d_kind=self._two_d_kind_modes.get(mode, TwoDKind.Q_CHI),
            **(
                common
                if common is not None
                else self._common_fields(frame, include_heavy=include_heavy)
            ),
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

    def read_record(
        self, frame: int, *, include_heavy: bool = True,
    ) -> FrameRecord:
        """Read every mode of ``frame`` into a multi-result :class:`FrameRecord`.

        On a single-mode/old file this is exactly
        ``FrameRecord.from_view(self.read(frame))`` (one ``DEFAULT_MODE_KEY``
        entry per dimension).  With ``include_heavy=False``, complete 1-D
        modes and per-frame metadata are retained while 2-D modes become
        array-free shells and thumbnail pixels are skipped.  This gives
        long-scan Browse projection its full cheap trace membership without
        eagerly reading detector/cake payloads that only the current frame
        needs."""
        frame = int(frame)
        common = self._common_fields(frame, include_heavy=include_heavy)
        r1d: dict = {}
        r2d: dict = {}
        for m in self.modes_1d():
            v = self._view_for(frame, "1d", m, common=common)
            if v is not None and v.has_1d:
                r1d[m] = v
        for m in self.modes_2d():
            v = self._view_for(
                frame, "2d", m,
                include_heavy=include_heavy,
                common=common,
            )
            if v is not None and (include_heavy is False or v.has_2d):
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
    include_heavy: bool = True,
):
    """Yield :class:`FrameRecord` objects one at a time from one open reader.

    ``include_heavy=False`` keeps complete 1-D modes and structural 2-D mode
    shells while avoiding per-frame 2-D and thumbnail pixel reads.
    """

    with FrameViewReader(
        scan_file, entry=entry, include_thumbnail=include_thumbnail,
        source_root=source_root,
    ) as reader:
        labels = reader.labels() if frames is None else frames
        for frame in labels:
            yield reader.read_record(
                int(frame), include_heavy=include_heavy,
            )


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
