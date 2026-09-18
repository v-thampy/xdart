"""FrameView readers for processed xdart/ssrl NeXus scans."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum
from os.path import isabs, normcase, normpath
from pathlib import Path
import math
import sys
from threading import RLock
from types import MappingProxyType
from typing import Callable, Iterable, Mapping
from weakref import ReferenceType, ref

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
from xrd_tools.core.physical_memory import (
    PhysicalRootAuthority,
    PhysicalRootLease,
    PhysicalRootReservation,
    physical_root_fact,
)
from xrd_tools.io.read import _decode
from xrd_tools.io.schema import (
    MONOTONIC_ATTR,
    THUMBNAIL_LUT_ATTRS,
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
_MAX_READER_RETAINED_BYTES = 64 * 1024 * 1024
# Scalar catalogs deliberately have a tighter Python-object projection budget
# than persisted numerical stacks.  100k rows and two million scalar facts
# comfortably cover a 651-frame scan even at the full 256-column metadata
# ceiling, while refusing object-heap amplification from a million-row file.
_MAX_SCALAR_CATALOG_ROWS = 100_000
_MAX_SCALAR_CATALOG_FACTS = 2_000_000
_SCALAR_CATALOG_FIXED_FACTS_PER_ROW = 13
# A 1-D row projection is a caller-owned Python/NumPy graph, not a streaming
# iterator.  Keep its label/membership graph bounded independently of HDF row
# width, and cap unique result roots/bytes at the established 64 MiB analysis
# payload ceiling.  651 frames across all five named GI 1-D modes, 1000 points,
# and intensity+sigma occupies about 52 MiB and remains admitted.
_MAX_1D_RESULT_LABELS = 100_000
_MAX_1D_RESULT_MEMBERSHIPS = 100_000
_MAX_1D_RESULT_ROOTS = 200_256
_MAX_1D_RESULT_BYTES = 64 * 1024 * 1024


_SCALAR_METADATA_TYPES = (type(None), bool, int, float, str)


def _is_current_average_frame_one(entry: object) -> bool:
    """Recognize the exact current averaged frame without reading counts."""

    if not isinstance(entry, h5py.Group):
        return False
    frames_link = entry.get("frames", getlink=True)
    frames = entry.get("frames")
    if (
        type(frames_link) is not h5py.HardLink
        or not isinstance(frames, h5py.Group)
        or "frame_0001" not in frames
    ):
        return False
    frame_link = frames.get("frame_0001", getlink=True)
    frame = frames.get("frame_0001")
    if (
        type(frame_link) is not h5py.HardLink
        or not isinstance(frame, h5py.Group)
        or "source" in frame
    ):
        return False
    counts_link = frame.get("finite_counts", getlink=True)
    counts = frame.get("finite_counts")
    required_attrs = {
        "average_scan_policy",
        "contributor_extent",
        "finite_counts_sha256",
        "finite_counts_min",
        "finite_counts_max",
        "finite_counts_zero_count",
    }
    return bool(
        type(counts_link) is h5py.HardLink
        and isinstance(counts, h5py.Dataset)
        and counts.dtype == np.dtype("<u4")
        and counts.ndim == 2
        and all(int(part) > 0 for part in counts.shape)
        and required_attrs.issubset(counts.attrs)
        and _bounded_text_attr(
            counts,
            "average_scan_policy",
            role=f"{counts.name} Average policy",
            max_bytes=32,
        ) == "average_scan_v1"
    )


def _frozen_scalar_metadata(
    value: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, float]]:
    """Validate and freeze one recursively scalar metadata projection."""

    if not isinstance(value, Mapping):
        raise TypeError("metadata_raw must be a mapping")
    raw: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise TypeError("metadata_raw keys must be exact nonempty strings")
        if type(item) not in _SCALAR_METADATA_TYPES:
            raise TypeError(
                "metadata_raw values must be exact scalar builtins or None"
            )
        raw[key] = item
    numeric: dict[str, float] = {}
    for key, item in raw.items():
        try:
            number = float(item)
        except (OverflowError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric[key] = number
    return MappingProxyType(raw), MappingProxyType(numeric)


def _exact_mode_tuple(value: object, *, role: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{role} must be an exact tuple")
    result: list[str] = []
    for mode in value:
        if type(mode) is not str or not mode:
            raise TypeError(f"{role} entries must be exact nonempty strings")
        if mode in result:
            raise ValueError(f"{role} contains a duplicate mode")
        result.append(mode)
    return tuple(result)


def _validate_scalar_catalog_projection(
    *,
    label_count: int,
    metadata_column_count: int,
    mode_membership_count: int,
    axis_descriptor_count: int = 0,
) -> None:
    """Refuse a scalar graph whose Python-object projection is excessive."""

    for role, value in (
        ("label_count", label_count),
        ("metadata_column_count", metadata_column_count),
        ("mode_membership_count", mode_membership_count),
        ("axis_descriptor_count", axis_descriptor_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{role} must be an exact nonnegative integer")
    if label_count > _MAX_SCALAR_CATALOG_ROWS:
        raise ValueError("Frame scalar catalog row projection exceeds limit")
    projected = (
        label_count
        * (_SCALAR_CATALOG_FIXED_FACTS_PER_ROW + metadata_column_count)
        + mode_membership_count
        + 4 * axis_descriptor_count
    )
    if projected > _MAX_SCALAR_CATALOG_FACTS:
        raise ValueError("Frame scalar catalog fact projection exceeds limit")


def _validate_1d_result_projection(
    *,
    label_count: int,
    membership_count: int,
    root_count: int,
    projected_bytes: int,
) -> None:
    """Refuse an excessive eager 1-D result graph before row allocation."""

    for role, value in (
        ("label_count", label_count),
        ("membership_count", membership_count),
        ("root_count", root_count),
        ("projected_bytes", projected_bytes),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"{role} must be an exact nonnegative integer")
    if label_count > _MAX_1D_RESULT_LABELS:
        raise ValueError("Frame 1-D result label projection exceeds limit")
    if membership_count > _MAX_1D_RESULT_MEMBERSHIPS:
        raise ValueError("Frame 1-D result membership projection exceeds limit")
    if root_count > _MAX_1D_RESULT_ROOTS:
        raise ValueError("Frame 1-D result root projection exceeds limit")
    if projected_bytes > _MAX_1D_RESULT_BYTES:
        raise ValueError("Frame 1-D result byte projection exceeds limit")


@dataclass(frozen=True, slots=True, eq=False)
class FrameScalarRow:
    """Array-free immutable catalog facts for one persisted frame."""

    label: int
    metadata_raw: Mapping[str, object] = field(default_factory=dict)
    metadata_numeric: Mapping[str, float] = field(init=False)
    geometry: FrameGeometry | None = None
    source_path: str | None = None
    source_frame_index: int | None = None
    has_thumbnail: bool = False
    mask_baked: bool = False
    averaged: bool = False
    modes_1d: tuple[str, ...] = ()
    modes_2d: tuple[str, ...] = ()
    active_mode_1d: str | None = None
    active_mode_2d: str | None = None
    two_d_kinds: tuple[tuple[str, TwoDKind], ...] = ()

    def __post_init__(self) -> None:
        if type(self.label) is not int or self.label < 0:
            raise TypeError("label must be an exact nonnegative integer")
        raw, numeric = _frozen_scalar_metadata(self.metadata_raw)
        object.__setattr__(self, "metadata_raw", raw)
        object.__setattr__(self, "metadata_numeric", numeric)

        geometry = self.geometry
        if geometry is not None:
            if type(geometry) is not FrameGeometry or geometry.poni is not None:
                raise TypeError("geometry must be an exact array-free FrameGeometry")
            for name in ("rot1", "rot2", "rot3", "incident_angle"):
                scalar = getattr(geometry, name)
                if scalar is not None and type(scalar) is not float:
                    raise TypeError(
                        "geometry values must be exact floats or None"
                    )
                if scalar is not None and not math.isfinite(scalar):
                    raise ValueError(
                        "geometry values must be finite floats or None"
                    )

        if self.source_path is not None:
            if type(self.source_path) is not str or not self.source_path:
                raise TypeError("source_path must be an exact nonempty string or None")
            if len(self.source_path.encode("utf-8")) > _MAX_SOURCE_PATH_BYTES:
                raise ValueError("source_path exceeds the scalar path limit")
        if self.source_frame_index is not None and (
            type(self.source_frame_index) is not int
            or self.source_frame_index < 0
        ):
            raise TypeError(
                "source_frame_index must be an exact nonnegative integer or None"
            )
        if (
            type(self.has_thumbnail) is not bool
            or type(self.mask_baked) is not bool
            or type(self.averaged) is not bool
        ):
            raise TypeError("frame marker facts must be exact booleans")

        modes_1d = _exact_mode_tuple(self.modes_1d, role="modes_1d")
        modes_2d = _exact_mode_tuple(self.modes_2d, role="modes_2d")
        object.__setattr__(self, "modes_1d", modes_1d)
        object.__setattr__(self, "modes_2d", modes_2d)
        for active, modes, role in (
            (self.active_mode_1d, modes_1d, "active_mode_1d"),
            (self.active_mode_2d, modes_2d, "active_mode_2d"),
        ):
            if active is not None and (
                type(active) is not str or active not in modes
            ):
                raise ValueError(f"{role} must be None or a present exact mode")

        if type(self.two_d_kinds) is not tuple:
            raise TypeError("two_d_kinds must be an exact tuple")
        kinds: list[tuple[str, TwoDKind]] = []
        for index, item in enumerate(self.two_d_kinds):
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not TwoDKind
            ):
                raise TypeError(
                    "two_d_kinds entries must be exact (mode, TwoDKind) tuples"
                )
            if index >= len(modes_2d) or item[0] != modes_2d[index]:
                raise ValueError("two_d_kinds must align exactly with modes_2d")
            kinds.append(item)
        if len(kinds) != len(modes_2d):
            raise ValueError("two_d_kinds must cover every 2-D mode")
        object.__setattr__(self, "two_d_kinds", tuple(kinds))


@dataclass(frozen=True, slots=True, eq=False)
class FrameScalarCatalog:
    """Stable array-free scalar inventory for one processed artifact."""

    artifact_path: str
    entry: str
    rows: tuple[FrameScalarRow, ...]
    axes_1d: tuple[tuple[str, str, str, bool], ...] = ()
    source_base: str | None = None
    labels: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if type(self.artifact_path) is not str or not self.artifact_path:
            raise TypeError("artifact_path must be an exact nonempty string")
        if type(self.entry) is not str or not self.entry:
            raise TypeError("entry must be an exact nonempty string")
        if type(self.rows) is not tuple:
            raise TypeError("rows must be an exact tuple")
        if self.source_base is not None and (
            type(self.source_base) is not str or not self.source_base
        ):
            raise TypeError("source_base must be nonempty text or None")
        labels: list[int] = []
        previous = -1
        for row in self.rows:
            if type(row) is not FrameScalarRow:
                raise TypeError("catalog rows must be exact FrameScalarRow objects")
            if row.label <= previous:
                raise ValueError("catalog row labels must be strictly increasing")
            labels.append(row.label)
            previous = row.label
        object.__setattr__(self, "labels", tuple(labels))

        if type(self.axes_1d) is not tuple:
            raise TypeError("axes_1d must be an exact tuple")
        axes_1d: list[tuple[str, str, str, bool]] = []
        axis_modes: list[str] = []
        for descriptor in self.axes_1d:
            if (
                type(descriptor) is not tuple
                or len(descriptor) != 4
                or type(descriptor[0]) is not str
                or not descriptor[0]
                or type(descriptor[1]) is not str
                or not descriptor[1]
                or type(descriptor[2]) is not str
                or type(descriptor[3]) is not bool
            ):
                raise TypeError(
                    "axes_1d entries must be exact "
                    "(mode, label, unit, log) tuples"
                )
            if descriptor[0] in axis_modes:
                raise ValueError("axes_1d contains a duplicate mode")
            axes_1d.append(descriptor)
            axis_modes.append(descriptor[0])
        for row in self.rows:
            if any(mode not in axis_modes for mode in row.modes_1d):
                raise ValueError("axes_1d must describe every row 1-D mode")
        object.__setattr__(self, "axes_1d", tuple(axes_1d))

    def row(self, label: int) -> FrameScalarRow | None:
        """Return one exact label by callback-free integer bisection."""

        if type(label) is not int or label < 0:
            raise TypeError("label must be an exact nonnegative integer")
        index = bisect_left(self.labels, label)
        if index == len(self.labels) or self.labels[index] != label:
            return None
        return self.rows[index]


def _exact_readonly_float64_vector(
    value: object,
    *,
    role: str,
    points: int | None = None,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.ndim != 1
        or not value.flags.c_contiguous
        or (points is not None and value.shape != (points,))
    ):
        raise TypeError(f"{role} must be an exact C float64 vector")
    if value.flags.writeable:
        value.setflags(write=False)
    if value.flags.writeable:
        raise ValueError(f"{role} must be read-only")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Frame1DModeRows:
    """Immutable rows for one persisted 1-D result mode."""

    mode: str
    axis: Axis
    labels: tuple[int, ...]
    intensity_rows: tuple[np.ndarray, ...]
    sigma_rows: tuple[np.ndarray, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or not self.mode:
            raise TypeError("mode must be an exact nonempty string")
        axis = self.axis
        if (
            type(axis) is not Axis
            or type(axis.label) is not str
            or not axis.label
            or type(axis.unit) is not str
            or type(axis.log) is not bool
            or axis.values is None
        ):
            raise TypeError("axis must be an exact sampled Axis")
        axis_values = _exact_readonly_float64_vector(
            axis.values, role="axis values",
        )
        points = int(axis_values.size)

        if type(self.labels) is not tuple:
            raise TypeError("labels must be an exact tuple")
        labels: list[int] = []
        previous = -1
        for label in self.labels:
            if type(label) is not int or label < 0:
                raise TypeError("labels must contain exact nonnegative integers")
            if label <= previous:
                raise ValueError("labels must be strictly increasing")
            labels.append(label)
            previous = label

        if (
            type(self.intensity_rows) is not tuple
            or len(self.intensity_rows) != len(labels)
        ):
            raise ValueError("intensity_rows must align exactly with labels")
        intensity_rows = tuple(
            _exact_readonly_float64_vector(
                row, role="intensity row", points=points,
            )
            for row in self.intensity_rows
        )

        sigma_rows = self.sigma_rows
        if sigma_rows is not None:
            if type(sigma_rows) is not tuple or len(sigma_rows) != len(labels):
                raise ValueError("sigma_rows must align exactly with labels")
            sigma_rows = tuple(
                _exact_readonly_float64_vector(
                    row, role="sigma row", points=points,
                )
                for row in sigma_rows
            )

        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "intensity_rows", intensity_rows)
        object.__setattr__(self, "sigma_rows", sigma_rows)

    def row(
        self, label: int,
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Return exact row arrays by callback-free integer bisection."""

        if type(label) is not int or label < 0:
            raise TypeError("label must be an exact nonnegative integer")
        index = bisect_left(self.labels, label)
        if index == len(self.labels) or self.labels[index] != label:
            return None
        sigma = None if self.sigma_rows is None else self.sigma_rows[index]
        return self.intensity_rows[index], sigma


@dataclass(frozen=True, slots=True, eq=False)
class Frame1DRows:
    """Immutable requested-label projection of all persisted 1-D modes."""

    artifact_path: str
    entry: str
    labels: tuple[int, ...]
    modes: tuple[Frame1DModeRows, ...]
    primary_mode: str | None = None

    def __post_init__(self) -> None:
        if type(self.artifact_path) is not str or not self.artifact_path:
            raise TypeError("artifact_path must be an exact nonempty string")
        if type(self.entry) is not str or not self.entry:
            raise TypeError("entry must be an exact nonempty string")
        if type(self.labels) is not tuple or not self.labels:
            raise TypeError("labels must be a nonempty exact tuple")
        labels: list[int] = []
        previous = -1
        for label in self.labels:
            if type(label) is not int or label < 0:
                raise TypeError("labels must contain exact nonnegative integers")
            if label <= previous:
                raise ValueError("labels must be strictly increasing")
            labels.append(label)
            previous = label
        if type(self.modes) is not tuple:
            raise TypeError("modes must be an exact tuple")
        modes: list[Frame1DModeRows] = []
        mode_names: list[str] = []
        for mode_rows in self.modes:
            if type(mode_rows) is not Frame1DModeRows:
                raise TypeError("modes must contain exact Frame1DModeRows")
            if mode_rows.mode in mode_names:
                raise ValueError("modes contains a duplicate mode")
            for label in mode_rows.labels:
                index = bisect_left(labels, label)
                if index == len(labels) or labels[index] != label:
                    raise ValueError("mode labels must be selected labels")
            modes.append(mode_rows)
            mode_names.append(mode_rows.mode)
        primary = self.primary_mode
        if primary is not None and (
            type(primary) is not str or primary not in mode_names
        ):
            raise ValueError("primary_mode must be None or a present exact mode")
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "modes", tuple(modes))

    def mode(self, name: str) -> Frame1DModeRows | None:
        if type(name) is not str or not name:
            raise TypeError("mode name must be an exact nonempty string")
        for mode_rows in self.modes:
            if mode_rows.mode == name:
                return mode_rows
        return None


@dataclass(frozen=True, slots=True, eq=False)
class _Frame1DReadModePlan:
    mode: str
    group: h5py.Group
    axis: Axis
    rows: tuple[tuple[int, int], ...]
    has_sigma: bool


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
    # The link was qualified above; open it without a second link lookup or
    # Group.get's redundant membership probe.
    try:
        value = group[name]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} is not a local hard-linked group") from error
    if not isinstance(value, h5py.Group):
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
    try:
        value = group[name]
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"{role} is not a dataset")
        external = tuple(value.external or ())
        if value.is_virtual or external:
            raise ValueError(f"{role} is not locally stored")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} is not a bounded local hard-linked dataset") from error
    return value


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


def _read_numeric_vector(
    dataset: object,
    *,
    role: str,
    bundle: "_ReaderArrayBundle | None" = None,
) -> np.ndarray:
    qualified = _qualify_vector(
        dataset,
        role=role,
        max_items=_MAX_AXIS_POINTS,
        max_bytes=_MAX_1D_ROW_BYTES,
        numeric=True,
    )
    if bundle is None:
        return np.asarray(qualified[()], dtype=np.float64, order="C")
    return bundle.read_hdf(
        qualified,
        role=role,
        validator=lambda value: _qualify_vector(
            value,
            role=role,
            max_items=_MAX_AXIS_POINTS,
            max_bytes=_MAX_1D_ROW_BYTES,
            numeric=True,
        ),
        selection=None,
        final_dtype=np.dtype(np.float64),
        transform="numeric-float64",
        retain=True,
    )


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
    group: h5py.Group,
    name: str,
    row: int,
    *,
    points: int,
    bundle: "_ReaderArrayBundle | None" = None,
) -> np.ndarray:
    dataset = _qualify_1d_stack(
        group,
        _required_direct_dataset(group, name, role=f"{group.name}/{name}"),
        points=points,
        role=name,
    )
    if not 0 <= int(row) < int(dataset.shape[0]):
        raise ValueError(f"{group.name}/{name} row is out of range")
    if bundle is None:
        return np.asarray(dataset[int(row)], dtype=np.float64, order="C")
    role = f"{group.name}/{name}"
    return bundle.read_hdf(
        dataset,
        role=role,
        validator=lambda value: _qualify_1d_stack(
            group, value, points=points, role=name,
        ),
        selection=int(row),
        final_dtype=np.dtype(np.float64),
        transform="numeric-float64",
        retain=False,
    )


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
    bundle: "_ReaderArrayBundle | None" = None,
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
    if bundle is None:
        return np.asarray(dataset[int(row)], dtype=np.float64, order="C")
    role = f"{group.name}/{name}"
    return bundle.read_hdf(
        dataset,
        role=role,
        validator=lambda value: _qualify_2d_stack(
            group,
            value,
            q_points=q_points,
            chi_points=chi_points,
            role=name,
        ),
        selection=int(row),
        final_dtype=np.dtype(np.float64),
        transform="numeric-float64",
        retain=False,
    )


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
    fixed_objects: set[int] = set()
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
            try:
                address = int(h5py.h5o.get_info(item.id).addr)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{group.name}/{key} object identity is unavailable"
                ) from error
            if address not in fixed_objects:
                fixed_objects.add(address)
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
    dataset: h5py.Dataset,
    *,
    role: str,
    allowance_bytes: int,
    bundle: "_ReaderArrayBundle | None" = None,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[np.ndarray, int]:
    if type(allowance_bytes) is not int or allowance_bytes < 0:
        raise ValueError(f"{role} has no remaining materialization allowance")
    if h5py.check_string_dtype(dataset.dtype) is None:
        if bundle is None:
            logical_bytes = int(dataset.size) * int(dataset.dtype.itemsize)
            if logical_bytes > allowance_bytes:
                raise ValueError(f"{role} materialized bytes exceed allowance")
            value = np.asarray(dataset[()])
        else:
            # Qualify this scientific role and consult the exact physical
            # object cache before charging the caller's remaining logical
            # allowance.  A second hard-link name costs no new storage/read.
            bundle._callback(
                _qualify_vector,
                dataset,
                role=role,
                max_items=_MAX_PERSISTED_FRAME_ROWS,
                max_bytes=_MAX_SCAN_DATA_BYTES,
                numeric=False,
            )
            key = bundle.cache_key(
                dataset,
                selection=None,
                final_dtype=dataset.dtype,
                transform="native",
            )
            cached = bundle.cached(key)
            if cached is not None:
                return cached, 0
            logical_bytes = int(dataset.size) * int(dataset.dtype.itemsize)
            if logical_bytes > allowance_bytes:
                raise ValueError(f"{role} materialized bytes exceed allowance")
            value = bundle.read_hdf(
                dataset,
                role=role,
                validator=lambda candidate: _qualify_vector(
                    candidate,
                    role=role,
                    max_items=_MAX_PERSISTED_FRAME_ROWS,
                    max_bytes=_MAX_SCAN_DATA_BYTES,
                    numeric=False,
                ),
                selection=None,
                final_dtype=dataset.dtype,
                transform="native",
                retain=True,
            )
        return value, logical_bytes
    cache_key = None
    if bundle is not None:
        # Repeat role qualification before consulting the physical cache.
        # Logical allowance is charged only for a cache miss.
        bundle._callback(
            _qualify_vector,
            dataset,
            role=role,
            max_items=_MAX_PERSISTED_FRAME_ROWS,
            max_bytes=_MAX_SCAN_DATA_BYTES,
            numeric=False,
        )
        cache_key = bundle.cache_key(
            dataset,
            selection=None,
            final_dtype=np.dtype(object),
            transform="bounded-utf8-object",
        )
        cached = bundle.cached(cache_key)
        if cached is not None:
            return cached, 0
    rows = int(dataset.shape[0])
    pointer_bytes = rows * int(np.dtype(object).itemsize)
    minimum = pointer_bytes + rows * 64
    if minimum > allowance_bytes:
        raise ValueError(f"{role} retained object bytes exceed limit")
    values = (
        np.empty((rows,), dtype=object)
        if bundle is None
        else bundle.allocate_cached(
            cache_key,
            (rows,),
            np.dtype(object),
            role=role,
        )
    )
    total = pointer_bytes
    for row in range(int(dataset.shape[0])):
        if checkpoint is not None:
            checkpoint()
        if total + 64 > allowance_bytes:
            raise ValueError(f"{role} retained object bytes exceed limit")
        value = (
            _bounded_utf8_item(dataset, row, role=role)
            if bundle is None
            else bundle._callback(
                _bounded_utf8_item, dataset, row, role=role,
            )
        )
        if checkpoint is not None:
            checkpoint()
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
    frame_group: h5py.Group,
    thumbnail: h5py.Dataset,
    *,
    bundle: "_ReaderArrayBundle | None" = None,
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

    if bundle is None:
        values = np.asarray(thumbnail[()])
    else:
        values = bundle.read_hdf(
            thumbnail,
            role=thumbnail.name,
            validator=lambda value: value
            if (
                isinstance(value, h5py.Dataset)
                and value.ndim == 2
                and all(1 <= int(part) <= 256 for part in value.shape)
                and value.dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}
                and int(value.size) * int(value.dtype.itemsize)
                <= 256 * 256 * 2
            )
            else (_ for _ in ()).throw(
                ValueError(f"{thumbnail.name} is not a bounded thumbnail")
            ),
            selection=None,
            final_dtype=thumbnail.dtype,
            transform="thumbnail-native",
            retain=False,
        )
    scale = 65535.0 if thumbnail.dtype == np.dtype(np.uint16) else 255.0
    if bundle is None:
        result = vmin + (values.astype(np.float64) / scale) * (vmax - vmin)
    else:
        result = bundle.allocate(
            tuple(int(part) for part in values.shape),
            np.dtype(np.float64),
            role=f"{thumbnail.name} decoded thumbnail",
        )
        np.multiply(
            values,
            (vmax - vmin) / scale,
            out=result,
            casting="unsafe",
        )
        np.add(result, vmin, out=result)
    if mask is not None:
        if bundle is None:
            invalid = np.asarray(mask[()], dtype=bool)
        else:
            invalid = bundle.read_hdf(
                mask,
                role=mask.name,
                validator=lambda value: value
                if (
                    isinstance(value, h5py.Dataset)
                    and value.shape == thumbnail.shape
                    and value.dtype == np.dtype(bool)
                    and value.ndim == 2
                )
                else (_ for _ in ()).throw(
                    ValueError(f"{mask.name} is not a bounded thumbnail mask")
                ),
                selection=None,
                final_dtype=np.dtype(bool),
                transform="thumbnail-mask",
                retain=False,
            )
        result[invalid] = np.nan
    return result


def _decode_kind(value, x_unit: str | None, y_unit: str | None) -> TwoDKind:
    if value is not None:
        try:
            return TwoDKind(str(_decode(value)))
        except ValueError:
            pass
    return two_d_kind_from_units(x_unit, y_unit)


def _frame_map(
    group: h5py.Group | None,
    target_frame: int | None = None,
    *,
    bundle: "_ReaderArrayBundle | None" = None,
) -> dict[int, int]:
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
    values = (
        np.asarray(dataset[()])
        if bundle is None
        else bundle.read_hdf(
            dataset,
            role=f"{group.name}/frame_index",
            validator=lambda value: _qualify_frame_index(group, value),
            selection=None,
            final_dtype=dataset.dtype,
            transform="native",
            retain=True,
        )
    )
    labels = [int(v) for v in values.ravel()]
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


class _ReaderCachePhase(Enum):
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    BUILDING = "building"
    CLOSING_LEASES = "closing-leases"
    CLOSE_AUTHORITY = "close-authority"
    CLOSE_HDF = "close-hdf"


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _ReaderBundleOwner:
    pass


@dataclass(frozen=True, slots=True, eq=False)
class _ReaderCacheEntry:
    array: np.ndarray
    lease: PhysicalRootLease


@dataclass(frozen=True, slots=True, eq=False)
class _ReaderPendingArray:
    key: tuple[object, ...]
    value: np.ndarray
    semantic: object
    retain: bool


@dataclass(frozen=True, slots=True, eq=False)
class _ReaderPendingBundle:
    owner_ref: ReferenceType[_ReaderBundleOwner]
    reservation: PhysicalRootReservation
    opening: bool


@dataclass(frozen=True, slots=True, eq=False)
class _ReaderCacheState:
    generation: int
    phase: _ReaderCachePhase
    open_token: object | None
    entries: Mapping[tuple[object, ...], _ReaderCacheEntry]
    scan_data_columns: Mapping[str, np.ndarray] | None
    pending: _ReaderPendingBundle | None
    close_leases: tuple[PhysicalRootLease, ...]
    close_h5: h5py.File | None


def _reader_cache_state_with(
    state: _ReaderCacheState,
    *,
    phase: _ReaderCachePhase | None = None,
    open_token: object | None = None,
    replace_open_token: bool = False,
    entries: Mapping[tuple[object, ...], _ReaderCacheEntry] | None = None,
    scan_data_columns: Mapping[str, np.ndarray] | None = None,
    replace_scan_data_columns: bool = False,
    pending: _ReaderPendingBundle | None = None,
    replace_pending: bool = False,
    close_leases: tuple[PhysicalRootLease, ...] | None = None,
    replace_close_leases: bool = False,
    close_h5: h5py.File | None = None,
    replace_close_h5: bool = False,
) -> _ReaderCacheState:
    return _ReaderCacheState(
        state.generation + 1,
        state.phase if phase is None else phase,
        open_token if replace_open_token else state.open_token,
        state.entries if entries is None else entries,
        (
            scan_data_columns
            if replace_scan_data_columns
            else state.scan_data_columns
        ),
        pending if replace_pending else state.pending,
        close_leases if replace_close_leases else state.close_leases,
        close_h5 if replace_close_h5 else state.close_h5,
    )


class _ReaderArrayBundle:
    """One all-or-none HDF array read bundle for one open reader."""

    def __init__(
        self, reader: "FrameViewReader", *, opening: bool = False,
    ) -> None:
        self._reader = reader
        self._owner = _ReaderBundleOwner()
        self._reservation = reader._begin_array_bundle(
            self._owner, opening=opening,
        )
        self._token = object()
        self._counter = 0
        self._local: dict[
            tuple[object, ...], tuple[np.ndarray, object, bool]
        ] = {}
        self._active = True
        self._terminal: str | None = None
        self.pending_scan_data_columns: dict[str, np.ndarray] | None = None

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("FrameView read bundle is closed")

    def _callback(self, callback, /, *args, **kwargs):
        """Run one callback window against the exact BUILDING/local state."""

        self._require_active()
        prior_local = self._local
        stamp = self._reader._bundle_callback_stamp(self._owner)
        result = callback(*args, **kwargs)
        self._reader._revalidate_bundle_callback(stamp, self._owner)
        if self._local is not prior_local:
            raise RuntimeError("FrameView read-bundle local mapping drifted")
        return result

    def _compact(self, direction: str) -> str:
        self._terminal = direction
        self._active = False
        self._local.clear()
        self.pending_scan_data_columns = None
        self._reservation = None
        self._owner = None
        self._reader = None
        return direction

    def __copy__(self) -> "_ReaderArrayBundle":
        raise TypeError("FrameView read bundle cannot be copied")

    def __deepcopy__(self, _memo: object) -> "_ReaderArrayBundle":
        raise TypeError("FrameView read bundle cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("FrameView read bundle cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("FrameView read bundle cannot be serialized")

    def _semantic(self, kind: str, key: object) -> tuple[object, ...]:
        self._counter += 1
        return (kind, self._token, self._counter, key)

    def _local_lookup(
        self, key: tuple[object, ...],
    ) -> tuple[
        dict[tuple[object, ...], tuple[np.ndarray, object, bool]],
        tuple[np.ndarray, object, bool] | None,
    ]:
        """Read one local entry across an exact callback/state stamp."""

        prior = self._local
        stamp = self._reader._bundle_callback_stamp(
            self._owner,
        )
        value = prior.get(key)
        self._reader._revalidate_bundle_callback(
            stamp, self._owner,
        )
        if self._local is not prior:
            raise RuntimeError("FrameView read-bundle local mapping drifted")
        return prior, value

    def _publish_local(
        self,
        prior: dict[
            tuple[object, ...], tuple[np.ndarray, object, bool]
        ],
        key: tuple[object, ...],
        value: tuple[np.ndarray, object, bool],
    ) -> None:
        """COW-install one local entry after every hash/equality callback."""

        if self._local is not prior:
            raise RuntimeError("FrameView read-bundle local mapping drifted")
        stamp = self._reader._bundle_callback_stamp(
            self._owner,
        )
        candidate = dict(prior)
        self._reader._revalidate_bundle_callback(
            stamp, self._owner,
        )
        if self._local is not prior:
            raise RuntimeError("FrameView read-bundle local mapping drifted")
        candidate[key] = value
        self._reader._revalidate_bundle_callback(stamp, self._owner)
        if self._local is not prior:
            raise RuntimeError("FrameView read-bundle local mapping drifted")
        self._local = candidate

    def _physical_dataset_key(
        self, dataset: h5py.Dataset,
    ) -> tuple[object, int]:
        self._require_active()
        self._reader._assert_bundle_owned(self._owner)
        owner = self._reader._hdf_owner_token
        if owner is None:
            raise RuntimeError("FrameViewReader HDF owner is unavailable")
        try:
            address = int(h5py.h5o.get_info(dataset.id).addr)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"{dataset.name} object identity is unavailable") from error
        if address < 0:
            raise ValueError(f"{dataset.name} object identity is invalid")
        self._reader._assert_bundle_owned(self._owner)
        return owner, address

    def cache_key(
        self,
        dataset: h5py.Dataset,
        *,
        selection: int | None,
        final_dtype: np.dtype,
        transform: str,
    ) -> tuple[object, ...]:
        physical = self._physical_dataset_key(dataset)
        selected = ("all",) if selection is None else ("row", int(selection))
        return (physical, selected, np.dtype(final_dtype).str, transform)

    def cached(self, key: tuple[object, ...]) -> np.ndarray | None:
        self._require_active()
        cached = self._reader._bundle_cached(self._owner, key)
        if cached is not None:
            return cached
        _prior, local = self._local_lookup(key)
        return None if local is None else local[0]

    def read_hdf(
        self,
        dataset: object,
        *,
        role: str,
        validator,
        selection: int | None,
        final_dtype: np.dtype,
        transform: str,
        retain: bool,
    ) -> np.ndarray:
        self._require_active()
        # Scientific role qualification deliberately precedes every cache hit.
        stamp = self._reader._bundle_callback_stamp(self._owner)
        qualified = validator(dataset)
        self._reader._revalidate_bundle_callback(stamp, self._owner)
        if not isinstance(qualified, h5py.Dataset):
            raise ValueError(f"{role} is not a qualified HDF dataset")
        dtype = np.dtype(final_dtype)
        key = self.cache_key(
            qualified,
            selection=selection,
            final_dtype=dtype,
            transform=transform,
        )
        cached = self._reader._bundle_cached(self._owner, key)
        if cached is not None:
            return cached
        local_prior, local = self._local_lookup(key)
        if local is not None:
            if retain and not local[2]:
                self._publish_local(
                    local_prior, key, (local[0], local[1], True),
                )
            return local[0]
        shape = (
            tuple(int(part) for part in qualified.shape)
            if selection is None
            else tuple(int(part) for part in qualified.shape[1:])
        )
        if selection is not None and not 0 <= int(selection) < int(
            qualified.shape[0]
        ):
            raise ValueError(f"{role} row is out of range")
        nbytes = int(math.prod(shape)) * int(dtype.itemsize)
        capacity = self._reader._bundle_reservation_claim(self._owner, nbytes)
        destination = np.empty(shape, dtype=dtype, order="C")
        semantic = self._semantic("reader-cache" if retain else "return", key)
        self._reader._bundle_reservation_bind(
            self._owner, capacity, destination, semantic,
        )
        if destination.size:
            stamp = self._reader._bundle_callback_stamp(self._owner)
            try:
                if selection is None:
                    qualified.read_direct(destination)
                else:
                    qualified.read_direct(
                        destination, source_sel=np.s_[int(selection)],
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError(f"{role} could not be read into final storage") from error
            self._reader._revalidate_bundle_callback(stamp, self._owner)
        if destination.dtype != dtype or not destination.flags.c_contiguous:
            raise AssertionError("FrameView HDF read did not retain final storage")
        self._publish_local(
            local_prior, key, (destination, semantic, retain),
        )
        return destination

    def allocate(
        self, shape: tuple[int, ...], dtype: np.dtype, *, role: str,
    ) -> np.ndarray:
        self._require_active()
        dtype = np.dtype(dtype)
        if (
            type(shape) is not tuple
            or any(type(part) is not int or part < 0 for part in shape)
        ):
            raise TypeError(f"{role} shape is invalid")
        local_prior = self._local
        nbytes = int(math.prod(shape)) * int(dtype.itemsize)
        capacity = self._reader._bundle_reservation_claim(self._owner, nbytes)
        destination = np.empty(shape, dtype=dtype, order="C")
        semantic = self._semantic("return", role)
        self._reader._bundle_reservation_bind(
            self._owner, capacity, destination, semantic,
        )
        key = ("allocated", semantic)
        self._publish_local(
            local_prior, key, (destination, semantic, False),
        )
        return destination

    def allocate_cached(
        self,
        key: tuple[object, ...] | None,
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        role: str,
    ) -> np.ndarray:
        """Claim and bind one cache-owned root before allocation/read."""

        self._require_active()
        if type(key) is not tuple:
            raise TypeError(f"{role} cache key is invalid")
        cached = self._reader._bundle_cached(self._owner, key)
        if cached is not None:
            return cached
        local_prior, local = self._local_lookup(key)
        if local is not None:
            return local[0]
        dtype = np.dtype(dtype)
        if (
            type(shape) is not tuple
            or any(type(part) is not int or part < 0 for part in shape)
        ):
            raise TypeError(f"{role} shape is invalid")
        nbytes = int(math.prod(shape)) * int(dtype.itemsize)
        capacity = self._reader._bundle_reservation_claim(self._owner, nbytes)
        destination = np.empty(shape, dtype=dtype, order="C")
        semantic = self._semantic("reader-cache", key)
        self._reader._bundle_reservation_bind(
            self._owner, capacity, destination, semantic,
        )
        self._publish_local(
            local_prior, key, (destination, semantic, True),
        )
        return destination

    def adopt(
        self,
        key: tuple[object, ...],
        value: np.ndarray,
        *,
        retain: bool,
    ) -> np.ndarray:
        self._require_active()
        cached = self._reader._bundle_cached(self._owner, key)
        if cached is not None:
            return cached
        local_prior, local = self._local_lookup(key)
        if local is not None:
            return local[0]
        semantic = self._semantic("reader-cache" if retain else "return", key)
        self._reader._bundle_reservation_reserve(self._owner, value, semantic)
        self._publish_local(
            local_prior, key, (value, semantic, retain),
        )
        return value

    def finish(self) -> None:
        self._require_active()
        plan = tuple(
            _ReaderPendingArray(key, value, semantic, retain)
            for key, (value, semantic, retain) in self._local.items()
        )
        try:
            direction = self._reader._finish_array_bundle(
                self._owner,
                plan,
                self.pending_scan_data_columns,
            )
        except BaseException as error:
            try:
                direction = self._reader._recover_array_bundle(self._owner)
            except BaseException as recovery_error:
                raise recovery_error from error
            self._compact(direction)
            raise
        self._compact(direction)

    def rollback(self) -> None:
        if not self._active:
            return
        direction = self._reader._rollback_array_bundle(self._owner)
        self._compact(direction)

    def recover(self) -> str:
        self._require_active()
        direction = self._reader._recover_array_bundle(self._owner)
        return self._compact(direction)

    def __enter__(self) -> "_ReaderArrayBundle":
        if not self._active:
            raise RuntimeError("FrameView read bundle is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            self.rollback()


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
        self._scan_data_items: tuple[tuple[str, h5py.Dataset], ...] = ()
        self._hdf_owner_token: object | None = None
        self._reader_cache_lock = RLock()
        self._memory_authority = PhysicalRootAuthority(
            _MAX_READER_RETAINED_BYTES,
        )
        self._memory_authority.close()
        self._reader_cache_state = _ReaderCacheState(
            0,
            _ReaderCachePhase.CLOSED,
            None,
            MappingProxyType({}),
            None,
            None,
            (),
            None,
        )

    def __copy__(self) -> "FrameViewReader":
        raise TypeError("FrameViewReader cannot be copied")

    def __deepcopy__(self, _memo: object) -> "FrameViewReader":
        raise TypeError("FrameViewReader cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("FrameViewReader cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("FrameViewReader cannot be serialized")

    @property
    def retained_bytes(self) -> int:
        """Bytes retained by reader-owned internal array/buffer roots."""
        return self._reader_accounting("bytes")

    @property
    def retained_root_count(self) -> int:
        return self._reader_accounting("roots")

    @property
    def semantic_references(self) -> int:
        return self._reader_accounting("semantics")

    def _reader_accounting(self, kind: str) -> int:
        state = self._snapshot_reader_cache_state()
        if state.phase is _ReaderCachePhase.CLOSED:
            return 0
        if state.phase is not _ReaderCachePhase.OPEN or state.pending is not None:
            raise RuntimeError("FrameView reader-cache accounting is busy")
        authority = self._memory_authority
        if kind == "bytes":
            value = authority.retained_bytes
        elif kind == "roots":
            value = authority.retained_root_count
        elif kind == "semantics":
            value = authority.semantic_references
        else:  # pragma: no cover - internal exact-string contract
            raise AssertionError("unknown FrameView accounting projection")
        if self._snapshot_reader_cache_state() is not state:
            raise RuntimeError("FrameView reader-cache accounting drifted")
        return int(value)

    @property
    def _read_cache(
        self,
    ) -> Mapping[tuple[object, ...], np.ndarray]:
        """Read-only values-only cache diagnostic; leases never escape."""

        state = self._snapshot_reader_cache_state()
        if state.phase is _ReaderCachePhase.CLOSED:
            return MappingProxyType({})
        if state.phase is not _ReaderCachePhase.OPEN or state.pending is not None:
            raise RuntimeError("FrameView reader cache publication is busy")
        values: dict[tuple[object, ...], np.ndarray] = {}
        for key, entry in state.entries.items():
            self._validate_reader_cache_entry(entry)
            values[key] = entry.array
        projection = MappingProxyType(values)
        if self._snapshot_reader_cache_state() is not state:
            raise RuntimeError("FrameView reader cache publication drifted")
        return projection

    @property
    def _scan_data_columns(self) -> Mapping[str, np.ndarray] | None:
        state = self._snapshot_reader_cache_state()
        if state.phase is _ReaderCachePhase.CLOSED:
            return None
        if state.phase is not _ReaderCachePhase.OPEN or state.pending is not None:
            raise RuntimeError("FrameView reader cache publication is busy")
        columns = state.scan_data_columns
        if self._snapshot_reader_cache_state() is not state:
            raise RuntimeError("FrameView scan-data projection drifted")
        return columns

    @staticmethod
    def _validate_reader_cache_entry(entry: _ReaderCacheEntry) -> None:
        if type(entry) is not _ReaderCacheEntry:
            raise RuntimeError("FrameView reader-cache entry is untrusted")
        if type(entry.array) is not np.ndarray or entry.array.flags.writeable:
            raise RuntimeError("FrameView reader-cache array is not read-only")
        if type(entry.lease) is not PhysicalRootLease or entry.lease.released:
            raise RuntimeError("FrameView reader-cache lease is unavailable")

    def _snapshot_reader_cache_state(self) -> _ReaderCacheState:
        with self._reader_cache_lock:
            return self._reader_cache_state

    def _cas_reader_cache_state(
        self,
        expected: _ReaderCacheState,
        replacement: _ReaderCacheState,
    ) -> bool:
        """Exact pointer CAS; no mapping work or callbacks occur under lock."""

        with self._reader_cache_lock:
            if self._reader_cache_state is not expected:
                return False
            self._reader_cache_state = replacement
            return True

    def _transition_reader_cache_state(
        self,
        expected: _ReaderCacheState,
        replacement: _ReaderCacheState,
    ) -> bool:
        try:
            swapped = self._cas_reader_cache_state(expected, replacement)
        except BaseException as error:
            current = self._snapshot_reader_cache_state()
            if current is expected or current is replacement:
                raise
            raise RuntimeError("FrameView reader-cache state drifted") from error
        if swapped:
            return True
        current = self._snapshot_reader_cache_state()
        if current is replacement:
            return True
        if current is expected:
            return False
        raise RuntimeError("FrameView reader-cache state drifted")

    def _enter_inner(self) -> "FrameViewReader":
        handle = self._h5
        if handle is None:
            raise RuntimeError("FrameViewReader has no open HDF handle")
        from xrd_tools.io.processed_scan_id import require_current_processed_groups
        processed = require_current_processed_groups(
            handle,
            self.entry_name,
            container=self.path,
        )
        entry = processed.entry
        x_name, y_name = processed.axis_names
        # N1: the project root the relative source paths point under (None on old
        # absolute-path files; harmless there).
        source_base = _bounded_text_attr(
            entry,
            "source_base",
            role=f"{entry.name} source base",
            max_bytes=_MAX_SOURCE_PATH_BYTES,
        )
        # The file stores portable POSIX spelling; runtime source identities
        # use native separators and case rules (notably on Windows).
        if source_base is not None:
            source_base = normcase(normpath(source_base))
            # A record reduced under another operating system keeps a root
            # that is not a path here ("C:/..." on POSIX; a POSIX root decodes
            # to a drive-less "\\..." on Windows).  It can own no locator on
            # this host: bind none, leave the record readable, and let a
            # selected Project root relocate the raw tree.  The stored
            # attribute is never rewritten.
            if not isabs(source_base):
                source_base = None
        # Admission binds canonical-or-complete-shadow groups and the complete
        # owned mode inventory once; downstream readers do not rediscover it.
        g1 = processed.integrated_1d
        g2 = processed.integrated_2d
        geom = _required_direct_group(
            entry,
            "per_frame_geometry",
            role=f"{entry.name}/per_frame_geometry",
        )
        scan_data = _required_direct_group(
            entry, "scan_data", role=f"{entry.name}/scan_data",
        )
        frames = _required_direct_group(
            entry, "frames", role=f"{entry.name}/frames",
        )
        with _ReaderArrayBundle(self, opening=True) as bundle:
            map_1d = _frame_map(g1, self.target_frame, bundle=bundle)
            map_2d = _frame_map(g2, self.target_frame, bundle=bundle)
            map_geom = _frame_map(geom, self.target_frame, bundle=bundle)
            map_scan_data = _frame_map(
                scan_data, self.target_frame, bundle=bundle,
            )
            scan_data_items = _scan_data_items(scan_data)

            primary_mode_1d = processed.primary_mode_1d
            primary_mode_2d = processed.primary_mode_2d
            multi_result_modes = bool(
                primary_mode_1d != DEFAULT_MODE_KEY
                or primary_mode_2d != DEFAULT_MODE_KEY
            )
            g1_modes: dict[str, h5py.Group] = {}
            g2_modes: dict[str, h5py.Group] = {}
            map_1d_modes: dict[str, dict[int, int]] = {}
            map_2d_modes: dict[str, dict[int, int]] = {}
            axis_1d_modes: dict[str, Axis] = {}
            axis_2d_x_modes: dict[str, Axis] = {}
            axis_2d_y_modes: dict[str, Axis] = {}
            two_d_kind_modes: dict[str, TwoDKind] = {}

            def register_1d(mode, group):
                intensity_node = _required_direct_dataset(
                    group, "intensity", role=f"{group.name}/intensity",
                )
                q_node = _required_direct_dataset(
                    group, x_name, role=f"{group.name}/{x_name}",
                )
                if intensity_node is None or q_node is None:
                    return
                q = _read_numeric_vector(
                    q_node, role=f"{group.name}/{x_name}", bundle=bundle,
                )
                _qualify_1d_stack(
                    group,
                    intensity_node,
                    points=int(q.size),
                    role="intensity",
                )
                sigma_node = _required_direct_dataset(
                    group, "sigma", role=f"{group.name}/sigma",
                )
                if sigma_node is not None:
                    _qualify_1d_stack(
                        group,
                        sigma_node,
                        points=int(q.size),
                        role="sigma",
                    )
                g1_modes[mode] = group
                map_1d_modes[mode] = _frame_map(
                    group, self.target_frame, bundle=bundle,
                )
                axis_1d_modes[mode] = axis_from_unit(
                    _dataset_unit(group, x_name), q,
                )

            def register_2d(mode, group):
                intensity_node = _required_direct_dataset(
                    group, "intensity", role=f"{group.name}/intensity",
                )
                q_node = _required_direct_dataset(
                    group, x_name, role=f"{group.name}/{x_name}",
                )
                chi_node = _required_direct_dataset(
                    group, y_name, role=f"{group.name}/{y_name}",
                )
                if (
                    intensity_node is None
                    or q_node is None
                    or chi_node is None
                ):
                    return
                q = _read_numeric_vector(
                    q_node, role=f"{group.name}/{x_name}", bundle=bundle,
                )
                chi = _read_numeric_vector(
                    chi_node, role=f"{group.name}/{y_name}", bundle=bundle,
                )
                _qualify_2d_stack(
                    group,
                    intensity_node,
                    q_points=int(q.size),
                    chi_points=int(chi.size),
                    role="intensity",
                )
                sigma_node = _required_direct_dataset(
                    group, "sigma", role=f"{group.name}/sigma",
                )
                if sigma_node is not None:
                    _qualify_2d_stack(
                        group,
                        sigma_node,
                        q_points=int(q.size),
                        chi_points=int(chi.size),
                        role="sigma",
                    )
                g2_modes[mode] = group
                map_2d_modes[mode] = _frame_map(
                    group, self.target_frame, bundle=bundle,
                )
                qu = _dataset_unit(group, x_name)
                cu = _dataset_unit(group, y_name)
                axis_2d_x_modes[mode] = axis_from_unit(qu, q)
                axis_2d_y_modes[mode] = axis_from_unit(cu, chi)
                two_d_kind_modes[mode] = _decode_kind(
                    _bounded_text_attr(
                        group,
                        "two_d_kind",
                        role=f"{group.name} 2-D kind",
                    ),
                    qu,
                    cu,
                )

            for key, group in processed.mode_groups_1d:
                register_1d(key, group)
            for key, group in processed.mode_groups_2d:
                register_2d(key, group)

            axis_1d = axis_1d_modes.get(primary_mode_1d)
            axis_2d_x = axis_2d_x_modes.get(primary_mode_2d)
            axis_2d_y = axis_2d_y_modes.get(primary_mode_2d)
            two_d_kind = two_d_kind_modes.get(
                primary_mode_2d, TwoDKind.Q_CHI,
            )
            bundle.finish()

        # Publish only after the complete local graph and retained-root bundle
        # are validated and committed.
        self._entry = entry
        self._source_base = source_base
        self._g1, self._g2 = g1, g2
        self._geom, self._scan_data, self._frames = geom, scan_data, frames
        self._map_1d, self._map_2d = map_1d, map_2d
        self._map_geom, self._map_scan_data = map_geom, map_scan_data
        self._scan_data_items = scan_data_items
        self._g1_modes, self._g2_modes = g1_modes, g2_modes
        self._map_1d_modes, self._map_2d_modes = (
            map_1d_modes, map_2d_modes,
        )
        self._axis_1d_modes = axis_1d_modes
        self._axis_2d_x_modes = axis_2d_x_modes
        self._axis_2d_y_modes = axis_2d_y_modes
        self._two_d_kind_modes = two_d_kind_modes
        self._primary_mode_1d = primary_mode_1d
        self._primary_mode_2d = primary_mode_2d
        self._multi_result_modes = multi_result_modes
        self._axis_1d, self._axis_2d_x, self._axis_2d_y = (
            axis_1d, axis_2d_x, axis_2d_y,
        )
        self._two_d_kind = two_d_kind
        state = self._snapshot_reader_cache_state()
        if (
            state.phase is not _ReaderCachePhase.OPENING
            or state.pending is not None
            or state.open_token is not self._hdf_owner_token
        ):
            raise RuntimeError("FrameViewReader open publication drifted")
        opened = _reader_cache_state_with(
            state, phase=_ReaderCachePhase.OPEN,
        )
        if not self._transition_reader_cache_state(state, opened):
            raise RuntimeError("FrameViewReader open publication raced")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._clear_open_state()

    def _row(self, mapping: dict[int, int], frame: int) -> int | None:
        return mapping.get(int(frame))

    def labels(self) -> tuple[int, ...]:
        """Frame labels known to this scan without reopening the file."""

        state = self._require_reader_open()
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
        result = tuple(sorted(labels))
        self._revalidate_reader_open(state)
        return result

    @property
    def source_base(self) -> str | None:
        """Host-normalized persisted Project root bound by this admission.

        ``None`` when the record has none, or when its root was written under
        another operating system's path rules and is not a path on this host.
        """

        state = self._require_reader_open()
        value = self._source_base
        self._revalidate_reader_open(state)
        return value

    def _scalar_catalog_inventory(self) -> tuple[int, ...]:
        """Build the complete admitted label inventory while a bundle owns it."""

        labels: set[int] = set(self._map_geom) | set(self._map_scan_data)
        for mode_map in (
            *self._map_1d_modes.values(), *self._map_2d_modes.values(),
        ):
            labels.update(mode_map)
        frames = self._frames
        if frames is not None:
            if len(frames) > _MAX_PERSISTED_FRAME_ROWS:
                raise ValueError(f"{frames.name} inventory exceeds limit")
            for name in frames:
                if type(name) is not str or not name.startswith("frame_"):
                    continue
                suffix = name.removeprefix("frame_")
                try:
                    label = int(suffix)
                except ValueError:
                    continue
                if str(label).zfill(4) != suffix:
                    continue
                labels.add(label)
        if len(labels) > _MAX_PERSISTED_FRAME_ROWS:
            raise ValueError("Frame scalar catalog inventory exceeds limit")
        if any(type(label) is not int or label < 0 for label in labels):
            raise ValueError(
                "Frame scalar catalog labels must be nonnegative integers"
            )
        return tuple(sorted(labels))

    def _scalar_catalog_projection_counts(self) -> tuple[int, int, int]:
        """Return a pre-inventory lower bound and exact per-row fan-out."""

        maps = (
            self._map_geom,
            self._map_scan_data,
            *self._map_1d_modes.values(),
            *self._map_2d_modes.values(),
        )
        counts = [len(mapping) for mapping in maps]
        frames = self._frames
        if frames is not None:
            counts.append(len(frames))
        return (
            max(counts, default=0),
            len(self._scan_data_items),
            sum(len(mapping) for mapping in (
                *self._map_1d_modes.values(),
                *self._map_2d_modes.values(),
            )),
        )

    def _scalar_catalog_scan_columns(
        self,
        bundle: _ReaderArrayBundle,
        checkpoint: Callable[[], None],
    ) -> Mapping[str, np.ndarray]:
        group = self._scan_data
        if group is None:
            return MappingProxyType({})
        retained = self._bundle_scan_data_columns(bundle._owner)
        if retained is not None:
            return retained
        columns: dict[str, np.ndarray] = {}
        total = 0
        for key, item in self._scan_data_items:
            checkpoint()
            column, logical_bytes = _scan_data_column(
                item,
                role=f"{group.name}/{key}",
                allowance_bytes=_MAX_SCAN_DATA_BYTES - total,
                bundle=bundle,
                checkpoint=checkpoint,
            )
            checkpoint()
            total += logical_bytes
            if total > _MAX_SCAN_DATA_BYTES:
                raise AssertionError("scan-data reader exceeded its allowance")
            columns[key] = column
        bundle.pending_scan_data_columns = columns
        return columns

    def _scalar_metadata_for_frame(
        self,
        frame: int,
        columns: Mapping[str, np.ndarray],
    ) -> dict[str, object]:
        row = self._row(self._map_scan_data, frame)
        if row is None:
            return {}
        metadata: dict[str, object] = {}
        for key, column in columns.items():
            if not 0 <= row < int(column.shape[0]):
                raise ValueError(f"scan-data row for frame {frame} is out of range")
            value = column[row]
            scalar = np.asarray(value)
            if scalar.shape != ():
                raise ValueError(f"scan-data value {key!r} is not scalar")
            metadata[key] = _decode(scalar.item())
        return metadata

    def _scalar_catalog_frame_facts(
        self, frame: int,
    ) -> tuple[str | None, int | None, bool, bool]:
        """Read source and thumbnail facts through one qualified frame group."""

        frames = self._frames
        if frames is None:
            return None, None, False, False
        frame_name = f"frame_{frame:04d}"
        group = _required_direct_group(
            frames, frame_name, role=f"{frames.name}/{frame_name}",
        )
        if group is None:
            return None, None, False, False
        source_path, source_index = self._persisted_source_in_group(group)
        thumbnail = _required_direct_dataset(
            group, "thumbnail", role=f"{group.name}/thumbnail",
        )
        if thumbnail is None:
            return source_path, source_index, False, False
        if (
            thumbnail.ndim != 2
            or not all(1 <= int(part) <= 256 for part in thumbnail.shape)
            or thumbnail.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}
            or int(thumbnail.size) * int(thumbnail.dtype.itemsize)
            > 256 * 256 * 2
        ):
            raise ValueError(f"{thumbnail.name} is not a bounded thumbnail")
        mask_baked = _bounded_bool_attr(
            thumbnail,
            "mask_baked",
            role=f"{thumbnail.name} mask marker",
        )
        return (
            source_path, source_index, True,
            True if mask_baked is None else mask_baked,
        )

    def read_scalar_catalog(
        self,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> FrameScalarCatalog:
        """Read one array-free all-frame catalog with atomic cache effects."""

        self._require_reader_open()
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        with _ReaderArrayBundle(self) as bundle:

            def checkpoint() -> None:
                if cancelled is None:
                    return
                decision = bundle._callback(cancelled)
                if type(decision) is not bool:
                    raise TypeError("cancelled must return an exact boolean")
                if decision:
                    raise InterruptedError("Frame scalar catalog read cancelled")

            checkpoint()
            if self.target_frame is not None:
                raise ValueError(
                    "Frame scalar catalog requires a full-inventory reader"
                )
            average_frame_one = bundle._callback(
                _is_current_average_frame_one, self._entry,
            )
            lower_bound, column_count, mode_memberships = bundle._callback(
                self._scalar_catalog_projection_counts,
            )
            axis_descriptor_count = len(self._axis_1d_modes)
            _validate_scalar_catalog_projection(
                label_count=lower_bound,
                metadata_column_count=column_count,
                mode_membership_count=mode_memberships,
                axis_descriptor_count=axis_descriptor_count,
            )
            labels = bundle._callback(self._scalar_catalog_inventory)
            _validate_scalar_catalog_projection(
                label_count=len(labels),
                metadata_column_count=column_count,
                mode_membership_count=mode_memberships,
                axis_descriptor_count=axis_descriptor_count,
            )
            axes_1d = tuple(
                (mode, axis.label, axis.unit, axis.log)
                for mode, axis in self._axis_1d_modes.items()
            )
            columns = self._scalar_catalog_scan_columns(bundle, checkpoint)
            rows: list[FrameScalarRow] = []
            for label in labels:
                checkpoint()
                metadata = self._scalar_metadata_for_frame(label, columns)
                geometry = bundle._callback(
                    self._scalar_catalog_geometry_for_frame, label,
                )
                frame_facts = bundle._callback(
                    self._scalar_catalog_frame_facts, label,
                )
                source_path, source_index, has_thumbnail, mask_baked = frame_facts
                modes_1d = tuple(
                    mode
                    for mode, mode_map in self._map_1d_modes.items()
                    if label in mode_map
                )
                modes_2d = tuple(
                    mode
                    for mode, mode_map in self._map_2d_modes.items()
                    if label in mode_map
                )
                active_1d = (
                    self._primary_mode_1d
                    if self._primary_mode_1d in modes_1d else None
                )
                active_2d = (
                    self._primary_mode_2d
                    if self._primary_mode_2d in modes_2d else None
                )
                kinds = tuple(
                    (mode, self._two_d_kind_modes[mode])
                    for mode in modes_2d
                )
                rows.append(bundle._callback(
                    FrameScalarRow,
                    label=label,
                    metadata_raw=metadata,
                    geometry=geometry,
                    source_path=source_path,
                    source_frame_index=source_index,
                    has_thumbnail=has_thumbnail,
                    mask_baked=mask_baked,
                    averaged=(
                        average_frame_one
                        and label == 1
                        and source_path is None
                    ),
                    modes_1d=modes_1d,
                    modes_2d=modes_2d,
                    active_mode_1d=active_1d,
                    active_mode_2d=active_2d,
                    two_d_kinds=kinds,
                ))
                checkpoint()
            checkpoint()
            catalog = bundle._callback(
                FrameScalarCatalog,
                artifact_path=str(self.path),
                entry=self.entry_name,
                rows=tuple(rows),
                axes_1d=axes_1d,
                source_base=self._source_base,
            )
            checkpoint()
            bundle.finish()
            return catalog

    def _preflight_1d_rows(
        self,
        labels: tuple[int, ...],
        bundle: _ReaderArrayBundle,
        checkpoint: Callable[[], None],
    ) -> tuple[_Frame1DReadModePlan, ...]:
        """Freeze an exact bounded row route without reading row payloads."""

        _validate_1d_result_projection(
            label_count=len(labels),
            membership_count=0,
            root_count=0,
            projected_bytes=0,
        )
        axis_roots: list[object] = []
        axis_bytes = 0
        row_roots: dict[tuple[int, int], int] = {}
        row_bytes_total = 0
        membership_count = 0
        plans: list[_Frame1DReadModePlan] = []

        for mode, group in self._g1_modes.items():
            checkpoint()
            axis = self._axis_1d_modes[mode]
            if axis.values is None:
                raise ValueError(f"1-D mode {mode!r} has no sampled axis")
            axis_fact = bundle._callback(physical_root_fact, axis.values)
            if not any(axis_fact.root is root for root in axis_roots):
                axis_roots.append(axis_fact.root)
                axis_bytes += axis_fact.nbytes

            points = int(axis.values.size)
            def qualify_mode_nodes() -> tuple[
                h5py.Dataset, h5py.Dataset | None, int,
            ]:
                intensity = _qualify_1d_stack(
                    group,
                    _required_direct_dataset(
                        group,
                        "intensity",
                        role=f"{group.name}/intensity",
                    ),
                    points=points,
                    role="intensity",
                )
                sigma = _required_direct_dataset(
                    group, "sigma", role=f"{group.name}/sigma",
                )
                if sigma is not None:
                    sigma = _qualify_1d_stack(
                        group, sigma, points=points, role="sigma",
                    )
                return intensity, sigma, int(intensity.shape[0])

            intensity_node, sigma_node, row_count = bundle._callback(
                qualify_mode_nodes,
            )

            datasets = (intensity_node,) + (
                () if sigma_node is None else (sigma_node,)
            )
            def dataset_addresses() -> tuple[int, ...]:
                addresses: list[int] = []
                for dataset in datasets:
                    try:
                        address = int(h5py.h5o.get_info(dataset.id).addr)
                    except (
                        OSError, RuntimeError, TypeError, ValueError,
                    ) as error:
                        raise ValueError(
                            f"{dataset.name} object identity is unavailable"
                        ) from error
                    if address < 0:
                        raise ValueError(
                            f"{dataset.name} object identity is invalid"
                        )
                    addresses.append(address)
                return tuple(addresses)

            addresses = bundle._callback(dataset_addresses)
            mode_rows: list[tuple[int, int]] = []
            mode_map = self._map_1d_modes[mode]
            for label in labels:
                checkpoint()
                row = mode_map.get(label)
                if row is None:
                    continue
                if type(row) is not int or row < 0 or row >= row_count:
                    raise ValueError(f"1-D mode {mode!r} row route is invalid")
                membership_count += 1
                if membership_count > _MAX_1D_RESULT_MEMBERSHIPS:
                    _validate_1d_result_projection(
                        label_count=len(labels),
                        membership_count=membership_count,
                        root_count=len(axis_roots) + len(row_roots),
                        projected_bytes=axis_bytes + row_bytes_total,
                    )
                mode_rows.append((label, row))
                for address in addresses:
                    key = (address, row)
                    prior_bytes = row_roots.get(key)
                    row_bytes = points * int(np.dtype(np.float64).itemsize)
                    if prior_bytes is None:
                        row_roots[key] = row_bytes
                        row_bytes_total += row_bytes
                    elif prior_bytes != row_bytes:
                        raise ValueError("1-D aliased row projection is inconsistent")
                    _validate_1d_result_projection(
                        label_count=len(labels),
                        membership_count=membership_count,
                        root_count=len(axis_roots) + len(row_roots),
                        projected_bytes=axis_bytes + row_bytes_total,
                    )
            plans.append(bundle._callback(
                _Frame1DReadModePlan,
                mode,
                group,
                axis,
                tuple(mode_rows),
                sigma_node is not None,
            ))
            checkpoint()

        _validate_1d_result_projection(
            label_count=len(labels),
            membership_count=membership_count,
            root_count=len(axis_roots) + len(row_roots),
            projected_bytes=axis_bytes + row_bytes_total,
        )
        checkpoint()
        return tuple(plans)

    def read_1d_rows(
        self,
        labels: tuple[int, ...],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Frame1DRows:
        """Read requested persisted 1-D rows without broad frame hydration.

        One atomic reader bundle owns every row read.  The method deliberately
        does not project metadata, geometry, source locators, thumbnails, or
        2-D results; its returned arrays are transferred as immutable caller
        payload only after the bundle settles.
        """

        self._require_reader_open()
        if type(labels) is not tuple or not labels:
            raise TypeError("labels must be a nonempty exact tuple")
        if len(labels) > _MAX_1D_RESULT_LABELS:
            raise ValueError("1-D row selection exceeds limit")
        selected: list[int] = []
        previous = -1
        for label in labels:
            if type(label) is not int or label < 0:
                raise TypeError("labels must contain exact nonnegative integers")
            if label <= previous:
                raise ValueError("labels must be strictly increasing")
            selected.append(label)
            previous = label
        selected_labels = tuple(selected)
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if self.target_frame is not None:
            raise ValueError("1-D rows require a full-inventory reader")

        with _ReaderArrayBundle(self) as bundle:

            def checkpoint() -> None:
                if cancelled is None:
                    return
                decision = bundle._callback(cancelled)
                if type(decision) is not bool:
                    raise TypeError("cancelled must return an exact boolean")
                if decision:
                    raise InterruptedError("Frame 1-D row read cancelled")

            checkpoint()
            mode_plans = self._preflight_1d_rows(
                selected_labels, bundle, checkpoint,
            )
            checkpoint()
            mode_results: list[Frame1DModeRows] = []
            for plan in mode_plans:
                checkpoint()
                mode = plan.mode
                group = plan.group
                axis = plan.axis
                points = int(axis.values.size)
                mode_labels: list[int] = []
                intensities: list[np.ndarray] = []
                sigmas: list[np.ndarray] | None = (
                    [] if plan.has_sigma else None
                )
                for label, row in plan.rows:
                    checkpoint()
                    intensity = _read_1d_row(
                        group,
                        "intensity",
                        row,
                        points=points,
                        bundle=bundle,
                    )
                    checkpoint()
                    sigma = None
                    if plan.has_sigma:
                        sigma = _read_1d_row(
                            group,
                            "sigma",
                            row,
                            points=points,
                            bundle=bundle,
                        )
                        checkpoint()
                    mode_labels.append(label)
                    intensities.append(intensity)
                    if sigmas is not None:
                        if sigma is None:
                            raise AssertionError("qualified sigma row is unavailable")
                        sigmas.append(sigma)
                mode_results.append(bundle._callback(
                    Frame1DModeRows,
                    mode=mode,
                    axis=axis,
                    labels=tuple(mode_labels),
                    intensity_rows=tuple(intensities),
                    sigma_rows=None if sigmas is None else tuple(sigmas),
                ))
                checkpoint()

            present_modes = tuple(result.mode for result in mode_results)
            primary_mode = (
                self._primary_mode_1d
                if self._primary_mode_1d in present_modes else None
            )
            result = bundle._callback(
                Frame1DRows,
                artifact_path=str(self.path),
                entry=self.entry_name,
                labels=selected_labels,
                modes=tuple(mode_results),
                primary_mode=primary_mode,
            )
            checkpoint()
            bundle.finish()
            return result

    def has_frame(self, frame: int) -> bool:
        """Whether one exact frame has any persisted processed record."""
        state = self._require_reader_open()
        frame = int(frame)
        maps = (self._map_1d, self._map_2d, self._map_geom, self._map_scan_data,
                *self._map_1d_modes.values(), *self._map_2d_modes.values())
        if any(frame in mapping for mapping in maps):
            self._revalidate_reader_open(state)
            return True
        record = (
            None
            if self._frames is None
            else _direct_group(self._frames, f"frame_{frame:04d}")
        )
        result = bool(record is not None and len(record))
        self._revalidate_reader_open(state)
        return result

    def _metadata_for_frame(
        self, frame: int, *, bundle: _ReaderArrayBundle,
    ) -> dict[str, object]:
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
        retained_cols = self._bundle_scan_data_columns(bundle._owner)
        cols = (
            retained_cols
            if retained_cols is not None
            else bundle.pending_scan_data_columns
        )
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
                    bundle=bundle,
                )
                total += logical_bytes
                if total > _MAX_SCAN_DATA_BYTES:  # defensive contract check
                    raise AssertionError("scan-data reader exceeded its allowance")
                cols[str(key)] = column
            bundle.pending_scan_data_columns = cols
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

    def _scalar_catalog_geometry_for_frame(
        self, frame: int,
    ) -> FrameGeometry | None:
        geometry = self._geometry_for_frame(frame)
        if geometry is None:
            return None

        def finite_or_none(value: float | None) -> float | None:
            return value if value is not None and math.isfinite(value) else None

        return FrameGeometry(
            rot1=finite_or_none(geometry.rot1),
            rot2=finite_or_none(geometry.rot2),
            rot3=finite_or_none(geometry.rot3),
            incident_angle=finite_or_none(geometry.incident_angle),
        )

    def _thumbnail_for_frame(
        self,
        frame: int,
        *,
        include_data: bool = True,
        bundle: _ReaderArrayBundle,
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
            _read_thumbnail(fg, thumbnail, bundle=bundle)
            if include_data else None,
            True if mask_baked is None else mask_baked,
        )

    def _persisted_source_for_frame(
        self, frame: int,
    ) -> tuple[str | None, int | None]:
        """Read the exact persisted locator without mutable path resolution."""

        frames = self._frames
        if frames is None:
            return None, None
        frame_name = f"frame_{int(frame):04d}"
        fg = _required_direct_group(
            frames, frame_name, role=f"{frames.name}/{frame_name}",
        )
        if fg is None:
            return None, None
        return self._persisted_source_in_group(fg)

    @staticmethod
    def _persisted_source_in_group(
        fg: h5py.Group,
    ) -> tuple[str | None, int | None]:
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
            path = _bounded_utf8_scalar(
                path_node,
                role=f"{src.name}/path",
                max_bytes=_MAX_SOURCE_PATH_BYTES,
            )
        source_index_node = _required_direct_dataset(
            src, "frame_index", role=f"{src.name}/frame_index",
        )
        if source_index_node is not None:
            source_idx = _bounded_integral_scalar(
                source_index_node,
                role=f"{src.name}/frame_index",
            )
        return path, source_idx

    def _source_for_frame(self, frame: int) -> tuple[str | None, int | None]:
        stored, source_idx = self._persisted_source_for_frame(frame)
        if stored is None or not self.resolve_source:
            return stored, source_idx
        # Resolution remains selected-frame hydration policy.  Scalar catalogs
        # deliberately use _persisted_source_for_frame and never consult the
        # mutable filesystem.
        from xrd_tools.io.read import resolve_source_master
        resolved = resolve_source_master(
            stored,
            scan_file=self.path,
            source_base=self._source_base,
            source_root=self.source_root,
        )
        return str(resolved) if resolved is not None else stored, source_idx

    def _common_fields(
        self,
        frame: int,
        *,
        include_heavy: bool = True,
        bundle: _ReaderArrayBundle,
    ) -> dict:
        """Shared per-frame fields (thumbnail/source/metadata/geometry) — the
        same for every mode of one frame."""
        thumbnail, mask_baked = self._thumbnail_for_frame(
            frame, include_data=include_heavy, bundle=bundle,
        )
        source_path, source_frame_index = self._source_for_frame(frame)
        metadata_raw = self._metadata_for_frame(frame, bundle=bundle)
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
        self._require_reader_open()
        frame = int(frame)
        with _ReaderArrayBundle(self) as bundle:
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
                    g1,
                    "intensity",
                    row_1d,
                    points=points,
                    bundle=bundle,
                )
                if _required_direct_dataset(
                    g1, "sigma", role=f"{g1.name}/sigma",
                ) is not None:
                    sigma_1d = _read_1d_row(
                        g1,
                        "sigma",
                        row_1d,
                        points=points,
                        bundle=bundle,
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
                    bundle=bundle,
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
                        bundle=bundle,
                    )

            view = FrameView(
                label=frame,
                axis_1d=(
                    self._axis_1d_modes.get(m1)
                    if intensity_1d is not None else None
                ),
                intensity_1d=intensity_1d,
                sigma_1d=sigma_1d,
                axis_2d_x=(
                    self._axis_2d_x_modes.get(m2)
                    if intensity_2d is not None else None
                ),
                axis_2d_y=(
                    self._axis_2d_y_modes.get(m2)
                    if intensity_2d is not None else None
                ),
                intensity_2d=intensity_2d,
                sigma_2d=sigma_2d,
                two_d_kind=self._two_d_kind_modes.get(m2, TwoDKind.Q_CHI),
                **self._common_fields(frame, bundle=bundle),
            )
            bundle.finish()
            return view

    def _view_for(
        self,
        frame: int,
        dim: str,
        mode: str,
        *,
        include_heavy: bool = True,
        common: dict | None = None,
        bundle: _ReaderArrayBundle,
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
                _read_1d_row(
                    g, "sigma", row, points=points, bundle=bundle,
                )
                if sigma_node is not None else None
            )
            return FrameView(
                label=frame, axis_1d=self._axis_1d_modes.get(mode),
                intensity_1d=_read_1d_row(
                    g, "intensity", row, points=points, bundle=bundle,
                ), sigma_1d=sig,
                **(
                    common
                    if common is not None
                    else self._common_fields(frame, bundle=bundle)
                ),
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
                bundle=bundle,
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
                    bundle=bundle,
                )
                if include_heavy else None
            ),
            sigma_2d=sig,
            two_d_kind=self._two_d_kind_modes.get(mode, TwoDKind.Q_CHI),
            **(
                common
                if common is not None
                else self._common_fields(
                    frame, include_heavy=include_heavy, bundle=bundle,
                )
            ),
        )

    def modes_1d(self) -> tuple:
        """GI 1D mode_keys present (primary first)."""
        state = self._require_reader_open()
        result = tuple(self._g1_modes)
        self._revalidate_reader_open(state)
        return result

    def modes_2d(self) -> tuple:
        """GI 2D mode_keys present (primary first)."""
        state = self._require_reader_open()
        result = tuple(self._g2_modes)
        self._revalidate_reader_open(state)
        return result

    def primary_mode_1d(self) -> str:
        state = self._require_reader_open()
        result = self._primary_mode_1d
        self._revalidate_reader_open(state)
        return result

    def primary_mode_2d(self) -> str:
        state = self._require_reader_open()
        result = self._primary_mode_2d
        self._revalidate_reader_open(state)
        return result

    def is_multi_mode(self) -> bool:
        """True if the file carries the per-mode capability marker."""
        state = self._require_reader_open()
        result = self._multi_result_modes
        self._revalidate_reader_open(state)
        return result

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
        self._require_reader_open()
        frame = int(frame)
        with _ReaderArrayBundle(self) as bundle:
            common = self._common_fields(
                frame, include_heavy=include_heavy, bundle=bundle,
            )
            r1d: dict = {}
            r2d: dict = {}
            for mode in tuple(self._g1_modes):
                view = self._view_for(
                    frame, "1d", mode, common=common, bundle=bundle,
                )
                if view is not None and view.has_1d:
                    r1d[mode] = view
            for mode in tuple(self._g2_modes):
                view = self._view_for(
                    frame,
                    "2d",
                    mode,
                    include_heavy=include_heavy,
                    common=common,
                    bundle=bundle,
                )
                if view is not None and (
                    include_heavy is False or view.has_2d
                ):
                    r2d[mode] = view
            a1 = (
                self._primary_mode_1d
                if self._primary_mode_1d in r1d
                else next(iter(r1d), DEFAULT_MODE_KEY)
            )
            a2 = (
                self._primary_mode_2d
                if self._primary_mode_2d in r2d
                else next(iter(r2d), DEFAULT_MODE_KEY)
            )
            record = FrameRecord(
                label=frame,
                results_1d=r1d,
                results_2d=r2d,
                active_mode_1d=a1,
                active_mode_2d=a2,
            )
            bundle.finish()
            return record

    # Reader reservations are deliberately smaller than the historical
    # exchange journal: the authority owns the gate; this state holds only the
    # live bundle and, during close, the actual leases still requiring release.
    def _pending_bundle(
        self, owner: _ReaderBundleOwner | None,
    ) -> tuple[_ReaderCacheState, _ReaderPendingBundle]:
        state = self._snapshot_reader_cache_state()
        pending = state.pending
        if state.phase is not _ReaderCachePhase.BUILDING or pending is None:
            raise RuntimeError("FrameView reader cache is not building")
        current = pending.owner_ref()
        if owner is not None and current is not owner:
            raise RuntimeError("FrameView read-bundle owner is stale")
        if owner is None and current is not None:
            raise RuntimeError("FrameView read-bundle owner is busy")
        if state.open_token is not self._hdf_owner_token:
            raise RuntimeError("FrameView read bundle lost its HDF owner")
        return state, pending

    def _assert_bundle_owned(
        self, owner: _ReaderBundleOwner,
    ) -> _ReaderCacheState:
        return self._pending_bundle(owner)[0]

    def _bundle_callback_stamp(
        self, owner: _ReaderBundleOwner,
    ) -> _ReaderCacheState:
        return self._assert_bundle_owned(owner)

    def _revalidate_bundle_callback(
        self, stamp: _ReaderCacheState, owner: _ReaderBundleOwner,
    ) -> None:
        if self._assert_bundle_owned(owner) is not stamp:
            raise RuntimeError("FrameView read-bundle callback drifted")

    def _begin_array_bundle(
        self, owner: _ReaderBundleOwner, *, opening: bool = False,
    ) -> PhysicalRootReservation:
        state = self._snapshot_reader_cache_state()
        expected = _ReaderCachePhase.OPENING if opening else _ReaderCachePhase.OPEN
        if (
            state.phase is not expected
            or state.pending is not None
            or state.open_token is not self._hdf_owner_token
        ):
            raise RuntimeError("FrameView reader cache publication is busy")
        reservation = self._memory_authority.reserve()
        replacement = _reader_cache_state_with(
            state,
            phase=_ReaderCachePhase.BUILDING,
            pending=_ReaderPendingBundle(ref(owner), reservation, opening),
            replace_pending=True,
        )
        if not self._transition_reader_cache_state(state, replacement):
            reservation.rollback()
            raise RuntimeError("FrameView reader cache admission raced")
        return reservation

    def _bundle_cached(
        self, owner: _ReaderBundleOwner, key: tuple[object, ...],
    ) -> np.ndarray | None:
        state = self._assert_bundle_owned(owner)
        entry = state.entries.get(key)
        if entry is None:
            return None
        self._validate_reader_cache_entry(entry)
        return entry.array

    def _bundle_scan_data_columns(
        self, owner: _ReaderBundleOwner,
    ) -> Mapping[str, np.ndarray] | None:
        return self._assert_bundle_owned(owner).scan_data_columns

    def _bundle_reservation_call(
        self, owner: _ReaderBundleOwner, method: str, *args: object,
    ) -> object:
        stamp, pending = self._pending_bundle(owner)
        result = getattr(pending.reservation, method)(*args)
        self._revalidate_bundle_callback(stamp, owner)
        return result

    def _bundle_reservation_claim(
        self, owner: _ReaderBundleOwner, nbytes: int,
    ) -> object:
        return self._bundle_reservation_call(owner, "claim", nbytes)

    def _bundle_reservation_bind(
        self, owner: _ReaderBundleOwner, token: object, value: object,
        semantic: object,
    ) -> object:
        return self._bundle_reservation_call(
            owner, "bind", token, value, semantic,
        )

    def _bundle_reservation_reserve(
        self, owner: _ReaderBundleOwner, value: object, semantic: object,
    ) -> object:
        return self._bundle_reservation_call(owner, "reserve", value, semantic)

    def _settle_bundle(
        self, owner: _ReaderBundleOwner | None, *, accept: bool,
        plan: tuple[_ReaderPendingArray, ...] = (),
        pending_scan_data_columns: Mapping[str, np.ndarray] | None = None,
    ) -> str:
        state, pending = self._pending_bundle(owner)
        reservation = pending.reservation
        if not accept:
            reservation.rollback()
            restored = _reader_cache_state_with(
                state,
                phase=_ReaderCachePhase.OPENING if pending.opening else _ReaderCachePhase.OPEN,
                pending=None,
                replace_pending=True,
            )
            if not self._transition_reader_cache_state(state, restored):
                raise RuntimeError("FrameView reader cache rollback raced")
            return "rolled-back"
        retained: dict[tuple[object, ...], _ReaderCacheEntry] = dict(state.entries)
        for item in plan:
            if type(item) is not _ReaderPendingArray:
                raise TypeError("FrameView pending array is untrusted")
            if item.retain:
                if item.key in retained:
                    raise RuntimeError("FrameView reader-cache key was already retained")
                item.value.setflags(write=False)
        leases = reservation.commit()
        transient: list[PhysicalRootLease] = []
        try:
            for item in plan:
                lease = leases[item.semantic]
                if (
                    type(lease) is not PhysicalRootLease
                    or lease.released
                    or lease._authority is not self._memory_authority
                ):
                    raise RuntimeError("FrameView reservation lease is invalid")
                if item.retain:
                    retained[item.key] = _ReaderCacheEntry(item.value, lease)
                else:
                    transient.append(lease)
            scan_columns = (
                state.scan_data_columns
                if pending_scan_data_columns is None
                else MappingProxyType(dict(pending_scan_data_columns))
            )
            if scan_columns is not None:
                values = tuple(entry.array for entry in retained.values())
                if any(not any(value is cached for cached in values)
                       for value in scan_columns.values()):
                    raise RuntimeError("FrameView scan-data column is not cache-owned")
            for lease in transient:
                lease.release()
        except BaseException:
            reservation.rollback()
            raise
        accepted = _reader_cache_state_with(
            state,
            phase=_ReaderCachePhase.OPENING if pending.opening else _ReaderCachePhase.OPEN,
            entries=MappingProxyType(retained),
            scan_data_columns=scan_columns,
            replace_scan_data_columns=True,
            pending=None,
            replace_pending=True,
        )
        if not self._transition_reader_cache_state(state, accepted):
            raise RuntimeError("FrameView reader cache publication raced")
        return "accepted"

    def _finish_array_bundle(
        self, owner: _ReaderBundleOwner, plan: tuple[_ReaderPendingArray, ...],
        pending_scan_data_columns: Mapping[str, np.ndarray] | None,
    ) -> str:
        return self._settle_bundle(
            owner, accept=True, plan=plan,
            pending_scan_data_columns=pending_scan_data_columns,
        )

    def _rollback_array_bundle(self, owner: _ReaderBundleOwner | None) -> str:
        return self._settle_bundle(owner, accept=False)

    def _recover_array_bundle(self, owner: _ReaderBundleOwner | None) -> str:
        return self._rollback_array_bundle(owner)

    def _require_reader_open(self) -> _ReaderCacheState:
        state = self._snapshot_reader_cache_state()
        if state.phase is _ReaderCachePhase.BUILDING and state.pending is not None:
            if state.pending.owner_ref() is not None:
                raise RuntimeError("FrameView reader cache publication is busy")
            self._rollback_array_bundle(None)
            state = self._snapshot_reader_cache_state()
        if (
            state.phase is not _ReaderCachePhase.OPEN
            or state.pending is not None
            or state.open_token is not self._hdf_owner_token
            or self._h5 is None
            or not bool(self._h5.id.valid)
        ):
            raise RuntimeError("FrameViewReader is not exactly open")
        return state

    def _revalidate_reader_open(self, state: _ReaderCacheState) -> None:
        if self._snapshot_reader_cache_state() is not state:
            raise RuntimeError("FrameViewReader open state drifted")

    def _clear_reader_fields(self) -> None:
        self._h5 = None
        self._hdf_owner_token = None
        self._detach_reader_science_fields()

    def _detach_reader_science_fields(self) -> None:
        self._entry = self._g1 = self._g2 = None
        self._geom = self._scan_data = self._frames = None
        self._source_base = None
        self._map_1d = self._map_2d = self._map_geom = self._map_scan_data = {}
        self._axis_1d = self._axis_2d_x = self._axis_2d_y = None
        self._two_d_kind = TwoDKind.Q_CHI
        self._g1_modes = self._g2_modes = {}
        self._map_1d_modes = self._map_2d_modes = {}
        self._axis_1d_modes = self._axis_2d_x_modes = self._axis_2d_y_modes = {}
        self._two_d_kind_modes = {}
        self._scan_data_items = ()

    def _clear_open_state(self) -> None:
        state = self._snapshot_reader_cache_state()
        if state.phase is _ReaderCachePhase.BUILDING:
            assert state.pending is not None
            if state.pending.owner_ref() is not None:
                raise RuntimeError("FrameView reader cache publication is busy")
            self._rollback_array_bundle(None)
            state = self._snapshot_reader_cache_state()
        if state.phase in {_ReaderCachePhase.OPEN, _ReaderCachePhase.OPENING}:
            leases = tuple(entry.lease for entry in state.entries.values())
            closing = _reader_cache_state_with(
                state,
                phase=_ReaderCachePhase.CLOSING_LEASES,
                entries=MappingProxyType({}),
                scan_data_columns=None,
                replace_scan_data_columns=True,
                close_leases=leases,
                replace_close_leases=True,
                close_h5=self._h5,
                replace_close_h5=True,
            )
            if not self._transition_reader_cache_state(state, closing):
                raise RuntimeError("FrameView reader close admission raced")
            self._detach_reader_science_fields()
            state = closing
        if state.phase is _ReaderCachePhase.CLOSED:
            return
        if state.phase is _ReaderCachePhase.CLOSING_LEASES:
            remaining = list(state.close_leases)
            while remaining:
                lease = remaining.pop(0)
                try:
                    lease.release()
                except BaseException:
                    if not lease.released:
                        held = _reader_cache_state_with(
                            state,
                            close_leases=tuple([lease, *remaining]),
                            replace_close_leases=True,
                        )
                        self._transition_reader_cache_state(state, held)
                        raise
                released = _reader_cache_state_with(
                    state,
                    close_leases=tuple(remaining),
                    replace_close_leases=True,
                )
                if not self._transition_reader_cache_state(
                    state, released,
                ):
                    raise RuntimeError("FrameView reader close lease release raced")
                state = released
            authority_phase = _reader_cache_state_with(
                state, phase=_ReaderCachePhase.CLOSE_AUTHORITY,
            )
            if not self._transition_reader_cache_state(state, authority_phase):
                raise RuntimeError("FrameView reader authority close raced")
            state = authority_phase
        if state.phase is _ReaderCachePhase.CLOSE_AUTHORITY:
            self._memory_authority.close()
            hdf_phase = _reader_cache_state_with(
                state, phase=_ReaderCachePhase.CLOSE_HDF,
            )
            if not self._transition_reader_cache_state(state, hdf_phase):
                raise RuntimeError("FrameView reader HDF close admission raced")
            state = hdf_phase
        if state.phase is not _ReaderCachePhase.CLOSE_HDF:
            raise RuntimeError("FrameView reader close state is invalid")
        handle = state.close_h5
        if handle is not None and bool(handle.id.valid):
            handle.close()
        closed = _reader_cache_state_with(
            state, phase=_ReaderCachePhase.CLOSED,
            open_token=None, replace_open_token=True,
            entries=MappingProxyType({}), scan_data_columns=None,
            replace_scan_data_columns=True, close_h5=None, replace_close_h5=True,
        )
        if not self._transition_reader_cache_state(state, closed):
            raise RuntimeError("FrameView reader final close raced")
        self._clear_reader_fields()

    def __enter__(self) -> "FrameViewReader":
        state = self._snapshot_reader_cache_state()
        if state.phase is not _ReaderCachePhase.CLOSED or state.pending is not None:
            raise RuntimeError("FrameViewReader cache is not closed")
        authority = PhysicalRootAuthority(_MAX_READER_RETAINED_BYTES)
        token = object()
        opening = _reader_cache_state_with(
            state, phase=_ReaderCachePhase.OPENING, open_token=token,
            replace_open_token=True, entries=MappingProxyType({}),
            scan_data_columns=None, replace_scan_data_columns=True,
        )
        if not self._transition_reader_cache_state(state, opening):
            authority.close()
            raise RuntimeError("FrameViewReader open admission raced")
        self._memory_authority = authority
        self._hdf_owner_token = token
        try:
            self._h5 = h5py.File(self.path, "r")
            return self._enter_inner()
        except BaseException as error:
            try:
                self._clear_open_state()
            except BaseException as cleanup_error:
                raise cleanup_error from error
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self._clear_open_state()


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
