"""Small, lazy NeXus/HDF5 inspection helpers.

These functions are intentionally read-only and GUI-agnostic.  They expose
enough structure for a thin xdart NeXus viewer without turning xdart into an
HDF5 browser or loading large detector/reduction arrays into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from ssrl_xrd_tools.core import TwoDKind, two_d_kind_from_units


def _readonly(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value[()])
        return [_decode(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(v) for v in value)
    return value


def _attrs(obj: h5py.Group | h5py.Dataset) -> Mapping[str, Any]:
    return _readonly({str(key): _decode(value) for key, value in obj.attrs.items()})


def _node_name(path: str) -> str:
    if path == "/":
        return "/"
    return path.rstrip("/").split("/")[-1]


@dataclass(frozen=True, slots=True)
class NexusNodeSummary:
    """A lazy summary of one HDF5 group, dataset, or unresolved link."""

    path: str
    kind: str
    name: str = ""
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    size: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    nx_class: str | None = None
    children: tuple["NexusNodeSummary", ...] = ()
    child_count: int = 0
    truncated: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", _readonly(self.attrs))
        object.__setattr__(self, "children", tuple(self.children))


@dataclass(frozen=True, slots=True)
class NexusAxisSummary:
    """Axis dataset metadata for an NXdata-style group."""

    name: str
    path: str
    shape: tuple[int, ...]
    units: str | None = None


@dataclass(frozen=True, slots=True)
class NexusReducedSummary:
    """Domain summary for an integrated 1D or 2D stack."""

    path: str
    frame_count: int
    frame_labels: tuple[int, ...]
    intensity_shape: tuple[int, ...] | None
    axes: tuple[NexusAxisSummary, ...]
    two_d_kind: TwoDKind | None = None


@dataclass(frozen=True, slots=True)
class NexusXDartSummary:
    """Schema-aware summary for processed xdart/ssrl scan files."""

    entry: str
    is_processed: bool
    integrated_1d: NexusReducedSummary | None = None
    integrated_2d: NexusReducedSummary | None = None
    frame_labels: tuple[int, ...] = ()
    scan_data_columns: tuple[str, ...] = ()
    geometry_columns: tuple[str, ...] = ()
    thumbnail_count: int = 0
    source_count: int = 0
    raw_image_dataset: str | None = None
    raw_image_shape: tuple[int, ...] | None = None
    raw_image_dtype: str | None = None


@dataclass(frozen=True, slots=True)
class NexusFileSummary:
    """Top-level lazy inspection result."""

    path: str
    entries: tuple[str, ...]
    tree: NexusNodeSummary
    xdart: NexusXDartSummary | None = None


@dataclass(frozen=True, slots=True)
class NexusDatasetPreview:
    """Small head-slice preview of a dataset."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    selection: str
    data: Any
    truncated: bool
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", _readonly(self.attrs))


@dataclass(frozen=True, slots=True)
class NexusDatasetData:
    """Dataset data loaded explicitly by a headless caller.

    ``read_nexus_dataset(..., selection=None)`` reads the full dataset.  The
    GUI should keep using :func:`preview_nexus_dataset` or pass an explicit
    bounded selection; notebooks can opt into the full read when that is the
    requested operation.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    selection: str
    data: Any
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", _readonly(self.attrs))


def inspect_nexus(
    path: str | Path,
    *,
    entry: str | None = "entry",
    max_depth: int = 4,
    max_children: int = 200,
) -> NexusFileSummary:
    """Return a lazy tree plus an xdart-schema summary when available."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NeXus/HDF5 file not found: {p}")

    with h5py.File(p, "r") as h5:
        entries = tuple(
            key for key, obj in h5.items()
            if isinstance(obj, h5py.Group)
            and _decode(obj.attrs.get("NX_class")) == "NXentry"
        )
        if not entries:
            entries = tuple(key for key, obj in h5.items() if isinstance(obj, h5py.Group))

        tree = _summarize_node(h5, "/", max_depth=max_depth, max_children=max_children)

        selected_entry = _select_entry(h5, entries, entry)
        xdart = (
            _summarize_xdart_entry(h5, selected_entry)
            if selected_entry is not None
            else None
        )

    return NexusFileSummary(path=str(p), entries=entries, tree=tree, xdart=xdart)


def preview_nexus_dataset(
    path: str | Path,
    dataset_path: str,
    *,
    max_items: int = 64,
) -> NexusDatasetPreview:
    """Read a small, bounded preview from one dataset."""

    p = Path(path)
    with h5py.File(p, "r") as h5:
        internal = dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"
        if internal not in h5:
            raise KeyError(f"Dataset {internal!r} not found in {p}")
        ds = h5[internal]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{internal!r} is not a dataset")

        selection = _preview_selection(ds.shape, max_items=max_items)
        data = _decode(ds[selection])
        text = _selection_text(selection, ds.shape)
        truncated = _selection_truncated(selection, ds.shape)
        return NexusDatasetPreview(
            path=internal,
            shape=tuple(int(v) for v in ds.shape),
            dtype=str(ds.dtype),
            selection=text,
            data=data,
            truncated=truncated,
            attrs=_attrs(ds),
        )


def read_nexus_dataset(
    path: str | Path,
    dataset_path: str,
    *,
    selection: Any = None,
) -> NexusDatasetData:
    """Read a NeXus/HDF5 dataset, optionally with an explicit selection.

    This is the headless "proper dataset" companion to the GUI preview
    helper.  Omitting ``selection`` reads the full dataset; pass NumPy-style
    slices/indices to bound the read.
    """

    p = Path(path)
    with h5py.File(p, "r") as h5:
        internal = dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"
        if internal not in h5:
            raise KeyError(f"Dataset {internal!r} not found in {p}")
        ds = h5[internal]
        if not isinstance(ds, h5py.Dataset):
            raise TypeError(f"{internal!r} is not a dataset")

        if selection is None:
            selection = _full_selection(ds.shape)
        data = ds[selection]
        return NexusDatasetData(
            path=internal,
            shape=tuple(int(v) for v in ds.shape),
            dtype=str(ds.dtype),
            selection=_selection_text(selection, ds.shape),
            data=data,
            attrs=_attrs(ds),
        )


def _select_entry(
    h5: h5py.File,
    entries: tuple[str, ...],
    requested: str | None,
) -> str | None:
    if requested and requested in h5 and isinstance(h5[requested], h5py.Group):
        return requested
    if entries:
        return entries[0]
    return None


def _summarize_node(
    obj: h5py.File | h5py.Group | h5py.Dataset,
    path: str,
    *,
    max_depth: int,
    max_children: int,
    depth: int = 0,
) -> NexusNodeSummary:
    if isinstance(obj, h5py.Dataset):
        return NexusNodeSummary(
            path=path,
            kind="dataset",
            name=_node_name(path),
            shape=tuple(int(v) for v in obj.shape),
            dtype=str(obj.dtype),
            size=int(obj.size),
            attrs=_attrs(obj),
        )

    attrs = _attrs(obj)
    child_count = len(obj)
    keys = list(islice(obj.keys(), max_children))
    children: list[NexusNodeSummary] = []
    if depth < max_depth:
        for key in keys[:max_children]:
            child_path = f"/{key}" if path == "/" else f"{path.rstrip('/')}/{key}"
            try:
                child = obj[key]
            except Exception as exc:
                children.append(
                    NexusNodeSummary(
                        path=child_path,
                        kind="link_error",
                        name=key,
                        error=str(exc),
                    )
                )
                continue
            children.append(
                _summarize_node(
                    child,
                    child_path,
                    max_depth=max_depth,
                    max_children=max_children,
                    depth=depth + 1,
                )
            )
    return NexusNodeSummary(
        path=path,
        kind="group",
        name=_node_name(path),
        attrs=attrs,
        nx_class=_decode(attrs.get("NX_class")),
        children=tuple(children),
        child_count=child_count,
        truncated=child_count > len(children),
    )


def _summarize_xdart_entry(h5: h5py.File, entry: str) -> NexusXDartSummary:
    e = h5[entry]
    integrated_1d = _reduced_summary(e, "integrated_1d")
    integrated_2d = _reduced_summary(e, "integrated_2d")
    labels = sorted(
        set(integrated_1d.frame_labels if integrated_1d else ())
        | set(integrated_2d.frame_labels if integrated_2d else ())
        | set(_frame_group_labels(e))
    )
    scan_data_columns = _dataset_names(e.get("scan_data"), exclude={"frame_index"})
    geometry_columns = _dataset_names(e.get("per_frame_geometry"), exclude={"frame_index"})
    thumbnail_count, source_count = _frame_artifact_counts(e)
    raw_image_dataset, raw_image_shape, raw_image_dtype = _find_raw_image_dataset(e)
    is_processed = bool(
        integrated_1d
        or integrated_2d
        or thumbnail_count
        or source_count
    )
    return NexusXDartSummary(
        entry=entry,
        is_processed=is_processed,
        integrated_1d=integrated_1d,
        integrated_2d=integrated_2d,
        frame_labels=tuple(int(v) for v in labels),
        scan_data_columns=scan_data_columns,
        geometry_columns=geometry_columns,
        thumbnail_count=thumbnail_count,
        source_count=source_count,
        raw_image_dataset=raw_image_dataset,
        raw_image_shape=raw_image_shape,
        raw_image_dtype=raw_image_dtype,
    )


def _reduced_summary(e: h5py.Group, name: str) -> NexusReducedSummary | None:
    if name not in e or not isinstance(e[name], h5py.Group):
        return None
    group = e[name]
    labels = (
        tuple(int(v) for v in np.asarray(group["frame_index"][()]).ravel())
        if "frame_index" in group
        else ()
    )
    intensity_shape = (
        tuple(int(v) for v in group["intensity"].shape)
        if "intensity" in group and isinstance(group["intensity"], h5py.Dataset)
        else None
    )
    axes = tuple(_axis_summaries(group, name))
    two_d_kind = None
    if name == "integrated_2d":
        two_d_kind = _two_d_kind(group, axes)
    return NexusReducedSummary(
        path=f"{e.name}/{name}",
        frame_count=len(labels),
        frame_labels=labels,
        intensity_shape=intensity_shape,
        axes=axes,
        two_d_kind=two_d_kind,
    )


def _axis_summaries(group: h5py.Group, group_name: str) -> list[NexusAxisSummary]:
    candidates = ("q",) if group_name == "integrated_1d" else ("q", "chi")
    axes: list[NexusAxisSummary] = []
    for name in candidates:
        if name not in group or not isinstance(group[name], h5py.Dataset):
            continue
        ds = group[name]
        units = _decode(ds.attrs.get("units")) if "units" in ds.attrs else None
        axes.append(
            NexusAxisSummary(
                name=name,
                path=ds.name,
                shape=tuple(int(v) for v in ds.shape),
                units=units,
            )
        )
    return axes


def _two_d_kind(group: h5py.Group, axes: tuple[NexusAxisSummary, ...]) -> TwoDKind:
    attr = _decode(group.attrs.get("two_d_kind"))
    if attr:
        try:
            return TwoDKind(str(attr))
        except ValueError:
            pass
    units = {axis.name: axis.units for axis in axes}
    return two_d_kind_from_units(units.get("q"), units.get("chi"))


def _frame_group_labels(e: h5py.Group) -> tuple[int, ...]:
    if "frames" not in e or not isinstance(e["frames"], h5py.Group):
        return ()
    labels: list[int] = []
    for name in e["frames"].keys():
        if not name.startswith("frame_"):
            continue
        group = e["frames"].get(name)
        if (
            group is None
            or not isinstance(group, h5py.Group)
            or ("thumbnail" not in group and "source" not in group)
        ):
            continue
        try:
            labels.append(int(name.removeprefix("frame_")))
        except ValueError:
            continue
    return tuple(labels)


def _dataset_names(group, *, exclude: set[str]) -> tuple[str, ...]:
    if group is None or not isinstance(group, h5py.Group):
        return ()
    return tuple(
        name for name, obj in group.items()
        if name not in exclude and isinstance(obj, h5py.Dataset)
    )


def _frame_artifact_counts(e: h5py.Group) -> tuple[int, int]:
    if "frames" not in e or not isinstance(e["frames"], h5py.Group):
        return 0, 0
    thumbnail_count = 0
    source_count = 0
    for _, group in e["frames"].items():
        if not isinstance(group, h5py.Group):
            continue
        thumbnail_count += int("thumbnail" in group)
        source_count += int("source" in group)
    return thumbnail_count, source_count


def _find_raw_image_dataset(e: h5py.Group) -> tuple[str | None, tuple[int, ...] | None, str | None]:
    candidates = (
        f"{e.name}/instrument/detector/data",
        f"{e.name}/data/data",
    )
    for path in candidates:
        root = e.file
        if path in root and isinstance(root[path], h5py.Dataset) and root[path].ndim >= 2:
            ds = root[path]
            return path, tuple(int(v) for v in ds.shape), str(ds.dtype)
    best_path: str | None = None
    best_shape: tuple[int, ...] | None = None
    best_dtype: str | None = None
    best_size = 0

    def visit_group(group: h5py.Group, prefix: str = "") -> None:
        nonlocal best_path, best_size
        for name, obj in group.items():
            rel = f"{prefix}/{name}" if prefix else name
            if rel.startswith(("integrated_1d", "integrated_2d", "frames")):
                continue
            if isinstance(obj, h5py.Group):
                visit_group(obj, rel)
                continue
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 2:
                continue
            size = int(np.prod(obj.shape))
            if size > best_size:
                best_size = size
                best_path = f"{e.name}/{rel}"
                best_shape = tuple(int(v) for v in obj.shape)
                best_dtype = str(obj.dtype)

    visit_group(e)
    return best_path, best_shape, best_dtype


def _preview_selection(shape: tuple[int, ...], *, max_items: int) -> Any:
    if shape == ():
        return ()
    max_items = max(1, int(max_items))
    ndim = len(shape)
    if ndim == 1:
        return np.s_[: min(shape[0], max_items)]

    if ndim >= 2:
        rows = min(shape[-2], max(1, int(np.sqrt(max_items))))
        cols = min(shape[-1], max(1, max_items // rows))
        prefix = tuple(0 for _ in range(ndim - 2))
        return prefix + (slice(0, rows), slice(0, cols))

    return np.s_[()]


def _full_selection(shape: tuple[int, ...]) -> Any:
    if shape == ():
        return ()
    return tuple(slice(None) for _ in shape)


def _selection_text(selection: Any, shape: tuple[int, ...]) -> str:
    if selection == ():
        return "scalar"
    if not isinstance(selection, tuple):
        selection = (selection,)
    parts: list[str] = []
    for sel, dim in zip(selection, shape):
        if isinstance(sel, slice):
            start = "" if sel.start in (None, 0) else str(sel.start)
            stop = "" if sel.stop is None or sel.stop == dim else str(sel.stop)
            if sel.step not in (None, 1):
                parts.append(f"{start}:{stop}:{sel.step}")
            else:
                parts.append(f"{start}:{stop}")
        else:
            parts.append(str(sel))
    return "[" + ", ".join(parts) + "]"


def _selection_truncated(selection: Any, shape: tuple[int, ...]) -> bool:
    if selection == ():
        return False
    if not isinstance(selection, tuple):
        selection = (selection,)
    if len(selection) < len(shape):
        return True
    for sel, dim in zip(selection, shape):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(dim)
            if start != 0 or stop != dim or step != 1:
                return True
        elif dim != 1:
            return True
    return False


__all__ = [
    "NexusAxisSummary",
    "NexusDatasetPreview",
    "NexusDatasetData",
    "NexusFileSummary",
    "NexusNodeSummary",
    "NexusReducedSummary",
    "NexusXDartSummary",
    "inspect_nexus",
    "preview_nexus_dataset",
    "read_nexus_dataset",
]
