# xrd_tools/io/processed_scan_id.py
"""Separate processed-result rejection from current-output admission.

A processed xdart output stores REDUCED results (the ``integrated_1d`` /
``integrated_2d`` NXdata stacks) and carries NO raw detector frames.  Fed to a
raw-detector resolver it is dangerous: the generic "largest 3-D dataset" fallback
happily returns ``/entry/integrated_2d/intensity`` — an integrated cake — as
though it were a detector stack (v1.1.2 finding F-NXS-2).

Raw-input guards must reject integrated result groups regardless of suffix or
schema stamp.  Positive processed-output admission is intentionally stricter:
only a ``.nexus`` file carrying the supported schema identity and integrated
results is an admitted xdart output (v2 read-only, v3 read/write). Keeping the two questions separate prevents
a raw ``.nxs`` with unrelated result groups from being positively claimed while
still preventing those groups from being mistaken for detector frames.

Import-light: depends only on :mod:`h5py` and the frozen schema-identity
constants — so :mod:`xrd_tools.io.image`, :mod:`~xrd_tools.io.nexus` and Qt-free
callers can all import it without a cycle or a heavy dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
from h5py._hl.files import File as _H5File
from h5py._hl.dataset import Dataset as _H5Dataset
from h5py._hl.group import Group as _H5Group

from xrd_tools.io.schema import (
    DEFAULT_MODE_KEY,
    MULTI_RESULT_MODES_ATTR,
    PRIMARY_MODE_ATTR,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    REINTEGRATE_SHADOW_SUFFIX,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    integrated_axis_names,
    axis_display_metadata,
    read_current_mode_layout,
    resolve_integrated_group,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ProcessedXdartInputError",
    "CurrentProcessedGroups",
    "has_processed_output_markers_entry",
    "has_processed_output_markers_file",
    "has_processed_output_markers_path",
    "is_current_processed_xdart_file",
    "is_current_processed_xdart_path",
    "require_current_output_path",
    "require_current_processed",
    "require_current_processed_groups",
    "require_current_writable_processed_groups",
    "require_raw_input",
]

# xdart's own reduced-result group names.  These are the output layout the GUI /
# headless writer produces; a raw acquisition (Eiger master, Bluesky NXWriter,
# plain detector NeXus) never contains them.
_PROCESSED_RESULT_GROUPS = ("integrated_1d", "integrated_2d")
_ANALYSIS_RESULT_GROUPS = ("stitched_1d", "stitched_2d", "rsm")
_ANALYSIS_SCHEMA_ATTR = "ssrl_schema"
_ANALYSIS_SCHEMA_NAME = "xrd_tools.analysis_artifact"
_MAX_CURRENT_RESULT_ROWS = 1_000_000
_LABEL_VALIDATION_CHUNK = 65_536


class ProcessedXdartInputError(ValueError):
    """Raised when a processed xdart scan file is offered where a RAW detector
    container is required (e.g. the directory-watch image reader).

    A :class:`ValueError` subclass so existing broad ``except ValueError`` /
    ``except Exception`` handling still catches it, while a caller that wants to
    *skip a processed candidate and continue* can catch it by type.
    """


@dataclass(frozen=True)
class CurrentProcessedGroups:
    """Exact groups and axis layout certified together by read admission."""

    entry: h5py.Group
    schema_version: int
    integrated_1d: h5py.Group | None
    integrated_2d: h5py.Group | None
    primary_mode_1d: str
    primary_mode_2d: str
    mode_groups_1d: tuple[tuple[str, h5py.Group], ...]
    mode_groups_2d: tuple[tuple[str, h5py.Group], ...]

    @property
    def axis_names(self) -> tuple[str, str]:
        return integrated_axis_names(self.schema_version)

    @property
    def modes_1d(self) -> tuple[str, ...]:
        return tuple(mode for mode, _group in self.mode_groups_1d)

    @property
    def modes_2d(self) -> tuple[str, ...]:
        return tuple(mode for mode, _group in self.mode_groups_2d)

    def mode_group(self, dimension: str, mode: str) -> h5py.Group | None:
        pairs = (
            self.mode_groups_1d if dimension == "1d"
            else self.mode_groups_2d if dimension == "2d"
            else ()
        )
        return next((group for key, group in pairs if key == mode), None)


def has_processed_output_markers_entry(entry: h5py.Group) -> bool:
    """Broad raw-negative guard for a schema stamp or result groups.

    A current-schema stamp is sufficient even when a write was interrupted
    before either integrated group was created.  Historical unstamped outputs
    remain identifiable by their result groups.  This deliberately says only
    "unsafe as raw"; strict positive admission remains a separate decision.
    """
    try:
        return isinstance(entry, h5py.Group) and (
            _attr_str(entry.attrs.get(SCHEMA_NAME_ATTR))
            == PROCESSED_SCHEMA_NAME
            or _attr_str(entry.attrs.get(_ANALYSIS_SCHEMA_ATTR))
            == _ANALYSIS_SCHEMA_NAME
            or any(group in entry for group in _PROCESSED_RESULT_GROUPS)
            or any(group in entry for group in _ANALYSIS_RESULT_GROUPS)
        )
    except Exception:
        logger.debug(
            "has_processed_output_markers_entry: traversal error",
            exc_info=True,
        )
        return False


def _attr_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _resolved_entry(f: h5py.File, entry: str) -> h5py.Group | None:
    try:
        grp = f.get(entry)
        if not isinstance(grp, h5py.Group):
            from xrd_tools.io.bluesky_nexus import resolve_nxentry
            grp = resolve_nxentry(f, entry)
        return grp if isinstance(grp, h5py.Group) else None
    except Exception:
        logger.debug("processed entry resolution failed", exc_info=True)
        return None


def has_processed_output_markers_file(
    f: h5py.File,
    entry: str = "entry",
) -> bool:
    """Broad negative guard across the owning HDF5 file's local entries."""
    grp = _resolved_entry(f, entry)
    if grp is not None and has_processed_output_markers_entry(grp):
        return True
    try:
        for name in f:
            if type(f.get(name, getlink=True)) is not h5py.HardLink:
                continue
            value = f.get(name)
            if (
                isinstance(value, h5py.Group)
                and has_processed_output_markers_entry(value)
            ):
                return True
    except Exception:
        logger.debug("processed marker file census failed", exc_info=True)
    return False


def has_processed_output_markers_path(path, entry: str = "entry") -> bool:
    """Broad negative guard for a path; never raises on open failure."""
    try:
        with h5py.File(Path(path), "r") as f:
            return has_processed_output_markers_file(f, entry)
    except OSError:
        return False


def require_raw_input(
    source: h5py.File | h5py.Group | h5py.Dataset | str | Path,
    entry: str = "entry",
) -> None:
    """Reject every processed-output marker before a raw HDF traversal.

    A bound :class:`h5py.Dataset` may belong to a different file after an
    ``ExternalLink`` dereference.  Authenticate that owning file and the
    dataset's actual top-level entry, rather than assuming the already-checked
    master also owns the pixels.
    """
    if isinstance(source, _H5File):
        marked = has_processed_output_markers_file(source, entry)
        shown = source.filename
    elif isinstance(source, (_H5Dataset, _H5Group)):
        owner = source.file
        parts = tuple(part for part in source.name.split("/") if part)
        owner_entry = parts[0] if parts else entry
        marked = has_processed_output_markers_file(owner, owner_entry)
        shown = owner.filename
    else:
        shown = Path(source)
        try:
            with h5py.File(shown, "r") as handle:
                marked = has_processed_output_markers_file(handle, entry)
        except OSError:
            return
    if marked:
        raise ProcessedXdartInputError(
            f"{shown} carries processed-output markers and cannot be opened "
            "as raw detector data"
        )


def _attr_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _direct_dataset(group: h5py.Group, name: str) -> h5py.Dataset | None:
    """Return one directly stored local dataset, never indirect storage."""
    try:
        if type(group.get(name, getlink=True)) is not h5py.HardLink:
            return None
        value = group.get(name)
        if not isinstance(value, h5py.Dataset):
            return None
        external = tuple(value.external or ())
        return None if value.is_virtual or external else value
    except Exception:
        return None


def require_current_output_path(path: str | Path) -> Path:
    """Require the one forward processed-output suffix without side effects."""
    target = Path(path)
    if target.suffix.casefold() != ".nexus":
        raise ValueError("current processed output target must end in .nexus")
    return target


def _valid_current_result_group(group: h5py.Group, name: str, version: int) -> bool:
    """Qualify one concrete supported stack, not a name-only marker."""
    try:
        x_name, y_name = integrated_axis_names(version)
        other_names = ("axis_x", "axis_y") if version == 2 else ("q", "chi")
        if any(key in group for key in other_names):
            return False
        expected_axes = (
            ("frame_index", x_name)
            if name == "integrated_1d"
            else ("frame_index", y_name, x_name)
        )
        axes = group.attrs.get("axes")
        if isinstance(axes, np.ndarray):
            axes = tuple(_attr_str(value) for value in axes.tolist())
        elif isinstance(axes, (tuple, list)):
            axes = tuple(_attr_str(value) for value in axes)
        else:
            axes = (_attr_str(axes),)
        if (
            _attr_str(group.attrs.get("NX_class")) != "NXdata"
            or _attr_str(group.attrs.get("signal")) != "intensity"
            or axes != expected_axes
        ):
            return False
        labels = _direct_dataset(group, "frame_index")
        intensity = _direct_dataset(group, "intensity")
        q = _direct_dataset(group, x_name)
        chi = None if name == "integrated_1d" else _direct_dataset(group, y_name)
        if (
            labels is None
            or intensity is None
            or q is None
            or labels.ndim != 1
            or labels.shape[0] < 1
            or labels.shape[0] > _MAX_CURRENT_RESULT_ROWS
            or labels.dtype != np.dtype(np.int64)
            or q.ndim != 1
            or q.shape[0] < 1
            or q.dtype != np.dtype(np.float32)
            or intensity.dtype != np.dtype(np.float32)
        ):
            return False
        if name == "integrated_2d" and (
            chi is None
            or chi.ndim != 1
            or chi.shape[0] < 1
            or chi.dtype != np.dtype(np.float32)
        ):
            return False
        expected_shape = (
            (labels.shape[0], q.shape[0])
            if name == "integrated_1d"
            else (labels.shape[0], chi.shape[0], q.shape[0])
        )
        if intensity.shape != expected_shape:
            return False
        sigma_link = group.get("sigma", getlink=True)
        sigma = _direct_dataset(group, "sigma")
        if sigma_link is not None and sigma is None:
            return False
        if sigma is not None and (
            sigma.dtype != np.dtype(np.float32) or sigma.shape != expected_shape
        ):
            return False
        row_datasets = (labels, intensity) + (() if sigma is None else (sigma,))
        if any(
            dataset.chunks is None
            or dataset.maxshape is None
            or dataset.maxshape[0] is not None
            for dataset in row_datasets
        ):
            return False
        if any(
            left.id == right.id
            for index, left in enumerate(row_datasets)
            for right in row_datasets[index + 1:]
        ):
            return False
        previous = -1
        for start in range(0, labels.shape[0], _LABEL_VALIDATION_CHUNK):
            rows = np.asarray(
                labels[start:start + _LABEL_VALIDATION_CHUNK],
                dtype=np.int64,
            )
            if (
                rows.ndim != 1
                or rows.size < 1
                or int(rows[0]) <= previous
                or np.any(rows < 0)
                or (rows.size > 1 and np.any(rows[1:] <= rows[:-1]))
            ):
                return False
            previous = int(rows[-1])
        return True
    except Exception:
        logger.debug("current result-group qualification failed", exc_info=True)
        return False


def _valid_current_result_leaf(group: h5py.Group, name: str, version: int) -> bool:
    """Qualify one owned result leaf, including its complete local graph."""
    if not _valid_current_result_group(group, name, version):
        return False
    if (
        PRIMARY_MODE_ATTR in group.attrs
        or MULTI_RESULT_MODES_ATTR in group.attrs
    ):
        return False
    try:
        for child_name in group:
            if type(group.get(child_name, getlink=True)) is not h5py.HardLink:
                return False
            if isinstance(group.get(child_name), h5py.Group):
                return False
        return True
    except Exception:
        return False


def _qualified_current_processed_entry(
    entry: h5py.Group,
    *,
    container: str | Path,
) -> CurrentProcessedGroups | None:
    """Qualify one local entry against its caller-supplied container path."""
    try:
        if (
            not isinstance(entry, h5py.Group)
            or Path(container).suffix.casefold() != ".nexus"
        ):
            return None
        version = _attr_int(entry.attrs.get(SCHEMA_VERSION_ATTR))
        if (_attr_str(entry.attrs.get(SCHEMA_NAME_ATTR)) != PROCESSED_SCHEMA_NAME
                or version not in (2, PROCESSED_SCHEMA_VERSION)):
            return None
        groups: dict[str, h5py.Group | None] = {}
        primaries = {
            "integrated_1d": DEFAULT_MODE_KEY,
            "integrated_2d": DEFAULT_MODE_KEY,
        }
        mode_groups: dict[str, tuple[tuple[str, h5py.Group], ...]] = {
            "integrated_1d": (),
            "integrated_2d": (),
        }
        row_dataset_owners: list[h5py.Dataset] = []
        present = False
        for name in _PROCESSED_RESULT_GROUPS:
            canonical_present = entry.get(name, getlink=True) is not None
            shadow_present = entry.get(
                f"{name}{REINTEGRATE_SHADOW_SUFFIX}", getlink=True,
            ) is not None
            group, _adopted = resolve_integrated_group(entry, name)
            slot_present = canonical_present or shadow_present
            present = present or slot_present
            if slot_present and (
                not isinstance(group, h5py.Group)
                or not _valid_current_result_group(group, name, version)
            ):
                return None
            if isinstance(group, h5py.Group):
                dimension = "1d" if name == "integrated_1d" else "2d"
                primary, _modes, pairs = read_current_mode_layout(
                    group,
                    dimension,
                )
                if any(
                    mode != primary and not _valid_current_result_leaf(child, name, version)
                    for mode, child in pairs
                ):
                    return None
                for _mode, owned_group in pairs:
                    for dataset_name in ("frame_index", "intensity", "sigma"):
                        dataset = _direct_dataset(owned_group, dataset_name)
                        if dataset is None:
                            if dataset_name == "sigma":
                                continue
                            return None
                        if any(
                            dataset.id == prior.id
                            for prior in row_dataset_owners
                        ):
                            return None
                        row_dataset_owners.append(dataset)
                primaries[name] = primary
                mode_groups[name] = pairs
            groups[name] = group if isinstance(group, h5py.Group) else None
        if not present:
            return None
        return CurrentProcessedGroups(
            entry=entry,
            schema_version=version,
            integrated_1d=groups["integrated_1d"],
            integrated_2d=groups["integrated_2d"],
            primary_mode_1d=primaries["integrated_1d"],
            primary_mode_2d=primaries["integrated_2d"],
            mode_groups_1d=mode_groups["integrated_1d"],
            mode_groups_2d=mode_groups["integrated_2d"],
        )
    except Exception:
        logger.debug(
            "current processed-entry qualification failed",
            exc_info=True,
        )
        return None


def _local_requested_entry(
    f: h5py.File,
    entry: str,
) -> h5py.Group | None:
    """Resolve the requested entry only when its link is local and hard."""
    try:
        if not isinstance(entry, str):
            return None
        name = entry[1:] if entry.startswith("/") else entry
        if not name or "/" in name or name in {".", ".."}:
            return None
        if type(f.get(name, getlink=True)) is not h5py.HardLink:
            return None
        value = f.get(name)
        return value if isinstance(value, h5py.Group) else None
    except Exception:
        return None


def _current_processed_groups_file(
    f: h5py.File,
    entry: str,
    *,
    container: str | Path | None = None,
) -> CurrentProcessedGroups | None:
    group = _local_requested_entry(f, entry)
    if group is None:
        return None
    container_path = Path(f.filename) if container is None else Path(container)
    return _qualified_current_processed_entry(group, container=container_path)


def is_current_processed_xdart_file(
    f: h5py.File,
    entry: str = "entry",
) -> bool:
    """Strict positive recognition of a supported v2/v3 processed output."""
    return _current_processed_groups_file(f, entry) is not None


def is_current_processed_xdart_path(path, entry: str = "entry") -> bool:
    """Strict positive admission for a current ``.nexus`` path.

    Suffix alone is never sufficient; the supported schema name/version and
    integrated result content must all be present.
    """
    try:
        with h5py.File(Path(path), "r") as f:
            return _current_processed_groups_file(
                f,
                entry,
                container=Path(path),
            ) is not None
    except OSError:
        return False


def require_current_processed(
    source: h5py.File | str | Path,
    entry: str = "entry",
    *,
    container: str | Path | None = None,
) -> None:
    """Require one exact supported v2/v3 record at a public read boundary."""
    if isinstance(source, _H5File):
        accepted = _current_processed_groups_file(
            source, entry, container=container,
        ) is not None
    else:
        accepted = is_current_processed_xdart_path(source, entry)
    if not accepted:
        raise ValueError("processed input is not a current xdart .nexus record")


def require_current_processed_groups(
    source: h5py.File,
    entry: str = "entry",
    *,
    container: str | Path | None = None,
) -> CurrentProcessedGroups:
    """Return the exact entry/result groups certified by one admission pass."""
    groups = _current_processed_groups_file(
        source,
        entry,
        container=container,
    )
    if groups is None:
        raise ValueError("processed input is not a current xdart .nexus record")
    return groups


def require_current_writable_processed_groups(
    source: h5py.File,
    entry: str = "entry",
    *,
    container: str | Path | None = None,
) -> CurrentProcessedGroups:
    """Require canonical current result slots suitable for ordinary append.

    A completed ``__reint`` shadow remains a valid read-recovery surface, but
    it is not silently repaired or treated as the canonical append target.
    """
    groups = require_current_processed_groups(
        source,
        entry,
        container=container,
    )
    if groups.schema_version != PROCESSED_SCHEMA_VERSION:
        raise ValueError("read-only processed v2 record; create a new v3 output instead of Append")
    for name, group in (
        ("integrated_1d", groups.integrated_1d),
        ("integrated_2d", groups.integrated_2d),
    ):
        if group is None:
            continue
        canonical = groups.entry.get(name)
        if not isinstance(canonical, h5py.Group) or canonical.id != group.id:
            raise ValueError(
                f"completed {name} reintegration shadow is read-only recovery"
            )
    return groups


def neutral_axis_upgrade(
    groups: CurrentProcessedGroups,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """Describe only the approved v2-to-v3 names and display-attribute changes.

    Shared by private-candidate conversion and its read-only preservation
    comparison. No arrays are copied or converted; the caller owns mutation.
    """
    if groups.schema_version != 2:
        return {}, {}
    renames = {}
    attributes = {
        groups.entry.name: {SCHEMA_VERSION_ATTR: np.int64(PROCESSED_SCHEMA_VERSION)},
    }
    for dimension, pairs in (
        ("1d", groups.mode_groups_1d), ("2d", groups.mode_groups_2d),
    ):
        for _mode, group in pairs:
            names = ("q",) if dimension == "1d" else ("q", "chi")
            axes = (
                ["frame_index", "axis_x"] if dimension == "1d"
                else ["frame_index", "axis_y", "axis_x"]
            )
            attributes[group.name] = {"axes": np.asarray(axes, dtype=object)}
            for old, new in zip(names, ("axis_x", "axis_y")):
                node = group[old]
                renames[node.name] = new
                unit = _attr_str(node.attrs.get("units", ""))
                label = axis_display_metadata(unit)["long_name"]
                attributes[node.name] = {"long_name": label}
    return renames, attributes


def upgrade_private_integrated_axes(
    document: h5py.File, entry: str, *, container: str | Path,
) -> None:
    """Normalize an already publisher-owned candidate, never a source file."""
    groups = require_current_processed_groups(document, entry, container=container)
    renames, attributes = neutral_axis_upgrade(groups)
    if not renames:
        return
    from .record_writer import _neutral_axis_result_seals
    seals = _neutral_axis_result_seals(document, entry, (renames, attributes))
    for path, values in attributes.items():
        for key, value in values.items():
            document[path].attrs[key] = value
    for path, new in renames.items():
        parent, old = path.rsplit("/", 1)
        document[parent].move(old, new)
    for path, value in seals.items():
        document[path][()] = value
