"""Qt-free lifecycle owner for incremental processed-NeXus records."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import shutil
import struct
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import uuid
import warnings

import h5py
import numpy as np

from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
from xrd_tools.core.frame_view import two_d_kind_from_units
from xrd_tools.core.provenance import write_provenance
from xrd_tools.io.nexus import (
    _axis_kind_1d,
    _axis_kind_from_group_1d,
    open_nexus_writer,
    upsert_per_frame_geometry,
    upsert_positioners,
    upsert_scan_metadata,
    validate_integrated_stack_write,
    write_diffractometer,
    write_integrated_stack,
    write_stitched,
)
from xrd_tools.io.nexus_record import (
    _PreparedThumbnail,
    _average_count_chunks,
    _background_pair,
    _prepare_thumbnail,
    ensure_frames_container,
    read_background_dependency,
    replace_frame_record,
    stamp_source_base,
    validate_source_base,
    write_average_finite_counts,
)
from xrd_tools.io.append import (
    AppendDecision,
    AppendDisposition,
    _MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
    _MAX_REPLACEMENT_LINEAGE_UTF8_BYTES,
    _MAX_REPLACEMENT_PATH_UTF8_BYTES,
    commit_append_lineage,
    decode_replacement_lineage,
    _replacement_hard_group,
    _replacement_utf8_attribute,
    _replacement_utf8_scalar,
    stage_append_lineage,
)
from xrd_tools.io.output_transaction import (
    OutputTransaction,
    OutputTransactionError,
    StreamAttempt,
    StreamTerminal,
    TargetLease,
)
from xrd_tools.io.finite_artifact import (
    FiniteArtifactRequest,
    FiniteCandidateBinding,
    FiniteFileSnapshot,
    FiniteSeedBinding,
    finite_lineage_hdf_path,
    write_finite_artifact_lineage,
)
from xrd_tools.io.processed_scan_id import (
    require_current_output_path,
    require_current_writable_processed_groups,
)
from xrd_tools.io.read import relative_source_path
from xrd_tools.io.schema import (
    GI_MODE_KEYS_1D,
    GI_MODE_KEYS_2D,
    INTEGRATED_ROW_ALIGNED,
    MONOTONIC_ATTR,
    PRIMARY_MODE_ATTR,
    SOURCE_BASE_ATTR,
    THUMBNAIL_LUT_ATTRS,
    canonical_gi_mode_key,
    local_hard_dataset,
    local_hard_group_path,
    mode_subgroup_name,
    read_current_mode_layout,
)
from xrd_tools.session import ResultMode, StageReceipt, get_pool


def _iter_persisted_background_bindings(path, labels):
    from xrd_tools.reduction.background import _frame
    nested = lambda value: tuple(nested(item) for item in value) if isinstance(value, list) else value
    with h5py.File(path, "r") as handle:
        frames = handle.get("entry/frames")
        for label in tuple(int(value) for value in labels):
            frame = None if not isinstance(frames, h5py.Group) else frames.get(f"frame_{label:04d}")
            pair = None if not isinstance(frame, h5py.Group) else read_background_dependency(frame)
            if pair is None: raise ValueError("persisted Background dependency is absent")
            try: fact = _frame(nested(json.loads(pair[0])["frame_fact"]), persisted=True)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError("persisted Background frame fact is malformed") from error
            if fact[0] != label: raise ValueError("persisted Background label differs")
            yield label, fact, pair[0], pair[1]
def _read_persisted_background_bindings(path, labels, limit):
    bindings = []; charge = 0
    for binding in _iter_persisted_background_bindings(path, labels):
        charge += 1024 + len(json.dumps(binding[1], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()) + len(binding[2]) + 64; charge <= limit or (_ for _ in ()).throw(ValueError("persisted Background dependency exceeds allocation")); bindings.append(binding)
    return tuple(bindings)
def _validate_persisted_background_bindings(path, bindings, labels) -> None:
    for observed in _iter_persisted_background_bindings(path, labels): observed == next((value for value in bindings if value[0] == observed[0]), None) or (_ for _ in ()).throw(ValueError("persisted Background dependency changed after qualification"))


class WriterPhase(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    PARTIAL = "partial"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class WriterOperationVector:
    prepare_rows: int = 0
    stacked_1d_rows: int = 0
    stacked_2d_rows: int = 0
    indexed_metadata_rows: int = 0
    source_record_rows: int = 0
    provenance_boundaries: int = 0
    frame_index_scan_rows: int = 0
    untouched_hydration_rows: int = 0
    harvested_source_rows: int = 0
    flush_boundaries: int = 0


@dataclass(frozen=True, slots=True)
class RecordWrite:
    label: int
    result_1d: Any | None = None
    result_2d: Any | None = None
    mode_1d: str = DEFAULT_MODE_KEY
    mode_2d: str = DEFAULT_MODE_KEY
    source_path: Path | str | None = None
    source_frame_index: int = 0
    source_snapshot: Mapping[str, Any] = field(default_factory=dict)
    timestamp: Any | None = None
    thumbnail: Any | None = None
    thumbnail_mask_baked: bool = True
    mask_baked: bool = True
    thumbnail_mask: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    write_frame_record: bool = True
    replace_existing: bool = False
    background_dependency_bytes: bytes | None = None
    background_dependency_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode_1d", canonical_gi_mode_key(self.mode_1d, "1d"),
        )
        object.__setattr__(
            self, "mode_2d", canonical_gi_mode_key(self.mode_2d, "2d"),
        )

        def exact_bool(value: Any, name: str) -> bool:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an exact boolean")
            return bool(value)

        def exact_nonnegative_int(value: Any, name: str) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an exact integer")
            normalized = int(value)
            if normalized < 0:
                raise ValueError(f"{name} must be non-negative")
            return normalized

        object.__setattr__(self, "label", exact_nonnegative_int(self.label, "label"))
        object.__setattr__(
            self, "source_frame_index",
            exact_nonnegative_int(self.source_frame_index, "source_frame_index"),
        )
        if self.background_dependency_bytes is not None or self.background_dependency_fingerprint is not None:
            pair = _background_pair(self.background_dependency_bytes, self.background_dependency_fingerprint); json.loads(pair[0])["frame_fact"][0] == self.label or (_ for _ in ()).throw(ValueError("Background dependency label differs from RecordWrite"))
            object.__setattr__(self, "background_dependency_bytes", pair[0])
            object.__setattr__(self, "background_dependency_fingerprint", pair[1])
        object.__setattr__(
            self, "thumbnail_mask_baked",
            exact_bool(self.thumbnail_mask_baked, "thumbnail_mask_baked"),
        )
        object.__setattr__(self, "mask_baked", exact_bool(self.mask_baked, "mask_baked"))
        object.__setattr__(
            self, "write_frame_record",
            exact_bool(self.write_frame_record, "write_frame_record"),
        )
        object.__setattr__(
            self, "replace_existing",
            exact_bool(self.replace_existing, "replace_existing"),
        )
        allowed_snapshot = {
            "adapter_id", "size", "mtime_ns", "frame_count",
            "dataset_path", "self_contained",
        }
        snapshot = {}
        for key, value in dict(self.source_snapshot or {}).items():
            key = str(key)
            if key not in allowed_snapshot:
                raise ValueError(f"unknown source snapshot field {key!r}")
            if key in {"size", "mtime_ns", "frame_count"}:
                value = exact_nonnegative_int(value, f"source snapshot {key}")
            elif key == "self_contained":
                value = exact_bool(value, "source snapshot self_contained")
            elif value is None:
                raise ValueError(f"source snapshot {key} may not be null")
            else:
                value = str(value)
                if not value:
                    raise ValueError(f"source snapshot {key} may not be empty")
                if len(value.encode("utf-8")) > _MAX_REPLACEMENT_PATH_UTF8_BYTES:
                    raise ValueError(f"source snapshot {key} exceeds the persisted path byte ceiling")
            snapshot[key] = value
        if snapshot and self.source_path is None:
            raise ValueError("source snapshot requires source_path")
        if self.source_path is None and self.source_frame_index != 0:
            raise ValueError(
                "absent source_path requires source_frame_index to be exactly 0"
            )
        if (self.source_path is not None
                and len(str(self.source_path).encode("utf-8"))
                    > _MAX_REPLACEMENT_PATH_UTF8_BYTES):
            raise ValueError("source_path exceeds the persisted path byte ceiling")
        object.__setattr__(self, "source_snapshot", MappingProxyType(snapshot))
        if self.thumbnail_mask is not None:
            incoming_mask = np.asarray(self.thumbnail_mask)
            if incoming_mask.dtype != np.dtype(bool):
                raise ValueError("record thumbnail_mask dtype must be bool")
            mask = np.array(incoming_mask, copy=True)
            if mask.ndim != 2:
                raise ValueError("record thumbnail_mask must be exactly 2-D")
            mask.setflags(write=False)
            object.__setattr__(self, "thumbnail_mask", mask)


@dataclass(frozen=True, slots=True)
class WriterFinalization:
    scan_data: Any | None = None
    frame_indices: tuple[int, ...] = ()
    geometry: Any | None = None
    diffractometer: Any | None = None
    provenance_config: Mapping[str, Any] | None = None
    provenance_inputs: Mapping[str, Any] | None = None
    program: str = "xrd-tools"
    program_version: str | None = None
    date: str | None = None
    host: str | None = None
    detector_calibration: Mapping[str, Any] | None = None
    global_mask: Any | None = None
    detector_shape: tuple[int, ...] | None = None
    stitched_1d: Any | None = None
    stitched_2d: Any | None = None
    stitched_provenance: Mapping[str, Any] | str | None = None
    average_finite_counts: Any | None = None


@dataclass(frozen=True, slots=True)
class WriterOutcome:
    phase: WriterPhase
    target: Path
    partial_path: Path | None
    pending_owner: str | None
    operation_vector: WriterOperationVector
    stream_terminal: StreamTerminal | None = None


@dataclass(frozen=True, slots=True)
class WriterTransactionBinding:
    transaction: OutputTransaction
    attempt: StreamAttempt
    lease: TargetLease


@dataclass(frozen=True, slots=True)
class _ExpectedModeRow:
    group_name: str
    label: int
    row: int
    dimension: str
    mode: str
    radial: np.ndarray
    azimuthal: np.ndarray | None
    intensity: np.ndarray
    sigma: np.ndarray | None
    unit: str
    azimuthal_unit: str | None
    two_d_kind: str | None
    source_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedFrameRow:
    label: int
    thumbnail: np.ndarray | None
    thumbnail_lut: tuple[float, float, str] | None
    thumbnail_mask_baked: bool
    mask_baked: bool
    thumbnail_mask: np.ndarray | None
    source_path: str | None
    source_frame_index: int | None
    source_snapshot: tuple[tuple[str, Any], ...]
    timestamp: str | None
    background_dependency: tuple[bytes, str] | None


@dataclass(frozen=True, slots=True)
class _PersistedSourceFact:
    present: bool
    path: str | None
    frame_index: int | None
    snapshot: tuple[tuple[str, Any | None], ...]


_SOURCE_SNAPSHOT_ATTRIBUTES = (
    ("adapter_id", "adapter_id"),
    ("size", "file_size"),
    ("mtime_ns", "file_mtime_ns"),
    ("frame_count", "frame_count"),
    ("dataset_path", "dataset_path"),
    ("self_contained", "self_contained"),
)
_REPLACEMENT_STATE_KEYS, _REPLACEMENT_EXECUTION_KEYS = {"path", "size", "mtime_ns", "ctime_ns", "device", "inode"}, {"path", "size", "mtime_ns", "ctime_ns", "device", "inode", "adapter_id", "frame_count", "first_label", "member_stamps", "external_members", "dependency_files", "admitted_motor_values", "metadata_sources"}
def _validate_replacement_execution(value: Any) -> Mapping[str, Any]:
    def state(item: Any, role: str) -> Mapping[str, Any]:
        (None if type(item) is dict and set(item) == _REPLACEMENT_STATE_KEYS and type(item["path"]) is str and os.path.isabs(item["path"]) and all(type(item[key]) is int and item[key] >= 0 for key in _REPLACEMENT_STATE_KEYS - {"path"}) else (_ for _ in ()).throw(WriterStateError(f"replacement {role} state is malformed"))); return item
    if type(value) is not dict or set(value) != _REPLACEMENT_EXECUTION_KEYS or type(value["adapter_id"]) is not str or not value["adapter_id"] or any(type(value[key]) is not int or value[key] < 0 for key in ("frame_count", "first_label")) or any(type(value[key]) is not list for key in ("member_stamps", "external_members", "dependency_files", "admitted_motor_values", "metadata_sources")): raise WriterStateError("replacement source_execution is absent or malformed")
    state({key: value[key] for key in _REPLACEMENT_STATE_KEYS}, "source"); [state(item, "member") for item in value["member_stamps"]]; [state(item, "dependency") for item in value["dependency_files"]]
    if any(type(item) is not dict or set(item) != {"file", "dataset", "first", "stop", "epoch"} or type(item["dataset"]) is not str or not item["dataset"].startswith("/") or any(type(item[key]) is not int or item[key] < 0 for key in ("first", "stop", "epoch")) or item["stop"] <= item["first"] for item in value["external_members"]): raise WriterStateError("replacement external member is malformed")
    [state(item["file"], "external member") for item in value["external_members"]]
    if any(type(item) is not dict or set(item) != {"source_path", "motor", "value"} or type(item["source_path"]) is not str or not os.path.isabs(item["source_path"]) or type(item["motor"]) is not str or not item["motor"] or item["motor"] == "Manual" or type(item["value"]) is not float or not np.isfinite(item["value"]) for item in value["admitted_motor_values"]): raise WriterStateError("replacement motor value is malformed")
    for item in value["metadata_sources"]:
        (None if type(item) is dict and set(item) == {"source_path", "metadata_file"} and type(item["source_path"]) is str and os.path.isabs(item["source_path"]) and (item["metadata_file"] is None or state(item["metadata_file"], "metadata") is item["metadata_file"]) else (_ for _ in ()).throw(WriterStateError("replacement metadata source is malformed")))
    if len({item["path"] for item in value["dependency_files"]}) != len(value["dependency_files"]) or value["admitted_motor_values"] and ([item["source_path"] for item in value["admitted_motor_values"]] != [item["path"] for item in value["member_stamps"]] or len({item["motor"] for item in value["admitted_motor_values"]}) != 1) or value["metadata_sources"] and (value["adapter_id"] != "tiff_series" or [item["source_path"] for item in value["metadata_sources"]] != [item["path"] for item in value["member_stamps"]]): raise WriterStateError("replacement source_execution member alignment is malformed")
    return value
def _replacement_scalar(value: Any, role: str) -> Any:
    try: return value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value.item() if isinstance(value, np.generic) else value
    except UnicodeDecodeError as error: raise WriterStateError(f"{role} is not UTF-8") from error
def _replacement_dtype_signature(dtype): dtype = np.dtype(dtype); string, vlen, enum, reference = h5py.check_string_dtype(dtype), h5py.check_vlen_dtype(dtype), h5py.check_enum_dtype(dtype), h5py.check_ref_dtype(dtype); token = lambda value: None if value is None else ("dtype", np.dtype(value).str, np.dtype(value).descr) if isinstance(value, np.dtype) else ("type", value.__module__, value.__qualname__) if isinstance(value, type) else (type(value).__module__, type(value).__qualname__, repr(value)); return (dtype.str, dtype.descr, None if string is None else (string.encoding, string.length), token(vlen), tuple(sorted((enum or {}).items())), token(reference))
def _replacement_atom(value): value = value.item() if isinstance(value, np.generic) else value; return ("bytes", value.hex()) if isinstance(value, bytes) else ("text", value) if isinstance(value, str) else ("array", _replacement_value_signature(value)) if isinstance(value, np.ndarray) else ("reference", type(value).__module__, type(value).__qualname__, bool(value), hash(value) if value else None) if isinstance(value, h5py.Reference) else ("value", type(value).__module__, type(value).__qualname__, repr(value))
def _replacement_value_signature(value, dtype=None): value = np.asarray(value); payload = json.dumps([_replacement_atom(item) for item in value.ravel()], ensure_ascii=False, separators=(",", ":")).encode() if value.dtype.kind in "OU" else value.tobytes(order="C"); return (_replacement_dtype_signature(value.dtype if dtype is None else dtype), value.shape, payload)
def _replacement_json_node(parent, path, role, *, required=True):
    node = _replacement_hard_group(parent, path, h5py.Dataset)
    if node is None: return None if not required else (_ for _ in ()).throw(ValueError(f"{role} is absent or not a local scalar"))
    raw = _replacement_utf8_scalar(node, role); value = json.loads(raw)
    (None if raw == json.dumps(value, sort_keys=True, separators=(",", ":")) else (_ for _ in ()).throw(ValueError(f"{role} is noncanonical"))); return value
def _decode_replacement_fact(handle: h5py.File, label: int, *, entry: str = "entry", rows: Mapping[str, Mapping[int, int]] | None = None, context: tuple[str, Mapping[str, Any] | None, Mapping[str, Any]] | None = None, metadata_keys: tuple[str, ...] = (), include_geometry: bool = False) -> Mapping[str, Any]:
    label, group = int(label), _replacement_hard_group(handle, entry); frames = _replacement_hard_group(group, "frames"); frame_name = f"frame_{label:04d}"; frame = _replacement_hard_group(frames, frame_name); source = _replacement_hard_group(frame, "source"); path_node = _replacement_hard_group(source, "path", h5py.Dataset); index_node = _replacement_hard_group(source, "frame_index", h5py.Dataset)
    try: source_class = _replacement_utf8_attribute(source, "NX_class", "replacement source class", max_bytes=_MAX_REPLACEMENT_PATH_UTF8_BYTES)
    except (AttributeError, TypeError, ValueError): source_class = None
    if not isinstance(source, h5py.Group) or set(source) != {"path", "frame_index"} or set(source.attrs) - {"NX_class", *(attr for _key, attr in _SOURCE_SNAPSHOT_ATTRIBUTES)} or source_class != "NXcollection" or type(source.get("path", getlink=True)) is not h5py.HardLink or type(source.get("frame_index", getlink=True)) is not h5py.HardLink or not isinstance(path_node, h5py.Dataset) or path_node.shape != () or path_node.maxshape != () or path_node.chunks is not None or path_node.compression is not None or len(path_node.attrs) != 0 or (encoding := h5py.check_string_dtype(path_node.dtype)) is None or encoding.encoding != "utf-8" or not isinstance(index_node, h5py.Dataset) or index_node.shape != () or index_node.maxshape != () or index_node.chunks is not None or index_node.compression is not None or len(index_node.attrs) != 0 or index_node.dtype.kind not in "iu": raise WriterStateError(f"replacement frame {label} has no exact source fact")
    path, index = _replacement_utf8_scalar(path_node, "replacement source path", max_bytes=_MAX_REPLACEMENT_PATH_UTF8_BYTES), _replacement_scalar(index_node[()], "replacement source index")
    if type(path) is not str or not path or type(index) is not int or index < 0: raise WriterStateError(f"replacement frame {label} source fact is malformed")
    snapshot = {}
    for key, attr in _SOURCE_SNAPSHOT_ATTRIBUTES:
        value = None if attr not in source.attrs else _read_replacement_source_attribute(source, attr, key)
        if value is not None and (key in {"size", "mtime_ns", "frame_count"} and (type(value) is not int or value < 0) or key == "self_contained" and type(value) is not bool or key not in {"size", "mtime_ns", "frame_count", "self_contained"} and type(value) is not str): raise WriterStateError(f"replacement source {attr} is malformed")
        snapshot[key] = value
    def indexed(name: str, columns: tuple[str, ...]) -> Mapping[str, Any]:
        if not columns: return MappingProxyType({})
        table = _replacement_hard_group(group, name)
        if not isinstance(table, h5py.Group) or _replacement_hard_group(table, "frame_index", h5py.Dataset) is None: return MappingProxyType({})
        if rows is None:
            labels = tuple(int(value) for value in _read_replacement_frame_index(_replacement_hard_group(table, "frame_index", h5py.Dataset), f"replacement {name}/frame_index"))
            (None if labels.count(label) == 1 and len(labels) == len(set(labels)) else (_ for _ in ()).throw(WriterStateError(f"replacement {name} inventory is malformed"))); row = labels.index(label)
        else:
            row = rows.get(name, {}).get(label); (None if row is not None and int(table["frame_index"][row]) == label else (_ for _ in ()).throw(WriterStateError(f"replacement {name} cursor changed")))
        values = {}
        for column in columns:
            shown = str(column)
            if shown in table:
                selected = shown
            else:
                matches = tuple(
                    name for name in table
                    if name != "frame_index" and str(name).casefold() == shown.casefold()
                )
                if len(matches) > 1:
                    raise WriterStateError(
                        f"replacement {name}/{shown} is ambiguous"
                    )
                selected = None if not matches else matches[0]
            node = None if selected is None else _replacement_hard_group(
                table, selected, h5py.Dataset,
            )
            if not isinstance(node, h5py.Dataset): raise WriterStateError(f"replacement {name}/{column} is absent")
            info = h5py.check_string_dtype(node.dtype)
            if info is None and (h5py.check_vlen_dtype(node.dtype) is not None
                                 or node.dtype.kind == "O"):
                raise WriterStateError(
                    f"replacement {column} has an unsupported vlen schema"
                )
            if info is not None and info.length is not None and info.length > _MAX_REPLACEMENT_CONFIG_UTF8_BYTES:
                raise WriterStateError(f"replacement {column} exceeds the persisted UTF-8 byte ceiling")
            try: value = (_read_replacement_utf8_element(node, int(row), f"replacement {column}") if info is not None and info.length is None else node[row])
            except (IndexError, TypeError, ValueError) as error: raise WriterStateError(f"replacement {name} row is malformed") from error
            if np.asarray(value).shape == (): values[str(selected)] = _replacement_scalar(value, f"replacement {column}")
        return MappingProxyType(values)
    if context is None:
        config = _replacement_hard_group(handle, f"{entry}/reduction/config"); node = _replacement_hard_group(config, "source_execution", h5py.Dataset)
        try: execution = json.loads(_replacement_utf8_scalar(node, "source_execution", max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES)); source_base, _lineage_bytes, lineage = decode_replacement_lineage(handle, entry=entry)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error: raise WriterStateError("replacement source context is absent or malformed") from error
        context = (source_base, lineage, _validate_replacement_execution(execution))
    source_base, lineage, execution = context
    try: background = read_background_dependency(frame)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error: raise WriterStateError("replacement Background fact is malformed") from error
    return MappingProxyType({"label": label, "path": path, "frame_index": index,
        "snapshot": MappingProxyType(snapshot), "source_base": source_base,
        "source_execution": execution, "append_lineage": lineage,
        "metadata": indexed("scan_data", metadata_keys), "geometry": indexed("per_frame_geometry", ("rot1", "rot2", "rot3", "incident_angle") if include_geometry else ()),
        "background_dependency": background})


@dataclass(frozen=True, slots=True)
class _ExpectedIndexedRow:
    group_name: str
    label: int
    values: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _DurableModeProof:
    """Small selector plus digest; never a duplicate ndarray owner."""

    group_name: str
    label: int
    row: int
    dimension: str
    mode: str
    unit: str
    azimuthal_unit: str | None
    two_d_kind: str | None
    source_shape: tuple[int, ...]
    sigma_expected: bool
    digest: str


@dataclass(frozen=True, slots=True)
class _FastArrayProof:
    """Shape plus value digest; never retains a scientific ndarray."""

    shape: tuple[int, ...]
    digest: str


@dataclass(slots=True)
class _FastModeGroupProof:
    """Compact scientific identity for one append-only finite result mode."""

    group_name: str
    dimension: str
    mode: str
    radial: _FastArrayProof
    azimuthal: _FastArrayProof | None
    unit: str
    axis_kind: str | None
    azimuthal_unit: str | None
    two_d_kind: str | None
    row_shape: tuple[int, ...]
    row_count: int
    sigma_presence: bytearray
    row_digest: Any


@dataclass(frozen=True, slots=True)
class _FastModeRowKey:
    """Ephemeral selector used only for one bounded terminal read."""

    group_name: str
    label: int
    row: int
    dimension: str


@dataclass(frozen=True, slots=True)
class _CloseModeGroupContext:
    group_name: str
    group_path: str
    top_path: str
    dimension: str
    primary_mode: str
    frame_index_shape: tuple[int, ...]
    frame_index_dtype: np.dtype
    intensity_shape: tuple[int, ...]
    intensity_dtype: np.dtype
    sigma_shape: tuple[int, ...] | None
    sigma_dtype: np.dtype | None
    radial: np.ndarray
    unit: str
    axis_kind: str | None
    azimuthal: np.ndarray | None
    azimuthal_unit: str | None
    two_d_kind: str | None
    sigma_present: bool


@dataclass(frozen=True, slots=True)
class _CloseModeObservation:
    context: _CloseModeGroupContext
    start: int
    frame_indices: np.ndarray
    intensities: np.ndarray
    sigmas: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _DurableAbsenceProof:
    """Small selector proving one publication row remains absent."""

    group_name: str
    label: int


@dataclass(frozen=True, slots=True)
class _DurableFrameProof:
    """Compact digest of a complete frame provenance row."""

    label: int
    digest: str


@dataclass(frozen=True, slots=True)
class _DurableAverageFrameProof:
    """Detached scalar proof for frame 1 plus its bounded count-map evidence."""
    label: int
    digest: str
    count_digest: str
    evidence: Any
class _EvidenceBuilder:
    def __init__(self, *, observed_only: bool = False) -> None:
        self._observed_only = bool(observed_only)
        self._digest = hashlib.sha256(b"xrd-tools-row-evidence-v1\0")
        self._observed_digest = hashlib.sha256(
            b"xrd-tools-row-observed-v1\0"
        )
        self.read_bytes = 0

    @staticmethod
    def _update(digest, role: str, side: str, payload: bytes) -> None:
        header = json.dumps(
            [str(role), str(side), len(payload)],
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(payload)

    def _part(self, role: str, side: str, payload: bytes) -> None:
        self._update(self._digest, role, side, payload)

    def _observed_part(self, role: str, side: str, payload: bytes) -> None:
        self._update(self._observed_digest, role, side, payload)

    def text(self, role: str, expected: Any, observed: Any) -> None:
        expected_text = str(expected)
        observed_text = str(observed)
        if observed_text != expected_text:
            raise WriterStateError(
                f"durability readback mismatch for {role}: "
                f"{observed_text!r} != {expected_text!r}"
            )
        expected_raw = expected_text.encode("utf-8")
        observed_raw = observed_text.encode("utf-8")
        if not self._observed_only:
            self._part(role, "expected", expected_raw)
            self._part(role, "observed", observed_raw)
        self._observed_part(role, "observed", observed_raw)
        self.read_bytes += len(observed_raw)

    def array(self, role: str, expected: Any, observed: Any) -> None:
        expected_array = np.asarray(expected)
        observed_array = np.asarray(observed)
        if expected_array.dtype != observed_array.dtype:
            raise WriterStateError(
                f"durability readback dtype mismatch for {role}: "
                f"{observed_array.dtype} != {expected_array.dtype}"
            )
        if expected_array.shape != observed_array.shape or not np.array_equal(
            expected_array,
            observed_array,
            equal_nan=True,
        ):
            raise WriterStateError(
                f"durability readback value/shape mismatch for {role}"
            )
        facts = json.dumps(
            [expected_array.dtype.str, list(expected_array.shape)],
            separators=(",", ":"),
        ).encode("utf-8")
        observed_raw = observed_array.tobytes(order="C")
        if not self._observed_only:
            self._part(role, "facts", facts)
            self._part(role, "expected", expected_array.tobytes(order="C"))
            self._part(role, "observed", observed_raw)
        self._observed_part(role, "facts", facts)
        self._observed_part(role, "observed", observed_raw)
        self.read_bytes += int(observed_array.nbytes)

    def absent(self, role: str, absent: bool) -> None:
        if not absent:
            raise WriterStateError(f"durability readback expected absent {role}")
        if not self._observed_only:
            self._part(role, "expected", b"absent")
            self._part(role, "observed", b"absent")
        self._observed_part(role, "observed", b"absent")

    def hexdigest(self) -> str:
        return (
            self._observed_digest.hexdigest()
            if self._observed_only
            else self._digest.hexdigest()
        )

    def observed_hexdigest(self) -> str:
        return self._observed_digest.hexdigest()


class WriterStateError(RuntimeError):
    pass


class ReplacementManifestTargetChanged(WriterStateError):
    """Manifest preparation lost the exact named source object."""


# Headless vNext replacement supports at most one million persisted frame rows.
# This value is intentionally independent of GUI admission modules.
_MAX_REPLACEMENT_FRAME_ROWS = 1_000_000


def _read_replacement_frame_index(
    node: h5py.Dataset | None, role: str, *, require_nonempty: bool = False,
) -> np.ndarray:
    length = None if not isinstance(node, h5py.Dataset) or node.ndim != 1 else int(node.shape[0])
    if (not isinstance(node, h5py.Dataset) or node.ndim != 1
            or node.dtype != np.dtype(np.int64) or length is None
            or length > _MAX_REPLACEMENT_FRAME_ROWS
            or require_nonempty and length < 1):
        raise WriterStateError(f"{role} is not an exact bounded int64 vector")
    return np.asarray(node[()])


def _read_replacement_utf8_element(
    node: h5py.Dataset, row: int, role: str,
) -> str:
    info = None if not isinstance(node, h5py.Dataset) else h5py.check_string_dtype(node.dtype)
    try:
        local = type(node.parent.get(node.name.rsplit("/", 1)[-1], getlink=True)) is h5py.HardLink
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        local = False
    if (not isinstance(node, h5py.Dataset) or not local or node.ndim != 1
            or not 0 <= int(row) < int(node.shape[0]) or info is None
            or info.encoding != "utf-8" or info.length is not None):
        raise WriterStateError(f"{role} is not a local UTF-8 vector element")
    ceiling = _MAX_REPLACEMENT_CONFIG_UTF8_BYTES
    capacity = min(64 << 10, ceiling + 1)
    while True:
        destination = np.empty((), dtype=f"S{capacity}")
        try:
            node.read_direct(destination, source_sel=np.s_[int(row)])
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise WriterStateError(f"{role} could not be read as bounded UTF-8") from error
        raw = bytes(destination[()])
        if len(raw) < capacity:
            break
        if capacity == ceiling + 1:
            raise WriterStateError(f"{role} exceeds the persisted UTF-8 byte ceiling")
        capacity = min(capacity * 2, ceiling + 1)
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise WriterStateError(f"{role} is not UTF-8") from error


def _read_replacement_source_attribute(
    source: h5py.Group, attr_name: str, key: str,
) -> Any:
    role = f"source {attr_name}"
    try:
        attribute = source.attrs.get_id(attr_name)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise WriterStateError(f"{role} is malformed") from error
    if attribute.shape != ():
        raise WriterStateError(f"{role} is not scalar")
    if key in {"adapter_id", "dataset_path"}:
        info = h5py.check_string_dtype(attribute.dtype)
        if info is None or info.encoding != "utf-8":
            raise WriterStateError(f"{role} is not UTF-8")
        destination = np.empty((), dtype=f"S{_MAX_REPLACEMENT_PATH_UTF8_BYTES + 1}")
        try:
            attribute.read(destination)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise WriterStateError(f"{role} could not be read as bounded UTF-8") from error
        raw = bytes(destination[()])
        if len(raw) > _MAX_REPLACEMENT_PATH_UTF8_BYTES:
            raise WriterStateError(f"{role} exceeds the persisted path byte ceiling")
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise WriterStateError(f"{role} is not UTF-8") from error
    expected = np.dtype(bool) if key == "self_contained" else None
    if ((expected is not None and attribute.dtype != expected)
            or expected is None and (attribute.dtype.kind not in "iu"
                                     or attribute.dtype.itemsize > 8)):
        raise WriterStateError(f"{role} has an invalid scalar dtype")
    destination = np.empty((), dtype=attribute.dtype)
    try:
        attribute.read(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise WriterStateError(f"{role} could not be read") from error
    return _replacement_scalar(destination[()], role)


def _read_replacement_attribute_value(
    owner: h5py.Group | h5py.Dataset, name: str, role: str,
) -> Any:
    try:
        attribute = owner.attrs.get_id(name)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise WriterStateError(f"{role} is malformed") from error
    info = h5py.check_string_dtype(attribute.dtype)
    if info is not None and info.length is None:
        if info.encoding != "utf-8":
            raise WriterStateError(f"{role} has an unsupported vlen schema")
        if attribute.shape != ():
            cardinality = 1
            for extent in attribute.shape:
                cardinality *= int(extent)
            width = _MAX_REPLACEMENT_PATH_UTF8_BYTES + 1
            if cardinality * width > _MAX_REPLACEMENT_CONFIG_UTF8_BYTES:
                raise WriterStateError(f"{role} exceeds the persisted attribute byte ceiling")
            destination = np.empty(attribute.shape, dtype=f"S{width}")
            try:
                attribute.read(destination)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise WriterStateError(f"{role} could not be read as bounded UTF-8") from error
            raw_values = tuple(bytes(value) for value in destination.ravel())
            if any(len(value) >= width for value in raw_values):
                raise WriterStateError(f"{role} exceeds the persisted UTF-8 element ceiling")
            try:
                decoded = tuple(value.decode("utf-8", errors="strict") for value in raw_values)
            except UnicodeDecodeError as error:
                raise WriterStateError(f"{role} is not UTF-8") from error
            return np.asarray(decoded, dtype=object).reshape(attribute.shape)
        ceiling = _MAX_REPLACEMENT_CONFIG_UTF8_BYTES
        capacity = min(64 << 10, ceiling + 1)
        while True:
            destination = np.empty((), dtype=f"S{capacity}")
            try:
                attribute.read(destination)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise WriterStateError(f"{role} could not be read as bounded UTF-8") from error
            raw = bytes(destination[()])
            if len(raw) < capacity:
                break
            if capacity == ceiling + 1:
                raise WriterStateError(f"{role} exceeds the persisted UTF-8 byte ceiling")
            capacity = min(capacity * 2, ceiling + 1)
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise WriterStateError(f"{role} is not UTF-8") from error
    if h5py.check_vlen_dtype(attribute.dtype) is not None or attribute.dtype.kind == "O":
        raise WriterStateError(f"{role} has an unsupported vlen schema")
    cardinality = 1
    for extent in attribute.shape:
        cardinality *= int(extent)
    if cardinality * int(attribute.dtype.itemsize) > _MAX_REPLACEMENT_CONFIG_UTF8_BYTES:
        raise WriterStateError(f"{role} exceeds the persisted attribute byte ceiling")
    destination = np.empty(attribute.shape, dtype=attribute.dtype)
    try:
        attribute.read(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise WriterStateError(f"{role} could not be read") from error
    return destination[()]


def _bounded_replacement_config_text(value: str, role: str) -> str:
    if type(value) is not str:
        raise WriterStateError(f"{role} must be exact text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise WriterStateError(f"{role} is not UTF-8") from error
    if size > _MAX_REPLACEMENT_CONFIG_UTF8_BYTES:
        raise WriterStateError(f"{role} exceeds the persisted UTF-8 byte ceiling")
    return value


_MANIFEST_FACTORY = object()
_LOWER_HEX_64 = frozenset("0123456789abcdef")


def _replacement_signature_projection(value):
    if value is None or type(value) in {str, bool, int, float}:
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is tuple:
        return {"tuple": [_replacement_signature_projection(item) for item in value]}
    if type(value) is list:
        return {"list": [_replacement_signature_projection(item) for item in value]}
    if type(value) is dict and all(type(key) is str for key in value):
        return {
            "dict": {
                key: _replacement_signature_projection(value[key])
                for key in sorted(value)
            }
        }
    raise WriterStateError(
        f"replacement signature contains unsupported {type(value).__name__}"
    )


def _replacement_signature_digest(value) -> str:
    payload = json.dumps(
        _replacement_signature_projection(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(
        b"xrd-tools-replacement-node-signature-v1\0" + payload
    ).hexdigest()


def _replacement_node_signature_value(node, *, max_bytes=None):
    if node is None:
        return None
    if not isinstance(node, h5py.Dataset):
        return ("invalid",)
    return (
        _replacement_dtype_signature(node.dtype),
        node.shape,
        node.maxshape,
        node.chunks,
        node.compression,
        node.compression_opts,
        tuple(
            (
                name,
                _replacement_value_signature(
                    _read_replacement_attribute_value(
                        node,
                        name,
                        f"replacement config {node.name}@{name}",
                    ),
                    node.attrs.get_id(name).dtype,
                ),
            )
            for name in sorted(node.attrs)
        ),
        _replacement_value_signature(
            _replacement_utf8_scalar(
                node,
                f"replacement config {node.name}",
                max_bytes=max_bytes,
            ),
            node.dtype,
        ),
    )


_FINITE_RESULT_HASH_SLAB_BYTES = 4 << 20
_FINITE_RESULT_MAX_NODES = 128
_FINITE_RESULT_SEAL_SCHEMA = "xrd_tools.finite_replacement_result_seal"
_REPLACEMENT_MANIFEST_MAX_NODES = 262_144
_REPLACEMENT_MANIFEST_MAX_VLEN_ELEMENTS = 262_144
_REPLACEMENT_MANIFEST_MAX_STRING_BYTES = 8 << 20


def _finite_result_dataset_blocks(
    dataset: h5py.Dataset,
    *,
    max_bytes: int = _FINITE_RESULT_HASH_SLAB_BYTES,
):
    if (
        not isinstance(dataset, h5py.Dataset)
        or type(max_bytes) is not int
        or max_bytes < 1
        or dataset.dtype.hasobject
        or dataset.dtype.itemsize < 1
    ):
        raise WriterStateError("finite result dataset is not bounded numeric data")
    if dataset.ndim == 0:
        value = np.asarray(dataset[()])
        if value.nbytes > max_bytes:
            raise WriterStateError("finite result scalar exceeds its hash slab")
        yield value.tobytes(order="C")
        return
    if any(type(value) is not int or value < 0 for value in dataset.shape):
        raise WriterStateError("finite result dataset shape is invalid")
    if any(value == 0 for value in dataset.shape):
        return
    maximum_elements = max(1, max_bytes // dataset.dtype.itemsize)
    split = dataset.ndim - 1
    trailing = 1
    while split > 0 and trailing * dataset.shape[split] <= maximum_elements:
        trailing *= dataset.shape[split]
        split -= 1
    step = max(1, maximum_elements // trailing)
    prefixes = np.ndindex(dataset.shape[:split]) if split else ((),)
    for prefix in prefixes:
        for start in range(0, dataset.shape[split], step):
            selection = (
                *prefix,
                slice(start, min(start + step, dataset.shape[split])),
                *(slice(None) for _ in dataset.shape[split + 1:]),
            )
            value = np.asarray(dataset[selection])
            if value.nbytes > max_bytes:
                raise WriterStateError("finite result hash slab exceeded its bound")
            yield value.tobytes(order="C")


def _finite_selected_result_digest(
    document: h5py.File,
    entry: str,
    dimension: str,
) -> str:
    if dimension not in {"1d", "2d"}:
        raise WriterStateError("finite result dimension is invalid")
    root = _replacement_hard_group(
        document, f"{entry}/integrated_{dimension}", h5py.Group,
    )
    if not isinstance(root, h5py.Group):
        raise WriterStateError("finite selected result is absent or nonlocal")
    digest = hashlib.sha256(b"xrd-tools-finite-selected-result-v1\0")
    nodes = 0

    def update(role: str, value) -> None:
        payload = json.dumps(
            _replacement_signature_projection(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
        digest.update(
            len(role.encode("utf-8")).to_bytes(8, "big")
            + role.encode("utf-8")
            + len(payload).to_bytes(8, "big")
            + payload
        )

    def walk(group: h5py.Group, role: str, active=()) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _FINITE_RESULT_MAX_NODES:
            raise WriterStateError("finite selected result exceeds its node bound")
        update(role, ("group", group.name))
        for name in sorted(group.attrs):
            update(
                f"{role}@{name}",
                _replacement_value_signature(
                    _read_replacement_attribute_value(
                        group, name, f"finite result {group.name}@{name}",
                    ),
                    group.attrs.get_id(name).dtype,
                ),
            )
        for name in sorted(group):
            link = group.get(name, getlink=True)
            child = group.get(name)
            child_role = f"{role}/{name}"
            if type(link) is not h5py.HardLink:
                raise WriterStateError("finite selected result contains a nonlocal link")
            if isinstance(child, h5py.Group):
                if any(child.id == owner for owner in (*active, group.id)):
                    raise WriterStateError("finite selected result contains a hard cycle")
                walk(child, child_role, (*active, group.id))
                continue
            if not isinstance(child, h5py.Dataset):
                raise WriterStateError("finite selected result contains an invalid node")
            nodes += 1
            if nodes > _FINITE_RESULT_MAX_NODES:
                raise WriterStateError("finite selected result exceeds its node bound")
            update(
                child_role,
                (
                    "dataset",
                    _replacement_dtype_signature(child.dtype),
                    child.shape,
                    child.maxshape,
                    child.chunks,
                    child.compression,
                    child.compression_opts,
                    child.shuffle,
                    child.fletcher32,
                    child.scaleoffset,
                ),
            )
            for attr in sorted(child.attrs):
                update(
                    f"{child_role}@{attr}",
                    _replacement_value_signature(
                        _read_replacement_attribute_value(
                            child, attr, f"finite result {child.name}@{attr}",
                        ),
                        child.attrs.get_id(attr).dtype,
                    ),
                )
            for block in _finite_result_dataset_blocks(child):
                digest.update(len(block).to_bytes(8, "big") + block)

    walk(root, f"integrated_{dimension}")
    return digest.hexdigest()


def require_finite_replacement_result_seal(
    document: h5py.File,
    *,
    entry: str,
    dimension: str,
    audit_identity: str,
) -> str:
    if (
        type(document) is not h5py.File
        or not document.id.valid
        or dimension not in {"1d", "2d"}
        or type(audit_identity) is not str
        or len(audit_identity) != 64
        or not set(audit_identity) <= _LOWER_HEX_64
    ):
        raise WriterStateError("finite result seal inputs are invalid")
    config = _replacement_hard_group(document, f"{entry}/reduction/config")
    name = f"dimension_replacement_{dimension}_result_seal"
    node = _replacement_hard_group(config, name, h5py.Dataset)
    if not isinstance(node, h5py.Dataset):
        raise WriterStateError("finite result seal is absent or nonlocal")
    raw = _replacement_utf8_scalar(
        node, "finite replacement result seal",
        max_bytes=_MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise WriterStateError("finite result seal is malformed") from error
    if (
        type(value) is not dict
        or set(value) != {
            "schema", "version", "dimension", "audit_identity",
            "result_sha256",
        }
        or value["schema"] != _FINITE_RESULT_SEAL_SCHEMA
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["dimension"] != dimension
        or value["audit_identity"] != audit_identity
        or type(value["result_sha256"]) is not str
        or len(value["result_sha256"]) != 64
        or not set(value["result_sha256"]) <= _LOWER_HEX_64
        or raw != json.dumps(value, sort_keys=True, separators=(",", ":"))
    ):
        raise WriterStateError("finite result seal is noncanonical")
    observed = _finite_selected_result_digest(document, entry, dimension)
    if observed != value["result_sha256"]:
        raise WriterStateError("finite selected result changed after sealing")
    return observed


def _replacement_exclusion_paths(
    entry_name: str,
    dimension: str,
    *,
    include_finite_lineage: bool,
) -> tuple[str, ...]:
    if (
        type(entry_name) is not str
        or not entry_name
        or dimension not in {"1d", "2d"}
    ):
        raise WriterStateError("replacement exclusion domain is invalid")
    root = "/" + "/".join(
        part for part in entry_name.split("/") if part
    )
    if root == "/":
        raise WriterStateError("replacement exclusion entry is invalid")
    values = (
        f"{root}/integrated_{dimension}",
        f"{root}/reduction/config/bai_{dimension}_args",
        f"{root}/reduction/config/gi_config",
        f"{root}/reduction/config/dimension_replacement_{dimension}",
        f"{root}/reduction/config/source_execution",
        f"{root}/reduction/config/append_lineage",
    )
    if include_finite_lineage:
        values += (
            finite_lineage_hdf_path(root.lstrip("/")),
            f"{root}/reduction/config/"
            f"dimension_replacement_{dimension}_result_seal",
        )
    return values


def _replacement_exclusions_for(
    entry: h5py.Group,
    dimension: str,
    *,
    include_finite_lineage: bool,
) -> tuple[str, ...]:
    if not isinstance(entry, h5py.Group):
        raise WriterStateError("replacement exclusion domain is invalid")
    return _replacement_exclusion_paths(
        entry.name,
        dimension,
        include_finite_lineage=include_finite_lineage,
    )


def _replacement_manifest_digest_for(
    root: h5py.File,
    entry_name: str,
    exclude: tuple[str, ...],
    *,
    ignore_source_base: bool,
    ignore_file_name: bool,
) -> str:
    digest = hashlib.sha256(b"xrd-tools-replacement-manifest-v2\0")
    config = _replacement_hard_group(root, f"{entry_name}/reduction/config")
    config_prefix = "" if config is None else f"{config.name.rstrip('/')}/"
    entry_path = "/" + "/".join(
        part for part in entry_name.split("/") if part
    )

    nodes = 1
    first_roles: dict[int, str] = {}

    def update(role, value):
        payload = value if isinstance(value, bytes) else repr(value).encode()
        encoded_role = role.encode("utf-8", errors="strict")
        digest.update(
            len(encoded_role).to_bytes(8, "big")
            + encoded_role
            + len(payload).to_bytes(8, "big")
            + payload
        )

    def string_value(dataset, index, role):
        info = h5py.check_string_dtype(dataset.dtype)
        if info is None or info.length is not None:
            raise WriterStateError(
                f"replacement manifest {role} has unsupported object data"
            )
        capacity = min(64 << 10, _REPLACEMENT_MANIFEST_MAX_STRING_BYTES + 1)
        while True:
            destination = np.empty((), dtype=f"S{capacity}")
            try:
                dataset.read_direct(destination, source_sel=index)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise WriterStateError(
                    f"replacement manifest {role} string could not be read"
                ) from error
            raw = bytes(destination[()])
            if len(raw) < capacity:
                break
            if capacity == _REPLACEMENT_MANIFEST_MAX_STRING_BYTES + 1:
                raise WriterStateError(
                    f"replacement manifest {role} string exceeds its byte bound"
                )
            capacity = min(
                capacity * 2, _REPLACEMENT_MANIFEST_MAX_STRING_BYTES + 1,
            )
        if info.encoding == "utf-8":
            try:
                raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise WriterStateError(
                    f"replacement manifest {role} string is not UTF-8"
                ) from error
        return raw

    def hash_dataset(dataset, role):
        if not dataset.dtype.hasobject:
            for block in _finite_result_dataset_blocks(dataset):
                update(f"{role}@data", block)
            return
        if h5py.check_string_dtype(dataset.dtype) is None:
            raise WriterStateError(
                f"replacement manifest {role} has unsupported variable data"
            )
        if dataset.size > _REPLACEMENT_MANIFEST_MAX_VLEN_ELEMENTS:
            raise WriterStateError(
                f"replacement manifest {role} exceeds its string cardinality bound"
            )
        indices = ((),) if dataset.ndim == 0 else np.ndindex(dataset.shape)
        for index in indices:
            update(f"{role}@data", string_value(dataset, index, role))

    def walk(group, prefix, active=()):
        nonlocal nodes
        for name in sorted(group.attrs):
            if ignore_source_base and group.name == entry_path and name == SOURCE_BASE_ATTR:
                continue
            if ignore_file_name and group.name == "/" and name == "file_name":
                continue
            update(
                f"{prefix}@{name}",
                _replacement_value_signature(
                    _read_replacement_attribute_value(
                        group,
                        name,
                        f"replacement manifest {group.name}@{name}",
                    ),
                    group.attrs.get_id(name).dtype,
                ),
            )
        for name in sorted(group):
            path = f"{group.name.rstrip('/')}/{name}"
            if path in exclude:
                continue
            link = group.get(name, getlink=True)
            role = f"{prefix}/{name}"
            update(
                f"{role}@link",
                (
                    type(link).__name__,
                    getattr(link, "filename", None),
                    getattr(link, "path", None),
                ),
            )
            if type(link) is not h5py.HardLink:
                continue
            child = group.get(name)
            if isinstance(child, h5py.Group):
                nodes += 1
                if nodes > _REPLACEMENT_MANIFEST_MAX_NODES:
                    raise WriterStateError(
                        "replacement manifest exceeds its node bound"
                    )
                update(role, b"group")
                if any(child.id == owner for owner in (*active, group.id)):
                    raise WriterStateError(
                        "replacement manifest contains a hard cycle"
                    )
                address = int(h5py.h5o.get_info(child.id).addr)
                prior_role = first_roles.get(address)
                if prior_role is not None:
                    update(f"{role}@alias", prior_role)
                    continue
                first_roles[address] = role
                walk(child, role, (*active, group.id))
            elif isinstance(child, h5py.Dataset):
                nodes += 1
                if nodes > _REPLACEMENT_MANIFEST_MAX_NODES:
                    raise WriterStateError(
                        "replacement manifest exceeds its node bound"
                    )
                address = int(h5py.h5o.get_info(child.id).addr)
                prior_role = first_roles.get(address)
                if prior_role is not None:
                    update(f"{role}@alias", prior_role)
                    continue
                first_roles[address] = role
                update(
                    f"{role}@layout",
                    (
                        _replacement_dtype_signature(child.dtype),
                        child.shape,
                        child.maxshape,
                        child.chunks,
                        child.compression,
                        child.compression_opts,
                        child.shuffle,
                        child.fletcher32,
                        child.scaleoffset,
                        tuple(child.external or ()),
                        bool(child.is_virtual),
                    ),
                )
                hash_dataset(child, role)
                for attr in sorted(child.attrs):
                    update(
                        f"{role}@{attr}",
                        _replacement_value_signature(
                            _read_replacement_attribute_value(
                                child,
                                attr,
                                f"replacement manifest {child.name}@{attr}",
                            ),
                            child.attrs.get_id(attr).dtype,
                        ),
                    )
            else:
                raise WriterStateError(
                    f"unsupported replacement manifest node {role}"
                )

    walk(root, "file")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalNodeSignatureDigest:
    schema: str
    version: int
    present: bool
    sha256: str | None
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _MANIFEST_FACTORY
            or self.schema != "xrd_tools.canonical_node_signature_digest"
            or type(self.version) is not int
            or self.version != 1
            or type(self.present) is not bool
            or self.present
            and (
                type(self.sha256) is not str
                or len(self.sha256) != 64
                or not set(self.sha256) <= _LOWER_HEX_64
            )
            or not self.present
            and self.sha256 is not None
        ):
            raise TypeError("node signature digest is not factory-owned")


def _canonical_signature_value(value):
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [_canonical_signature_value(item) for item in value],
        }
    if type(value) is list:
        return {
            "kind": "list",
            "items": [_canonical_signature_value(item) for item in value],
        }
    if type(value) in {str, int, float, bool} or value is None:
        if type(value) is float and not math.isfinite(value):
            raise WriterStateError("replacement signature float is non-finite")
        return {"kind": type(value).__name__, "value": value}
    raise WriterStateError(
        f"replacement signature contains unsupported {type(value).__name__}"
    )


def _canonical_node_signature_digest(value) -> CanonicalNodeSignatureDigest:
    if value is None:
        return CanonicalNodeSignatureDigest(
            "xrd_tools.canonical_node_signature_digest",
            1,
            False,
            None,
            _MANIFEST_FACTORY,
        )
    payload = _canonical_signature_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    return CanonicalNodeSignatureDigest(
        "xrd_tools.canonical_node_signature_digest",
        1,
        True,
        hashlib.sha256(encoded).hexdigest(),
        _MANIFEST_FACTORY,
    )


def _node_digest_mapping(
    value: CanonicalNodeSignatureDigest,
) -> dict[str, Any]:
    if type(value) is not CanonicalNodeSignatureDigest:
        raise TypeError("node signature projection requires an exact receipt")
    return {
        "schema": value.schema,
        "version": value.version,
        "present": value.present,
        "sha256": value.sha256,
    }


def _admit_node_digest(value: Mapping[str, Any]) -> CanonicalNodeSignatureDigest:
    if type(value) is not dict or set(value) != {
        "schema", "version", "present", "sha256",
    }:
        raise TypeError("node signature receipt is noncanonical")
    return CanonicalNodeSignatureDigest(
        value["schema"],
        value["version"],
        value["present"],
        value["sha256"],
        _MANIFEST_FACTORY,
    )


@dataclass(frozen=True, slots=True)
class ReplacementManifestReceipt:
    schema: str
    version: int
    manifest_algorithm_version: int
    source_snapshot: FiniteFileSnapshot
    entry: str
    dimension: str
    exclusions: tuple[str, ...]
    manifest_digest: str
    gi_name: str
    gi_node_signature: CanonicalNodeSignatureDigest
    gi_structure_signature: CanonicalNodeSignatureDigest
    selected_bai_signature: CanonicalNodeSignatureDigest
    selected_audit_signature: CanonicalNodeSignatureDigest
    source_execution_signature: CanonicalNodeSignatureDigest
    append_lineage_signature: CanonicalNodeSignatureDigest
    gi_preserved_json: str
    gi_preserved_digest: str
    facts_digest: str
    receipt_digest: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        try:
            payload = _replacement_manifest_receipt_preimage(self)
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8", errors="strict")
            expected = hashlib.sha256(encoded).hexdigest()
        except (TypeError, ValueError, UnicodeError) as error:
            raise TypeError("replacement manifest receipt is invalid") from error
        node_receipts = (
            self.gi_node_signature,
            self.gi_structure_signature,
            self.selected_bai_signature,
            self.selected_audit_signature,
            self.source_execution_signature,
            self.append_lineage_signature,
        )
        digests = (
            self.manifest_digest,
            self.gi_preserved_digest,
            self.facts_digest,
            self.receipt_digest,
        )
        if (
            _claim is not _MANIFEST_FACTORY
            or self.schema != "xrd_tools.replacement_manifest_receipt"
            or type(self.version) is not int
            or self.version != 1
            or type(self.manifest_algorithm_version) is not int
            or self.manifest_algorithm_version != 2
            or type(self.source_snapshot) is not FiniteFileSnapshot
            or type(self.entry) is not str
            or not self.entry
            or self.dimension not in {"1d", "2d"}
            or type(self.exclusions) is not tuple
            or any(type(value) is not str for value in self.exclusions)
            or self.gi_name != f"gi_mode_{self.dimension}"
            or any(
                type(value) is not CanonicalNodeSignatureDigest
                for value in node_receipts
            )
            or type(self.gi_preserved_json) is not str
            or hashlib.sha256(
                self.gi_preserved_json.encode("utf-8", errors="strict")
            ).hexdigest() != self.gi_preserved_digest
            or any(
                type(value) is not str
                or len(value) != 64
                or not set(value) <= _LOWER_HEX_64
                for value in digests
            )
            or self.receipt_digest != expected
        ):
            raise TypeError("replacement manifest receipt is not factory-owned")


def _replacement_manifest_receipt_preimage(
    receipt: ReplacementManifestReceipt,
) -> dict[str, Any]:
    snapshot = receipt.source_snapshot
    return {
        "schema": receipt.schema,
        "version": receipt.version,
        "manifest_algorithm_version": receipt.manifest_algorithm_version,
        "source_snapshot": {
            name: getattr(snapshot, name)
            for name in FiniteFileSnapshot.__dataclass_fields__
        },
        "entry": receipt.entry,
        "dimension": receipt.dimension,
        "exclusions": list(receipt.exclusions),
        "manifest_digest": receipt.manifest_digest,
        "gi_name": receipt.gi_name,
        "gi_node_signature": _node_digest_mapping(
            receipt.gi_node_signature
        ),
        "gi_structure_signature": _node_digest_mapping(
            receipt.gi_structure_signature
        ),
        "selected_bai_signature": _node_digest_mapping(
            receipt.selected_bai_signature
        ),
        "selected_audit_signature": _node_digest_mapping(
            receipt.selected_audit_signature
        ),
        "source_execution_signature": _node_digest_mapping(
            receipt.source_execution_signature
        ),
        "append_lineage_signature": _node_digest_mapping(
            receipt.append_lineage_signature
        ),
        "gi_preserved_json": receipt.gi_preserved_json,
        "gi_preserved_digest": receipt.gi_preserved_digest,
        "facts_digest": receipt.facts_digest,
    }


def _require_replacement_manifest_source(
    document: h5py.File,
    source_snapshot: FiniteFileSnapshot,
) -> None:
    """Fence an open manifest document against its exact named source."""

    try:
        descriptor = document.id.get_vfd_handle()
        observed = os.fstat(descriptor)
        named = os.stat(source_snapshot.path, follow_symlinks=False)
        shown = os.path.normcase(os.path.abspath(os.fspath(document.filename)))
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise ReplacementManifestTargetChanged(
            "replacement manifest document identity is unavailable"
        ) from error
    expected_state = (
        source_snapshot.device,
        source_snapshot.inode,
        source_snapshot.mode,
        source_snapshot.size,
        source_snapshot.mtime_ns,
        source_snapshot.ctime_ns,
    )
    if (
        type(descriptor) is not int
        or shown != os.path.normcase(os.path.abspath(source_snapshot.path))
        or tuple(
            int(getattr(observed, name))
            for name in (
                "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
                "st_ctime_ns",
            )
        ) != expected_state
        or tuple(
            int(getattr(named, name))
            for name in (
                "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns",
                "st_ctime_ns",
            )
        ) != expected_state
    ):
        raise ReplacementManifestTargetChanged(
            "replacement manifest document differs from its source snapshot"
        )


def prepare_replacement_manifest_receipt(
    document: h5py.File,
    source_snapshot: FiniteFileSnapshot,
    *,
    entry: str,
    dimension: str,
    facts_digest: str,
) -> ReplacementManifestReceipt:
    if (
        type(document) is not h5py.File
        or not document.id.valid
        or document.mode != "r"
        or type(source_snapshot) is not FiniteFileSnapshot
        or type(facts_digest) is not str
        or len(facts_digest) != 64
        or not set(facts_digest) <= _LOWER_HEX_64
    ):
        raise TypeError("replacement manifest preparation requires exact inputs")
    _require_replacement_manifest_source(document, source_snapshot)
    entry_group = _replacement_hard_group(document, entry)
    config = _replacement_hard_group(entry_group, "reduction/config")
    if not isinstance(entry_group, h5py.Group) or not isinstance(config, h5py.Group):
        raise WriterStateError("replacement manifest entry/config is not local")
    exclusions = _replacement_exclusions_for(
        entry_group,
        dimension,
        include_finite_lineage=True,
    )
    gi_name = f"gi_mode_{dimension}"
    gi_node = _replacement_hard_group(config, "gi_config", h5py.Dataset)
    gi_values = _replacement_json_node(
        config,
        "gi_config",
        "replacement GI config",
        required=False,
    )
    gi_values = {} if gi_values is None else gi_values
    if type(gi_values) is not dict:
        raise WriterStateError("replacement GI config is malformed")
    signature = _replacement_node_signature_value(gi_node)
    bai_signature = _replacement_node_signature_value(
        _replacement_hard_group(
            config, f"bai_{dimension}_args", h5py.Dataset,
        ),
        max_bytes=_MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
    )
    audit_signature = _replacement_node_signature_value(
        _replacement_hard_group(
            config, f"dimension_replacement_{dimension}", h5py.Dataset,
        ),
        max_bytes=_MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
    )
    execution_signature = _replacement_node_signature_value(
        _replacement_hard_group(config, "source_execution", h5py.Dataset),
        max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES,
    )
    lineage_signature = _replacement_node_signature_value(
        _replacement_hard_group(config, "append_lineage", h5py.Dataset),
        max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES,
    )
    preserved = json.dumps(
        {key: value for key, value in gi_values.items() if key != gi_name},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    manifest_digest = _replacement_manifest_digest_for(
        document,
        entry,
        exclusions,
        ignore_source_base=True,
        ignore_file_name=True,
    )
    gi_node_receipt = _canonical_node_signature_digest(signature)
    gi_structure_receipt = _canonical_node_signature_digest(
        None if signature is None else signature[:-1]
    )
    bai_receipt = _canonical_node_signature_digest(bai_signature)
    audit_receipt = _canonical_node_signature_digest(audit_signature)
    execution_receipt = _canonical_node_signature_digest(execution_signature)
    lineage_receipt = _canonical_node_signature_digest(lineage_signature)
    preserved_digest = hashlib.sha256(
        preserved.encode("utf-8", errors="strict")
    ).hexdigest()
    payload = {
        "schema": "xrd_tools.replacement_manifest_receipt",
        "version": 1,
        "manifest_algorithm_version": 2,
        "source_snapshot": {
            name: getattr(source_snapshot, name)
            for name in FiniteFileSnapshot.__dataclass_fields__
        },
        "entry": entry,
        "dimension": dimension,
        "exclusions": list(exclusions),
        "manifest_digest": manifest_digest,
        "gi_name": gi_name,
        "gi_node_signature": _node_digest_mapping(gi_node_receipt),
        "gi_structure_signature": _node_digest_mapping(gi_structure_receipt),
        "selected_bai_signature": _node_digest_mapping(bai_receipt),
        "selected_audit_signature": _node_digest_mapping(audit_receipt),
        "source_execution_signature": _node_digest_mapping(execution_receipt),
        "append_lineage_signature": _node_digest_mapping(lineage_receipt),
        "gi_preserved_json": preserved,
        "gi_preserved_digest": preserved_digest,
        "facts_digest": facts_digest,
    }
    receipt_digest = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")).hexdigest()
    receipt = ReplacementManifestReceipt(
        "xrd_tools.replacement_manifest_receipt",
        1,
        2,
        source_snapshot,
        entry,
        dimension,
        exclusions,
        manifest_digest,
        gi_name,
        gi_node_receipt,
        gi_structure_receipt,
        bai_receipt,
        audit_receipt,
        execution_receipt,
        lineage_receipt,
        preserved,
        preserved_digest,
        facts_digest,
        receipt_digest,
        _MANIFEST_FACTORY,
    )
    _require_replacement_manifest_source(document, source_snapshot)
    return receipt


def replacement_manifest_receipt_mapping(
    receipt: ReplacementManifestReceipt,
) -> dict[str, Any]:
    """Project one authenticated receipt onto its exact JSON boundary."""

    if type(receipt) is not ReplacementManifestReceipt:
        raise TypeError("replacement manifest projection requires an exact receipt")
    return {
        **_replacement_manifest_receipt_preimage(receipt),
        "receipt_digest": receipt.receipt_digest,
    }


def admit_replacement_manifest_receipt(
    value: Mapping[str, Any],
) -> ReplacementManifestReceipt:
    """Authenticate a detached JSON receipt without touching the filesystem."""

    expected = {
        "schema", "version", "manifest_algorithm_version",
        "source_snapshot", "entry", "dimension", "exclusions",
        "manifest_digest", "gi_name", "gi_node_signature",
        "gi_structure_signature", "selected_bai_signature",
        "selected_audit_signature", "source_execution_signature",
        "append_lineage_signature", "gi_preserved_json",
        "gi_preserved_digest", "facts_digest", "receipt_digest",
    }
    if type(value) is not dict or set(value) != expected:
        raise TypeError("replacement manifest receipt has a noncanonical keyset")
    snapshot_value = value["source_snapshot"]
    snapshot_fields = set(FiniteFileSnapshot.__dataclass_fields__)
    if type(snapshot_value) is not dict or set(snapshot_value) != snapshot_fields:
        raise TypeError("replacement manifest source snapshot is noncanonical")
    exclusions = value["exclusions"]
    if type(exclusions) is not list:
        raise TypeError("replacement manifest exclusions are not a JSON array")
    if (
        value["schema"] != "xrd_tools.replacement_manifest_receipt"
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["manifest_algorithm_version"]) is not int
        or value["manifest_algorithm_version"] != 2
    ):
        raise TypeError("replacement manifest receipt schema is unsupported")
    return ReplacementManifestReceipt(
        value["schema"],
        value["version"],
        value["manifest_algorithm_version"],
        FiniteFileSnapshot(**snapshot_value),
        value["entry"],
        value["dimension"],
        tuple(exclusions),
        value["manifest_digest"],
        value["gi_name"],
        _admit_node_digest(value["gi_node_signature"]),
        _admit_node_digest(value["gi_structure_signature"]),
        _admit_node_digest(value["selected_bai_signature"]),
        _admit_node_digest(value["selected_audit_signature"]),
        _admit_node_digest(value["source_execution_signature"]),
        _admit_node_digest(value["append_lineage_signature"]),
        value["gi_preserved_json"],
        value["gi_preserved_digest"],
        value["facts_digest"],
        value["receipt_digest"],
        _MANIFEST_FACTORY,
    )


@dataclass(frozen=True, slots=True)
class PreparedManifestAdmission:
    receipt: ReplacementManifestReceipt
    facts_digest: str
    version_identity: str
    operation_identity: str
    route_identity: str
    seed_receipt_digest: str
    candidate_identity: str
    admission_digest: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        payload = {
            "candidate_identity": self.candidate_identity,
            "domain": "xrd-tools-prepared-manifest-admission-v1",
            "facts_digest": self.facts_digest,
            "operation_identity": self.operation_identity,
            "receipt_digest": self.receipt.receipt_digest,
            "route_identity": self.route_identity,
            "seed_receipt_digest": self.seed_receipt_digest,
            "version_identity": self.version_identity,
        }
        try:
            expected = hashlib.sha256(json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8", errors="strict")).hexdigest()
        except (TypeError, ValueError, UnicodeError) as error:
            raise TypeError("prepared manifest admission is invalid") from error
        digests = (
            self.facts_digest,
            self.version_identity,
            self.operation_identity,
            self.route_identity,
            self.seed_receipt_digest,
            self.candidate_identity,
            self.admission_digest,
        )
        if (
            _claim is not _MANIFEST_FACTORY
            or type(self.receipt) is not ReplacementManifestReceipt
            or self.receipt.facts_digest != self.facts_digest
            or any(
                type(value) is not str
                or len(value) != 64
                or not set(value) <= _LOWER_HEX_64
                for value in digests
            )
            or self.admission_digest != expected
        ):
            raise TypeError("prepared manifest admission is not factory-owned")


def bind_prepared_manifest_receipt(
    receipt: ReplacementManifestReceipt,
    *,
    facts_digest: str,
    request: FiniteArtifactRequest,
    seed_binding: FiniteSeedBinding,
    candidate_binding: FiniteCandidateBinding,
) -> PreparedManifestAdmission:
    """Join prepared facts to one exact publisher-owned seeded candidate."""

    if (
        type(receipt) is not ReplacementManifestReceipt
        or type(request) is not FiniteArtifactRequest
        or type(seed_binding) is not FiniteSeedBinding
        or type(candidate_binding) is not FiniteCandidateBinding
        or seed_binding.request is not request
        or not seed_binding.authorizes(candidate_binding)
        or receipt.source_snapshot != seed_binding.receipt.source_snapshot
        or receipt.source_snapshot != request.operation_context.source_snapshot
        or receipt.facts_digest != facts_digest
    ):
        raise ValueError("prepared manifest cannot bind to this finite candidate")
    values = (
        receipt,
        facts_digest,
        request.version_identity,
        request.operation_identity,
        request.operation_context.route_identity,
        seed_binding.receipt.receipt_digest,
        candidate_binding.candidate_identity,
    )
    payload = {
        "candidate_identity": values[6],
        "domain": "xrd-tools-prepared-manifest-admission-v1",
        "facts_digest": values[1],
        "operation_identity": values[3],
        "receipt_digest": receipt.receipt_digest,
        "route_identity": values[4],
        "seed_receipt_digest": values[5],
        "version_identity": values[2],
    }
    digest = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")).hexdigest()
    return PreparedManifestAdmission(
        *values, digest, _MANIFEST_FACTORY,
    )


class WriterIncomplete(RuntimeError):
    def __init__(self, message: str, outcome: WriterOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


_VECTOR_FIELDS = tuple(WriterOperationVector.__dataclass_fields__)
_SEMANTIC_READ_GROUP_SIZE = 8


class NexusRecordWriter:
    """Own one open writer, dirty cursors, receipts and terminal outcome.

    ``file_lock`` is borrowed, never manufactured here.  With no supplied lock,
    callers retain the pre-cutover single-session serialization contract.  The
    shared HDF5 read pool remains excluded for the complete open-handle lifetime.
    Cached row absence is authoritative only while that caller contract prevents
    every other writer from mutating the target between ``begin`` and terminal
    ``finish``/``abort``; C3 is responsible for mounting the C1 target lease.
    """

    def __init__(
        self,
        target: Path | str,
        *,
        entry: str = "entry",
        compression: str | None = "gzip",
        overwrite: bool = False,
        atomic: bool | None = None,
        flush_every: int | None = 16,
        complete_record: bool = True,
        source_base: Path | str | None = None,
        file_lock: Any | None = None,
        pool: Any | None = None,
        replace_attempts: int = 3,
        opener=open_nexus_writer,
        transaction_binding: WriterTransactionBinding | None = None,
        append_decision: AppendDecision | None = None,
        fast_regenerable: bool = False,
        defer_epoch_durability: bool = False,
        replacement_dimension: str | None = None,
        replacement_labels: tuple[int, ...] | None = None,
        replacement_audit: bytes | None = None,
        replacement_selected_plan: Mapping[str, Any] | None = None,
        replacement_gi_mode: str | None = None,
        replacement_source_execution: Mapping[str, Any] | None = None,
        replacement_append_lineage: bytes | None = None,
        seeded_document: h5py.File | None = None,
        finite_request: FiniteArtifactRequest | None = None,
        finite_seed_binding: FiniteSeedBinding | None = None,
        finite_candidate_binding: FiniteCandidateBinding | None = None,
        prepared_manifest_admission: PreparedManifestAdmission | None = None,
    ) -> None:
        if flush_every is not None and int(flush_every) <= 0:
            raise ValueError(f"flush_every must be > 0 or None; got {flush_every}")
        if int(replace_attempts) < 1:
            raise ValueError("replace_attempts must be >= 1")
        if type(fast_regenerable) is not bool:
            raise TypeError("fast_regenerable must be an exact bool")
        if type(defer_epoch_durability) is not bool:
            raise TypeError("defer_epoch_durability must be an exact bool")
        self.target = require_current_output_path(target)
        self.entry = str(entry)
        self.compression = compression
        self.overwrite = bool(overwrite)
        self.atomic = atomic
        self.flush_every = flush_every
        self.complete_record = bool(complete_record)
        if (source_base is not None
                and len(str(source_base).encode("utf-8"))
                    > _MAX_REPLACEMENT_PATH_UTF8_BYTES):
            raise ValueError("source_base exceeds the persisted path byte ceiling")
        self.source_base = source_base
        self.file_lock = file_lock
        self._pool = get_pool() if pool is None else pool
        self._replace_attempts = int(replace_attempts)
        self._opener = opener
        self._transaction_binding = transaction_binding
        if seeded_document is not None and (
            type(seeded_document) is not h5py.File
            or not seeded_document.id.valid
            or seeded_document.mode != "r+"
        ):
            raise TypeError("seeded replacement document must be one open h5py.File")
        if not (
            (seeded_document is None)
            == (finite_request is None)
            == (finite_seed_binding is None)
            == (finite_candidate_binding is None)
        ):
            raise ValueError(
                "seeded document, finite request, and seed binding must be supplied together"
            )
        if finite_request is not None and (
            type(finite_request) is not FiniteArtifactRequest
            or type(finite_seed_binding) is not FiniteSeedBinding
            or type(finite_candidate_binding) is not FiniteCandidateBinding
            or finite_seed_binding.request is not finite_request
            or not finite_seed_binding.authorizes(finite_candidate_binding)
            or not finite_seed_binding.active
            or replacement_dimension not in {"1d", "2d"}
            or finite_request.output_artifact != str(self.target)
            or finite_request.entry != self.entry
            or finite_request.operation_kind
            != f"reintegrate-{replacement_dimension}"
        ):
            raise ValueError("seeded replacement finite request is inconsistent")
        if prepared_manifest_admission is not None and (
            type(prepared_manifest_admission) is not PreparedManifestAdmission
            or finite_seed_binding is None
            or finite_candidate_binding is None
            or prepared_manifest_admission.receipt.source_snapshot
            != finite_seed_binding.receipt.source_snapshot
            or prepared_manifest_admission.receipt.entry != self.entry
            or prepared_manifest_admission.receipt.dimension
            != replacement_dimension
            or prepared_manifest_admission.version_identity
            != finite_request.version_identity
            or prepared_manifest_admission.operation_identity
            != finite_request.operation_identity
            or prepared_manifest_admission.route_identity
            != finite_request.operation_context.route_identity
            or prepared_manifest_admission.seed_receipt_digest
            != finite_seed_binding.receipt.receipt_digest
            or prepared_manifest_admission.candidate_identity
            != finite_candidate_binding.candidate_identity
        ):
            raise ValueError("prepared replacement manifest admission is inconsistent")
        if seeded_document is not None and (
            transaction_binding is not None
            or self.overwrite
            or atomic not in {None, False}
        ):
            raise ValueError(
                "seeded replacement cannot use a transaction, overwrite, or atomic path"
            )
        self._seeded_document = seeded_document
        self._finite_request = finite_request
        self._finite_seed_binding = finite_seed_binding
        self._finite_candidate_binding = finite_candidate_binding
        self._prepared_manifest_admission = prepared_manifest_admission
        self._prepared_manifest_receipt = (
            None
            if prepared_manifest_admission is None
            else prepared_manifest_admission.receipt
        )
        self._fast_regenerable = fast_regenerable
        self._defer_epoch_durability = defer_epoch_durability
        self._append_decision = append_decision
        replacement_values = (replacement_dimension, replacement_labels, replacement_audit, replacement_selected_plan)
        # The ordinary non-Live, non-Append Run binds a same-run intent, so a
        # fresh finite Overwrite legitimately carries an append decision that
        # records its first lineage epoch.  A successor epoch that extends an
        # already committed artifact is excluded by overwrite plus the
        # transaction's own fast_regenerable admission: NexusSink refuses an
        # Append preflight under Overwrite and hard-codes fast_regenerable=False
        # on the successor-epoch writer.  Rejecting every non-None decision here
        # instead disabled the exact route this fast path exists for.
        if fast_regenerable and (
            transaction_binding is None
            or type(transaction_binding) is not WriterTransactionBinding
            or transaction_binding.transaction.fast_regenerable is not True
            or not self.overwrite
            or defer_epoch_durability
            or seeded_document is not None
            or any(value is not None for value in replacement_values)
        ):
            raise ValueError(
                "fast regenerable writes require a fresh transaction-bound Overwrite"
            )
        replacement_owner_count = int(transaction_binding is not None) + int(
            seeded_document is not None
        )
        if any(value is not None for value in replacement_values) and (not all(value is not None for value in replacement_values) or replacement_owner_count != 1 or fast_regenerable or replacement_source_execution is None or not self.source_base): raise ValueError("selected-dimension replacement configuration is incomplete or unbound")
        if replacement_dimension is None and (replacement_source_execution is not None or replacement_append_lineage is not None): raise ValueError("replacement source context requires a selected dimension")
        if seeded_document is not None and replacement_dimension is None:
            raise ValueError("seeded document mode is replacement-only")
        if replacement_dimension is not None:
            try:
                source_execution = dict(_validate_replacement_execution(dict(replacement_source_execution)))
                source_base_text = os.fspath(self.source_base)
                if (not os.path.isabs(source_base_text)
                        or os.path.normcase(os.path.normpath(source_base_text)) != source_base_text):
                    raise ValueError
                append_bytes = None if replacement_append_lineage is None else bytes(replacement_append_lineage)
                append_value = None if append_bytes is None else json.loads(append_bytes.decode("utf-8", errors="strict"))
                if append_value is not None and (type(append_value) is not dict or append_bytes != json.dumps(append_value, sort_keys=True, separators=(",", ":")).encode() or append_value.get("source_base") != source_base_text):
                    raise ValueError
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, WriterStateError) as error:
                raise ValueError("replacement source context is malformed") from error
        self._replacement_configuration = None if replacement_dimension is None else (replacement_dimension, tuple(replacement_labels), bytes(replacement_audit), dict(replacement_selected_plan), replacement_gi_mode, source_execution, append_bytes, append_value)
        self._replacement_labels: tuple[int, ...] = ()
        self._replacement_read_context = self._replacement_manifest = self._replacement_expected = None
        if append_decision is not None and (
            self._replacement_configuration is not None or append_decision.disposition is not AppendDisposition.WRITE
        ):
            raise ValueError("writer requires a WRITE Append decision")
        if append_decision is not None and transaction_binding is None:
            raise ValueError("Append writer requires a live transaction binding")
        if append_decision is not None and seeded_document is not None:
            raise ValueError("seeded finite replacement cannot Append")
        if transaction_binding is not None and atomic:
            raise ValueError("transaction-bound writer cannot replace its owned inode")
        self.phase = WriterPhase.NEW
        self._h5 = None
        self._fast_entry_group: h5py.Group | None = None
        self._active_path: Path | None = None
        self._pool_owned = False
        self._in_boundary = False
        self._facade = None
        self._pending: dict[tuple[int, ResultMode, str], StageReceipt] = {}
        # Receipts staged since the last checkpoint.  `_pending` is drained only
        # at finish on the fast path, so a checkpoint must never walk all of it.
        self._staged_since_checkpoint: dict[
            tuple[int, ResultMode, str], StageReceipt
        ] = {}
        self._pending_publication_drops: dict[
            tuple[int, ResultMode], int
        ] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._pending_metadata_labels: set[int] = set()
        self._source_paths: list[str] = []
        self._row_cursors: dict[str, dict[int, int]] = {}
        self._prepared_stack_authority: Mapping[str, Any] | None = None
        self._primary_mode_1d = DEFAULT_MODE_KEY
        self._primary_mode_2d = DEFAULT_MODE_KEY
        self._since_flush = 0
        self._vector = {name: 0 for name in _VECTOR_FIELDS}
        self._pending_owner: str | None = None
        self._finish_step = 0
        self._finalization = WriterFinalization()
        self._fresh = False
        self._dirty_modes: dict[tuple[str, int], _ExpectedModeRow] = {}
        self._dirty_absent_modes: set[tuple[str, int]] = set()
        self._dirty_frames: dict[int, _ExpectedFrameRow] = {}
        self._dirty_indexed: dict[tuple[str, int], _ExpectedIndexedRow] = {}
        self._append_written_labels: set[int] = set()
        self._lineage_dirty = False
        self._lineage_expected: str | None = None
        self._checkpoint_rows = 0
        self._checkpoint_read_bytes = 0
        self._semantic_read_groups = {"checkpoint": 0, "close": 0}
        self._semantic_read_rows = {"checkpoint": 0, "close": 0}
        self._semantic_mode_observation = None
        self._stream_close_attempt = None
        self._stream_terminal: StreamTerminal | None = None
        self._close_verification_descriptor = None
        self._durable_mode_proofs: dict[
            tuple[str, int], _DurableModeProof
        ] = {}
        self._fast_mode_groups: dict[str, _FastModeGroupProof] = {}
        self._durable_absence_proofs: dict[
            tuple[str, int], _DurableAbsenceProof
        ] = {}
        self._durable_frame_proofs: dict[
            int, _DurableFrameProof | _DurableAverageFrameProof
        ] = {}

    @classmethod
    def for_seeded_replacement(
        cls,
        logical_target: Path | str,
        document: h5py.File,
        request: FiniteArtifactRequest,
        seed_binding: FiniteSeedBinding,
        candidate_binding: FiniteCandidateBinding,
        *,
        entry: str,
        source_base: Path | str,
        file_lock: Any | None,
        replacement_dimension: str,
        replacement_labels: tuple[int, ...],
        replacement_audit: bytes,
        replacement_selected_plan: Mapping[str, Any],
        replacement_gi_mode: str | None,
        replacement_source_execution: Mapping[str, Any],
        replacement_append_lineage: bytes | None,
        prepared_manifest_admission: PreparedManifestAdmission | None = None,
        compression: str | None = "gzip",
        flush_every: int | None = None,
    ) -> "NexusRecordWriter":
        """Own selected-dimension writes inside one publisher-owned seed."""

        return cls(
            logical_target,
            entry=entry,
            compression=compression,
            overwrite=False,
            atomic=False,
            flush_every=flush_every,
            complete_record=True,
            source_base=source_base,
            file_lock=file_lock,
            replacement_dimension=replacement_dimension,
            replacement_labels=replacement_labels,
            replacement_audit=replacement_audit,
            replacement_selected_plan=replacement_selected_plan,
            replacement_gi_mode=replacement_gi_mode,
            replacement_source_execution=replacement_source_execution,
            replacement_append_lineage=replacement_append_lineage,
            seeded_document=document,
            finite_request=request,
            finite_seed_binding=seed_binding,
            finite_candidate_binding=candidate_binding,
            prepared_manifest_admission=prepared_manifest_admission,
        )

    @property
    def active_path(self) -> Path | None:
        return self._active_path

    @property
    def fresh(self) -> bool:
        return self._fresh

    @property
    def checkpoint_read_volume(self) -> tuple[int, int]:
        return self._checkpoint_rows, self._checkpoint_read_bytes

    @property
    def grouped_semantic_read_volume(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType({
            phase: (self._semantic_read_groups[phase], self._semantic_read_rows[phase])
            for phase in ("checkpoint", "close")
        })

    @staticmethod
    def _semantic_groups(rows):
        group = []
        for row in rows:
            if group and (
                len(group) == _SEMANTIC_READ_GROUP_SIZE
                or row.group_name != group[-1].group_name
                or row.row != group[-1].row + 1
            ):
                yield tuple(group)
                group = []
            group.append(row)
        if group:
            yield tuple(group)

    def _read_grouped_mode(self, rows, phase: str, close_context=None, close_datasets=None):
        if close_context is not None:
            if close_datasets is None:
                raise WriterStateError("close mode read has no datasets")
            labels, intensity, sigma = close_datasets
            start = rows[0].row
            stop = start + len(rows)
            observed_labels = np.asarray(labels[start:stop])
            observed_intensity = np.asarray(intensity[start:stop])
            observed_sigma = None if sigma is None else np.asarray(sigma[start:stop])
            if (
                observed_labels.ndim != 1 or observed_labels.shape != (len(rows),)
                or observed_intensity.shape[:1] != (len(rows),)
                or (
                    observed_sigma is not None
                    and observed_sigma.shape[:1] != (len(rows),)
                )
            ):
                raise WriterStateError(
                    f"grouped durability read lost rows for {rows[0].group_name}"
                )
            self._semantic_read_groups[phase] += 1
            self._semantic_read_rows[phase] += len(rows)
            return _CloseModeObservation(
                close_context, start, observed_labels, observed_intensity, observed_sigma,
            )
        group = local_hard_group_path(
            self._entry_group(), rows[0].group_name, role=rows[0].group_name,
        )
        intensity = local_hard_dataset(
            group, "intensity", role=f"{rows[0].group_name}/intensity",
        ) if isinstance(group, h5py.Group) else None
        radial = local_hard_dataset(
            group, "q", role=f"{rows[0].group_name}/q",
        ) if isinstance(group, h5py.Group) else None
        azimuthal = (
            local_hard_dataset(
                group, "chi", role=f"{rows[0].group_name}/chi",
            )
            if rows[0].dimension == "2d" and isinstance(group, h5py.Group)
            else None
        )
        if not isinstance(intensity, h5py.Dataset) or not isinstance(
            radial, h5py.Dataset,
        ) or (rows[0].dimension == "2d" and not isinstance(
            azimuthal, h5py.Dataset,
        )):
            return None
        start = rows[0].row
        observed = np.asarray(intensity[start:start + len(rows)])
        if observed.shape[:1] != (len(rows),):
            raise WriterStateError(
                f"grouped durability read lost rows for {rows[0].group_name}"
            )
        self._semantic_read_groups[phase] += 1
        self._semantic_read_rows[phase] += len(rows)
        return (
            np.asarray(radial[()]), observed,
            None if azimuthal is None else np.asarray(azimuthal[()]),
        )

    def _close_mode_groups(self, proofs):
        entry = self._entry_group()
        index = 0
        while index < len(proofs):
            end = index + 1
            while end < len(proofs) and proofs[end].group_name == proofs[index].group_name:
                end += 1
            group_proofs = proofs[index:end]
            first = group_proofs[0]
            group = local_hard_group_path(
                entry, first.group_name, role=first.group_name,
            )
            if not isinstance(group, h5py.Group):
                raise WriterStateError(f"durable-row proof lost mode group {first.group_name}")
            labels = local_hard_dataset(group, "frame_index", role=f"{first.group_name}/frame_index")
            intensity = local_hard_dataset(group, "intensity", role=f"{first.group_name}/intensity")
            radial = local_hard_dataset(group, "q", role=f"{first.group_name}/q")
            sigma = local_hard_dataset(group, "sigma", role=f"{first.group_name}/sigma")
            if not all(isinstance(value, h5py.Dataset) for value in (
                labels, intensity, radial,
            )) or (sigma is not None and not isinstance(sigma, h5py.Dataset)):
                raise WriterStateError(f"durable-row proof lost arrays for {first.group_name}")
            if any(proof.dimension != first.dimension for proof in group_proofs):
                raise WriterStateError(f"durable-row proof changed dimension for {first.group_name}")
            top = local_hard_group_path(
                entry,
                f"integrated_{first.dimension}",
                role=f"integrated_{first.dimension}",
            )
            if not isinstance(top, h5py.Group):
                raise WriterStateError(f"durable-row proof lost integrated_{first.dimension}")
            chi = local_hard_dataset(
                group, "chi", role=f"{first.group_name}/chi",
            ) if first.dimension == "2d" else None
            if first.dimension == "2d" and not isinstance(chi, h5py.Dataset):
                raise WriterStateError(
                    f"durable-row proof lost chi for {first.group_name}"
                )
            context = _CloseModeGroupContext(
                first.group_name,
                group.name,
                top.name,
                first.dimension,
                self._text_value(top.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)),
                tuple(int(size) for size in labels.shape),
                labels.dtype,
                tuple(int(size) for size in intensity.shape),
                intensity.dtype,
                (
                    None
                    if sigma is None
                    else tuple(int(size) for size in sigma.shape)
                ),
                None if sigma is None else sigma.dtype,
                np.asarray(radial[()]),
                self._text_value(radial.attrs.get("units", "")),
                (
                    _axis_kind_from_group_1d(group)
                    if first.dimension == "1d" else None
                ),
                None if chi is None else np.asarray(chi[()]),
                None if chi is None else self._text_value(chi.attrs.get("units", "")),
                None if chi is None else self._text_value(
                    group.attrs.get("two_d_kind", "")
                ),
                sigma is not None,
            )
            datasets = (labels, intensity, sigma)
            for rows in self._semantic_groups(group_proofs):
                yield rows, self._read_grouped_mode(
                    rows, "close", context, datasets,
                )
            index = end

    def bind_session(self, facade) -> None:
        if self.phase is not WriterPhase.NEW:
            raise WriterStateError("session facade must be bound before begin")
        self._facade = facade

    @contextmanager
    def _boundary(self):
        with (nullcontext() if self.file_lock is None else self.file_lock):
            if self._in_boundary:
                raise WriterStateError("concurrent or re-entrant writer boundary")
            self._in_boundary = True
            try:
                yield
            finally:
                self._in_boundary = False

    def _authorize_transaction_mutation(self) -> None:
        self._invalidate_checkpoint_recovery()
        binding = self._transaction_binding
        if binding is None:
            return
        binding.transaction.authorize_stream_mutation(
            binding.attempt,
            lease=binding.lease,
        )

    def _vfd_descriptor(self) -> int:
        if self._h5 is None:
            raise WriterStateError("durability checkpoint has no open HDF5 owner")
        try:
            handle = self._h5.id.get_vfd_handle()
        except BaseException as exc:
            raise WriterStateError(
                "HDF5 driver did not expose a bindable durability descriptor"
            ) from exc
        if isinstance(handle, tuple) and len(handle) == 1:
            handle = handle[0]
        if not isinstance(handle, (int, np.integer)):
            raise WriterStateError(
                f"HDF5 VFD handle is not one descriptor: {type(handle).__name__}"
            )
        descriptor = int(handle)
        try:
            os.fstat(descriptor)
        except OSError as exc:
            raise WriterStateError("HDF5 VFD descriptor is not live") from exc
        return descriptor

    @staticmethod
    def _text_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8", errors="strict")
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    @staticmethod
    def _scan_text_value(value: Any) -> str:
        if value is None:
            return ""
        try:
            scalar = np.asarray(value)
            if scalar.shape == () and not np.isfinite(float(value)):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _remember_mode_row(
        self,
        record: RecordWrite,
        *,
        dimension: str,
    ) -> None:
        result = record.result_1d if dimension == "1d" else record.result_2d
        if result is None:
            return
        mode = str(
            (record.mode_1d if dimension == "1d" else record.mode_2d)
            or DEFAULT_MODE_KEY
        )
        group_name = self._mode_cursor_name(dimension, mode)
        label = int(record.label)
        row = self._row_cursors[group_name].get(label)
        if row is None:
            raise WriterStateError(
                f"{group_name} did not install a cursor for dirty label {label}"
            )
        radial = np.asarray(result.radial, dtype=np.float32).copy()
        if dimension == "1d":
            azimuthal = None
            intensity = np.asarray(result.intensity, dtype=np.float32).copy()
            sigma = (
                None
                if result.sigma is None
                else np.asarray(result.sigma, dtype=np.float32).copy()
            )
            azimuthal_unit = None
            two_d_kind = None
        else:
            azimuthal = np.asarray(result.azimuthal, dtype=np.float32).copy()
            intensity_source = np.asarray(result.intensity, dtype=np.float32)
            intensity = intensity_source.T.copy()
            sigma = (
                None
                if result.sigma is None
                else np.asarray(result.sigma, dtype=np.float32).T.copy()
            )
            azimuthal_unit = str(result.azimuthal_unit or "")
            two_d_kind = two_d_kind_from_units(
                result.unit,
                result.azimuthal_unit,
            ).value
        self._dirty_modes[(group_name, label)] = _ExpectedModeRow(
            group_name=group_name,
            label=label,
            row=int(row),
            dimension=dimension,
            mode=mode,
            radial=radial,
            azimuthal=azimuthal,
            intensity=intensity,
            sigma=sigma,
            unit=str(result.unit or ""),
            azimuthal_unit=azimuthal_unit,
            two_d_kind=two_d_kind,
            source_shape=tuple(np.asarray(result.intensity).shape),
        )
        self._dirty_absent_modes.discard((group_name, label))
        self._durable_absence_proofs.pop((group_name, label), None)

    @staticmethod
    def _fast_array_proof(value: Any, *, transpose: bool = False) -> _FastArrayProof:
        array = np.asarray(value, dtype=np.float32)
        if transpose:
            array = array.T
        contiguous = np.ascontiguousarray(array)
        return _FastArrayProof(
            tuple(int(size) for size in contiguous.shape),
            hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
        )

    @staticmethod
    def _update_fast_row_digest(
        digest,
        label: int,
        intensity: Any,
        sigma: Any | None,
        *,
        transpose: bool = False,
    ) -> None:
        """Append one canonical stored row to a rolling finite digest."""

        def update_array(value: Any) -> None:
            array = np.asarray(value, dtype=np.float32)
            if transpose:
                array = array.T
            array = np.ascontiguousarray(array)
            digest.update(struct.pack("<I", array.ndim))
            for size in array.shape:
                digest.update(struct.pack("<Q", int(size)))
            digest.update(memoryview(array).cast("B"))

        digest.update(b"row\0")
        digest.update(struct.pack("<q", int(label)))
        update_array(intensity)
        if sigma is None:
            digest.update(b"sigma-absent\0")
        else:
            digest.update(b"sigma-present\0")
            update_array(sigma)

    def _remember_fast_mode_proof(
        self,
        record: RecordWrite,
        *,
        dimension: str,
    ) -> None:
        """Roll one successful append into compact terminal proof state."""
        result = record.result_1d if dimension == "1d" else record.result_2d
        if result is None:
            return
        mode = str(
            (record.mode_1d if dimension == "1d" else record.mode_2d)
            or DEFAULT_MODE_KEY
        )
        group_name = self._mode_cursor_name(dimension, mode)
        label = int(record.label)
        row = self._row_cursors[group_name].get(label)
        if row is None:
            raise WriterStateError(
                f"{group_name} did not install a cursor for finite label {label}"
            )
        two_d = dimension == "2d"
        group_proof = self._fast_mode_groups.get(group_name)
        if group_proof is None:
            unit = str(result.unit or "")
            group_proof = _FastModeGroupProof(
                group_name=group_name,
                dimension=dimension,
                mode=mode,
                radial=self._fast_array_proof(result.radial),
                azimuthal=(
                    self._fast_array_proof(result.azimuthal) if two_d else None
                ),
                unit=unit,
                axis_kind=(_axis_kind_1d(unit) if not two_d else None),
                azimuthal_unit=(
                    str(result.azimuthal_unit or "") if two_d else None
                ),
                two_d_kind=(
                    two_d_kind_from_units(
                        result.unit, result.azimuthal_unit,
                    ).value
                    if two_d else None
                ),
                row_shape=tuple(int(size) for size in (
                    np.asarray(result.intensity).T.shape
                    if two_d else np.asarray(result.intensity).shape
                )),
                row_count=0,
                sigma_presence=bytearray(),
                row_digest=hashlib.sha256(),
            )
            self._fast_mode_groups[group_name] = group_proof
        elif (
            group_proof.dimension != dimension
            or group_proof.mode != mode
            or group_proof.radial.shape
            != tuple(int(size) for size in np.asarray(result.radial).shape)
            or group_proof.unit != str(result.unit or "")
            or (
                two_d
                and (
                    group_proof.azimuthal is None
                    or group_proof.azimuthal.shape
                    != tuple(
                        int(size) for size in np.asarray(result.azimuthal).shape
                    )
                    or group_proof.azimuthal_unit
                    != str(result.azimuthal_unit or "")
                    or group_proof.two_d_kind
                    != two_d_kind_from_units(
                        result.unit, result.azimuthal_unit,
                    ).value
                )
            )
            or group_proof.row_shape
            != tuple(int(size) for size in (
                np.asarray(result.intensity).T.shape
                if two_d else np.asarray(result.intensity).shape
            ))
            or (
                result.sigma is not None
                and group_proof.row_shape
                != tuple(int(size) for size in (
                    np.asarray(result.sigma).T.shape
                    if two_d else np.asarray(result.sigma).shape
                ))
            )
        ):
            raise WriterStateError(
                f"finite mode identity changed for {group_name}"
            )
        if int(row) != group_proof.row_count:
            raise WriterStateError(
                f"finite mode row order changed for {group_name}"
            )
        self._update_fast_row_digest(
            group_proof.row_digest,
            label,
            result.intensity,
            result.sigma,
            transpose=two_d,
        )
        group_proof.sigma_presence.append(int(result.sigma is not None))
        group_proof.row_count += 1
        self._dirty_absent_modes.discard((group_name, label))
        self._durable_absence_proofs.pop((group_name, label), None)

    def _expected_frame_row(
        self,
        record: RecordWrite,
        prepared_thumbnail: _PreparedThumbnail | None = None,
    ) -> _ExpectedFrameRow:
        thumbnail = None
        lut = None
        if record.thumbnail is not None:
            prepared_thumbnail = prepared_thumbnail or _prepare_thumbnail(
                record.thumbnail,
                thumbnail_mask=record.thumbnail_mask,
            )
            thumbnail = prepared_thumbnail.array
            lut = prepared_thumbnail.lut
        source_path = None
        source_frame_index = None
        if record.source_path is not None:
            source_path = str(record.source_path) if self._replacement_configuration is not None else relative_source_path(record.source_path, self.source_base)
            source_frame_index = int(record.source_frame_index)
        timestamp = None if record.timestamp is None else str(record.timestamp)
        thumbnail_mask = (
            None if prepared_thumbnail is None else prepared_thumbnail.mask
        )
        return _ExpectedFrameRow(
            int(record.label),
            thumbnail,
            lut,
            bool(record.thumbnail_mask_baked),
            bool(record.mask_baked),
            thumbnail_mask,
            source_path,
            source_frame_index,
            tuple(sorted(record.source_snapshot.items())),
            timestamp,
            (None if record.background_dependency_bytes is None else
             (record.background_dependency_bytes, record.background_dependency_fingerprint)),
        )

    def _remember_frame_row(
        self,
        record: RecordWrite,
        prepared_thumbnail: _PreparedThumbnail | None = None,
    ) -> None:
        if not self.complete_record:
            return
        self._dirty_frames[int(record.label)] = self._expected_frame_row(
            record,
            prepared_thumbnail,
        )

    def _expected_source_fact(
        self,
        record: RecordWrite,
        prepared_thumbnail: _PreparedThumbnail | None = None,
    ) -> _PersistedSourceFact:
        expected = self._expected_frame_row(record, prepared_thumbnail)
        return self._source_fact_for_expected_frame(expected)

    @staticmethod
    def _source_fact_for_expected_frame(
        expected: _ExpectedFrameRow,
    ) -> _PersistedSourceFact:
        if expected.source_path is None:
            return _PersistedSourceFact(False, None, None, ())
        supplied = dict(expected.source_snapshot)
        return _PersistedSourceFact(
            True,
            expected.source_path,
            expected.source_frame_index,
            tuple(
                (key, supplied.get(key))
                for key, _attr_name in _SOURCE_SNAPSHOT_ATTRIBUTES
            ),
        )

    @staticmethod
    def _decode_persisted_source_scalar(
        value: Any,
        *,
        kind: str,
        role: str,
    ) -> str | int | bool:
        if kind == "text":
            if type(value) in (str, np.str_):
                decoded = str(value)
            elif type(value) in (bytes, np.bytes_):
                try:
                    decoded = bytes(value).decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise WriterStateError(
                        f"{role} must be a valid UTF-8 text scalar"
                    ) from error
            else:
                raise WriterStateError(
                    f"{role} must be a text scalar, got {type(value).__name__}"
                )
            if not decoded:
                raise WriterStateError(f"{role} must not be empty")
            try:
                decoded.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise WriterStateError(
                    f"{role} must be a valid UTF-8 text scalar"
                ) from error
            return decoded
        if kind == "integer":
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral,
            ):
                raise WriterStateError(
                    f"{role} must be a non-negative integer scalar, "
                    f"got {type(value).__name__}"
                )
            decoded = int(value)
            if decoded < 0:
                raise WriterStateError(f"{role} must be non-negative")
            return decoded
        if kind == "boolean":
            if type(value) not in (bool, np.bool_):
                raise WriterStateError(
                    f"{role} must be a boolean scalar, "
                    f"got {type(value).__name__}"
                )
            return bool(value)
        raise AssertionError(f"unknown persisted source scalar kind {kind!r}")

    def _authoritative_source_fact(self, label: int) -> _PersistedSourceFact:
        label = int(label)
        frame = _replacement_hard_group(self._entry_group(), f"frames/frame_{label:04d}") if self._replacement_configuration is not None else self._entry_group().get(f"frames/frame_{label:04d}")
        source = (_replacement_hard_group(frame, "source") if self._replacement_configuration is not None else frame.get("source")) if isinstance(frame, h5py.Group) else None
        if source is None:
            return _PersistedSourceFact(False, None, None, ())
        if not isinstance(source, h5py.Group):
            raise WriterStateError(f"existing frame {label} source is not a group")
        try:
            path_node = (_replacement_hard_group(source, "path", h5py.Dataset) if self._replacement_configuration is not None else source["path"])
            index_node = (_replacement_hard_group(source, "frame_index", h5py.Dataset) if self._replacement_configuration is not None else source["frame_index"])
            stored_path = (_replacement_utf8_scalar(path_node, f"existing frame {label} source/path", max_bytes=_MAX_REPLACEMENT_PATH_UTF8_BYTES) if self._replacement_configuration is not None else path_node[()])
            stored_frame_index = index_node[()]
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise WriterStateError(
                f"existing frame {label} has malformed source identity"
            ) from error
        path = self._decode_persisted_source_scalar(
            stored_path,
            kind="text",
            role=f"existing frame {label} source/path",
        )
        frame_index = self._decode_persisted_source_scalar(
            stored_frame_index,
            kind="integer",
            role=f"existing frame {label} source/frame_index",
        )
        snapshot = []
        for key, attr_name in _SOURCE_SNAPSHOT_ATTRIBUTES:
            if attr_name not in source.attrs:
                value = None
            elif self._replacement_configuration is not None:
                value = _read_replacement_source_attribute(source, attr_name, key)
            else:
                observed = source.attrs[attr_name]
                if key in {"size", "mtime_ns", "frame_count"}:
                    kind = "integer"
                elif key == "self_contained":
                    kind = "boolean"
                else:
                    kind = "text"
                value = self._decode_persisted_source_scalar(
                    observed,
                    kind=kind,
                    role=f"existing frame {label} source@{attr_name}",
                )
            snapshot.append((key, value))
        return _PersistedSourceFact(True, path, frame_index, tuple(snapshot))

    def _detach_replacement_fact(self, label: int, *, metadata_keys=(), include_geometry=False) -> Mapping[str, Any]:
        self._require_active()
        with self._boundary():
            return _decode_replacement_fact(self._h5, int(label), entry=self.entry, rows=self._row_cursors, context=self._replacement_read_context, metadata_keys=metadata_keys, include_geometry=include_geometry)
    def _verify_supplied_source_identity(
        self,
        record: RecordWrite,
        prepared_thumbnail: _PreparedThumbnail | None = None,
    ) -> None:
        """Require complete source-fact equality for a mode-only sibling."""
        expected = self._expected_source_fact(record, prepared_thumbnail)
        observed = self._authoritative_source_fact(int(record.label))
        if observed != expected:
            raise WriterStateError(
                f"existing frame {int(record.label)} complete source fact "
                "does not match the mode-only write"
            )
        frame = _replacement_hard_group(self._entry_group(), f"frames/frame_{int(record.label):04d}") if self._replacement_configuration is not None else self._entry_group().get(f"frames/frame_{int(record.label):04d}")
        if not isinstance(frame, h5py.Group) or read_background_dependency(frame) != self._expected_frame_row(record, prepared_thumbnail).background_dependency:
            raise WriterStateError("existing frame background dependency does not match the mode-only write")

    def _remember_written_rows(
        self,
        records: tuple[RecordWrite, ...],
        prepared_thumbnails: Mapping[int, _PreparedThumbnail],
    ) -> None:
        for record in records:
            # A finite regenerable Overwrite starts from an empty working
            # artifact.  Retaining another owned copy of every result row
            # solely to reread it at each checkpoint adds memory and GIL
            # pressure.  Compact proofs are checked at terminal close rather
            # than at each checkpoint.
            if self._fast_regenerable:
                self._remember_fast_mode_proof(record, dimension="1d")
                self._remember_fast_mode_proof(record, dimension="2d")
            else:
                self._remember_mode_row(record, dimension="1d")
                self._remember_mode_row(record, dimension="2d")
            if record.write_frame_record and not self._fast_regenerable:
                self._remember_frame_row(
                    record,
                    prepared_thumbnails.get(int(record.label)),
                )
            self._append_written_labels.add(int(record.label))

    def _verify_mode_row(
        self,
        evidence: _EvidenceBuilder,
        expected: _ExpectedModeRow,
        observed: _ExpectedModeRow | None = None,
    ) -> _DurableModeProof:
        entry = self._entry_group()
        group = entry.get(expected.group_name)
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durability readback missing mode group {expected.group_name}"
            )
        # The row is only the current lookup coordinate.  A publication drop
        # compacts later labels in place, so including that mutable coordinate
        # in the retained digest makes unchanged label/content evidence stale.
        # Keep the proof bound to the stable label and the observed payload;
        # frame_index[row] below still proves that the lookup found this label.
        role = f"{group.name}[label={expected.label}]"
        evidence.text(f"{role}/dimension", expected.dimension, expected.dimension)
        evidence.text(f"{role}/mode", expected.mode, expected.mode)
        evidence.text(f"{role}/group", expected.group_name, expected.group_name)
        wanted_primary = (
            self._primary_mode_1d
            if expected.dimension == "1d"
            else self._primary_mode_2d
        )
        top_name = f"integrated_{expected.dimension}"
        top = entry.get(top_name)
        if not isinstance(top, h5py.Group):
            raise WriterStateError(f"durability readback missing {top_name}")
        observed_primary = self._text_value(
            top.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)
        )
        evidence.text(f"{top.name}/primary_mode", wanted_primary, observed_primary)
        if expected.row < 0 or expected.row >= group["frame_index"].shape[0]:
            raise WriterStateError(f"durability readback row is absent for {role}")
        frame_index = group["frame_index"]
        evidence.array(
            f"{role}/frame_index",
            np.asarray(expected.label, dtype=frame_index.dtype),
            np.asarray(frame_index[expected.row]),
        )
        grouped = self._semantic_mode_observation
        observed_radial = (
            grouped[0] if grouped is not None else
            np.asarray(group["q"][()]) if observed is None else observed.radial
        )
        evidence.array(f"{role}/q", expected.radial, observed_radial)
        evidence.text(
            f"{role}/q@units",
            expected.unit,
            self._text_value(group["q"].attrs.get("units", "")),
        )
        observed_intensity = None if grouped is None else grouped[1]
        if observed_intensity is None:
            observed_intensity = (
                np.asarray(group["intensity"][expected.row])
                if observed is None else observed.intensity
            )
        evidence.array(f"{role}/intensity", expected.intensity, observed_intensity)
        evidence.text(
            f"{role}/shape-facts",
            json.dumps(
                {
                    "source": expected.source_shape,
                    "stored": tuple(expected.intensity.shape),
                    "transpose": expected.dimension == "2d",
                },
                sort_keys=True,
            ),
            json.dumps(
                {
                    "source": expected.source_shape,
                    "stored": tuple(observed_intensity.shape),
                    "transpose": expected.dimension == "2d",
                },
                sort_keys=True,
            ),
        )
        sigma = local_hard_dataset(
            group, "sigma", role=f"{expected.group_name}/sigma",
        )
        if expected.sigma is None:
            observed_sigma = (
                None if observed is None and sigma is None
                else (np.asarray(sigma[expected.row])
                      if observed is None else observed.sigma)
            )
            if observed_sigma is None:
                evidence.absent(f"{role}/sigma", True)
            else:
                evidence.array(
                    f"{role}/sigma-absent-row",
                    np.full(expected.intensity.shape, np.nan, dtype=np.float32),
                    observed_sigma,
                )
        else:
            if not isinstance(sigma, h5py.Dataset):
                raise WriterStateError(f"durability readback missing {role}/sigma")
            evidence.array(
                f"{role}/sigma",
                expected.sigma,
                (np.asarray(sigma[expected.row])
                 if observed is None else observed.sigma),
            )
        if expected.dimension == "2d":
            if expected.azimuthal is None:
                raise WriterStateError("2-D evidence has no azimuthal axis")
            evidence.array(
                f"{role}/chi",
                expected.azimuthal,
                (grouped[2] if grouped is not None else
                 np.asarray(group["chi"][()])
                 if observed is None else observed.azimuthal),
            )
            evidence.text(
                f"{role}/chi@units",
                str(expected.azimuthal_unit or ""),
                self._text_value(group["chi"].attrs.get("units", "")),
            )
            evidence.text(
                f"{role}/two_d_kind",
                str(expected.two_d_kind),
                self._text_value(group.attrs.get("two_d_kind", "")),
            )
        return _DurableModeProof(
            group_name=expected.group_name,
            label=expected.label,
            row=expected.row,
            dimension=expected.dimension,
            mode=expected.mode,
            unit=expected.unit,
            azimuthal_unit=expected.azimuthal_unit,
            two_d_kind=expected.two_d_kind,
            source_shape=expected.source_shape,
            sigma_expected=expected.sigma is not None,
            digest=evidence.observed_hexdigest(),
        )

    def _expected_frame_children(self, expected: _ExpectedFrameRow) -> set[str]: return {name for name, value in (("thumbnail", expected.thumbnail), ("thumbnail_mask", expected.thumbnail_mask), ("source", expected.source_path), ("timestamp", expected.timestamp), ("background_dependency", expected.background_dependency)) if value is not None}

    def _verify_frame_row(self, evidence: _EvidenceBuilder, expected: _ExpectedFrameRow) -> None: self._verify_frame_row_with_children(evidence, expected, self._expected_frame_children(expected))

    def _verify_average_frame_row(self, evidence: _EvidenceBuilder, expected: _ExpectedFrameRow) -> None: children = self._expected_frame_children(expected); children.add("finite_counts"); self._verify_frame_row_with_children(evidence, expected, children)

    def _verify_frame_row_with_children(self, evidence: _EvidenceBuilder, expected: _ExpectedFrameRow, expected_children: set[str]) -> None:
        role = f"{self.entry}/frames/frame_{expected.label:04d}"
        frame = self._entry_group().get(f"frames/frame_{expected.label:04d}")
        if not isinstance(frame, h5py.Group):
            raise WriterStateError(f"durability readback missing {role}")
        expected_source_fact = self._source_fact_for_expected_frame(expected)
        observed_source_fact = self._authoritative_source_fact(expected.label)
        if observed_source_fact != expected_source_fact:
            raise WriterStateError(
                f"durability readback mismatch for {role}/source complete fact"
            )
        evidence.text(
            f"{role}/children",
            json.dumps(sorted(expected_children)),
            json.dumps(sorted(frame.keys())),
        )
        if expected.thumbnail is None:
            evidence.absent(f"{role}/thumbnail", "thumbnail" not in frame)
        else:
            dataset = frame.get("thumbnail")
            if not isinstance(dataset, h5py.Dataset):
                raise WriterStateError(f"durability readback missing {role}/thumbnail")
            evidence.array(
                f"{role}/thumbnail",
                expected.thumbnail,
                np.asarray(dataset[()]),
            )
            assert expected.thumbnail_lut is not None
            for key, value in zip(THUMBNAIL_LUT_ATTRS, expected.thumbnail_lut):
                observed = dataset.attrs.get(key)
                if isinstance(value, float):
                    evidence.array(
                        f"{role}/thumbnail@{key}",
                        np.asarray(value, dtype=np.asarray(observed).dtype),
                        np.asarray(observed),
                    )
                else:
                    evidence.text(
                        f"{role}/thumbnail@{key}",
                        value,
                        self._text_value(observed),
                    )
            evidence.text(
                f"{role}/thumbnail@mask_baked",
                str(expected.thumbnail_mask_baked),
                str(bool(dataset.attrs.get("mask_baked", True))),
            )
        evidence.text(
            f"{role}@mask_baked",
            str(expected.mask_baked),
            str(bool(frame.attrs.get("mask_baked", True))),
        )
        if expected.thumbnail_mask is None:
            evidence.absent(
                f"{role}/thumbnail_mask", "thumbnail_mask" not in frame,
            )
        else:
            mask = frame.get("thumbnail_mask")
            if not isinstance(mask, h5py.Dataset):
                raise WriterStateError(
                    f"durability readback missing {role}/thumbnail_mask"
                )
            evidence.array(
                f"{role}/thumbnail_mask",
                expected.thumbnail_mask,
                np.asarray(mask[()]),
            )
        if expected.source_path is None:
            evidence.absent(f"{role}/source", "source" not in frame)
        else:
            source = frame.get("source")
            if not isinstance(source, h5py.Group):
                raise WriterStateError(f"durability readback missing {role}/source")
            evidence.text(
                f"{role}/source/path",
                expected.source_path,
                observed_source_fact.path,
            )
            evidence.text(
                f"{role}/source/frame_index",
                expected.source_frame_index,
                observed_source_fact.frame_index,
            )
            attr_names = dict(_SOURCE_SNAPSHOT_ATTRIBUTES)
            expected_attr_names = {
                attr_names[key]
                for key, value in expected.source_snapshot
                if value is not None
            }
            observed_attr_names = set(source.attrs) - {"NX_class"}
            evidence.text(
                f"{role}/source@fields",
                json.dumps(sorted(expected_attr_names)),
                json.dumps(sorted(observed_attr_names)),
            )
            observed_snapshot = dict(observed_source_fact.snapshot)
            for key, value in expected.source_snapshot:
                attr_name = attr_names.get(key)
                if attr_name is None or value is None:
                    continue
                evidence.text(
                    f"{role}/source@{attr_name}",
                    value,
                    observed_snapshot[key],
                )
        if expected.timestamp is None:
            evidence.absent(f"{role}/timestamp", "timestamp" not in frame)
        else:
            evidence.text(
                f"{role}/timestamp",
                expected.timestamp,
                self._text_value(frame["timestamp"][()]),
            )
        observed_background = read_background_dependency(frame)
        if observed_background != expected.background_dependency:
            raise WriterStateError(f"durability readback mismatch for {role}/background_dependency")
        if observed_background is not None:
            evidence.text(f"{role}/background_dependency", expected.background_dependency[1], observed_background[1])

    def _frame_row_digest(self, label: int) -> tuple[str, int]:
        """Digest the complete persisted provenance row without retaining data."""
        frame = self._entry_group().get(f"frames/frame_{int(label):04d}")
        if not isinstance(frame, h5py.Group):
            raise WriterStateError(f"durable frame proof lost label {label}")
        return self._frame_row_digest_nodes(frame, frame)

    def _average_frame_row_digest(self, label: int) -> tuple[str, int]:
        frame = self._entry_group().get(f"frames/frame_{int(label):04d}")
        if not isinstance(frame, h5py.Group): raise WriterStateError(f"durable frame proof lost label {label}")
        return self._frame_row_digest_nodes(frame, (name for name in frame if name != "finite_counts"))

    def _frame_row_digest_nodes(self, frame: h5py.Group, root_names) -> tuple[str, int]:
        digest = hashlib.sha256(b"xrd-tools-frame-row-v1\0")
        read_bytes = 0

        def update(role: str, payload: bytes) -> None:
            digest.update(len(role).to_bytes(8, "big"))
            digest.update(role.encode("utf-8"))
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)

        def scalar(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="strict")
            if isinstance(value, np.generic):
                return value.item()
            return value

        def walk(group: h5py.Group, prefix: str, names) -> None:
            nonlocal read_bytes
            for name in sorted(group.attrs):
                value = scalar(group.attrs[name])
                update(f"{prefix}@{name}", repr(value).encode("utf-8"))
            for name in sorted(names):
                child = group[name]
                role = f"{prefix}/{name}"
                if isinstance(child, h5py.Group):
                    update(role, b"group")
                    walk(child, role, child)
                    continue
                if not isinstance(child, h5py.Dataset):
                    raise WriterStateError(f"unsupported frame proof node {role}")
                observed = np.asarray(child[()])
                facts = json.dumps(
                    [observed.dtype.str, list(observed.shape)],
                    separators=(",", ":"),
                ).encode("utf-8")
                if observed.dtype.kind in {"O", "S", "U"}:
                    values = [self._text_value(item) for item in observed.ravel()]
                    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
                else:
                    payload = observed.tobytes(order="C")
                update(role, facts + payload)
                read_bytes += len(payload)
                for attr_name in sorted(child.attrs):
                    value = scalar(child.attrs[attr_name])
                    update(
                        f"{role}@{attr_name}", repr(value).encode("utf-8"),
                    )

        walk(frame, frame.name, root_names)
        return digest.hexdigest(), read_bytes

    def _verify_average_finite_counts(
        self,
        expected,
    ) -> tuple[str, int]:
        from xrd_tools.reduction.average import AverageFiniteCountsEvidence
        if type(expected) is not AverageFiniteCountsEvidence:
            raise WriterStateError("Average count proof has invalid evidence")
        entry = self._entry_group()
        frames = entry.get("frames")
        frame = None if not isinstance(frames, h5py.Group) else frames.get(
            "frame_0001",
        )
        if (
            not isinstance(frame, h5py.Group)
            or "source" in frame
            or set(self._append_written_labels) != {1}
        ):
            raise WriterStateError(
                "Average count proof requires sole source-free frame 1",
            )
        link = frame.get("finite_counts", getlink=True)
        dataset = frame.get("finite_counts")
        expected_attrs = {
            "average_scan_policy",
            "contributor_extent",
            "finite_counts_sha256",
            "finite_counts_min",
            "finite_counts_max",
            "finite_counts_zero_count",
        }
        if (
            type(link) is not h5py.HardLink
            or not isinstance(dataset, h5py.Dataset)
            or dataset.dtype != np.dtype("<u4")
            or dataset.shape != expected.shape
            or dataset.chunks != _average_count_chunks(expected.shape)
            or dataset.compression != "gzip"
            or dataset.compression_opts != 1
            or dataset.shuffle is not True
            or dataset.fletcher32 is not False
            or dataset.is_virtual
            or dataset.external is not None
            or set(dataset.attrs) != expected_attrs
        ):
            raise WriterStateError("Average count durability schema is invalid")
        def scalar_attr(name: str, dtype: str):
            value = dataset.attrs[name]
            if (
                np.asarray(value).shape != ()
                or dataset.attrs.get_id(name).dtype != np.dtype(dtype)
            ):
                raise WriterStateError(
                    f"Average count attribute {name} is invalid",
                )
            return value
        policy = scalar_attr("average_scan_policy", "S15")
        extent = scalar_attr("contributor_extent", "<u4")
        sha256 = scalar_attr("finite_counts_sha256", "S64")
        minimum = scalar_attr("finite_counts_min", "<u4")
        maximum = scalar_attr("finite_counts_max", "<u4")
        zero_count = scalar_attr("finite_counts_zero_count", "<u8")
        try:
            policy_text = bytes(policy).decode("ascii")
            digest_text = bytes(sha256).decode("ascii")
        except (TypeError, UnicodeDecodeError) as error:
            raise WriterStateError("Average count text evidence is invalid") from error
        scalar_observed = (
            policy_text,
            int(extent),
            digest_text,
            int(minimum),
            int(maximum),
            int(zero_count),
        )
        scalar_expected = (
            expected.policy,
            expected.contributor_extent,
            expected.sha256,
            expected.minimum,
            expected.maximum,
            expected.zero_count,
        )
        if scalar_observed != scalar_expected:
            raise WriterStateError("Average count scalar evidence changed")
        height, width = expected.shape
        rows = expected.chunks[0]
        digest = hashlib.sha256(b"xdart.average-finite-counts.v1\0")
        digest.update(struct.pack(
            "<QQQ", height, width, expected.contributor_extent,
        ))
        observed_min = expected.contributor_extent
        observed_max = 0
        observed_zero = 0
        read_bytes = 0
        for start in range(0, height, rows):
            slab = np.asarray(dataset[start:start + rows])
            if slab.dtype != np.dtype("<u4") or slab.ndim != 2:
                raise WriterStateError("Average count slab is invalid")
            digest.update(slab.tobytes(order="C"))
            observed_min = min(observed_min, int(slab.min()))
            observed_max = max(observed_max, int(slab.max()))
            observed_zero += int(np.count_nonzero(slab == 0))
            read_bytes += int(slab.nbytes)
        observed_digest = digest.hexdigest()
        if (
            observed_digest != expected.sha256
            or observed_min != expected.minimum
            or observed_max != expected.maximum
            or observed_zero != expected.zero_count
            or observed_max > expected.contributor_extent
            or observed_zero >= height * width
        ):
            raise WriterStateError("Average count durability census is invalid")
        return observed_digest, read_bytes
    def _replacement_manifest_digest(self, exclude, handle=None) -> str:
        root = self._h5 if handle is None else handle
        return _replacement_manifest_digest_for(
            root,
            self.entry,
            tuple(exclude),
            ignore_source_base=self._replacement_configuration is not None,
            ignore_file_name=self._seeded_document is not None,
        )
    def _verify_indexed_row(
        self,
        evidence: _EvidenceBuilder,
        expected: _ExpectedIndexedRow,
    ) -> None:
        group = local_hard_group_path(
            self._entry_group(), expected.group_name, role=expected.group_name,
        )
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durability readback missing indexed group {expected.group_name}"
            )
        cursor = self._row_cursors[expected.group_name]
        row = cursor.get(expected.label)
        if row is None:
            raise WriterStateError(
                f"durability readback lacks {expected.group_name} label {expected.label}"
            )
        role = f"{group.name}[label={expected.label},row={row}]"
        labels = local_hard_dataset(
            group, "frame_index", role=f"{expected.group_name}/frame_index",
        )
        if labels is None:
            raise WriterStateError(
                f"durability readback missing {expected.group_name}/frame_index"
            )
        evidence.array(
            f"{role}/frame_index",
            np.asarray(expected.label, dtype=labels.dtype),
            np.asarray(labels[row]),
        )
        for name, value in expected.values:
            dataset = local_hard_dataset(
                group, name, role=f"{expected.group_name}/{name}",
            )
            if not isinstance(dataset, h5py.Dataset):
                raise WriterStateError(f"durability readback missing {role}/{name}")
            observed = dataset[row]
            if dataset.dtype.kind in "OSU":
                evidence.text(
                    f"{role}/{name}",
                    self._scan_text_value(value),
                    self._text_value(observed),
                )
            else:
                evidence.array(
                    f"{role}/{name}",
                    np.asarray(value, dtype=dataset.dtype),
                    np.asarray(observed),
                )

    def _verify_absent_mode_row(
        self,
        evidence: _EvidenceBuilder,
        group_name: str,
        label: int,
    ) -> None:
        group = local_hard_group_path(
            self._entry_group(), group_name, role=group_name,
        )
        role = f"/{self.entry}/{group_name}[label={int(label)}]"
        if group is None:
            evidence.absent(role, True)
            return
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durability readback found non-group mode path {group_name}"
            )
        labels_ds = local_hard_dataset(
            group, "frame_index", role=f"{group_name}/frame_index",
        )
        if not isinstance(labels_ds, h5py.Dataset):
            raise WriterStateError(
                f"durability readback missing {group_name}/frame_index"
            )
        labels = np.asarray(labels_ds[()], dtype=np.int64).ravel()
        if len(labels) != len(set(int(value) for value in labels)):
            raise WriterStateError(
                f"{group_name}/frame_index contains duplicate labels"
            )
        evidence.absent(role, int(label) not in {int(value) for value in labels})
        digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
        evidence.text(f"{role}/remaining-labels", digest, digest)

    def _verify_lineage(self, evidence: _EvidenceBuilder) -> None:
        if not self._lineage_dirty:
            return
        if self._lineage_expected is None:
            raise WriterStateError("dirty Append lineage has no expected value")
        dataset = self._entry_group().get("reduction/config/append_lineage")
        if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
            raise WriterStateError("durability readback missing Append lineage")
        evidence.text(
            f"{dataset.name}",
            self._lineage_expected,
            self._text_value(dataset[()]),
        )

    def _verify_dirty_evidence(self) -> tuple[
        str,
        int,
        int,
        dict[tuple[str, int], _DurableModeProof],
        dict[int, _DurableFrameProof | _DurableAverageFrameProof],
    ]:
        aggregate = _EvidenceBuilder()
        read_bytes = 0
        mode_proofs: dict[tuple[str, int], _DurableModeProof] = {}
        frame_proofs: dict[
            int, _DurableFrameProof | _DurableAverageFrameProof
        ] = {}
        def absorb(role: str, evidence: _EvidenceBuilder) -> None:
            nonlocal read_bytes
            digest = evidence.hexdigest()
            aggregate.text(role, digest, digest)
            read_bytes += evidence.read_bytes

        mode_rows = tuple(
            self._dirty_modes[key] for key in sorted(self._dirty_modes)
        )
        for rows in self._semantic_groups(mode_rows):
            grouped = self._read_grouped_mode(rows, "checkpoint")
            for offset, expected in enumerate(rows):
                self._semantic_mode_observation = (
                    None if grouped is None else
                    (grouped[0], grouped[1][offset], grouped[2])
                )
                try:
                    evidence = _EvidenceBuilder()
                    proof = self._verify_mode_row(evidence, expected)
                    mode_proofs[(proof.group_name, proof.label)] = proof
                    absorb(f"mode:{proof.group_name}:{proof.label}", evidence)
                finally:
                    self._semantic_mode_observation = None
        for group_name, label in sorted(self._dirty_absent_modes):
            evidence = _EvidenceBuilder()
            self._verify_absent_mode_row(evidence, group_name, label)
            absorb(f"mode-absent:{group_name}:{label}", evidence)
        for label in sorted(self._dirty_frames):
            evidence = _EvidenceBuilder()
            self._verify_frame_row(evidence, self._dirty_frames[label])
            absorb(f"frame:{label}", evidence)
            frame_digest, frame_bytes = self._frame_row_digest(label)
            frame_proofs[label] = _DurableFrameProof(label, frame_digest)
            read_bytes += frame_bytes
        for key in sorted(self._dirty_indexed):
            evidence = _EvidenceBuilder()
            self._verify_indexed_row(evidence, self._dirty_indexed[key])
            absorb(f"indexed:{key[0]}:{key[1]}", evidence)
        if self._lineage_dirty:
            evidence = _EvidenceBuilder()
            self._verify_lineage(evidence)
            absorb("append-lineage", evidence)
        rows = (
            len(self._dirty_modes) + len(self._dirty_absent_modes)
            + len(self._dirty_frames)
            + len(self._dirty_indexed)
        )
        return aggregate.hexdigest(), read_bytes, rows, mode_proofs, frame_proofs

    def _verify_average_dirty_evidence(self, counts):
        dirty, labels = self._dirty_frames, tuple(sorted(self._dirty_frames))
        if labels not in ((), (1,)): raise WriterStateError("Average count proof requires exact frame 1")
        self._dirty_frames = {}
        try: digest, read_bytes, rows, mode_proofs, frame_proofs = self._verify_dirty_evidence()
        finally: self._dirty_frames = dirty
        expected = getattr(counts, "evidence", None); prior = self._durable_frame_proofs.get(1); evidence = _EvidenceBuilder()
        if labels: self._verify_average_frame_row(evidence, dirty[1])
        elif type(prior) not in (_DurableFrameProof, _DurableAverageFrameProof): raise WriterStateError("Average count proof requires retained frame 1")
        frame_digest, frame_bytes = self._average_frame_row_digest(1)
        if not labels and frame_digest != prior.digest: raise WriterStateError("Average count proof changed retained frame 1")
        count_digest, count_bytes = self._verify_average_finite_counts(expected); aggregate = _EvidenceBuilder(); aggregate.text("ordinary-dirty", digest, digest); semantic = evidence.hexdigest() if labels else prior.digest; aggregate.text("frame:1", semantic, semantic); aggregate.text("average-counts:1", expected.sha256, count_digest)
        frame_proofs[1] = _DurableAverageFrameProof(1, frame_digest, count_digest, expected)
        return aggregate.hexdigest(), read_bytes + evidence.read_bytes + frame_bytes + count_bytes, rows + len(labels) + 1, mode_proofs, frame_proofs

    def _reverify_durable_mode_proof(
        self,
        proof: _DurableModeProof,
        grouped: _CloseModeObservation | None = None,
        offset: int = 0,
    ) -> _EvidenceBuilder:
        if grouped is not None:
            context = grouped.context
            if (
                offset < 0 or offset >= len(grouped.frame_indices)
                or proof.row != grouped.start + offset
                or context.group_name != proof.group_name
                or context.dimension != proof.dimension
            ):
                raise WriterStateError(f"durable-row proof lost label {proof.label}")
            label = int(np.asarray(grouped.frame_indices[offset]).item())
            if label != proof.label:
                raise WriterStateError(f"durable-row proof changed label {proof.label}")
            intensity = np.asarray(grouped.intensities[offset])
            sigma = None if grouped.sigmas is None else np.asarray(grouped.sigmas[offset])
            evidence = _EvidenceBuilder(observed_only=True)
            role = f"{context.group_path}[label={proof.label}]"
            evidence.text(f"{role}/dimension", proof.dimension, proof.dimension)
            evidence.text(f"{role}/mode", proof.mode, proof.mode)
            evidence.text(f"{role}/group", proof.group_name, proof.group_name)
            wanted_primary = (
                self._primary_mode_1d if proof.dimension == "1d"
                else self._primary_mode_2d
            )
            evidence.text(
                f"{context.top_path}/primary_mode",
                wanted_primary, context.primary_mode,
            )
            evidence.array(
                f"{role}/frame_index",
                np.asarray(proof.label, dtype=context.frame_index_dtype),
                np.asarray(label, dtype=context.frame_index_dtype),
            )
            evidence.array(f"{role}/q", context.radial, context.radial)
            evidence.text(f"{role}/q@units", proof.unit, context.unit)
            evidence.array(f"{role}/intensity", intensity, intensity)
            shape_facts = json.dumps({
                "source": proof.source_shape,
                "stored": tuple(intensity.shape),
                "transpose": proof.dimension == "2d",
            }, sort_keys=True)
            evidence.text(f"{role}/shape-facts", shape_facts, shape_facts)
            if proof.sigma_expected:
                if sigma is None:
                    raise WriterStateError(f"durable-row proof lost sigma for label {proof.label}")
                evidence.array(f"{role}/sigma", sigma, sigma)
            elif sigma is None:
                evidence.absent(f"{role}/sigma", True)
            else:
                evidence.array(
                    f"{role}/sigma-absent-row",
                    np.full(intensity.shape, np.nan, dtype=np.float32), sigma,
                )
            if proof.dimension == "2d":
                if context.azimuthal is None:
                    raise WriterStateError(f"durable-row proof lost chi for label {proof.label}")
                evidence.array(
                    f"{role}/chi", context.azimuthal, context.azimuthal,
                )
                evidence.text(
                    f"{role}/chi@units", str(proof.azimuthal_unit or ""),
                    str(context.azimuthal_unit or ""),
                )
                evidence.text(
                    f"{role}/two_d_kind", str(proof.two_d_kind),
                    str(context.two_d_kind),
                )
            if evidence.observed_hexdigest() != proof.digest:
                raise WriterStateError(
                    f"durable-row proof changed for {proof.group_name} "
                    f"label {proof.label}"
                )
            return evidence
        if self._h5 is None:
            raise WriterStateError("durable-row proof has no HDF5 reader")
        entry = local_hard_group_path(self._h5, self.entry, role=self.entry)
        if not isinstance(entry, h5py.Group):
            raise WriterStateError("durable-row proof lost the NeXus entry")
        group = local_hard_group_path(
            entry, proof.group_name, role=proof.group_name,
        )
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durable-row proof lost mode group {proof.group_name}"
            )
        intensity = local_hard_dataset(
            group, "intensity", role=f"{proof.group_name}/intensity",
        )
        radial = local_hard_dataset(
            group, "q", role=f"{proof.group_name}/q",
        )
        if not isinstance(intensity, h5py.Dataset) or not isinstance(
            radial, h5py.Dataset,
        ):
            raise WriterStateError(
                f"durable-row proof lost arrays for {proof.group_name}"
            )
        if proof.row < 0 or proof.row >= intensity.shape[0]:
            raise WriterStateError(
                f"durable-row proof lost label {proof.label}"
            )
        sigma_dataset = local_hard_dataset(
            group, "sigma", role=f"{proof.group_name}/sigma",
        )
        if proof.sigma_expected:
            if not isinstance(sigma_dataset, h5py.Dataset):
                raise WriterStateError(
                    f"durable-row proof lost sigma for label {proof.label}"
                )
        sigma = (
            None if sigma_dataset is None
            else np.asarray(sigma_dataset[proof.row])
        )
        grouped = self._semantic_mode_observation
        azimuthal = None
        if proof.dimension == "2d":
            chi = local_hard_dataset(
                group, "chi", role=f"{proof.group_name}/chi",
            )
            if not isinstance(chi, h5py.Dataset):
                raise WriterStateError(
                    f"durable-row proof lost chi for label {proof.label}"
                )
            azimuthal = (
                np.asarray(chi[()]) if grouped is None else grouped[2]
            )
        observed_intensity = None if grouped is None else grouped[1]
        if observed_intensity is None:
            observed_intensity = np.asarray(intensity[proof.row])
        observation = _ExpectedModeRow(
            group_name=proof.group_name,
            label=proof.label,
            row=proof.row,
            dimension=proof.dimension,
            mode=proof.mode,
            radial=np.asarray(radial[()]) if grouped is None else grouped[0],
            azimuthal=azimuthal,
            intensity=observed_intensity,
            sigma=sigma,
            unit=proof.unit,
            azimuthal_unit=proof.azimuthal_unit,
            two_d_kind=proof.two_d_kind,
            source_shape=proof.source_shape,
        )
        expected = _ExpectedModeRow(
            group_name=proof.group_name,
            label=proof.label,
            row=proof.row,
            dimension=proof.dimension,
            mode=proof.mode,
            radial=observation.radial,
            azimuthal=observation.azimuthal,
            intensity=observation.intensity,
            sigma=observation.sigma if proof.sigma_expected else None,
            unit=proof.unit,
            azimuthal_unit=proof.azimuthal_unit,
            two_d_kind=proof.two_d_kind,
            source_shape=proof.source_shape,
        )
        evidence = _EvidenceBuilder(observed_only=True)
        self._verify_mode_row(evidence, expected, observation)
        if evidence.observed_hexdigest() != proof.digest:
            raise WriterStateError(
                f"durable-row proof changed for {proof.group_name} "
                f"label {proof.label}"
            )
        return evidence

    @staticmethod
    def _descriptor_stat_tuple(descriptor: int) -> tuple[int, int, int, int, int]:
        observed = os.fstat(descriptor)
        return (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )

    def _seal_verified_stream_close(
        self, binding, verification_path=None, verification_descriptor=None,
        *, seal_transaction: bool = True,
    ) -> None:
        if (not self._durable_mode_proofs
                and not self._durable_absence_proofs
                and not self._durable_frame_proofs):
            raise WriterStateError(
                "durable close has no retained mode-row or absence proof"
            )
        previous = self._h5
        if verification_descriptor is None:
            verification_descriptor = self._close_verification_descriptor
        aggregate = _EvidenceBuilder()
        read_bytes = 0
        verification_stream = None
        try:
            if verification_descriptor is not None:
                # Reopen the exact duplicated live inode/handle, not a pathname
                # that an atomic transaction may already have replaced.  A
                # file-object VFD is portable across POSIX and Windows; the
                # duplicated descriptor itself remains the stat/seal authority.
                verification_stream = os.fdopen(
                    verification_descriptor, "rb", closefd=False,
                )
                verification_source = verification_stream
            else:
                verification_source = (
                    Path(verification_path)
                    if verification_path is not None
                    else (self._active_path or self.target)
                )
            with h5py.File(verification_source, "r") as handle:
                self._h5 = handle
                descriptor = (
                    verification_descriptor
                    if verification_descriptor is not None
                    else self._vfd_descriptor()
                )
                before = self._descriptor_stat_tuple(descriptor)
                proofs = tuple(
                    self._durable_mode_proofs[key]
                    for key in sorted(self._durable_mode_proofs)
                )
                for proofs_group, grouped in self._close_mode_groups(proofs):
                    for offset, proof in enumerate(proofs_group):
                        evidence = self._reverify_durable_mode_proof(
                            proof, grouped, offset,
                        )
                        digest = evidence.hexdigest()
                        aggregate.text(
                            f"mode:{proof.group_name}:{proof.label}",
                            digest,
                            digest,
                        )
                        read_bytes += evidence.read_bytes
                for key in sorted(self._durable_absence_proofs):
                    proof = self._durable_absence_proofs[key]
                    evidence = _EvidenceBuilder()
                    self._verify_absent_mode_row(
                        evidence, proof.group_name, proof.label,
                    )
                    digest = evidence.hexdigest()
                    aggregate.text(
                        f"mode-absent:{proof.group_name}:{proof.label}",
                        digest,
                        digest,
                    )
                    read_bytes += evidence.read_bytes
                frame_labels = sorted(self._durable_frame_proofs)
                average_proof = self._durable_frame_proofs.get(1)
                if isinstance(average_proof, _DurableAverageFrameProof):
                    frame_labels.remove(1)
                for label in frame_labels:
                    proof = self._durable_frame_proofs[label]
                    observed, observed_bytes = self._frame_row_digest(label)
                    if observed != proof.digest:
                        raise WriterStateError(
                            f"durable frame proof changed for label {label}"
                        )
                    aggregate.text(f"frame:{label}", proof.digest, observed)
                    read_bytes += observed_bytes
                if isinstance(average_proof, _DurableAverageFrameProof):
                    observed, observed_bytes = self._average_frame_row_digest(1)
                    if observed != average_proof.digest:
                        raise WriterStateError("durable frame proof changed for label 1")
                    aggregate.text("frame:1", average_proof.digest, observed)
                    count_digest, count_bytes = self._verify_average_finite_counts(average_proof.evidence)
                    if count_digest != average_proof.count_digest:
                        raise WriterStateError("durable Average count proof changed")
                    aggregate.text("average-counts:1", average_proof.count_digest, count_digest)
                    read_bytes += observed_bytes + count_bytes
                after = self._descriptor_stat_tuple(descriptor)
                if after != before:
                    raise WriterStateError(
                        "durable rows changed during post-close verification"
                    )
                if seal_transaction:
                    binding.transaction.seal_stream_close(
                        binding.attempt,
                        self._stream_close_attempt,
                        lease=binding.lease,
                        descriptor=descriptor,
                        expected_stat=after,
                        evidence_digest=aggregate.hexdigest(),
                        evidence_bytes=read_bytes,
                    )
        except BaseException:
            if seal_transaction:
                binding.transaction.hold_stream_close(
                    binding.attempt,
                    self._stream_close_attempt,
                    lease=binding.lease,
                )
            raise
        finally:
            self._h5 = previous
            if verification_stream is not None:
                verification_stream.close()
            if verification_descriptor is not None:
                os.close(verification_descriptor)
                self._close_verification_descriptor = None

    def _verify_fast_mode_inventory(
        self,
        entry: h5py.Group,
        evidence: _EvidenceBuilder,
    ) -> None:
        """Bind every expected mode key to its persisted NeXus group."""
        for dimension in ("1d", "2d"):
            groups = tuple(
                proof for proof in self._fast_mode_groups.values()
                if proof.dimension == dimension
            )
            top_name = f"integrated_{dimension}"
            top = local_hard_group_path(entry, top_name, role=top_name)
            if not groups:
                if top is not None:
                    raise WriterStateError(
                        f"fast close found unexpected finite {dimension} output"
                    )
                evidence.absent(f"finite-{dimension}-mode-inventory", True)
                continue
            if not isinstance(top, h5py.Group):
                raise WriterStateError(
                    f"fast close lost finite {dimension} mode inventory"
                )
            wanted_primary = (
                self._primary_mode_1d
                if dimension == "1d" else self._primary_mode_2d
            )
            path_by_mode = {proof.mode: proof.group_name for proof in groups}
            if len(path_by_mode) != len(groups):
                raise WriterStateError(
                    f"fast close found duplicate {dimension} mode identity"
                )
            expected_modes = (
                wanted_primary,
                *(mode for mode in path_by_mode if mode != wanted_primary),
            )
            expected_paths = tuple(
                (mode, path_by_mode[mode]) for mode in expected_modes
            )
            try:
                observed_primary, observed_modes, observed_pairs = (
                    read_current_mode_layout(top, dimension)
                )
            except (TypeError, ValueError) as error:
                raise WriterStateError(
                    f"fast close found invalid {dimension} mode inventory"
                ) from error
            observed_paths = tuple(
                (mode, group.name.removeprefix(f"{entry.name}/"))
                for mode, group in observed_pairs
            )
            if (
                observed_primary != wanted_primary
                or observed_modes != expected_modes
                or observed_paths != expected_paths
            ):
                raise WriterStateError(
                    f"fast close changed {dimension} mode inventory"
                )
            evidence.text(
                f"finite-{dimension}-mode-inventory",
                json.dumps(expected_paths, separators=(",", ":")),
                json.dumps(observed_paths, separators=(",", ":")),
            )

    def _verify_fast_group_context(
        self,
        evidence: _EvidenceBuilder,
        proof: _FastModeGroupProof,
        context: _CloseModeGroupContext,
    ) -> int:
        """Verify shared axes and units once for one persisted mode."""
        wanted_primary = (
            self._primary_mode_1d
            if proof.dimension == "1d" else self._primary_mode_2d
        )
        observed_radial = self._fast_array_proof(context.radial)
        sigma_expected = any(proof.sigma_presence)
        expected_shape = (proof.row_count, *proof.row_shape)
        if (
            context.group_name != proof.group_name
            or context.dimension != proof.dimension
            or context.primary_mode != wanted_primary
            or context.frame_index_shape != (proof.row_count,)
            or context.frame_index_dtype != np.dtype(np.int64)
            or context.intensity_shape != expected_shape
            or observed_radial != proof.radial
            or context.radial.dtype != np.dtype(np.float32)
            or context.unit != proof.unit
            or context.axis_kind != proof.axis_kind
            or context.intensity_dtype != np.dtype(np.float32)
            or context.sigma_present != sigma_expected
            or (
                context.sigma_shape is not None
                and context.sigma_shape != expected_shape
            )
            or (
                context.sigma_dtype is not None
                and context.sigma_dtype != np.dtype(np.float32)
            )
        ):
            raise WriterStateError(
                f"fast close changed shared science for {proof.group_name}"
            )
        evidence.text(
            f"{proof.group_name}/q",
            proof.radial.digest,
            observed_radial.digest,
        )
        schema = json.dumps(
            {
                "dimension": proof.dimension,
                "mode": proof.mode,
                "primary_mode": wanted_primary,
                "frame_index_shape": context.frame_index_shape,
                "frame_index_dtype": str(context.frame_index_dtype),
                "intensity_shape": context.intensity_shape,
                "intensity_dtype": str(context.intensity_dtype),
                "sigma_present": context.sigma_present,
                "sigma_shape": context.sigma_shape,
                "sigma_dtype": (
                    None
                    if context.sigma_dtype is None else str(context.sigma_dtype)
                ),
                "q_shape": observed_radial.shape,
                "q_dtype": str(context.radial.dtype),
                "q_unit": context.unit,
                "axis_kind": context.axis_kind,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        evidence.text(f"{proof.group_name}/schema", schema, schema)
        read_bytes = int(context.radial.nbytes)
        if proof.dimension == "2d":
            observed_azimuthal = (
                None
                if context.azimuthal is None
                else self._fast_array_proof(context.azimuthal)
            )
            if (
                observed_azimuthal != proof.azimuthal
                or context.azimuthal.dtype != np.dtype(np.float32)
                or context.azimuthal_unit != proof.azimuthal_unit
                or context.two_d_kind != proof.two_d_kind
            ):
                raise WriterStateError(
                    f"fast close changed azimuthal science for {proof.group_name}"
                )
            evidence.text(
                f"{proof.group_name}/chi",
                proof.azimuthal.digest,
                observed_azimuthal.digest,
            )
            angular = json.dumps(
                {
                    "chi_shape": observed_azimuthal.shape,
                    "chi_dtype": str(context.azimuthal.dtype),
                    "chi_unit": context.azimuthal_unit,
                    "two_d_kind": context.two_d_kind,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            evidence.text(f"{proof.group_name}/angular-schema", angular, angular)
            read_bytes += int(context.azimuthal.nbytes)
        return read_bytes

    def _update_observed_fast_rows(
        self,
        proof: _FastModeGroupProof,
        rows: tuple[_FastModeRowKey, ...],
        observation: _CloseModeObservation,
        digest,
    ) -> int:
        """Add one bounded persisted slab to its mode-level digest."""
        observed_labels = np.asarray(observation.frame_indices)
        expected_labels = np.asarray(
            [row.label for row in rows], dtype=np.int64,
        )
        if not np.array_equal(observed_labels, expected_labels):
            raise WriterStateError(
                f"fast close changed frame labels for {proof.group_name}"
            )
        read_bytes = int(observed_labels.nbytes + observation.intensities.nbytes)
        if observation.sigmas is not None:
            read_bytes += int(observation.sigmas.nbytes)
        for offset, row in enumerate(rows):
            sigma_row = (
                None
                if observation.sigmas is None else observation.sigmas[offset]
            )
            sigma_expected = bool(proof.sigma_presence[row.row])
            if not sigma_expected:
                if sigma_row is not None and not np.isnan(sigma_row).all():
                    raise WriterStateError(
                        f"fast close changed absent sigma for {proof.group_name} "
                        f"label {row.label}"
                    )
            elif sigma_row is None:
                raise WriterStateError(
                    f"fast close lost sigma for {proof.group_name} "
                    f"label {row.label}"
                )
            self._update_fast_row_digest(
                digest,
                row.label,
                observation.intensities[offset],
                sigma_row if sigma_expected else None,
            )
        return read_bytes

    def _verify_fast_integrated_results(self) -> tuple[str, int]:
        """Verify integrated-result stacks and Average counts once at close."""
        evidence = _EvidenceBuilder(observed_only=True)
        entry = self._h5.get(self.entry)
        if not isinstance(entry, h5py.Group) or self._text_value(
            entry.attrs.get("NX_class", "")
        ) != "NXentry":
            raise WriterStateError("fast close lost its NXentry metadata")
        evidence.text(f"/{self.entry}/NX_class", "NXentry", "NXentry")
        self._verify_fast_mode_inventory(entry, evidence)
        if not self._fast_mode_groups:
            evidence.absent("finite-result-science", True)
        read_bytes = 0
        for group_name, proof in self._fast_mode_groups.items():
            cursor = self._row_cursors.get(group_name, {})
            labels: list[int | None] = [None] * proof.row_count
            for label, row in cursor.items():
                if (
                    isinstance(row, (bool, np.bool_))
                    or not isinstance(row, (int, np.integer))
                    or int(row) < 0
                    or int(row) >= proof.row_count
                    or labels[int(row)] is not None
                ):
                    raise WriterStateError(
                        f"fast close lost finite cursor coverage for {group_name}"
                    )
                labels[int(row)] = int(label)
            if (
                proof.row_count < 1
                or len(cursor) != proof.row_count
                or len(proof.sigma_presence) != proof.row_count
                or any(label is None for label in labels)
            ):
                raise WriterStateError(
                    f"fast close lost finite proof coverage for {group_name}"
                )
            rows = tuple(
                _FastModeRowKey(group_name, int(label), row, proof.dimension)
                for row, label in enumerate(labels)
            )
            observed_digest = hashlib.sha256()
            first_slab = True
            for slab, observation in self._close_mode_groups(rows):
                if first_slab:
                    read_bytes += self._verify_fast_group_context(
                        evidence, proof, observation.context,
                    )
                    first_slab = False
                read_bytes += self._update_observed_fast_rows(
                    proof, slab, observation, observed_digest,
                )
            expected = proof.row_digest.copy().hexdigest()
            observed = observed_digest.hexdigest()
            if observed != expected:
                raise WriterStateError(
                    f"fast close changed result rows for {group_name}"
                )
            evidence.text(f"{group_name}/rows", expected, observed)
        counts = self._finalization.average_finite_counts
        if counts is not None:
            expected = getattr(counts, "evidence", None)
            count_digest, count_bytes = self._verify_average_finite_counts(
                expected,
            )
            evidence.text("average-counts:1", expected.sha256, count_digest)
            read_bytes += count_bytes
        return evidence.hexdigest(), read_bytes

    def _clear_dirty_evidence(self) -> None:
        self._dirty_modes.clear()
        self._dirty_absent_modes.clear()
        self._dirty_frames.clear()
        self._dirty_indexed.clear()
        self._lineage_dirty = False

    def _entry_group(self):
        if self._fast_regenerable:
            cached = self._fast_entry_group
            if cached is not None:
                return cached
            cached = self._h5.require_group(self.entry)
            self._fast_entry_group = cached
            return cached
        return self._h5.require_group(self.entry)

    def _bump(self, name: str, value: int = 1) -> None:
        self._vector[name] += int(value)

    def operation_vector(self) -> WriterOperationVector:
        return WriterOperationVector(**self._vector)

    def reset_operation_vector(self) -> None:
        self._vector = {name: 0 for name in _VECTOR_FIELDS}

    def _load_cursor_from(self, entry: h5py.Group, name: str, *, local_hard=False) -> dict[int, int]:
        group = (
            _replacement_hard_group(entry, name)
            if local_hard else local_hard_group_path(entry, name, role=name)
        )
        return self._load_cursor_group(group, name, local_hard=local_hard)

    def _load_cursor_group(
        self,
        group: h5py.Group | None,
        name: str,
        *,
        local_hard: bool = False,
    ) -> dict[int, int]:
        if group is None:
            return {}
        labels_node = (
            _replacement_hard_group(group, "frame_index", h5py.Dataset)
            if local_hard else local_hard_dataset(
                group, "frame_index", role=f"{name}/frame_index",
            )
        )
        if not isinstance(group, h5py.Group) or not isinstance(labels_node, h5py.Dataset):
            raise WriterStateError(f"{name} has no indexed frame_index")
        values = (_read_replacement_frame_index(labels_node, f"{name}/frame_index") if local_hard else np.asarray(labels_node[()]).ravel())
        labels = [int(x) for x in values]
        if len(labels) != len(set(labels)):
            raise WriterStateError(f"{name}/frame_index contains duplicate labels")
        self._bump("frame_index_scan_rows", len(labels))
        return {label: row for row, label in enumerate(labels)}

    def _load_cursor(self, name: str) -> dict[int, int]:
        return self._load_cursor_from(self._entry_group(), name)

    def _replacement_exclusions(
        self,
        entry: h5py.Group,
        dimension: str,
    ) -> tuple[str, ...]:
        return _replacement_exclusions_for(
            entry,
            dimension,
            include_finite_lineage=self._seeded_document is not None,
        )

    def _validate_existing_contract(
        self,
        handle: h5py.File | None = None,
    ) -> dict[str, dict[int, int]]:
        owner = h5py.File(self.target, "r") if handle is None else nullcontext(handle)
        with owner as h5:
            admitted = None
            if self._replacement_configuration is not None:
                entry = _replacement_hard_group(h5, self.entry)
            else:
                admitted = require_current_writable_processed_groups(
                    h5,
                    self.entry,
                    container=self.target,
                )
                entry = admitted.entry
            if self._replacement_configuration is not None:
                dimension = self._replacement_configuration[0]; excluded = self._replacement_exclusions(entry, dimension); config = _replacement_hard_group(entry, "reduction/config"); (None if isinstance(entry, h5py.Group) and isinstance(config, h5py.Group) else (_ for _ in ()).throw(WriterStateError("replacement entry/config is not local"))); gi_node = _replacement_hard_group(config, "gi_config", h5py.Dataset); gi_link = config.get("gi_config", getlink=True); (None if gi_link is None or type(gi_link) is h5py.HardLink else (_ for _ in ()).throw(WriterStateError("replacement GI config is not local"))); gi_values = _replacement_json_node(config, "gi_config", "replacement GI config", required=False)
                gi_values = {} if gi_values is None else gi_values
                gi_name = f"gi_mode_{dimension}"; (None if type(gi_values) is dict else (_ for _ in ()).throw(WriterStateError("replacement GI config is malformed")))
                gi_signature = self._replacement_node_signature(gi_node)
                gi_preserved = json.dumps({key: value for key, value in gi_values.items() if key != gi_name}, sort_keys=True, separators=(",", ":")).encode()
                receipt = self._prepared_manifest_receipt
                if receipt is not None:
                    bai_signature = self._replacement_node_signature(
                        _replacement_hard_group(
                            config, f"bai_{dimension}_args", h5py.Dataset,
                        ),
                        max_bytes=_MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
                    )
                    audit_signature = self._replacement_node_signature(
                        _replacement_hard_group(
                            config,
                            f"dimension_replacement_{dimension}",
                            h5py.Dataset,
                        ),
                        max_bytes=_MAX_REPLACEMENT_CONFIG_UTF8_BYTES,
                    )
                    execution_signature = self._replacement_node_signature(
                        _replacement_hard_group(
                            config, "source_execution", h5py.Dataset,
                        ),
                        max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES,
                    )
                    lineage_signature = self._replacement_node_signature(
                        _replacement_hard_group(
                            config, "append_lineage", h5py.Dataset,
                        ),
                        max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES,
                    )
                    if (
                        receipt.exclusions != excluded
                        or receipt.gi_name != gi_name
                        or receipt.gi_node_signature
                        != _canonical_node_signature_digest(gi_signature)
                        or receipt.gi_structure_signature
                        != _canonical_node_signature_digest(
                            None if gi_signature is None else gi_signature[:-1]
                        )
                        or receipt.selected_bai_signature
                        != _canonical_node_signature_digest(bai_signature)
                        or receipt.selected_audit_signature
                        != _canonical_node_signature_digest(audit_signature)
                        or receipt.source_execution_signature
                        != _canonical_node_signature_digest(execution_signature)
                        or receipt.append_lineage_signature
                        != _canonical_node_signature_digest(lineage_signature)
                        or receipt.gi_preserved_json.encode("utf-8") != gi_preserved
                        or receipt.gi_preserved_digest
                        != hashlib.sha256(gi_preserved).hexdigest()
                    ):
                        raise WriterStateError(
                            "prepared replacement manifest receipt changed"
                        )
                    manifest_digest = receipt.manifest_digest
                else:
                    manifest_digest = self._replacement_manifest_digest(excluded, h5)
                self._replacement_manifest = (
                    excluded, manifest_digest, gi_name, gi_signature, gi_preserved,
                )
            if entry is None:
                return {
                    name: {} for name in (
                        "integrated_1d", "integrated_2d", "scan_data",
                        "per_frame_geometry",
                    )
                }
            if self._replacement_configuration is None and entry.get("frames/frame_0001/finite_counts", getlink=True) is not None: self.write_batch = self._write_batch_preserving_average_counts
            if (
                self._replacement_configuration is None
                and self.complete_record
                and self.source_base
            ):
                validate_source_base(entry, self.source_base)
            for name, requested in (
                ("integrated_1d", self._primary_mode_1d),
                ("integrated_2d", self._primary_mode_2d),
            ):
                if self._replacement_configuration is not None:
                    group = _replacement_hard_group(entry, name)
                elif name == "integrated_1d":
                    group = admitted.integrated_1d
                else:
                    group = admitted.integrated_2d
                if group is None:
                    continue
                existing = (
                    group.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)
                    if self._replacement_configuration is not None
                    else (
                        admitted.primary_mode_1d
                        if name == "integrated_1d"
                        else admitted.primary_mode_2d
                    )
                )
                if self._replacement_configuration is None and existing != requested:
                    raise ValueError(
                        f"{name} primary mode {existing!r} != requested "
                        f"{requested!r}; sparse writes cannot reinterpret "
                        "untouched rows"
                    )
            if self._replacement_configuration is not None:
                cursors = {
                    name: self._load_cursor_from(entry, name, local_hard=True)
                    for name in (
                        "integrated_1d", "integrated_2d", "scan_data",
                        "per_frame_geometry",
                    )
                }
            else:
                cursors = {
                    "integrated_1d": self._load_cursor_group(
                        admitted.integrated_1d,
                        "integrated_1d",
                    ),
                    "integrated_2d": self._load_cursor_group(
                        admitted.integrated_2d,
                        "integrated_2d",
                    ),
                    "scan_data": self._load_cursor_from(entry, "scan_data"),
                    "per_frame_geometry": self._load_cursor_from(
                        entry,
                        "per_frame_geometry",
                    ),
                }
            for name in ("integrated_1d", "integrated_2d"):
                if self._replacement_configuration is not None:
                    group = _replacement_hard_group(entry, name)
                elif name == "integrated_1d":
                    group = admitted.integrated_1d
                else:
                    group = admitted.integrated_2d
                if group is None:
                    continue
                if self._replacement_configuration is not None:
                    children = tuple(
                        (child_name, _replacement_hard_group(group, child_name))
                        for child_name in group
                    )
                    nested_groups = tuple(
                        (f"{name}/{child_name}", child)
                        for child_name, child in children
                        if isinstance(child, h5py.Group)
                    )
                else:
                    pairs = (
                        admitted.mode_groups_1d
                        if name == "integrated_1d"
                        else admitted.mode_groups_2d
                    )
                    nested_groups = tuple(
                        (f"{name}/{mode_subgroup_name(mode)}", child)
                        for mode, child in pairs[1:]
                    )
                for nested, child in nested_groups:
                    cursors[nested] = (
                        self._load_cursor_from(entry, nested, local_hard=True)
                        if self._replacement_configuration is not None
                        else self._load_cursor_group(child, nested)
                    )
            return cursors

    def begin(
        self,
        *,
        metadata=None,
        primary_mode_1d: str = DEFAULT_MODE_KEY,
        primary_mode_2d: str = DEFAULT_MODE_KEY,
    ) -> None:
        if self._replacement_configuration is not None and metadata is not None: raise WriterStateError("replacement begin forbids metadata mutation")
        if self.phase is not WriterPhase.NEW:
            raise WriterStateError(f"begin requires NEW, got {self.phase.value}")
        self._primary_mode_1d = canonical_gi_mode_key(primary_mode_1d, "1d")
        self._primary_mode_2d = canonical_gi_mode_key(primary_mode_2d, "2d")
        try:
            with self._boundary():
                binding = self._transaction_binding
                seeded = self._seeded_document
                exists = False if seeded is not None else self.target.exists()
                logical_exists = (
                    True
                    if seeded is not None
                    else bool(binding.transaction.stream_base_snapshot.exists)
                    if binding is not None else exists
                )
                use_atomic = False if binding is not None or seeded is not None else (
                    bool(self.atomic)
                    if self.atomic is not None
                    else (self.overwrite or not exists)
                )
                preserve_existing = bool(
                    use_atomic and logical_exists and not self.overwrite
                )
                self._fresh = bool(self.overwrite or not logical_exists)
                self._active_path = self.target
                if use_atomic:
                    self._active_path = self.target.with_name(
                        f".{self.target.stem}.{os.getpid()}."
                        f"{uuid.uuid4().hex}.tmp{self.target.suffix}"
                    )
                if binding is None and seeded is None:
                    self._pool.pause(self.target)
                    self._pool_owned = True
                existing_cursors = None
                if logical_exists and not self.overwrite:
                    existing_cursors = self._validate_existing_contract(seeded)
                if use_atomic:
                    self.target.parent.mkdir(parents=True, exist_ok=True)
                    if preserve_existing:
                        shutil.copyfile(self.target, self._active_path)
                if seeded is None:
                    self._authorize_transaction_mutation()
                    self._h5 = self._opener(
                        self._active_path, metadata=metadata, entry=self.entry,
                        compression=self.compression,
                        overwrite=(
                            not preserve_existing
                            if use_atomic
                            else bool(self.overwrite or (binding is not None and not logical_exists))
                        ),
                    )
                else:
                    self._h5 = seeded
                disable_terminal_admission = getattr(
                    self._h5,
                    "_disable_terminal_admission",
                    None,
                )
                if callable(disable_terminal_admission):
                    disable_terminal_admission()
                # The opener may be writing an atomic private path.  Root
                # provenance names the logical record, never that temporary
                # implementation detail; an Append also refreshes a relocated
                # legacy file's path to its current truthful location.
                if (
                    self._replacement_configuration is None
                    or seeded is not None
                ):
                    self._h5.attrs["file_name"] = (
                        self._finite_request.output_artifact
                        if self._finite_request is not None
                        else str(self.target.resolve())
                    )
                if self._replacement_configuration is None and self.complete_record and self.source_base:
                    stamp_source_base(self._entry_group(), self.source_base)
                if self._append_decision is not None:
                    stage_append_lineage(
                        self._entry_group(),
                        self._append_decision,
                    )
                    lineage = dict(self._append_decision.lineage or {})
                    lineage["state"] = "pending"
                    self._lineage_expected = json.dumps(
                        lineage,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    self._lineage_dirty = True
                if existing_cursors is not None:
                    self._row_cursors.update(existing_cursors)
                else:
                    for name in (
                        "integrated_1d", "integrated_2d", "scan_data",
                        "per_frame_geometry",
                    ):
                        self._row_cursors[name] = self._load_cursor(name)
                if self._replacement_configuration is not None:
                    config = _replacement_hard_group(self._h5, f"{self.entry}/reduction/config"); node = _replacement_hard_group(config, "source_execution", h5py.Dataset)
                    try: execution = json.loads(_replacement_utf8_scalar(node, "source_execution", max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES))
                    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                        raise WriterStateError("replacement source_execution is absent or malformed") from error
                    execution = _validate_replacement_execution(execution)
                    source_base, _raw, lineage = decode_replacement_lineage(self._h5, entry=self.entry)
                    self._replacement_read_context = (source_base, lineage, execution)
            self.phase = WriterPhase.ACTIVE
            if self._replacement_configuration is not None:
                dimension, labels, audit, selected, gi_mode = (
                    self._replacement_configuration[:5]
                )
                self._reset_selected_dimension(
                    dimension, labels, audit, selected_plan=selected,
                    selected_gi_mode=gi_mode,
                )
            if self._finite_request is not None:
                with self._boundary():
                    self._authorize_transaction_mutation()
                    write_finite_artifact_lineage(
                        self._h5, self._finite_request,
                    )
        except BaseException as exc:
            self._pending_owner = "begin"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer begin incomplete: {exc}", self._outcome()) from exc

    def _require_active(self) -> None:
        if self.phase is not WriterPhase.ACTIVE:
            raise WriterStateError(f"writer requires ACTIVE, got {self.phase.value}")

    def _reset_selected_dimension(self, dimension: str, labels: tuple[int, ...], audit_bytes: bytes, *, selected_plan: Mapping[str, Any], selected_gi_mode: str | None) -> None:
        self._require_active(); (None if dimension in {"1d", "2d"} else (_ for _ in ()).throw(ValueError("replacement dimension must be '1d' or '2d'")))
        labels = tuple(labels); (None if labels and labels == tuple(sorted(set(labels))) and all(type(label) is int and label >= 0 for label in labels) else (_ for _ in ()).throw(ValueError("replacement labels must be ascending unique nonnegative ints")))
        if type(audit_bytes) is not bytes: raise TypeError("replacement audit must be exact bytes")
        try: audit_text = audit_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error: raise WriterStateError("replacement audit is not UTF-8") from error
        audit_text = _bounded_replacement_config_text(audit_text, "replacement audit")
        bai_text = _bounded_replacement_config_text(json.dumps(dict(selected_plan), sort_keys=True, separators=(",", ":")), "replacement selected BAI")
        (_dimension, _labels, _audit, _selected, _gi_mode,
         source_execution, append_bytes,
         append_value) = self._replacement_configuration
        execution_text = json.dumps(
            source_execution, sort_keys=True, separators=(",", ":"),
        )
        if len(execution_text.encode("utf-8")) > _MAX_REPLACEMENT_LINEAGE_UTF8_BYTES:
            raise WriterStateError("replacement source execution exceeds its byte ceiling")
        append_text = None if append_bytes is None else append_bytes.decode("utf-8")
        top = f"integrated_{dimension}"
        with self._boundary():
            entry = self._entry_group()
            if tuple(self._load_cursor_from(entry, top, local_hard=True)) != labels: raise WriterStateError("replacement label inventory changed")
            excluded = self._replacement_exclusions(entry, dimension)
            frozen = self._replacement_manifest
            if frozen is None or frozen[0] != excluded or self._prepared_manifest_receipt is None and self._replacement_manifest_digest(excluded) != frozen[1]: raise WriterStateError("replacement opener changed preserved artifact")
            config = _replacement_hard_group(entry, "reduction/config"); gi_config = _replacement_hard_group(config, "gi_config", h5py.Dataset); gi_link = None if config is None else config.get("gi_config", getlink=True)
            if (gi_link is not None and type(gi_link) is not h5py.HardLink) or self._replacement_node_signature(gi_config) != frozen[3]: raise WriterStateError("replacement opener changed physical GI config")
            self._authorize_transaction_mutation()
            entry.attrs[SOURCE_BASE_ATTR] = Path(os.fspath(self.source_base)).as_posix()
            for name in ("source_execution", "append_lineage"):
                if name in config:
                    del config[name]
            config.create_dataset("source_execution", data=execution_text)
            if append_text is not None:
                config.create_dataset("append_lineage", data=append_text)
            self._replacement_read_context = (
                os.fspath(self.source_base), append_value, source_execution,
            )
            del entry[top]
            bai_name = f"bai_{dimension}_args"; (config.__delitem__(bai_name) if bai_name in config else None); config.create_dataset(bai_name, data=bai_text)
            gi_name, gi_values = f"gi_mode_{dimension}", {}
            if gi_config is not None:
                try: raw = _replacement_utf8_scalar(gi_config, "replacement GI config"); gi_values = json.loads(raw); del config["gi_config"]
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error: raise WriterStateError("replacement GI config is malformed") from error
                if not isinstance(gi_config, h5py.Dataset) or gi_config.shape != () or type(gi_values) is not dict or raw != json.dumps(gi_values, sort_keys=True, separators=(",", ":")): raise WriterStateError("replacement GI config is noncanonical")
            staged_preserved = json.dumps({key: value for key, value in gi_values.items() if key != gi_name}, sort_keys=True, separators=(",", ":")).encode(); (None if frozen[2] == gi_name and staged_preserved == frozen[4] else (_ for _ in ()).throw(WriterStateError("replacement opener changed sibling GI config"))); gi_preserved = frozen[4]; gi_values.pop(gi_name, None); gi_values.update({} if selected_gi_mode is None else {gi_name: str(selected_gi_mode)})
            if gi_values: config.create_dataset("gi_config", data=_bounded_replacement_config_text(json.dumps(gi_values, sort_keys=True, separators=(",", ":")), "replacement GI config"))
            stored_node = _replacement_hard_group(config, "gi_config", h5py.Dataset); stored_gi = {} if stored_node is None else json.loads(_replacement_utf8_scalar(stored_node, "replacement GI config")); stored_signature = self._replacement_node_signature(stored_node); (None if json.dumps({key: value for key, value in stored_gi.items() if key != gi_name}, sort_keys=True, separators=(",", ":")).encode() == gi_preserved and (None if stored_signature is None else stored_signature[:-1]) == (None if frozen[3] is None else frozen[3][:-1]) else (_ for _ in ()).throw(WriterStateError("replacement changed sibling GI config")))
            audit_name = f"dimension_replacement_{dimension}"; (config.__delitem__(audit_name) if audit_name in config else None); config.create_dataset(audit_name, data=audit_text)
            result_seal_name = f"{audit_name}_result_seal"
            if result_seal_name in config:
                del config[result_seal_name]
            self._row_cursors = {name: cursor for name, cursor in self._row_cursors.items() if name != top and not name.startswith(f"{top}/")}; self._row_cursors[top] = {}
            self._replacement_labels = labels; self._replacement_manifest = frozen; self._replacement_expected = (bai_name, self._replacement_node_signature(config.get(bai_name)), "gi_config", self._replacement_node_signature(config.get("gi_config")), audit_name, self._replacement_node_signature(config.get(audit_name)), gi_name, gi_preserved, os.fspath(self.source_base), self._replacement_node_signature(config.get("source_execution"), max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES), self._replacement_node_signature(config.get("append_lineage"), max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES))
    def _replacement_node_signature(self, node, *, max_bytes=None):
        return _replacement_node_signature_value(node, max_bytes=max_bytes)
    def _seal_finite_replacement_result(self) -> None:
        if self._finite_request is None:
            return
        if self._replacement_configuration is None:
            raise WriterStateError("finite result seal lost replacement facts")
        dimension = self._replacement_configuration[0]
        audit_bytes = self._replacement_configuration[2]
        audit_identity = hashlib.sha256(audit_bytes).hexdigest()
        config = _replacement_hard_group(
            self._entry_group(), "reduction/config",
        )
        if not isinstance(config, h5py.Group):
            raise WriterStateError("finite result seal lost replacement config")
        name = f"dimension_replacement_{dimension}_result_seal"
        self._authorize_transaction_mutation()
        if name in config:
            del config[name]
        payload = {
            "schema": _FINITE_RESULT_SEAL_SCHEMA,
            "version": 1,
            "dimension": dimension,
            "audit_identity": audit_identity,
            "result_sha256": _finite_selected_result_digest(
                self._h5, self.entry, dimension,
            ),
        }
        config.create_dataset(
            name,
            data=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        if self._prepared_manifest_admission is None:
            require_finite_replacement_result_seal(
                self._h5,
                entry=self.entry,
                dimension=dimension,
                audit_identity=audit_identity,
            )
    def _verify_replacement_manifest(self) -> None:
        entry, selected_expected = self._entry_group(), self._replacement_expected; excluded, manifest_expected, _gi_name, gi_original, _gi_preserved = self._replacement_manifest
        if self._finite_request is not None:
            dimension = self._replacement_configuration[0]
            require_finite_replacement_result_seal(
                self._h5,
                entry=self.entry,
                dimension=dimension,
                audit_identity=hashlib.sha256(
                    self._replacement_configuration[2]
                ).hexdigest(),
            )
        if self._replacement_manifest_digest(excluded) != manifest_expected: raise WriterStateError("replacement final verification changed preserved artifact")
        config = _replacement_hard_group(entry, "reduction/config"); bai, bai_expected, gi, gi_expected, audit, audit_expected, gi_name, gi_preserved, source_base, execution_expected, lineage_expected = selected_expected; nodes = {name: _replacement_hard_group(config, name, h5py.Dataset) for name in (bai, gi, audit)}
        if config is None or any(config.get(name, getlink=True) is not None and type(config.get(name, getlink=True)) is not h5py.HardLink for name in nodes): raise WriterStateError("replacement selected science/audit link changed")
        raw_gi = {} if nodes[gi] is None else json.loads(_replacement_utf8_scalar(nodes[gi], "replacement GI config")); gi_signature = self._replacement_node_signature(nodes[gi]); preserved = json.dumps({key: value for key, value in raw_gi.items() if key != gi_name}, sort_keys=True, separators=(",", ":")).encode()
        execution = _replacement_hard_group(config, "source_execution", h5py.Dataset); lineage = _replacement_hard_group(config, "append_lineage", h5py.Dataset); stored_base = entry.attrs.get(SOURCE_BASE_ATTR); stored_base = stored_base.decode("utf-8", errors="strict") if isinstance(stored_base, bytes) else stored_base
        if self._replacement_node_signature(nodes[bai]) != bai_expected or gi_signature != gi_expected or self._replacement_node_signature(nodes[audit]) != audit_expected or preserved != gi_preserved or (None if gi_signature is None else gi_signature[:-1]) != (None if gi_original is None else gi_original[:-1]) or stored_base != Path(source_base).as_posix() or self._replacement_node_signature(execution, max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES) != execution_expected or self._replacement_node_signature(lineage, max_bytes=_MAX_REPLACEMENT_LINEAGE_UTF8_BYTES) != lineage_expected: raise WriterStateError("replacement selected science/audit/source final verification changed")
    def _verify_cursor(self, name: str, label: int) -> None:
        cursor = self._row_cursors[name]
        row = cursor.get(label)
        if row is None:
            return
        group = (
            _replacement_hard_group(self._entry_group(), name)
            if self._replacement_configuration is not None
            else local_hard_group_path(self._entry_group(), name, role=name)
        )
        if (group is None or row < 0 or row >= group["frame_index"].shape[0]
                or int(group["frame_index"][row]) != label):
            raise WriterStateError(
                f"{name} cursor row {row} does not contain exact label {label}"
            )

    def _mode_cursor_name(self, dimension: str, mode: str) -> str:
        if dimension == "1d":
            primary = self._primary_mode_1d
            allowed = GI_MODE_KEYS_1D
            top = "integrated_1d"
        else:
            primary = self._primary_mode_2d
            allowed = GI_MODE_KEYS_2D
            top = "integrated_2d"
        mode = canonical_gi_mode_key(mode, dimension)
        if mode == primary:
            return top
        if primary == DEFAULT_MODE_KEY:
            raise ValueError(
                f"named {dimension} mode {mode!r} requires a named primary mode"
            )
        if mode not in allowed:
            raise ValueError(f"unknown {dimension} mode {mode!r}")
        return f"{top}/{mode_subgroup_name(mode)}"

    def _prepared_stack_records(self, records: tuple[RecordWrite, ...]):
        """Select only the manifest-admitted immutable-successor primary."""
        if (
            self._prepared_manifest_admission is None
            or self._finite_request is None
            or self._finite_candidate_binding is None
            or self._seeded_document is not self._h5
            or self._transaction_binding is not None
            or self._append_decision is not None
            or self._replacement_configuration is None
        ):
            return None
        dimension = self._replacement_configuration[0]
        primary = (
            self._primary_mode_1d if dimension == "1d"
            else self._primary_mode_2d
        )
        selected = []
        for record in records:
            result = record.result_1d if dimension == "1d" else record.result_2d
            other = record.result_2d if dimension == "1d" else record.result_1d
            mode = record.mode_1d if dimension == "1d" else record.mode_2d
            if record.write_frame_record or other is not None:
                return None
            if result is not None:
                if canonical_gi_mode_key(mode, dimension) != primary:
                    return None
                selected.append(record)
        return dimension, f"integrated_{dimension}", primary, tuple(selected)

    def _capture_prepared_stack_authority(
        self, records: tuple[RecordWrite, ...],
    ) -> None:
        """Retain the first fully validated stack objects as a private capability."""
        qualified = self._prepared_stack_records(records)
        if qualified is None:
            self._prepared_stack_authority = None
            return
        dimension, group_name, primary, selected = qualified
        if not selected:
            return
        results = tuple(
            record.result_1d if dimension == "1d" else record.result_2d
            for record in selected
        )
        group = _replacement_hard_group(self._entry_group(), group_name)
        if not isinstance(group, h5py.Group):
            raise WriterStateError("prepared stack admission lost its exact group")
        names = ("intensity", "frame_index", "q") + (
            ("chi",) if dimension == "2d" else ()
        )
        datasets = {
            name: _replacement_hard_group(group, name, h5py.Dataset)
            for name in names
        }
        if not isinstance(group, h5py.Group) or any(
            not isinstance(node, h5py.Dataset) for node in datasets.values()
        ):
            raise WriterStateError("prepared stack admission lost its exact schema")
        cursor = self._row_cursors[group_name]
        row_count = int(datasets["frame_index"].shape[0])
        if (
            row_count < 1 or len(cursor) != row_count
            or set(cursor.values()) != set(range(row_count))
        ):
            raise WriterStateError("prepared stack cursor is not a complete row authority")
        last_label = int(datasets["frame_index"][row_count - 1])
        if cursor.get(last_label) != row_count - 1:
            raise WriterStateError("prepared stack tail differs from its row authority")
        observed_primary = self._text_value(
            group.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)
        )
        if observed_primary != primary:
            raise WriterStateError("prepared stack primary mode changed")
        first = results[0]
        axes = [np.asarray(datasets["q"][()], np.float32).copy()]
        units = [str(first.unit or "")]
        if dimension == "2d":
            axes.append(np.asarray(datasets["chi"][()], np.float32).copy())
            units.append(str(first.azimuthal_unit or ""))
            expected_kind = two_d_kind_from_units(*units).value
            if self._text_value(group.attrs.get("two_d_kind", "")) != expected_kind:
                raise WriterStateError("prepared 2-D stack kind changed")
        sigma = _replacement_hard_group(group, "sigma", h5py.Dataset)
        if sigma is not None:
            datasets["sigma"] = sigma
        self._prepared_stack_authority = MappingProxyType({
            "dimension": dimension, "group_name": group_name, "primary": primary,
            "cursor": cursor, "group": group,
            "datasets": MappingProxyType(datasets),
            "axes": tuple(axes), "units": tuple(units),
            "row_shape": tuple(datasets["intensity"].shape[1:]),
            "last_label": last_label,
        })

    def _prepare_prepared_stack_append(
        self, records: tuple[RecordWrite, ...],
    ) -> dict[str, Any] | None:
        """Prebuild an append after O(1) identity checks, without HDF mutation."""
        authority = self._prepared_stack_authority
        qualified = self._prepared_stack_records(records)
        if authority is None or qualified is None:
            return None
        dimension, group_name, primary, selected = qualified
        if not selected or (
            dimension, group_name, primary
        ) != (
            authority["dimension"], authority["group_name"], authority["primary"],
        ):
            return None
        cursor = authority["cursor"]
        labels = tuple(int(record.label) for record in selected)
        if any(right <= left for left, right in zip(labels, labels[1:])):
            raise ValueError("prepared stack labels must be strictly increasing")
        if any(label in cursor for label in labels) or labels[0] <= authority["last_label"]:
            return None
        group = _replacement_hard_group(self._entry_group(), group_name)
        if not isinstance(group, h5py.Group):
            raise WriterStateError("prepared stack authority changed before append")
        datasets = authority["datasets"]
        current = {
            name: _replacement_hard_group(group, name, h5py.Dataset)
            for name in datasets
        }
        row_count = len(cursor)
        row_shape = authority["row_shape"]
        if (
            group.id != authority["group"].id
            or self._row_cursors.get(group_name) is not cursor
            or any(
                not isinstance(current[name], h5py.Dataset)
                or current[name].id != dataset.id
                for name, dataset in datasets.items()
            )
            or datasets["frame_index"].shape != (row_count,)
            or datasets["intensity"].shape != (row_count,) + row_shape
            or cursor.get(authority["last_label"]) != row_count - 1
            or int(datasets["frame_index"][row_count - 1]) != authority["last_label"]
            or self._text_value(group.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY))
            != authority["primary"]
            or (
                _replacement_hard_group(group, "sigma", h5py.Dataset) is None
            ) != ("sigma" not in datasets)
            or any(
                self._text_value(datasets[name].attrs.get("units", "")) != unit
                for name, unit in zip(("q", "chi"), authority["units"])
            )
        ):
            raise WriterStateError("prepared stack authority changed before append")
        for offset, axis in enumerate(authority["axes"]):
            name = "q" if offset == 0 else "chi"
            if tuple(datasets[name].shape) != tuple(axis.shape):
                raise WriterStateError("prepared stack axis shape changed before append")
        sigma_node = datasets.get("sigma")
        if sigma_node is not None and sigma_node.shape != (row_count,) + row_shape:
            raise WriterStateError("prepared stack sigma shape changed before append")
        intensity_rows, sigma_rows = [], []
        for record in selected:
            result = record.result_1d if dimension == "1d" else record.result_2d
            incoming_axes = (np.asarray(result.radial, np.float32),)
            incoming_units = (str(result.unit or ""),)
            intensity = np.asarray(result.intensity, np.float32)
            sigma = None if result.sigma is None else np.asarray(result.sigma, np.float32)
            if dimension == "2d":
                incoming_axes += (np.asarray(result.azimuthal, np.float32),)
                incoming_units += (str(result.azimuthal_unit or ""),)
                intensity = intensity.T
                sigma = None if sigma is None else sigma.T
            if incoming_units != authority["units"] or any(
                incoming.shape != expected.shape
                or not np.allclose(incoming, expected, rtol=1e-5, atol=1e-8)
                for incoming, expected in zip(incoming_axes, authority["axes"])
            ):
                raise ValueError("prepared stack axis or unit differs from its authority")
            if intensity.shape != row_shape or (
                sigma is not None and sigma.shape != row_shape
            ):
                raise ValueError("prepared stack result row shape changed")
            if sigma is not None and sigma_node is None:
                return None
            intensity_rows.append(intensity)
            if sigma_node is not None:
                sigma_rows.append(
                    np.full(row_shape, np.nan, np.float32)
                    if sigma is None else sigma
                )
        return {
            "authority": authority, "labels": np.asarray(labels, np.int64),
            "intensity": np.stack(intensity_rows).astype(np.float32, copy=False),
            "sigma": None if sigma_node is None else
            np.stack(sigma_rows).astype(np.float32, copy=False),
        }

    def _append_prepared_stack(self, prepared: Mapping[str, Any]) -> None:
        authority = prepared["authority"]
        datasets, cursor = authority["datasets"], authority["cursor"]
        start, stop = len(cursor), len(cursor) + len(prepared["labels"])
        intensity, frame_index = datasets["intensity"], datasets["frame_index"]
        intensity.resize((stop,) + authority["row_shape"])
        intensity[start:stop] = prepared["intensity"]
        frame_index.resize((stop,))
        frame_index[start:stop] = prepared["labels"]
        if "sigma" in datasets:
            datasets["sigma"].resize((stop,) + authority["row_shape"])
            datasets["sigma"][start:stop] = prepared["sigma"]
        cursor.update({int(label): start + offset for offset, label in enumerate(prepared["labels"])})
        self._prepared_stack_authority = MappingProxyType({
            **authority, "last_label": int(prepared["labels"][-1]),
        })

    def _validate_fresh_fast_records(
        self,
        records: tuple[RecordWrite, ...],
    ) -> dict[int, _PreparedThumbnail]:
        """Validate source rows for a fresh finite artifact before mutation."""
        prepared: dict[int, _PreparedThumbnail] = {}
        for record in records:
            if record.replace_existing or not record.write_frame_record:
                raise WriterStateError(
                    "fresh finite Overwrite requires new frame records"
                )
            thumbnail = (
                None
                if record.thumbnail is None
                else np.asarray(record.thumbnail)
            )
            if thumbnail is not None and thumbnail.ndim != 2:
                raise ValueError("record thumbnail must be exactly 2-D")
            if record.thumbnail_mask is not None:
                mask = np.asarray(record.thumbnail_mask)
                if thumbnail is None:
                    raise ValueError("thumbnail_mask requires a thumbnail")
                if mask.shape != thumbnail.shape:
                    raise ValueError(
                        "record thumbnail_mask shape must equal thumbnail shape"
                    )
            if record.source_path is not None:
                relative_source_path(record.source_path, self.source_base)
            if thumbnail is not None and self.complete_record:
                prepared[int(record.label)] = _prepare_thumbnail(
                    thumbnail,
                    thumbnail_mask=record.thumbnail_mask,
                )
            label = int(record.label)
            if label in self._append_written_labels:
                supplied_groups = []
                if record.result_1d is not None:
                    supplied_groups.append(
                        self._mode_cursor_name("1d", record.mode_1d)
                    )
                if record.result_2d is not None:
                    supplied_groups.append(
                        self._mode_cursor_name("2d", record.mode_2d)
                    )
                if (
                    not supplied_groups
                    or any(
                        label in self._row_cursors.get(group_name, {})
                        for group_name in supplied_groups
                    )
                ):
                    raise WriterStateError(
                        f"fresh finite output already owns frame label {label}"
                    )
                # A GI frame can legitimately arrive once for its primary
                # modes and again for new sibling modes.  Bind that second
                # result to the exact already-written frame fact; otherwise a
                # repeated ordinary write could silently replace provenance.
                self._verify_frame_row(
                    _EvidenceBuilder(),
                    self._expected_frame_row(record, prepared.get(label)),
                )
        return prepared

    def _validate_records(
        self,
        records: tuple[RecordWrite, ...],
        *,
        trusted_group: str | None = None,
    ) -> dict[int, _PreparedThumbnail]:
        labels = [int(record.label) for record in records]
        prepared_thumbnails: dict[int, _PreparedThumbnail] = {}
        if len(labels) != len(set(labels)):
            raise ValueError(f"dirty batch contains duplicate labels: {labels}")
        if self._replacement_labels and (any(label not in self._replacement_labels for label in labels) or any(record.write_frame_record or self._replacement_configuration[0] == "1d" and record.result_2d is not None or self._replacement_configuration[0] == "2d" and record.result_1d is not None for record in records)):
            raise WriterStateError("replacement write escaped its frozen label inventory")
        one_d = [record for record in records if record.result_1d is not None]
        two_d = [record for record in records if record.result_2d is not None]
        groups_1d: dict[str, list[RecordWrite]] = {}
        groups_2d: dict[str, list[RecordWrite]] = {}
        for record in one_d:
            name = self._mode_cursor_name("1d", record.mode_1d)
            self._row_cursors.setdefault(name, {})
            if name != trusted_group:
                self._verify_cursor(name, int(record.label))
            groups_1d.setdefault(name, []).append(record)
        for record in two_d:
            name = self._mode_cursor_name("2d", record.mode_2d)
            self._row_cursors.setdefault(name, {})
            if name != trusted_group:
                self._verify_cursor(name, int(record.label))
            groups_2d.setdefault(name, []).append(record)
        write_validated_groups = (
            {trusted_group} if trusted_group is not None else set()
        )
        for name, grouped in (*groups_1d.items(), *groups_2d.items()):
            grouped_labels = tuple(int(record.label) for record in grouped)
            if any(
                right <= left
                for left, right in zip(grouped_labels, grouped_labels[1:])
            ):
                raise ValueError(
                    f"new labels for {name} must be strictly increasing "
                    "within one batch"
                )
            cursor = self._row_cursors[name]
            new_labels = tuple(
                label for label in grouped_labels if label not in cursor
            )
            if name != trusted_group and new_labels and cursor and new_labels[0] <= max(cursor):
                raise ValueError(
                    f"new labels for {name} must be strictly increasing "
                    "after its persisted cursor"
                )
        for record in records:
            label = int(record.label)
            self._verify_cursor("scan_data", label)
            self._verify_cursor("per_frame_geometry", label)
        entry = self._entry_group()
        for top, groups in (
            ("integrated_1d", groups_1d), ("integrated_2d", groups_2d),
        ):
            if any(name != top for name in groups):
                if (
                    _replacement_hard_group(entry, top)
                    if self._replacement_configuration is not None
                    else local_hard_group_path(entry, top, role=top)
                ) is None and top not in groups:
                    raise ValueError(
                        f"named modes require an established {top} primary group"
                    )
        for name, grouped in groups_1d.items():
            if name in write_validated_groups:
                continue
            validate_integrated_stack_write(
                entry, frame_indices=[int(r.label) for r in grouped],
                results_1d=[r.result_1d for r in grouped],
                group_name_1d=name, allow_rebuild=False,
            )
        for name, grouped in groups_2d.items():
            if name in write_validated_groups:
                continue
            validate_integrated_stack_write(
                entry, frame_indices=[int(r.label) for r in grouped],
                results_2d=[r.result_2d for r in grouped],
                group_name_2d=name, allow_rebuild=False,
            )
        if self._fast_regenerable:
            return self._validate_fresh_fast_records(records)
        existing_scan_data = _replacement_hard_group(self._entry_group(), "scan_data") if self._replacement_configuration is not None else self._entry_group().get("scan_data")
        existing_columns = (
            {str(name) for name in existing_scan_data if name != "frame_index"}
            if existing_scan_data is not None else None
        )
        for record in records:
            thumb = None if record.thumbnail is None else np.asarray(record.thumbnail)
            if thumb is not None and thumb.ndim != 2:
                raise ValueError("record thumbnail must be exactly 2-D")
            if record.thumbnail_mask is not None:
                mask = np.asarray(record.thumbnail_mask)
                if thumb is None:
                    raise ValueError("thumbnail_mask requires a thumbnail")
                if mask.shape != thumb.shape:
                    raise ValueError(
                        "record thumbnail_mask shape must equal thumbnail shape"
                    )
            if record.source_path is not None and self._replacement_configuration is None:
                relative_source_path(record.source_path, self.source_base)
            frame = _replacement_hard_group(entry, f"frames/frame_{int(record.label):04d}") if self._replacement_configuration is not None else entry.get(f"frames/frame_{int(record.label):04d}")
            prepared_thumbnail = None
            if thumb is not None and (
                (self.complete_record and record.write_frame_record)
                or isinstance(frame, h5py.Group)
            ):
                prepared_thumbnail = _prepare_thumbnail(
                    thumb,
                    thumbnail_mask=record.thumbnail_mask,
                )
                prepared_thumbnails[int(record.label)] = prepared_thumbnail
            expected_background = (None if record.background_dependency_bytes is None else
                (record.background_dependency_bytes, record.background_dependency_fingerprint))
            if isinstance(frame, h5py.Group) and read_background_dependency(frame) != expected_background:
                raise WriterStateError("existing frame Background dependency changed")
            if record.replace_existing and not isinstance(frame, h5py.Group):
                raise WriterStateError(
                    f"explicit replacement requires existing frame label "
                    f"{int(record.label)}"
                )
            if not record.write_frame_record and not isinstance(frame, h5py.Group):
                raise WriterStateError(
                    f"mode-only write requires existing frame label "
                    f"{int(record.label)}"
                )
            if (isinstance(frame, h5py.Group) and record.write_frame_record
                    and not record.replace_existing):
                # Existing-row replacement is an exact provenance identity
                # assertion, including child presence/absence, source selector,
                # snapshot fields, thumbnail mask flags, shape, dtype and bytes.
                self._verify_frame_row(
                    _EvidenceBuilder(),
                    self._expected_frame_row(record, prepared_thumbnail),
                )
            elif isinstance(frame, h5py.Group) and not record.write_frame_record:
                self._verify_supplied_source_identity(
                    record,
                    prepared_thumbnail,
                )
            if existing_columns is not None:
                unexpected = set(map(str, record.metadata)) - existing_columns
                if unexpected:
                    raise ValueError(
                        f"dirty metadata adds columns absent on disk: {sorted(unexpected)}"
                    )
        return prepared_thumbnails

    @staticmethod
    def _modes(record: RecordWrite) -> tuple[ResultMode, ...]:
        modes = []
        if record.result_1d is not None:
            modes.append(ResultMode.one_d(canonical_gi_mode_key(record.mode_1d, "1d")))
        if record.result_2d is not None:
            modes.append(ResultMode.two_d(canonical_gi_mode_key(record.mode_2d, "2d")))
        return tuple(modes)

    def _target_matches(self, target: str) -> bool:
        if not str(target).startswith("nexus:"):
            return False
        candidate = str(target)[len("nexus:"):]
        return os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(
            os.path.abspath(self.target)
        )

    def _capture(self, records: tuple[RecordWrite, ...]) -> tuple[StageReceipt, ...]:
        if self._facade is None:
            return ()
        receipts = []
        for record in records:
            for mode in self._modes(record):
                for target in self._facade.targets_for(mode):
                    if self._target_matches(target):
                        receipts.append(
                            self._facade.capture_receipt(int(record.label), mode, target)
                        )
        return tuple(receipts)

    def write(self, record: RecordWrite) -> None:
        self.write_batch((record,))

    def _write_batch_preserving_average_counts(self, records: Iterable[RecordWrite]) -> None:
        self._require_active(); batch = tuple(records)
        if any(int(record.label) == 1 and record.write_frame_record for record in batch): raise WriterStateError("AVERAGE_FINITE_COUNTS_REPLACEMENT_REQUIRES_AVERAGE_FINALIZATION")
        NexusRecordWriter.write_batch(self, batch)

    def write_batch(self, records: Iterable[RecordWrite]) -> None:
        self._require_active()
        batch = tuple(records)
        if not batch:
            return
        metadata_updates = {
            int(record.label): dict(record.metadata)
            for record in batch
            if record.write_frame_record
        }
        receipts = self._capture(batch)
        one_d = [record for record in batch if record.result_1d is not None]
        two_d = [record for record in batch if record.result_2d is not None]
        primary_1d = [
            record for record in one_d
            if canonical_gi_mode_key(record.mode_1d, "1d") == self._primary_mode_1d
        ]
        primary_2d = [
            record for record in two_d
            if canonical_gi_mode_key(record.mode_2d, "2d") == self._primary_mode_2d
        ]
        extra_1d: dict[str, list[RecordWrite]] = {}
        extra_2d: dict[str, list[RecordWrite]] = {}
        for record in one_d:
            mode = canonical_gi_mode_key(record.mode_1d, "1d")
            if mode != self._primary_mode_1d:
                extra_1d.setdefault(mode, []).append(record)
        for record in two_d:
            mode = canonical_gi_mode_key(record.mode_2d, "2d")
            if mode != self._primary_mode_2d:
                extra_2d.setdefault(mode, []).append(record)
        validation_complete = False
        try:
            with self._boundary():
                prepared_stack = self._prepare_prepared_stack_append(batch)
                prepared_thumbnails = self._validate_records(
                    batch,
                    trusted_group=(
                        None if prepared_stack is None else
                        prepared_stack["authority"]["group_name"]
                    ),
                )
                validation_complete = True
                self._authorize_transaction_mutation()
                labels_1d = [int(r.label) for r in primary_1d]
                labels_2d = [int(r.label) for r in primary_2d]
                if prepared_stack is not None:
                    self._append_prepared_stack(prepared_stack)
                elif primary_1d and primary_2d and labels_1d == labels_2d:
                    write_integrated_stack(
                        self._entry_group(), frame_indices=labels_1d,
                        results_1d=[r.result_1d for r in primary_1d],
                        results_2d=[r.result_2d for r in primary_2d],
                        primary_mode_1d=self._primary_mode_1d,
                        primary_mode_2d=self._primary_mode_2d,
                        compression=self.compression,
                        known_rows_1d=self._row_cursors["integrated_1d"],
                        known_rows_2d=self._row_cursors["integrated_2d"],
                        bulk_new_rows=self._fast_regenerable,
                    )
                else:
                    if primary_1d:
                        write_integrated_stack(
                            self._entry_group(), frame_indices=labels_1d,
                            results_1d=[r.result_1d for r in primary_1d],
                            primary_mode_1d=self._primary_mode_1d,
                            compression=self.compression,
                            known_rows_1d=self._row_cursors["integrated_1d"],
                            bulk_new_rows=self._fast_regenerable,
                        )
                    if primary_2d:
                        write_integrated_stack(
                            self._entry_group(), frame_indices=labels_2d,
                            results_2d=[r.result_2d for r in primary_2d],
                            primary_mode_2d=self._primary_mode_2d,
                            compression=self.compression,
                            known_rows_2d=self._row_cursors["integrated_2d"],
                            bulk_new_rows=self._fast_regenerable,
                        )
                if prepared_stack is None and (extra_1d or extra_2d):
                    write_integrated_stack(
                        self._entry_group(), frame_indices=[],
                        extra_modes_1d={
                            mode: [record.result_1d for record in grouped]
                            for mode, grouped in extra_1d.items()
                        } or None,
                        extra_modes_2d={
                            mode: [record.result_2d for record in grouped]
                            for mode, grouped in extra_2d.items()
                        } or None,
                        extra_mode_indices_1d={
                            mode: [int(record.label) for record in grouped]
                            for mode, grouped in extra_1d.items()
                        } or None,
                        extra_mode_indices_2d={
                            mode: [int(record.label) for record in grouped]
                            for mode, grouped in extra_2d.items()
                        } or None,
                        primary_mode_1d=self._primary_mode_1d,
                        primary_mode_2d=self._primary_mode_2d,
                        compression=self.compression,
                        known_rows_extra_1d={
                            mode: self._row_cursors[
                                f"integrated_1d/{mode_subgroup_name(mode)}"
                            ]
                            for mode in extra_1d
                        },
                        known_rows_extra_2d={
                            mode: self._row_cursors[
                                f"integrated_2d/{mode_subgroup_name(mode)}"
                            ]
                            for mode in extra_2d
                        },
                        bulk_new_rows=self._fast_regenerable,
                    )
                if prepared_stack is None and (one_d or two_d):
                    self._capture_prepared_stack_authority(batch)
                if primary_1d:
                    self._entry_group().attrs.setdefault(
                        "default", "integrated_1d",
                    )
                if self.complete_record and any(record.write_frame_record for record in batch):
                    frames = ensure_frames_container(self._entry_group())
                    for record in batch:
                        if not record.write_frame_record:
                            continue
                        replace_frame_record(
                            frames, f"frame_{int(record.label):04d}",
                            thumbnail=record.thumbnail,
                            thumbnail_mask_baked=record.thumbnail_mask_baked,
                            mask_baked=record.mask_baked,
                            thumbnail_mask=record.thumbnail_mask,
                            source_path=record.source_path,
                            source_frame_index=int(record.source_frame_index),
                            timestamp=record.timestamp,
                            source_base=self.source_base,
                            source_snapshot=record.source_snapshot,
                            background_dependency_bytes=record.background_dependency_bytes,
                            background_dependency_fingerprint=record.background_dependency_fingerprint,
                            _prepared_thumbnail=prepared_thumbnails.get(
                                int(record.label),
                            ),
                        )
                self._remember_written_rows(batch, prepared_thumbnails)
            self._bump("prepare_rows", len(batch))
            self._bump("stacked_1d_rows", len(one_d))
            self._bump("stacked_2d_rows", len(two_d))
            if self.complete_record:
                self._bump("source_record_rows", sum(
                    int(record.write_frame_record) for record in batch
                ))
            for label, metadata in metadata_updates.items():
                self._metadata[label] = metadata
                self._pending_metadata_labels.add(label)
            for record in batch:
                if record.source_path is not None:
                    source = str(record.source_path)
                    if source not in self._source_paths:
                        self._source_paths.append(source)
            for receipt in receipts:
                key = (receipt.label, receipt.mode, receipt.target)
                self._pending[key] = receipt
                self._staged_since_checkpoint[key] = receipt
            self._since_flush += len(batch)
            if self.flush_every is not None and self._since_flush >= self.flush_every:
                self.flush()
        except WriterIncomplete:
            raise
        except BaseException as exc:
            if not validation_complete:
                raise
            self._pending_owner = "write"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer mutation incomplete: {exc}", self._outcome()) from exc

    def _current_receipts(self) -> tuple[StageReceipt, ...]:
        if self._facade is None:
            self._pending.clear()
            self._staged_since_checkpoint.clear()
            return ()
        current = []
        for key, receipt in tuple(self._pending.items()):
            observed = self._facade.capture_receipt(
                receipt.label, receipt.mode, receipt.target
            )
            if observed != receipt:
                self._pending.pop(key, None)
                continue
            current.append(receipt)
        return tuple(current)

    def _checkpoint_staged_receipts(self) -> tuple[StageReceipt, ...]:
        """Validate only what was staged since the last checkpoint.

        `_current_receipts` walks the whole pending set, and a fast-regenerable
        Run drains that set only at finish, so calling it per checkpoint is
        quadratic in frame count: measured +2.9% wall over 3000 frames at the
        production cadence, and it grows.  A checkpoint only ever needs to mark
        the rows it newly covers, so read the staged delta and leave `_pending`
        for the single durable publication at finish.

        READ-ONLY on the staged window.  An earlier revision emptied it here,
        which lost the delta when the publication that follows raised: `flush()`
        stays retryable and `_pending` survives, but the retry sealed a fresh
        checkpoint that never covered those labels, so their heavy arrays
        stayed resident until close.  The window is closed by a COMPLETED seal
        (`_seal_checkpoint_and_receipts`) or by abort, never here.
        """
        staged = self._staged_since_checkpoint
        if self._facade is None:
            return ()
        current = []
        for key, receipt in tuple(staged.items()):
            if key not in self._pending:
                continue
            observed = self._facade.capture_receipt(
                receipt.label, receipt.mode, receipt.target
            )
            if observed != receipt:
                self._pending.pop(key, None)
                continue
            current.append(receipt)
        return tuple(current)

    def _current_publication_drops(
        self,
    ) -> tuple[StageReceipt, ...]:
        if self._facade is None:
            self._pending_publication_drops.clear()
            return ()
        current = []
        for (label, mode), revision in tuple(
            self._pending_publication_drops.items()
        ):
            receipt = self._current_nexus_receipt(label, mode)
            if receipt is None or receipt.revision != revision:
                self._pending_publication_drops.pop((label, mode), None)
                continue
            current.append(receipt)
        return tuple(current)

    def _commit_receipts(
        self,
        batch: tuple[StageReceipt, ...] | None = None,
    ) -> None:
        if batch is None:
            batch = self._current_receipts()
        if not batch:
            return
        self._facade.commit_durable(batch)
        for receipt in batch:
            self._pending.pop((receipt.label, receipt.mode, receipt.target), None)

    def _commit_publication_drops(
        self,
        batch: tuple[StageReceipt, ...],
    ) -> None:
        for receipt in batch:
            self._facade.commit_publication_drop(
                receipt.label, receipt.mode, receipt.revision)
            self._pending_publication_drops.pop(
                (receipt.label, receipt.mode), None)

    def _flush_handle(self) -> None:
        self._h5.flush()

    def _invalidate_checkpoint_recovery(self) -> None:
        revoke = getattr(self._facade, "revoke_checkpoint_recovery", None)
        if callable(revoke): revoke()

    @staticmethod
    def _fast_checkpoint_evidence():
        """Seal only the current inode boundary; make no row durability claim.

        A finite regenerable Overwrite publishes no receipts until terminal
        integrated-science verification succeeds.  Re-reading thumbnails,
        source provenance, and indexed presentation metadata at every flush
        therefore added no recoverable guarantee; the incomplete new artifact
        is disposable.  Keep dirty owners bounded, and leave those non-result
        values to the normal HDF5 write/schema path.
        """
        digest = hashlib.sha256(
            b"xrd-tools-fast-regenerable-checkpoint-v1\0"
        ).hexdigest()
        return digest, 0, 0, {}, {}

    def _seal_checkpoint_and_receipts(self, *, publish_receipts: bool = True, verified=None) -> None:
        publish_checkpoint_receipts = (
            publish_receipts and not self._fast_regenerable
        )
        batch = self._current_receipts() if publish_checkpoint_receipts else ()
        dropped = (
            self._current_publication_drops()
            if publish_checkpoint_receipts else ()
        )
        if verified is None:
            if self._fast_regenerable:
                verified = self._fast_checkpoint_evidence()
            else:
                verified = self._verify_dirty_evidence()
        (digest, read_bytes, rows, mode_proofs, frame_proofs) = verified
        binding = self._transaction_binding
        # A dynamic same-run lineage facade cannot publish additive H10
        # durability until H23 has committed that rollback-capable epoch.  It
        # still receives the exact verified batch below, but owns it only as a
        # pending overlay.  Static R7 facades lack this opt-in and retain the
        # existing immediate irreversible-floor contract.
        defer_epoch = self._defer_epoch_durability or bool(
            self._facade is not None
            and getattr(self._facade, "defer_epoch_durability", False)
        )
        checkpoint = None
        if binding is not None:
            checkpoint = binding.transaction.seal_stream_checkpoint(
                binding.attempt,
                lease=binding.lease,
                descriptor=self._vfd_descriptor(),
                evidence_digest=digest,
                evidence_bytes=read_bytes,
            )
            if batch:
                self._durable_mode_proofs.update(mode_proofs)
            self._durable_frame_proofs.update(frame_proofs)
            if dropped:
                for receipt in dropped:
                    group_name = self._mode_cursor_name(
                        receipt.mode.kind, receipt.mode.key)
                    proof = _DurableAbsenceProof(group_name, int(receipt.label))
                    self._durable_absence_proofs[
                        (group_name, int(receipt.label))] = proof
            if (
                not defer_epoch
                and (
                    batch
                    or dropped
                    or binding.transaction.snapshot().durable_floor is not None
                )
            ):
                binding.transaction.promote_stream_checkpoint(
                    binding.attempt,
                    checkpoint,
                    lease=binding.lease,
                )
        if batch:
            self._commit_receipts(batch)
        recover = getattr(self._facade, "commit_checkpoint_recoverable", None)
        # A fast-regenerable checkpoint still publishes NO durability receipts:
        # `batch` and `dropped` stay empty above, so neither commit_durable nor
        # promote_stream_checkpoint runs and nothing is recorded durable without
        # the readback proof this path declines (see _fast_checkpoint_evidence).
        # It DOES hand those same staged receipts to the checkpoint-recovery
        # route, which asserts only "written and re-readable from the artifact".
        # Without it the display store never releases a heavy payload: under a
        # live projection release_heavy is licensed by the recoverable set alone
        # (frame_record_store _heavy_releasable_modes_locked), so every frame's
        # 2-D array stayed resident and memory grew with the frame count -- about
        # 7 GB on a 3621-frame Run against a ~64-frame cap.
        recoverable_batch = (
            self._checkpoint_staged_receipts()
            if (
                self._fast_regenerable
                and publish_receipts
                and self._facade is not None
            )
            else batch
        )
        checkpoint_recovered = (
            checkpoint is not None
            and callable(recover)
            and (not self._fast_regenerable or bool(recoverable_batch))
        )
        if checkpoint_recovered:
            frame_labels = tuple(sorted(
                set(frame_proofs)
                | {
                    int(receipt.label)
                    for receipt in (*recoverable_batch, *dropped)
                }
            ))
            recover(
                checkpoint, recoverable_batch, dropped,
                frame_labels,
                tuple(label for label in frame_labels
                      if label in self._dirty_frames
                      and self._dirty_frames[label].thumbnail is not None),
            )
        if dropped:
            if checkpoint_recovered:
                for receipt in dropped:
                    self._pending_publication_drops.pop(
                        (receipt.label, receipt.mode), None)
            else:
                self._commit_publication_drops(dropped)
        self._checkpoint_rows += rows
        self._checkpoint_read_bytes += read_bytes
        # The ONLY reset point for the staged window, on both paths: the
        # ordinary writer never reads it (it publishes the whole validated
        # batch), so without this the delta would grow for the life of the run.
        # It sits after recover() deliberately -- a seal that raises before
        # here leaves the window intact so the retry republishes the same
        # delta, which is what the pre-31a89da4 writer did from `_pending`.
        # Re-marking an already-recoverable label is idempotent (the store
        # holds a set), so a late raise costs a repeat, never a loss.
        self._staged_since_checkpoint.clear()
        self._clear_dirty_evidence()

    def _close_handle(self) -> None:
        self._prepared_stack_authority = None
        self._fast_entry_group = None
        allow_unverified = self._pending_owner == "abort"
        binding = self._transaction_binding
        fast_evidence = None
        if self._h5 is not None and self._h5 is self._seeded_document:
            # The finite publisher owns this HDF5 context.  Quiesce and detach
            # the writer, but leave the actual close to that enclosing context
            # so descriptor revocation remains single-owner and pathless.
            self._h5 = None
            return
        if allow_unverified and self._fast_regenerable:
            # Fast abort rejects this artifact as canonical output.  Close its
            # HDF5 handle without starting, repeating, or sealing terminal
            # science proof;
            # the transaction can then retire the untrusted bytes directly.
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
            self._stream_close_attempt = None
            return
        verification_path = (
            Path(self._h5.filename)
            if self._h5 is not None and binding is not None
            and hasattr(self._h5, "filename")
            else None
        )
        verification_descriptor = None
        if (
            self._h5 is not None
            and binding is not None
            and self._durable_frame_proofs
            and not self._fast_regenerable
        ):
            verification_descriptor = os.dup(self._vfd_descriptor())
            self._close_verification_descriptor = verification_descriptor
        if self._h5 is not None and binding is not None:
            if self._stream_close_attempt is None:
                try:
                    self._stream_close_attempt = (
                        binding.transaction.begin_stream_close(
                            binding.attempt,
                            lease=binding.lease,
                        )
                    )
                except OutputTransactionError:
                    if verification_descriptor is not None:
                        os.close(verification_descriptor)
                        verification_descriptor = None
                        self._close_verification_descriptor = None
                    if not allow_unverified:
                        raise
                    # Release the HDF5 handle even when an earlier unsealed
                    # mutation has already forced the transaction to retain
                    # its integrity owner.  No receipt is installed here.
                    self._h5.close()
                    self._h5 = None
                    return
            if self._fast_regenerable and self._stream_close_attempt is not None:
                try:
                    fast_evidence = self._verify_fast_integrated_results()
                except BaseException:
                    binding.transaction.hold_stream_close(
                        binding.attempt, self._stream_close_attempt,
                        lease=binding.lease,
                    )
                    raise
        if self._h5 is not None:
            try:
                self._h5.close()
            except BaseException:
                if verification_descriptor is not None:
                    os.close(verification_descriptor)
                    verification_descriptor = None
                    self._close_verification_descriptor = None
                raise
            self._h5 = None
        if binding is not None:
            retained_proofs = bool(
                self._durable_mode_proofs or self._durable_absence_proofs
                or self._durable_frame_proofs
            )
            if retained_proofs and not self._fast_regenerable:
                if self._stream_close_attempt is not None:
                    self._seal_verified_stream_close(binding)
                else:
                    self._seal_verified_stream_close(
                        binding, verification_path, verification_descriptor,
                        seal_transaction=False,
                    )
            elif self._stream_close_attempt is not None:
                binding.transaction.seal_stream_close(
                    binding.attempt,
                    self._stream_close_attempt,
                    lease=binding.lease,
                    evidence_digest=(
                        None if fast_evidence is None else fast_evidence[0]
                    ),
                    evidence_bytes=(
                        None if fast_evidence is None else fast_evidence[1]
                    ),
                )
            self._stream_close_attempt = None

    def _resume_pool(self) -> None:
        if self._pool_owned:
            self._pool.resume(self.target)
            self._pool_owned = False

    def flush(self, *, force: bool = False) -> None:
        if self.phase is WriterPhase.PARTIAL and self._pending_owner == "flush":
            self.phase = WriterPhase.ACTIVE
        self._require_active()
        if self._since_flush <= 0:
            return
        try:
            with self._boundary():
                if self._pending_metadata_labels:
                    self._authorize_transaction_mutation()
                    self._write_pending_metadata()
                self._flush_handle()
                self._bump("flush_boundaries")
                if self._active_path == self.target:
                    self._seal_checkpoint_and_receipts()
            self._since_flush = 0
        except BaseException as exc:
            self._pending_owner = "flush"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer flush incomplete: {exc}", self._outcome()) from exc

    def _drop_mode_row(self, label: int, mode: ResultMode) -> None:
        """Remove one exact indexed result row without disturbing GI siblings."""
        self._prepared_stack_authority = None
        label = int(label)
        if self._replacement_configuration is not None and mode.kind != self._replacement_configuration[0]: raise WriterStateError("replacement publication drop escaped its selected dimension")
        group_name = self._mode_cursor_name(mode.kind, mode.key)
        cursor = self._row_cursors.setdefault(group_name, {})
        group = local_hard_group_path(
            self._entry_group(), group_name, role=group_name,
        )
        if group is None:
            cursor.pop(label, None)
            self._dirty_modes.pop((group_name, label), None)
            self._dirty_absent_modes.add((group_name, label))
            return
        if not isinstance(group, h5py.Group):
            raise WriterStateError(f"{group_name} is not an indexed mode group")
        labels_ds = local_hard_dataset(
            group, "frame_index", role=f"{group_name}/frame_index",
        )
        if not isinstance(labels_ds, h5py.Dataset):
            raise WriterStateError(f"{group_name} has no indexed frame_index")
        labels = [int(value) for value in np.asarray(labels_ds[()]).ravel()]
        if len(labels) != len(set(labels)):
            raise WriterStateError(f"{group_name}/frame_index contains duplicate labels")
        observed_row = cursor.get(label)
        actual_row = labels.index(label) if label in labels else None
        if observed_row != actual_row:
            raise WriterStateError(
                f"{group_name} cached row for label {label} is not exact"
            )
        if actual_row is not None:
            row_count = len(labels)
            if row_count == 1:
                raise WriterStateError(
                    f"cannot remove the last row of mode {group_name}"
                )
            for name in INTEGRATED_ROW_ALIGNED:
                dataset = local_hard_dataset(
                    group, name, role=f"{group_name}/{name}",
                )
                if dataset is None:
                    if name == "sigma":
                        continue
                    raise WriterStateError(
                        f"{group_name} is missing row-aligned dataset {name}"
                    )
                if not isinstance(dataset, h5py.Dataset):
                    raise WriterStateError(
                        f"{group_name}/{name} is not a dataset"
                    )
                if (
                    dataset.ndim < 1
                    or dataset.shape[0] != row_count
                    or dataset.maxshape is None
                    or dataset.maxshape[0] is not None
                ):
                    raise WriterStateError(
                        f"{group_name}/{name} is not an appendable aligned stack"
                    )
            for name in INTEGRATED_ROW_ALIGNED:
                dataset = local_hard_dataset(
                    group, name, role=f"{group_name}/{name}",
                )
                if not isinstance(dataset, h5py.Dataset):
                    continue
                if actual_row + 1 < row_count:
                    dataset[actual_row:-1] = np.asarray(
                        dataset[actual_row + 1:row_count]
                    )
                dataset.resize((row_count - 1,) + tuple(dataset.shape[1:]))
            labels.pop(actual_row)
            group.attrs[MONOTONIC_ATTR] = bool(
                all(left < right for left, right in zip(labels, labels[1:]))
            )
            cursor.clear()
            cursor.update({value: row for row, value in enumerate(labels)})

            for key, expected in tuple(self._dirty_modes.items()):
                if expected.group_name != group_name:
                    continue
                if expected.label == label:
                    self._dirty_modes.pop(key, None)
                    continue
                row = cursor.get(expected.label)
                if row is None:
                    raise WriterStateError(
                        f"dirty mode proof lost {group_name} label {expected.label}"
                    )
                self._dirty_modes[key] = replace(expected, row=int(row))
            for key, proof in tuple(self._durable_mode_proofs.items()):
                if proof.group_name != group_name:
                    continue
                if proof.label == label:
                    self._durable_mode_proofs.pop(key, None)
                    continue
                row = cursor.get(proof.label)
                if row is None:
                    raise WriterStateError(
                        f"durable mode proof lost {group_name} label {proof.label}"
                    )
                self._durable_mode_proofs[key] = replace(proof, row=int(row))
        self._dirty_modes.pop((group_name, label), None)
        self._dirty_absent_modes.add((group_name, label))

    def _preflight_mode_row_drop(self, label: int, mode: ResultMode) -> None:
        """Refuse a last-row removal before authorizing any mutation."""
        group_name = self._mode_cursor_name(mode.kind, mode.key)
        cursor = self._row_cursors.setdefault(group_name, {})
        if self._fast_regenerable and int(label) in cursor:
            raise WriterStateError(
                "finite append-only output cannot drop an already-written row"
            )
        group = local_hard_group_path(
            self._entry_group(), group_name, role=group_name,
        )
        if group is None:
            return
        if not isinstance(group, h5py.Group):
            raise WriterStateError(f"{group_name} is not an indexed mode group")
        labels_ds = local_hard_dataset(
            group, "frame_index", role=f"{group_name}/frame_index",
        )
        if not isinstance(labels_ds, h5py.Dataset):
            raise WriterStateError(f"{group_name} has no indexed frame_index")
        labels = tuple(int(value) for value in np.asarray(labels_ds[()]).ravel())
        if len(labels) != len(set(labels)):
            raise WriterStateError(
                f"{group_name}/frame_index contains duplicate labels"
            )
        observed_row = cursor.get(int(label))
        actual_row = labels.index(int(label)) if int(label) in labels else None
        if observed_row != actual_row:
            raise WriterStateError(
                f"{group_name} cached row for label {int(label)} is not exact"
            )
        if actual_row is not None and len(labels) == 1:
            raise WriterStateError(
                f"cannot remove the last row of mode {group_name}"
            )

    def _current_nexus_receipt(
        self, label: int, mode: ResultMode,
    ) -> StageReceipt | None:
        for target in self._facade.targets_for(mode):
            if self._target_matches(target):
                return self._facade.capture_receipt(int(label), mode, target)
        return None

    def mark_publication_dropped(
        self, label: int, mode: ResultMode, *, expected_revision: int
    ) -> None:
        self._require_active()
        if self._facade is None:
            raise WriterStateError("publication drop requires a bound session facade")
        label, expected = int(label), int(expected_revision)
        current = self._current_nexus_receipt(label, mode)
        if current is None or current.revision != expected:
            # Preserve StageLedger's revision-qualified stale no-op/future
            # refusal without deleting a row that belongs to another revision.
            self._facade.commit_publication_drop(label, mode, expected)
            return
        validation_complete = False
        try:
            with self._boundary():
                self._preflight_mode_row_drop(label, mode)
                validation_complete = True
                self._authorize_transaction_mutation()
                self._drop_mode_row(label, mode)
                self._pending_publication_drops[(label, mode)] = expected
            for key, receipt in tuple(self._pending.items()):
                if receipt.label == label and receipt.mode == mode:
                    self._pending.pop(key, None)
            # A direct drop after an earlier checkpoint still needs an exact
            # absence checkpoint even though it did not append a result row.
            self._since_flush = max(1, self._since_flush)
        except BaseException as exc:
            if not validation_complete:
                raise
            self._pending_owner = "publication-drop"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(
                f"writer publication drop incomplete: {exc}", self._outcome(),
            ) from exc

    def drop_publication(self, label: int, mode: ResultMode) -> None:
        if self._facade is None:
            return
        for target in self._facade.targets_for(mode):
            if self._target_matches(target):
                receipt = self._facade.capture_receipt(int(label), mode, target)
                self.mark_publication_dropped(
                    int(label), mode, expected_revision=receipt.revision,
                )

    @staticmethod
    def _decode(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value.item() if hasattr(value, "item") else value

    def _metadata_table(self, supplied=None, labels: tuple[int, ...] = ()):
        if supplied is not None:
            if not len(supplied.columns):
                return None, ()
            frame_indices = labels or tuple(int(x) for x in supplied.index)
            return supplied, frame_indices
        if not self._metadata:
            return None, ()
        import pandas as pd

        labels = tuple(int(label) for label in labels) or tuple(sorted(self._metadata))
        group = self._entry_group().get("scan_data")
        if group is None:
            columns = sorted({str(k) for row in self._metadata.values() for k in row})
        else:
            columns = [str(name) for name in group if name != "frame_index"]
        if not columns:
            return None, ()
        rows = []
        cursor = self._row_cursors["scan_data"]
        for label in labels:
            incoming = self._metadata[label]
            row = cursor.get(label)
            values = {}
            for column in columns:
                if column in incoming:
                    values[column] = incoming[column]
                elif group is not None and row is not None:
                    if row >= group["frame_index"].shape[0] or int(group["frame_index"][row]) != label:
                        raise WriterStateError("scan_data cursor does not match exact label")
                    values[column] = self._decode(group[column][row])
                else:
                    values[column] = "" if group is not None and group[column].dtype.kind in "OSU" else np.nan
            rows.append(values)
        return pd.DataFrame(rows, index=labels, columns=columns), labels

    def _upsert_metadata_table(self, scan_data, labels: tuple[int, ...]) -> None:
        aligned_scan_data = (
            scan_data
            if list(scan_data.index) == list(labels)
            else scan_data.reindex(list(labels))
        )
        for label in labels:
            self._verify_cursor("scan_data", int(label))
        upsert_scan_metadata(
            self._entry_group(), scan_data, labels,
            known_rows=self._row_cursors["scan_data"],
        )
        for position, label in enumerate(labels):
            self._dirty_indexed[("scan_data", int(label))] = (
                _ExpectedIndexedRow(
                    "scan_data",
                    int(label),
                    tuple(
                        (str(column), aligned_scan_data.iloc[position][column])
                        for column in aligned_scan_data.columns
                    ),
                )
            )
        self._bump("indexed_metadata_rows", len(labels))

    def _write_pending_metadata(self) -> None:
        labels = tuple(sorted(self._pending_metadata_labels))
        if not labels:
            return
        scan_data, labels = self._metadata_table(labels=labels)
        if scan_data is None or not labels:
            return
        self._upsert_metadata_table(scan_data, labels)
        self._pending_metadata_labels.difference_update(labels)

    def _write_finalization(self, finalization: WriterFinalization) -> None:
        self._authorize_transaction_mutation()
        if finalization.scan_data is None:
            self._write_pending_metadata()
            scan_data, labels = self._metadata_table()
        else:
            scan_data, labels = self._metadata_table(
                finalization.scan_data, tuple(finalization.frame_indices)
            )
            if scan_data is not None and labels:
                self._upsert_metadata_table(scan_data, labels)
                self._pending_metadata_labels.clear()
        if scan_data is not None and labels:
            aligned_scan_data = (
                scan_data
                if list(scan_data.index) == list(labels)
                else scan_data.reindex(list(labels))
            )
            if finalization.geometry is not None:
                upsert_per_frame_geometry(
                    self._entry_group(), scan_data, labels,
                    finalization.geometry,
                    known_rows=self._row_cursors["per_frame_geometry"],
                )
                motors = {
                    motor: np.asarray(aligned_scan_data[motor].values, dtype=float)
                    for motor in finalization.geometry.all_referenced_motors()
                    if motor in aligned_scan_data.columns
                }
                derived = {
                    str(key): np.asarray(value, dtype=np.float32)
                    for key, value in finalization.geometry.derive_per_frame(
                        motors
                    ).items()
                } if motors else {}
                if derived:
                    for position, label in enumerate(labels):
                        self._dirty_indexed[("per_frame_geometry", int(label))] = (
                            _ExpectedIndexedRow(
                                "per_frame_geometry",
                                int(label),
                                tuple(
                                    (name, values[position])
                                    for name, values in sorted(derived.items())
                                ),
                            )
                        )
                upsert_positioners(
                    self._entry_group(),
                    scan_data,
                    labels,
                    finalization.geometry,
                )
        if finalization.diffractometer is not None:
            write_diffractometer(self._entry_group(), finalization.diffractometer)
        self._write_detector_values(finalization)
        if finalization.average_finite_counts is not None:
            write_average_finite_counts(
                self._entry_group(), finalization.average_finite_counts,
            )
        if finalization.stitched_1d is not None or finalization.stitched_2d is not None:
            write_stitched(
                self._entry_group(),
                stitched_1d=finalization.stitched_1d,
                stitched_2d=finalization.stitched_2d,
                provenance=finalization.stitched_provenance,
                compression=self.compression,
            )
        if finalization.provenance_config is not None:
            inputs = finalization.provenance_inputs
            if inputs is None and self._fresh and self._source_paths:
                inputs = {"raw_files": list(self._source_paths)}
            write_provenance(
                self._h5, entry=self.entry, program=finalization.program,
                program_version=finalization.program_version,
                config=finalization.provenance_config, inputs=inputs,
                date=finalization.date, host=finalization.host,
            )
            self._bump("provenance_boundaries")
        if self._append_decision is not None:
            written_labels = tuple(sorted(self._append_written_labels))
            commit_append_lineage(
                self._entry_group(),
                self._append_decision,
                written_labels=written_labels,
            )
            lineage = dict(self._append_decision.lineage or {})
            lineage["state"] = "committed"
            self._lineage_expected = json.dumps(
                lineage,
                sort_keys=True,
                separators=(",", ":"),
            )
            self._lineage_dirty = True

    def _write_detector_values(self, finalization: WriterFinalization) -> None:
        calibration = dict(finalization.detector_calibration or {})
        if (not calibration and finalization.global_mask is None
                and finalization.detector_shape is None):
            return
        instrument = self._entry_group().require_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        detector = instrument.require_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        for name in (
            "dist", "poni1", "poni2", "rot1", "rot2", "rot3",
            "detector_name", "x_pixel_size", "y_pixel_size",
            "sensor_material", "sensor_thickness", "parallax",
        ):
            if name in detector:
                del detector[name]
            value = calibration.get(name)
            if value is None or (
                name in {"detector_name", "sensor_material"}
                and not str(value)
            ):
                continue
            if name == "parallax" and type(value) is not bool:
                raise TypeError("detector parallax must be an exact boolean")
            dataset = detector.create_dataset(
                name,
                data=(
                    str(value)
                    if name in {"detector_name", "sensor_material"}
                    else bool(value) if name == "parallax" else float(value)
                ),
            )
            if name in {
                "x_pixel_size", "y_pixel_size", "sensor_thickness",
            }:
                dataset.attrs["units"] = "m"
        if "mask" in detector:
            del detector["mask"]
        if finalization.global_mask is not None:
            mask = np.asarray(finalization.global_mask, dtype=np.int64).ravel()
            if mask.size:
                dataset = detector.create_dataset("mask", data=mask)
                dataset.attrs["description"] = "flat pixel indices, shape (N,)"
        if "detector_shape" in detector:
            del detector["detector_shape"]
        if finalization.detector_shape is not None:
            dataset = detector.create_dataset(
                "detector_shape",
                data=np.asarray(finalization.detector_shape, dtype=np.int64),
            )
            dataset.attrs["description"] = (
                "full-resolution detector (raw) shape (H, W)"
            )

    def extend_append(self, decision: AppendDecision) -> None:
        self._require_active()
        if (self._replacement_configuration is not None or self._append_decision is None
                or decision.disposition is not AppendDisposition.WRITE):
            raise WriterStateError("Append extension requires one active Append writer")
        if decision.skip_labels != self._append_decision.skip_labels:
            raise WriterStateError("Append extension changed the committed prefix")
        old = self._append_decision.write_labels
        new = decision.write_labels
        expanding = new[:len(old)] == old
        truncating = old[:len(new)] == new and tuple(
            sorted(self._append_written_labels)
        ) == new
        if not (expanding or truncating):
            raise WriterStateError("Append extension remapped the pending suffix")
        with self._boundary():
            self._authorize_transaction_mutation()
            stage_append_lineage(self._entry_group(), decision)
            self._append_decision = decision
            lineage = dict(decision.lineage or {})
            lineage["state"] = "pending"
            self._lineage_expected = json.dumps(
                lineage, sort_keys=True, separators=(",", ":"),
            )
            self._lineage_dirty = True

    def adopt_append(self, decision: AppendDecision) -> None:
        """Bind first lineage to an already-owned, still-empty live writer."""
        self._require_active()
        if (self._replacement_configuration is not None or self._append_decision is not None
                or decision.disposition is not AppendDisposition.WRITE):
            raise WriterStateError(
                "Append adoption requires one unbound active writer")
        if (any(self._row_cursors.values()) or self._metadata
                or self._dirty_modes or self._dirty_frames
                or self._dirty_indexed):
            raise WriterStateError(
                "Append lineage must be adopted before the first record")
        with self._boundary():
            self._authorize_transaction_mutation()
            stage_append_lineage(self._entry_group(), decision)
            self._append_decision = decision
            lineage = dict(decision.lineage or {})
            lineage["state"] = "pending"
            self._lineage_expected = json.dumps(
                lineage, sort_keys=True, separators=(",", ":"),
            )
            self._lineage_dirty = True

    @property
    def written_labels(self) -> tuple[int, ...]:
        return tuple(sorted(self._append_written_labels))

    @property
    def append_decision(self) -> AppendDecision | None:
        return self._append_decision

    def _replace_target(self) -> None:
        if self._active_path == self.target:
            return
        last = None
        for _attempt in range(self._replace_attempts):
            try:
                os.replace(self._active_path, self.target)
                self._active_path = self.target
                return
            except OSError as exc:
                last = exc
        raise last  # type: ignore[misc]

    def finish(self, finalization: WriterFinalization | None = None) -> WriterOutcome:
        if self.phase is WriterPhase.FINISHED:
            if self._seeded_document is not None:
                return self._outcome()
            with h5py.File(self.target, "r") as finished:
                require_current_writable_processed_groups(
                    finished,
                    self.entry,
                    container=self.target,
                )
            return self._outcome()
        if self.phase is WriterPhase.ABORTED:
            raise WriterStateError("an aborted writer cannot finish")
        if self.phase is WriterPhase.PARTIAL and self._pending_owner == "write":
            raise WriterStateError("an indeterminate write requires abort")
        if self.phase is WriterPhase.ACTIVE:
            (None if self._replacement_configuration is None or finalization in (None, WriterFinalization()) else (_ for _ in ()).throw(WriterStateError("replacement finish forbids finalization mutation"))); self._finalization = finalization or WriterFinalization()
            self._finish_step = 0
        elif finalization is not None and finalization != self._finalization:
            raise WriterStateError("retry must use the frozen finalization values")
        def checkpoint(*, publish_receipts=True):
            counts = self._finalization.average_finite_counts
            verified = (
                None
                if counts is None or self._fast_regenerable
                else self._verify_average_dirty_evidence(counts)
            )
            self._seal_checkpoint_and_receipts(publish_receipts=publish_receipts, verified=verified)
        def require_terminal_admission() -> None:
            handle = self._h5
            if not isinstance(handle, h5py.File):
                handle = getattr(handle, "_h5", None)
            if not isinstance(handle, h5py.File):
                raise WriterStateError("terminal admission lost the writer handle")
            require_current_writable_processed_groups(
                handle,
                self.entry,
                container=self.target,
            )
        try:
            with self._boundary():
                if self._transaction_binding is None:
                    common = (
                        ("metadata", lambda: self._write_finalization(self._finalization)),
                        ("flush", self._flush_handle),
                    )
                    if self._seeded_document is not None:
                        verification = (
                            ()
                            if self._prepared_manifest_admission is not None
                            else (("verify", self._verify_replacement_manifest),)
                        )
                        steps = (
                            common[0],
                            ("result-seal", self._seal_finite_replacement_result),
                            common[1],
                        ) + verification + (
                            ("admission", require_terminal_admission),
                            ("checkpoint", lambda: checkpoint(publish_receipts=False)),
                            ("close", self._close_handle),
                            ("receipt", self._commit_receipts),
                            (
                                "publication-drop",
                                lambda: self._commit_publication_drops(
                                    self._current_publication_drops(),
                                ),
                            ),
                        )
                    else:
                        steps = common + (
                        ("admission", require_terminal_admission),
                        ("checkpoint", lambda: checkpoint(publish_receipts=False)),
                        ("close", self._close_handle),
                        ("replace", self._replace_target),
                        ("resume", self._resume_pool),
                        ("receipt", self._commit_receipts),
                        (
                            "publication-drop",
                            lambda: self._commit_publication_drops(
                                self._current_publication_drops(),
                            ),
                        ),
                        )
                else:
                    binding = self._transaction_binding

                    def seal_terminal() -> None:
                        terminal = binding.transaction.seal_stream_terminal(
                            binding.attempt, lease=binding.lease,
                        )
                        if (
                            self._stream_terminal is not None
                            and terminal != self._stream_terminal
                        ):
                            raise WriterStateError(
                                "retry changed the exact stream terminal identity"
                            )
                        self._stream_terminal = terminal

                    steps = (
                        (
                            "metadata",
                            lambda: self._write_finalization(self._finalization),
                        ),
                        ("flush", self._flush_handle),
                    ) + (
                        (("verify", self._verify_replacement_manifest),)
                        if self._replacement_configuration is not None else ()
                    )
                    steps += (
                        ("admission", require_terminal_admission),
                        ("checkpoint", checkpoint),
                        ("close", self._close_handle),
                        ("terminal", seal_terminal),
                    )
                    if self._fast_regenerable:
                        # Fast result receipts remain pending until terminal
                        # science and the exact stream terminal are both sealed.
                        steps += (
                            ("receipt", self._commit_receipts),
                            (
                                "publication-drop",
                                lambda: self._commit_publication_drops(
                                    self._current_publication_drops(),
                                ),
                            ),
                        )
                while self._finish_step < len(steps):
                    owner, action = steps[self._finish_step]
                    self._pending_owner = owner
                    action()
                    if owner == "flush":
                        self._bump("flush_boundaries")
                    self._finish_step += 1
            self._pending_owner = None
            self.phase = WriterPhase.FINISHED
            return self._outcome()
        except BaseException as exc:
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(
                f"writer {self._pending_owner} incomplete: {exc}", self._outcome()
            ) from exc

    def abort(self) -> WriterOutcome:
        if self.phase is WriterPhase.FINISHED:
            raise WriterStateError("a finished writer cannot abort")
        if self.phase is WriterPhase.ABORTED:
            return self._outcome()
        try:
            self._pending_owner = "abort"
            self._invalidate_checkpoint_recovery()
            with self._boundary():
                if (
                    self._h5 is not None
                    or self._stream_close_attempt is not None
                ):
                    self._close_handle()
                partial = None
                if (
                    self._transaction_binding is None
                    and self._seeded_document is None
                    and self._active_path is not None
                    and self._active_path != self.target
                ):
                    partial = Path(str(self.target) + ".partial")
                    if self._active_path.exists():
                        os.replace(self._active_path, partial)
                        self._active_path = partial
                        warnings.warn(
                            f"writer abort preserved non-final data at {partial}",
                            RuntimeWarning, stacklevel=2,
                        )
                if (
                    self._transaction_binding is None
                    and self._seeded_document is None
                ):
                    self._resume_pool()
            self._pending.clear()
            self._staged_since_checkpoint.clear()
            self._pending_owner = None
            self.phase = WriterPhase.ABORTED
            return self._outcome()
        except BaseException as exc:
            self._pending_owner = "abort"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer abort incomplete: {exc}", self._outcome()) from exc

    def _outcome(self) -> WriterOutcome:
        partial = None
        if (
            self._seeded_document is None
            and self.phase is WriterPhase.PARTIAL
            and self._active_path is not None
        ):
            partial = self._active_path
        elif (
            self._seeded_document is None
            and self.phase is WriterPhase.ABORTED
            and self._active_path != self.target
        ):
            partial = self._active_path
        return WriterOutcome(
            phase=self.phase, target=self.target, partial_path=partial,
            pending_owner=self._pending_owner,
            operation_vector=self.operation_vector(),
            stream_terminal=self._stream_terminal,
        )


__all__ = [
    "NexusRecordWriter", "RecordWrite", "WriterFinalization",
    "ReplacementManifestTargetChanged",
    "WriterIncomplete", "WriterOperationVector", "WriterOutcome",
    "WriterPhase", "WriterStateError", "WriterTransactionBinding",
]
