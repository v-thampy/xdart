"""Qt-free lifecycle owner for incremental processed-NeXus records."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import shutil
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
    ensure_frames_container,
    quantize_thumbnail,
    replace_frame_record,
    stamp_source_base,
    validate_source_base,
)
from xrd_tools.io.append import (
    AppendDecision,
    AppendDisposition,
    commit_append_lineage,
    stage_append_lineage,
)
from xrd_tools.io.output_transaction import (
    OutputTransaction,
    OutputTransactionError,
    StreamAttempt,
    StreamTerminal,
    TargetLease,
)
from xrd_tools.io.read import relative_source_path
from xrd_tools.io.schema import (
    GI_MODE_KEYS_1D,
    GI_MODE_KEYS_2D,
    INTEGRATED_ROW_ALIGNED,
    MONOTONIC_ATTR,
    PRIMARY_MODE_ATTR,
    THUMBNAIL_LUT_ATTRS,
    mode_subgroup_name,
)
from xrd_tools.session import ResultMode, StageReceipt, get_pool


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

    def __post_init__(self) -> None:
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
            snapshot[key] = value
        if snapshot and self.source_path is None:
            raise ValueError("source snapshot requires source_path")
        if self.source_path is None and self.source_frame_index != 0:
            raise ValueError(
                "absent source_path requires source_frame_index to be exactly 0"
            )
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
class _DurableAbsenceProof:
    """Small selector proving one publication row remains absent."""

    group_name: str
    label: int


@dataclass(frozen=True, slots=True)
class _DurableFrameProof:
    """Compact digest of a complete frame provenance row."""

    label: int
    digest: str


class _EvidenceBuilder:
    def __init__(self) -> None:
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
        self._part(role, "facts", facts)
        self._part(role, "expected", expected_array.tobytes(order="C"))
        self._part(role, "observed", observed_array.tobytes(order="C"))
        self._observed_part(role, "facts", facts)
        self._observed_part(
            role,
            "observed",
            observed_array.tobytes(order="C"),
        )
        self.read_bytes += int(observed_array.nbytes)

    def absent(self, role: str, absent: bool) -> None:
        if not absent:
            raise WriterStateError(f"durability readback expected absent {role}")
        self._part(role, "expected", b"absent")
        self._part(role, "observed", b"absent")
        self._observed_part(role, "observed", b"absent")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def observed_hexdigest(self) -> str:
        return self._observed_digest.hexdigest()


class WriterStateError(RuntimeError):
    pass


class WriterIncomplete(RuntimeError):
    def __init__(self, message: str, outcome: WriterOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


_VECTOR_FIELDS = tuple(WriterOperationVector.__dataclass_fields__)


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
    ) -> None:
        if flush_every is not None and int(flush_every) <= 0:
            raise ValueError(f"flush_every must be > 0 or None; got {flush_every}")
        if int(replace_attempts) < 1:
            raise ValueError("replace_attempts must be >= 1")
        self.target = Path(target)
        self.entry = str(entry)
        self.compression = compression
        self.overwrite = bool(overwrite)
        self.atomic = atomic
        self.flush_every = flush_every
        self.complete_record = bool(complete_record)
        self.source_base = source_base
        self.file_lock = file_lock
        self._pool = get_pool() if pool is None else pool
        self._replace_attempts = int(replace_attempts)
        self._opener = opener
        self._transaction_binding = transaction_binding
        self._append_decision = append_decision
        if append_decision is not None and (
            append_decision.disposition is not AppendDisposition.WRITE
        ):
            raise ValueError("writer requires a WRITE Append decision")
        if append_decision is not None and transaction_binding is None:
            raise ValueError("Append writer requires a live transaction binding")
        if transaction_binding is not None and atomic:
            raise ValueError("transaction-bound writer cannot replace its owned inode")
        self.phase = WriterPhase.NEW
        self._h5 = None
        self._active_path: Path | None = None
        self._pool_owned = False
        self._in_boundary = False
        self._facade = None
        self._pending: dict[tuple[int, ResultMode, str], StageReceipt] = {}
        self._pending_publication_drops: dict[
            tuple[int, ResultMode], int
        ] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._pending_metadata_labels: set[int] = set()
        self._source_paths: list[str] = []
        self._row_cursors: dict[str, dict[int, int]] = {}
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
        self._stream_close_attempt = None
        self._stream_terminal: StreamTerminal | None = None
        self._close_verification_descriptor = None
        self._durable_mode_proofs: dict[
            tuple[str, int], _DurableModeProof
        ] = {}
        self._durable_absence_proofs: dict[
            tuple[str, int], _DurableAbsenceProof
        ] = {}
        self._durable_frame_proofs: dict[int, _DurableFrameProof] = {}

    @property
    def active_path(self) -> Path | None:
        return self._active_path

    @property
    def fresh(self) -> bool:
        return self._fresh

    @property
    def checkpoint_read_volume(self) -> tuple[int, int]:
        return self._checkpoint_rows, self._checkpoint_read_bytes

    def bind_session(self, facade) -> None:
        if self.phase is not WriterPhase.NEW:
            raise WriterStateError("session facade must be bound before begin")
        self._facade = facade

    @contextmanager
    def _boundary(self):
        if self._in_boundary:
            raise WriterStateError("concurrent or re-entrant writer boundary")
        self._in_boundary = True
        try:
            with (nullcontext() if self.file_lock is None else self.file_lock):
                yield
        finally:
            self._in_boundary = False

    def _authorize_transaction_mutation(self) -> None:
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

    def _expected_frame_row(self, record: RecordWrite) -> _ExpectedFrameRow:
        thumbnail = None
        lut = None
        if record.thumbnail is not None:
            thumbnail, lut = quantize_thumbnail(np.asarray(record.thumbnail))
            thumbnail = np.asarray(thumbnail).copy()
            lut = (float(lut[0]), float(lut[1]), str(lut[2]))
        source_path = None
        source_frame_index = None
        if record.source_path is not None:
            source_path = relative_source_path(record.source_path, self.source_base)
            source_frame_index = int(record.source_frame_index)
        timestamp = None if record.timestamp is None else str(record.timestamp)
        thumbnail_mask = None
        if record.thumbnail is not None:
            thumbnail_mask = (
                np.asarray(record.thumbnail_mask, dtype=bool).copy()
                if record.thumbnail_mask is not None
                else ~np.isfinite(np.asarray(record.thumbnail))
            )
            if not thumbnail_mask.any() and record.thumbnail_mask is None:
                thumbnail_mask = None
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
        )

    def _remember_frame_row(self, record: RecordWrite) -> None:
        if not self.complete_record:
            return
        self._dirty_frames[int(record.label)] = self._expected_frame_row(record)

    def _expected_source_fact(self, record: RecordWrite) -> _PersistedSourceFact:
        expected = self._expected_frame_row(record)
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
        frame = self._entry_group().get(f"frames/frame_{label:04d}")
        source = frame.get("source") if isinstance(frame, h5py.Group) else None
        if source is None:
            return _PersistedSourceFact(False, None, None, ())
        if not isinstance(source, h5py.Group):
            raise WriterStateError(f"existing frame {label} source is not a group")
        try:
            stored_path = source["path"][()]
            stored_frame_index = source["frame_index"][()]
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

    def _verify_supplied_source_identity(self, record: RecordWrite) -> None:
        """Require complete source-fact equality for a mode-only sibling."""
        expected = self._expected_source_fact(record)
        observed = self._authoritative_source_fact(int(record.label))
        if observed != expected:
            raise WriterStateError(
                f"existing frame {int(record.label)} complete source fact "
                "does not match the mode-only write"
            )

    def _remember_written_rows(self, records: tuple[RecordWrite, ...]) -> None:
        for record in records:
            self._remember_mode_row(record, dimension="1d")
            self._remember_mode_row(record, dimension="2d")
            if record.write_frame_record:
                self._remember_frame_row(record)
            self._append_written_labels.add(int(record.label))

    def _verify_mode_row(
        self,
        evidence: _EvidenceBuilder,
        expected: _ExpectedModeRow,
    ) -> _DurableModeProof:
        entry = self._entry_group()
        group = entry.get(expected.group_name)
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durability readback missing mode group {expected.group_name}"
            )
        role = f"{group.name}[label={expected.label},row={expected.row}]"
        evidence.text(f"{role}/dimension", expected.dimension, expected.dimension)
        evidence.text(f"{role}/mode", expected.mode, expected.mode)
        evidence.text(f"{role}/group", expected.group_name, expected.group_name)
        evidence.text(f"{role}/row", expected.row, expected.row)
        top_name = f"integrated_{expected.dimension}"
        top = entry.get(top_name)
        if not isinstance(top, h5py.Group):
            raise WriterStateError(f"durability readback missing {top_name}")
        wanted_primary = (
            self._primary_mode_1d
            if expected.dimension == "1d"
            else self._primary_mode_2d
        )
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
        evidence.array(f"{role}/q", expected.radial, np.asarray(group["q"][()]))
        evidence.text(
            f"{role}/q@units",
            expected.unit,
            self._text_value(group["q"].attrs.get("units", "")),
        )
        observed_intensity = np.asarray(group["intensity"][expected.row])
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
        sigma = group.get("sigma")
        if expected.sigma is None:
            if sigma is None:
                evidence.absent(f"{role}/sigma", True)
            else:
                observed_sigma = np.asarray(sigma[expected.row])
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
                np.asarray(sigma[expected.row]),
            )
        if expected.dimension == "2d":
            if expected.azimuthal is None:
                raise WriterStateError("2-D evidence has no azimuthal axis")
            evidence.array(
                f"{role}/chi",
                expected.azimuthal,
                np.asarray(group["chi"][()]),
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

    def _verify_frame_row(
        self,
        evidence: _EvidenceBuilder,
        expected: _ExpectedFrameRow,
    ) -> None:
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
        expected_children = set()
        if expected.thumbnail is not None:
            expected_children.add("thumbnail")
        if expected.thumbnail_mask is not None:
            expected_children.add("thumbnail_mask")
        if expected.source_path is not None:
            expected_children.add("source")
        if expected.timestamp is not None:
            expected_children.add("timestamp")
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

    def _frame_row_digest(self, label: int) -> tuple[str, int]:
        """Digest the complete persisted provenance row without retaining data."""
        frame = self._entry_group().get(f"frames/frame_{int(label):04d}")
        if not isinstance(frame, h5py.Group):
            raise WriterStateError(f"durable frame proof lost label {label}")
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

        def walk(group: h5py.Group, prefix: str) -> None:
            nonlocal read_bytes
            for name in sorted(group.attrs):
                value = scalar(group.attrs[name])
                update(f"{prefix}@{name}", repr(value).encode("utf-8"))
            for name in sorted(group):
                child = group[name]
                role = f"{prefix}/{name}"
                if isinstance(child, h5py.Group):
                    update(role, b"group")
                    walk(child, role)
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

        walk(frame, frame.name)
        return digest.hexdigest(), read_bytes

    def _verify_indexed_row(
        self,
        evidence: _EvidenceBuilder,
        expected: _ExpectedIndexedRow,
    ) -> None:
        group = self._entry_group().get(expected.group_name)
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
        labels = group["frame_index"]
        evidence.array(
            f"{role}/frame_index",
            np.asarray(expected.label, dtype=labels.dtype),
            np.asarray(labels[row]),
        )
        for name, value in expected.values:
            dataset = group.get(name)
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
        group = self._entry_group().get(group_name)
        role = f"/{self.entry}/{group_name}[label={int(label)}]"
        if group is None:
            evidence.absent(role, True)
            return
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durability readback found non-group mode path {group_name}"
            )
        labels_ds = group.get("frame_index")
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

    def _verify_dirty_evidence(
        self,
    ) -> tuple[
        str,
        int,
        int,
        dict[tuple[str, int], _DurableModeProof],
        dict[int, _DurableFrameProof],
    ]:
        aggregate = _EvidenceBuilder()
        read_bytes = 0
        mode_proofs: dict[tuple[str, int], _DurableModeProof] = {}
        frame_proofs: dict[int, _DurableFrameProof] = {}

        def absorb(role: str, evidence: _EvidenceBuilder) -> None:
            nonlocal read_bytes
            digest = evidence.hexdigest()
            aggregate.text(role, digest, digest)
            read_bytes += evidence.read_bytes

        for key in sorted(self._dirty_modes):
            evidence = _EvidenceBuilder()
            proof = self._verify_mode_row(evidence, self._dirty_modes[key])
            mode_proofs[(proof.group_name, proof.label)] = proof
            absorb(f"mode:{proof.group_name}:{proof.label}", evidence)
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
            len(self._dirty_modes)
            + len(self._dirty_absent_modes)
            + len(self._dirty_frames)
            + len(self._dirty_indexed)
        )
        return aggregate.hexdigest(), read_bytes, rows, mode_proofs, frame_proofs

    def _reverify_durable_mode_proof(
        self,
        proof: _DurableModeProof,
    ) -> _EvidenceBuilder:
        if self._h5 is None:
            raise WriterStateError("durable-row proof has no HDF5 reader")
        entry = self._h5.get(self.entry)
        if not isinstance(entry, h5py.Group):
            raise WriterStateError("durable-row proof lost the NeXus entry")
        group = entry.get(proof.group_name)
        if not isinstance(group, h5py.Group):
            raise WriterStateError(
                f"durable-row proof lost mode group {proof.group_name}"
            )
        intensity = group.get("intensity")
        radial = group.get("q")
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
        sigma = None
        if proof.sigma_expected:
            sigma_dataset = group.get("sigma")
            if not isinstance(sigma_dataset, h5py.Dataset):
                raise WriterStateError(
                    f"durable-row proof lost sigma for label {proof.label}"
                )
            sigma = np.asarray(sigma_dataset[proof.row])
        azimuthal = None
        if proof.dimension == "2d":
            chi = group.get("chi")
            if not isinstance(chi, h5py.Dataset):
                raise WriterStateError(
                    f"durable-row proof lost chi for label {proof.label}"
                )
            azimuthal = np.asarray(chi[()])
        expected = _ExpectedModeRow(
            group_name=proof.group_name,
            label=proof.label,
            row=proof.row,
            dimension=proof.dimension,
            mode=proof.mode,
            radial=np.asarray(radial[()]),
            azimuthal=azimuthal,
            intensity=np.asarray(intensity[proof.row]),
            sigma=sigma,
            unit=proof.unit,
            azimuthal_unit=proof.azimuthal_unit,
            two_d_kind=proof.two_d_kind,
            source_shape=proof.source_shape,
        )
        evidence = _EvidenceBuilder()
        observed = self._verify_mode_row(evidence, expected)
        if observed.digest != proof.digest:
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
                for key in sorted(self._durable_mode_proofs):
                    proof = self._durable_mode_proofs[key]
                    evidence = self._reverify_durable_mode_proof(proof)
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
                for label in sorted(self._durable_frame_proofs):
                    proof = self._durable_frame_proofs[label]
                    observed, observed_bytes = self._frame_row_digest(label)
                    if observed != proof.digest:
                        raise WriterStateError(
                            f"durable frame proof changed for label {label}"
                        )
                    aggregate.text(f"frame:{label}", proof.digest, observed)
                    read_bytes += observed_bytes
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

    def _clear_dirty_evidence(self) -> None:
        self._dirty_modes.clear()
        self._dirty_absent_modes.clear()
        self._dirty_frames.clear()
        self._dirty_indexed.clear()
        self._lineage_dirty = False

    def _entry_group(self):
        return self._h5.require_group(self.entry)

    def _bump(self, name: str, value: int = 1) -> None:
        self._vector[name] += int(value)

    def operation_vector(self) -> WriterOperationVector:
        return WriterOperationVector(**self._vector)

    def reset_operation_vector(self) -> None:
        self._vector = {name: 0 for name in _VECTOR_FIELDS}

    def _load_cursor_from(self, entry: h5py.Group, name: str) -> dict[int, int]:
        group = entry.get(name)
        if group is None:
            return {}
        if not isinstance(group, h5py.Group) or "frame_index" not in group:
            raise WriterStateError(f"{name} has no indexed frame_index")
        if not isinstance(group["frame_index"], h5py.Dataset):
            raise WriterStateError(f"{name}/frame_index is not a dataset")
        labels = [int(x) for x in np.asarray(group["frame_index"][()]).ravel()]
        if len(labels) != len(set(labels)):
            raise WriterStateError(f"{name}/frame_index contains duplicate labels")
        self._bump("frame_index_scan_rows", len(labels))
        return {label: row for row, label in enumerate(labels)}

    def _load_cursor(self, name: str) -> dict[int, int]:
        return self._load_cursor_from(self._entry_group(), name)

    def _validate_existing_contract(self) -> dict[str, dict[int, int]]:
        with h5py.File(self.target, "r") as h5:
            entry = h5.get(self.entry)
            if entry is None:
                return {
                    name: {} for name in (
                        "integrated_1d", "integrated_2d", "scan_data",
                        "per_frame_geometry",
                    )
                }
            if self.complete_record and self.source_base:
                validate_source_base(entry, self.source_base)
            for name, requested in (
                ("integrated_1d", self._primary_mode_1d),
                ("integrated_2d", self._primary_mode_2d),
            ):
                group = entry.get(name)
                if group is None:
                    continue
                existing = group.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)
                if isinstance(existing, bytes):
                    existing = existing.decode("utf-8", errors="replace")
                if str(existing) != requested:
                    raise ValueError(
                        f"{name} primary mode {existing!r} != requested "
                        f"{requested!r}; sparse writes cannot reinterpret "
                        "untouched rows"
                    )
            cursors = {
                name: self._load_cursor_from(entry, name)
                for name in (
                    "integrated_1d", "integrated_2d", "scan_data",
                    "per_frame_geometry",
                )
            }
            for name in ("integrated_1d", "integrated_2d"):
                group = entry.get(name)
                if group is None:
                    continue
                for child_name, child in group.items():
                    if isinstance(child, h5py.Group):
                        nested = f"{name}/{child_name}"
                        cursors[nested] = self._load_cursor_from(entry, nested)
            return cursors

    def begin(
        self,
        *,
        metadata=None,
        primary_mode_1d: str = DEFAULT_MODE_KEY,
        primary_mode_2d: str = DEFAULT_MODE_KEY,
    ) -> None:
        if self.phase is not WriterPhase.NEW:
            raise WriterStateError(f"begin requires NEW, got {self.phase.value}")
        self._primary_mode_1d = str(primary_mode_1d or DEFAULT_MODE_KEY)
        self._primary_mode_2d = str(primary_mode_2d or DEFAULT_MODE_KEY)
        if (self._primary_mode_1d != DEFAULT_MODE_KEY
                and self._primary_mode_1d not in GI_MODE_KEYS_1D):
            raise ValueError(f"unknown 1d primary mode {self._primary_mode_1d!r}")
        if (self._primary_mode_2d != DEFAULT_MODE_KEY
                and self._primary_mode_2d not in GI_MODE_KEYS_2D):
            raise ValueError(f"unknown 2d primary mode {self._primary_mode_2d!r}")
        try:
            with self._boundary():
                binding = self._transaction_binding
                exists = self.target.exists()
                logical_exists = (
                    bool(binding.transaction.stream_base_snapshot.exists)
                    if binding is not None
                    else exists
                )
                use_atomic = False if binding is not None else (
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
                        f".{self.target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                    )
                if binding is None:
                    self._pool.pause(self.target)
                    self._pool_owned = True
                existing_cursors = None
                if logical_exists and not self.overwrite:
                    existing_cursors = self._validate_existing_contract()
                if use_atomic:
                    self.target.parent.mkdir(parents=True, exist_ok=True)
                    if preserve_existing:
                        shutil.copyfile(self.target, self._active_path)
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
                # The opener may be writing an atomic private path.  Root
                # provenance names the logical record, never that temporary
                # implementation detail; an Append also refreshes a relocated
                # legacy file's path to its current truthful location.
                self._h5.attrs["file_name"] = str(self.target.resolve())
                if self.complete_record and self.source_base:
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
            self.phase = WriterPhase.ACTIVE
        except BaseException as exc:
            self._pending_owner = "begin"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer begin incomplete: {exc}", self._outcome()) from exc

    def _require_active(self) -> None:
        if self.phase is not WriterPhase.ACTIVE:
            raise WriterStateError(f"writer requires ACTIVE, got {self.phase.value}")

    def _verify_cursor(self, name: str, label: int) -> None:
        cursor = self._row_cursors[name]
        row = cursor.get(label)
        if row is None:
            return
        group = self._entry_group().get(name)
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
        mode = str(mode or DEFAULT_MODE_KEY)
        if mode == primary:
            return top
        if primary == DEFAULT_MODE_KEY:
            raise ValueError(
                f"named {dimension} mode {mode!r} requires a named primary mode"
            )
        if mode not in allowed:
            raise ValueError(f"unknown {dimension} mode {mode!r}")
        return f"{top}/{mode_subgroup_name(mode)}"

    def _validate_records(self, records: tuple[RecordWrite, ...]) -> None:
        labels = [int(record.label) for record in records]
        if len(labels) != len(set(labels)):
            raise ValueError(f"dirty batch contains duplicate labels: {labels}")
        one_d = [record for record in records if record.result_1d is not None]
        two_d = [record for record in records if record.result_2d is not None]
        groups_1d: dict[str, list[RecordWrite]] = {}
        groups_2d: dict[str, list[RecordWrite]] = {}
        for record in one_d:
            name = self._mode_cursor_name("1d", record.mode_1d)
            self._row_cursors.setdefault(name, {})
            self._verify_cursor(name, int(record.label))
            groups_1d.setdefault(name, []).append(record)
        for record in two_d:
            name = self._mode_cursor_name("2d", record.mode_2d)
            self._row_cursors.setdefault(name, {})
            self._verify_cursor(name, int(record.label))
            groups_2d.setdefault(name, []).append(record)
        for record in records:
            label = int(record.label)
            self._verify_cursor("scan_data", label)
            self._verify_cursor("per_frame_geometry", label)
        entry = self._entry_group()
        for top, groups in (
            ("integrated_1d", groups_1d), ("integrated_2d", groups_2d),
        ):
            if any(name != top for name in groups):
                if entry.get(top) is None and top not in groups:
                    raise ValueError(
                        f"named modes require an established {top} primary group"
                    )
        for name, grouped in groups_1d.items():
            validate_integrated_stack_write(
                entry, frame_indices=[int(r.label) for r in grouped],
                results_1d=[r.result_1d for r in grouped],
                group_name_1d=name, allow_rebuild=False,
            )
        for name, grouped in groups_2d.items():
            validate_integrated_stack_write(
                entry, frame_indices=[int(r.label) for r in grouped],
                results_2d=[r.result_2d for r in grouped],
                group_name_2d=name, allow_rebuild=False,
            )
        existing_scan_data = self._entry_group().get("scan_data")
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
            if record.source_path is not None:
                relative_source_path(record.source_path, self.source_base)
            frame = entry.get(f"frames/frame_{int(record.label):04d}")
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
                    _EvidenceBuilder(), self._expected_frame_row(record),
                )
            elif isinstance(frame, h5py.Group) and not record.write_frame_record:
                self._verify_supplied_source_identity(record)
            if existing_columns is not None:
                unexpected = set(map(str, record.metadata)) - existing_columns
                if unexpected:
                    raise ValueError(
                        f"dirty metadata adds columns absent on disk: {sorted(unexpected)}"
                    )

    @staticmethod
    def _modes(record: RecordWrite) -> tuple[ResultMode, ...]:
        modes = []
        if record.result_1d is not None:
            modes.append(ResultMode.one_d(str(record.mode_1d or DEFAULT_MODE_KEY)))
        if record.result_2d is not None:
            modes.append(ResultMode.two_d(str(record.mode_2d or DEFAULT_MODE_KEY)))
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
            if str(record.mode_1d or DEFAULT_MODE_KEY) == self._primary_mode_1d
        ]
        primary_2d = [
            record for record in two_d
            if str(record.mode_2d or DEFAULT_MODE_KEY) == self._primary_mode_2d
        ]
        extra_1d: dict[str, list[RecordWrite]] = {}
        extra_2d: dict[str, list[RecordWrite]] = {}
        for record in one_d:
            mode = str(record.mode_1d or DEFAULT_MODE_KEY)
            if mode != self._primary_mode_1d:
                extra_1d.setdefault(mode, []).append(record)
        for record in two_d:
            mode = str(record.mode_2d or DEFAULT_MODE_KEY)
            if mode != self._primary_mode_2d:
                extra_2d.setdefault(mode, []).append(record)
        validation_complete = False
        try:
            with self._boundary():
                self._validate_records(batch)
                validation_complete = True
                self._authorize_transaction_mutation()
                labels_1d = [int(r.label) for r in primary_1d]
                labels_2d = [int(r.label) for r in primary_2d]
                if primary_1d and primary_2d and labels_1d == labels_2d:
                    write_integrated_stack(
                        self._entry_group(), frame_indices=labels_1d,
                        results_1d=[r.result_1d for r in primary_1d],
                        results_2d=[r.result_2d for r in primary_2d],
                        primary_mode_1d=self._primary_mode_1d,
                        primary_mode_2d=self._primary_mode_2d,
                        compression=self.compression,
                        known_rows_1d=self._row_cursors["integrated_1d"],
                        known_rows_2d=self._row_cursors["integrated_2d"],
                    )
                else:
                    if primary_1d:
                        write_integrated_stack(
                            self._entry_group(), frame_indices=labels_1d,
                            results_1d=[r.result_1d for r in primary_1d],
                            primary_mode_1d=self._primary_mode_1d,
                            compression=self.compression,
                            known_rows_1d=self._row_cursors["integrated_1d"],
                        )
                    if primary_2d:
                        write_integrated_stack(
                            self._entry_group(), frame_indices=labels_2d,
                            results_2d=[r.result_2d for r in primary_2d],
                            primary_mode_2d=self._primary_mode_2d,
                            compression=self.compression,
                            known_rows_2d=self._row_cursors["integrated_2d"],
                        )
                if extra_1d or extra_2d:
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
                    )
                if primary_1d:
                    self._entry_group().attrs.setdefault(
                        "default", "integrated_1d",
                    )
                if self.complete_record:
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
                        )
                self._remember_written_rows(batch)
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
                self._pending[(receipt.label, receipt.mode, receipt.target)] = receipt
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

    def _current_publication_drops(
        self,
    ) -> tuple[tuple[int, ResultMode, int], ...]:
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
            current.append((label, mode, revision))
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
        batch: tuple[tuple[int, ResultMode, int], ...],
    ) -> None:
        for label, mode, revision in batch:
            self._facade.commit_publication_drop(label, mode, revision)
            self._pending_publication_drops.pop((label, mode), None)

    def _flush_handle(self) -> None:
        self._h5.flush()

    def _seal_checkpoint_and_receipts(self, *, publish_receipts: bool = True) -> None:
        batch = self._current_receipts() if publish_receipts else ()
        dropped = self._current_publication_drops() if publish_receipts else ()
        (digest, read_bytes, rows, mode_proofs,
         frame_proofs) = self._verify_dirty_evidence()
        binding = self._transaction_binding
        # A dynamic same-run lineage facade cannot publish additive H10
        # durability until H23 has committed that rollback-capable epoch.  It
        # still receives the exact verified batch below, but owns it only as a
        # pending overlay.  Static R7 facades lack this opt-in and retain the
        # existing immediate irreversible-floor contract.
        defer_epoch = bool(
            self._facade is not None
            and getattr(self._facade, "defer_epoch_durability", False)
        )
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
                for label, mode, _revision in dropped:
                    group_name = self._mode_cursor_name(mode.kind, mode.key)
                    proof = _DurableAbsenceProof(group_name, int(label))
                    self._durable_absence_proofs[(group_name, int(label))] = proof
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
        if dropped:
            self._commit_publication_drops(dropped)
        self._checkpoint_rows += rows
        self._checkpoint_read_bytes += read_bytes
        self._clear_dirty_evidence()

    def _close_handle(self) -> None:
        allow_unverified = self._pending_owner == "abort"
        binding = self._transaction_binding
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
            if retained_proofs:
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
        label = int(label)
        group_name = self._mode_cursor_name(mode.kind, mode.key)
        cursor = self._row_cursors.setdefault(group_name, {})
        group = self._entry_group().get(group_name)
        if group is None:
            cursor.pop(label, None)
            self._dirty_modes.pop((group_name, label), None)
            self._dirty_absent_modes.add((group_name, label))
            return
        if not isinstance(group, h5py.Group):
            raise WriterStateError(f"{group_name} is not an indexed mode group")
        labels_ds = group.get("frame_index")
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
            for name in INTEGRATED_ROW_ALIGNED:
                dataset = group.get(name)
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
                dataset = group.get(name)
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
        try:
            with self._boundary():
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
        ):
            if name in detector:
                del detector[name]
            value = calibration.get(name)
            if value is None or (name == "detector_name" and not str(value)):
                continue
            dataset = detector.create_dataset(
                name,
                data=(str(value) if name == "detector_name" else float(value)),
            )
            if name in {"x_pixel_size", "y_pixel_size"}:
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
        if (self._append_decision is None
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
        if (self._append_decision is not None
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
            return self._outcome()
        if self.phase is WriterPhase.ABORTED:
            raise WriterStateError("an aborted writer cannot finish")
        if self.phase is WriterPhase.PARTIAL and self._pending_owner == "write":
            raise WriterStateError("an indeterminate write requires abort")
        if self.phase is WriterPhase.ACTIVE:
            self._finalization = finalization or WriterFinalization()
            self._finish_step = 0
        elif finalization is not None and finalization != self._finalization:
            raise WriterStateError("retry must use the frozen finalization values")
        try:
            with self._boundary():
                if self._transaction_binding is None:
                    steps = (
                        ("metadata", lambda: self._write_finalization(self._finalization)),
                        ("flush", self._flush_handle),
                        ("checkpoint", lambda: self._seal_checkpoint_and_receipts(
                            publish_receipts=False,
                        )),
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
                        ("metadata", lambda: self._write_finalization(self._finalization)),
                        ("flush", self._flush_handle),
                        ("checkpoint", self._seal_checkpoint_and_receipts),
                        ("close", self._close_handle),
                        ("terminal", seal_terminal),
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
            with self._boundary():
                if (
                    self._h5 is not None
                    or self._stream_close_attempt is not None
                ):
                    self._close_handle()
                partial = None
                if (
                    self._transaction_binding is None
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
                if self._transaction_binding is None:
                    self._resume_pool()
            self._pending.clear()
            self._pending_owner = None
            self.phase = WriterPhase.ABORTED
            return self._outcome()
        except BaseException as exc:
            self._pending_owner = "abort"
            self.phase = WriterPhase.PARTIAL
            raise WriterIncomplete(f"writer abort incomplete: {exc}", self._outcome()) from exc

    def _outcome(self) -> WriterOutcome:
        partial = None
        if self.phase is WriterPhase.PARTIAL and self._active_path is not None:
            partial = self._active_path
        elif self.phase is WriterPhase.ABORTED and self._active_path != self.target:
            partial = self._active_path
        return WriterOutcome(
            phase=self.phase, target=self.target, partial_path=partial,
            pending_owner=self._pending_owner,
            operation_vector=self.operation_vector(),
            stream_terminal=self._stream_terminal,
        )


__all__ = [
    "NexusRecordWriter", "RecordWrite", "WriterFinalization",
    "WriterIncomplete", "WriterOperationVector", "WriterOutcome",
    "WriterPhase", "WriterStateError", "WriterTransactionBinding",
]
