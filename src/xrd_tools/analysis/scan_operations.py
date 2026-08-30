"""Blocking, Qt-free retained-scan analysis operations.

The public values in this module are deliberately operation-specific.  Source
handles are local to one call, results are detached values, and no operation
writes an artifact or participates in acquisition.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import struct
import threading
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from xrd_tools.analysis.plans import RoiSignal, run_roi_signals
from xrd_tools.core.scan import SourceCapabilities, SourceKind, SourceSpec, coerce_source_kind
from xrd_tools.io.metadata import read_image_metadata_observed
from xrd_tools.sources.discover import discover_scans
from xrd_tools.sources.registry import guess_source_kind, open_source
from xrd_tools.sources.selection import image_series_spec
__all__ = [
    "AnalysisDisposition", "AnalysisSourceLeaseRefused", "AnalysisSourceReceipt",
    "MetadataColumn", "MetadataTablePlan",
    "MetadataTableResult", "MetadataTableRequalificationPlan",
    "MetadataTableRequalificationResult", "ScanPlotPlan", "ScanPlotResult", "RoiPreviewPlan",
    "RoiPreviewResult", "RoiScanPlan", "RoiScanResult", "run_metadata_table",
    "run_metadata_table_requalification", "run_scan_plot", "run_roi_preview", "run_roi_scan",
    "analysis_canonical_fingerprint", "requalified_analysis_source",
]
# SSRL SPEC scans commonly expose roughly 60 point columns plus the complete
# 159-motor #O/#P snapshot.  Keep that physical catalog admissible while the
# independent 64 MiB canonical table budget remains the actual memory bound.
_MAX_TABLE_ROWS, _MAX_TABLE_COLUMNS = 100_000, 256
_MAX_CANDIDATES, _MAX_CANDIDATE_BYTES = 256, 1 << 20
_MAX_TABLE_BYTES = _MAX_PLOT_BYTES = 64 << 20
_MAX_PREVIEW_BYTES = _MAX_ROI_BYTES = 64 << 20
_MAX_PLOT_TRACES, _METADATA_INPUT_BYTES = 16, 1 << 20
_OUTPUT_QNAN_BITS = 0x7FF8000000000000
_CANONICAL_PREFIX = b"xrd_tools.analysis.canonical.v1\x00"
_CANONICAL_TAGS = {
    type(None): 0x00, bool: 0x01, int: 0x02, float: 0x03, str: 0x04,
    bytes: 0x05, Path: 0x06, Enum: 0x07, tuple: 0x08,
    Mapping: 0x09, np.ndarray: 0x0A,
}
FileRevision = tuple[int, int, int, int, int, int]
class InvalidCanonicalValue(ValueError):
    pass
class AnalysisSourceLeaseRefused(RuntimeError):
    """An exact retained source could not remain qualified for one body."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
class AnalysisDisposition(str, Enum):
    COMPLETED = "completed"
    REFUSED = "refused"
    CANCELLED = "cancelled"
def _canonical_array(value: Any, *, allow_missing: bool) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype.kind not in "biufc" or value.dtype.fields is not None:
        raise InvalidCanonicalValue("unsupported canonical ndarray")
    array = np.ascontiguousarray(value)
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        valid_missing = False
        if allow_missing and array.dtype.str == "<f8":
            invalid = ~np.isfinite(array)
            valid_missing = bool(
                np.all(array.view("<u8")[invalid] == _OUTPUT_QNAN_BITS)
            )
        if not valid_missing:
            raise InvalidCanonicalValue("nonfinite canonical ndarray")
    return array
def _canonical_frame(
    value: Any, active: set[int], emit=None, *, allow_missing: bool = False,
) -> int:
    recursive = isinstance(value, (tuple, list, Mapping, np.ndarray, Enum))
    marker = id(value)
    if recursive:
        if marker in active:
            raise InvalidCanonicalValue("cyclic canonical value")
        active.add(marker)
    try:
        if value is None:
            tag, payload = 0x00, b""
        elif type(value) is bool:
            tag, payload = 0x01, b"\x01" if value else b"\x00"
        elif isinstance(value, Enum):
            identity = f"{type(value).__module__}.{type(value).__qualname__}"
            size = _canonical_frame(identity, active, allow_missing=allow_missing)
            size += _canonical_frame(value.value, active, allow_missing=allow_missing)
            if emit is not None:
                emit(b"\x07" + struct.pack(">Q", size))
                _canonical_frame(identity, active, emit, allow_missing=allow_missing)
                _canonical_frame(value.value, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif type(value) is int:
            magnitude = abs(value)
            raw = b"" if magnitude == 0 else magnitude.to_bytes(
                (magnitude.bit_length() + 7) // 8, "big"
            )
            tag, payload = 0x02, (b"\x01" if value < 0 else b"\x00") + raw
        elif type(value) is float:
            if not math.isfinite(value):
                raise InvalidCanonicalValue("nonfinite canonical float")
            tag, payload = 0x03, struct.pack(">d", value)
        elif type(value) is str:
            tag, payload = 0x04, value.encode("utf-8")
        elif type(value) is bytes:
            tag, payload = 0x05, value
        elif isinstance(value, Path):
            tag, payload = 0x06, os.fspath(value).encode("utf-8")
        elif type(value) in {list, tuple}:
            items = tuple(value) if type(value) is list else value
            size = 8 + sum(
                _canonical_frame(item, active, allow_missing=allow_missing)
                for item in items
            )
            if emit is not None:
                emit(b"\x08" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for item in items:
                    _canonical_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise InvalidCanonicalValue("canonical mapping keys must be exact str")
            items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
            size = 8 + sum(
                _canonical_frame(key, active, allow_missing=allow_missing)
                + _canonical_frame(item, active, allow_missing=allow_missing)
                for key, item in items
            )
            if emit is not None:
                emit(b"\x09" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for key, item in items:
                    _canonical_frame(key, active, emit, allow_missing=allow_missing)
                    _canonical_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif type(value) is np.ndarray:
            array = _canonical_array(value, allow_missing=allow_missing)
            dtype_size = _canonical_frame(array.dtype.str, active, allow_missing=allow_missing)
            size = dtype_size + 8 + 8 * array.ndim + array.nbytes
            if emit is not None:
                emit(b"\x0a" + struct.pack(">Q", size))
                _canonical_frame(array.dtype.str, active, emit, allow_missing=allow_missing)
                emit(struct.pack(">Q", array.ndim))
                for dimension in array.shape:
                    emit(struct.pack(">Q", dimension))
                emit(memoryview(array).cast("B"))
            return 9 + size
        else:
            raise InvalidCanonicalValue(f"unsupported canonical value {type(value)!r}")
        if emit is not None:
            emit(bytes((tag,)) + struct.pack(">Q", len(payload)))
            emit(payload)
        return 9 + len(payload)
    finally:
        if recursive:
            active.discard(marker)
def _canonical_stream(value: Any, emit, *, allow_missing: bool = False) -> int:
    emit(_CANONICAL_PREFIX)
    return len(_CANONICAL_PREFIX) + _canonical_frame(
        value, set(), emit, allow_missing=allow_missing,
    )
def _canonical_bytes(value: Any, *, allow_missing: bool = False) -> bytes:
    output = bytearray()
    _canonical_stream(value, output.extend, allow_missing=allow_missing)
    return bytes(output)
def _canonical_frame_charge(value: Any, *, allow_missing: bool = False) -> int:
    return _canonical_frame(value, set(), allow_missing=allow_missing)
def _canonical_charge(value: Any, *, allow_missing: bool = False) -> int:
    return len(_CANONICAL_PREFIX) + _canonical_frame_charge(
        value, allow_missing=allow_missing,
    )
def _digest(value: Any, *, allow_missing: bool = False) -> str:
    digest = hashlib.sha256()
    _canonical_stream(value, digest.update, allow_missing=allow_missing)
    return digest.hexdigest()
class _PublicFingerprintContainer(Enum):
    LIST = "list"
    TUPLE = "tuple"
    MAPPING = "mapping"
def _public_fingerprint_projection(value: Any, active: set[int]) -> Any:
    recursive = type(value) in {list, tuple} or isinstance(value, Mapping)
    marker = id(value)
    if recursive:
        if marker in active:
            raise InvalidCanonicalValue("cyclic canonical value")
        active.add(marker)
    try:
        if type(value) is list:
            return (
                _PublicFingerprintContainer.LIST,
                tuple(_public_fingerprint_projection(item, active) for item in value),
            )
        if type(value) is tuple:
            return (
                _PublicFingerprintContainer.TUPLE,
                tuple(_public_fingerprint_projection(item, active) for item in value),
            )
        if isinstance(value, Mapping):
            return (
                _PublicFingerprintContainer.MAPPING,
                {
                    key: _public_fingerprint_projection(item, active)
                    for key, item in value.items()
                },
            )
        return value
    finally:
        if recursive:
            active.remove(marker)
def analysis_canonical_fingerprint(
    domain: str, value: Any, *, allow_missing: bool = False,
) -> str:
    """Hash one type-framed value in an explicit public identity domain."""
    if type(domain) is not str or not domain or domain.strip() != domain:
        raise InvalidCanonicalValue("fingerprint domain must be an exact token")
    if type(allow_missing) is not bool:
        raise InvalidCanonicalValue("allow_missing must be an exact bool")
    return _digest(
        (
            "analysis-public-fingerprint-v1",
            domain,
            _public_fingerprint_projection(value, set()),
        ),
        allow_missing=allow_missing,
    )
def _missing_numeric_array(size: int) -> np.ndarray:
    result = np.empty(int(size), dtype="<f8")
    result.view("<u8")[:] = _OUTPUT_QNAN_BITS
    return result
def _owned(array: Any, *, dtype=None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    if result.dtype.kind == "O":
        raise ValueError("object arrays are not publishable")
    result.flags.writeable = False
    return result
def _owned_missing(array: Any) -> np.ndarray:
    result = np.array(array, dtype="<f8", copy=True, order="C")
    result.view("<u8")[np.isnan(result)] = _OUTPUT_QNAN_BITS
    result.flags.writeable = False
    return result
def _freeze(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    if isinstance(value, np.generic):
        raise InvalidCanonicalValue("NumPy scalar is not canonical")
    if isinstance(value, (Mapping, list, tuple)):
        if id(value) in active:
            raise InvalidCanonicalValue("cyclic value")
        active.add(id(value))
        try:
            if isinstance(value, Mapping):
                if any(type(key) is not str for key in value):
                    raise InvalidCanonicalValue("mapping keys must be exact str")
                return MappingProxyType({key: _freeze(item, active) for key, item in value.items()})
            return tuple(_freeze(item, active) for item in value)
        finally:
            active.remove(id(value))
    if isinstance(value, np.ndarray):
        _canonical_frame(value, set())
        return _owned(value)
    if value is None or type(value) in {bool, int, float, str, bytes} or isinstance(value, (Path, Enum)):
        _canonical_frame(value, set())
        return value
    raise InvalidCanonicalValue(f"unsupported value {type(value)!r}")
def _freeze_spec(spec: SourceSpec) -> SourceSpec:
    return SourceSpec(
        spec.uri, spec.kind, metadata_uri=spec.metadata_uri, entry=spec.entry,
        options=_freeze(dict(spec.options)),
    )
def _tuple_once(value: Any) -> tuple[Any, ...]:
    if type(value) not in {tuple, list}:
        raise InvalidCanonicalValue("value must be an exact tuple or list")
    return tuple(value)
def _path_revision(path: Path) -> FileRevision | None:
    try:
        state = Path(path).stat()
    except OSError:
        return None
    return (
        state.st_mode, state.st_dev, state.st_ino, state.st_size,
        state.st_mtime_ns, state.st_ctime_ns,
    )
def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
@dataclass(frozen=True, slots=True)
class CandidateProjection:
    source_spec: SourceSpec
    fingerprint: str
    storage_bytes: int
@dataclass(frozen=True, slots=True)
class AnalysisSourceReceipt:
    schema_version: str
    source_spec: SourceSpec
    source_spec_digest: str
    lexical_root: Path
    resolved_root: Path
    resolved_kind: SourceKind
    resolved_entry: str | None
    resolved_scan: str | None
    primary_state: FileRevision | None
    primary_post_state: FileRevision | None
    labels: tuple[int, ...]
    labels_digest: str
    catalog_digest: str
    metadata_observation_modes: tuple[str, ...]
    dependency_revisions: tuple[
        tuple[Path, Path, FileRevision | None], ...
    ]
    source_fingerprint: str
@dataclass(frozen=True, slots=True)
class MetadataColumn:
    name: str
    kind: str
    missing: str
    numeric: np.ndarray | None
    text: tuple[str | None, ...] | None
    storage_bytes: int
@dataclass(frozen=True, slots=True)
class MetadataTablePlan:
    source: SourceSpec | str | Path
    selection: str = "exact"
    kind: SourceKind | str | None = None
    recursive: bool = False
    entry: str | None = None
    scan: str | None = None
    image_dir: str | Path | None = None
    image_stem: str | None = None
    source_root: str | Path | None = None
    metadata_format: str | None = None
    def __post_init__(self) -> None:
        if isinstance(self.source, SourceSpec):
            object.__setattr__(self, "source", _freeze_spec(self.source))
@dataclass(frozen=True, slots=True)
class MetadataTableResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    receipt: AnalysisSourceReceipt | None = None
    labels: tuple[int, ...] = ()
    columns: tuple[MetadataColumn, ...] = ()
    declared_scanned_positioner: str | None = None
    selected_scanned_positioner: str | None = None
    candidates: tuple[CandidateProjection, ...] = ()
    policy_fingerprint: str = ""
    table_fingerprint: str = ""
    storage_bytes: int = 0
@dataclass(frozen=True, slots=True)
class MetadataTableRequalificationPlan:
    receipt: AnalysisSourceReceipt
    table_fingerprint: str
    def __post_init__(self) -> None:
        if type(self.receipt) is not AnalysisSourceReceipt:
            raise TypeError("metadata requalification requires an exact receipt")
        if type(self.table_fingerprint) is not str or not self.table_fingerprint:
            raise ValueError("metadata requalification requires a table fingerprint")
@dataclass(frozen=True, slots=True)
class MetadataTableRequalificationResult:
    disposition: AnalysisDisposition
    code: str
    receipt: AnalysisSourceReceipt | None = None
    table_fingerprint: str = ""
@dataclass(frozen=True, slots=True)
class ScanPlotPlan:
    x: str | None = None
    y: tuple[str, ...] = ()
    normalization: str | None = None
    def __post_init__(self) -> None:
        values = _tuple_once(self.y)
        if any(type(value) is not str for value in values) or any(
            value is not None and type(value) is not str
            for value in (self.x, self.normalization)
        ):
            raise InvalidCanonicalValue("scan plot names must be exact strings")
        object.__setattr__(self, "y", values)
@dataclass(frozen=True, slots=True)
class ScanPlotResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    table_fingerprint: str = ""
    roi_fingerprint: str | None = None
    x_name: str = ""
    x: np.ndarray | None = None
    trace_names: tuple[str, ...] = ()
    trace_origins: tuple[str, ...] = ()
    original_identities: tuple[str, ...] = ()
    traces: tuple[np.ndarray, ...] = ()
    normalization: str | None = None
    normalization_invalid_count: int = 0
    policy_fingerprint: str = ""
    storage_bytes: int = 0
@dataclass(frozen=True, slots=True)
class RoiPreviewPlan:
    receipt: AnalysisSourceReceipt
    table_fingerprint: str
    labels: tuple[int, ...]
    label: int
    def __post_init__(self) -> None:
        labels = _tuple_once(self.labels)
        if any(type(value) is not int for value in labels) or type(self.label) is not int:
            raise InvalidCanonicalValue("ROI labels must be exact integers")
        object.__setattr__(self, "labels", labels)
    @classmethod
    def from_table(cls, table: MetadataTableResult, *, label: int) -> "RoiPreviewPlan":
        if table.receipt is None:
            raise ValueError("completed metadata table required")
        return cls(table.receipt, table.table_fingerprint, table.labels, label)
@dataclass(frozen=True, slots=True)
class RoiPreviewResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    receipt: AnalysisSourceReceipt | None = None
    table_fingerprint: str = ""
    labels: tuple[int, ...] = ()
    label: int | None = None
    image: np.ndarray | None = None
    policy_fingerprint: str = ""
    result_fingerprint: str = ""
    storage_bytes: int = 0
@dataclass(frozen=True, slots=True)
class RoiScanPlan:
    receipt: AnalysisSourceReceipt
    table_fingerprint: str
    labels: tuple[int, ...]
    selected_labels: tuple[int, ...] | None
    signals: tuple[RoiSignal, ...]
    mask: Any = None
    mask_saturation: bool = False
    def __post_init__(self) -> None:
        labels = _tuple_once(self.labels)
        selected = None if self.selected_labels is None else _tuple_once(self.selected_labels)
        signals = _tuple_once(self.signals)
        if any(type(value) is not int for value in labels + (() if selected is None else selected)):
            raise InvalidCanonicalValue("ROI labels must be exact integers")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "selected_labels", selected)
        object.__setattr__(self, "signals", signals)
        if isinstance(self.mask, np.ndarray):
            object.__setattr__(self, "mask", _owned(self.mask))
    @classmethod
    def from_table(
        cls, table: MetadataTableResult, *, signals: tuple[RoiSignal, ...],
        selected_labels: tuple[int, ...] | None = None, mask: Any = None,
        mask_saturation: bool = False,
    ) -> "RoiScanPlan":
        if table.receipt is None:
            raise ValueError("completed metadata table required")
        return cls(table.receipt, table.table_fingerprint, table.labels,
                   selected_labels, signals, mask, mask_saturation)
@dataclass(frozen=True, slots=True)
class RoiScanResult:
    disposition: AnalysisDisposition
    code: str
    diagnostics: tuple[str, ...] = ()
    receipt: AnalysisSourceReceipt | None = None
    table_fingerprint: str = ""
    requested_labels: tuple[int, ...] = ()
    completed_labels: tuple[int, ...] = ()
    signal_names: tuple[str, ...] = ()
    signal_values: tuple[np.ndarray, ...] = ()
    valid_counts: tuple[np.ndarray, ...] = ()
    no_raw_labels: tuple[int, ...] = ()
    policy_fingerprint: str = ""
    result_fingerprint: str = ""
    storage_bytes: int = 0
class _Refusal(Exception):
    def __init__(self, code: str, diagnostics: tuple[str, ...] = ()):
        self.code, self.diagnostics = code, diagnostics
@dataclass(slots=True)
class _Snapshot:
    source: Any
    receipt: AnalysisSourceReceipt
    catalog: tuple[tuple[str, tuple[Any, ...]], ...]
    scanned: tuple[str, ...]
    revisions: tuple[tuple[Path, Path, FileRevision | None], ...]
def _valid_cancel(token: threading.Event | None) -> bool:
    return token is None or type(token) is threading.Event
def _notify(callback: Callable[[int, int], None] | None, done: int, total: int) -> None:
    if callback is not None:
        try:
            callback(done, total)
        except Exception:
            pass
def _close(source: Any) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()
def _resolve_spec_scan(requested: str | None, scans: tuple[str, ...]) -> str | None:
    if requested is None:
        return scans[0] if len(scans) == 1 else None
    if "." in requested:
        return requested if scans.count(requested) == 1 else None
    matches = tuple(value for value in scans if value == requested or value.startswith(requested + "."))
    return matches[0] if len(matches) == 1 else None
def _resolve_nexus_entry(requested: str | None, entries: tuple[str, ...]) -> str | None:
    if requested is None:
        return entries[0] if len(entries) == 1 else None
    return requested if entries.count(requested) == 1 else None
def _candidate(
    spec: SourceSpec, *, remaining_bytes: int | None = None,
) -> CandidateProjection:
    frozen = _freeze_spec(spec)
    identity = (
        str(frozen.uri), frozen.kind, frozen.metadata_uri, frozen.entry,
        dict(frozen.options),
    )
    charge = _canonical_charge(identity)
    if remaining_bytes is not None and charge > remaining_bytes:
        raise _Refusal("SOURCE_CANDIDATE_LIMIT_EXCEEDED")
    return CandidateProjection(frozen, _digest(identity), charge)
def _select(plan: MetadataTablePlan) -> tuple[
    SourceSpec | None, tuple[CandidateProjection, ...], str | None, Any | None,
    FileRevision | None,
]:
    opened = selected_source = None
    selected_revision = None
    try:
        if type(plan.selection) is not str or plan.selection not in {"exact", "image_series", "directory"} or type(plan.recursive) is not bool:
            raise _Refusal("INVALID_SOURCE_SELECTION")
        series_container = False
        if isinstance(plan.source, SourceSpec):
            if plan.selection != "exact" or plan.recursive or any(value is not None for value in (plan.kind, plan.entry, plan.scan, plan.image_dir, plan.image_stem, plan.source_root, plan.metadata_format)):
                raise _Refusal("INVALID_SOURCE_SELECTION")
            spec = _freeze_spec(plan.source)
        elif plan.selection == "image_series":
            spec = image_series_spec(plan.source, metadata_format=plan.metadata_format)
            options = dict(spec.options)
            if plan.source_root is not None:
                options["source_root"] = plan.source_root
            series_container = coerce_source_kind(spec.kind) is SourceKind.NEXUS_STACK
            kind = guess_source_kind(spec.uri) if series_container else spec.kind
            spec = SourceSpec(
                spec.uri, kind, metadata_uri=spec.metadata_uri,
                entry=plan.entry, options=options,
            )
        elif plan.selection == "directory":
            if plan.kind is None:
                raise _Refusal("INVALID_SOURCE_SELECTION")
            specs = discover_scans(
                plan.source, coerce_source_kind(plan.kind), recursive=plan.recursive,
                metadata_format=plan.metadata_format,
            )
            projected: list[CandidateProjection] = []
            charge = 0
            for raw in specs:
                if len(projected) == _MAX_CANDIDATES:
                    raise _Refusal("SOURCE_CANDIDATE_LIMIT_EXCEEDED")
                item = _candidate(
                    raw, remaining_bytes=_MAX_CANDIDATE_BYTES - charge,
                )
                charge += item.storage_bytes
                projected.append(item)
            return None, tuple(projected), (
                "SOURCE_NOT_FOUND" if not projected else "SOURCE_SELECTION_REQUIRED"
            ), None, None
        else:
            kind = coerce_source_kind(plan.kind) if plan.kind is not None else guess_source_kind(plan.source)
            options = {
                key: value for key, value in {
                    "scan": plan.scan, "image_dir": plan.image_dir,
                    "image_stem": plan.image_stem, "source_root": plan.source_root,
                    "metadata_format": plan.metadata_format,
                }.items() if value is not None
            }
            spec = SourceSpec(plan.source, kind, entry=plan.entry, options=options)
        kind = coerce_source_kind(spec.kind)
        if kind in {SourceKind.MEMORY, SourceKind.TILED, SourceKind.LIVE, SourceKind.UNKNOWN}:
            raise _Refusal("SOURCE_KIND_UNSUPPORTED")
        if kind is SourceKind.SPEC:
            from xrd_tools.io.spec import list_spec_scans
            options = dict(spec.options); selected = _resolve_spec_scan(options.get("scan"), tuple(list_spec_scans(spec.uri)))
            if selected is None: raise _Refusal("SOURCE_SELECTION_REQUIRED")
            spec = SourceSpec(spec.uri, kind, metadata_uri=spec.metadata_uri, entry=spec.entry, options={**options, "scan": selected})
        if kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER, SourceKind.PROCESSED_NEXUS}:
            from xrd_tools.io.nexus import list_entries
            selected_revision = _path_revision(
                Path(spec.uri).expanduser().resolve(strict=False)
            )
            if selected_revision is None:
                raise _Refusal("SOURCE_NOT_FOUND")
            if kind is SourceKind.PROCESSED_NEXUS and coerce_source_kind(
                guess_source_kind(spec.uri)
            ) is not kind:
                raise _Refusal("SOURCE_IDENTITY_MISMATCH")
            entry = _resolve_nexus_entry(spec.entry, tuple(list_entries(spec.uri)))
            if entry is None: raise _Refusal("SOURCE_SELECTION_REQUIRED")
            if kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
                from xrd_tools.sources.nexus import NexusStackSource
                opened = open_source(SourceSpec(
                    spec.uri, kind, metadata_uri=spec.metadata_uri,
                    entry=entry, options=dict(spec.options),
                ))
                if type(opened) is NexusStackSource:
                    cursor = opened.open_cursor().open()
                    _close(opened); opened = None
                    selected_source = _CursorSource(cursor)
                else:
                    selected_source, opened = opened, None
                descriptor = getattr(selected_source, "descriptor", None)
                if descriptor is None:
                    raise _Refusal("SOURCE_IDENTITY_MISMATCH")
                actual = coerce_source_kind(descriptor.kind)
                if (
                    (not series_container and actual is not kind)
                    or descriptor.resolved_entry != entry
                ):
                    raise _Refusal("SOURCE_IDENTITY_MISMATCH")
                kind = actual
            spec = SourceSpec(
                spec.uri, kind, metadata_uri=spec.metadata_uri, entry=entry,
                options=dict(spec.options),
            )
        return _freeze_spec(spec), (), None, selected_source, selected_revision
    except _Refusal:
        _close(selected_source); _close(opened)
        raise
    except OSError:
        _close(selected_source); _close(opened)
        raise _Refusal("SOURCE_NOT_FOUND")
    except (InvalidCanonicalValue, TypeError, ValueError):
        _close(selected_source); _close(opened)
        raise _Refusal("INVALID_CANONICAL_VALUE")
    except BaseException:
        _close(selected_source); _close(opened)
        raise
def _metadata_row(values: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in values.items():
        if type(key) is not str:
            raise _Refusal("INVALID_CANONICAL_VALUE")
        if isinstance(value, np.generic):
            if value.dtype.kind not in "biuf":
                raise _Refusal("INVALID_CANONICAL_VALUE")
            value = value.item()
        if type(value) is float and not math.isfinite(value):
            value = None
        elif type(value) not in {type(None), bool, int, float, str, bytes}:
            raise _Refusal("INVALID_CANONICAL_VALUE")
        row[key] = value
    return row
class _CursorSource:
    def __init__(self, cursor):
        self.cursor = cursor
        try:
            self.descriptor = cursor.descriptor
            self.provider = cursor.metadata_provider()
            self.frame_indices = range(self.descriptor.frame_count)
            self.scanned_axes = tuple(self.provider.motors())
            self.capabilities = SourceCapabilities(supports_random_access=True, supports_chunks=True, has_metadata=True, has_raw_references=True)
        except BaseException:
            cursor.close()
            raise
    def metadata_for(self, label, *, max_input_bytes=None):
        return self.provider.complete_metadata_for(label)
    def load_frame(self, label): return self.cursor.read_frame(label)
    def close(self): self.cursor.close()
def _snapshot(
    spec: SourceSpec, *, cancel_token: threading.Event | None = None,
    selected_source: Any | None = None,
    selected_revision: FileRevision | None = None,
) -> _Snapshot:
    lexical = _lexical_absolute(spec.uri)
    resolved = lexical.resolve(strict=False)
    current = _path_revision(resolved)
    before = current if selected_revision is None else selected_revision
    if before is None:
        _close(selected_source)
        raise _Refusal("SOURCE_NOT_FOUND")
    if current != before:
        _close(selected_source)
        raise _Refusal("SOURCE_REVISION_CHANGED")
    source = selected_source if selected_source is not None else open_source(spec)
    try:
        from xrd_tools.sources.nexus import NexusStackSource
        if type(source) is NexusStackSource:
            cursor = source.open_cursor().open()
            _close(source); source = None
            source = _CursorSource(cursor)
        actual_kind = coerce_source_kind(getattr(source, "kind", spec.kind))
        descriptor = getattr(source, "descriptor", None)
        if descriptor is not None:
            actual_kind = coerce_source_kind(descriptor.kind)
        if actual_kind is not coerce_source_kind(spec.kind) or Path(getattr(source, "path", resolved)).resolve(strict=False) != resolved:
            raise _Refusal("SOURCE_IDENTITY_MISMATCH")
        resolved_entry = descriptor.resolved_entry if descriptor is not None else spec.entry
        if resolved_entry != spec.entry or (actual_kind is SourceKind.SPEC and getattr(source, "scan_key", None) != dict(spec.options).get("scan")):
            raise _Refusal("SOURCE_IDENTITY_MISMATCH")
        labels = tuple(
            int(value) for value in islice(iter(source.frame_indices), _MAX_TABLE_ROWS + 1)
        )
        if len(labels) > _MAX_TABLE_ROWS:
            raise _Refusal("METADATA_ROW_LIMIT_EXCEEDED")
        column_values: dict[str, list[Any]] = {}
        row_count = 0
        value_floor_extra = 0
        modes: list[str] = []
        observed_states: list[Any] = []
        discovery_states: list[tuple[Path, FileRevision | None]] = []
        member_states: list[tuple[Path, FileRevision | None]] = []
        member_revisions: dict[Path, FileRevision | None] = {}
        raw_facts: list[tuple[int, str | None, int | None, FileRevision | None]] = []
        options = dict(spec.options)
        metadata_format = getattr(
            source, "metadata_format", options.get("metadata_format"),
        )
        def append_row(row: Mapping[str, Any]) -> None:
            nonlocal row_count, value_floor_extra
            for values in column_values.values():
                values.append(None)
            for key, value in row.items():
                if key not in column_values:
                    if len(column_values) + 1 >= _MAX_TABLE_COLUMNS:
                        raise _Refusal("METADATA_COLUMN_LIMIT_EXCEEDED")
                    column_values[key] = [None] * row_count + [value]
                else:
                    column_values[key][-1] = value
                if type(value) not in {int, float}:
                    text = None if value is None else str(value)
                    value_floor_extra += max(
                        0, _canonical_frame_charge(text) - 8,
                    )
            row_count += 1
            floor = len(labels) * (len(column_values) + 1) * 8 + value_floor_extra
            if len(_CANONICAL_PREFIX) + floor > _MAX_TABLE_BYTES:
                raise _Refusal("METADATA_TABLE_LIMIT_EXCEEDED")
        from xrd_tools.sources.image import ImageFileSource, TiffSeriesSource
        if type(source) is ImageFileSource:
            observed = None if metadata_format is None else read_image_metadata_observed(
                source.path, metadata_format, meta_dir=getattr(source, "meta_dir", None),
                max_input_bytes=_METADATA_INPUT_BYTES,
            )
            if observed is not None and observed.source_path is not None and (
                observed.source_revision is None or _path_revision(observed.source_path) != observed.source_revision
            ): raise _Refusal("SOURCE_REVISION_CHANGED")
            if observed is not None:
                discovery_states.extend(
                    getattr(observed, "discovery_revisions", ())
                )
            row = {} if observed is None else _metadata_row(observed.values)
            for _label in labels:
                if cancel_token is not None and cancel_token.is_set():
                    raise _Refusal("CANCELLED")
                append_row(row)
                modes.append("metadata_off" if observed is None else "same_read")
            observed_states.append(None if observed is None else (observed.source_path, observed.source_revision))
        elif type(source) is TiffSeriesSource:
            paths = tuple(Path(path) for path in source.files)
            member_states = [(path, _path_revision(path)) for path in paths]
            if any(revision is None for _path, revision in member_states):
                raise _Refusal("SOURCE_NOT_FOUND")
            member_revisions.update(member_states)
            admitted_values = tuple(options.get("admitted_motor_values", ()))
            for label in labels:
                if cancel_token is not None and cancel_token.is_set():
                    raise _Refusal("CANCELLED")
                try:
                    path = paths[label - 1]
                except (IndexError, TypeError):
                    raise _Refusal("SOURCE_IDENTITY_MISMATCH")
                observed = None if metadata_format is None else read_image_metadata_observed(
                    path, metadata_format, meta_dir=getattr(source, "meta_dir", None),
                    max_input_bytes=_METADATA_INPUT_BYTES,
                )
                if observed is not None and observed.source_path is not None and (
                    observed.source_revision is None or _path_revision(observed.source_path) != observed.source_revision
                ): raise _Refusal("SOURCE_REVISION_CHANGED")
                if observed is not None:
                    discovery_states.extend(
                        getattr(observed, "discovery_revisions", ())
                    )
                row = {} if observed is None else _metadata_row(observed.values)
                if admitted_values:
                    admitted_path, motor, value = admitted_values[label - 1]
                    if Path(admitted_path).resolve(strict=False) != path.resolve(strict=False):
                        raise _Refusal("SOURCE_IDENTITY_MISMATCH")
                    for key in tuple(row):
                        if key.casefold() == str(motor).casefold():
                            del row[key]
                    row[str(motor)] = value
                append_row(_metadata_row(row))
                modes.append("metadata_off" if observed is None else "same_read")
                observed_states.append(None if observed is None else (observed.source_path, observed.source_revision))
        elif actual_kind in {SourceKind.SPEC, SourceKind.PROCESSED_NEXUS}:
            for label in labels:
                if cancel_token is not None and cancel_token.is_set():
                    raise _Refusal("CANCELLED")
                frame = source.frame_for(label)
                append_row(_metadata_row(frame.metadata))
                modes.append("provider_only")
                observed_states.append(None)
                locator = frame.source_path
                if locator is not None:
                    candidate = Path(locator).expanduser().resolve(strict=False)
                    if candidate in member_revisions:
                        revision = member_revisions[candidate]
                    else:
                        revision = _path_revision(candidate)
                        member_revisions[candidate] = revision
                        member_states.append((candidate, revision))
                else:
                    revision = None
                raw_facts.append(
                    (int(label), None if locator is None else str(locator),
                     frame.source_frame_index, revision)
                )
        else:
            method = source.metadata_for
            try:
                accepts_cap = "max_input_bytes" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                accepts_cap = False
            for label in labels:
                if cancel_token is not None and cancel_token.is_set():
                    raise _Refusal("CANCELLED")
                values = (
                    method(label, max_input_bytes=_METADATA_INPUT_BYTES)
                    if accepts_cap else method(label)
                )
                append_row(_metadata_row(values))
                modes.append("provider_only")
                observed_states.append(None)
        catalog = tuple(
            (key, tuple(values)) for key, values in column_values.items()
        )
        post = _path_revision(resolved)
        if before != post:
            raise _Refusal("SOURCE_REVISION_CHANGED")
        scanned_value = getattr(source, "scanned_axes", ())
        scanned_value = scanned_value() if callable(scanned_value) else scanned_value
        scanned = tuple(str(value) for value in (scanned_value or ()))
        source_spec_identity = (str(spec.uri), spec.kind, spec.metadata_uri,
                                spec.entry, dict(spec.options))
        dependency_revisions = _canonical_dependency_revisions(
            lexical,
            resolved,
            post,
            tuple(value for value in observed_states
                  if value is not None and value[0] is not None)
            + tuple(discovery_states)
            + tuple(member_states),
        )
        source_identity = (
            "analysis-source-v2", source_spec_identity, str(lexical), str(resolved),
            actual_kind, resolved_entry, dict(spec.options).get("scan"), before, post,
            labels, catalog, scanned, tuple(modes), tuple(observed_states),
            tuple(discovery_states),
            tuple(member_states), tuple(raw_facts), dependency_revisions,
        )
        fingerprint = _digest(source_identity)
        receipt = AnalysisSourceReceipt(
            schema_version="analysis-source-v2", source_spec=spec,
            source_spec_digest=_digest(source_spec_identity), lexical_root=lexical,
            resolved_root=resolved, resolved_kind=actual_kind,
            resolved_entry=resolved_entry,
            resolved_scan=dict(spec.options).get("scan"), primary_state=before,
            primary_post_state=post, labels=labels, labels_digest=_digest(labels),
            catalog_digest=_digest(catalog), metadata_observation_modes=tuple(modes),
            dependency_revisions=dependency_revisions,
            source_fingerprint=fingerprint,
        )
        return _Snapshot(
            source, receipt, catalog, scanned, dependency_revisions,
        )
    except BaseException:
        _close(source)
        raise
def _terminal_fence(snapshot: _Snapshot) -> None:
    if not _receipt_fence(snapshot.receipt):
        raise _Refusal("SOURCE_REVISION_CHANGED")
def _source_spec_storage_value(spec: SourceSpec) -> tuple[Any, ...]:
    return (
        str(spec.uri), spec.kind, spec.metadata_uri, spec.entry,
        dict(spec.options),
    )
def _receipt_storage_value(receipt: AnalysisSourceReceipt) -> tuple[Any, ...]:
    return (
        receipt.schema_version,
        _source_spec_storage_value(receipt.source_spec),
        receipt.source_spec_digest,
        receipt.lexical_root,
        receipt.resolved_root,
        receipt.resolved_kind,
        receipt.resolved_entry,
        receipt.resolved_scan,
        receipt.primary_state,
        receipt.primary_post_state,
        receipt.labels,
        receipt.labels_digest,
        receipt.catalog_digest,
        receipt.metadata_observation_modes,
        receipt.dependency_revisions,
        receipt.source_fingerprint,
    )
def _table_storage_value(result: MetadataTableResult) -> tuple[Any, ...]:
    return (
        result.code, result.diagnostics,
        None if result.receipt is None else _receipt_storage_value(result.receipt),
        result.labels,
        tuple((column.name, column.kind, column.missing,
               column.numeric if column.numeric is not None else column.text)
              for column in result.columns),
        result.declared_scanned_positioner, result.selected_scanned_positioner,
        result.policy_fingerprint, result.table_fingerprint,
    )
def _canonical_dependency_revisions(
    primary_lexical: Path,
    primary_resolved: Path,
    primary_revision: FileRevision | None,
    values: tuple[tuple[Path, FileRevision | None], ...],
) -> tuple[tuple[Path, Path, FileRevision | None], ...]:
    root_lexical = _lexical_absolute(primary_lexical)
    root_resolved = root_lexical.resolve(strict=False)
    if root_resolved != primary_resolved or _path_revision(root_resolved) != primary_revision:
        raise _Refusal("SOURCE_REVISION_CHANGED")
    seen: dict[Path, tuple[Path, FileRevision | None]] = {
        root_lexical: (root_resolved, primary_revision),
    }
    ordered: list[tuple[Path, Path, FileRevision | None]] = []
    for raw_path, revision in values:
        lexical = _lexical_absolute(raw_path)
        resolved = lexical.resolve(strict=False)
        observation = (resolved, revision)
        if lexical in seen:
            if seen[lexical] != observation:
                raise _Refusal("SOURCE_REVISION_CHANGED")
            continue
        if _path_revision(resolved) != revision:
            raise _Refusal("SOURCE_REVISION_CHANGED")
        seen[lexical] = observation
        ordered.append((lexical, resolved, revision))
    return tuple(ordered)
def _dependency_matches(
    lexical: Path, resolved: Path, revision: FileRevision | None,
) -> bool:
    try:
        return (
            _lexical_absolute(lexical).resolve(strict=False) == resolved
            and _path_revision(resolved) == revision
        )
    except OSError:
        return False
def _receipt_dependencies(
    receipt: AnalysisSourceReceipt,
) -> tuple[tuple[Path, Path, FileRevision | None], ...]:
    return (
        (receipt.lexical_root, receipt.resolved_root, receipt.primary_post_state),
        *receipt.dependency_revisions,
    )
def _receipt_fence(receipt: AnalysisSourceReceipt) -> bool:
    return all(_dependency_matches(*dependency)
               for dependency in _receipt_dependencies(receipt))
def _metadata_table_storage_value(result: MetadataTableResult) -> tuple[Any, ...]:
    return _table_storage_value(result)
def _columns(
    catalog: tuple[tuple[str, tuple[Any, ...]], ...], labels: tuple[int, ...],
    *, cancel_token: threading.Event | None = None,
) -> tuple[MetadataColumn, ...]:
    if len(catalog) + 1 > _MAX_TABLE_COLUMNS:
        raise _Refusal("METADATA_COLUMN_LIMIT_EXCEEDED")
    result: list[MetadataColumn] = []
    frame_charge = 0
    for name, values in (("frame_index", labels), *catalog):
        if cancel_token is not None and cancel_token.is_set():
            raise _Refusal("CANCELLED")
        numeric = all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in values if value is not None
        )
        if numeric:
            array = _missing_numeric_array(len(values))
            for index, value in enumerate(values):
                if value is not None:
                    array[index] = float(value)
            array.flags.writeable = False
            identity = (name, "numeric", "nan", array)
            column = MetadataColumn(
                name, "numeric", "nan", array, None,
                _canonical_charge(identity, allow_missing=True),
            )
        else:
            text = tuple(None if value is None else str(value) for value in values)
            identity = (name, "text", "none", text)
            column = MetadataColumn(
                name, "text", "none", None, text, _canonical_charge(identity),
            )
        frame_charge += _canonical_frame_charge(identity, allow_missing=True)
        if len(_CANONICAL_PREFIX) + frame_charge > _MAX_TABLE_BYTES:
            raise _Refusal("METADATA_TABLE_LIMIT_EXCEEDED")
        result.append(column)
    return tuple(result)
def run_metadata_table(
    plan: MetadataTablePlan, *, cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MetadataTableResult:
    if not _valid_cancel(cancel_token):
        return MetadataTableResult(AnalysisDisposition.REFUSED, "INVALID_CANCEL_TOKEN")
    if cancel_token is not None and cancel_token.is_set():
        return MetadataTableResult(AnalysisDisposition.CANCELLED, "CANCELLED")
    snapshot = None
    selected_source = None
    try:
        spec, candidates, selection_code, selected_source, selected_revision = _select(plan)
        if selection_code is not None:
            return MetadataTableResult(AnalysisDisposition.REFUSED, selection_code,
                                       candidates=candidates)
        assert spec is not None
        handoff, selected_source = selected_source, None
        snapshot = _snapshot(
            spec, cancel_token=cancel_token, selected_source=handoff,
            selected_revision=selected_revision,
        )
        if cancel_token is not None and cancel_token.is_set():
            return MetadataTableResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        columns = _columns(
            snapshot.catalog, snapshot.receipt.labels, cancel_token=cancel_token,
        )
        declared = snapshot.scanned[0] if snapshot.scanned else None
        selected = declared if declared in {column.name for column in columns} else None
        policy = _digest(("metadata-table-v1", _MAX_TABLE_ROWS, _MAX_TABLE_COLUMNS,
                          _MAX_TABLE_BYTES))
        table_fp = _digest((snapshot.receipt.source_fingerprint, declared, selected,
                            tuple((c.name, c.kind, c.numeric if c.numeric is not None else c.text)
                                  for c in columns)), allow_missing=True)
        provisional = MetadataTableResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=snapshot.receipt,
            labels=snapshot.receipt.labels, columns=columns,
            declared_scanned_positioner=declared,
            selected_scanned_positioner=selected, policy_fingerprint=policy,
            table_fingerprint=table_fp,
        )
        charge = _canonical_charge(
            _table_storage_value(provisional), allow_missing=True,
        )
        if charge > _MAX_TABLE_BYTES:
            raise _Refusal("METADATA_TABLE_LIMIT_EXCEEDED")
        _notify(progress_callback, len(columns), len(columns))
        if cancel_token is not None and cancel_token.is_set():
            return MetadataTableResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        _terminal_fence(snapshot)
        return MetadataTableResult(**{
            field.name: (charge if field.name == "storage_bytes" else getattr(provisional, field.name))
            for field in provisional.__dataclass_fields__.values()
        })
    except _Refusal as error:
        if error.code == "CANCELLED":
            return MetadataTableResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        return MetadataTableResult(AnalysisDisposition.REFUSED, error.code,
                                   diagnostics=error.diagnostics)
    finally:
        _close(selected_source)
        if snapshot is not None:
            _close(snapshot.source)
def run_metadata_table_requalification(
    plan: MetadataTableRequalificationPlan, *,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MetadataTableRequalificationResult:
    if type(plan) is not MetadataTableRequalificationPlan:
        return MetadataTableRequalificationResult(
            AnalysisDisposition.REFUSED, "INVALID_REQUALIFICATION_PLAN"
        )
    if not _valid_cancel(cancel_token):
        return MetadataTableRequalificationResult(
            AnalysisDisposition.REFUSED, "INVALID_CANCEL_TOKEN"
        )
    receipt = plan.receipt
    if receipt.schema_version != "analysis-source-v2":
        return MetadataTableRequalificationResult(
            AnalysisDisposition.REFUSED, "SOURCE_IDENTITY_MISMATCH"
        )
    revisions = _receipt_dependencies(receipt)
    total = 2 * len(revisions)
    completed = 0
    for _sweep in range(2):
        for lexical, resolved, revision in revisions:
            if cancel_token is not None and cancel_token.is_set():
                return MetadataTableRequalificationResult(
                    AnalysisDisposition.CANCELLED, "CANCELLED"
                )
            if not _dependency_matches(lexical, resolved, revision):
                return MetadataTableRequalificationResult(
                    AnalysisDisposition.REFUSED, "SOURCE_REVISION_CHANGED"
                )
            completed += 1
            _notify(progress_callback, completed, total)
    if cancel_token is not None and cancel_token.is_set():
        return MetadataTableRequalificationResult(
            AnalysisDisposition.CANCELLED, "CANCELLED"
        )
    return MetadataTableRequalificationResult(
        AnalysisDisposition.COMPLETED, "OK", receipt, plan.table_fingerprint,
    )
def _numeric_column(table: MetadataTableResult, name: str) -> np.ndarray:
    matches = [column for column in table.columns if column.name == name]
    if len(matches) != 1 or matches[0].kind != "numeric" or matches[0].numeric is None:
        raise _Refusal("SCAN_PLOT_NONNUMERIC_COLUMN")
    return matches[0].numeric
def run_scan_plot(
    plan: ScanPlotPlan, table: MetadataTableResult, *, roi_result: RoiScanResult | None = None,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ScanPlotResult:
    if not _valid_cancel(cancel_token):
        return ScanPlotResult(AnalysisDisposition.REFUSED, "INVALID_CANCEL_TOKEN")
    if table.disposition is not AnalysisDisposition.COMPLETED:
        return ScanPlotResult(AnalysisDisposition.REFUSED, "METADATA_TABLE_REQUIRED")
    if cancel_token is not None and cancel_token.is_set():
        return ScanPlotResult(AnalysisDisposition.CANCELLED, "CANCELLED")
    try:
        names = {column.name for column in table.columns}
        x_name = plan.x or table.selected_scanned_positioner or "frame_index"
        x = _numeric_column(table, x_name)
        requested = plan.y or next(
            ((name,) for name in ("Photod", "bs", "mon", "i2", "i1", "i0")
             if name != x_name and name in names),
            ("frame_index",) if x_name != "frame_index" else (),
        )
        if not requested:
            raise _Refusal("SCAN_PLOT_Y_REQUIRED")
        if len(requested) > _MAX_PLOT_TRACES:
            raise _Refusal("SCAN_PLOT_TRACE_LIMIT_EXCEEDED")
        trace_names: list[str] = []
        origins: list[str] = []
        identities: list[str] = []
        traces: list[np.ndarray] = []
        roi_by_name = {} if roi_result is None else {
            name: values for name, values in zip(roi_result.signal_names, roi_result.signal_values)
        }
        if roi_result is not None and (
            roi_result.disposition is not AnalysisDisposition.COMPLETED
            or roi_result.table_fingerprint != table.table_fingerprint
            or table.receipt is None or roi_result.receipt is None
            or roi_result.receipt.source_fingerprint
            != table.receipt.source_fingerprint
            or roi_result.requested_labels != table.labels
            or roi_result.completed_labels != table.labels
        ):
            raise _Refusal("ROI_IDENTITY_MISMATCH")
        for requested_name in requested:
            if cancel_token is not None and cancel_token.is_set():
                return ScanPlotResult(AnalysisDisposition.CANCELLED, "CANCELLED")
            choices: list[tuple[str, np.ndarray]] = []
            if requested_name in names:
                choices.append(("metadata", _numeric_column(table, requested_name)))
            if requested_name in roi_by_name:
                choices.append(("derived_roi", roi_by_name[requested_name]))
            if not choices:
                raise _Refusal("SCAN_PLOT_COLUMN_NOT_FOUND")
            for origin, values in choices:
                if len(trace_names) == _MAX_PLOT_TRACES:
                    raise _Refusal("SCAN_PLOT_TRACE_LIMIT_EXCEEDED")
                if np.asarray(values).size != x.size:
                    raise _Refusal("ROI_IDENTITY_MISMATCH")
                display = requested_name
                suffix = 2
                while display in trace_names:
                    display = f"{requested_name}_{suffix}"; suffix += 1
                trace_names.append(display); origins.append(origin); identities.append(requested_name)
                traces.append(np.asarray(values, dtype=float))
        divisor = None if plan.normalization is None else _numeric_column(table, plan.normalization)
        bad = None if divisor is None else (~np.isfinite(divisor) | (divisor == 0))
        invalid = 0 if bad is None else int(np.count_nonzero(bad))
        published: list[np.ndarray] = []
        for values in traces:
            current = np.array(values, dtype="<f8", copy=True)
            if bad is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    current = current / divisor
                current[bad] = np.nan
            published.append(_owned_missing(current))
        x_out = _owned(x, dtype="<f8")
        identity = (table.table_fingerprint,
                    None if roi_result is None else roi_result.result_fingerprint,
                    x_name, tuple(trace_names), tuple(origins), tuple(identities),
                    x_out, tuple(published), plan.normalization, invalid)
        charge = _canonical_charge(identity, allow_missing=True)
        if charge > _MAX_PLOT_BYTES:
            raise _Refusal("SCAN_PLOT_LIMIT_EXCEEDED")
        _notify(progress_callback, len(published), len(published))
        if cancel_token is not None and cancel_token.is_set():
            return ScanPlotResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        return ScanPlotResult(
            AnalysisDisposition.COMPLETED, "OK", table_fingerprint=table.table_fingerprint,
            roi_fingerprint=None if roi_result is None else roi_result.result_fingerprint,
            x_name=x_name, x=x_out, trace_names=tuple(trace_names),
            trace_origins=tuple(origins), original_identities=tuple(identities),
            traces=tuple(published), normalization=plan.normalization,
            normalization_invalid_count=invalid,
            policy_fingerprint=_digest(("scan-plot-v1",)), storage_bytes=charge,
        )
    except _Refusal as error:
        return ScanPlotResult(AnalysisDisposition.REFUSED, error.code)
def _source_close_note(cleanup: BaseException) -> str:
    try:
        message = str(cleanup)
    except BaseException:
        message = "exception message unavailable"
    kind = type(cleanup)
    return (
        "source close also failed: "
        f"{kind.__module__}.{kind.__qualname__}: {message[:4096]}"
    )


def _lease_refusal(refusal: _Refusal) -> AnalysisSourceLeaseRefused:
    converted = AnalysisSourceLeaseRefused(refusal.code)
    for note in getattr(refusal, "__notes__", ()):
        try:
            converted.add_note(note)
        except BaseException:
            pass
    return converted


def _requalify(receipt: AnalysisSourceReceipt) -> _Snapshot:
    snapshot = _snapshot(receipt.source_spec)
    exact_fields = (
        "schema_version", "source_spec_digest", "lexical_root", "resolved_root",
        "resolved_kind", "resolved_entry", "resolved_scan", "primary_state",
        "primary_post_state", "labels", "labels_digest", "catalog_digest",
        "metadata_observation_modes", "dependency_revisions", "source_fingerprint",
    )
    if any(
        getattr(snapshot.receipt, name) != getattr(receipt, name)
        for name in exact_fields
    ):
        refusal = _Refusal("SOURCE_IDENTITY_MISMATCH")
        try:
            _close(snapshot.source)
        except BaseException as cleanup:
            refusal.add_note(_source_close_note(cleanup))
        raise refusal
    return snapshot
@contextmanager
def requalified_analysis_source(
    receipt: AnalysisSourceReceipt, *,
    cancel_token: threading.Event | None = None,
):
    """Yield one exact reopened source under before/after revision fences.

    The source handle is closed exactly once.  A body exception remains the
    primary outcome; terminal fencing is applied only when the body returns
    normally, so cleanup never disguises a scientific failure.
    """
    if type(receipt) is not AnalysisSourceReceipt:
        raise TypeError("source lease requires an exact AnalysisSourceReceipt")
    if not _valid_cancel(cancel_token):
        raise TypeError("source lease requires a threading.Event cancellation token")
    if cancel_token is not None and cancel_token.is_set():
        raise AnalysisSourceLeaseRefused("CANCELLED")
    snapshot = None
    primary: BaseException | None = None
    try:
        try:
            snapshot = _requalify(receipt)
        except _Refusal as error:
            raise _lease_refusal(error) from None
        if cancel_token is not None and cancel_token.is_set():
            raise AnalysisSourceLeaseRefused("CANCELLED")
        try:
            yield snapshot.source
        except BaseException:
            raise
        else:
            if cancel_token is not None and cancel_token.is_set():
                raise AnalysisSourceLeaseRefused("CANCELLED")
            try:
                _terminal_fence(snapshot)
            except _Refusal as error:
                raise _lease_refusal(error) from None
    except BaseException as error:
        primary = error
        raise
    finally:
        if snapshot is not None:
            try:
                _close(snapshot.source)
            except BaseException as cleanup:
                if primary is None:
                    raise
                try:
                    primary.add_note(_source_close_note(cleanup))
                except BaseException:
                    pass
class _RoiSourceView:
    def __init__(self, source: Any, labels: tuple[int, ...]):
        self._source = source
        self._labels = labels
        self.capabilities = source.capabilities
    @property
    def frame_indices(self) -> list[int]:
        return list(self._labels)
    def load_frame(self, label: int) -> np.ndarray:
        return self._source.load_frame(label)
    def iter_chunks(self, chunk_size: int):
        for start in range(0, len(self._labels), chunk_size):
            labels = self._labels[start:start + chunk_size]
            yield np.stack([self.load_frame(label) for label in labels]), list(labels)
def run_roi_preview(
    plan: RoiPreviewPlan, *, cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RoiPreviewResult:
    if not _valid_cancel(cancel_token):
        return RoiPreviewResult(AnalysisDisposition.REFUSED, "INVALID_CANCEL_TOKEN")
    if cancel_token is not None and cancel_token.is_set():
        return RoiPreviewResult(AnalysisDisposition.CANCELLED, "CANCELLED")
    snapshot = None
    try:
        snapshot = _requalify(plan.receipt)
        if snapshot.receipt.labels != plan.labels or plan.label not in plan.labels:
            raise _Refusal("SOURCE_IDENTITY_MISMATCH")
        try:
            image = np.asarray(snapshot.source.load_frame(plan.label))
        except Exception:
            raise _Refusal("ROI_RAW_UNAVAILABLE")
        if cancel_token is not None and cancel_token.is_set(): return RoiPreviewResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        if image.ndim != 2:
            raise _Refusal("ROI_RAW_SHAPE_INVALID")
        identity_prefix = (plan.table_fingerprint, plan.receipt.source_fingerprint,
                           plan.labels, plan.label)
        charge = _canonical_charge((*identity_prefix, image), allow_missing=True)
        if charge > _MAX_PREVIEW_BYTES:
            raise _Refusal("ROI_PREVIEW_LIMIT_EXCEEDED")
        output = _owned(image)
        identity = (*identity_prefix, output)
        _notify(progress_callback, 1, 1)
        if cancel_token is not None and cancel_token.is_set():
            return RoiPreviewResult(AnalysisDisposition.CANCELLED, "CANCELLED")
        _terminal_fence(snapshot)
        return RoiPreviewResult(
            AnalysisDisposition.COMPLETED, "OK", receipt=plan.receipt,
            table_fingerprint=plan.table_fingerprint, labels=plan.labels,
            label=plan.label, image=output,
            policy_fingerprint=_digest(("roi-preview-v1",)),
            result_fingerprint=_digest(identity, allow_missing=True), storage_bytes=charge,
        )
    except _Refusal as error:
        return RoiPreviewResult(AnalysisDisposition.REFUSED, error.code)
    finally:
        if snapshot is not None:
            _close(snapshot.source)
def run_roi_scan(
    plan: RoiScanPlan, *, cancel_token: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> RoiScanResult:
    if not _valid_cancel(cancel_token):
        return RoiScanResult(AnalysisDisposition.REFUSED, "INVALID_CANCEL_TOKEN")
    if not 1 <= len(plan.signals) <= 5:
        return RoiScanResult(AnalysisDisposition.REFUSED, "ROI_SIGNAL_COUNT_INVALID")
    if cancel_token is not None and cancel_token.is_set():
        return RoiScanResult(AnalysisDisposition.CANCELLED, "CANCELLED")
    snapshot = None
    try:
        snapshot = _requalify(plan.receipt)
        if snapshot.receipt.labels != plan.labels:
            raise _Refusal("SOURCE_IDENTITY_MISMATCH")
        requested = plan.labels if plan.selected_labels is None else plan.selected_labels
        selected = set(requested)
        if (
            len(requested) > len(plan.labels)
            or len(selected) != len(requested)
            or tuple(label for label in plan.labels if label in selected) != requested
        ):
            raise _Refusal("SOURCE_IDENTITY_MISMATCH")
        def progress(done, total):
            _notify(progress_callback, done, total)
        payload = run_roi_signals(
            plan.signals, _RoiSourceView(snapshot.source, tuple(requested)), mask=plan.mask,
            mask_saturation=plan.mask_saturation, frame_indices=requested,
            on_progress=progress,
            should_cancel=None if cancel_token is None else cancel_token.is_set,
        ).payload
        completed = tuple(int(value) for value in payload.frames)
        names = tuple(payload.series)
        values = tuple(_owned_missing(payload.series[name]) for name in names)
        counts = tuple(_owned(payload.valid_counts[name]) for name in names)
        no_raw = tuple(int(value) for value in payload.diagnostics["no_raw_frames"])
        diagnostics: list[str] = []
        if no_raw and len(no_raw) == len(completed):
            diagnostics.append("ROI_RAW_UNAVAILABLE_ALL")
        if payload.diagnostics.get("mask_resolution_attempted") and not payload.diagnostics.get("mask_resolved"):
            diagnostics.append("ROI_INVALID_MASK_IGNORED")
        cancelled = bool(payload.diagnostics["cancelled"] or (cancel_token is not None and cancel_token.is_set()))
        identity = (plan.table_fingerprint, plan.receipt.source_fingerprint,
                    requested, completed, names, values, counts, no_raw,
                    tuple(diagnostics))
        charge = _canonical_charge(identity, allow_missing=True)
        if charge > _MAX_ROI_BYTES:
            raise _Refusal("ROI_RESULT_LIMIT_EXCEEDED")
        _terminal_fence(snapshot)
        disposition = AnalysisDisposition.CANCELLED if cancelled else AnalysisDisposition.COMPLETED
        code = "CANCELLED" if cancelled else "OK"
        return RoiScanResult(
            disposition, code, tuple(diagnostics), plan.receipt,
            plan.table_fingerprint, tuple(requested), completed, names, values,
            counts, no_raw, _digest(("roi-scan-v1",)),
            _digest(identity, allow_missing=True), charge,
        )
    except _Refusal as error:
        return RoiScanResult(AnalysisDisposition.REFUSED, error.code)
    finally:
        if snapshot is not None:
            _close(snapshot.source)
