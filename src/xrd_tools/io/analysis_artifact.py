"""Strict standalone NeXus publication for Stitch and RSM results.

An analysis artifact is deliberately not a processed scan: it contains exactly
one scan-level result and never fabricates an integrated-frame group.  This
module composes the accepted file transaction with a small, independently
admitted schema and strict semantic readback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import h5py
import numpy as np

from xrd_tools.io.output_transaction import (
    LeaseOwner,
    OwnerToken,
    OutputTransactionCoordinator,
    StreamTerminal,
    TargetSnapshot,
    TargetChanged,
    TransactionPhase,
    TransactionStateError,
    capture_target_snapshot,
    get_output_transaction_coordinator,
    revalidate_stream_terminal,
    stream_terminal_object_revision,
)


ANALYSIS_SCHEMA_ATTR = "ssrl_schema"
ANALYSIS_SCHEMA_NAME = "xrd_tools.analysis_artifact"
ANALYSIS_SCHEMA_VERSION_ATTR = "ssrl_schema_version"
ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION_V2 = 2
ANALYSIS_KIND_ATTR = "analysis_kind"
_ENTRY = "entry"
_PROVENANCE = "provenance_json"
_EXECUTION_ATTESTATION = "execution_attestation_json"
_EXECUTION_ATTESTATION_DIGEST_ATTR = "execution_attestation_digest"
_MAX_PROVENANCE_BYTES = 1 << 20
_MAX_PROVENANCE_NESTING = 64
_MAX_EXECUTION_ATTESTATION_BYTES = 16_384
_MAX_EXECUTION_ATTESTATION_NESTING = 4
_MAX_ATTRIBUTE_BYTES = 4096
_MAX_ATTRIBUTE_ITEMS = 16
_MAX_AXIS_POINTS = 1_000_000
_MAX_RESULT_ELEMENTS = 64_000_000
_MAX_READ_BLOCK_BYTES = 8 << 20
_MAX_ARTIFACT_FILE_BYTES = 1 << 30
_RESULT_FINGERPRINT_PREFIX = b"xrd_tools.analysis-artifact-result.v1\0"
_STORED_RESULT_PROJECTION_POLICY = "analysis_artifact_stored_le_f4_v1"
_STORED_QNAN_F4_BITS = np.uint32(0x7FC00000)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ANALYSIS_RECEIPT_FACTORY = object()
_ANALYSIS_PAYLOAD_FACTORY = object()
_ANALYSIS_CANDIDATE_INSPECTION = object()
_ANALYSIS_PROJECTION_FACTORY = object()
_ANALYSIS_V1_DECODE_PROJECTION = object()


class AnalysisArtifactKind(str, Enum):
    STITCH_1D = "stitch-1d"
    STITCH_2D = "stitch-2d"
    RSM = "rsm"

    @property
    def group(self) -> str:
        return {
            AnalysisArtifactKind.STITCH_1D: "stitched_1d",
            AnalysisArtifactKind.STITCH_2D: "stitched_2d",
            AnalysisArtifactKind.RSM: "rsm",
        }[self]


class AnalysisArtifactOverwrite(str, Enum):
    CREATE_NEW = "create-new"
    REPLACE = "replace"


class AnalysisArtifactError(RuntimeError):
    pass


class AnalysisArtifactInvalid(AnalysisArtifactError):
    pass


class AnalysisArtifactProjectionInvalid(AnalysisArtifactError, ValueError):
    """A scientific result cannot be represented by the stored artifact schema."""

    code = "STITCH_RESULT_STORAGE_PROJECTION_INVALID"


class AnalysisArtifactCleanupPending(AnalysisArtifactError):
    def __init__(self, snapshot: object) -> None:
        self.snapshot = snapshot
        super().__init__("analysis artifact cleanup remains retryable")


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{name} must be lowercase SHA-256 hex")
    return value


def _absolute_path(value: object) -> str:
    try:
        return os.path.normcase(
            os.path.abspath(os.path.expanduser(os.fspath(value)))
        )
    except TypeError as error:
        raise TypeError("analysis artifact target must be path-like") from error


def _normalize_target(value: object) -> str:
    target = _absolute_path(value)
    if Path(target).suffix.casefold() != ".nexus":
        raise ValueError("analysis artifact target must end in .nexus")
    return target


def _charge_json(budget: list[int], amount: int) -> None:
    budget[0] += int(amount)
    if budget[0] > _MAX_PROVENANCE_BYTES:
        raise ValueError("analysis provenance exceeds the bounded input size")


def _json_string_size(value: str) -> int:
    if len(value) > _MAX_PROVENANCE_BYTES:
        return _MAX_PROVENANCE_BYTES + 1
    size = 2
    for character in value:
        code = ord(character)
        if character in {'"', "\\"}:
            size += 2
        elif character in {"\b", "\t", "\n", "\f", "\r"}:
            size += 2
        elif code < 0x20:
            size += 6
        else:
            try:
                size += len(character.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError(
                    "analysis provenance contains invalid Unicode"
                ) from error
        if size > _MAX_PROVENANCE_BYTES:
            break
    return size


def _json_value(
    value: object,
    active: set[int],
    depth: int,
    budget: list[int],
) -> object:
    if type(value) in {dict, list, tuple} and depth > _MAX_PROVENANCE_NESTING:
        raise ValueError("analysis provenance exceeds the nesting bound")
    recursive = type(value) in {dict, list, tuple}
    marker = id(value)
    if recursive:
        if marker in active:
            raise ValueError("analysis provenance is cyclic")
        active.add(marker)
    try:
        if value is None:
            _charge_json(budget, 4)
            return value
        if type(value) is str:
            _charge_json(budget, _json_string_size(value))
            return value
        if type(value) is bool:
            _charge_json(budget, 4 if value else 5)
            return value
        if type(value) is int:
            if value.bit_length() > 3_500_000:
                raise ValueError("analysis provenance exceeds the bounded input size")
            _charge_json(budget, len(str(value)))
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("analysis provenance contains a nonfinite number")
            _charge_json(budget, len(repr(value)))
            return value
        if type(value) in {list, tuple}:
            _charge_json(budget, 2)
            frozen = []
            for index, item in enumerate(value):
                if index:
                    _charge_json(budget, 1)
                frozen.append(_json_value(item, active, depth + 1, budget))
            return frozen
        if type(value) is dict:
            _charge_json(budget, 2)
            frozen = {}
            for index, (key, item) in enumerate(value.items()):
                if type(key) is not str:
                    raise TypeError("analysis provenance keys must be exact strings")
                if index:
                    _charge_json(budget, 1)
                _charge_json(budget, _json_string_size(key) + 1)
                frozen[key] = _json_value(item, active, depth + 1, budget)
            return frozen
        raise TypeError(
            f"analysis provenance contains unsupported {type(value).__name__}"
        )
    finally:
        if recursive:
            active.remove(marker)


def canonical_analysis_provenance(value: Mapping[str, object]) -> str:
    if type(value) is not dict:
        raise TypeError("analysis provenance must be an exact dictionary")
    frozen = _json_value(value, set(), 1, [0])
    text = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(text.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise ValueError("analysis provenance exceeds the bounded input size")
    return text


def _execution_attestation_value(
    value: object,
    active: set[int],
    depth: int,
    budget: list[int],
) -> object:
    if type(value) in {dict, list, tuple} and depth > _MAX_EXECUTION_ATTESTATION_NESTING:
        raise ValueError("analysis execution attestation exceeds the nesting bound")
    recursive = type(value) in {dict, list, tuple}
    marker = id(value)
    if recursive:
        if marker in active:
            raise ValueError("analysis execution attestation is cyclic")
        active.add(marker)

    def charge(amount: int) -> None:
        budget[0] += int(amount)
        if budget[0] > _MAX_EXECUTION_ATTESTATION_BYTES:
            raise ValueError("analysis execution attestation is oversized")

    try:
        if value is None:
            charge(4)
            return None
        if type(value) is str:
            charge(_json_string_size(value))
            return value
        if type(value) is bool:
            charge(4 if value else 5)
            return value
        if type(value) is int:
            if value.bit_length() > 65_536:
                raise ValueError("analysis execution attestation is oversized")
            charge(len(str(value)))
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(
                    "analysis execution attestation contains a nonfinite number"
                )
            charge(len(repr(value)))
            return value
        if type(value) in {list, tuple}:
            charge(2)
            frozen = []
            for index, item in enumerate(value):
                if index:
                    charge(1)
                frozen.append(
                    _execution_attestation_value(
                        item, active, depth + 1, budget
                    )
                )
            return frozen
        if type(value) is dict:
            charge(2)
            frozen = {}
            for index, (key, item) in enumerate(value.items()):
                if type(key) is not str:
                    raise TypeError(
                        "analysis execution attestation keys must be exact strings"
                    )
                if index:
                    charge(1)
                charge(_json_string_size(key) + 1)
                frozen[key] = _execution_attestation_value(
                    item, active, depth + 1, budget
                )
            return frozen
        raise TypeError(
            "analysis execution attestation contains unsupported "
            f"{type(value).__name__}"
        )
    finally:
        if recursive:
            active.remove(marker)


def canonical_analysis_execution_attestation(
    value: Mapping[str, object],
) -> str:
    if type(value) is not dict:
        raise TypeError("analysis execution attestation must be an exact dictionary")
    frozen = _execution_attestation_value(value, set(), 1, [0])
    text = json.dumps(
        frozen,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(text.encode("utf-8")) > _MAX_EXECUTION_ATTESTATION_BYTES:
        raise ValueError("analysis execution attestation is oversized")
    return text


def _validated_execution_attestation(
    value: Mapping[str, object],
    *,
    kind: AnalysisArtifactKind,
    request_fingerprint: str,
) -> tuple[str, dict[str, object]]:
    text = canonical_analysis_execution_attestation(value)
    parsed = json.loads(text)
    top_keys = {
        "schema_version",
        "module_request_fingerprint",
        "result_projection_policy",
        "result_fingerprint",
        "selected_frame_count",
        "release_check_frame_count",
        "release_check_passed",
        "q_root_policy",
        "xu_runtime",
    }
    runtime_keys = {
        "lock_policy",
        "xrayutilities_distribution_version",
        "xrayutilities_module_version",
        "numpy_version",
        "config_epsilon",
        "config_digits",
        "nthreads_before",
        "nthreads_effective",
        "nthreads_restored",
        "restore_passed",
    }
    runtime = parsed.get("xu_runtime")
    selected = parsed.get("selected_frame_count")
    released = parsed.get("release_check_frame_count")
    before = runtime.get("nthreads_before") if type(runtime) is dict else None
    restored = runtime.get("nthreads_restored") if type(runtime) is dict else None
    if (
        kind is not AnalysisArtifactKind.STITCH_1D
        or set(parsed) != top_keys
        or parsed.get("schema_version") != "analysis-execution-attestation-v1"
        or parsed.get("module_request_fingerprint") != request_fingerprint
        or parsed.get("result_projection_policy")
        != _STORED_RESULT_PROJECTION_POLICY
        or type(parsed.get("result_fingerprint")) is not str
        or _SHA256.fullmatch(parsed["result_fingerprint"]) is None
        or type(selected) is not int
        or selected < 1
        or type(released) is not int
        or released != selected
        or parsed.get("release_check_passed") is not True
        or parsed.get("q_root_policy")
        != "shared_ultimate_ndarray_root_weakref_v1"
        or type(runtime) is not dict
        or set(runtime) != runtime_keys
        or runtime.get("lock_policy") != "shared_xrd_tools_xu_rlock_v1"
        or runtime.get("xrayutilities_distribution_version") != "1.7.12"
        or runtime.get("xrayutilities_module_version") != "1.7.12"
        or runtime.get("numpy_version") != "2.5.1"
        or type(runtime.get("config_epsilon")) is not float
        or runtime.get("config_epsilon") != 1e-8
        or type(runtime.get("config_digits")) is not int
        or runtime.get("config_digits") != 8
        or type(before) is not int
        or before < 0
        or type(runtime.get("nthreads_effective")) is not int
        or runtime.get("nthreads_effective") != 1
        or type(restored) is not int
        or restored != before
        or runtime.get("restore_passed") is not True
    ):
        raise ValueError("analysis execution attestation contract is invalid")
    return text, parsed


def analysis_execution_attestation_digest(
    kind: AnalysisArtifactKind,
    value: Mapping[str, object],
    *,
    request_fingerprint: str,
) -> str:
    """Return the exact v1 execution-attestation identity domain."""

    if type(kind) is not AnalysisArtifactKind:
        raise TypeError("analysis attestation kind must be exact")
    _sha256(request_fingerprint, "request fingerprint")
    _text, parsed = _validated_execution_attestation(
        value,
        kind=kind,
        request_fingerprint=request_fingerprint,
    )
    # Importing the engine-light transaction enum lazily avoids the artifact /
    # transaction import cycle while preserving the frozen Enum framing.
    from xrd_tools.analysis.module_transaction import ModuleKind
    from xrd_tools.analysis.scan_operations import analysis_canonical_fingerprint

    module_kind = ModuleKind.STITCH
    return analysis_canonical_fingerprint(
        "analysis-artifact-execution-attestation-v1",
        (module_kind, parsed),
    )


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactRequest:
    target: str | Path
    kind: AnalysisArtifactKind
    overwrite: AnalysisArtifactOverwrite
    request_fingerprint: str
    source_fingerprint: str
    plan_fingerprint: str
    provenance_digest: str
    provenance: InitVar[Mapping[str, object]]
    provenance_json: str = field(init=False, repr=False)
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    execution_attestation_digest: str | None = None
    execution_attestation: InitVar[Mapping[str, object] | None] = None
    execution_attestation_json: str | None = field(init=False, default=None, repr=False)
    module_owner: InitVar[object | None] = None
    _module_owner: object | None = field(init=False, default=None, repr=False)

    def __post_init__(
        self,
        provenance: Mapping[str, object],
        execution_attestation: Mapping[str, object] | None,
        module_owner: object | None,
    ) -> None:
        if (
            type(self.kind) is not AnalysisArtifactKind
            or type(self.overwrite) is not AnalysisArtifactOverwrite
        ):
            raise TypeError("analysis artifact request enums must be exact")
        object.__setattr__(self, "_module_owner", module_owner)
        object.__setattr__(self, "target", _normalize_target(self.target))
        for value, name in (
            (self.request_fingerprint, "request fingerprint"),
            (self.source_fingerprint, "source fingerprint"),
            (self.plan_fingerprint, "plan fingerprint"),
            (self.provenance_digest, "provenance digest"),
        ):
            _sha256(value, name)
        object.__setattr__(
            self,
            "provenance_json",
            canonical_analysis_provenance(provenance),
        )
        if type(self.schema_version) is not int or self.schema_version not in {
            ANALYSIS_SCHEMA_VERSION,
            ANALYSIS_SCHEMA_VERSION_V2,
        }:
            raise TypeError("analysis artifact schema version is unsupported")
        if self.schema_version == ANALYSIS_SCHEMA_VERSION:
            if (
                self.execution_attestation_digest is not None
                or execution_attestation is not None
            ):
                raise ValueError("analysis artifact v1 cannot carry an attestation")
            return
        if (
            type(self.execution_attestation_digest) is not str
            or execution_attestation is None
        ):
            raise ValueError("analysis artifact v2 requires an execution attestation")
        _sha256(
            self.execution_attestation_digest,
            "execution attestation digest",
        )
        text, parsed = _validated_execution_attestation(
            execution_attestation,
            kind=self.kind,
            request_fingerprint=self.request_fingerprint,
        )
        observed = analysis_execution_attestation_digest(
            self.kind,
            parsed,
            request_fingerprint=self.request_fingerprint,
        )
        if observed != self.execution_attestation_digest:
            raise ValueError("analysis execution attestation digest changed")
        object.__setattr__(self, "execution_attestation_json", text)


@dataclass(frozen=True, slots=True)
class AnalysisArtifactInspection:
    path: str
    storage_revision: tuple[int, int, int, int, int]
    kind: AnalysisArtifactKind
    group: str
    shape: tuple[int, ...]
    axes: tuple[tuple[str, int], ...]
    axis_units: tuple[tuple[str, str | None], ...]
    has_sigma: bool
    has_stitch_diagnostics: bool
    request_fingerprint: str
    source_fingerprint: str
    plan_fingerprint: str
    provenance_digest: str
    provenance_json: str = field(repr=False)
    provenance_sha256: str
    result_fingerprint: str
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    execution_attestation_digest: str | None = None
    execution_attestation_json: str | None = field(default=None, repr=False)


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactResultProjection:
    """The one exact little-endian float32 projection stored in an artifact."""

    kind: AnalysisArtifactKind
    axes: tuple[tuple[str, np.ndarray], ...]
    axis_units: tuple[tuple[str, str | None], ...]
    intensity: np.ndarray
    sigma: np.ndarray | None
    coverage: np.ndarray | None
    normalization: np.ndarray | None
    result_fingerprint: str
    policy: str = field(init=False, default=_STORED_RESULT_PROJECTION_POLICY)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        arrays = tuple(values for _name, values in self.axes) + (
            self.intensity,
        ) + (() if self.sigma is None else (self.sigma,)) + (
            ()
            if self.coverage is None
            else (self.coverage, self.normalization)
        )
        if (
            _claim is not _ANALYSIS_PROJECTION_FACTORY
            or type(self.kind) is not AnalysisArtifactKind
            or type(self.axes) is not tuple
            or type(self.axis_units) is not tuple
            or type(self.intensity) is not np.ndarray
            or (self.sigma is not None and type(self.sigma) is not np.ndarray)
            or (self.coverage is None) is not (self.normalization is None)
            or any(
                type(values) is not np.ndarray
                or values.dtype != np.dtype("<f4")
                or not values.flags.c_contiguous
                or values.flags.writeable
                for values in arrays
            )
        ):
            raise TypeError("analysis artifact result projection is invalid")
        _sha256(self.result_fingerprint, "result fingerprint")

    def __copy__(self):
        raise TypeError("analysis artifact result projection is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact result projection is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact result projection is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact result projection is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactPayload:
    """Detached scientific bytes with fresh read-only NumPy views."""

    inspection: AnalysisArtifactInspection
    _axis_values: InitVar[tuple[tuple[str, np.ndarray], ...]]
    _intensity_values: InitVar[np.ndarray]
    _sigma_values: InitVar[np.ndarray | None]
    _coverage_values: InitVar[np.ndarray | None]
    _normalization_values: InitVar[np.ndarray | None]
    _claim: InitVar[object] = None
    _axis_buffers: tuple[tuple[str, bytes], ...] = field(init=False, repr=False)
    _intensity_buffer: bytes = field(init=False, repr=False)
    _sigma_buffer: bytes | None = field(init=False, repr=False)
    _coverage_buffer: bytes | None = field(init=False, repr=False)
    _normalization_buffer: bytes | None = field(init=False, repr=False)

    def __post_init__(
        self,
        _axis_values: tuple[tuple[str, np.ndarray], ...],
        _intensity_values: np.ndarray,
        _sigma_values: np.ndarray | None,
        _coverage_values: np.ndarray | None,
        _normalization_values: np.ndarray | None,
        _claim: object,
    ) -> None:
        if (
            _claim is not _ANALYSIS_PAYLOAD_FACTORY
            or type(self.inspection) is not AnalysisArtifactInspection
            or type(_axis_values) is not tuple
            or type(_intensity_values) is not np.ndarray
            or (
                _sigma_values is not None
                and type(_sigma_values) is not np.ndarray
            )
            or (
                _coverage_values is not None
                and type(_coverage_values) is not np.ndarray
            )
            or (
                _normalization_values is not None
                and type(_normalization_values) is not np.ndarray
            )
        ):
            raise TypeError("analysis artifact payload is invalid")
        expected_axes = self.inspection.axes
        if (
            len(_axis_values) != len(expected_axes)
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not np.ndarray
                or item[0] != expected_axes[index][0]
                or item[1].shape != (expected_axes[index][1],)
                or item[1].dtype != np.dtype(np.float32)
                or item[1].flags.writeable
                or not item[1].flags.c_contiguous
                for index, item in enumerate(_axis_values)
            )
            or _intensity_values.shape != self.inspection.shape
            or _intensity_values.dtype != np.dtype(np.float32)
            or _intensity_values.flags.writeable
            or not _intensity_values.flags.c_contiguous
            or (_sigma_values is None) is self.inspection.has_sigma
            or (_coverage_values is None)
            is not (_normalization_values is None)
            or (_coverage_values is not None)
            is not self.inspection.has_stitch_diagnostics
            or (
                _sigma_values is not None
                and (
                    _sigma_values.shape != self.inspection.shape
                    or _sigma_values.dtype != np.dtype(np.float32)
                    or _sigma_values.flags.writeable
                    or not _sigma_values.flags.c_contiguous
                )
            )
            or any(
                values is not None
                and (
                    values.shape != self.inspection.shape
                    or values.dtype != np.dtype(np.float32)
                    or values.flags.writeable
                    or not values.flags.c_contiguous
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                )
                for values in (_coverage_values, _normalization_values)
            )
        ):
            raise TypeError("analysis artifact payload arrays are invalid")
        object.__setattr__(
            self,
            "_axis_buffers",
            tuple(
                (name, values.tobytes(order="C"))
                for name, values in _axis_values
            ),
        )
        object.__setattr__(
            self,
            "_intensity_buffer",
            _intensity_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_sigma_buffer",
            None if _sigma_values is None else _sigma_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_coverage_buffer",
            None
            if _coverage_values is None
            else _coverage_values.tobytes(order="C"),
        )
        object.__setattr__(
            self,
            "_normalization_buffer",
            None
            if _normalization_values is None
            else _normalization_values.tobytes(order="C"),
        )

    @property
    def kind(self) -> AnalysisArtifactKind:
        return self.inspection.kind

    @property
    def provenance_json(self) -> str:
        return self.inspection.provenance_json

    @property
    def result_fingerprint(self) -> str:
        return self.inspection.result_fingerprint

    @property
    def schema_version(self) -> int:
        return self.inspection.schema_version

    @property
    def execution_attestation_digest(self) -> str | None:
        return self.inspection.execution_attestation_digest

    @property
    def execution_attestation_json(self) -> str | None:
        return self.inspection.execution_attestation_json

    @staticmethod
    def _view(buffer: bytes, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(buffer, dtype=np.float32).reshape(shape)

    @property
    def axes(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple(
            (
                name,
                self._view(buffer, (self.inspection.axes[index][1],)),
            )
            for index, (name, buffer) in enumerate(self._axis_buffers)
        )

    @property
    def intensity(self) -> np.ndarray:
        return self._view(self._intensity_buffer, self.inspection.shape)

    @property
    def sigma(self) -> np.ndarray | None:
        if self._sigma_buffer is None:
            return None
        return self._view(self._sigma_buffer, self.inspection.shape)

    @property
    def coverage(self) -> np.ndarray | None:
        if self._coverage_buffer is None:
            return None
        return self._view(self._coverage_buffer, self.inspection.shape)

    @property
    def normalization(self) -> np.ndarray | None:
        if self._normalization_buffer is None:
            return None
        return self._view(self._normalization_buffer, self.inspection.shape)

    def axis(self, name: str) -> np.ndarray:
        if type(name) is not str:
            raise TypeError("analysis artifact axis name must be an exact string")
        for index, (axis_name, buffer) in enumerate(self._axis_buffers):
            if axis_name == name:
                return self._view(buffer, (self.inspection.axes[index][1],))
        raise KeyError(name)

    def __copy__(self):
        raise TypeError("analysis artifact payload is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact payload is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact payload is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact payload is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class AnalysisArtifactReceipt:
    request: AnalysisArtifactRequest
    terminal: StreamTerminal
    inspection: AnalysisArtifactInspection
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _ANALYSIS_RECEIPT_FACTORY
            or type(self.request) is not AnalysisArtifactRequest
            or type(self.terminal) is not StreamTerminal
            or self.terminal.target != self.request.target
            or type(self.inspection) is not AnalysisArtifactInspection
            or self.inspection.path != self.request.target
            or self.inspection.storage_revision
            != stream_terminal_object_revision(self.terminal)
            or self.inspection.kind is not self.request.kind
            or self.inspection.request_fingerprint
            != self.request.request_fingerprint
            or self.inspection.source_fingerprint
            != self.request.source_fingerprint
            or self.inspection.plan_fingerprint != self.request.plan_fingerprint
            or self.inspection.provenance_digest
            != self.request.provenance_digest
            or self.inspection.schema_version != self.request.schema_version
            or self.inspection.execution_attestation_digest
            != self.request.execution_attestation_digest
            or self.inspection.execution_attestation_json
            != self.request.execution_attestation_json
        ):
            raise TypeError("analysis artifact receipt is invalid")

    def __copy__(self):
        raise TypeError("analysis artifact receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact receipt is not copyable")

    def __reduce__(self):
        raise TypeError("analysis artifact receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("analysis artifact receipt is not serializable")


@dataclass(frozen=True, slots=True)
class AnalysisArtifactOutputSnapshot:
    target: str
    phase: TransactionPhase
    writer_started: bool
    writer_finished: bool
    remaining_lease_owners: tuple[LeaseOwner, ...]
    retryable: bool
    receipt: AnalysisArtifactReceipt | None = None


def _decode_fixed_text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict")
    raise AnalysisArtifactInvalid("analysis artifact attribute is not fixed text")


def _hdf_type_class(identifier) -> int:
    data_type = identifier.get_type()
    try:
        return int(data_type.get_class())
    finally:
        data_type.close()


def _bounded_attr_text(owner, name: str) -> str:
    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        attribute.shape != ()
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize < 1
        or attribute.dtype.itemsize > _MAX_ATTRIBUTE_BYTES
        or attribute.get_storage_size() > _MAX_ATTRIBUTE_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not bounded fixed text"
        )
    return _decode_fixed_text(owner.attrs[name])


def _exact_sha256_attr_text(owner, name: str) -> str:
    """Read one v2-only digest with no fixed-width NUL padding."""

    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        attribute.shape != ()
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize != 64
        or attribute.get_storage_size() != 64
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not exact digest text"
        )
    value = _decode_fixed_text(owner.attrs[name])
    if len(value.encode("utf-8")) != 64:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is padded digest text"
        )
    return value


def _bounded_attr_texts(owner, name: str) -> tuple[str, ...]:
    try:
        attribute = owner.attrs.get_id(name)
    except KeyError as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is missing"
        ) from error
    if (
        len(attribute.shape) != 1
        or attribute.shape[0] < 1
        or attribute.shape[0] > _MAX_ATTRIBUTE_ITEMS
        or _hdf_type_class(attribute) != h5py.h5t.STRING
        or attribute.dtype.kind != "S"
        or attribute.dtype.itemsize < 1
        or attribute.dtype.itemsize > _MAX_ATTRIBUTE_BYTES
        or attribute.get_storage_size()
        > _MAX_ATTRIBUTE_ITEMS * _MAX_ATTRIBUTE_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"analysis artifact attribute {name} is not bounded fixed text"
        )
    values = owner.attrs[name]
    return tuple(_decode_fixed_text(value) for value in values.tolist())


def _direct(group: h5py.Group, name: str, expected_type):
    try:
        link_type = group.id.links.get_info(name.encode("utf-8")).type
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(f"{group.name}/{name} is missing") from error
    if link_type != h5py.h5l.TYPE_HARD:
        raise AnalysisArtifactInvalid(f"{group.name}/{name} is not a local hard link")
    value = group.get(name)
    if not isinstance(value, expected_type):
        raise AnalysisArtifactInvalid(f"{group.name}/{name} has the wrong object type")
    if isinstance(value, h5py.Dataset):
        creation = value.id.get_create_plist()
        try:
            indirect = value.is_virtual or creation.get_external_count() != 0
        finally:
            creation.close()
        if indirect:
            raise AnalysisArtifactInvalid(f"{value.name} uses indirect storage")
    return value


def _object_address(value: h5py.Group | h5py.Dataset) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def _require_local_graph(group: h5py.Group, seen: set[int]) -> None:
    address = _object_address(group)
    if address in seen:
        raise AnalysisArtifactInvalid(f"hard-link alias detected at {group.name}")
    seen.add(address)
    for name in group:
        value = _direct(group, name, (h5py.Group, h5py.Dataset))
        address = _object_address(value)
        if address in seen:
            raise AnalysisArtifactInvalid(f"hard-link alias detected at {value.name}")
        if isinstance(value, h5py.Group):
            _require_local_graph(value, seen)
        else:
            seen.add(address)


def _read_scalar_text(group: h5py.Group, name: str) -> str:
    dataset = _direct(group, name, h5py.Dataset)
    if dataset.shape != ():
        raise AnalysisArtifactInvalid(f"{dataset.name} must be a scalar string")
    if len(dataset.attrs) != 0:
        raise AnalysisArtifactInvalid(f"{dataset.name} has unexpected attributes")
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.STRING
        or dataset.dtype.kind != "S"
    ):
        raise AnalysisArtifactInvalid(
            "analysis provenance must use bounded fixed-width storage"
        )
    if (
        dataset.dtype.itemsize < 1
        or dataset.dtype.itemsize > _MAX_PROVENANCE_BYTES
        or dataset.id.get_storage_size() > _MAX_PROVENANCE_BYTES
    ):
        raise AnalysisArtifactInvalid("analysis provenance is oversized")
    raw = dataset[()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="strict")
    elif isinstance(raw, str):
        text = raw
    else:
        raise AnalysisArtifactInvalid(f"{dataset.name} must be UTF-8 text")
    if len(text.encode("utf-8")) > _MAX_PROVENANCE_BYTES:
        raise AnalysisArtifactInvalid("analysis provenance is oversized")
    _require_json_nesting_bound(text)
    try:
        parsed = json.loads(text)
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid("analysis provenance is invalid JSON") from error
    if type(parsed) is not dict:
        raise AnalysisArtifactInvalid("analysis provenance must decode to an object")
    try:
        canonical = canonical_analysis_provenance(parsed)
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid("analysis provenance is not canonicalizable") from error
    if canonical != text:
        raise AnalysisArtifactInvalid("analysis provenance is not canonical JSON")
    return text


def _require_execution_attestation_nesting_bound(text: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_EXECUTION_ATTESTATION_NESTING:
                raise AnalysisArtifactInvalid(
                    "analysis execution attestation exceeds the nesting bound"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                break


def _read_scalar_execution_attestation(
    group: h5py.Group,
    name: str,
    *,
    kind: AnalysisArtifactKind,
    request_fingerprint: str,
    expected_digest: str,
) -> str:
    dataset = _direct(group, name, h5py.Dataset)
    if dataset.shape != () or len(dataset.attrs) != 0:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be an attribute-free scalar"
        )
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.STRING
        or dataset.dtype.kind != "S"
        or dataset.dtype.itemsize < 1
        or dataset.dtype.itemsize > _MAX_EXECUTION_ATTESTATION_BYTES
        or dataset.id.get_storage_size() > _MAX_EXECUTION_ATTESTATION_BYTES
    ):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must use bounded fixed-width storage"
        )
    raw = dataset[()]
    if not isinstance(raw, bytes):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be UTF-8 text"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation must be UTF-8 text"
        ) from error
    if len(text.encode("utf-8")) > _MAX_EXECUTION_ATTESTATION_BYTES:
        raise AnalysisArtifactInvalid("analysis execution attestation is oversized")
    if (
        dataset.dtype.itemsize != len(text.encode("utf-8"))
        or dataset.id.get_storage_size() != len(text.encode("utf-8"))
    ):
        raise AnalysisArtifactInvalid(
            "analysis execution attestation storage is not byte-exact"
        )
    _require_execution_attestation_nesting_bound(text)
    try:
        parsed = json.loads(text)
        canonical, frozen = _validated_execution_attestation(
            parsed,
            kind=kind,
            request_fingerprint=request_fingerprint,
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation is invalid"
        ) from error
    if canonical != text:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation is not canonical JSON"
        )
    if analysis_execution_attestation_digest(
        kind,
        frozen,
        request_fingerprint=request_fingerprint,
    ) != expected_digest:
        raise AnalysisArtifactInvalid(
            "analysis execution attestation digest changed"
        )
    return text


def _require_json_nesting_bound(text: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_PROVENANCE_NESTING:
                raise AnalysisArtifactInvalid(
                    "analysis provenance exceeds the nesting bound"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                break


def _axis(group: h5py.Group, name: str) -> np.ndarray:
    dataset = _direct(group, name, h5py.Dataset)
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.ndim != 1
        or dataset.size < 1
        or dataset.size > _MAX_AXIS_POINTS
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} is not a nonempty float32 axis")
    _require_bounded_chunk(dataset)
    values = np.asarray(dataset[()], dtype=np.float32)
    if not np.isfinite(values).all() or (
        values.size > 1 and not np.all(np.diff(values.astype(np.float64)) > 0)
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be finite and increasing")
    return values


def _digest_text(digest, value: str) -> None:
    raw = value.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _digest_array(digest, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype="<f4")
    _digest_text(digest, name)
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(memoryview(array).cast("B"))


def _require_bounded_chunk(dataset: h5py.Dataset) -> None:
    if dataset.chunks is not None and (
        math.prod(int(dimension) for dimension in dataset.chunks)
        * int(dataset.dtype.itemsize)
        > _MAX_READ_BLOCK_BYTES
    ):
        raise AnalysisArtifactInvalid(
            f"{dataset.name} storage chunks exceed the read bound"
        )


def _require_bounded_values(
    dataset: h5py.Dataset,
    *,
    digest=None,
    digest_name: str = "",
    require_finite: bool = False,
    require_nonnegative: bool = False,
) -> None:
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.size < 1
        or dataset.size > _MAX_RESULT_ELEMENTS
        or dataset.ndim < 1
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be nonempty float32")
    _require_bounded_chunk(dataset)
    finite_seen = False
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * 4)
    if row_bytes > _MAX_READ_BLOCK_BYTES:
        raise AnalysisArtifactInvalid(f"{dataset.name} rows exceed the read bound")
    rows = max(1, _MAX_READ_BLOCK_BYTES // row_bytes)
    if digest is not None:
        _digest_text(digest, digest_name)
        digest.update(dataset.ndim.to_bytes(8, "big"))
        for dimension in dataset.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
    pieces = (
        dataset[start : min(dataset.shape[0], start + rows)]
        for start in range(0, dataset.shape[0], rows)
    )
    for piece in pieces:
        values = np.asarray(piece)
        if np.isinf(values).any() or (
            require_finite and not np.isfinite(values).all()
        ):
            raise AnalysisArtifactInvalid(f"{dataset.name} contains infinity")
        if require_nonnegative and np.any(values < 0):
            raise AnalysisArtifactInvalid(
                f"{dataset.name} contains negative values"
            )
        finite_seen = finite_seen or bool(np.isfinite(values).any())
        if digest is not None:
            digest.update(
                memoryview(np.ascontiguousarray(values, dtype="<f4")).cast("B")
            )
    if not finite_seen:
        raise AnalysisArtifactInvalid(f"{dataset.name} has no finite result")


def _stat_revision(state: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _opened_hdf_revision(handle: h5py.File) -> tuple[int, int, int, int, int]:
    try:
        raw_handle = handle.id.get_vfd_handle()
        descriptor = raw_handle[0] if type(raw_handle) is tuple else raw_handle
        if isinstance(descriptor, bool):
            raise TypeError("boolean VFD handle")
        return _stat_revision(os.fstat(int(descriptor)))
    except (OSError, OverflowError, TypeError, ValueError) as error:
        raise AnalysisArtifactInvalid(
            "analysis artifact storage handle cannot be fenced"
        ) from error


def _freeze_float32(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    frozen = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.dtype("<f4")
    )
    return frozen.reshape(contiguous.shape)


def _project_stored_float32(
    values: object,
    *,
    name: str,
    require_finite: bool,
    require_nonnegative: bool = False,
    max_elements: int = _MAX_RESULT_ELEMENTS,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype.kind not in "biuf"
        or values.dtype.fields is not None
        or values.ndim < 1
        or values.size < 1
        or values.size > max_elements
    ):
        raise AnalysisArtifactProjectionInvalid(
            f"{name} is not an exact bounded real ndarray"
        )
    source = np.asarray(values)
    if source.dtype == np.dtype("<f4") and np.isnan(source).any():
        nan_bits = source.view(np.uint32)[np.isnan(source)]
        if np.any(nan_bits != _STORED_QNAN_F4_BITS):
            raise AnalysisArtifactProjectionInvalid(
                f"{name} contains a noncanonical stored NaN payload"
            )
    if np.isinf(source).any() or (require_finite and not np.isfinite(source).all()):
        raise AnalysisArtifactProjectionInvalid(f"{name} contains a nonfinite value")
    if require_nonnegative and np.any(source < 0):
        raise AnalysisArtifactProjectionInvalid(f"{name} contains a negative value")
    with np.errstate(over="ignore", invalid="ignore"):
        projected = np.array(source, dtype=np.dtype("<f4"), order="C", copy=True)
    if np.isinf(projected).any() or (
        require_finite and not np.isfinite(projected).all()
    ):
        raise AnalysisArtifactProjectionInvalid(
            f"{name} overflows the stored float32 representation"
        )
    if not require_finite:
        if not np.isfinite(projected).any():
            raise AnalysisArtifactProjectionInvalid(f"{name} has no finite result")
        projected.view(np.uint32)[np.isnan(projected)] = _STORED_QNAN_F4_BITS
    return _freeze_float32(projected)


def project_analysis_artifact_result(
    *,
    kind: AnalysisArtifactKind,
    axes: tuple[tuple[str, np.ndarray], ...],
    axis_units: tuple[tuple[str, str | None], ...],
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    coverage: np.ndarray | None,
    normalization: np.ndarray | None,
    _stored_v1_claim: object = None,
) -> AnalysisArtifactResultProjection:
    """Project one result exactly once to the bytes stored by the artifact.

    The function deliberately retains the v1 result-fingerprint framing.  It
    is shared by runners, writers, strict inspection, and payload hydration so
    a pre-publication fingerprint cannot describe different bytes from those
    admitted after publication.
    """

    if type(kind) is not AnalysisArtifactKind:
        raise TypeError("analysis result kind must be exact AnalysisArtifactKind")
    expected_axis_names = {
        AnalysisArtifactKind.STITCH_1D: ("q",),
        AnalysisArtifactKind.STITCH_2D: ("q", "chi"),
        AnalysisArtifactKind.RSM: ("h", "k", "l"),
    }[kind]

    def bounded_unit(value: object) -> bool:
        if type(value) is not str or not value or "\x00" in value:
            return False
        try:
            return len(value.encode("utf-8")) <= _MAX_ATTRIBUTE_BYTES
        except UnicodeEncodeError:
            return False

    if (
        type(axes) is not tuple
        or len(axes) != len(expected_axis_names)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != expected_axis_names[index]
            or type(item[1]) is not np.ndarray
            for index, item in enumerate(axes)
        )
        or type(axis_units) is not tuple
        or len(axis_units) != len(expected_axis_names)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or item[0] != expected_axis_names[index]
            or (
                kind in {
                    AnalysisArtifactKind.STITCH_1D,
                    AnalysisArtifactKind.STITCH_2D,
                }
                and not bounded_unit(item[1])
            )
            or (kind is AnalysisArtifactKind.RSM and item[1] is not None)
            for index, item in enumerate(axis_units)
        )
        or (coverage is None) is not (normalization is None)
        or (
            kind is AnalysisArtifactKind.RSM
            and (coverage is not None or sigma is not None)
        )
    ):
        raise AnalysisArtifactProjectionInvalid(
            "analysis result axes, units, or optional channels are invalid"
        )
    projected_axes: list[tuple[str, np.ndarray]] = []
    for name, values in axes:
        projected = _project_stored_float32(
            values,
            name=name,
            require_finite=True,
            max_elements=_MAX_AXIS_POINTS,
        )
        if projected.ndim != 1 or (
            projected.size > 1
            and not np.all(np.diff(projected.astype(np.float64)) > 0)
        ):
            raise AnalysisArtifactProjectionInvalid(
                f"{name} is not strictly increasing after float32 projection"
            )
        projected_axes.append((name, projected))
    expected_shape = tuple(values.size for _name, values in projected_axes)
    if (
        max(1, int(np.prod(expected_shape[1:], dtype=np.int64)) * 4)
        > _MAX_READ_BLOCK_BYTES
    ):
        raise AnalysisArtifactProjectionInvalid(
            "analysis result rows exceed the stored read bound"
        )
    projected_intensity = _project_stored_float32(
        intensity,
        name="intensity",
        require_finite=False,
    )
    if projected_intensity.shape != expected_shape:
        raise AnalysisArtifactProjectionInvalid(
            "analysis intensity shape does not match projected axes"
        )
    projected_sigma = None
    if sigma is not None:
        projected_sigma = _project_stored_float32(
            sigma,
            name="sigma",
            require_finite=False,
        )
        if projected_sigma.shape != expected_shape:
            raise AnalysisArtifactProjectionInvalid(
                "analysis sigma shape does not match projected axes"
            )
    projected_coverage = projected_normalization = None
    if coverage is not None:
        if (
            kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }
            and _stored_v1_claim is not _ANALYSIS_V1_DECODE_PROJECTION
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                invalid_counts = (
                    not np.isfinite(coverage).all()
                    or np.any(coverage < 0)
                    or np.any(coverage > 2**24)
                    or np.any(coverage != np.floor(coverage))
                )
            if invalid_counts:
                raise AnalysisArtifactProjectionInvalid(
                    "Stitch coverage must be exact bounded integer counts"
                )
            coverage_f4 = np.asarray(coverage, dtype=np.dtype("<f4"))
            if not np.array_equal(
                coverage_f4.astype(coverage.dtype), coverage
            ):
                raise AnalysisArtifactProjectionInvalid(
                    "Stitch coverage loses count identity in float32 storage"
                )
        projected_coverage = _project_stored_float32(
            coverage,
            name="coverage",
            require_finite=True,
            require_nonnegative=True,
        )
        projected_normalization = _project_stored_float32(
            normalization,
            name="normalization",
            require_finite=True,
            require_nonnegative=True,
        )
        if (
            projected_coverage.shape != expected_shape
            or projected_normalization.shape != expected_shape
        ):
            raise AnalysisArtifactProjectionInvalid(
                "analysis diagnostics shape does not match projected axes"
            )
    digest = hashlib.sha256(_RESULT_FINGERPRINT_PREFIX)
    _digest_text(digest, kind.value)
    for name, values in projected_axes:
        _digest_array(digest, name, values)
    for _name, units in axis_units:
        if units is not None:
            _digest_text(digest, units)
    _digest_array(digest, "intensity", projected_intensity)
    if projected_sigma is None:
        _digest_text(digest, "sigma:absent")
    else:
        _digest_array(digest, "sigma", projected_sigma)
    if projected_coverage is not None:
        _digest_array(digest, "coverage", projected_coverage)
        _digest_array(digest, "normalization", projected_normalization)
    return AnalysisArtifactResultProjection(
        kind,
        tuple(projected_axes),
        axis_units,
        projected_intensity,
        projected_sigma,
        projected_coverage,
        projected_normalization,
        digest.hexdigest(),
        _ANALYSIS_PROJECTION_FACTORY,
    )


def _project_analysis_artifact_result_v1_compat(
    *,
    kind: AnalysisArtifactKind,
    axes: tuple[tuple[str, np.ndarray], ...],
    axis_units: tuple[tuple[str, str | None], ...],
    intensity: np.ndarray,
    sigma: np.ndarray | None,
    coverage: np.ndarray | None,
    normalization: np.ndarray | None,
) -> AnalysisArtifactResultProjection:
    """Use the central projection for byte-compatible historical v1 Stitch."""

    return project_analysis_artifact_result(
        kind=kind,
        axes=axes,
        axis_units=axis_units,
        intensity=intensity,
        sigma=sigma,
        coverage=coverage,
        normalization=normalization,
        _stored_v1_claim=_ANALYSIS_V1_DECODE_PROJECTION,
    )


def _materialize_bounded_values(
    dataset: h5py.Dataset,
    *,
    require_finite: bool = False,
    require_nonnegative: bool = False,
) -> np.ndarray:
    if (
        _hdf_type_class(dataset.id) != h5py.h5t.FLOAT
        or dataset.dtype != np.dtype(np.float32)
        or dataset.size < 1
        or dataset.size > _MAX_RESULT_ELEMENTS
        or dataset.ndim < 1
    ):
        raise AnalysisArtifactInvalid(f"{dataset.name} must be nonempty float32")
    _require_bounded_chunk(dataset)
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * 4)
    if row_bytes > _MAX_READ_BLOCK_BYTES:
        raise AnalysisArtifactInvalid(f"{dataset.name} rows exceed the read bound")
    rows = max(1, _MAX_READ_BLOCK_BYTES // row_bytes)
    values = np.empty(dataset.shape, dtype=np.float32)
    finite_seen = False
    for start in range(0, dataset.shape[0], rows):
        stop = min(dataset.shape[0], start + rows)
        piece = np.asarray(dataset[start:stop], dtype=np.float32)
        if np.isinf(piece).any() or (
            require_finite and not np.isfinite(piece).all()
        ):
            raise AnalysisArtifactInvalid(f"{dataset.name} contains infinity")
        if require_nonnegative and np.any(piece < 0):
            raise AnalysisArtifactInvalid(
                f"{dataset.name} contains negative values"
            )
        finite_seen = finite_seen or bool(np.isfinite(piece).any())
        values[start:stop] = piece
    if not finite_seen:
        raise AnalysisArtifactInvalid(f"{dataset.name} has no finite result")
    return _freeze_float32(values)


# Detached admission and later payload hydration deliberately retain separate
# race seams even though both feed the same public stored-result projection.
_materialize_inspection_values = _materialize_bounded_values


def inspect_analysis_artifact(
    path: str | Path,
    *,
    expected_request: AnalysisArtifactRequest | None = None,
    expected_kind: AnalysisArtifactKind | None = None,
    _candidate_claim: object = None,
) -> AnalysisArtifactInspection:
    shown = _absolute_path(path)
    if expected_request is None and Path(shown).suffix.casefold() != ".nexus":
        raise ValueError("analysis artifact path must end in .nexus")
    if expected_request is not None and type(expected_request) is not AnalysisArtifactRequest:
        raise TypeError("expected request must be exact AnalysisArtifactRequest")
    if expected_kind is not None and type(expected_kind) is not AnalysisArtifactKind:
        raise TypeError("expected kind must be exact AnalysisArtifactKind")
    if (
        expected_request is not None
        and shown != expected_request.target
        and _candidate_claim is not _ANALYSIS_CANDIDATE_INSPECTION
    ):
        raise AnalysisArtifactInvalid(
            "analysis artifact path does not match the expected request"
        )
    if (
        expected_request is not None
        and expected_kind is not None
        and expected_kind is not expected_request.kind
    ):
        raise ValueError("expected artifact kind conflicts with expected request")
    try:
        admitted_state = os.stat(shown)
        if (
            not stat.S_ISREG(admitted_state.st_mode)
            or admitted_state.st_size < 1
            or admitted_state.st_size > _MAX_ARTIFACT_FILE_BYTES
        ):
            raise AnalysisArtifactInvalid(
                "analysis artifact must be a regular bounded file"
            )
        admitted_revision = _stat_revision(admitted_state)
        with h5py.File(shown, "r") as handle:
            if _opened_hdf_revision(handle) != admitted_revision:
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed while opening detached admission"
                )
            if _stat_revision(os.stat(shown)) != admitted_revision:
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed while opening detached admission"
                )
            if len(handle.attrs) != 0 or len(handle) != 1 or _ENTRY not in handle:
                raise AnalysisArtifactInvalid("analysis artifact must contain only /entry")
            entry = _direct(handle, _ENTRY, h5py.Group)
            try:
                version_id = entry.attrs.get_id(ANALYSIS_SCHEMA_VERSION_ATTR)
            except KeyError as error:
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is missing"
                ) from error
            if (
                version_id.shape != ()
                or _hdf_type_class(version_id) != h5py.h5t.INTEGER
                or version_id.dtype.kind not in "iu"
                or version_id.dtype.itemsize > 8
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is invalid"
                )
            version = int(entry.attrs[ANALYSIS_SCHEMA_VERSION_ATTR])
            if version not in {
                ANALYSIS_SCHEMA_VERSION,
                ANALYSIS_SCHEMA_VERSION_V2,
            }:
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version is unsupported"
                )
            expected_entry_attrs = {
                "NX_class",
                ANALYSIS_SCHEMA_ATTR,
                ANALYSIS_SCHEMA_VERSION_ATTR,
                ANALYSIS_KIND_ATTR,
                "request_fingerprint",
                "source_fingerprint",
                "plan_fingerprint",
                "provenance_digest",
                "file_name",
            }
            if version == ANALYSIS_SCHEMA_VERSION_V2:
                expected_entry_attrs.add(_EXECUTION_ATTESTATION_DIGEST_ATTR)
            if len(entry.attrs) != len(expected_entry_attrs) or any(
                name not in expected_entry_attrs for name in entry.attrs
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact entry attributes are not exact"
                )
            if _bounded_attr_text(entry, "NX_class") != "NXentry":
                raise AnalysisArtifactInvalid("analysis artifact entry is not NXentry")
            if _bounded_attr_text(entry, ANALYSIS_SCHEMA_ATTR) != ANALYSIS_SCHEMA_NAME:
                raise AnalysisArtifactInvalid("analysis artifact schema marker is wrong")
            try:
                kind = AnalysisArtifactKind(_bounded_attr_text(entry, ANALYSIS_KIND_ATTR))
            except ValueError as error:
                raise AnalysisArtifactInvalid("analysis artifact kind is invalid") from error
            expected = expected_request.kind if expected_request is not None else expected_kind
            if expected is not None and kind is not expected:
                raise AnalysisArtifactInvalid("analysis artifact kind does not match request")
            if (
                expected_request is not None
                and version != expected_request.schema_version
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact schema version changed"
                )
            required_attrs = (
                ("request_fingerprint", "request_fingerprint"),
                ("source_fingerprint", "source_fingerprint"),
                ("plan_fingerprint", "plan_fingerprint"),
                ("provenance_digest", "provenance_digest"),
            )
            identity_fingerprints: dict[str, str] = {}
            for attr, request_attr in required_attrs:
                value = _sha256(_bounded_attr_text(entry, attr), attr)
                identity_fingerprints[attr] = value
                if expected_request is not None and value != getattr(expected_request, request_attr):
                    raise AnalysisArtifactInvalid(f"analysis artifact {attr} changed")
            execution_attestation_digest = None
            if version == ANALYSIS_SCHEMA_VERSION_V2:
                execution_attestation_digest = _sha256(
                    _exact_sha256_attr_text(
                        entry, _EXECUTION_ATTESTATION_DIGEST_ATTR
                    ),
                    "execution attestation digest",
                )
                if (
                    expected_request is not None
                    and execution_attestation_digest
                    != expected_request.execution_attestation_digest
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact execution attestation digest changed"
                    )
            final_name = _bounded_attr_text(entry, "file_name")
            expected_name = (
                expected_request.target
                if expected_request is not None
                else shown
            )
            if final_name != expected_name:
                raise AnalysisArtifactInvalid("analysis artifact records a private candidate path")
            expected_entry_children = 2 + int(
                version == ANALYSIS_SCHEMA_VERSION_V2
            )
            if len(entry) != expected_entry_children:
                raise AnalysisArtifactInvalid(
                    "analysis artifact contains an unknown entry graph"
                )
            present = tuple(
                item for item in ("stitched_1d", "stitched_2d", "rsm")
                if item in entry
            )
            if present != (kind.group,):
                raise AnalysisArtifactInvalid("analysis artifact result groups are ambiguous")
            # C0 excludes frame/source-base graphs until S1/R1 bind one exact
            # typed manifest into the operation request.  Provenance carries
            # source identity meanwhile; an unbound raw-popup graph must not
            # acquire artifact authority merely by being local-hard HDF5.
            allowed = {kind.group, _PROVENANCE}
            if version == ANALYSIS_SCHEMA_VERSION_V2:
                allowed.add(_EXECUTION_ATTESTATION)
            if any(name not in allowed for name in entry):
                raise AnalysisArtifactInvalid("analysis artifact contains an unknown entry graph")
            provenance = _read_scalar_text(entry, _PROVENANCE)
            if expected_request is not None and provenance != expected_request.provenance_json:
                raise AnalysisArtifactInvalid("analysis artifact provenance changed")
            execution_attestation_json = None
            if version == ANALYSIS_SCHEMA_VERSION_V2:
                execution_attestation_json = _read_scalar_execution_attestation(
                    entry,
                    _EXECUTION_ATTESTATION,
                    kind=kind,
                    request_fingerprint=identity_fingerprints[
                        "request_fingerprint"
                    ],
                    expected_digest=execution_attestation_digest,
                )
                if (
                    expected_request is not None
                    and execution_attestation_json
                    != expected_request.execution_attestation_json
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact execution attestation changed"
                    )
            result = _direct(entry, kind.group, h5py.Group)
            expected_result_attrs = {"NX_class", "signal", "axes"}
            if len(result.attrs) != len(expected_result_attrs) or any(
                name not in expected_result_attrs for name in result.attrs
            ):
                raise AnalysisArtifactInvalid(
                    "analysis result attributes are not exact"
                )
            if _bounded_attr_text(result, "NX_class") != "NXdata":
                raise AnalysisArtifactInvalid("analysis result group is not NXdata")
            if _bounded_attr_text(result, "signal") != "intensity":
                raise AnalysisArtifactInvalid("analysis result signal is invalid")
            group_provenance = _read_scalar_text(result, _PROVENANCE)
            if group_provenance != provenance:
                raise AnalysisArtifactInvalid("result and artifact provenance differ")
            axis_names = {
                AnalysisArtifactKind.STITCH_1D: ("q",),
                AnalysisArtifactKind.STITCH_2D: ("q", "chi"),
                AnalysisArtifactKind.RSM: ("h", "k", "l"),
            }[kind]
            if _bounded_attr_texts(result, "axes") != axis_names:
                raise AnalysisArtifactInvalid("analysis result axes are invalid")
            allowed_result = {"intensity", _PROVENANCE, *axis_names}
            if kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }:
                allowed_result.update({"sigma", "coverage", "normalization"})
            if any(name not in allowed_result for name in result):
                raise AnalysisArtifactInvalid(
                    "analysis result contains an unknown graph"
                )
            has_sigma = "sigma" in result
            has_stitch_diagnostics = (
                "coverage" in result and "normalization" in result
            )
            if ("coverage" in result) != ("normalization" in result):
                raise AnalysisArtifactInvalid(
                    "stitched coverage and normalization must be paired"
                )
            expected_result_size = (
                2
                + len(axis_names)
                + int(has_sigma)
                + 2 * int(has_stitch_diagnostics)
            )
            if len(result) != expected_result_size:
                raise AnalysisArtifactInvalid(
                    "analysis result contains an unknown graph"
                )
            axes = tuple((name, _axis(result, name)) for name in axis_names)
            for name, _values in axes:
                axis = _direct(result, name, h5py.Dataset)
                expected_axis_attrs = (
                    {"units"}
                    if kind in {
                        AnalysisArtifactKind.STITCH_1D,
                        AnalysisArtifactKind.STITCH_2D,
                    }
                    else set()
                )
                if len(axis.attrs) != len(expected_axis_attrs) or any(
                    attr not in expected_axis_attrs for attr in axis.attrs
                ):
                    raise AnalysisArtifactInvalid(
                        f"analysis axis {name} attributes are not exact"
                    )
            axis_units: list[tuple[str, str | None]] = [
                (name, None) for name, _values in axes
            ]
            if kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }:
                q = _direct(result, "q", h5py.Dataset)
                q_units = _bounded_attr_text(q, "units")
                if not q_units:
                    raise AnalysisArtifactInvalid("stitched q units are invalid")
                axis_units[0] = ("q", q_units)
            if kind is AnalysisArtifactKind.STITCH_2D:
                chi = _direct(result, "chi", h5py.Dataset)
                chi_units = _bounded_attr_text(chi, "units")
                if not chi_units:
                    raise AnalysisArtifactInvalid("stitched chi units are invalid")
                axis_units[1] = ("chi", chi_units)
            if version == ANALYSIS_SCHEMA_VERSION_V2 and (
                kind is not AnalysisArtifactKind.STITCH_1D
                or axis_units != [("q", "q_A^-1")]
                or has_sigma
                or not has_stitch_diagnostics
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact v2 requires the exact XU Stitch result schema"
                )
            intensity = _direct(result, "intensity", h5py.Dataset)
            if len(intensity.attrs) != 0:
                raise AnalysisArtifactInvalid(
                    "analysis intensity attributes are not exact"
                )
            expected_shape = tuple(len(values) for _name, values in axes)
            if intensity.shape != expected_shape:
                raise AnalysisArtifactInvalid("analysis intensity shape does not match axes")
            intensity_values = _materialize_inspection_values(intensity)
            sigma_values = None
            if has_sigma:
                sigma = _direct(result, "sigma", h5py.Dataset)
                if len(sigma.attrs) != 0:
                    raise AnalysisArtifactInvalid(
                        "stitched sigma attributes are not exact"
                    )
                if sigma.shape != expected_shape:
                    raise AnalysisArtifactInvalid(
                        "stitched sigma shape does not match axes"
                    )
                sigma_values = _materialize_inspection_values(sigma)
            coverage_values = normalization_values = None
            if has_stitch_diagnostics:
                for name in ("coverage", "normalization"):
                    values = _direct(result, name, h5py.Dataset)
                    if len(values.attrs) != 0 or values.shape != expected_shape:
                        raise AnalysisArtifactInvalid(
                            f"stitched {name} must be an exact result-shaped dataset"
                        )
                coverage_values = _materialize_inspection_values(
                    _direct(result, "coverage", h5py.Dataset),
                    require_finite=True,
                    require_nonnegative=True,
                )
                normalization_values = _materialize_inspection_values(
                    _direct(result, "normalization", h5py.Dataset),
                    require_finite=True,
                    require_nonnegative=True,
                )
            try:
                projection = project_analysis_artifact_result(
                    kind=kind,
                    axes=axes,
                    axis_units=tuple(axis_units),
                    intensity=intensity_values,
                    sigma=sigma_values,
                    coverage=coverage_values,
                    normalization=normalization_values,
                    _stored_v1_claim=(
                        _ANALYSIS_V1_DECODE_PROJECTION
                        if version == ANALYSIS_SCHEMA_VERSION
                        else None
                    ),
                )
            except AnalysisArtifactProjectionInvalid as error:
                raise AnalysisArtifactInvalid(str(error)) from error
            if version == ANALYSIS_SCHEMA_VERSION_V2:
                empty_coverage = projection.coverage == 0
                empty_normalization = projection.normalization == 0
                empty_intensity = np.isnan(projection.intensity)
                if (
                    not np.array_equal(empty_coverage, empty_normalization)
                    or not np.array_equal(empty_coverage, empty_intensity)
                    or not np.isfinite(
                        projection.intensity[~empty_coverage]
                    ).all()
                    or np.any(projection.coverage[~empty_coverage] <= 0)
                    or np.any(
                        projection.normalization[~empty_coverage] <= 0
                    )
                ):
                    raise AnalysisArtifactInvalid(
                        "analysis artifact v2 empty and occupied bins disagree"
                    )
            if (
                execution_attestation_json is not None
                and json.loads(execution_attestation_json).get(
                    "result_fingerprint"
                )
                != projection.result_fingerprint
            ):
                raise AnalysisArtifactInvalid(
                    "analysis execution attestation result changed"
                )
            seen: set[int] = set()
            _require_local_graph(entry, seen)
            inspection = AnalysisArtifactInspection(
                path=shown,
                storage_revision=admitted_revision,
                kind=kind,
                group=kind.group,
                shape=expected_shape,
                axes=tuple((name, len(values)) for name, values in axes),
                axis_units=tuple(axis_units),
                has_sigma=has_sigma,
                has_stitch_diagnostics=has_stitch_diagnostics,
                request_fingerprint=identity_fingerprints["request_fingerprint"],
                source_fingerprint=identity_fingerprints["source_fingerprint"],
                plan_fingerprint=identity_fingerprints["plan_fingerprint"],
                provenance_digest=identity_fingerprints["provenance_digest"],
                provenance_json=provenance,
                provenance_sha256=hashlib.sha256(
                    provenance.encode("utf-8")
                ).hexdigest(),
                result_fingerprint=projection.result_fingerprint,
                schema_version=version,
                execution_attestation_digest=execution_attestation_digest,
                execution_attestation_json=execution_attestation_json,
            )
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise AnalysisArtifactInvalid(
                    "analysis artifact changed during detached admission"
                )
        final_state = os.stat(shown)
        if _stat_revision(final_state) != admitted_revision:
            raise AnalysisArtifactInvalid(
                "analysis artifact changed during detached admission"
            )
        return inspection
    except AnalysisArtifactInvalid:
        raise
    except (
        OSError,
        KeyError,
        TargetChanged,
        TypeError,
        ValueError,
        UnicodeError,
    ) as error:
        raise AnalysisArtifactInvalid(f"analysis artifact admission failed: {error}") from error


def read_analysis_artifact(
    path: str | Path,
    *,
    expected_request: AnalysisArtifactRequest | None = None,
    expected_kind: AnalysisArtifactKind | None = None,
    expected_receipt: AnalysisArtifactReceipt | None = None,
) -> AnalysisArtifactPayload:
    """Strictly admit and detach one standalone Stitch or RSM payload."""

    if expected_request is not None and type(expected_request) is not AnalysisArtifactRequest:
        raise TypeError("expected request must be exact AnalysisArtifactRequest")
    if expected_kind is not None and type(expected_kind) is not AnalysisArtifactKind:
        raise TypeError("expected kind must be exact AnalysisArtifactKind")
    if expected_receipt is not None:
        if type(expected_receipt) is not AnalysisArtifactReceipt:
            raise TypeError("expected receipt must be exact AnalysisArtifactReceipt")
        if (
            expected_request is not None
            and expected_request is not expected_receipt.request
        ):
            raise ValueError("expected request conflicts with expected receipt")
        if (
            expected_kind is not None
            and expected_kind is not expected_receipt.request.kind
        ):
            raise ValueError("expected artifact kind conflicts with expected receipt")
        expected_request = expected_receipt.request
        expected_kind = expected_receipt.request.kind

    shown = _absolute_path(path)
    try:
        if expected_receipt is not None:
            revalidate_stream_terminal(shown, expected_receipt.terminal)
        inspection = inspect_analysis_artifact(
            shown,
            expected_request=expected_request,
            expected_kind=expected_kind,
        )
        if (
            expected_receipt is not None
            and inspection != expected_receipt.inspection
        ):
            raise TargetChanged(
                "analysis artifact no longer matches its commit receipt"
            )
        admitted_revision = inspection.storage_revision
        if _stat_revision(os.stat(shown)) != admitted_revision:
            raise TargetChanged("analysis artifact changed before payload hydration")

        with h5py.File(shown, "r") as handle:
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise TargetChanged(
                    "analysis artifact changed while opening payload"
                )
            entry = _direct(handle, _ENTRY, h5py.Group)
            result = _direct(entry, inspection.group, h5py.Group)
            axes = tuple(
                (name, _axis(result, name))
                for name, _length in inspection.axes
            )
            intensity = _materialize_bounded_values(
                _direct(result, "intensity", h5py.Dataset)
            )
            if intensity.shape != inspection.shape:
                raise AnalysisArtifactInvalid(
                    "analysis payload shape changed after admission"
                )
            sigma = None
            if inspection.has_sigma:
                sigma = _materialize_bounded_values(
                    _direct(result, "sigma", h5py.Dataset)
                )
                if sigma.shape != inspection.shape:
                    raise AnalysisArtifactInvalid(
                        "analysis sigma shape changed after admission"
                    )
            coverage = normalization = None
            if inspection.has_stitch_diagnostics:
                coverage = _materialize_bounded_values(
                    _direct(result, "coverage", h5py.Dataset),
                    require_finite=True,
                    require_nonnegative=True,
                )
                normalization = _materialize_bounded_values(
                    _direct(result, "normalization", h5py.Dataset),
                    require_finite=True,
                    require_nonnegative=True,
                )
                if (
                    coverage.shape != inspection.shape
                    or normalization.shape != inspection.shape
                ):
                    raise AnalysisArtifactInvalid(
                        "stitched diagnostics changed shape after admission"
                    )

            units = dict(inspection.axis_units)
            if inspection.kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }:
                q_units = _bounded_attr_text(
                    _direct(result, "q", h5py.Dataset),
                    "units",
                )
                if q_units != units["q"]:
                    raise AnalysisArtifactInvalid(
                        "analysis q units changed after admission"
                    )
            if inspection.kind is AnalysisArtifactKind.STITCH_2D:
                chi_units = _bounded_attr_text(
                    _direct(result, "chi", h5py.Dataset),
                    "units",
                )
                if chi_units != units["chi"]:
                    raise AnalysisArtifactInvalid(
                        "analysis chi units changed after admission"
                    )
            try:
                projection = project_analysis_artifact_result(
                    kind=inspection.kind,
                    axes=axes,
                    axis_units=inspection.axis_units,
                    intensity=intensity,
                    sigma=sigma,
                    coverage=coverage,
                    normalization=normalization,
                    _stored_v1_claim=(
                        _ANALYSIS_V1_DECODE_PROJECTION
                        if inspection.schema_version
                        == ANALYSIS_SCHEMA_VERSION
                        else None
                    ),
                )
            except AnalysisArtifactProjectionInvalid as error:
                raise AnalysisArtifactInvalid(str(error)) from error
            if projection.result_fingerprint != inspection.result_fingerprint:
                raise TargetChanged(
                    "analysis payload changed after detached admission"
                )
            if (
                _opened_hdf_revision(handle) != admitted_revision
                or _stat_revision(os.stat(shown)) != admitted_revision
            ):
                raise TargetChanged(
                    "analysis artifact changed during payload hydration"
                )
        if _stat_revision(os.stat(shown)) != admitted_revision:
            raise TargetChanged(
                "analysis artifact changed during payload hydration"
            )
        if expected_receipt is not None:
            revalidate_stream_terminal(shown, expected_receipt.terminal)
        return AnalysisArtifactPayload(
            inspection,
            projection.axes,
            projection.intensity,
            projection.sigma,
            projection.coverage,
            projection.normalization,
            _ANALYSIS_PAYLOAD_FACTORY,
        )
    except AnalysisArtifactInvalid:
        raise
    except (
        OSError,
        KeyError,
        TargetChanged,
        TypeError,
        ValueError,
        UnicodeError,
    ) as error:
        raise AnalysisArtifactInvalid(
            f"analysis artifact payload admission failed: {error}"
        ) from error


class AnalysisArtifactOutput:
    """Factory-owned exact lease and one-shot artifact publisher."""

    def __init__(
        self,
        request: AnalysisArtifactRequest,
        *,
        coordinator: OutputTransactionCoordinator,
    ) -> None:
        if type(request) is not AnalysisArtifactRequest:
            raise TypeError("analysis output requires exact request")
        parent = Path(request.target).parent
        if not parent.is_dir():
            raise ValueError("analysis artifact parent directory must already exist")
        if (
            request.overwrite is AnalysisArtifactOverwrite.CREATE_NEW
            and os.path.lexists(request.target)
        ):
            raise FileExistsError(request.target)
        self.request = request
        self._transaction_owner = OwnerToken("analysis-artifact-transaction")
        self._target_owner = OwnerToken("analysis-artifact-target")
        self._owners = {
            role: OwnerToken(f"analysis-artifact-{role.value}")
            for role in LeaseOwner
        }
        self._transaction = coordinator.admit(
            request.target,
            transaction_owner=self._transaction_owner,
            target_owner=self._target_owner,
        )
        if (
            request.overwrite is AnalysisArtifactOverwrite.CREATE_NEW
            and self._transaction.admission.snapshot.exists
        ):
            raise FileExistsError(request.target)
        self._lease = self._transaction.acquire_lease(
            admission=self._transaction.admission,
            transaction_owner=self._transaction_owner,
            target_owner=self._target_owner,
            owners=self._owners,
        )
        self._remaining = list(LeaseOwner)
        self._writer_started = False
        self._writer_finished = False
        self._validated_candidate: TargetSnapshot | None = None
        self._receipt: AnalysisArtifactReceipt | None = None

    def __copy__(self):
        raise TypeError("analysis artifact output owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("analysis artifact output owner is not copyable")

    @property
    def snapshot(self) -> AnalysisArtifactOutputSnapshot:
        state = self._transaction.snapshot()
        return AnalysisArtifactOutputSnapshot(
            self.request.target,
            state.phase,
            self._writer_started,
            self._writer_finished,
            tuple(self._remaining),
            state.retryable or bool(self._remaining and state.phase in {
                TransactionPhase.COMMITTED,
                TransactionPhase.ABORTED,
            }),
            self._receipt,
        )

    def _release(self) -> None:
        for role in tuple(LeaseOwner):
            if role not in self._remaining:
                continue
            self._transaction.release_lease_owner(
                self._lease, role, self._owners[role]
            )
            self._remaining.remove(role)

    def _write_candidate(
        self,
        candidate: Path,
        write_result: Callable[[h5py.Group], object],
    ) -> None:
        self._writer_started = True
        with h5py.File(candidate, "w") as handle:
            entry = handle.create_group(_ENTRY)
            entry.attrs.create("NX_class", np.bytes_(b"NXentry"))
            entry.attrs.create(
                ANALYSIS_SCHEMA_ATTR,
                np.bytes_(ANALYSIS_SCHEMA_NAME.encode("utf-8")),
            )
            entry.attrs[ANALYSIS_SCHEMA_VERSION_ATTR] = self.request.schema_version
            for name, value in (
                (ANALYSIS_KIND_ATTR, self.request.kind.value),
                ("request_fingerprint", self.request.request_fingerprint),
                ("source_fingerprint", self.request.source_fingerprint),
                ("plan_fingerprint", self.request.plan_fingerprint),
                ("provenance_digest", self.request.provenance_digest),
                ("file_name", self.request.target),
            ):
                entry.attrs.create(name, np.bytes_(value.encode("utf-8")))
            if self.request.schema_version == ANALYSIS_SCHEMA_VERSION_V2:
                entry.attrs.create(
                    _EXECUTION_ATTESTATION_DIGEST_ATTR,
                    np.bytes_(
                        self.request.execution_attestation_digest.encode(
                            "utf-8"
                        )
                    ),
                )
            entry.create_dataset(
                _PROVENANCE,
                data=np.bytes_(self.request.provenance_json.encode("utf-8")),
            )
            if self.request.schema_version == ANALYSIS_SCHEMA_VERSION_V2:
                entry.create_dataset(
                    _EXECUTION_ATTESTATION,
                    data=np.bytes_(
                        self.request.execution_attestation_json.encode("utf-8")
                    ),
                )
            write_result(entry)
            handle.flush()
        descriptor = os.open(candidate, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _validate_candidate(
        self,
        candidate: Path,
        captured: TargetSnapshot,
        prepublish: Callable[[], object] | None,
    ) -> None:
        before = capture_target_snapshot(candidate)
        if before != captured:
            raise TargetChanged(
                "analysis candidate changed before semantic readback"
            )
        inspect_analysis_artifact(
            candidate,
            expected_request=self.request,
            _candidate_claim=_ANALYSIS_CANDIDATE_INSPECTION,
        )
        after = capture_target_snapshot(candidate)
        if not after.exists or before != after:
            raise TargetChanged(
                "analysis candidate changed during semantic readback"
            )
        if prepublish is not None:
            prepublish()
        final = capture_target_snapshot(candidate)
        if final != after:
            raise TargetChanged(
                "analysis candidate changed after prepublication checks"
            )
        self._validated_candidate = final
        self._writer_finished = True

    def _terminal(self) -> tuple[StreamTerminal, AnalysisArtifactInspection]:
        before = capture_target_snapshot(self.request.target)
        inspection = inspect_analysis_artifact(
            self.request.target, expected_request=self.request
        )
        captured = capture_target_snapshot(self.request.target)
        if (
            before != captured
            or self._validated_candidate is None
            or captured != self._validated_candidate
            or not captured.exists
            or captured.size is None
            or captured.digest is None
            or captured.device is None
            or captured.inode is None
            or captured.mtime_ns is None
        ):
            raise TargetChanged("committed analysis artifact lacks exact storage facts")
        state = os.stat(self.request.target)
        if (
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_size),
            int(state.st_mtime_ns),
        ) != (
            captured.device,
            captured.inode,
            captured.size,
            captured.mtime_ns,
        ):
            raise TargetChanged("analysis artifact changed while sealing commit")
        terminal = StreamTerminal(
            self.request.target,
            captured.size,
            captured.digest,
            self._transaction.admission.ordinal,
            int(state.st_dev),
            int(state.st_ino),
            int(state.st_mtime_ns),
            int(state.st_ctime_ns),
        )
        revalidate_stream_terminal(self.request.target, terminal)
        final_capture = capture_target_snapshot(self.request.target)
        final_state = os.stat(self.request.target)
        if final_capture != captured or (
            int(final_state.st_dev),
            int(final_state.st_ino),
            int(final_state.st_size),
            int(final_state.st_mtime_ns),
            int(final_state.st_ctime_ns),
        ) != (
            terminal.device,
            terminal.inode,
            terminal.size,
            terminal.mtime_ns,
            terminal.ctime_ns,
        ):
            raise TargetChanged("analysis artifact changed after terminal validation")
        return terminal, inspection

    def _seal_receipt(self) -> AnalysisArtifactReceipt:
        if self._receipt is None:
            terminal, inspection = self._terminal()
            self._receipt = AnalysisArtifactReceipt(
                self.request,
                terminal,
                inspection,
                _ANALYSIS_RECEIPT_FACTORY,
            )
        else:
            revalidate_stream_terminal(
                self.request.target,
                self._receipt.terminal,
            )
            inspection = inspect_analysis_artifact(
                self.request.target,
                expected_request=self.request,
            )
            revalidate_stream_terminal(
                self.request.target,
                self._receipt.terminal,
            )
            if inspection != self._receipt.inspection:
                raise TargetChanged(
                    "cached analysis artifact receipt no longer matches the target"
                )
        return self._receipt

    def _settle_failed_writer(self) -> None:
        phase = self._transaction.snapshot().phase
        if phase is TransactionPhase.READY_TO_RETRY:
            self._transaction.abort(
                admission=self._transaction.admission,
                transaction_owner=self._transaction_owner,
                target_owner=self._target_owner,
                lease=self._lease,
            )
            phase = TransactionPhase.ABORTED
        elif phase is TransactionPhase.LEASED:
            self._transaction.abandon(self._lease)
            phase = TransactionPhase.ABORTED
        if phase is TransactionPhase.ABORTED:
            self._release()

    def _settle_or_raise(self, primary: BaseException) -> None:
        settlement_error: BaseException | None = None
        try:
            self._settle_failed_writer()
        except BaseException as error:
            settlement_error = error
            try:
                primary.add_note(
                    f"analysis artifact settlement also failed: "
                    f"{type(error).__module__}.{type(error).__qualname__}: {error}"
                )
            except BaseException:
                pass
        state = self.snapshot
        if settlement_error is not None or state.retryable:
            raise AnalysisArtifactCleanupPending(state) from primary
        raise primary.with_traceback(primary.__traceback__)

    def _publish_owned(
        self,
        write_result: Callable[[h5py.Group], object],
        *,
        prepublish: Callable[[], object] | None = None,
        retain_lease: bool,
    ) -> AnalysisArtifactReceipt:
        if not callable(write_result):
            raise TypeError("analysis artifact writer must be callable")
        if prepublish is not None and not callable(prepublish):
            raise TypeError("analysis artifact prepublication check must be callable")
        if self._writer_started or self._receipt is not None:
            raise TransactionStateError("analysis artifact writer is one-shot")
        try:
            outcome = self._transaction.execute(
                lambda candidate: self._write_candidate(candidate, write_result),
                admission=self._transaction.admission,
                transaction_owner=self._transaction_owner,
                target_owner=self._target_owner,
                lease=self._lease,
                validate_writer_result=lambda candidate, captured: (
                    self._validate_candidate(candidate, captured, prepublish)
                ),
            )
        except BaseException as error:
            self._settle_or_raise(error)
        if outcome.phase is not TransactionPhase.COMMITTED:
            raise TransactionStateError("analysis artifact did not reach committed phase")
        try:
            receipt = self._seal_receipt()
        except BaseException as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from error
        if not retain_lease:
            try:
                self._release()
            except BaseException as error:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
        return receipt

    def publish(
        self,
        write_result: Callable[[h5py.Group], object],
        *,
        prepublish: Callable[[], object] | None = None,
    ) -> AnalysisArtifactReceipt:
        return self._publish_owned(
            write_result,
            prepublish=prepublish,
            retain_lease=False,
        )

    def _publish_retained(
        self,
        write_result: Callable[[h5py.Group], object],
        *,
        prepublish: Callable[[], object] | None = None,
    ) -> AnalysisArtifactReceipt:
        return self._publish_owned(
            write_result,
            prepublish=prepublish,
            retain_lease=True,
        )

    def abort(self) -> AnalysisArtifactOutputSnapshot:
        if self._receipt is not None:
            if self._remaining:
                try:
                    self._release()
                except BaseException as error:
                    raise AnalysisArtifactCleanupPending(self.snapshot) from error
            return self.snapshot
        phase = self._transaction.snapshot().phase
        if phase in {TransactionPhase.LEASED, TransactionPhase.READY_TO_RETRY}:
            try:
                if phase is TransactionPhase.LEASED:
                    self._transaction.abandon(self._lease)
                else:
                    self._transaction.abort(
                        admission=self._transaction.admission,
                        transaction_owner=self._transaction_owner,
                        target_owner=self._target_owner,
                        lease=self._lease,
                    )
            except BaseException as error:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
        elif phase not in {TransactionPhase.ABORTED}:
            raise TransactionStateError(
                f"analysis artifact cannot abort in phase {phase.value}"
            )
        try:
            self._release()
        except BaseException as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from error
        return self.snapshot

    def _retry_cleanup_owned(
        self,
        *,
        retain_lease: bool,
    ) -> AnalysisArtifactOutputSnapshot:
        try:
            state = self._transaction.snapshot()
            if state.cleanup_token is not None and state.retryable:
                state = self._transaction.retry_cleanup(state.cleanup_token)
            if (
                state.phase in {
                    TransactionPhase.LEASED,
                    TransactionPhase.INTEGRITY_HOLD,
                }
                and not self._writer_started
            ):
                self._transaction.abandon(self._lease)
                state = self._transaction.snapshot()
            if state.phase is TransactionPhase.READY_TO_RETRY:
                self._transaction.abort(
                    admission=self._transaction.admission,
                    transaction_owner=self._transaction_owner,
                    target_owner=self._target_owner,
                    lease=self._lease,
                )
                state = self._transaction.snapshot()
            if state.phase is TransactionPhase.COMMITTED:
                self._seal_receipt()
                if not retain_lease:
                    self._release()
                return self.snapshot
            if state.phase is TransactionPhase.ABORTED:
                self._release()
                return self.snapshot
        except AnalysisArtifactCleanupPending:
            raise
        except BaseException as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from error
        raise AnalysisArtifactCleanupPending(self.snapshot)

    def retry_cleanup(self) -> AnalysisArtifactOutputSnapshot:
        return self._retry_cleanup_owned(retain_lease=False)

    def _retry_retained(self) -> AnalysisArtifactOutputSnapshot:
        return self._retry_cleanup_owned(retain_lease=True)

    def _complete_retained(self) -> AnalysisArtifactOutputSnapshot:
        state = self._transaction.snapshot()
        if state.phase is not TransactionPhase.COMMITTED or self._receipt is None:
            raise TransactionStateError(
                "retained analysis artifact is not ready for final release"
            )
        self._seal_receipt()
        self._release()
        return self.snapshot


def admit_analysis_artifact(
    request: AnalysisArtifactRequest,
    *,
    coordinator: OutputTransactionCoordinator | None = None,
) -> AnalysisArtifactOutput:
    if type(request) is not AnalysisArtifactRequest:
        raise TypeError("analysis artifact admission requires exact request")
    selected = get_output_transaction_coordinator() if coordinator is None else coordinator
    if type(selected) is not OutputTransactionCoordinator:
        raise TypeError("analysis artifact coordinator must be exact")
    return AnalysisArtifactOutput(request, coordinator=selected)


__all__ = [
    "ANALYSIS_KIND_ATTR",
    "ANALYSIS_SCHEMA_ATTR",
    "ANALYSIS_SCHEMA_NAME",
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_SCHEMA_VERSION_V2",
    "ANALYSIS_SCHEMA_VERSION_ATTR",
    "AnalysisArtifactCleanupPending",
    "AnalysisArtifactError",
    "AnalysisArtifactInspection",
    "AnalysisArtifactInvalid",
    "AnalysisArtifactKind",
    "AnalysisArtifactOutput",
    "AnalysisArtifactOutputSnapshot",
    "AnalysisArtifactOverwrite",
    "AnalysisArtifactPayload",
    "AnalysisArtifactProjectionInvalid",
    "AnalysisArtifactReceipt",
    "AnalysisArtifactResultProjection",
    "AnalysisArtifactRequest",
    "admit_analysis_artifact",
    "analysis_execution_attestation_digest",
    "canonical_analysis_execution_attestation",
    "canonical_analysis_provenance",
    "inspect_analysis_artifact",
    "project_analysis_artifact_result",
    "read_analysis_artifact",
]
