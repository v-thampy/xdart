"""Exact, blocking headless operation for one standalone Stitch artifact.

The generic module transaction owns source and output custody.  This module
adds the Stitch-specific boundary: bounded geometry assets, a canonical plan,
cooperative frame-load cancellation, deterministic provenance, and strict
post-commit payload admission.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field, replace
from enum import Enum
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import threading

import numpy as np

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleArtifactOutput,
    ModuleArtifactRefused,
    ModuleDisposition,
    ModuleKind,
    ModuleOperationRequest,
    ModuleOutputRequest,
    ModuleProgress,
    ModuleSourceReceipt,
    ModuleTerminalResult,
    admit_module_artifact,
    module_artifact_request,
    module_plan_fingerprint,
    module_provenance_digest,
    xu_stitch_module_request,
)
from xrd_tools.analysis.plans import (
    StitchCancelled,
    StitchPlan,
    run_stitch,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    AnalysisSourceLeaseRefused,
    MetadataTablePlan,
    requalified_analysis_source,
    run_metadata_table,
)
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationRefused,
    XuStitchCalibrationReceipt,
    revalidate_xu_stitch_calibration,
)
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.geometry import AngleMapping, Diffractometer, ImageOrientation
from xrd_tools.core.scan import SourceKind
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL,
    ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactPayload,
    AnalysisArtifactProjectionInvalid,
    _project_analysis_artifact_result_v1_compat,
    analysis_execution_attestation_digest,
    project_analysis_artifact_result,
    read_analysis_artifact,
)
from xrd_tools.io.nexus import write_stitched
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.integrate.multi import StitchDiagnostics
from xrd_tools.integrate.xu_stitch import (
    XuStitchCancelled,
    XuStitchEffectiveGeometryProjection,
    XuStitchScienceObservations,
    XuStitchScienceRefused,
    resolve_xu_stitch_effective_geometry,
    run_xu_hist_stitch_1d,
)
from xrd_tools.core.geometry.xu_runtime import (
    XuRuntimeRequirements,
    XuRuntimeUnsupported,
    xu_runtime_session,
)
from xrd_tools.sources.base import BaseFrameSource


_MAX_GEOMETRY_BYTES = 1 << 20
_MAX_MANIFEST_BYTES = 512 << 10
_MAX_CONTRIBUTIONS = 4096
_MAX_MANIFEST_FILES = 8192
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 65_536
_MAX_AXIS_POINTS = 1_000_000
_MAX_2D_POINTS = 64_000_000
_GEOMETRY_FACTORY = object()
_REQUEST_FACTORY = object()


class StitchGeometryKind(str, Enum):
    PYFAI_GONIOMETER_JSON = "pyfai_goniometer_json"
    PONI = "poni"


class StitchOperationRefused(ValueError):
    """An input could not be admitted as the requested Stitch operation."""

    def __init__(self, code: str, message: str | None = None):
        self.code = code
        super().__init__(message or code)


class StitchOperationVerificationError(RuntimeError):
    """A committed output could not be strictly re-admitted for presentation."""

    def __init__(self, execution: "StitchOperationExecution", message: str):
        self.execution = execution
        super().__init__(message)


class StitchOperationCleanupPending(AnalysisArtifactCleanupPending):
    """Retryable cleanup retaining the exact Stitch execution owner."""

    def __init__(self, execution: "StitchOperationExecution"):
        self.execution = execution
        super().__init__(execution.output_snapshot)

    def retry_cleanup(self) -> "StitchOperationResult":
        return self.execution.retry_cleanup()


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise TypeError(f"{name} must be lowercase SHA-256 hex")
    return value


def _file_state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        identity_ctime_ns(value.st_ctime_ns),
    )


def _failure_diagnostic(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        message = "exception message unavailable"
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}: {message}"[:4096]


def _motor_mapping(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError("source motor mapping must be an exact tuple")
    items: list[tuple[str, str]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(part) is not str or not part or part.strip() != part for part in item)
        ):
            raise TypeError("source motor mapping entries must be exact string pairs")
        items.append(item)
    result = tuple(items)
    if (
        result != tuple(sorted(result))
        or len({left for left, _right in result}) != len(result)
        or len({right for _left, right in result}) != len(result)
    ):
        raise ValueError("source motor mapping must be sorted and one-to-one")
    return result


def _motor_references(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) is not tuple:
        raise TypeError("PONI motor references must be an exact tuple")
    items: list[tuple[str, float]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or item[0].strip() != item[0]
            or type(item[1]) not in {int, float}
            or not math.isfinite(float(item[1]))
        ):
            raise TypeError(
                "PONI motor references must be exact finite name/value pairs"
            )
        items.append((item[0], float(item[1])))
    result = tuple(items)
    if result != tuple(sorted(result)) or len({name for name, _ in result}) != len(result):
        raise ValueError("PONI motor references must be sorted and unique")
    return result


def _range(value: object, name: str, *, required: bool) -> tuple[float, float] | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) not in {int, float} for item in value)
        or any(not math.isfinite(float(item)) for item in value)
        or float(value[0]) >= float(value[1])
    ):
        raise ValueError(f"{name} must be a finite ordered pair")
    return float(value[0]), float(value[1])


def _json_position_names(value: Mapping[str, object]) -> tuple[str, ...]:
    transformation = value.get("trans_function")
    if type(transformation) is not dict:
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED",
            "goniometer JSON has no exact transformation object",
        )
    raw_positions = transformation.get("pos_names")
    if (
        type(raw_positions) is not list
        or not 1 <= len(raw_positions) <= 32
        or any(
            type(name) is not str
            or not name
            or name.strip() != name
            or len(name) > 128
            for name in raw_positions
        )
        or len(set(raw_positions)) != len(raw_positions)
    ):
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED",
            "goniometer JSON position names are invalid",
        )
    top_level = value.get("pos_names")
    if top_level is not None and top_level != raw_positions:
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED",
            "goniometer JSON position-name declarations disagree",
        )
    return tuple(raw_positions)


def _json_mapping(raw: bytes) -> dict[str, object]:
    def no_duplicates(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    try:
        value = json.loads(
            raw.decode("utf-8-sig", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as error:
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED", "goniometer JSON could not be parsed"
        ) from error
    if type(value) is not dict:
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED", "goniometer JSON must contain one object"
        )
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise StitchOperationRefused(
                "GEOMETRY_PARSE_FAILED", "goniometer JSON exceeds structural bounds"
            )
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or len(key) > 4096:
                    raise StitchOperationRefused(
                        "GEOMETRY_PARSE_FAILED", "goniometer JSON key is invalid"
                    )
                stack.append((child, depth + 1))
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) in {int, float}:
            try:
                finite = math.isfinite(float(item))
            except OverflowError:
                finite = False
            if not finite:
                raise StitchOperationRefused(
                    "GEOMETRY_PARSE_FAILED", "goniometer JSON number is non-finite"
                )
        elif type(item) is str:
            if len(item) > 65_536:
                raise StitchOperationRefused(
                    "GEOMETRY_PARSE_FAILED", "goniometer JSON string is oversized"
                )
        elif item is not None and type(item) is not bool:
            raise StitchOperationRefused(
                "GEOMETRY_PARSE_FAILED", "goniometer JSON value is unsupported"
            )
    detector = value.get("detector")
    config = value.get("detector_config")
    if (
        type(detector) is not str
        or not detector
        or len(detector) > 128
        or any(mark in detector for mark in ("\0", "/", "\\", "://"))
        or type(config) is not dict
        or not set(config) <= {"orientation", "binning"}
    ):
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED", "goniometer detector configuration is unsafe"
        )
    orientation = config.get("orientation", 3)
    if type(orientation) is not int or orientation not in range(1, 5):
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED", "detector orientation must be 1 through 4"
        )
    if "binning" in config and (
        type(config["binning"]) not in {list, tuple}
        or len(config["binning"]) != 2
        or any(type(item) is not int or item < 1 for item in config["binning"])
    ):
        raise StitchOperationRefused(
            "GEOMETRY_PARSE_FAILED", "detector binning must contain two positive integers"
        )
    _json_position_names(value)
    return value


@dataclass(frozen=True, slots=True)
class StitchGeometryInput:
    path: str | Path
    kind: StitchGeometryKind
    expected_sha256: str | None = None
    source_motors: tuple[tuple[str, str], ...] = ()
    reference_motor_positions: tuple[tuple[str, float], ...] = ()
    image_rotation: int = 0
    base_preset: str = "psic"

    def __post_init__(self) -> None:
        if type(self.kind) is not StitchGeometryKind:
            raise TypeError("geometry kind must be exact StitchGeometryKind")
        try:
            path = os.path.abspath(os.path.expanduser(os.fspath(self.path)))
        except TypeError as error:
            raise TypeError("geometry path must be path-like") from error
        suffix = Path(path).suffix.casefold()
        expected_suffix = (
            ".json"
            if self.kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON
            else ".poni"
        )
        if suffix != expected_suffix:
            raise ValueError(f"geometry path must end in {expected_suffix}")
        if self.expected_sha256 is not None:
            _sha256(self.expected_sha256, "expected geometry digest")
        motors = _motor_mapping(self.source_motors)
        references = _motor_references(self.reference_motor_positions)
        if not motors:
            raise ValueError("Stitch geometry requires an explicit source motor mapping")
        if self.kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON and references:
            raise ValueError("fitted goniometer JSON cannot carry PONI references")
        if self.kind is StitchGeometryKind.PONI and (
            {name for name, _source in motors} != {"del", "nu"}
            or {name for name, _value in references} != {"del", "nu"}
        ):
            raise ValueError(
                "PONI geometry requires del/nu source mappings and reference positions"
            )
        if type(self.image_rotation) is not int or self.image_rotation not in {0, 90, 180, 270}:
            raise ValueError("image rotation must be an exact 0/90/180/270 integer")
        if self.base_preset != "psic":
            raise ValueError("S1 supports only the canonical psic base preset")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "source_motors", motors)
        object.__setattr__(self, "reference_motor_positions", references)


@dataclass(eq=False, frozen=True, slots=True)
class StitchGeometryReceipt:
    request: StitchGeometryInput
    lexical_path: str
    resolved_path: str
    byte_count: int
    sha256: str
    pre_state: tuple[int, int, int, int, int, int]
    post_state: tuple[int, int, int, int, int, int]
    fingerprint: str = field(init=False)
    _content_input: InitVar[bytes] = b""
    _claim: InitVar[object] = None
    _content: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self, _content_input: bytes, _claim: object) -> None:
        if (
            _claim is not _GEOMETRY_FACTORY
            or type(self.request) is not StitchGeometryInput
            or type(self.lexical_path) is not str
            or type(self.resolved_path) is not str
            or type(self.byte_count) is not int
            or type(_content_input) is not bytes
            or self.byte_count != len(_content_input)
            or not 0 < self.byte_count <= _MAX_GEOMETRY_BYTES
            or type(self.pre_state) is not tuple
            or type(self.post_state) is not tuple
            or len(self.pre_state) != 6
            or self.pre_state != self.post_state
        ):
            raise TypeError("stitch geometry receipt is invalid")
        _sha256(self.sha256, "geometry digest")
        if hashlib.sha256(_content_input).hexdigest() != self.sha256:
            raise ValueError("geometry receipt content does not match its digest")
        fingerprint = module_plan_fingerprint(
            ModuleKind.STITCH,
            (
                "stitch-geometry-v1",
                self.request.kind,
                self.lexical_path,
                self.resolved_path,
                self.byte_count,
                self.sha256,
                self.pre_state,
                self.request.source_motors,
                self.request.reference_motor_positions,
                self.request.image_rotation,
                self.request.base_preset,
            ),
        )
        object.__setattr__(self, "_content", bytes(_content_input))
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def content(self) -> bytes:
        return bytes(self._content)

    def __copy__(self):
        raise TypeError("stitch geometry receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("stitch geometry receipt is not copyable")

    def __reduce__(self):
        raise TypeError("stitch geometry receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("stitch geometry receipt is not serializable")


def capture_stitch_geometry(request: StitchGeometryInput) -> StitchGeometryReceipt:
    """Capture and validate one bounded, non-symlink geometry file."""

    if type(request) is not StitchGeometryInput:
        raise TypeError("geometry capture requires exact StitchGeometryInput")
    lexical = Path(request.path)
    try:
        lexical_state = os.lstat(lexical)
        if stat.S_ISLNK(lexical_state.st_mode):
            raise StitchOperationRefused(
                "GEOMETRY_SYMLINK_REFUSED", "geometry file cannot be a symlink"
            )
        resolved = lexical.resolve(strict=True)
        with builtins.open(resolved, "rb") as handle:
            pre = os.fstat(handle.fileno())
            if not stat.S_ISREG(pre.st_mode):
                raise StitchOperationRefused(
                    "GEOMETRY_UNAVAILABLE", "geometry asset is not a regular file"
                )
            raw = handle.read(_MAX_GEOMETRY_BYTES + 1)
            post = os.fstat(handle.fileno())
        current = os.stat(resolved, follow_symlinks=False)
    except StitchOperationRefused:
        raise
    except OSError as error:
        raise StitchOperationRefused(
            "GEOMETRY_UNAVAILABLE", "geometry asset is unavailable"
        ) from error
    pre_state, post_state = _file_state(pre), _file_state(post)
    if pre_state != post_state or post_state != _file_state(current):
        raise StitchOperationRefused(
            "GEOMETRY_REVISION_CHANGED", "geometry changed during capture"
        )
    if not raw:
        raise StitchOperationRefused("GEOMETRY_PARSE_FAILED", "geometry asset is empty")
    if len(raw) > _MAX_GEOMETRY_BYTES:
        raise StitchOperationRefused(
            "GEOMETRY_BYTE_LIMIT_EXCEEDED", "geometry asset exceeds 1 MiB"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if request.expected_sha256 is not None and digest != request.expected_sha256:
        raise StitchOperationRefused(
            "GEOMETRY_HASH_MISMATCH", "geometry digest does not match expectation"
        )
    if request.kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON:
        declared_positions = _json_position_names(_json_mapping(raw))
        bound_positions = tuple(name for name, _source in request.source_motors)
        if set(bound_positions) != set(declared_positions):
            raise StitchOperationRefused(
                "GEOMETRY_MOTOR_MAPPING_MISMATCH",
                "goniometer JSON source mapping must bind every declared "
                "position exactly once and no others",
            )
    else:
        try:
            from xrd_tools.integrate.calibration import load_detector_calibration

            load_detector_calibration(resolved, data=raw)
        except (ImportError, TypeError, ValueError) as error:
            raise StitchOperationRefused(
                "GEOMETRY_PARSE_FAILED", "PONI geometry is not one strict configured record"
            ) from error
    return StitchGeometryReceipt(
        request,
        str(lexical),
        str(resolved),
        len(raw),
        digest,
        pre_state,
        post_state,
        _content_input=raw,
        _claim=_GEOMETRY_FACTORY,
    )


def _revalidate_geometry(receipt: StitchGeometryReceipt) -> bytes:
    current = capture_stitch_geometry(receipt.request)
    fields = (
        "lexical_path",
        "resolved_path",
        "byte_count",
        "sha256",
        "pre_state",
        "post_state",
        "fingerprint",
    )
    if any(getattr(current, name) != getattr(receipt, name) for name in fields):
        raise StitchOperationRefused(
            "GEOMETRY_IDENTITY_MISMATCH", "geometry no longer matches its receipt"
        )
    return current.content


def _relative_locator(
    path: str | Path,
    project_root: Path,
    name: str,
    *,
    must_exist: bool = True,
) -> str:
    try:
        resolved = Path(path).expanduser().resolve(strict=must_exist)
        relative = resolved.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise StitchOperationRefused(
            "INPUT_OUTSIDE_PROJECT",
            f"{name} must resolve inside the selected Project root",
        ) from error
    value = relative.as_posix()
    if not value or value == "." or len(value.encode("utf-8")) > 4096:
        raise StitchOperationRefused(
            "INPUT_LOCATOR_INVALID", f"{name} has no bounded Project-relative locator"
        )
    return value


def _portable_json_value(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise StitchOperationRefused(
            "SOURCE_SPEC_INVALID", "source options exceed the nesting bound"
        )
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key or len(key) > 4096:
                raise StitchOperationRefused(
                    "SOURCE_SPEC_INVALID", "source option key is invalid"
                )
            result[key] = _portable_json_value(item, depth=depth + 1)
        return result
    if type(value) in {tuple, list}:
        return [_portable_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _portable_json_value(value.value, depth=depth + 1)
    if isinstance(value, np.generic):
        return _portable_json_value(value.item(), depth=depth + 1)
    if type(value) in {str, bool, int} or value is None:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise StitchOperationRefused(
        "SOURCE_SPEC_INVALID", "source options contain an unsupported value"
    )


def _revision(value: object, name: str) -> tuple[int, int, int, int, int, int]:
    if (
        type(value) is not tuple
        or len(value) != 6
        or any(type(item) is not int for item in value)
    ):
        raise TypeError(f"{name} must be one exact file revision")
    return value


def _manifest_relative_path(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096:
        raise TypeError(f"{name} must be a bounded relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class StitchManifestFile:
    relative_path: str
    revision: tuple[int, int, int, int, int, int] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _manifest_relative_path(self.relative_path, "manifest file path"),
        )
        if self.revision is not None:
            _revision(self.revision, "manifest file revision")


@dataclass(frozen=True, slots=True)
class StitchContribution:
    label: int
    file_ordinal: int
    source_frame_index: int
    values: tuple[tuple[str, int, float], ...]

    def __post_init__(self) -> None:
        if (
            type(self.label) is not int
            or type(self.file_ordinal) is not int
            or self.file_ordinal < 0
            or type(self.source_frame_index) is not int
            or self.source_frame_index < 0
            or type(self.values) is not tuple
        ):
            raise TypeError("Stitch contribution is invalid")
        parsed: list[tuple[str, int, float]] = []
        for item in self.values:
            if (
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 0
                or type(item[2]) is not float
                or not math.isfinite(item[2])
            ):
                raise TypeError("Stitch contribution metadata value is invalid")
            parsed.append(item)
        if tuple(parsed) != tuple(sorted(parsed, key=lambda item: (item[0], item[1]))):
            raise ValueError("Stitch contribution metadata values must be sorted")
        if len({(name, occurrence) for name, occurrence, _value in parsed}) != len(parsed):
            raise ValueError("Stitch contribution metadata values must be unique")


@dataclass(eq=False, frozen=True, slots=True)
class StitchInputManifestReceipt:
    project_root: str
    source_relative_path: str
    source_kind: str
    source_entry: str | None
    source_scan: str | None
    source_spec_digest: str
    source_options_json: str = field(repr=False)
    primary_revision: tuple[int, int, int, int, int, int]
    files: tuple[StitchManifestFile, ...]
    contributions: tuple[StitchContribution, ...]
    source_fingerprint: str
    module_source_fingerprint: str
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        try:
            root = Path(self.project_root)
            options = json.loads(self.source_options_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("Stitch input manifest is invalid") from error
        if (
            _claim is not _REQUEST_FACTORY
            or type(self.project_root) is not str
            or not root.is_absolute()
            or type(self.source_kind) is not str
            or not self.source_kind
            or (self.source_entry is not None and type(self.source_entry) is not str)
            or (self.source_scan is not None and type(self.source_scan) is not str)
            or type(options) is not dict
            or type(self.files) is not tuple
            or not self.files
            or len(self.files) > _MAX_MANIFEST_FILES
            or any(type(item) is not StitchManifestFile for item in self.files)
            or len({item.relative_path for item in self.files}) != len(self.files)
            or type(self.contributions) is not tuple
            or not self.contributions
            or len(self.contributions) > _MAX_CONTRIBUTIONS
            or any(type(item) is not StitchContribution for item in self.contributions)
            or any(item.file_ordinal >= len(self.files) for item in self.contributions)
            or len({item.label for item in self.contributions})
            != len(self.contributions)
        ):
            raise TypeError("Stitch input manifest is invalid")
        canonical_options = json.dumps(
            options,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if canonical_options != self.source_options_json:
            raise ValueError("Stitch input manifest source options are not canonical")
        object.__setattr__(
            self,
            "source_relative_path",
            _manifest_relative_path(self.source_relative_path, "source path"),
        )
        _revision(self.primary_revision, "source primary revision")
        _sha256(self.source_spec_digest, "source spec digest")
        _sha256(self.source_fingerprint, "source fingerprint")
        _sha256(self.module_source_fingerprint, "module source fingerprint")
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(ModuleKind.STITCH, self._canonical_value()),
        )
        encoded = json.dumps(
            self.to_provenance(),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise StitchOperationRefused(
                "MANIFEST_BYTE_LIMIT_EXCEEDED",
                "Stitch persisted input manifest exceeds 512 KiB",
            )

    def _canonical_value(self) -> tuple[object, ...]:
        return (
            "stitch-input-manifest-v1",
            self.source_relative_path,
            self.source_kind,
            self.source_entry,
            self.source_scan,
            self.source_spec_digest,
            json.loads(self.source_options_json),
            self.primary_revision,
            tuple((item.relative_path, item.revision) for item in self.files),
            tuple(
                (
                    item.label,
                    item.file_ordinal,
                    item.source_frame_index,
                    item.values,
                )
                for item in self.contributions
            ),
            self.source_fingerprint,
            self.module_source_fingerprint,
        )

    def to_provenance(self) -> dict[str, object]:
        return {
            "schema_version": "stitch-input-manifest-v1",
            "source": {
                "relative_path": self.source_relative_path,
                "kind": self.source_kind,
                "entry": self.source_entry,
                "scan": self.source_scan,
                "source_spec_digest": self.source_spec_digest,
                "options": json.loads(self.source_options_json),
                "primary_revision": list(self.primary_revision),
                "source_fingerprint": self.source_fingerprint,
                "module_source_fingerprint": self.module_source_fingerprint,
            },
            "files": [
                {
                    "relative_path": item.relative_path,
                    "revision": (
                        None if item.revision is None else list(item.revision)
                    ),
                }
                for item in self.files
            ],
            "contributions": [
                {
                    "label": item.label,
                    "file_ordinal": item.file_ordinal,
                    "source_frame_index": item.source_frame_index,
                    "values": [
                        {"name": name, "occurrence": occurrence, "value": value}
                        for name, occurrence, value in item.values
                    ],
                }
                for item in self.contributions
            ],
            "fingerprint": self.fingerprint,
        }

    def __copy__(self):
        raise TypeError("Stitch input manifest is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("Stitch input manifest is not copyable")

    def __reduce__(self):
        raise TypeError("Stitch input manifest is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("Stitch input manifest is not serializable")


def _project_source_options(analysis, root: Path) -> str:
    """Return the closed, Project-relative S1 source-policy projection."""

    if analysis.source_spec.metadata_uri is not None:
        raise StitchOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            "S1 Stitch does not accept the unused SourceSpec.metadata_uri field",
        )
    options = dict(analysis.source_spec.options)
    if analysis.resolved_kind is SourceKind.SPEC:
        allowed = {"scan", "image_dir", "image_stem", "read_image_kwargs"}
        if (
            analysis.resolved_entry is not None
            or not analysis.resolved_scan
            or options.get("scan") != analysis.resolved_scan
            or not options.get("image_dir")
        ):
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 SPEC Stitch requires one resolved scan and image directory",
            )
        unexpected = set(options) - allowed
        if unexpected:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                f"S1 SPEC source has unsupported options {sorted(unexpected)}",
            )
        projected = dict(options)
        projected["image_dir"] = _relative_locator(
            options["image_dir"], root, "SPEC image directory"
        )
        read_options = options.get("read_image_kwargs", {})
        if type(read_options) is not dict:
            read_options = dict(read_options)
        allowed_read = {
            "detector_shape",
            "detector",
            "raw_dtype",
            "raw_header_skip",
            "threshold",
            "rotation",
        }
        unexpected_read = set(read_options) - allowed_read
        if unexpected_read:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 SPEC image reader has unsupported options "
                f"{sorted(unexpected_read)}",
            )
        projected["read_image_kwargs"] = read_options
    elif analysis.resolved_kind is SourceKind.TIFF_SERIES:
        allowed = {
            "selected_file",
            "files",
            "pattern",
            "scan_name",
            "metadata_format",
            "selection_mode",
            "meta_dir",
            "detector_shape",
            "detector",
            "raw_dtype",
            "raw_header_skip",
            "admitted_motor_values",
        }
        if analysis.resolved_entry is not None or analysis.resolved_scan is not None:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 TIFF Stitch does not accept container entry or scan fields",
            )
        unexpected = set(options) - allowed
        if unexpected:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                f"S1 TIFF source has unsupported options {sorted(unexpected)}",
            )
        projected = dict(options)
        selected = options.get("selected_file")
        files = options.get("files")
        if type(selected) is not str or not selected or type(files) is not tuple or not files:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 TIFF Stitch requires one frozen selected file and member tuple",
            )
        if any(type(item) is not str or not item for item in files):
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 TIFF Stitch members must be exact nonempty path strings",
            )
        try:
            source_root = Path(analysis.resolved_root).resolve(strict=True)
            selected_resolved = Path(selected).expanduser().resolve(strict=True)
            members_resolved = tuple(
                Path(item).expanduser().resolve(strict=True) for item in files
            )
        except OSError as error:
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 TIFF Stitch source intent is no longer available",
            ) from error
        selection_mode = options.get("selection_mode")
        if (
            not source_root.is_dir()
            or selected_resolved.parent != source_root
            or any(member.parent != source_root for member in members_resolved)
            or sum(member == selected_resolved for member in members_resolved) != 1
            or len(set(members_resolved)) != len(members_resolved)
            or selection_mode not in {None, "single_image"}
            or (selection_mode == "single_image" and len(members_resolved) != 1)
        ):
            raise StitchOperationRefused(
                "SOURCE_SPEC_UNSUPPORTED",
                "S1 TIFF Stitch requires one coherent selected member, series root, "
                "and single-image marker",
            )
        projected["selected_file"] = _relative_locator(
            selected_resolved, root, "TIFF selected file"
        )
        projected["files"] = tuple(
            _relative_locator(item, root, "TIFF member")
            for item in members_resolved
        )
        if options.get("meta_dir") not in {None, ""}:
            projected["meta_dir"] = _relative_locator(
                options["meta_dir"], root, "TIFF metadata directory"
            )
        admitted = options.get("admitted_motor_values")
        if admitted is not None:
            if type(admitted) is not tuple:
                raise StitchOperationRefused(
                    "SOURCE_SPEC_UNSUPPORTED",
                    "TIFF admitted motor values must be one frozen tuple",
                )
            projected_rows: list[tuple[str, str, float]] = []
            for item in admitted:
                if (
                    type(item) is not tuple
                    or len(item) != 3
                    or type(item[1]) is not str
                    or not item[1]
                    or type(item[2]) is not float
                    or not math.isfinite(item[2])
                ):
                    raise StitchOperationRefused(
                        "SOURCE_SPEC_UNSUPPORTED",
                        "TIFF admitted motor value is invalid",
                    )
                projected_rows.append(
                    (
                        _relative_locator(
                            item[0], root, "TIFF admitted motor member"
                        ),
                        item[1],
                        item[2],
                    )
                )
            projected["admitted_motor_values"] = tuple(projected_rows)
    else:
        raise StitchOperationRefused("SOURCE_KIND_UNSUPPORTED")

    portable = _portable_json_value(projected)
    return json.dumps(
        portable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _manifest_numeric_lookup(
    table,
    selectors: tuple[MetadataColumnSelector, ...],
) -> tuple[dict[int, int], tuple[tuple[MetadataColumnSelector, np.ndarray], ...]]:
    positions: dict[int, int] = {}
    for index, label in enumerate(table.labels):
        if type(label) is not int or label in positions:
            raise StitchOperationRefused(
                "SOURCE_IDENTITY_MISMATCH", "metadata labels are not exact and unique"
            )
        positions[label] = index
    columns: dict[str, list[object]] = {}
    for column in table.columns:
        columns.setdefault(column.name, []).append(column)
    selected: list[tuple[MetadataColumnSelector, np.ndarray]] = []
    for selector in selectors:
        matches = columns.get(selector.name, ())
        if (
            selector.occurrence >= len(matches)
            or matches[selector.occurrence].numeric is None
        ):
            raise StitchOperationRefused(
                "INVALID_METADATA_SELECTOR",
                f"metadata column {selector.name!r} is not a numeric input",
            )
        values = matches[selector.occurrence].numeric
        if values.shape != (len(positions),):
            raise StitchOperationRefused(
                "SOURCE_IDENTITY_MISMATCH", "metadata column length changed"
            )
        selected.append((selector, values))
    return positions, tuple(selected)


def _capture_input_manifest(
    source: ModuleSourceReceipt,
    table,
    project_root: str | Path,
    selectors: tuple[MetadataColumnSelector, ...],
) -> StitchInputManifestReceipt:
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except (OSError, TypeError) as error:
        raise StitchOperationRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is unavailable"
        ) from error
    if not root.is_dir():
        raise StitchOperationRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is not a directory"
        )
    analysis = source.analysis
    source_relative = _relative_locator(analysis.resolved_root, root, "source")
    source_options_json = _project_source_options(analysis, root)
    if analysis.primary_post_state is None:
        raise StitchOperationRefused(
            "SOURCE_IDENTITY_MISMATCH", "source has no exact primary revision"
        )
    dependency_rows = (
        (
            analysis.lexical_root,
            analysis.resolved_root,
            analysis.primary_post_state,
        ),
        *analysis.dependency_revisions,
    )
    files: list[StitchManifestFile] = []
    file_ordinals: dict[Path, int] = {}
    dependency_by_resolved: dict[
        Path, tuple[int, int, int, int, int, int] | None
    ] = {}
    contributions: list[StitchContribution] = []
    charge = len(source_options_json.encode("utf-8")) + 1024

    def charge_piece(value: object) -> None:
        nonlocal charge
        charge += len(
            json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ) + 1
        if charge > _MAX_MANIFEST_BYTES:
            raise StitchOperationRefused(
                "MANIFEST_BYTE_LIMIT_EXCEEDED",
                "Stitch input manifest exceeds 512 KiB while being captured",
            )

    for _lexical, raw_resolved, revision in dependency_rows:
        resolved = Path(raw_resolved)
        previous = dependency_by_resolved.get(resolved, ...)
        if previous is not ...:
            if previous != revision:
                raise StitchOperationRefused(
                    "SOURCE_IDENTITY_MISMATCH",
                    "source dependency revisions disagree",
                )
            continue
        if len(files) >= _MAX_MANIFEST_FILES:
            raise StitchOperationRefused(
                "MANIFEST_FILE_LIMIT_EXCEEDED",
                "Stitch input manifest exceeds 8192 dependency files",
            )
        relative = _relative_locator(
            resolved,
            root,
            "source dependency",
            must_exist=revision is not None,
        )
        item = StitchManifestFile(relative, revision)
        ordinal = len(files)
        files.append(item)
        file_ordinals[resolved] = ordinal
        dependency_by_resolved[resolved] = revision
        charge_piece(
            {
                "relative_path": item.relative_path,
                "revision": None if revision is None else list(revision),
            }
        )

    label_positions, selected_columns = _manifest_numeric_lookup(table, selectors)
    try:
        with requalified_analysis_source(analysis) as opened:
            for label in source.selected_labels:
                if len(contributions) >= _MAX_CONTRIBUTIONS:
                    raise StitchOperationRefused(
                        "MANIFEST_CONTRIBUTION_LIMIT_EXCEEDED",
                        "Stitch selection exceeds 4096 contributions",
                    )
                frame = opened.frame_for(label)
                if frame.source_path is None or type(frame.source_frame_index) is not int:
                    raise StitchOperationRefused(
                        "RAW_LOCATOR_MISSING",
                        f"selected frame {label} has no exact raw source locator",
                    )
                try:
                    resolved = Path(frame.source_path).expanduser().resolve(strict=True)
                    state = _file_state(os.stat(resolved, follow_symlinks=False))
                except OSError as error:
                    raise StitchOperationRefused(
                        "RAW_SOURCE_UNAVAILABLE",
                        f"selected frame {label} raw source is unavailable",
                    ) from error
                if not stat.S_ISREG(state[0]) or dependency_by_resolved.get(resolved) != state:
                    raise StitchOperationRefused(
                        "SOURCE_IDENTITY_MISMATCH",
                        f"selected frame {label} raw source is outside its receipt",
                    )
                try:
                    ordinal = file_ordinals[resolved]
                    position = label_positions[label]
                except KeyError as error:
                    raise StitchOperationRefused(
                        "SOURCE_IDENTITY_MISMATCH",
                        "selected raw source or label is outside its receipt",
                    ) from error
                values_list: list[tuple[str, int, float]] = []
                for selector, column in selected_columns:
                    value = float(column[position])
                    if not math.isfinite(value):
                        raise StitchOperationRefused(
                            "INVALID_METADATA_VALUE",
                            f"metadata column {selector.name!r} contains a non-finite value",
                        )
                    values_list.append(
                        (selector.name, selector.occurrence, value)
                    )
                contribution = StitchContribution(
                    label,
                    ordinal,
                    frame.source_frame_index,
                    tuple(sorted(values_list)),
                )
                charge_piece(
                    {
                        "label": contribution.label,
                        "file_ordinal": contribution.file_ordinal,
                        "source_frame_index": contribution.source_frame_index,
                        "values": contribution.values,
                    }
                )
                contributions.append(contribution)
    except AnalysisSourceLeaseRefused as error:
        raise StitchOperationRefused(error.code) from error
    return StitchInputManifestReceipt(
        str(root),
        source_relative,
        analysis.resolved_kind.value,
        analysis.resolved_entry,
        analysis.resolved_scan,
        analysis.source_spec_digest,
        source_options_json,
        analysis.primary_post_state,
        tuple(files),
        tuple(contributions),
        analysis.source_fingerprint,
        source.fingerprint,
        _REQUEST_FACTORY,
    )


@dataclass(frozen=True, slots=True)
class StitchOperationPlan:
    geometry: StitchGeometryReceipt
    backend: str = "multigeometry"
    mode: str = "1d"
    npt_1d: int = 1500
    npt_rad_2d: int = 1000
    npt_azim_2d: int = 360
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    unit: str = "q_A^-1"
    method: str = "full"
    monitor_selector: MetadataColumnSelector | None = None
    use_detector_mask: bool = True
    max_frame_bytes: int = 256 * 1024 * 1024
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.geometry) is not StitchGeometryReceipt:
            raise TypeError("stitch plan requires an exact geometry receipt")
        if self.backend != "multigeometry":
            raise ValueError("S1 supports only the multigeometry backend")
        if self.mode not in {"1d", "2d"}:
            raise ValueError("stitch mode must be '1d' or '2d'")
        for name in ("npt_1d", "npt_rad_2d", "npt_azim_2d"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= _MAX_AXIS_POINTS:
                raise ValueError(f"{name} must be a bounded positive exact integer")
        if self.npt_rad_2d * self.npt_azim_2d > _MAX_2D_POINTS:
            raise ValueError("2-D Stitch output exceeds the bounded artifact point limit")
        radial = _range(self.radial_range, "radial range", required=True)
        azimuth = _range(
            self.azimuth_range,
            "azimuth range",
            required=self.mode == "2d",
        )

        if self.mode == "1d" and azimuth is not None:
            raise ValueError("a 1-D Stitch plan cannot carry an azimuth range")
        if self.unit != "q_A^-1" or self.method != "full":
            raise ValueError("S1 requires unit='q_A^-1' and notebook method='full'")
        if (
            self.monitor_selector is not None
            and type(self.monitor_selector) is not MetadataColumnSelector
        ):
            raise TypeError("monitor selector must be exact MetadataColumnSelector")
        if type(self.use_detector_mask) is not bool:
            raise TypeError("use_detector_mask must be an exact bool")
        if (
            type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= 4 * 1024 * 1024 * 1024
        ):
            raise ValueError("max_frame_bytes must be a bounded positive exact integer")
        object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "azimuth_range", azimuth)
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(ModuleKind.STITCH, self._canonical_value()),
        )

    def _canonical_value(self) -> tuple[object, ...]:
        selector = self.monitor_selector
        return (
            "stitch-operation-plan-v1",
            self.geometry.fingerprint,
            self.backend,
            self.mode,
            self.npt_1d,
            self.npt_rad_2d,
            self.npt_azim_2d,
            self.radial_range,
            self.azimuth_range,
            self.unit,
            self.method,
            None if selector is None else (selector.name, selector.occurrence),
            self.use_detector_mask,
            "streaming-multigeometry-v1",
            self.max_frame_bytes,
        )


@dataclass(frozen=True, slots=True)
class XuStitchOperationPlan:
    calibration: XuStitchCalibrationReceipt
    effective_geometry: XuStitchEffectiveGeometryProjection
    q_min_A_inverse: float
    q_max_A_inverse: float
    npt_1d: int = 1500
    monitor_selector: MetadataColumnSelector | None = None
    max_frame_bytes: int = 256 * 1024 * 1024
    backend: str = field(init=False, default="xu_hist")
    mode: str = field(init=False, default="1d")
    unit: str = field(init=False, default="q_A^-1")
    method: str = field(init=False, default="numpy_histogram_center_v1")
    use_detector_mask: bool = field(init=False, default=True)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.calibration) is not XuStitchCalibrationReceipt
            or type(self.effective_geometry)
            is not XuStitchEffectiveGeometryProjection
            or self.effective_geometry.asset_semantic_fingerprint
            != self.calibration.semantic_fingerprint
            or type(self.q_min_A_inverse) is not float
            or type(self.q_max_A_inverse) is not float
            or not math.isfinite(self.q_min_A_inverse)
            or not math.isfinite(self.q_max_A_inverse)
            or self.q_min_A_inverse >= self.q_max_A_inverse
            or type(self.npt_1d) is not int
            or not 1 <= self.npt_1d <= _MAX_AXIS_POINTS
            or (
                self.monitor_selector is not None
                and type(self.monitor_selector) is not MetadataColumnSelector
            )
            or type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= 4 * 1024 * 1024 * 1024
        ):
            raise TypeError("XU Stitch operation plan is invalid")
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(
                ModuleKind.STITCH,
                (
                    "stitch-operation-plan-v2-xu-hist",
                    self.backend,
                    self.mode,
                    self.unit,
                    self.method,
                    (self.q_min_A_inverse, self.q_max_A_inverse),
                    self.npt_1d,
                    None
                    if self.monitor_selector is None
                    else (
                        self.monitor_selector.name,
                        self.monitor_selector.occurrence,
                    ),
                    self.use_detector_mask,
                    self.max_frame_bytes,
                    self.calibration.fingerprint,
                    self.effective_geometry.fingerprint,
                ),
            ),
        )

    @property
    def radial_range(self) -> tuple[float, float]:
        return (self.q_min_A_inverse, self.q_max_A_inverse)


def _required_stitch_selectors(
    plan: StitchOperationPlan | XuStitchOperationPlan,
    table_names: tuple[str, ...] = (),
) -> tuple[MetadataColumnSelector, ...]:
    if type(plan) is StitchOperationPlan:
        requested = tuple(
            MetadataColumnSelector(name, 0)
            for _position, name in plan.geometry.request.source_motors
        )
    elif type(plan) is XuStitchOperationPlan:
        acquisition = plan.calibration.projection.value["acquisition"]
        requested = tuple(
            MetadataColumnSelector(str(name), 0)
            for name in acquisition["source_motors"].values()
        )
        energy_name = str(acquisition["source_energy_selector"])
        energy_count = table_names.count(energy_name)
        if energy_count > 1:
            raise StitchOperationRefused(
                "AMBIGUOUS_METADATA_SELECTOR",
                f"metadata column {energy_name!r} must occur at most once",
            )
        if energy_count == 1:
            requested += (MetadataColumnSelector(energy_name, 0),)
    else:
        raise TypeError("stitch selectors require an exact Stitch plan")
    requested += (() if plan.monitor_selector is None else (plan.monitor_selector,))
    return tuple(
        sorted(
            {
                (selector.name, selector.occurrence): selector
                for selector in requested
            }.values(),
            key=lambda selector: (selector.name, selector.occurrence),
        )
    )


def _xu_energy_selector(
    plan: XuStitchOperationPlan,
    table_names: tuple[str, ...],
) -> MetadataColumnSelector | None:
    name = str(
        plan.calibration.projection.value["acquisition"][
            "source_energy_selector"
        ]
    )
    count = table_names.count(name)
    if count > 1:
        raise StitchOperationRefused(
            "AMBIGUOUS_METADATA_SELECTOR",
            f"metadata column {name!r} must occur at most once",
        )
    return None if count == 0 else MetadataColumnSelector(name, 0)


def _xu_observations(
    plan: XuStitchOperationPlan,
    manifest: StitchInputManifestReceipt,
    energy_selector: MetadataColumnSelector | None,
) -> XuStitchScienceObservations:
    asset = plan.calibration.projection.value
    acquisition = asset["acquisition"]
    validation = asset["validation"]
    del_key = (str(acquisition["source_motors"]["del"]), 0)
    nu_key = (str(acquisition["source_motors"]["nu"]), 0)
    monitor_key = (
        None
        if plan.monitor_selector is None
        else (plan.monitor_selector.name, plan.monitor_selector.occurrence)
    )
    energy_key = (
        None
        if energy_selector is None
        else (energy_selector.name, energy_selector.occurrence)
    )
    del_values: list[float] = []
    nu_values: list[float] = []
    energy_values: list[float] = []
    extrapolation_count = 0
    validated = validation["validated_scan_domain_source_deg"]
    control = validation["control_domain_source_deg"]
    expected_energy = float(acquisition["energy_eV"])
    for contribution in manifest.contributions:
        values = {
            (name, occurrence): value
            for name, occurrence, value in contribution.values
        }
        try:
            del_value = float(values[del_key])
            nu_value = float(values[nu_key])
        except KeyError as error:
            raise StitchOperationRefused(
                "UNBOUND_METADATA_SELECTOR",
                "XU angle metadata is absent from the exact input manifest",
            ) from error
        if not (
            float(validated["del"][0])
            <= del_value
            <= float(validated["del"][1])
            and float(validated["nu"][0])
            <= nu_value
            <= float(validated["nu"][1])
        ):
            raise StitchOperationRefused(
                "XU_CALIBRATION_DOMAIN_EXCEEDED",
                "selected source angles exceed the validated XU domain",
            )
        if not (
            float(control["del"][0])
            <= del_value
            <= float(control["del"][1])
            and float(control["nu"][0])
            <= nu_value
            <= float(control["nu"][1])
        ):
            extrapolation_count += 1
        if monitor_key is not None and values[monitor_key] <= 0:
            raise StitchOperationRefused(
                "INVALID_MONITOR_VALUE",
                "Stitch monitor values must be finite and strictly positive",
            )
        if energy_key is not None:
            energy = float(values[energy_key])
            if (
                energy <= 0
                or abs(energy - expected_energy) > 0.001 * expected_energy
            ):
                raise StitchOperationRefused(
                    "XU_SOURCE_ENERGY_CONFLICT",
                    "source energy conflicts with the calibrated XU energy",
                )
            energy_values.append(energy)
        del_values.append(del_value)
        nu_values.append(nu_value)
    return XuStitchScienceObservations(
        (float(min(del_values)), float(max(del_values))),
        (float(min(nu_values)), float(max(nu_values))),
        (
            None
            if not energy_values
            else (float(min(energy_values)), float(max(energy_values)))
        ),
        extrapolation_count,
        (
            ()
            if extrapolation_count == 0
            else ("XU_CALIBRATION_EXTRAPOLATION_WITHIN_VALIDATED_SCAN",)
        ),
    )


def _engine_version() -> str:
    try:
        return importlib.metadata.version("pyFAI")
    except importlib.metadata.PackageNotFoundError:
        try:
            import pyFAI

            return str(pyFAI.version)
        except (ImportError, AttributeError):
            return "unavailable"


def _effective_poni_calibration(
    geometry: StitchGeometryReceipt,
    raw: bytes,
) -> dict[str, object] | None:
    """Return the strict effective PONI projection bound to one receipt."""

    if geometry.request.kind is not StitchGeometryKind.PONI:
        return None
    from xrd_tools.integrate.calibration import (
        detector_calibration_projection,
        load_detector_calibration,
    )

    calibration = load_detector_calibration(
        geometry.resolved_path,
        data=raw,
    )
    projection = detector_calibration_projection(calibration)
    # Force a finite JSON-native value at the provenance boundary.
    return json.loads(json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _provenance(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan: StitchOperationPlan,
    manifest: StitchInputManifestReceipt,
    geometry_relative_path: str,
) -> dict[str, object]:
    geometry = plan.geometry
    geometry_request = geometry.request
    selector = plan.monitor_selector
    geometry_projection: dict[str, object] = {
        "kind": geometry_request.kind.value,
        "relative_path": geometry_relative_path,
        "byte_count": geometry.byte_count,
        "sha256": geometry.sha256,
        "receipt_fingerprint": geometry.fingerprint,
        "source_motors": [list(item) for item in geometry_request.source_motors],
        "reference_motor_positions": [
            [name, value]
            for name, value in geometry_request.reference_motor_positions
        ],
        "image_orientation": {
            "rotation": geometry_request.image_rotation,
            "flip_vertical": False,
            "flip_horizontal": False,
            "transpose": False,
        },
        "base_preset": geometry_request.base_preset,
    }
    effective_calibration = _effective_poni_calibration(
        geometry,
        geometry.content,
    )
    if effective_calibration is not None:
        geometry_projection["effective_calibration"] = effective_calibration
    return {
        "schema_version": "stitch-operation-v1",
        "kind": "stitch",
        "source": {
            "source_fingerprint": source.analysis.source_fingerprint,
            "module_source_fingerprint": source.fingerprint,
            "metadata_table_fingerprint": source.table_fingerprint,
            "selected_labels": list(source.selected_labels),
            "input_manifest": manifest.to_provenance(),
        },
        "geometry": geometry_projection,
        "plan": {
            "backend": plan.backend,
            "mode": plan.mode,
            "npt_1d": plan.npt_1d,
            "npt_rad_2d": plan.npt_rad_2d,
            "npt_azim_2d": plan.npt_azim_2d,
            "radial_range": list(plan.radial_range),
            "azimuth_range": (
                None if plan.azimuth_range is None else list(plan.azimuth_range)
            ),
            "unit": plan.unit,
            "method": plan.method,
            "monitor_selector": (
                None
                if selector is None
                else {"name": selector.name, "occurrence": selector.occurrence}
            ),
            "use_detector_mask": plan.use_detector_mask,
            "streaming_multigeometry": True,
            "max_frame_bytes": plan.max_frame_bytes,
            "plan_fingerprint": plan.fingerprint,
        },
        "engine": {"pyfai": _engine_version(), "numpy": np.__version__},
        "output": {"kind": output.kind.value},
        "holds": [
            "2d-orientation-parity",
            "gi-and-custom-corrections",
        ],
    }


def _xu_provenance(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan: XuStitchOperationPlan,
    manifest: StitchInputManifestReceipt,
    observations: XuStitchScienceObservations,
    energy_selector: MetadataColumnSelector | None,
) -> dict[str, object]:
    receipt = plan.calibration
    asset = receipt.projection.value
    requirements = XuRuntimeRequirements()
    monitor = plan.monitor_selector
    effective = plan.effective_geometry.to_provenance()
    effective["asset_semantic_fingerprint"] = receipt.semantic_fingerprint
    effective["acquisition"] = _portable_json_value(asset["acquisition"])
    effective["xrayutilities"] = _portable_json_value(asset["xrayutilities"])
    observed = observations.to_provenance()
    observed.update(
        {
            "selected_frame_count": len(source.selected_labels),
            "source_energy_selector": (
                None
                if energy_selector is None
                else {
                    "name": energy_selector.name,
                    "occurrence": energy_selector.occurrence,
                }
            ),
            "monitor_selector": (
                None
                if monitor is None
                else {"name": monitor.name, "occurrence": monitor.occurrence}
            ),
        }
    )
    return {
        "schema_version": "stitch-operation-v2-xu-intent",
        "kind": "stitch",
        "backend": "xu_hist",
        "source": {
            "source_fingerprint": source.analysis.source_fingerprint,
            "module_source_fingerprint": source.fingerprint,
            "metadata_table_fingerprint": source.table_fingerprint,
            "selected_labels": list(source.selected_labels),
            "input_manifest": manifest.to_provenance(),
        },
        "asset": {
            "lexical_relative_path": receipt.lexical_relative_path,
            "resolved_relative_path": receipt.resolved_relative_path,
            "byte_count": receipt.byte_count,
            "raw_sha256": receipt.raw_sha256,
            "semantic_fingerprint": receipt.semantic_fingerprint,
            "receipt_fingerprint": receipt.fingerprint,
        },
        "effective_geometry": effective,
        "detector": _portable_json_value(asset["detector"]),
        "corrections": _portable_json_value(asset["corrections"]),
        "plan": {
            "backend": plan.backend,
            "mode": plan.mode,
            "unit": plan.unit,
            "method": plan.method,
            "radial_range": list(plan.radial_range),
            "npt_1d": plan.npt_1d,
            "monitor_selector": (
                None
                if monitor is None
                else {"name": monitor.name, "occurrence": monitor.occurrence}
            ),
            "use_detector_mask": plan.use_detector_mask,
            "max_frame_bytes": plan.max_frame_bytes,
            "asset_receipt_fingerprint": receipt.fingerprint,
            "effective_geometry_fingerprint": plan.effective_geometry.fingerprint,
            "plan_fingerprint": plan.fingerprint,
        },
        "observations": observed,
        "runtime_requirements": {
            "xrayutilities_distribution_version": requirements.distribution_version,
            "xrayutilities_module_version": requirements.module_version,
            "numpy_version": requirements.numpy_version,
            "config_epsilon": requirements.config_epsilon,
            "config_digits": requirements.config_digits,
            "nthreads_effective": 1,
            "python_implementation": requirements.python_implementation,
            "python_min_version": list(requirements.python_min_version),
            "platform_systems": list(requirements.platform_systems),
            "runtime_policy": asset["xrayutilities"]["runtime_policy"],
        },
        "output": {
            "target": output.target,
            "kind": output.kind.value,
            "overwrite": output.overwrite.value,
            "output_fingerprint": output.fingerprint,
        },
        "holds": [
            "xu-2d-and-chi",
            "gi-stitch-on-xu",
            "explicit-ub-and-sample-placement",
            "operator-configured-sample-circles",
            "generic-xu-diffractometers-and-detectors",
            "external-or-combined-masks",
            "sensor-parallax-xu-geometry",
            "new-corrections-and-variance",
            "mixed-energy-grouped-xu-stitch",
            "historical-json-migration",
            "unvalidated-platforms-and-live-gates",
        ],
    }


@dataclass(eq=False, frozen=True, slots=True)
class StitchOperationRequest:
    module: ModuleOperationRequest
    plan: StitchOperationPlan | XuStitchOperationPlan
    manifest: StitchInputManifestReceipt
    provenance_json: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _REQUEST_FACTORY
            or type(self.module) is not ModuleOperationRequest
            or self.module.kind is not ModuleKind.STITCH
            or type(self.plan) not in {StitchOperationPlan, XuStitchOperationPlan}
            or type(self.manifest) is not StitchInputManifestReceipt
            or type(self.provenance_json) is not str
            or self.module.plan_fingerprint != self.plan.fingerprint
            or self.manifest.module_source_fingerprint
            != self.module.source.fingerprint
            or tuple(item.label for item in self.manifest.contributions)
            != self.module.source.selected_labels
        ):
            raise TypeError("stitch operation request is invalid")
        table = _fresh_table(self.module.source)
        table_names = tuple(column.name for column in table.columns)
        required = _required_stitch_selectors(self.plan, table_names)
        expected_manifest = _capture_input_manifest(
            self.module.source,
            table,
            self.manifest.project_root,
            required,
        )
        if expected_manifest.fingerprint != self.manifest.fingerprint:
            raise ValueError(
                "stitch input manifest is not the exact bound source projection"
            )
        try:
            provenance = json.loads(self.provenance_json)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("stitch operation provenance is invalid") from error
        if (
            type(provenance) is not dict
            or module_provenance_digest(ModuleKind.STITCH, provenance)
            != self.module.provenance_digest
        ):
            raise ValueError("stitch provenance does not match module request")
        if type(self.plan) is StitchOperationPlan:
            geometry_relative = _relative_locator(
                self.plan.geometry.resolved_path,
                Path(self.manifest.project_root),
                "geometry",
            )
            expected = _provenance(
                self.module.source,
                self.module.output,
                self.plan,
                self.manifest,
                geometry_relative,
            )
        else:
            if (
                self.module._xu_stitch_v2_bound is not True
                or self.plan.calibration.project_root
                != self.manifest.project_root
            ):
                raise ValueError("XU Stitch request lacks exact Project authority")
            energy_selector = _xu_energy_selector(self.plan, table_names)
            observations = _xu_observations(
                self.plan,
                self.manifest,
                energy_selector,
            )
            expected = _xu_provenance(
                self.module.source,
                self.module.output,
                self.plan,
                self.manifest,
                observations,
                energy_selector,
            )
        if provenance != expected:
            raise ValueError("stitch provenance is not the exact request projection")

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(self.provenance_json)

    def __copy__(self):
        raise TypeError("stitch operation request is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("stitch operation request is not copyable")


def _fresh_table(source: ModuleSourceReceipt):
    table = run_metadata_table(MetadataTablePlan(source.analysis.source_spec))
    if (
        table.disposition is not AnalysisDisposition.COMPLETED
        or table.receipt != source.analysis
        or table.table_fingerprint != source.table_fingerprint
        or table.labels != source.analysis.labels
    ):
        raise StitchOperationRefused(
            "SOURCE_IDENTITY_MISMATCH", "source metadata no longer matches its receipt"
        )
    return table


def prepare_stitch_operation(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan: StitchOperationPlan | XuStitchOperationPlan,
    *,
    project_root: str | Path,
) -> StitchOperationRequest:
    """Bind exact source, geometry, plan, provenance, and output intent."""

    if (
        type(source) is not ModuleSourceReceipt
        or source.kind is not ModuleKind.STITCH
        or type(output) is not ModuleOutputRequest
        or type(plan) not in {StitchOperationPlan, XuStitchOperationPlan}
    ):
        raise TypeError("stitch preparation requires exact Stitch module values")
    if source.analysis.resolved_kind not in {
        SourceKind.SPEC,
        SourceKind.TIFF_SERIES,
    }:
        raise StitchOperationRefused(
            "SOURCE_KIND_UNSUPPORTED",
            "S1 Stitch accepts only an extensionless SPEC scan or TIFF series",
        )
    expected_kind = (
        AnalysisArtifactKind.STITCH_1D
        if type(plan) is XuStitchOperationPlan or plan.mode == "1d"
        else AnalysisArtifactKind.STITCH_2D
    )
    if output.kind is not expected_kind:
        raise StitchOperationRefused(
            "OUTPUT_KIND_MISMATCH", "Stitch mode and output artifact kind differ"
        )
    table = _fresh_table(source)
    names = tuple(column.name for column in table.columns)
    required = _required_stitch_selectors(plan, names)
    for selector in required:
        if selector.occurrence != 0 or names.count(selector.name) != 1:
            raise StitchOperationRefused(
                "AMBIGUOUS_METADATA_SELECTOR",
                f"metadata column {selector.name!r} must occur exactly once",
            )
        if selector not in source.resolved_selectors:
            raise StitchOperationRefused(
                "UNBOUND_METADATA_SELECTOR",
                f"metadata column {selector.name!r} is not bound by the source receipt",
            )
    options = dict(source.analysis.source_spec.options)
    read_options = options.get("read_image_kwargs", {})
    if type(read_options) is not dict:
        read_options = dict(read_options)
    if read_options.get("rotation", 0) != 0:
        raise StitchOperationRefused(
            "SOURCE_ORIENTATION_CONFLICT",
            "source rotation must be zero; Stitch owns the recorded image orientation",
        )
    manifest = _capture_input_manifest(source, table, project_root, required)
    if type(plan) is StitchOperationPlan:
        _revalidate_geometry(plan.geometry)
        if plan.monitor_selector is not None:
            monitor_key = (
                plan.monitor_selector.name,
                plan.monitor_selector.occurrence,
            )
            for contribution in manifest.contributions:
                values = {
                    (name, occurrence): value
                    for name, occurrence, value in contribution.values
                }
                if values[monitor_key] <= 0:
                    raise StitchOperationRefused(
                        "INVALID_MONITOR_VALUE",
                        "Stitch monitor values must be finite and strictly positive",
                    )
        geometry_relative = _relative_locator(
            plan.geometry.resolved_path,
            Path(manifest.project_root),
            "geometry",
        )
        provenance = _provenance(
            source,
            output,
            plan,
            manifest,
            geometry_relative,
        )
        module = ModuleOperationRequest(
            source,
            output,
            plan.fingerprint,
            module_provenance_digest(ModuleKind.STITCH, provenance),
        )
    else:
        if plan.calibration.project_root != manifest.project_root:
            raise StitchOperationRefused(
                "XU_CALIBRATION_PROJECT_MISMATCH",
                "XU calibration and source must share the exact Project root",
            )
        energy_selector = _xu_energy_selector(plan, names)
        observations = _xu_observations(plan, manifest, energy_selector)
        try:
            revalidate_xu_stitch_calibration(plan.calibration)
            runtime_owner = xu_runtime_session()
            with runtime_owner as session:
                effective = resolve_xu_stitch_effective_geometry(
                    plan.calibration,
                    session,
                ).projection
        except (
            XuRuntimeUnsupported,
            XuStitchCalibrationRefused,
            XuStitchScienceRefused,
        ) as error:
            raise StitchOperationRefused(error.code) from error
        if (
            runtime_owner.execution_record is None
            or effective.fingerprint != plan.effective_geometry.fingerprint
            or effective.to_provenance()
            != plan.effective_geometry.to_provenance()
        ):
            raise StitchOperationRefused(
                "XU_EFFECTIVE_GEOMETRY_MISMATCH",
                "effective XU geometry differs from the frozen operation plan",
            )
        provenance = _xu_provenance(
            source,
            output,
            plan,
            manifest,
            observations,
            energy_selector,
        )
        module = xu_stitch_module_request(
            source,
            output,
            plan.fingerprint,
            module_provenance_digest(ModuleKind.STITCH, provenance),
        )
    canonical = module_artifact_request(module, provenance).provenance_json
    return StitchOperationRequest(
        module,
        plan,
        manifest,
        canonical,
        _REQUEST_FACTORY,
    )


class _OrientedSource(BaseFrameSource):
    def __init__(
        self,
        source,
        orientation: ImageOrientation,
        max_frame_bytes: int,
    ):
        super().__init__(
            name=getattr(source, "name", type(source).__name__),
            frame_indices=source.frame_indices,
            spec=getattr(source, "spec", None),
            capabilities=source.capabilities,
        )
        self._source = source
        self._orientation = orientation
        self._max_frame_bytes = max_frame_bytes

    @property
    def motors(self):
        return getattr(self._source, "motors", {})

    def load_frame(self, index: int) -> np.ndarray:
        image = np.asarray(self._source.load_frame(index))
        if image.nbytes > self._max_frame_bytes:
            raise MemoryError(
                f"Stitch frame requires {image.nbytes} bytes, exceeding "
                f"max_frame_bytes={self._max_frame_bytes}"
            )
        # run_stitch consumes float64 frames.  Convert here exactly once and
        # enforce the envelope against the representation used by pyFAI, not
        # merely the smaller on-disk integer dtype.
        oriented = np.asarray(
            self._orientation.apply(image),
            dtype=np.float64,
            order="C",
        )
        if oriented.nbytes > self._max_frame_bytes:
            raise MemoryError(
                f"float64 oriented Stitch frame requires {oriented.nbytes} bytes, exceeding "
                f"max_frame_bytes={self._max_frame_bytes}"
            )
        return oriented

    def metadata_for(self, index: int) -> Mapping[str, object]:
        return self._source.metadata_for(index)

    def frame_for(self, index: int):
        return self._source.frame_for(index)


def _runtime_geometry(receipt: StitchGeometryReceipt, raw: bytes):
    request = receipt.request
    orientation = ImageOrientation(rotation=request.image_rotation)
    if request.kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON:
        base = Diffractometer.psic()
        diffractometer = Diffractometer.from_pyfai_goniometer(
            _json_mapping(raw),
            source_motors=dict(request.source_motors),
            base=base,
            image_orientation=orientation,
        )
    else:
        from xrd_tools.integrate.calibration import load_detector_calibration

        calibration = load_detector_calibration(receipt.resolved_path, data=raw)
        calibration = replace(calibration, image_orientation=orientation)
        motors = dict(request.source_motors)
        references = dict(request.reference_motor_positions)
        base = Diffractometer.psic(del_=motors["del"], nu=motors["nu"])
        base = replace(
            base,
            rot1=AngleMapping(
                source_motor=motors["nu"],
                offset=-references["nu"],
            ),
            rot2=AngleMapping(
                source_motor=motors["del"],
                offset=-references["del"],
            ),
        )
        diffractometer = replace(base, calibration=calibration)
    return diffractometer, orientation


def _detector_mask(diffractometer) -> np.ndarray | None:
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator

    detector = detector_calibration_to_integrator(diffractometer.calibration).detector
    if detector is None or not hasattr(detector, "calc_mask"):
        return None
    value = detector.calc_mask()
    if value is None:
        return None
    mask = np.ascontiguousarray(np.asarray(value, dtype=bool))
    mask.setflags(write=False)
    return mask


def _runtime_plan(plan: StitchOperationPlan, diffractometer) -> StitchPlan:
    return StitchPlan(
        diffractometer=diffractometer,
        backend=plan.backend,
        mode=plan.mode,
        npt_1d=plan.npt_1d,
        npt_rad_2d=plan.npt_rad_2d,
        npt_azim_2d=plan.npt_azim_2d,
        unit=plan.unit,
        method=plan.method,
        radial_range=plan.radial_range,
        azimuth_range=plan.azimuth_range,
        monitor_key=(
            None if plan.monitor_selector is None else plan.monitor_selector.name
        ),
        mask=_detector_mask(diffractometer) if plan.use_detector_mask else None,
        max_eager_bytes=None,
        streaming_multigeometry=True,
    )


@dataclass(eq=False, frozen=True, slots=True)
class StitchOperationResult:
    request: StitchOperationRequest
    terminal: ModuleTerminalResult
    payload: AnalysisArtifactPayload | None = None

    def __post_init__(self) -> None:
        committed = self.terminal.disposition is ModuleDisposition.COMMITTED
        if (
            type(self.request) is not StitchOperationRequest
            or type(self.terminal) is not ModuleTerminalResult
            or self.terminal.request is not self.request.module
            or committed != (type(self.payload) is AnalysisArtifactPayload)
            or (
                committed
                and self.payload.inspection.result_fingerprint
                != self.terminal.commit.result_fingerprint
            )
            or (
                committed
                and not self.payload.inspection.has_stitch_diagnostics
            )
        ):
            raise TypeError("stitch operation result is invalid")


def _protected_stitch_inputs(request: StitchOperationRequest) -> tuple[Path, ...]:
    root = Path(request.manifest.project_root)
    inputs = [root / request.manifest.source_relative_path]
    inputs.extend(root / item.relative_path for item in request.manifest.files)
    if type(request.plan) is XuStitchOperationPlan:
        geometry = request.plan.calibration
        inputs.extend((
            root / geometry.lexical_relative_path,
            root / geometry.resolved_relative_path,
        ))
    else:
        geometry = request.plan.geometry
        inputs.extend((Path(geometry.lexical_path), Path(geometry.resolved_path)))
    return tuple(dict.fromkeys(inputs))


class StitchOperationExecution:
    """One-shot execution owner retained across retryable artifact cleanup."""

    def __init__(self, request: StitchOperationRequest, *, coordinator=None):
        if type(request) is not StitchOperationRequest:
            raise TypeError("Stitch execution requires exact StitchOperationRequest")
        self.request = request
        self.coordinator = coordinator
        self._state = "new"
        self._output: ModuleArtifactOutput | None = None
        self._result: StitchOperationResult | None = None
        self._verification_terminal: ModuleTerminalResult | None = None
        self._execution_attestation_json: str | None = None
        self._execution_attestation_digest: str | None = None
        self._revision = 0
        self._progress_callback: Callable[[ModuleProgress], object] | None = None

    def __copy__(self):
        raise TypeError("stitch operation execution is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("stitch operation execution is not copyable")

    @property
    def output_snapshot(self):
        return None if self._output is None else self._output.snapshot

    def _terminal(
        self,
        disposition: ModuleDisposition,
        code: str,
        diagnostic: str = "",
    ) -> StitchOperationResult:
        value = StitchOperationResult(
            self.request,
            ModuleTerminalResult(
                self.request.module,
                disposition,
                code,
                diagnostic=diagnostic,
            ),
        )
        self._result = value
        self._state = "done"
        return value

    def _emit(self, stage: str, completed: int, total: int) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        self._revision += 1
        value = ModuleProgress(
            self.request.module,
            self._revision,
            stage,
            completed,
            total,
        )
        try:
            callback(value)
        except Exception:
            pass

    def _strict_result(
        self,
        terminal: ModuleTerminalResult,
        *,
        total: int,
    ) -> StitchOperationResult:
        if terminal.disposition is not ModuleDisposition.COMMITTED:
            value = StitchOperationResult(self.request, terminal)
            self._result = value
            self._state = "done"
            return value
        if self._output is None or self._output.snapshot.receipt is None:
            raise StitchOperationVerificationError(
                self, "committed Stitch output has no exact artifact receipt"
            )
        try:
            payload = read_analysis_artifact(
                self.request.module.output.target,
                expected_receipt=self._output.snapshot.receipt,
            )
        except BaseException as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self, f"strict Stitch reload failed: {_failure_diagnostic(error)}"
            ) from error
        if payload.inspection.result_fingerprint != terminal.commit.result_fingerprint:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self, "strict Stitch reload result fingerprint changed"
            )
        if (
            not payload.inspection.has_stitch_diagnostics
            or payload.coverage is None
            or payload.normalization is None
        ):
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self, "strict Stitch reload omitted coverage or normalization"
            )
        try:
            persisted = json.loads(payload.provenance_json)
            source_provenance = persisted["source"]
            manifest_provenance = source_provenance["input_manifest"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self, "strict Stitch provenance is malformed"
            ) from error
        if (
            payload.provenance_json != self.request.provenance_json
            or manifest_provenance != self.request.manifest.to_provenance()
            or source_provenance.get("selected_labels")
            != list(self.request.module.source.selected_labels)
        ):
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self, "strict Stitch provenance no longer matches its request"
            )
        if type(self.request.plan) is XuStitchOperationPlan:
            if (
                payload.schema_version != ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL
                or payload.execution_attestation_json
                != self._execution_attestation_json
                or payload.execution_attestation_digest
                != self._execution_attestation_digest
                or terminal.commit.execution_attestation_digest
                != self._execution_attestation_digest
            ):
                self._verification_terminal = terminal
                self._state = "verification_failed"
                raise StitchOperationVerificationError(
                    self,
                    "strict XU Stitch attestation no longer matches execution",
                )
        elif (
            payload.schema_version != ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL
            or payload.execution_attestation_json is not None
            or payload.execution_attestation_digest is not None
            or terminal.commit.execution_attestation_digest is not None
        ):
            self._verification_terminal = terminal
            self._state = "verification_failed"
            raise StitchOperationVerificationError(
                self,
                "strict MultiGeometry Stitch unexpectedly changed artifact version",
            )
        self._emit("reload", total, total)
        value = StitchOperationResult(self.request, terminal, payload)
        self._result = value
        self._verification_terminal = terminal
        self._state = "done"
        return value

    def _run_xu(
        self,
        *,
        cancel_token: threading.Event | None,
    ) -> StitchOperationResult:
        plan = self.request.plan
        if type(plan) is not XuStitchOperationPlan:
            raise TypeError("XU execution requires an exact XU Stitch plan")
        labels = self.request.module.source.selected_labels
        total = len(labels) + 4
        self._emit("calibration", 0, total)
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        provenance = self.request.provenance
        energy_projection = provenance["observations"]["source_energy_selector"]
        energy_key = (
            None
            if energy_projection is None
            else energy_projection["name"]
        )
        try:
            revalidate_xu_stitch_calibration(plan.calibration)
            self._emit("calibration", 1, total)
            with requalified_analysis_source(
                self.request.module.source.analysis,
                cancel_token=cancel_token,
            ) as source:

                def science_progress(done: int, _inner_total: int) -> None:
                    self._emit("science", 1 + done, total)

                scientific = run_xu_hist_stitch_1d(
                    plan.calibration,
                    source,
                    frame_indices=labels,
                    q_min_A_inverse=plan.q_min_A_inverse,
                    q_max_A_inverse=plan.q_max_A_inverse,
                    npt=plan.npt_1d,
                    monitor_key=(
                        None
                        if plan.monitor_selector is None
                        else plan.monitor_selector.name
                    ),
                    source_energy_key=energy_key,
                    max_frame_bytes=plan.max_frame_bytes,
                    cancel_token=cancel_token,
                    progress_callback=science_progress,
                )
            revalidate_xu_stitch_calibration(plan.calibration)
        except XuStitchCancelled:
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        except AnalysisSourceLeaseRefused as error:
            if error.code == "CANCELLED":
                return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except (
            XuRuntimeUnsupported,
            XuStitchCalibrationRefused,
            XuStitchScienceRefused,
        ) as error:
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "SCIENCE_FAILED",
                _failure_diagnostic(error),
            )
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        expected_observations = {
            name: provenance["observations"][name]
            for name in scientific.observations.to_provenance()
        }
        if (
            scientific.effective_geometry.fingerprint
            != plan.effective_geometry.fingerprint
            or scientific.effective_geometry.to_provenance()
            != plan.effective_geometry.to_provenance()
            or scientific.observations.to_provenance()
            != expected_observations
            or scientific.selected_frame_count != len(labels)
            or scientific.release_check_frame_count != len(labels)
        ):
            return self._terminal(
                ModuleDisposition.REFUSED,
                "XU_EXECUTION_INTENT_MISMATCH",
            )
        payload = scientific.payload
        diagnostics = scientific.diagnostics
        try:
            result_projection = project_analysis_artifact_result(
                kind=AnalysisArtifactKind.STITCH_1D,
                axes=(("q", payload.radial),),
                axis_units=(("q", payload.unit),),
                intensity=payload.intensity,
                sigma=payload.sigma,
                coverage=diagnostics.coverage,
                normalization=diagnostics.normalization,
            )
        except AnalysisArtifactProjectionInvalid:
            return self._terminal(
                ModuleDisposition.FAILED,
                "STITCH_RESULT_STORAGE_PROJECTION_INVALID",
            )
        attestation = {
            "schema_version": "analysis-execution-attestation-v1",
            "module_request_fingerprint": self.request.module.fingerprint,
            "result_projection_policy": result_projection.policy,
            "result_fingerprint": result_projection.result_fingerprint,
            "selected_frame_count": scientific.selected_frame_count,
            "release_check_frame_count": scientific.release_check_frame_count,
            "release_check_passed": True,
            "q_root_policy": "shared_ultimate_ndarray_root_weakref_v1",
            "xu_runtime": scientific.runtime.to_attestation(),
        }
        attestation_digest = analysis_execution_attestation_digest(
            self.request.module.output.kind,
            attestation,
            request_fingerprint=self.request.module.fingerprint,
        )
        try:
            bound = module_artifact_request(
                self.request.module,
                provenance,
                execution_attestation=attestation,
                execution_attestation_digest=attestation_digest,
            )
            self._execution_attestation_json = bound.execution_attestation_json
            self._execution_attestation_digest = attestation_digest
            self._emit("projection", total - 2, total)
            self._output = admit_module_artifact(
                self.request.module,
                provenance,
                execution_attestation=attestation,
                execution_attestation_digest=attestation_digest,
                cancel_token=cancel_token,
                coordinator=self.coordinator,
                protected_inputs=_protected_stitch_inputs(self.request),
            )
        except ModuleArtifactRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        except FileExistsError:
            return self._terminal(ModuleDisposition.REFUSED, "OUTPUT_EXISTS")
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "OUTPUT_ADMISSION_FAILED",
                _failure_diagnostic(error),
            )

        def require_current_effective_geometry() -> None:
            try:
                revalidate_xu_stitch_calibration(plan.calibration)
                runtime_owner = xu_runtime_session()
                with runtime_owner as session:
                    current = resolve_xu_stitch_effective_geometry(
                        plan.calibration,
                        session,
                    ).projection
                if (
                    runtime_owner.execution_record is None
                    or current.fingerprint != plan.effective_geometry.fingerprint
                    or current.to_provenance()
                    != plan.effective_geometry.to_provenance()
                ):
                    raise ModuleArtifactRefused(
                        "XU_EFFECTIVE_GEOMETRY_MISMATCH"
                    )
            except ModuleArtifactRefused:
                raise
            except (
                XuRuntimeUnsupported,
                XuStitchCalibrationRefused,
                XuStitchScienceRefused,
            ) as error:
                raise ModuleArtifactRefused(error.code) from error

        def writer(entry) -> None:
            revalidate_xu_stitch_calibration(plan.calibration)
            write_stitched(
                entry,
                result_projection=result_projection,
                provenance=bound.provenance_json,
                bounded_artifact=True,
            )
            revalidate_xu_stitch_calibration(plan.calibration)

        try:
            terminal = self._output.publish(
                writer,
                cancel_token=cancel_token,
                prepublish_check=require_current_effective_geometry,
            )
        except AnalysisArtifactCleanupPending as error:
            self._state = "cleanup_pending"
            raise StitchOperationCleanupPending(self) from error
        self._emit("publish", total - 1, total)
        return self._strict_result(terminal, total=total)

    def run(
        self,
        *,
        cancel_token: threading.Event | None = None,
        progress_callback: Callable[[ModuleProgress], object] | None = None,
    ) -> StitchOperationResult:
        if self._state != "new":
            raise RuntimeError("Stitch execution is one-shot")
        if cancel_token is not None and type(cancel_token) is not threading.Event:
            raise TypeError("Stitch cancellation token must be exact threading.Event")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("Stitch progress callback must be callable")
        self._state = "running"
        self._progress_callback = progress_callback
        if type(self.request.plan) is XuStitchOperationPlan:
            return self._run_xu(cancel_token=cancel_token)
        labels = self.request.module.source.selected_labels
        science_total = len(labels) + 1
        total = science_total + 3
        self._emit("geometry", 0, total)
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        try:
            raw = _revalidate_geometry(self.request.plan.geometry)
            diffractometer, orientation = _runtime_geometry(
                self.request.plan.geometry, raw
            )
            effective_calibration = None
            if (
                self.request.plan.geometry.request.kind
                is StitchGeometryKind.PONI
            ):
                from xrd_tools.integrate.calibration import (
                    detector_calibration_projection,
                )

                effective_calibration = detector_calibration_projection(
                    diffractometer.calibration
                )
            expected_calibration = self.request.provenance["geometry"].get(
                "effective_calibration"
            )
            if (
                effective_calibration is None
            ) != (
                expected_calibration is None
            ) or (
                effective_calibration is not None
                and _canonical_json(effective_calibration)
                != _canonical_json(expected_calibration)
            ):
                raise StitchOperationRefused(
                    "GEOMETRY_EFFECTIVE_CALIBRATION_MISMATCH",
                    "runtime calibration differs from the receipt-bound projection",
                )
            runtime_plan = _runtime_plan(self.request.plan, diffractometer)
            self._emit("geometry", 1, total)
            with requalified_analysis_source(
                self.request.module.source.analysis,
                cancel_token=cancel_token,
            ) as source:
                oriented = _OrientedSource(
                    source,
                    orientation,
                    self.request.plan.max_frame_bytes,
                )

                def science_progress(done: int, _inner_total: int) -> None:
                    self._emit("science", 1 + done, total)

                scientific = run_stitch(
                    runtime_plan,
                    oriented,
                    frame_indices=labels,
                    progress_callback=science_progress,
                    cancel_token=cancel_token,
                    collect_frame_records=False,
                )
            _revalidate_geometry(self.request.plan.geometry)
        except StitchCancelled:
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        except AnalysisSourceLeaseRefused as error:
            if error.code == "CANCELLED":
                return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except StitchOperationRefused as error:
            return self._terminal(ModuleDisposition.REFUSED, error.code)
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "SCIENCE_FAILED",
                _failure_diagnostic(error),
            )
        if cancel_token is not None and cancel_token.is_set():
            return self._terminal(ModuleDisposition.CANCELLED, "CANCELLED")
        payload = scientific.payload
        if self.request.plan.mode == "1d":
            if type(payload) is not IntegrationResult1D:
                return self._terminal(ModuleDisposition.FAILED, "INVALID_SCIENCE_RESULT")
        elif type(payload) is not IntegrationResult2D:
            return self._terminal(ModuleDisposition.FAILED, "INVALID_SCIENCE_RESULT")
        diagnostics = scientific.auxiliary
        if type(diagnostics) is not StitchDiagnostics:
            return self._terminal(
                ModuleDisposition.FAILED,
                "INVALID_STITCH_DIAGNOSTICS",
            )
        try:
            if self.request.plan.mode == "1d":
                axes = (("q", payload.radial),)
                axis_units = (("q", payload.unit),)
                artifact_kind = AnalysisArtifactKind.STITCH_1D
            else:
                axes = (
                    ("q", payload.radial),
                    ("chi", payload.azimuthal),
                )
                axis_units = (
                    ("q", payload.unit),
                    ("chi", payload.azimuthal_unit),
                )
                artifact_kind = AnalysisArtifactKind.STITCH_2D
            result_projection = _project_analysis_artifact_result_v1_compat(
                kind=artifact_kind,
                axes=axes,
                axis_units=axis_units,
                intensity=payload.intensity,
                sigma=payload.sigma,
                coverage=diagnostics.coverage,
                normalization=diagnostics.normalization,
            )
        except AnalysisArtifactProjectionInvalid:
            return self._terminal(
                ModuleDisposition.FAILED,
                "STITCH_RESULT_STORAGE_PROJECTION_INVALID",
            )
        provenance = self.request.provenance
        bound = module_artifact_request(self.request.module, provenance)
        try:
            self._output = admit_module_artifact(
                self.request.module,
                provenance,
                cancel_token=cancel_token,
                coordinator=self.coordinator,
                protected_inputs=_protected_stitch_inputs(self.request),
            )
        except ModuleArtifactRefused as error:
            disposition = (
                ModuleDisposition.CANCELLED
                if error.code == "CANCELLED"
                else ModuleDisposition.REFUSED
            )
            return self._terminal(disposition, error.code)
        except FileExistsError:
            return self._terminal(ModuleDisposition.REFUSED, "OUTPUT_EXISTS")
        except BaseException as error:
            return self._terminal(
                ModuleDisposition.FAILED,
                "OUTPUT_ADMISSION_FAILED",
                _failure_diagnostic(error),
            )

        def writer(entry) -> None:
            _revalidate_geometry(self.request.plan.geometry)
            write_stitched(
                entry,
                result_projection=result_projection,
                legacy_v1_unit_layout=False,
                provenance=bound.provenance_json,
                bounded_artifact=True,
            )
            _revalidate_geometry(self.request.plan.geometry)

        def prepublish_check() -> None:
            try:
                _revalidate_geometry(self.request.plan.geometry)
            except StitchOperationRefused as error:
                raise ModuleArtifactRefused(error.code) from error

        try:
            terminal = self._output.publish(
                writer,
                cancel_token=cancel_token,
                prepublish_check=prepublish_check,
            )
        except AnalysisArtifactCleanupPending as error:
            self._state = "cleanup_pending"
            raise StitchOperationCleanupPending(self) from error
        self._emit("publish", total - 1, total)
        return self._strict_result(terminal, total=total)

    def retry_cleanup(self) -> StitchOperationResult:
        if self._result is not None:
            return self._result
        if self._state != "cleanup_pending" or self._output is None:
            raise RuntimeError("Stitch execution has no retryable cleanup")
        try:
            terminal = self._output.retry_cleanup()
        except AnalysisArtifactCleanupPending as error:
            raise StitchOperationCleanupPending(self) from error
        total = len(self.request.module.source.selected_labels) + 4
        self._emit("publish", total - 1, total)
        return self._strict_result(terminal, total=total)

    def retry_verification(self) -> StitchOperationResult:
        """Retry only strict detached reload; never replay science or writing."""

        if self._result is not None:
            return self._result
        if (
            self._state != "verification_failed"
            or self._verification_terminal is None
            or self._verification_terminal.disposition is not ModuleDisposition.COMMITTED
        ):
            raise RuntimeError("Stitch execution has no retryable verification")
        total = len(self.request.module.source.selected_labels) + 4
        return self._strict_result(self._verification_terminal, total=total)


def run_stitch_operation(
    request: StitchOperationRequest,
    *,
    cancel_token: threading.Event | None = None,
    progress_callback: Callable[[ModuleProgress], object] | None = None,
    coordinator=None,
) -> StitchOperationResult:
    """Run one prepared Stitch operation, retaining cleanup in its exception."""

    execution = StitchOperationExecution(request, coordinator=coordinator)
    return execution.run(
        cancel_token=cancel_token,
        progress_callback=progress_callback,
    )


__all__ = [
    "StitchContribution",
    "StitchGeometryInput",
    "StitchGeometryKind",
    "StitchGeometryReceipt",
    "StitchInputManifestReceipt",
    "StitchManifestFile",
    "StitchOperationCleanupPending",
    "StitchOperationExecution",
    "StitchOperationPlan",
    "StitchOperationRefused",
    "StitchOperationRequest",
    "StitchOperationResult",
    "StitchOperationVerificationError",
    "XuStitchOperationPlan",
    "capture_stitch_geometry",
    "prepare_stitch_operation",
    "run_stitch_operation",
]
