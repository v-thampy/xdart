"""Exact, blocking headless operation for one standalone Stitch artifact.

The generic module transaction owns source and output custody.  This module
adds the Stitch-specific boundary: bounded geometry assets, a canonical plan,
cooperative frame-load cancellation, deterministic provenance, and strict
post-commit payload admission.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from contextlib import contextmanager, ExitStack
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
    ModuleSourceGroupReceipt,
    ModuleTerminalResult,
    admit_module_artifact,
    module_artifact_request,
    module_plan_fingerprint,
    module_provenance_digest,
    xu_stitch_module_request,
    _native_stitch_module_request,
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
    MetadataTableResult,
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
from xrd_tools.sources.composite import CompositeFrameSource
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL,
    ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL,
    ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE,
    ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE,
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactPayload,
    AnalysisArtifactProjectionInvalid,
    _project_analysis_artifact_result_v1_compat,
    analysis_execution_attestation_digest,
    project_analysis_artifact_result,
    read_analysis_artifact,
    stitch_record_identity,
)
from xrd_tools.io.nexus import write_stitched
from xrd_tools.io.stitch_record import write_stitch_projection
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy
from xrd_tools.integrate.multi import StitchDiagnostics
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.integrate.stitch_gi import StitchGI
from xrd_tools.analysis.stitch_native import (
    capture_mask, revalidate_mask, native_mask, native_plan_identity,
    native_provenance, capture_callback, geometry_facts, label_saturation,
    capture_detector_image, native_detector_image,
)
from xrd_tools.corrections.detector import DEFAULT_MIN_FLAT_FRACTION
from xrd_tools.corrections.stack import CorrectionStack
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
#: prepare_stitch_operation's own claim: it derived the manifest and provenance from the same receipt and
#: admitted tables just before, so the request does not derive them again (PERF-STITCH-1). Any other
#: construction re-derives them.
_PREPARED_REQUEST_FACTORY = object()
#: Where the sample sits relative to a stacked arm's calibrated sample offset.
SAMPLE_POSITIONS = ("calibrated", "rotation_axis")
#: pyFAI error models a MultiGeometry Stitch may request (None = no sigma).
STITCH_ERROR_MODELS = (None, "poisson")


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
        # A single detector arm (a fourc/twoc 2θ) maps its one moving circle, del
        # (vertical) or nu (horizontal); the other is held at zero.
        if self.kind is StitchGeometryKind.PONI and (
            {name for name, _source in motors} not in ({"del", "nu"}, {"del"}, {"nu"})
            or {name for name, _value in references} != {name for name, _source in motors}
        ):
            raise ValueError(
                "PONI geometry requires del and/or nu source mappings and reference positions"
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
        record = _json_mapping(raw)
        declared_positions = _json_position_names(record)
        bound_positions = tuple(name for name, _source in request.source_motors)
        if set(bound_positions) != set(declared_positions):
            raise StitchOperationRefused(
                "GEOMETRY_MOTOR_MAPPING_MISMATCH",
                "goniometer JSON source mapping must bind every declared "
                "position exactly once and no others (declared: "
                f"{', '.join(declared_positions)}; bound: "
                f"{', '.join(bound_positions) or 'none'})",
            )
        from xrd_tools.core.geometry.stacked_arm import STACKED_ARM_MARKERS

        constants = record["trans_function"].get("constants", {})
        if type(constants) is dict and any(marker in constants for marker in STACKED_ARM_MARKERS):
            from xrd_tools.core.geometry import Diffractometer

            try:
                Diffractometer.from_pyfai_goniometer(
                    record, source_motors=dict(request.source_motors)
                )
            except (KeyError, TypeError, ValueError, NotImplementedError) as error:
                raise StitchOperationRefused(
                    "GEOMETRY_PARSE_FAILED", "invalid stacked-arm calibration"
                ) from error
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


@dataclass(eq=False, frozen=True, slots=True)
class StitchGroupManifestReceipt:
    """Ordered per-source manifests with one explicit global frame map."""

    members: tuple[StitchInputManifestReceipt, ...]
    module_source_fingerprint: str
    files: tuple[StitchManifestFile, ...] = field(init=False)
    contributions: tuple[StitchContribution, ...] = field(init=False)
    member_frames: tuple[tuple[int, int], ...] = field(init=False)
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim):
        if (_claim is not _REQUEST_FACTORY or type(self.members) is not tuple
                or not self.members or any(type(m) is not StitchInputManifestReceipt for m in self.members)
                or len({m.project_root for m in self.members}) != 1):
            raise TypeError("invalid Stitch group manifest")
        _sha256(self.module_source_fingerprint, "group source fingerprint")
        files, contributions, member_frames = [], [], []
        by_path = {}
        for index, member in enumerate(self.members):
            for item in member.files:
                if item.relative_path not in by_path:
                    by_path[item.relative_path] = len(files)
                    files.append(item)
                elif files[by_path[item.relative_path]] != item:
                    raise StitchOperationRefused("SOURCE_IDENTITY_MISMATCH")
            for item in member.contributions:
                contributions.append(replace(item, label=len(contributions), file_ordinal=
                    by_path[member.files[item.file_ordinal].relative_path]))
                member_frames.append((index, item.label))
        if len(contributions) > _MAX_CONTRIBUTIONS or len(files) > _MAX_MANIFEST_FILES:
            raise StitchOperationRefused("MANIFEST_CONTRIBUTION_LIMIT_EXCEEDED")
        object.__setattr__(self, "files", tuple(files))
        object.__setattr__(self, "contributions", tuple(contributions))
        object.__setattr__(self, "member_frames", tuple(member_frames))
        object.__setattr__(self, "fingerprint", module_plan_fingerprint(ModuleKind.STITCH,
            ("stitch-group-manifest-v1", self.module_source_fingerprint, tuple(m.fingerprint for m in self.members))))
        if len(json.dumps(self.to_provenance()).encode()) > _MAX_MANIFEST_BYTES:
            raise StitchOperationRefused("MANIFEST_BYTE_LIMIT_EXCEEDED")

    @property
    def project_root(self): return self.members[0].project_root
    @property
    def source_relative_path(self): return self.members[0].source_relative_path
    @property
    def source_scan(self): return None

    def to_provenance(self):
        return {"schema_version": "stitch-group-manifest-v1",
                "members": [member.to_provenance() for member in self.members],
                "member_frames": [list(value) for value in self.member_frames],
                "fingerprint": self.fingerprint}


def _source_members(source):
    return source.members if type(source) is ModuleSourceGroupReceipt else (source,)


def _table_names(table):
    if type(table) is tuple:
        names = tuple(column.name for column in table[0].columns)
        # Every selected motor must have the same unambiguous availability.
        return names
    return tuple(column.name for column in table.columns)


def _project_source_options(analysis, root: Path) -> str:
    """Return the closed, Project-relative S1 source-policy projection."""

    if analysis.source_spec.metadata_uri is not None:
        raise StitchOperationRefused(
            "SOURCE_SPEC_UNSUPPORTED",
            "S1 Stitch does not accept the unused SourceSpec.metadata_uri field",
        )
    options = dict(analysis.source_spec.options)
    if analysis.resolved_kind is SourceKind.SPEC:
        allowed = {"scan", "image_dir", "image_stem", "read_image_kwargs", "require_complete_images"}
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
    elif analysis.resolved_kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
        if options or not analysis.resolved_entry:
            raise StitchOperationRefused("SOURCE_SPEC_UNSUPPORTED", "NeXus requires an exact entry and no reader overrides")
        projected = {}
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
    if type(source) is ModuleSourceGroupReceipt:
        return StitchGroupManifestReceipt(tuple(
            _capture_input_manifest(member, member_table, project_root, selectors)
            for member, member_table in zip(source.members, table, strict=True)
        ), source.fingerprint, _REQUEST_FACTORY)
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
    native_records: bool = False
    conditioning: StitchConditioning | None = None
    mask_path: str | Path | None = None
    frame_screen: FrameScreenPolicy = field(default_factory=FrameScreenPolicy)
    #: 'calibrated' keeps a stacked arm's fitted sample offset; 'rotation_axis'
    #: drops it for a sample centred on the rotation axis (e.g. after GI alignment).
    sample_position: str = "calibrated"
    #: pyFAI corrections. Solid angle stays on for a stacked arm with a sample
    #: offset, whose per-frame distances differ. polarization_factor None = off.
    correct_solid_angle: bool = True
    polarization_factor: float | None = 0.99
    error_model: str | None = None
    dark_path: str | Path | None = None
    flat_path: str | Path | None = None
    #: Flat pixels below this fraction of the flat's median are masked as dead.
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION
    #: Grazing incidence: GI cuts written beside I(q), and a Stitch 2D's q_ip-q_oop map (native, in Q).
    gi: StitchGI | None = None
    #: A native Stitch 2D's map axis, as Int 2D's: its radial unit and range (None: its frames' extent);
    #: ``unit`` and ``radial_range`` are I(q)'s beside it.
    map_unit: str = "q_A^-1"
    map_radial_range: tuple[float, float] | None = None
    _mask_receipt: object = field(init=False, default=None, repr=False)
    _dark_receipt: object = field(init=False, default=None, repr=False)
    _flat_receipt: object = field(init=False, default=None, repr=False)
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
        _native_plan_fields(self)
        radial = _range(self.radial_range, "radial range", required=not self.native_records)
        azimuth = _range(
            self.azimuth_range,
            "azimuth range",
            # a native map's None: its frames' own χ extent, scouted as the radial one is
            required=self.mode == "2d" and not self.native_records,
        )

        if self.mode == "1d" and azimuth is not None:
            raise ValueError("a 1-D Stitch plan cannot carry an azimuth range")
        if self.unit not in ({"q_A^-1", "2th_deg"} if self.native_records else {"q_A^-1"}) or self.method != "full":
            raise ValueError("S1 requires unit='q_A^-1' and notebook method='full'")
        if (
            self.monitor_selector is not None
            and type(self.monitor_selector) is not MetadataColumnSelector
        ):
            raise TypeError("monitor selector must be exact MetadataColumnSelector")
        if type(self.use_detector_mask) is not bool:
            raise TypeError("use_detector_mask must be an exact bool")
        if type(self.frame_screen) is not FrameScreenPolicy:
            raise TypeError("frame screen must be an exact FrameScreenPolicy")
        if self.sample_position not in SAMPLE_POSITIONS:
            raise ValueError(f"sample position must be one of {SAMPLE_POSITIONS}")
        if (
            type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= 4 * 1024 * 1024 * 1024
        ):
            raise ValueError("max_frame_bytes must be a bounded positive exact integer")
        _correction_fields(self)
        _gi_field(self)
        _map_axis_fields(self)
        object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "azimuth_range", azimuth)
        object.__setattr__(
            self,
            "fingerprint",
            module_plan_fingerprint(ModuleKind.STITCH, self._canonical_value()),
        )

    def _canonical_value(self) -> tuple[object, ...]:
        selector = self.monitor_selector
        value = (
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
            self.frame_screen.identity(),
            ("sample-position", self.sample_position),
            ("corrections-v1", self.correct_solid_angle, self.polarization_factor,
             self.error_model,
             None if self._dark_receipt is None else self._dark_receipt.fingerprint,
             None if self._flat_receipt is None else self._flat_receipt.fingerprint),
        ) + _flat_cutoff_identity(self) + _gi_identity(self) + _map_axis_identity(self)
        return value + native_plan_identity(self) if self.native_records else value


@dataclass(frozen=True, slots=True)
class XuStitchOperationPlan:
    calibration: XuStitchCalibrationReceipt
    effective_geometry: XuStitchEffectiveGeometryProjection
    q_min_A_inverse: float | None = None
    q_max_A_inverse: float | None = None
    npt_1d: int = 1500
    monitor_selector: MetadataColumnSelector | None = None
    max_frame_bytes: int = 256 * 1024 * 1024
    unit: str = "q_A^-1"
    radial_range: tuple[float, float] | None = None
    native_records: bool = False
    conditioning: StitchConditioning | None = None
    mask_path: str | Path | None = None
    frame_screen: FrameScreenPolicy = field(default_factory=FrameScreenPolicy)
    #: pyFAI's polarization, dark and flat, applied like MultiGeometry; None = off.
    polarization_factor: float | None = 0.99
    dark_path: str | Path | None = None
    flat_path: str | Path | None = None
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION
    #: Grazing incidence: a GI cut written beside I(q) (native Stitch 1D only).
    gi: StitchGI | None = None
    #: '2d' (native records): the (radial, χ) map beside I(q), its points and χ range (°).
    mode: str = "1d"
    npt_rad_2d: int = 1000
    npt_azim_2d: int = 360
    azimuth_range: tuple[float, float] | None = None
    #: The map's radial unit and range (None: its frames' extent), as Int 2D's; I(q)'s are ``unit``, ``radial_range``.
    map_unit: str = "q_A^-1"
    map_radial_range: tuple[float, float] | None = None
    _mask_receipt: object = field(init=False, default=None, repr=False)
    _dark_receipt: object = field(init=False, default=None, repr=False)
    _flat_receipt: object = field(init=False, default=None, repr=False)
    backend: str = field(init=False, default="xu_hist")
    method: str = field(init=False, default="numpy_histogram_center_v1")
    use_detector_mask: bool = field(init=False, default=True)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _native_plan_fields(self)
        if type(self.frame_screen) is not FrameScreenPolicy:
            raise TypeError("frame screen must be an exact FrameScreenPolicy")
        if not self.native_records and (self.dark_path is not None or self.flat_path is not None):
            raise ValueError("XU dark and flat corrections require native records")
        _polarization_field(self)
        _detector_image_fields(self)
        _gi_field(self)
        if self.q_min_A_inverse is not None or self.q_max_A_inverse is not None:
            if (type(self.q_min_A_inverse) is not float or type(self.q_max_A_inverse) is not float
                    or self.unit != "q_A^-1" or self.radial_range not in
                    (None, (self.q_min_A_inverse, self.q_max_A_inverse))):
                raise TypeError("legacy XU bounds require two floats and no competing range")
            object.__setattr__(self, "radial_range", (self.q_min_A_inverse, self.q_max_A_inverse))
        object.__setattr__(self, "radial_range", _range(self.radial_range, "radial range", required=not self.native_records))
        if (
            type(self.calibration) is not XuStitchCalibrationReceipt
            or type(self.effective_geometry)
            is not XuStitchEffectiveGeometryProjection
            or self.effective_geometry.asset_semantic_fingerprint
            != self.calibration.semantic_fingerprint
            or self.unit not in ({"q_A^-1", "2th_deg"} if self.native_records else {"q_A^-1"})
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
        if not self.native_records and self.calibration.projection.value["version"] != 1:
            raise ValueError("non-native XU plans require canonical version 1 calibration")
        _xu_map_fields(self)
        _map_axis_fields(self)
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
                    self.radial_range,
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
                    self.frame_screen.identity(),
                ) + _xu_corrections_identity(self) + _gi_identity(self)
                + (native_plan_identity(self) if self.native_records else ())
                # a 1-D plan keeps the identity it had
                + (() if self.mode == "1d" else (("xu-map-v1",) + _xu_map_grid(self),))
                + _map_axis_identity(self),
            ),
        )

def _xu_map_fields(plan):
    """A Stitch 2D's map (native records, no GI): its points and its χ range; a 1-D plan has none."""
    if plan.mode not in {"1d", "2d"}:
        raise ValueError("stitch mode must be '1d' or '2d'")
    azimuth = _range(plan.azimuth_range, "azimuth range", required=False)  # None: the frames' χ extent
    if plan.mode == "1d":
        if azimuth is not None:
            raise ValueError("a 1-D Stitch plan cannot carry an azimuth range")
        return
    if not plan.native_records:
        raise ValueError("an xrayutilities Stitch 2D is a native record")
    for name in ("npt_rad_2d", "npt_azim_2d"):
        value = getattr(plan, name)
        if type(value) is not int or not 1 <= value <= _MAX_AXIS_POINTS:
            raise ValueError(f"{name} must be a bounded positive exact integer")
    if plan.npt_rad_2d * plan.npt_azim_2d > _MAX_2D_POINTS:
        raise ValueError("2-D Stitch output exceeds the bounded artifact point limit")
    object.__setattr__(plan, "azimuth_range", azimuth)


def _xu_map_grid(plan):
    """The map's (radial points, χ points, χ range) of an xrayutilities Stitch 2D."""
    return plan.npt_rad_2d, plan.npt_azim_2d, plan.azimuth_range


def _xu_corrections_identity(plan):
    """Only a corrected XU plan changes its identity; an uncorrected one keeps the old one."""
    if plan.polarization_factor is None and plan._dark_receipt is None and plan._flat_receipt is None:
        return ()
    return (("xu-corrections-v1", plan.polarization_factor,
             None if plan._dark_receipt is None else plan._dark_receipt.fingerprint,
             None if plan._flat_receipt is None else plan._flat_receipt.fingerprint),
            ) + _flat_cutoff_identity(plan)


def _gi_field(plan):
    if plan.gi is None:
        return
    if type(plan.gi) is not StitchGI:
        raise TypeError("Stitch GI must be an exact StitchGI")
    if not plan.native_records or plan.unit != "q_A^-1":
        raise ValueError("a GI cut is written beside I(q) by a native Stitch in Q")


def _gi_identity(plan):
    """Only a GI plan changes its identity; every other plan keeps the one it had."""
    return () if plan.gi is None else (plan.gi.identity(),)


def _map_axis_fields(plan):
    """A native Stitch 2D's map axis (Q or 2θ; Q with GI) and its range; any other plan has none."""
    if type(plan.map_unit) is not str:
        raise TypeError("the map unit must be a string")
    if not (plan.native_records and plan.mode == "2d"):
        if (plan.map_unit, plan.map_radial_range) != ("q_A^-1", None):
            raise ValueError("only a native Stitch 2D has a map axis of its own")
        return
    if plan.map_unit not in {"q_A^-1", "2th_deg"} or (plan.gi is not None and plan.map_unit != "q_A^-1"):
        raise ValueError("a Stitch 2D map is binned in Q or 2theta (Q with GI)")
    object.__setattr__(plan, "map_radial_range", _range(plan.map_radial_range, "map radial range", required=False))


def _map_axis_identity(plan):
    """A native Stitch 2D's map axis; a 1-D plan keeps the identity it had."""
    if not (plan.native_records and plan.mode == "2d"):
        return ()
    return (("map-axis-v1", plan.map_unit, plan.map_radial_range),)


def _flat_cutoff_identity(plan):
    """A flat cutoff away from 10 % changes the identity; the default keeps the old one."""
    if plan._flat_receipt is None or plan.flat_min_fraction == DEFAULT_MIN_FLAT_FRACTION:
        return ()
    return (("flat-min-fraction", plan.flat_min_fraction),)


def _correction_fields(plan):
    if type(plan.correct_solid_angle) is not bool:
        raise TypeError("correct_solid_angle must be an exact bool")
    _polarization_field(plan)
    if plan.error_model not in STITCH_ERROR_MODELS:
        raise ValueError(f"Stitch error model must be one of {STITCH_ERROR_MODELS}")
    _detector_image_fields(plan)


def _polarization_field(plan):
    factor = plan.polarization_factor
    if factor is not None:
        if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                or not math.isfinite(factor) or not -1.0 <= factor <= 1.0):
            raise ValueError("polarization factor must be a finite number from -1 to 1, or None")
        object.__setattr__(plan, "polarization_factor", float(factor))


def _detector_image_fields(plan):
    fraction = plan.flat_min_fraction
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0.0 <= fraction < 1.0):
        raise ValueError("the flat cutoff must be a number from 0 to below 1")
    object.__setattr__(plan, "flat_min_fraction", float(fraction))
    for role in ("dark", "flat"):
        path = getattr(plan, f"{role}_path")
        if path is not None:
            receipt, _pixels = capture_detector_image(path, role, plan.max_frame_bytes,
                                                      plan.flat_min_fraction)
            object.__setattr__(plan, f"{role}_path", receipt.lexical_path)
            object.__setattr__(plan, f"_{role}_receipt", receipt)


def _native_plan_fields(plan):
    if type(plan.native_records) is not bool:
        raise TypeError("native_records must be an exact bool")
    if plan.conditioning is not None and type(plan.conditioning) is not StitchConditioning:
        raise TypeError("conditioning must be exact StitchConditioning")
    if not plan.native_records and (plan.conditioning is not None or plan.mask_path is not None):
        raise ValueError("native conditioning and mask require native_records")
    if plan.native_records and plan.conditioning is None:
        object.__setattr__(plan, "conditioning", StitchConditioning("off"))
    if plan.mask_path is not None:
        receipt = capture_mask(plan.mask_path)
        object.__setattr__(plan, "mask_path", receipt.lexical_path)
        object.__setattr__(plan, "_mask_receipt", receipt)


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
            if name is not None  # a single 2θ arm reads no nu
        )
        energy_name = str(acquisition["source_energy_selector"])
        energy_count = table_names.count(energy_name)
        if energy_count > 1:
            raise StitchOperationRefused(
                "AMBIGUOUS_METADATA_SELECTOR",
                f"metadata column {energy_name!r} must occur at most once",
            )
        if energy_count == 1 and not plan.native_records:
            requested += (MetadataColumnSelector(energy_name, 0),)
    else:
        raise TypeError("stitch selectors require an exact Stitch plan")
    requested += (() if plan.monitor_selector is None else (plan.monitor_selector,))
    if plan.gi is not None:  # the incidence (and xrayutilities' sample circles), per frame
        requested += tuple(MetadataColumnSelector(name, 0) for name in plan.gi.metadata_names())
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
    if plan.native_records:
        # Each member's optional energy is bound independently by native provenance.
        return None
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
    # None: the circle a single 2θ arm holds at zero
    del_key, nu_key = (None if acquisition["source_motors"][circle] is None
                       else (str(acquisition["source_motors"][circle]), 0) for circle in ("del", "nu"))
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
    # a refined calibration's scans are checked against the energy they recorded
    expected_energy = float(acquisition.get("source_energy_eV", acquisition["energy_eV"]))
    for contribution in manifest.contributions:
        values = {
            (name, occurrence): value
            for name, occurrence, value in contribution.values
        }
        try:
            del_value = 0.0 if del_key is None else float(values[del_key])
            nu_value = 0.0 if nu_key is None else float(values[nu_key])
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
    *,
    preparation: dict | None = None,
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
    value = {
        "schema_version": "stitch-operation-v1",
        "kind": "stitch",
        "source": {
            "source_fingerprint": (source.fingerprint if type(source) is ModuleSourceGroupReceipt
                                   else source.analysis.source_fingerprint),
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
            "radial_range": None if plan.radial_range is None else list(plan.radial_range),
            "azimuth_range": (
                None if plan.azimuth_range is None else list(plan.azimuth_range)
            ),
            "unit": plan.unit,
            **({} if not (plan.native_records and plan.mode == "2d") else {
                "map_unit": plan.map_unit,
                "map_radial_range": None if plan.map_radial_range is None else list(plan.map_radial_range)}),
            "method": plan.method,
            "monitor_selector": (
                None
                if selector is None
                else {"name": selector.name, "occurrence": selector.occurrence}
            ),
            "use_detector_mask": plan.use_detector_mask,
            "streaming_multigeometry": True,
            "frame_screen": plan.frame_screen.to_provenance(),
            "sample_position": plan.sample_position,
            "corrections": {
                "solid_angle": plan.correct_solid_angle,
                "polarization_factor": plan.polarization_factor,
                "error_model": plan.error_model,
                "dark": (None if plan._dark_receipt is None
                         else plan._dark_receipt.to_provenance(Path(manifest.project_root))),
                "flat": (None if plan._flat_receipt is None
                         else plan._flat_receipt.to_provenance(Path(manifest.project_root))),
            },
            "max_frame_bytes": plan.max_frame_bytes,
            "plan_fingerprint": plan.fingerprint,
        },
        "engine": {"pyfai": _engine_version(), "numpy": np.__version__},
        "output": {"kind": output.kind.value},
        "holds": [
            "2d-orientation-parity",
            "gi-corrections",
        ],
    }
    return native_provenance(value, source, output, plan, manifest, preparation=preparation)


def _xu_provenance(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan: XuStitchOperationPlan,
    manifest: StitchInputManifestReceipt,
    observations: XuStitchScienceObservations,
    energy_selector: MetadataColumnSelector | None,
    *,
    preparation: dict | None = None,
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
    value = {
        "schema_version": "stitch-operation-v2-xu-intent",
        "kind": "stitch",
        "backend": "xu_hist",
        "source": {
            "source_fingerprint": (source.fingerprint if type(source) is ModuleSourceGroupReceipt
                                   else source.analysis.source_fingerprint),
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
            "radial_range": None if plan.radial_range is None else list(plan.radial_range),
            "npt_1d": plan.npt_1d,
            "monitor_selector": (
                None
                if monitor is None
                else {"name": monitor.name, "occurrence": monitor.occurrence}
            ),
            "use_detector_mask": plan.use_detector_mask,
            "max_frame_bytes": plan.max_frame_bytes,
            "frame_screen": plan.frame_screen.to_provenance(),
            **({} if not _xu_corrections_identity(plan) else {"corrections": {
                "polarization_factor": plan.polarization_factor,
                "dark": (None if plan._dark_receipt is None
                         else plan._dark_receipt.to_provenance(Path(manifest.project_root))),
                "flat": (None if plan._flat_receipt is None
                         else plan._flat_receipt.to_provenance(Path(manifest.project_root))),
            }}),
            "asset_receipt_fingerprint": receipt.fingerprint,
            "effective_geometry_fingerprint": plan.effective_geometry.fingerprint,
            "plan_fingerprint": plan.fingerprint,
            **({} if plan.mode == "1d" else {"map": {
                "npt_rad_2d": plan.npt_rad_2d, "npt_azim_2d": plan.npt_azim_2d,
                # None (Auto): the frames' own χ extent, scouted when stitching
                "azimuth_range": None if plan.azimuth_range is None else list(plan.azimuth_range),
                "unit": plan.map_unit,
                "radial_range": None if plan.map_radial_range is None else list(plan.map_radial_range)}}),
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
            "explicit-ub-and-sample-placement",
            "operator-configured-sample-circles",
            "generic-xu-diffractometers-and-detectors",
            "external-or-combined-masks",
            "sensor-parallax-xu-geometry",
            "xu-sigma-and-variance",
            "mixed-energy-grouped-xu-stitch",
            "historical-json-migration",
            "unvalidated-platforms-and-live-gates",
        ],
    }
    return native_provenance(value, source, output, plan, manifest, preparation=preparation)


@dataclass(eq=False, frozen=True, slots=True)
class StitchOperationRequest:
    module: ModuleOperationRequest
    plan: StitchOperationPlan | XuStitchOperationPlan
    manifest: StitchInputManifestReceipt
    provenance_json: str
    _claim: InitVar[object] = None
    #: The contributing frames' display facts its preparation bound, from the same layout read
    #: (``stitch_native.native_frame_members`` returns them); None: read them again.
    _preview_members: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim not in (_REQUEST_FACTORY, _PREPARED_REQUEST_FACTORY)
            or type(self.module) is not ModuleOperationRequest
            or self.module.kind is not ModuleKind.STITCH
            or type(self.plan) not in {StitchOperationPlan, XuStitchOperationPlan}
            or type(self.manifest) not in {StitchInputManifestReceipt, StitchGroupManifestReceipt}
            or type(self.provenance_json) is not str
            or self.module.plan_fingerprint != self.plan.fingerprint
            or self.manifest.module_source_fingerprint
            != self.module.source.fingerprint
            or tuple(item.label for item in self.manifest.contributions)
            != self.module.source.selected_labels
        ):
            raise TypeError("stitch operation request is invalid")
        if _claim is _PREPARED_REQUEST_FACTORY:
            self._prepared_checks()
            return
        table = _fresh_table(self.module.source)
        table_names = _table_names(table)
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

    def _prepared_checks(self) -> None:
        """What a prepared request still checks without reading its sources again."""
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
        if type(self.plan) is XuStitchOperationPlan and (
            self.module._xu_stitch_v2_bound is not True
            or self.plan.calibration.project_root != self.manifest.project_root
        ):
            raise ValueError("XU Stitch request lacks exact Project authority")

    @property
    def provenance(self) -> dict[str, object]:
        return json.loads(self.provenance_json)

    def __copy__(self):
        raise TypeError("stitch operation request is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("stitch operation request is not copyable")


def _admitted_tables(source: ModuleSourceReceipt, tables):
    """The metadata tables this preparation already read for ``source``, without reading them again: each
    must be the exact table its receipt was made from (PERF-STITCH-1). Operations re-verify their sources
    when they run."""
    members = _source_members(source)
    values = tables if type(tables) is tuple else (tables,)
    if len(values) != len(members) or any(
        type(table) is not MetadataTableResult
        or table.disposition is not AnalysisDisposition.COMPLETED
        or table.receipt != member.analysis
        or table.table_fingerprint != member.table_fingerprint
        or table.labels != member.analysis.labels
        for member, table in zip(members, values)
    ):
        raise StitchOperationRefused(
            "SOURCE_IDENTITY_MISMATCH", "admitted metadata differs from the source receipt"
        )
    return values if type(source) is ModuleSourceGroupReceipt else values[0]


def _fresh_table(source: ModuleSourceReceipt):
    if type(source) is ModuleSourceGroupReceipt:
        return tuple(_fresh_table(member) for member in source.members)
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
    tables=None,
) -> StitchOperationRequest:
    """Bind exact source, geometry, plan, provenance, and output intent.

    ``tables`` are the metadata tables the caller just read and made ``source`` from (one per member):
    reused, not read again (PERF-STITCH-1); absent, they are read here.
    """

    if (
        type(source) not in {ModuleSourceReceipt, ModuleSourceGroupReceipt}
        or source.kind is not ModuleKind.STITCH
        or type(output) is not ModuleOutputRequest
        or type(plan) not in {StitchOperationPlan, XuStitchOperationPlan}
    ):
        raise TypeError("stitch preparation requires exact Stitch module values")
    if any(member.analysis.resolved_kind not in {
        SourceKind.SPEC,
        SourceKind.TIFF_SERIES,
        SourceKind.NEXUS_STACK,
        SourceKind.EIGER_MASTER,
    } for member in _source_members(source)):
        raise StitchOperationRefused(
            "SOURCE_KIND_UNSUPPORTED",
            "S1 Stitch accepts only an extensionless SPEC scan or TIFF series",
        )
    expected_kind = (
        AnalysisArtifactKind.STITCH_1D
        if plan.mode == "1d"
        else AnalysisArtifactKind.STITCH_2D
    )
    if output.kind is not expected_kind:
        raise StitchOperationRefused(
            "OUTPUT_KIND_MISMATCH", "Stitch mode and output artifact kind differ"
        )
    table = _fresh_table(source) if tables is None else _admitted_tables(source, tables)
    # this preparation's admitted facts (its tables, then its frame facts), never kept past it
    preparation = {"tables": table if type(table) is tuple else (table,)}
    names = _table_names(table)
    required = _required_stitch_selectors(plan, names)
    tables = table if type(table) is tuple else (table,)
    for member, member_table in zip(_source_members(source), tables, strict=True):
        member_names = _table_names(member_table)
        for selector in required:
            if selector.occurrence != 0 or member_names.count(selector.name) != 1:
                raise StitchOperationRefused("AMBIGUOUS_METADATA_SELECTOR",
                    f"metadata column {selector.name!r} must occur exactly once in every member")
            if selector not in member.resolved_selectors:
                raise StitchOperationRefused("UNBOUND_METADATA_SELECTOR", selector.name)
        options = dict(member.analysis.source_spec.options)
        if dict(options.get("read_image_kwargs", {})).get("rotation", 0) != 0:
            raise StitchOperationRefused("SOURCE_ORIENTATION_CONFLICT",
                "source rotation must be zero; Stitch owns image orientation")
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
            preparation=preparation,
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
            preparation=preparation,
        )
        # a native plan's request is the native one, below
        module = None if plan.native_records else xu_stitch_module_request(
            source,
            output,
            plan.fingerprint,
            module_provenance_digest(ModuleKind.STITCH, provenance),
        )
    if plan.native_records:
        native_mask(plan)  # Admit mask orientation/shape before any detector read.
        module = _native_stitch_module_request(source, output, plan, manifest, provenance)
    canonical = module_artifact_request(module, provenance).provenance_json
    request = StitchOperationRequest(
        module,
        plan,
        manifest,
        canonical,
        _PREPARED_REQUEST_FACTORY,
    )
    if "members" in preparation:
        object.__setattr__(request, "_preview_members", tuple(preparation["members"]))
    return request


class _SelectedStitchSource(BaseFrameSource):
    def __init__(self, source, labels):
        super().__init__(name=source.name, frame_indices=labels, capabilities=source.capabilities)
        self._source = source

    def load_frame(self, index): return self._source.load_frame(index)
    def load_native_frame(self, index):
        return getattr(self._source, "load_native_frame", self._source.load_frame)(index)
    def metadata_for(self, index): return self._source.metadata_for(index)
    def frame_for(self, index): return self._source.frame_for(index)
    @property
    def motors(self):
        rows = [self.metadata_for(index) for index in self.frame_indices]
        keys = set.intersection(*(set(row) for row in rows)) if rows else set()
        return {key: np.asarray([row[key] for row in rows]) for key in keys}


@contextmanager
def _requalified_stitch_source(source, *, cancel_token=None):
    with ExitStack() as stack:
        opened = [stack.enter_context(requalified_analysis_source(member.analysis, cancel_token=cancel_token))
                  for member in _source_members(source)]
        if type(source) is ModuleSourceGroupReceipt:
            yield CompositeFrameSource([_SelectedStitchSource(item, member.selected_labels)
                                        for item, member in zip(opened, source.members, strict=True)])
        else:
            yield opened[0]


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

    def load_native_frame(self, index: int) -> np.ndarray:
        image = getattr(self._source, "load_native_frame", self._source.load_frame)(index)
        if image.size * 8 > self._max_frame_bytes:
            raise MemoryError("oriented float64 frame exceeds max_frame_bytes")
        return self._orientation.apply(image)

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
        base = Diffractometer.psic(del_=motors.get("del", "del"), nu=motors.get("nu", "nu"))
        base = replace(  # a single arm's unmapped circle is held at zero
            base,
            rot1=(AngleMapping(source_motor=motors["nu"], offset=-references["nu"])
                  if "nu" in motors else AngleMapping()),
            rot2=(AngleMapping(source_motor=motors["del"], offset=-references["del"])
                  if "del" in motors else AngleMapping()),
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


def _sample_positioned(plan: StitchOperationPlan, diffractometer):
    """Apply the plan's sample position to a stacked arm's calibrated sample offset."""
    arm = getattr(diffractometer, "stacked_arm", None)
    if plan.sample_position == "rotation_axis" and arm is not None and arm.has_sample_offset:
        return replace(diffractometer, stacked_arm=arm.without_sample_offset())
    return diffractometer


def _runtime_plan(plan: StitchOperationPlan, diffractometer, saturation=None) -> StitchPlan:
    diffractometer = _sample_positioned(plan, diffractometer)
    mask = _detector_mask(diffractometer) if plan.use_detector_mask else None
    external = native_mask(plan)
    if external is not None:
        mask = external if mask is None else np.logical_or(mask, external)
    return StitchPlan(
        corrections=CorrectionStack(solid_angle=plan.correct_solid_angle,
                                    polarization_factor=plan.polarization_factor),
        error_model=plan.error_model,
        dark=native_detector_image(plan, plan._dark_receipt),
        flat=native_detector_image(plan, plan._flat_receipt),
        flat_min_fraction=plan.flat_min_fraction,
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
        mask=mask,
        conditioning=plan.conditioning,
        saturation=saturation,
        max_eager_bytes=None,
        streaming_multigeometry=True,
        stitch_gi=plan.gi,
        # a native Stitch 2D is an Int 2D record: I(q) beside the map, each on its own axis
        stitch_profile=plan.native_records and plan.mode == "2d",
        **({} if not (plan.native_records and plan.mode == "2d") else dict(
            map_unit=plan.map_unit, map_radial_range=plan.map_radial_range)),
    )


@dataclass(eq=False, frozen=True, slots=True)
class StitchOperationResult:
    request: StitchOperationRequest
    terminal: ModuleTerminalResult
    payload: AnalysisArtifactPayload | None = None
    #: dropped-frame screening summary (policy, excluded/empty/partial labels)
    frame_screen: Mapping[str, object] | None = None

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


def _screen_row_motor(plan) -> str:
    """The scan motor whose value is a frame's screening row: psic's outer circle (nu), when
    the calibration moves both detector circles — read from its geometry as bound to the
    scan, never from how its positions are spelled. A single arm is one row ('')."""
    if type(plan) is XuStitchOperationPlan:  # the asset's keys are del and nu themselves
        motors = plan.calibration.projection.value["acquisition"]["source_motors"]
        return str(motors.get("nu") or "") if motors.get("del") else ""
    diffractometer, _orientation = _runtime_geometry(plan.geometry, plan.geometry.content)
    arm = diffractometer.stacked_arm
    if arm is not None:
        return arm.nu_motor if arm.nu_motor and arm.del_motor else ""
    # independent mappings (a goniometer JSON, or PONI + offsets), as psic's: rot1 follows
    # nu, rot2 follows del
    rot1, rot2 = diffractometer.rot1, diffractometer.rot2
    return rot1.source_motor if rot1.is_active and rot2.is_active else ""


def _frame_screen_rows(request: StitchOperationRequest) -> tuple[float, ...]:
    """Each selected frame's outer-circle (nu) value: its dropped-frame screening row. A
    single arm (one circle held at zero) is one row, whichever circle moves."""
    name = _screen_row_motor(request.plan)
    manifest = request.manifest
    values = {item.label: {(n, o): v for n, o, v in item.values}
              for item in manifest.contributions}
    # A group's members are separate scans; never compare frames across them.
    members = ({item.label: member for item, (member, _label)
                in zip(manifest.contributions, manifest.member_frames, strict=True)}
               if type(manifest) is StitchGroupManifestReceipt else {})
    return tuple((members.get(label, 0),
                  round(float(values[label].get((name, 0), 0.0)), 6) if name else 0.0)
                 for label in request.module.source.selected_labels)


def _protected_stitch_inputs(request: StitchOperationRequest) -> tuple[Path, ...]:
    root = Path(request.manifest.project_root)
    inputs = [root / request.manifest.source_relative_path]
    inputs.extend(root / item.relative_path for item in request.manifest.files)
    receipts = (request.plan._mask_receipt, getattr(request.plan, "_dark_receipt", None),
                getattr(request.plan, "_flat_receipt", None))
    for receipt in receipts:
        if receipt is not None:
            inputs.extend((Path(receipt.lexical_path), Path(receipt.resolved_path)))
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


def _record_frame_screen(records, verdicts):
    """Stamp each captured frame record with its dropped-frame verdict.

    Verdicts join on stream position, which is the record's contribution index;
    grouped sources repeat frame labels across members.
    """
    if not records:
        return
    by_position = {verdict.position: verdict for verdict in verdicts}
    if sorted(by_position) != list(range(len(records))):
        raise StitchOperationRefused("FRAME_SCREEN_VERDICT_MISMATCH",
                                     "frame check verdicts differ from the captured frames")
    for record in records:
        verdict = by_position[record["identity"]["contribution_index"]]
        record["screen"], record["excluded"] = verdict.status, verdict.excluded


def _native_artifact_args(request, records):
    if not request.plan.native_records:
        return {}
    if len(records) != len(request.module.source.selected_labels):
        raise StitchOperationRefused("FRAME_RECORD_COUNT_MISMATCH")
    return {"wavelength_A": request.provenance["wavelength_A"],
            "frame_records": records, "source_base": request.manifest.project_root}


def _gi_cuts(plan, cuts):
    """A GI Stitch 1D's cuts as the projection takes them, in the plan's order (() without GI)."""
    if len(cuts) != (0 if plan.gi is None else len(plan.gi.cuts)):
        raise AnalysisArtifactProjectionInvalid("the science returned other GI cuts than the plan's")
    return tuple((mode, cut.unit, cut.radial, cut.intensity, diagnostics.coverage, diagnostics.normalization)
                 for mode, (cut, diagnostics) in zip(() if plan.gi is None else plan.gi.cuts, cuts))


def _gi_map(gi_map):
    """A GI Stitch 2D's q_ip-q_oop map as the projection takes it (None without one)."""
    if gi_map is None:
        return None
    stitched, diagnostics = gi_map
    return (stitched.unit, stitched.azimuthal_unit, stitched.radial, stitched.azimuthal, stitched.intensity,
            diagnostics.coverage, diagnostics.normalization)


def _write_record(entry, target, projection, native_args) -> None:
    """A Stitch 1D result, or a native Stitch 2D's map and I(q), as an Int record (``stitch_record``),
    from its exact projection."""
    write_stitch_projection(entry, target, projection, frame_records=native_args.get("frame_records"),
                            source_base=native_args.get("source_base"))


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
        self._frame_screen = None
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
            value = StitchOperationResult(self.request, terminal, frame_screen=self._frame_screen)
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
                payload.schema_version != (ANALYSIS_SCHEMA_VERSION_XU_STITCH_NATIVE
                    if self.request.plan.native_records else ANALYSIS_SCHEMA_VERSION_XU_STITCH_NEUTRAL)
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
            payload.schema_version != (ANALYSIS_SCHEMA_VERSION_STITCH_NATIVE
                if self.request.plan.native_records else ANALYSIS_SCHEMA_VERSION_STITCH_NEUTRAL)
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
        value = StitchOperationResult(self.request, terminal, payload, frame_screen=self._frame_screen)
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
            records, frame_callback = capture_callback(self.request)
            external_mask = native_mask(plan)
            self._emit("calibration", 1, total)
            with _requalified_stitch_source(
                self.request.module.source,
                cancel_token=cancel_token,
            ) as source:
                if plan.native_records:
                    _wavelength, _shape, orientation = geometry_facts(plan)
                    source = _OrientedSource(source, orientation, plan.max_frame_bytes)

                def science_progress(done: int, _inner_total: int) -> None:
                    self._emit("science", 1 + done, total)

                screen = FrameScreen(plan.frame_screen, labels, _frame_screen_rows(self.request))
                scientific = run_xu_hist_stitch_1d(
                    plan.calibration,
                    source,
                    frame_indices=labels,
                    frame_screen=screen,
                    unit=plan.unit,
                    radial_range=plan.radial_range,
                    conditioning=plan.conditioning,
                    saturation=label_saturation(self.request),
                    mask=external_mask,
                    frame_record_callback=frame_callback,
                    polarization_factor=plan.polarization_factor,
                    dark=native_detector_image(plan, plan._dark_receipt),
                    flat=native_detector_image(plan, plan._flat_receipt),
                    flat_min_fraction=plan.flat_min_fraction,
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
                    gi=plan.gi,
                    map_grid=None if plan.mode == "1d" else _xu_map_grid(plan),
                    **({} if plan.mode == "1d" else dict(map_unit=plan.map_unit,
                                                        map_radial_range=plan.map_radial_range)),
                )
                self._frame_screen = screen.summary()
                _record_frame_screen(records, screen.verdicts)
            revalidate_xu_stitch_calibration(plan.calibration)
            revalidate_mask(plan._mask_receipt)
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
            StitchOperationRefused,
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
        if (scientific.map is None) is not (plan.mode == "1d"):
            return self._terminal(ModuleDisposition.FAILED, "INVALID_SCIENCE_RESULT")
        try:
            if scientific.map is None:
                result = dict(kind=AnalysisArtifactKind.STITCH_1D, axes=(("q", payload.radial),),
                              axis_units=(("q", payload.unit),), intensity=payload.intensity,
                              sigma=payload.sigma, coverage=diagnostics.coverage,
                              normalization=diagnostics.normalization)
            else:  # the map, and I(q) beside it
                stitched, map_diagnostics = scientific.map
                result = dict(kind=AnalysisArtifactKind.STITCH_2D,
                              axes=(("q", stitched.radial), ("chi", stitched.azimuthal)),
                              axis_units=(("q", stitched.unit), ("chi", stitched.azimuthal_unit)),
                              intensity=stitched.intensity, sigma=stitched.sigma,
                              coverage=map_diagnostics.coverage, normalization=map_diagnostics.normalization,
                              profile=(payload.unit, payload.radial, payload.intensity, payload.sigma,
                                       diagnostics.coverage, diagnostics.normalization))
            result_projection = project_analysis_artifact_result(
                **result,
                **_native_artifact_args(self.request, records),
                gi_cuts=_gi_cuts(plan, scientific.gi_cuts),
                gi_map=_gi_map(scientific.gi_map),
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
                revalidate_mask(plan._mask_receipt)
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
                StitchOperationRefused,
            ) as error:
                raise ModuleArtifactRefused(error.code) from error

        def writer(entry) -> None:
            revalidate_xu_stitch_calibration(plan.calibration)
            _write_record(entry, self.request.module.output.target, result_projection,
                          _native_artifact_args(self.request, records))
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
            records, frame_callback = capture_callback(self.request)
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
            runtime_plan = _runtime_plan(self.request.plan, diffractometer, label_saturation(self.request))
            self._emit("geometry", 1, total)
            with _requalified_stitch_source(
                self.request.module.source,
                cancel_token=cancel_token,
            ) as source:
                oriented = _OrientedSource(
                    source,
                    orientation,
                    self.request.plan.max_frame_bytes,
                )

                def science_progress(done: int, _inner_total: int) -> None:
                    self._emit("science", 1 + done, total)

                screen = FrameScreen(self.request.plan.frame_screen, labels,
                                     _frame_screen_rows(self.request))
                scientific = run_stitch(
                    runtime_plan,
                    oriented,
                    frame_indices=labels,
                    progress_callback=science_progress,
                    cancel_token=cancel_token,
                    collect_frame_records=False,
                    frame_record_callback=frame_callback,
                    frame_screen=screen,
                )
                self._frame_screen = screen.summary()
                _record_frame_screen(records, screen.verdicts)
            _revalidate_geometry(self.request.plan.geometry)
            revalidate_mask(self.request.plan._mask_receipt)
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
        native_map = self.request.plan.native_records and self.request.plan.mode == "2d"
        profile = scientific.profile
        if native_map and (type(profile) is not tuple or len(profile) != 2
                           or type(profile[0]) is not IntegrationResult1D
                           or type(profile[1]) is not StitchDiagnostics):
            return self._terminal(ModuleDisposition.FAILED, "INVALID_SCIENCE_RESULT")
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
                **_native_artifact_args(self.request, records),
                gi_cuts=_gi_cuts(self.request.plan, scientific.gi_cuts),
                gi_map=_gi_map(scientific.gi_map),
                profile=None if not native_map else (
                    profile[0].unit, profile[0].radial, profile[0].intensity, profile[0].sigma,
                    profile[1].coverage, profile[1].normalization),
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
            if self.request.plan.mode == "1d" or native_map:
                _write_record(entry, self.request.module.output.target, result_projection,
                              _native_artifact_args(self.request, records))
            else:
                write_stitched(
                    entry,
                    result_projection=result_projection,
                    legacy_v1_unit_layout=False,
                    provenance=bound.provenance_json,
                    bounded_artifact=True,
                    **_native_artifact_args(self.request, records),
                )
            _revalidate_geometry(self.request.plan.geometry)

        def prepublish_check() -> None:
            try:
                _revalidate_geometry(self.request.plan.geometry)
                revalidate_mask(self.request.plan._mask_receipt)
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


#: What a source file's revision changes in a stored input manifest.
_REVISION_KEYS = frozenset({"fingerprint", "revision", "primary_revision",
                            "module_source_fingerprint", "source_fingerprint"})


def _without_revisions(value):
    if type(value) is dict:
        return {key: _without_revisions(item) for key, item in value.items() if key not in _REVISION_KEYS}
    if type(value) is list:
        return [_without_revisions(item) for item in value]
    return value


def stitch_output_current(request: StitchOperationRequest) -> bool:
    """Whether ``request``'s output already holds its stitch: the same settings (the plan) over the
    same frames (sources, scans, images and motor readings). A Run in Append mode skips such an output
    (maintainer, 2026-09-29: "skip if something has already been processed with the same settings as
    before, else overwrite").

    File revisions are left out, so a SPEC file that grew by later scans leaves an earlier scan's
    stitch standing; so does an image rewritten under the same name, which this does not detect. It
    reads the file's envelope only (``stitch_record_identity``): a processed file can always be stitched
    again, with Replace. A missing, unreadable or foreign file at the target is not current.

    The recorded threshold policy must also match: it holds each member's resolved automatic ceiling
    (decision 39), which the plan does not, so an output made under the data-type rule is redone."""
    target = Path(request.module.output.target)
    identity = stitch_record_identity(target) if target.is_file() else None
    if identity is None:
        return False
    plan_fingerprint, provenance = identity
    try:
        stored_provenance = json.loads(provenance)
        stored = stored_provenance["source"]["input_manifest"]
        planned = json.loads(json.dumps(request.manifest.to_provenance()))
        conditioning = (stored_provenance["plan"].get("conditioning"),
                        request.provenance["plan"].get("conditioning"))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
    return (plan_fingerprint == request.module.plan_fingerprint
            and _without_revisions(stored) == _without_revisions(planned)
            and conditioning[0] == conditioning[1])


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
    "stitch_output_current",
]
