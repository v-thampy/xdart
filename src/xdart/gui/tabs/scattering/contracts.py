from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.run_configuration import FrozenRunConfiguration
from xrd_tools.sources.descriptor import ContainerDescriptor
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.selection import DirectorySourceSpec

from .events import (
    CleanupStatus,
    DetachedDiagnostic,
    DurablePaused,
    ExecutorAccepted,
    ExecutorClosed,
    ExecutorStartFailed,
    RequestId,
    RunIdentity,
)
from .display_retirement import (
    DisplayRetirementReceipt,
    NO_DISPLAY_RETIREMENT,
)
from .display_values import StandardRunEvent

SourceSelection: TypeAlias = SourceSpec | DirectorySourceSpec

class OutputDisposition(str, Enum):
    WRITE = "write"

from xrd_tools.sources.execution_graph import (
    AdmittedMetadataSource,
    AdmittedMotorValue,
    CanonicalSourceTarget,
    ExternalSourceState,
    SourceAliasBinding,
    SourceExecutionIdentityV1,
    SourceExecutionStamp,
    SourceFileState,
)

@dataclass(frozen=True, slots=True)
class SourceGroupIdentity:
    member_paths: tuple[Path, ...]
    adapter_key: str
    group_key: str
    target: Path
    source_stamp: SourceExecutionStamp
    motor_names: tuple[str, ...] | None

    def __post_init__(self) -> None:
        motors = self.motor_names
        if not (
            type(self.member_paths) is tuple and self.member_paths
            and all(isinstance(value, Path) for value in self.member_paths)
            and type(self.adapter_key) is str and self.adapter_key
            and type(self.group_key) is str and self.group_key
            and isinstance(self.target, Path)
            and type(self.source_stamp) is SourceExecutionStamp
            and (motors is None or type(motors) is tuple
                 and all(type(value) is str for value in motors))
        ):
            raise TypeError("source group identity is invalid")
        members = tuple(path.resolve(strict=False) for path in self.member_paths)
        object.__setattr__(self, "member_paths", members)
        object.__setattr__(self, "target", self.target.resolve(strict=False))

    @property
    def key(self) -> tuple[str, str, Path]:
        return self.adapter_key, self.group_key, self.target

@dataclass(frozen=True, slots=True)
class PlannedOutput:
    source_spec: SourceSpec
    source_path: Path
    target: Path
    source_stamp: SourceExecutionStamp
    candidate: Candidate | None = None
    descriptor: ContainerDescriptor | None = None
    motor_names: tuple[str, ...] | None = None
    #: ROOT FAMILY this run publishes into, carried from the moment the target
    #: was named.  Recorded rather than re-derived: the public name is
    #: `<family><slot>.nexus`, so recovering the family from `target` would mean
    #: stripping the slot, which is an explicit stop condition.
    artifact_family: str = ""
    group: SourceGroupIdentity = field(init=False)

    def __post_init__(self) -> None:
        members = tuple(Path(value.path) for value in self.source_stamp.members)
        members = members or (self.source_path,)
        members += tuple(
            Path(value.file.path)
            for value in self.source_stamp.external_members
        )
        members += tuple(
            Path(value.path)
            for value in self.source_stamp.dependency_files
        )
        members = tuple(dict.fromkeys(members))
        options = dict(self.source_spec.options)
        key = str(
            options.get("scan_name")
            or getattr(self.descriptor, "scan_name", "")
            or Path(self.source_stamp.path).stem
        )
        motors = self.motor_names
        if motors is None and self.descriptor is not None:
            motors = self.descriptor.motor_names
        if self.source_spec.kind is SourceKind.TIFF_SERIES:
            raw_admitted = options.get("admitted_motor_values", ())
            try:
                serialized_admitted = tuple(
                    tuple(value) for value in raw_admitted
                )
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "TIFF execution motor values are malformed"
                ) from error
            stamped_admitted = tuple(
                (value.source_path, value.motor, value.value)
                for value in self.source_stamp.admitted_motor_values
            )
            if serialized_admitted != stamped_admitted:
                raise ValueError(
                    "TIFF execution motor values do not match source stamp"
                )
            if stamped_admitted and (
                motors is None
                or stamped_admitted[0][1] not in motors
            ):
                raise ValueError(
                    "admitted TIFF motor is absent from motor catalog"
                )
        group = SourceGroupIdentity(
            members, self.source_stamp.adapter_id, key, self.target,
            self.source_stamp, motors,
        )
        object.__setattr__(self, "group", group)

@dataclass(frozen=True, slots=True)
class OutputFact:
    target_state: tuple[int, int, int, int, int] | bool

@dataclass(frozen=True, slots=True)
class AdmittedOutput:
    item: PlannedOutput
    disposition: OutputDisposition
    labels: tuple[int, ...]
    fact: OutputFact
    reason: str = ""
    background_bindings: tuple[tuple[int, tuple[object, ...], bytes, str], ...] = ()

    def __post_init__(self) -> None:
        labels, state = self.labels, self.fact.target_state
        bindings = self.background_bindings
        if not (type(self.item) is PlannedOutput
                and type(self.disposition) is OutputDisposition
                and type(labels) is tuple and labels == tuple(sorted(set(labels)))
                and all(type(label) is int and label >= 0 for label in labels)
                and type(self.fact) is OutputFact
                and (type(state) is bool or type(state) is tuple
                     and len(state) == 5
                     and all(type(value) is int and value >= 0 for value in state))
                and type(self.reason) is str
                and type(bindings) is tuple
                and all(type(value) is tuple and len(value) == 4 and type(value[0]) is int
                        and type(value[1]) is tuple and len(value[1]) == 6
                        and type(value[2]) is bytes and 0 < len(value[2]) <= 262_144
                        and type(value[3]) is str and hashlib.sha256(value[2]).hexdigest() == value[3]
                        for value in bindings)
                and tuple(value[0] for value in bindings) == tuple(sorted(set(value[0] for value in bindings)))):
            raise TypeError("admitted output is invalid")

@dataclass(frozen=True, slots=True)
class AcceptedScientificAssets:
    poni_values: tuple[float, float, float, float, float, float, float, str] | None
    mask_dtype: str | None
    mask_shape: tuple[int, ...] | None
    mask_bytes: bytes | None
    poni_sha256: str | None
    mask_sha256: str | None
    poni_detector_config_json: str | None = None
    poni_parallax: bool | None = None

    def __post_init__(self) -> None:
        values = self.poni_values
        if values is not None and not (
                type(values) is tuple and len(values) == 8
                and all(type(value) is float and math.isfinite(value) for value in values[:7])
                and values[0] > 0 and values[6] >= 0 and type(values[7]) is str):
            raise TypeError("accepted PONI is invalid")
        parts = self.mask_dtype, self.mask_shape, self.mask_bytes
        if any(value is not None for value in parts) and not (
                type(self.mask_dtype) is str and type(self.mask_shape) is tuple
                and all(type(value) is int and value >= 0 for value in self.mask_shape)
                and type(self.mask_bytes) is bytes):
            raise TypeError("accepted detector mask is invalid")
        if not all(value is None or type(value) is str
                   for value in (self.poni_sha256, self.mask_sha256)):
            raise TypeError("accepted scientific assets are invalid")
        config_text = self.poni_detector_config_json
        if (values is None) != (config_text is None):
            raise TypeError("accepted PONI and detector config must be paired")
        if self.poni_parallax is not None and type(self.poni_parallax) is not bool:
            raise TypeError("accepted PONI parallax is invalid")
        if values is None and self.poni_parallax is not None:
            raise TypeError("accepted PONI parallax has no calibration")
        if config_text is not None:
            try:
                config = json.loads(config_text)
                canonical = json.dumps(
                    config, sort_keys=True, separators=(",", ":"), allow_nan=False,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TypeError("accepted detector config is invalid") from exc
            orientation = config.get("orientation") if type(config) is dict else None
            if (canonical != config_text or type(orientation) is not int
                    or orientation not in range(1, 5)):
                raise TypeError("accepted detector config is not canonical")
            sensor = config.get("sensor")
            if self.poni_parallax is None:
                if sensor is not None:
                    raise TypeError("legacy accepted PONI cannot contain a sensor")
            else:
                try:
                    from xrd_tools.integrate.calibration import (
                        validate_sensor_parallax,
                    )
                    validate_sensor_parallax(
                        sensor.get("material") if type(sensor) is dict else None,
                        sensor.get("thickness") if type(sensor) is dict else None,
                        self.poni_parallax,
                        wavelength_m=values[6],
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "accepted PONI sensor/parallax is invalid"
                    ) from exc

    @property
    def poni(self) -> object | None:
        from xrd_tools.core.containers import PONI
        return None if self.poni_values is None else PONI(*self.poni_values)

    @property
    def detector_calibration(self) -> object | None:
        if self.poni_values is None:
            return None
        from xrd_tools.integrate.calibration import (
            detector_calibration_from_projection,
        )

        projection = dict(self.poni.to_dict())
        config = json.loads(self.poni_detector_config_json)
        if self.poni_parallax is not None:
            projection["detector_config"] = config
            projection["parallax"] = self.poni_parallax
        return detector_calibration_from_projection(
            projection,
            detector_config=config,
        )

    @property
    def poni_projection(self) -> dict[str, object] | None:
        if self.poni_values is None:
            return None
        from xrd_tools.integrate.calibration import (
            detector_calibration_projection,
        )
        return detector_calibration_projection(self.detector_calibration)

    @property
    def mask(self) -> np.ndarray | None:
        return None if self.mask_bytes is None else np.frombuffer(
            self.mask_bytes, dtype=np.dtype(self.mask_dtype)).reshape(
                self.mask_shape).copy()

def _copy_source(value: SourceSelection) -> SourceSelection:
    if type(value) is SourceSpec:
        return SourceSpec(
            uri=value.uri,
            kind=value.kind,
            metadata_uri=value.metadata_uri,
            entry=value.entry,
            options=copy.deepcopy(dict(value.options)),
        )
    if type(value) is DirectorySourceSpec:
        return DirectorySourceSpec(
            root=value.root,
            recursive=value.recursive,
            suffixes=value.suffixes,
            name_filter=value.name_filter,
            generation=value.generation,
            metadata_format=value.metadata_format,
        )
    raise TypeError("source must be SourceSpec or DirectorySourceSpec")

@dataclass(frozen=True, slots=True)
class SourceCapture:
    request_id: RequestId
    source_epoch: int
    source: SourceSelection
    gi_motor_choices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.source_epoch) is not int or self.source_epoch < 0:
            raise ValueError("source_epoch cannot be negative")

@dataclass(frozen=True, slots=True)
class StartCapture:
    request_id: RequestId
    capture_sequence: int
    intent_snapshot: RunIntentSnapshot
    source_capture: SourceCapture

@dataclass(frozen=True, slots=True)
class AdmissionToken:
    request_id: RequestId
    revision: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not RequestId:
            raise TypeError("admission token request is invalid")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("admission token revision is invalid")

    def qualifies(self, capture: object) -> bool:
        return (
            type(capture) is StartCapture
            and self.request_id is capture.request_id
            and self.revision == capture.intent_snapshot.revision
        )

@dataclass(frozen=True, slots=True)
class AdmissionFailure:
    token: AdmissionToken
    reason: str

    def __post_init__(self) -> None:
        if type(self.token) is not AdmissionToken or type(self.reason) is not str:
            raise TypeError("admission failure values are invalid")

@dataclass(frozen=True, slots=True)
class AdmissionReleased:
    token: AdmissionToken
    cleanup_status: CleanupStatus
    cleanup_failures: tuple[DetachedDiagnostic, ...] = ()
    display_retirement: DisplayRetirementReceipt = (
        NO_DISPLAY_RETIREMENT
    )

    def __post_init__(self) -> None:
        if not (
            type(self.token) is AdmissionToken
            and type(self.cleanup_status) is CleanupStatus
            and type(self.cleanup_failures) is tuple
            and all(type(value) is DetachedDiagnostic
                    for value in self.cleanup_failures)
            and type(self.display_retirement)
            is DisplayRetirementReceipt
        ):
            raise TypeError("admission cleanup values are invalid")

@dataclass(frozen=True, slots=True)
class AdmissionReceipt:
    request_id: RequestId
    revision: int
    source_capture: SourceCapture
    candidate: object
    outputs: tuple[object, ...]
    scientific_assets: object
    gi_motor_choices: tuple[str, ...] | None = None
    display_retirement: DisplayRetirementReceipt = (
        NO_DISPLAY_RETIREMENT
    )
    deferred_directory: object | None = None
    directory_discovered_file_count: int = 0
    directory_discovered_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        from .output_preflight import (
            DeferredDirectoryPlan,
            OutputCandidate,
        )

        choices = self.gi_motor_choices
        valid = (type(self.request_id) is RequestId and type(self.revision) is int
                 and self.revision >= 0 and type(self.source_capture) is SourceCapture
                 and self.source_capture.request_id is self.request_id
                 and type(self.candidate) is OutputCandidate and type(self.outputs) is tuple
                 and all(type(item) is AdmittedOutput for item in self.outputs)
                 and type(self.scientific_assets) is AcceptedScientificAssets
                 and (
                     self.deferred_directory is None
                     or type(self.deferred_directory)
                     is DeferredDirectoryPlan
                 )
                 and not (self.outputs and self.deferred_directory is not None)
                 and type(self.directory_discovered_file_count) is int
                 and self.directory_discovered_file_count >= 0
                 and type(self.directory_discovered_paths) is tuple
                 and all(
                     isinstance(value, Path)
                     and value.is_absolute()
                     for value in self.directory_discovered_paths
                 )
                 and len(set(self.directory_discovered_paths))
                 == len(self.directory_discovered_paths)
                 and self.directory_discovered_file_count
                 == len(self.directory_discovered_paths)
                 and (choices is None or type(choices) is tuple
                      and all(type(value) is str for value in choices))
                 and type(self.display_retirement)
                 is DisplayRetirementReceipt
                 and self.display_retirement.cleanup_status
                 is CleanupStatus.CLEANED)
        if not valid:
            raise TypeError("admission values are invalid")

    def qualifies(self, capture: object) -> bool:
        return (
            AdmissionToken(self.request_id, self.revision).qualifies(capture)
            and self.source_capture is capture.source_capture
        )

def admitted_capture(admission: object, capture: object) -> object | None:
    return capture if (
        type(admission) in {AdmissionReceipt, AdmissionToken}
        and admission.qualifies(capture)
    ) else None

def executor_start_inputs_are_valid(
    configuration: object, source: object, identity: object,
) -> bool:
    try:
        return (
            type(source) is SourceCapture
            and type(identity) is RunIdentity
            and type(configuration) is FrozenRunConfiguration
            and identity.generation == configuration.generation
            and identity.fingerprint == configuration.fingerprint
            and source.source == configuration.thaw_source_spec()
            and bool(configuration.poni_file)
            and bool(configuration.save_path)
        )
    except Exception:
        return False

class SourceObservationStatus(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class SourceCountScope(str, Enum):
    DIRECT_ONLY = "direct_only"
    SELECTED_PLUS_IMMEDIATE = "selected_plus_immediate"

@dataclass(frozen=True, slots=True)
class SourceObservationRequest:
    observation_id: int
    intent_revision: int
    source: SourceSelection

    def __post_init__(self) -> None:
        if type(self.observation_id) is not int or self.observation_id <= 0:
            raise ValueError("observation_id must be a positive integer")
        if type(self.intent_revision) is not int or self.intent_revision < 0:
            raise ValueError("intent_revision must be a non-negative integer")
        object.__setattr__(self, "source", _copy_source(self.source))

@dataclass(frozen=True, slots=True)
class SourceObservation:
    observation_id: int
    intent_revision: int
    source: SourceSelection
    status: SourceObservationStatus
    selected_name: str
    exists: bool
    is_directory: bool
    size_bytes: int | None = None
    mtime_ns: int | None = None
    direct_child_count: int | None = None
    subdirectories_deferred: bool = False
    reason: str = ""
    gi_motor_choices: tuple[str, ...] | None = None
    candidate_fingerprint: str = ""
    one_level_file_count: int | None = None
    file_count_scope: SourceCountScope = SourceCountScope.DIRECT_ONLY

    def __post_init__(self) -> None:
        if type(self.observation_id) is not int or self.observation_id <= 0:
            raise ValueError("observation_id must be a positive integer")
        if type(self.intent_revision) is not int or self.intent_revision < 0:
            raise ValueError("intent_revision must be a non-negative integer")
        if type(self.status) is not SourceObservationStatus:
            raise TypeError("status must be SourceObservationStatus")
        if type(self.file_count_scope) is not SourceCountScope:
            raise TypeError("file_count_scope must be SourceCountScope")
        for value in (self.exists, self.is_directory, self.subdirectories_deferred):
            if type(value) is not bool:
                raise TypeError("observation flags must be booleans")
        for value in (
            self.size_bytes,
            self.mtime_ns,
            self.direct_child_count,
            self.one_level_file_count,
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("observation sizes and counts must be non-negative integers")
        if (
            type(self.selected_name) is not str
            or type(self.reason) is not str
            or type(self.candidate_fingerprint) is not str
        ):
            raise TypeError("observation text must be strings")
        if self.gi_motor_choices is not None and (
            type(self.gi_motor_choices) is not tuple
            or not all(type(value) is str for value in self.gi_motor_choices)
        ):
            raise TypeError("observed GI motor choices are invalid")
        if self.file_count_scope is SourceCountScope.SELECTED_PLUS_IMMEDIATE:
            if (
                self.one_level_file_count is None
                or self.direct_child_count is None
                or self.one_level_file_count < self.direct_child_count
            ):
                raise ValueError(
                    "one-level count must include every direct matching file"
                )
        elif self.one_level_file_count is not None:
            raise ValueError(
                "one-level count requires selected-plus-immediate scope"
            )
        object.__setattr__(self, "source", _copy_source(self.source))

    @property
    def observed_file_count(self) -> int | None:
        """Return the bounded count intended for passive display only.

        Direct observations qualify the direct-child fingerprint.  Recursive
        directory observations may display a bounded selected-folder plus
        one-level count without opening file content.  Recursive TIFF also
        widens its fingerprint to that shallow universe because its bounded
        motor preview consumes the same members; container preview retains
        its established direct-only fingerprint fallback.
        """

        return (
            self.one_level_file_count
            if self.one_level_file_count is not None
            else self.direct_child_count
        )

    def qualifies(self, request: SourceObservationRequest,
                  candidate_fingerprint: str | None = None) -> bool:
        return (
            type(request) is SourceObservationRequest
            and self.observation_id == request.observation_id
            and self.intent_revision == request.intent_revision
            and self.source == request.source
            and (
                candidate_fingerprint is None
                or self.candidate_fingerprint == candidate_fingerprint
            )
        )

@runtime_checkable
class SourcePort(Protocol):
    def capture(
        self,
        source: SourceSelection,
        request_id: RequestId,
    ) -> SourceCapture: ...

    def cancel(self, request_id: RequestId) -> None: ...

    def observe(self, request: SourceObservationRequest) -> SourceObservation: ...

    def preview_motors(
        self, request: SourceObservationRequest,
    ) -> SourceObservation: ...

    def publish_motor_knowledge(
        self, observation: SourceObservation,
    ) -> None: ...

    def project_motor_knowledge(
        self, source: SourceSelection,
        candidate_fingerprint: str | None = None,
    ) -> SourceObservation | None: ...

    def cancel_observation(self, observation_id: int) -> None: ...

@runtime_checkable
class RunExecutorPort(Protocol):
    def begin_admission(self, capture: StartCapture) -> AdmissionToken: ...

    def poll_admission(
        self, token: AdmissionToken,
    ) -> AdmissionReceipt | AdmissionFailure | None: ...

    def cancel_admission(self, token: AdmissionToken) -> AdmissionReleased: ...

    def release_admission(self, token: AdmissionToken) -> AdmissionReleased: ...

    def start(
        self,
        configuration: FrozenRunConfiguration,
        source: SourceCapture,
        run_identity: RunIdentity,
        admission: AdmissionReceipt,
    ) -> ExecutorAccepted | ExecutorStartFailed: ...

    def stop(self, run_identity: RunIdentity) -> None: ...

    def pause(self, run_identity: RunIdentity) -> DurablePaused: ...

    def resume(self, run_identity: RunIdentity) -> None: ...

    def acquisition_context(
        self, run_identity: RunIdentity,
    ) -> object | None: ...

    def close(self, run_identity: RunIdentity) -> ExecutorClosed: ...

    def drain_events(self) -> tuple[StandardRunEvent, ...]: ...

__all__ = [
    "RunExecutorPort",
    "AdmissionFailure",
    "AdmissionReceipt",
    "AdmissionToken",
    "DisplayRetirementReceipt",
    "admitted_capture",
    "executor_start_inputs_are_valid",
    "AdmittedOutput",
    "OutputDisposition",
    "OutputFact",
    "PlannedOutput",
    "AcceptedScientificAssets",
    "AdmittedMetadataSource",
    "AdmittedMotorValue",
    "CanonicalSourceTarget",
    "ExternalSourceState",
    "SourceAliasBinding",
    "SourceExecutionIdentityV1",
    "SourceFileState",
    "SourceExecutionStamp",
    "SourceCapture",
    "StartCapture",
    "SourceObservation",
    "SourceObservationRequest",
    "SourceObservationStatus",
    "SourceCountScope",
    "SourcePort",
    "SourceSelection",
]
