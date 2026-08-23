from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
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

@dataclass(frozen=True, slots=True)
class SourceFileState:
    path: str
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    _resolved_path_at_capture: str | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise TypeError("source file path is invalid")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.size,
                self.mtime_ns,
                self.ctime_ns,
                self.device,
                self.inode,
            )
        ):
            raise TypeError("source file state is invalid")

    @classmethod
    def capture(cls, path: Path) -> "SourceFileState":
        selected = Path(os.path.abspath(path.expanduser()))
        resolved_before = selected.resolve(strict=True)
        stat = selected.stat()
        target_stat = resolved_before.stat()
        resolved_after = selected.resolve(strict=True)
        revision = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(stat.st_dev),
            int(stat.st_ino),
        )
        target_revision = (
            int(target_stat.st_size),
            int(target_stat.st_mtime_ns),
            int(target_stat.st_ctime_ns),
            int(target_stat.st_dev),
            int(target_stat.st_ino),
        )
        if (
            os.path.normcase(os.path.normpath(resolved_before))
            != os.path.normcase(os.path.normpath(resolved_after))
            or revision != target_revision
        ):
            raise OSError(f"source alias changed during capture: {selected}")
        state = cls(
            str(selected),
            *revision,
        )
        object.__setattr__(
            state,
            "_resolved_path_at_capture",
            str(resolved_before),
        )
        return state

    def matches_disk(self) -> bool:
        try:
            return self == type(self).capture(Path(self.path))
        except OSError:
            return False

    def as_dict(self) -> dict[str, int | str]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class ExternalSourceState:
    file: SourceFileState
    dataset: str
    first: int
    stop: int
    epoch: int

    def __post_init__(self) -> None:
        if (
            type(self.file) is not SourceFileState
            or type(self.dataset) is not str
            or not self.dataset
            or any(
                type(value) is not int or value < 0
                for value in (self.first, self.stop, self.epoch)
            )
            or self.stop <= self.first
        ):
            raise TypeError("external source state is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file.as_dict(),
            "dataset": self.dataset,
            "first": self.first,
            "stop": self.stop,
            "epoch": self.epoch,
        }


@dataclass(frozen=True, slots=True)
class AdmittedMotorValue:
    """One exact per-image motor value owned by TIFF admission."""

    source_path: str
    motor: str
    value: float

    def __post_init__(self) -> None:
        if (
            type(self.source_path) is not str
            or not os.path.isabs(self.source_path)
            or type(self.motor) is not str
            or not self.motor
            or self.motor == "Manual"
            or type(self.value) is not float
            or not math.isfinite(self.value)
        ):
            raise TypeError("admitted motor value is invalid")

    def as_dict(self) -> dict[str, str | float]:
        return {
            "source_path": self.source_path,
            "motor": self.motor,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class AdmittedMetadataSource:
    """The exact nullable metadata input aligned with one TIFF member."""

    source_path: str
    metadata_file: SourceFileState | None

    def __post_init__(self) -> None:
        if (
            type(self.source_path) is not str
            or not os.path.isabs(self.source_path)
            or (
                self.metadata_file is not None
                and type(self.metadata_file) is not SourceFileState
            )
        ):
            raise TypeError("admitted metadata source is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "metadata_file": (
                None
                if self.metadata_file is None
                else self.metadata_file.as_dict()
            ),
        }


_SOURCE_ROLE_ORDER = (
    "source_file",
    "source_member",
    "external_member",
    "detector_dependency",
    "image_metadata",
)


@dataclass(frozen=True, slots=True)
class SourceAliasBinding:
    """One admitted lexical source name bound to one canonical target."""

    raw_path: str
    resolved_path: str
    target_id: int
    roles: tuple[str, ...]
    candidate_owner_id: str | None = None

    def __post_init__(self) -> None:
        order = {
            value: index for index, value in enumerate(_SOURCE_ROLE_ORDER)
        }
        if (
            type(self.raw_path) is not str
            or not os.path.isabs(self.raw_path)
            or type(self.resolved_path) is not str
            or not os.path.isabs(self.resolved_path)
            or type(self.target_id) is not int
            or self.target_id < 0
            or type(self.roles) is not tuple
            or not self.roles
            or not all(value in order for value in self.roles)
            or tuple(sorted(self.roles, key=order.__getitem__)) != self.roles
            or len(set(self.roles)) != len(self.roles)
            or (
                self.candidate_owner_id is not None
                and (
                    type(self.candidate_owner_id) is not str
                    or not self.candidate_owner_id
                )
            )
        ):
            raise TypeError("source alias binding is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_path": self.raw_path,
            "resolved_path": self.resolved_path,
            "target_id": self.target_id,
            "roles": list(self.roles),
            "candidate_owner_id": self.candidate_owner_id,
        }


@dataclass(frozen=True, slots=True)
class CanonicalSourceTarget:
    """One resolved pathname and its exact admitted target revision."""

    target_id: int
    resolved_path: str
    state: SourceFileState
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        order = {
            value: index for index, value in enumerate(_SOURCE_ROLE_ORDER)
        }
        if (
            type(self.target_id) is not int
            or self.target_id < 0
            or type(self.resolved_path) is not str
            or not os.path.isabs(self.resolved_path)
            or type(self.state) is not SourceFileState
            or self.state.path != self.resolved_path
            or type(self.roles) is not tuple
            or not self.roles
            or not all(value in order for value in self.roles)
            or tuple(sorted(self.roles, key=order.__getitem__)) != self.roles
            or len(set(self.roles)) != len(self.roles)
        ):
            raise TypeError("canonical source target is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "resolved_path": self.resolved_path,
            "state": self.state.as_dict(),
            "roles": list(self.roles),
        }


@dataclass(frozen=True, slots=True)
class SourceExecutionIdentityV1:
    """Resolved source topology used without changing legacy stamp equality."""

    aliases: tuple[SourceAliasBinding, ...]
    targets: tuple[CanonicalSourceTarget, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or type(self.aliases) is not tuple
            or not self.aliases
            or not all(
                type(value) is SourceAliasBinding for value in self.aliases
            )
            or type(self.targets) is not tuple
            or not self.targets
            or not all(
                type(value) is CanonicalSourceTarget for value in self.targets
            )
            or tuple(value.target_id for value in self.targets)
            != tuple(range(len(self.targets)))
            or any(value.target_id >= len(self.targets) for value in self.aliases)
        ):
            raise TypeError("source execution identity is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "aliases": [value.as_dict() for value in self.aliases],
            "targets": [value.as_dict() for value in self.targets],
        }


def _absolute_source_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.fsdecode(path)))


def _source_path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _state_resolved_source_path(state: SourceFileState) -> str:
    captured = state._resolved_path_at_capture
    return (
        _absolute_source_path(state.path)
        if captured is None
        else captured
    )


def _same_source_revision(
    left: SourceFileState,
    right: SourceFileState,
) -> bool:
    return (
        left.size,
        left.mtime_ns,
        left.ctime_ns,
        left.device,
        left.inode,
    ) == (
        right.size,
        right.mtime_ns,
        right.ctime_ns,
        right.device,
        right.inode,
    )


@dataclass(frozen=True, slots=True)
class SourceExecutionStamp:
    file: SourceFileState
    adapter_id: str
    frame_count: int
    first_label: int
    members: tuple[SourceFileState, ...] = ()
    external_members: tuple[ExternalSourceState, ...] = ()
    dependency_files: tuple[SourceFileState, ...] = ()
    admitted_motor_values: tuple[AdmittedMotorValue, ...] = ()
    metadata_sources: tuple[AdmittedMetadataSource, ...] = ()
    source_aliases: tuple[SourceAliasBinding, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    canonical_targets: tuple[CanonicalSourceTarget, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    execution_identity_v1: SourceExecutionIdentityV1 = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.file) is not SourceFileState or not self.adapter_id:
            raise ValueError("source stamp identity is empty")
        if any(type(value) is not int or value < 0 for value in (
            self.frame_count, self.first_label,
        )):
            raise ValueError("source stamp counts are invalid")
        if (
            type(self.members) is not tuple
            or not all(type(value) is SourceFileState for value in self.members)
            or type(self.external_members) is not tuple
            or not all(
                type(value) is ExternalSourceState
                for value in self.external_members
            )
            or type(self.dependency_files) is not tuple
            or not all(
                type(value) is SourceFileState
                for value in self.dependency_files
            )
            or len({value.path for value in self.dependency_files})
            != len(self.dependency_files)
            or type(self.admitted_motor_values) is not tuple
            or not all(
                type(value) is AdmittedMotorValue
                for value in self.admitted_motor_values
            )
            or type(self.metadata_sources) is not tuple
            or not all(
                type(value) is AdmittedMetadataSource
                for value in self.metadata_sources
            )
        ):
            raise TypeError("source stamp members are invalid")
        values = self.admitted_motor_values
        if values:
            if (
                not self.members
                or tuple(value.source_path for value in values)
                != tuple(member.path for member in self.members)
                or len({value.motor for value in values}) != 1
            ):
                raise ValueError(
                    "admitted motor values must align with every source member"
                )
        metadata = self.metadata_sources
        if metadata and (
            self.adapter_id != "tiff_series"
            or not self.members
            or tuple(value.source_path for value in metadata)
            != tuple(member.path for member in self.members)
        ):
            raise ValueError(
                "metadata sources must align with every TIFF member"
            )
        expected_owner = (
            "image_file" if self.adapter_id == "tiff_series"
            else self.adapter_id
        )
        occurrences = (
            ((self.file, "source_file", expected_owner),)
            + tuple(
                (value, "source_member", expected_owner)
                for value in self.members
            )
            + tuple(
                (value.file, "external_member", None)
                for value in self.external_members
            )
            + tuple(
                (value, "detector_dependency", None)
                for value in self.dependency_files
            )
            + tuple(
                (value.metadata_file, "image_metadata", None)
                for value in self.metadata_sources
                if value.metadata_file is not None
            )
        )
        raw_records: dict[str, dict[str, Any]] = {}
        raw_order: list[str] = []
        target_records: dict[str, dict[str, Any]] = {}
        target_order: list[str] = []
        for raw_state, role, owner_id in occurrences:
            raw_path = _absolute_source_path(raw_state.path)
            raw_key = _source_path_key(raw_path)
            resolved_path = _state_resolved_source_path(raw_state)
            resolved_key = _source_path_key(resolved_path)
            target_state = SourceFileState(
                resolved_path,
                raw_state.size,
                raw_state.mtime_ns,
                raw_state.ctime_ns,
                raw_state.device,
                raw_state.inode,
            )
            raw_record = raw_records.get(raw_key)
            if raw_record is not None and (
                raw_record["raw_path"] != raw_path
                or raw_record["resolved_key"] != resolved_key
            ):
                raise ValueError(
                    f"conflicting source alias binding for {raw_path}"
                )
            if raw_record is None:
                raw_record = {
                    "raw_path": raw_path,
                    "resolved_path": resolved_path,
                    "resolved_key": resolved_key,
                    "roles": set(),
                    "owner_id": owner_id,
                }
                raw_records[raw_key] = raw_record
                raw_order.append(raw_key)
            elif (
                owner_id is not None
                and raw_record["owner_id"] is not None
                and raw_record["owner_id"] != owner_id
            ):
                raise ValueError(
                    f"conflicting source candidate owners for {raw_path}"
                )
            elif raw_record["owner_id"] is None:
                raw_record["owner_id"] = owner_id
            raw_record["roles"].add(role)

            target_record = target_records.get(resolved_key)
            if target_record is not None and not _same_source_revision(
                target_record["state"],
                target_state,
            ):
                raise ValueError(
                    "conflicting canonical source states for "
                    f"{resolved_path}"
                )
            if target_record is None:
                target_record = {
                    "resolved_path": resolved_path,
                    "state": target_state,
                    "roles": set(),
                }
                target_records[resolved_key] = target_record
                target_order.append(resolved_key)
            target_record["roles"].add(role)
        role_order = {
            value: index
            for index, value in enumerate(_SOURCE_ROLE_ORDER)
        }
        target_ids = {
            key: index for index, key in enumerate(target_order)
        }
        targets = tuple(
            CanonicalSourceTarget(
                target_ids[key],
                target_records[key]["resolved_path"],
                target_records[key]["state"],
                tuple(sorted(
                    target_records[key]["roles"],
                    key=role_order.__getitem__,
                )),
            )
            for key in target_order
        )
        aliases = tuple(
            SourceAliasBinding(
                raw_records[key]["raw_path"],
                raw_records[key]["resolved_path"],
                target_ids[raw_records[key]["resolved_key"]],
                tuple(sorted(
                    raw_records[key]["roles"],
                    key=role_order.__getitem__,
                )),
                raw_records[key]["owner_id"],
            )
            for key in raw_order
        )
        object.__setattr__(
            self,
            "source_aliases",
            aliases,
        )
        object.__setattr__(
            self,
            "canonical_targets",
            targets,
        )
        object.__setattr__(
            self,
            "execution_identity_v1",
            SourceExecutionIdentityV1(aliases, targets),
        )

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def size(self) -> int:
        return self.file.size

    @property
    def mtime_ns(self) -> int:
        return self.file.mtime_ns

    @property
    def member_stamps(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (value.path, value.size, value.mtime_ns)
            for value in self.members
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.file.as_dict(),
            "adapter_id": self.adapter_id,
            "frame_count": self.frame_count,
            "first_label": self.first_label,
            "member_stamps": [
                value.as_dict() for value in self.members
            ],
            "external_members": [
                value.as_dict() for value in self.external_members
            ],
            "dependency_files": [
                value.as_dict() for value in self.dependency_files
            ],
            "admitted_motor_values": [
                value.as_dict() for value in self.admitted_motor_values
            ],
            "metadata_sources": [
                value.as_dict() for value in self.metadata_sources
            ],
        }

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

    @property
    def poni(self) -> object | None:
        from xrd_tools.core.containers import PONI
        return None if self.poni_values is None else PONI(*self.poni_values)

    @property
    def detector_calibration(self) -> object | None:
        from xrd_tools.core.geometry.diffractometer import DetectorCalibration
        return None if self.poni_values is None else DetectorCalibration(
            self.poni, json.loads(self.poni_detector_config_json),
        )

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

def threshold_pair_is_canonical(threshold: object) -> bool:
    """LV-UI-11 canonical run identity: exactly ONE of the two exclusive
    threshold facts is active (``apply_threshold != mask_saturation``).

    The vNext start capture canonicalizes a degenerate (equal) pair through
    the intent store BEFORE any admission signature, frozen configuration,
    fingerprint, or provenance is derived; every later boundary REFUSES a
    degenerate pair instead of silently reinterpreting it (Codex P1,
    2026-08-04: execution-vs-identity split).  Accepts any carrier with the
    two attributes (mutable ``ThresholdIntent`` or ``FrozenThresholdPolicy``).
    """
    return bool(threshold.apply_threshold) != bool(threshold.mask_saturation)


__all__ = [
    "threshold_pair_is_canonical",
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
