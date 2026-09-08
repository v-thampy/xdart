from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib, json, math, os, posixpath
from pathlib import Path
import threading, time
from typing import Any, Callable
import h5py
import numpy as np
_HDF5_FILE_OPEN = h5py.File
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.append import (
    AppendExternalMember, AppendImageMember, AppendSource, science_fingerprint,
)
from xrd_tools.io.image import read_detector_image_layout
from xrd_tools.sources.descriptor import ContainerDescriptor, describe_container_from_open
class SourceRevisionChanged(ValueError): pass


class SourceCleanupPending(RuntimeError):
    """One exact source owner needs another creating-thread close command."""

    def __init__(self, owner: object, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.owner = owner
        self.diagnostic = diagnostic


class SourceCleanupFailed(RuntimeError):
    """A source sub-owner finalized but reported a non-retryable failure."""


class _QualificationCleanupOwner:
    """Retain one qualification handle graph across explicit close commands."""

    def __init__(self, handle: Any, slot: Any, completion: Callable[[], Any]) -> None:
        self._handle = handle
        self._slot = slot
        self._completion = completion
        self._completed = False
        self._value = None
        self._error: BaseException | None = None
        self._thread = threading.get_ident()

    @property
    def closed(self) -> bool:
        owner = self._slot.owner
        return (
            (owner is None or owner.state == "CLOSED")
            and not self._handle.id.valid
        )

    def close(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("qualification cleanup is creating-thread affine")
        owner = self._slot.owner
        if owner is not None and owner.state != "CLOSED":
            try:
                owner.close()
            except BaseException as error:
                if owner.state != "CLOSED":
                    raise SourceCleanupPending(
                        self, f"AVERAGE_QUALIFICATION_CLEANUP_PENDING: {error}",
                    ) from error
        if owner is not None and owner.state == "CLOSED":
            self._slot.owner = None
        if self._handle.id.valid:
            try:
                self._handle.close()
            except BaseException as error:
                if self._handle.id.valid:
                    raise SourceCleanupPending(
                        self, f"AVERAGE_QUALIFICATION_CLEANUP_PENDING: {error}",
                    ) from error
        if not self.closed:
            raise SourceCleanupPending(
                self, "AVERAGE_QUALIFICATION_CLEANUP_PENDING",
            )

    def take(self) -> Any:
        """Complete once after the exact retained handle graph is closed."""
        if threading.get_ident() != self._thread:
            raise RuntimeError("qualification completion is creating-thread affine")
        if not self.closed:
            raise SourceCleanupPending(
                self, "AVERAGE_QUALIFICATION_CLEANUP_PENDING",
            )
        if not self._completed:
            completion, self._completion = self._completion, None
            try:
                self._value = completion()
            except BaseException as error:
                self._error = error
            finally:
                self._completed = True
        if self._error is not None:
            raise self._error
        return self._value
@dataclass(frozen=True, slots=True)
class SourceFileState:
    path: str; size: int; mtime_ns: int; ctime_ns: int; device: int; inode: int
    _resolved_path_at_capture: str | None = field(init=False, repr=False, compare=False, default=None)
    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise TypeError("source file path is invalid")
        if any(type(value) is not int or value < 0 for value in (
            self.size, self.mtime_ns, self.ctime_ns, self.device, self.inode,
        )):
            raise TypeError("source file state is invalid")
    @classmethod
    def capture(cls, path: Path) -> "SourceFileState":
        selected = Path(os.path.abspath(path.expanduser()))
        resolved_before = selected.resolve(strict=True)
        stat = selected.stat(); target_stat = resolved_before.stat()
        resolved_after = selected.resolve(strict=True)
        revision = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns),
                    int(stat.st_dev), int(stat.st_ino))
        target_revision = (int(target_stat.st_size), int(target_stat.st_mtime_ns),
                           int(target_stat.st_ctime_ns), int(target_stat.st_dev),
                           int(target_stat.st_ino))
        if (os.path.normcase(os.path.normpath(resolved_before))
                != os.path.normcase(os.path.normpath(resolved_after))
                or revision != target_revision):
            raise OSError(f"source alias changed during capture: {selected}")
        value = cls(str(selected), *revision)
        object.__setattr__(value, "_resolved_path_at_capture", str(resolved_before))
        return value
    def matches_disk(self) -> bool:
        try: return self == type(self).capture(Path(self.path))
        except OSError: return False
    def as_dict(self) -> dict[str, int | str]:
        return {name: getattr(self, name) for name in (
            "path", "size", "mtime_ns", "ctime_ns", "device", "inode",
        )}


@dataclass(frozen=True, slots=True)
class SelectedContainerInput:
    """Already-admitted facts for one directory-selected container."""

    file: SourceFileState
    adapter_id: str
    descriptor: ContainerDescriptor

    def __post_init__(self) -> None:
        if (
            type(self.file) is not SourceFileState
            or type(self.adapter_id) is not str
            or not self.adapter_id
            or type(self.descriptor) is not ContainerDescriptor
        ):
            raise TypeError("selected container input requires exact admitted facts")
        # Direct adapter probes leave identity binding to their caller, just
        # as DirectoryIndex does. Bind only that missing identity; all probed
        # layout, readiness and revision facts remain exact.
        if self.descriptor.adapter_id is None:
            object.__setattr__(self, "descriptor", replace(
                self.descriptor, adapter_id=self.adapter_id,
            ))
        elif self.descriptor.adapter_id != self.adapter_id:
            raise ValueError("selected container descriptor has a different adapter")


@dataclass(frozen=True, slots=True)
class _CapturedSourceTopology:
    """Private pre-stamp owner of one lexical source binding."""

    raw_path: str
    resolved_path: str
    followed_state: SourceFileState
    candidate_owner_id: str | None = None

    def __post_init__(self) -> None:
        captured = (
            self.followed_state._resolved_path_at_capture
            if type(self.followed_state) is SourceFileState
            else None
        )
        if (
            type(self.raw_path) is not str
            or not os.path.isabs(self.raw_path)
            or type(self.resolved_path) is not str
            or not os.path.isabs(self.resolved_path)
            or type(self.followed_state) is not SourceFileState
            or _raw_source_key(self.raw_path)
            != _raw_source_key(self.followed_state.path)
            or type(captured) is not str
            or _resolved_source_key(self.resolved_path)
            != _resolved_source_key(captured)
            or (
                self.candidate_owner_id is not None
                and (
                    type(self.candidate_owner_id) is not str
                    or not self.candidate_owner_id
                )
            )
        ):
            raise TypeError("captured source topology is invalid")

    def matches_disk(self) -> bool:
        """Retain the legacy internal-map diagnostic without rebaselining."""

        return self.followed_state.matches_disk()


def _raw_source_path(path: str | Path) -> str:
    return os.path.abspath(os.path.expanduser(os.fsdecode(path)))


def _raw_source_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(_raw_source_path(path)))


def _resolved_source_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(_raw_source_path(path)))


def _resolve_source_alias(path: str) -> str:
    return str(Path(path).resolve(strict=True))


def _capture_canonical_source_target(path: str) -> SourceFileState:
    return SourceFileState.capture(Path(path))


def _candidate_owner_id(path: str) -> str | None:
    from xrd_tools.sources.adapters import candidate_owner

    owner = candidate_owner(Path(path))
    return None if owner is None else owner.id
@dataclass(frozen=True, slots=True)
class ExternalSourceState:
    file: SourceFileState; dataset: str; first: int; stop: int; epoch: int
    def __post_init__(self) -> None:
        if (type(self.file) is not SourceFileState or type(self.dataset) is not str
                or not self.dataset or any(type(value) is not int or value < 0
                for value in (self.first, self.stop, self.epoch))
                or self.stop <= self.first):
            raise TypeError("external source state is invalid")
    def as_dict(self) -> dict[str, Any]:
        return {"file": self.file.as_dict(), "dataset": self.dataset,
                "first": self.first, "stop": self.stop, "epoch": self.epoch}
@dataclass(frozen=True, slots=True)
class AdmittedMotorValue:
    source_path: str; motor: str; value: float
    def __post_init__(self) -> None:
        if (type(self.source_path) is not str or not os.path.isabs(self.source_path)
                or type(self.motor) is not str or not self.motor or self.motor == "Manual"
                or type(self.value) is not float or not math.isfinite(self.value)):
            raise TypeError("admitted motor value is invalid")
    def as_dict(self) -> dict[str, str | float]:
        return {"source_path": self.source_path, "motor": self.motor, "value": self.value}
@dataclass(frozen=True, slots=True)
class AdmittedMetadataSource:
    source_path: str; metadata_file: SourceFileState | None
    def __post_init__(self) -> None:
        if (type(self.source_path) is not str or not os.path.isabs(self.source_path)
                or self.metadata_file is not None
                and type(self.metadata_file) is not SourceFileState):
            raise TypeError("admitted metadata source is invalid")
    def as_dict(self) -> dict[str, Any]:
        return {"source_path": self.source_path, "metadata_file": (
            None if self.metadata_file is None else self.metadata_file.as_dict())}
_SOURCE_ROLE_ORDER = ("source_file", "source_member", "external_member",
                      "detector_dependency", "image_metadata")
@dataclass(frozen=True, slots=True)
class SourceAliasBinding:
    raw_path: str; resolved_path: str; target_id: int; roles: tuple[str, ...]
    candidate_owner_id: str | None = None
    def __post_init__(self) -> None:
        order = {value: index for index, value in enumerate(_SOURCE_ROLE_ORDER)}
        if (type(self.raw_path) is not str or not os.path.isabs(self.raw_path)
                or type(self.resolved_path) is not str or not os.path.isabs(self.resolved_path)
                or type(self.target_id) is not int or self.target_id < 0
                or type(self.roles) is not tuple or not self.roles
                or not all(value in order for value in self.roles)
                or tuple(sorted(self.roles, key=order.__getitem__)) != self.roles
                or len(set(self.roles)) != len(self.roles)
                or self.candidate_owner_id is not None
                and (type(self.candidate_owner_id) is not str or not self.candidate_owner_id)):
            raise TypeError("source alias binding is invalid")
    def as_dict(self) -> dict[str, Any]:
        return {"raw_path": self.raw_path, "resolved_path": self.resolved_path,
                "target_id": self.target_id, "roles": list(self.roles),
                "candidate_owner_id": self.candidate_owner_id}
@dataclass(frozen=True, slots=True)
class CanonicalSourceTarget:
    target_id: int; resolved_path: str; state: SourceFileState; roles: tuple[str, ...]
    def __post_init__(self) -> None:
        order = {value: index for index, value in enumerate(_SOURCE_ROLE_ORDER)}
        if (type(self.target_id) is not int or self.target_id < 0
                or type(self.resolved_path) is not str or not os.path.isabs(self.resolved_path)
                or type(self.state) is not SourceFileState or self.state.path != self.resolved_path
                or type(self.roles) is not tuple or not self.roles
                or not all(value in order for value in self.roles)
                or tuple(sorted(self.roles, key=order.__getitem__)) != self.roles
                or len(set(self.roles)) != len(self.roles)):
            raise TypeError("canonical source target is invalid")
    def as_dict(self) -> dict[str, Any]:
        return {"target_id": self.target_id, "resolved_path": self.resolved_path,
                "state": self.state.as_dict(), "roles": list(self.roles)}
@dataclass(frozen=True, slots=True)
class SourceExecutionIdentityV1:
    aliases: tuple[SourceAliasBinding, ...]; targets: tuple[CanonicalSourceTarget, ...]; schema_version: int = 1
    def __post_init__(self) -> None:
        if (self.schema_version != 1 or type(self.aliases) is not tuple or not self.aliases
                or not all(type(value) is SourceAliasBinding for value in self.aliases)
                or type(self.targets) is not tuple or not self.targets
                or not all(type(value) is CanonicalSourceTarget for value in self.targets)
                or tuple(value.target_id for value in self.targets) != tuple(range(len(self.targets)))
                or any(value.target_id >= len(self.targets) for value in self.aliases)):
            raise TypeError("source execution identity is invalid")
    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "aliases": [value.as_dict() for value in self.aliases],
                "targets": [value.as_dict() for value in self.targets]}
def _absolute(path: str) -> str: return os.path.abspath(os.path.expanduser(os.fsdecode(path)))
def _path_key(path: str) -> str: return os.path.normcase(os.path.normpath(path))
def _resolved(state: SourceFileState) -> str:
    return _absolute(state.path) if state._resolved_path_at_capture is None else state._resolved_path_at_capture
def _revision(state: SourceFileState) -> tuple[int, ...]:
    return state.size, state.mtime_ns, state.ctime_ns, state.device, state.inode
@dataclass(frozen=True, slots=True)
class SourceExecutionStamp:
    file: SourceFileState; adapter_id: str; frame_count: int; first_label: int
    members: tuple[SourceFileState, ...] = (); external_members: tuple[ExternalSourceState, ...] = ()
    dependency_files: tuple[SourceFileState, ...] = (); admitted_motor_values: tuple[AdmittedMotorValue, ...] = ()
    metadata_sources: tuple[AdmittedMetadataSource, ...] = ()
    source_aliases: tuple[SourceAliasBinding, ...] = field(init=False, repr=False, compare=False)
    canonical_targets: tuple[CanonicalSourceTarget, ...] = field(init=False, repr=False, compare=False)
    execution_identity_v1: SourceExecutionIdentityV1 = field(init=False, repr=False, compare=False)
    def __post_init__(self) -> None:
        if type(self.file) is not SourceFileState or not self.adapter_id:
            raise ValueError("source stamp identity is empty")
        if any(type(value) is not int or value < 0 for value in (self.frame_count, self.first_label)):
            raise ValueError("source stamp counts are invalid")
        typed = ((self.members, SourceFileState), (self.external_members, ExternalSourceState),
                 (self.dependency_files, SourceFileState),
                 (self.admitted_motor_values, AdmittedMotorValue),
                 (self.metadata_sources, AdmittedMetadataSource))
        if any(type(values) is not tuple or not all(type(value) is kind for value in values)
               for values, kind in typed):
            raise TypeError("source stamp members are invalid")
        if len({value.path for value in self.dependency_files}) != len(self.dependency_files):
            raise TypeError("source stamp members are invalid")
        if self.admitted_motor_values and (
            not self.members or tuple(value.source_path for value in self.admitted_motor_values)
            != tuple(member.path for member in self.members)
            or len({value.motor for value in self.admitted_motor_values}) != 1
        ):
            raise ValueError("admitted motor values must align with every source member")
        if self.metadata_sources and (
            self.adapter_id != "tiff_series" or not self.members
            or tuple(value.source_path for value in self.metadata_sources)
            != tuple(member.path for member in self.members)
        ):
            raise ValueError("metadata sources must align with every TIFF member")
        owner = "image_file" if self.adapter_id == "tiff_series" else self.adapter_id
        occurrences = (((self.file, "source_file", owner),)
            + tuple((value, "source_member", owner) for value in self.members)
            + tuple((value.file, "external_member", None) for value in self.external_members)
            + tuple((value, "detector_dependency", None) for value in self.dependency_files)
            + tuple((value.metadata_file, "image_metadata", None)
                    for value in self.metadata_sources if value.metadata_file is not None))
        raws: dict[str, dict[str, Any]] = {}; targets: dict[str, dict[str, Any]] = {}
        raw_order: list[str] = []; target_order: list[str] = []
        for state, role, owner_id in occurrences:
            raw = _absolute(state.path); raw_key = _path_key(raw)
            resolved = _resolved(state); resolved_key = _path_key(resolved)
            target_state = SourceFileState(resolved, *_revision(state))
            seen = raws.get(raw_key)
            if seen is not None and (seen["raw"] != raw or seen["resolved_key"] != resolved_key):
                raise ValueError(f"conflicting source alias binding for {raw}")
            if seen is None:
                seen = {"raw": raw, "resolved": resolved, "resolved_key": resolved_key,
                        "roles": set(), "owner": owner_id}
                raws[raw_key] = seen; raw_order.append(raw_key)
            elif owner_id is not None and seen["owner"] not in (None, owner_id):
                raise ValueError(f"conflicting source candidate owners for {raw}")
            elif seen["owner"] is None: seen["owner"] = owner_id
            seen["roles"].add(role)
            target = targets.get(resolved_key)
            if target is not None and _revision(target["state"]) != _revision(target_state):
                raise ValueError(f"conflicting canonical source states for {resolved}")
            if target is None:
                target = {"resolved": resolved, "state": target_state, "roles": set()}
                targets[resolved_key] = target; target_order.append(resolved_key)
            target["roles"].add(role)
        role_order = {value: index for index, value in enumerate(_SOURCE_ROLE_ORDER)}
        ids = {key: index for index, key in enumerate(target_order)}
        canonical = tuple(CanonicalSourceTarget(ids[key], targets[key]["resolved"],
            targets[key]["state"], tuple(sorted(targets[key]["roles"], key=role_order.__getitem__)))
            for key in target_order)
        aliases = tuple(SourceAliasBinding(raws[key]["raw"], raws[key]["resolved"],
            ids[raws[key]["resolved_key"]], tuple(sorted(raws[key]["roles"], key=role_order.__getitem__)),
            raws[key]["owner"]) for key in raw_order)
        object.__setattr__(self, "source_aliases", aliases)
        object.__setattr__(self, "canonical_targets", canonical)
        object.__setattr__(self, "execution_identity_v1", SourceExecutionIdentityV1(aliases, canonical))
    @property
    def path(self) -> str: return self.file.path
    @property
    def size(self) -> int: return self.file.size
    @property
    def mtime_ns(self) -> int: return self.file.mtime_ns
    @property
    def member_stamps(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((value.path, value.size, value.mtime_ns) for value in self.members)
    def as_dict(self) -> dict[str, Any]:
        return {**self.file.as_dict(), "adapter_id": self.adapter_id,
            "frame_count": self.frame_count, "first_label": self.first_label,
            "member_stamps": [value.as_dict() for value in self.members],
            "external_members": [value.as_dict() for value in self.external_members],
            "dependency_files": [value.as_dict() for value in self.dependency_files],
            "admitted_motor_values": [value.as_dict() for value in self.admitted_motor_values],
            "metadata_sources": [value.as_dict() for value in self.metadata_sources]}
def validate_source_aliases(
    stamp: SourceExecutionStamp,
    *,
    cancelled: Callable[[], bool] | None = None,
    resolve_alias: Callable[[str], str] | None = None,
    capture_target: Callable[[str], SourceFileState] | None = None,
    candidate_owner_id: Callable[[str], str | None] | None = None, check_candidate_owner: bool = True,
) -> tuple[SourceExecutionIdentityV1, tuple[CanonicalSourceTarget, ...]]:
    """Prove every admitted lexical alias and canonical target twice."""
    if type(stamp) is not SourceExecutionStamp:
        raise TypeError("source alias validation requires an execution stamp")
    if resolve_alias is None:
        resolve_alias = _resolve_source_alias
    if capture_target is None:
        capture_target = _capture_canonical_source_target
    if check_candidate_owner and candidate_owner_id is None:
        candidate_owner_id = _candidate_owner_id
    is_cancelled = (lambda: False) if cancelled is None else cancelled
    targets = stamp.canonical_targets
    for sweep in range(2):
        for binding in stamp.source_aliases:
            if is_cancelled():
                raise RuntimeError("admission cancelled")
            try:
                resolved = resolve_alias(binding.raw_path)
            except (OSError, RuntimeError) as error:
                raise SourceRevisionChanged(
                    f"source alias is unavailable: {binding.raw_path}"
                ) from error
            if _path_key(resolved) != _path_key(binding.resolved_path):
                prefix = ("source candidate changed after admission: "
                          if binding.candidate_owner_id is not None
                          else "source alias retargeted after admission: ")
                raise SourceRevisionChanged(prefix + binding.raw_path)
            expected = targets[binding.target_id]
            try:
                current = capture_target(resolved)
            except OSError as error:
                raise SourceRevisionChanged(
                    f"source target is unavailable: {binding.resolved_path}"
                ) from error
            if current != expected.state:
                raise SourceRevisionChanged(
                    "source target changed after admission: "
                    f"{binding.resolved_path}"
                )
            if (check_candidate_owner and sweep == 0 and binding.candidate_owner_id is not None
                    and candidate_owner_id(binding.raw_path)
                    != binding.candidate_owner_id):
                raise SourceRevisionChanged(
                    "source candidate owner changed after admission: "
                    f"{binding.raw_path}"
                )
    return stamp.execution_identity_v1, targets


def validate_source_state_sweep(
    stamp: SourceExecutionStamp,
    *,
    cancelled: Callable[[], bool] | None = None,
    capture: Callable[[Path], SourceFileState] | None = None,
) -> None:
    """Perform one cheap exact-state sweep of the admitted source aliases.

    Qualification owns structural discovery and graph comparison.  Once that
    graph is fixed, Average only needs to prove that every admitted lexical
    path still resolves to the same canonical target with the same captured
    file revision before it can publish derived data.
    """
    if type(stamp) is not SourceExecutionStamp:
        raise TypeError("source state sweep requires an execution stamp")
    is_cancelled = (lambda: False) if cancelled is None else cancelled
    capture = SourceFileState.capture if capture is None else capture
    targets = stamp.canonical_targets
    for binding in stamp.source_aliases:
        if is_cancelled():
            raise InterruptedError("source state sweep cancelled")
        expected = targets[binding.target_id]
        try:
            current = capture(Path(binding.raw_path))
        except OSError as error:
            raise SourceRevisionChanged(
                f"source alias is unavailable: {binding.raw_path}"
            ) from error
        if _path_key(_resolved(current)) != _path_key(binding.resolved_path):
            raise SourceRevisionChanged(
                f"source alias retargeted after admission: {binding.raw_path}"
            )
        if _revision(current) != _revision(expected.state):
            raise SourceRevisionChanged(
                f"source target changed after admission: {binding.resolved_path}"
            )
    if is_cancelled():
        raise InterruptedError("source state sweep cancelled")
@dataclass(frozen=True, slots=True)
class PreparedSourceExecutionGraph:
    execution_source: SourceSpec; source_path: str; stamp: SourceExecutionStamp
    descriptor: ContainerDescriptor | None; group_key: str
    motor_names: tuple[str, ...] | None; scanned_motor_names: tuple[str, ...] | None
    dataset_paths: tuple[str, ...]; detector_shape: tuple[int, int] | None; native_dtype: str | None
    reader_binding: str | None = None
    def __post_init__(self) -> None:
        if type(self.execution_source) is not SourceSpec or type(self.source_path) is not str \
                or not os.path.isabs(self.source_path) or type(self.stamp) is not SourceExecutionStamp \
                or type(self.group_key) is not str or not self.group_key:
            raise TypeError("prepared source execution graph is invalid")
        if self.reader_binding not in (None, "average_closed_v1"):
            raise ValueError("source reader binding is unsupported")
        if (self.detector_shape is None) != (self.native_dtype is None):
            raise ValueError("source detector layout must be wholly known or absent")
        if self.reader_binding == "average_closed_v1":
            if type(self.motor_names) is not tuple or type(self.scanned_motor_names) is not tuple: raise TypeError("Average source graph requires exact motor tuples")
        elif self.scanned_motor_names is not None:
            raise TypeError("ordinary source graph cannot retain scanned motors")
        if type(self.dataset_paths) is not tuple or any(type(value) is not str or not value for value in self.dataset_paths):
            raise TypeError("source graph dataset paths are invalid")
def _frozen_options(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _frozen_options(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return tuple(_frozen_options(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}: return value
    return str(value)
def freeze_source_execution_graph(selection: SourceSpec, execution_source: SourceSpec, *,
        source_path: str | Path, group_key: str, reader_binding: str | None = None,
        file: SourceFileState, adapter_id: str, frame_count: int, first_label: int,
        detector_shape: tuple[int, int] | None, native_dtype: str | None,
        members: tuple[SourceFileState, ...] = (), external_members: tuple[ExternalSourceState, ...] = (),
        dependency_files: tuple[SourceFileState, ...] = (), admitted_motor_values: tuple[AdmittedMotorValue, ...] = (),
        metadata_sources: tuple[AdmittedMetadataSource, ...] = (), descriptor: ContainerDescriptor | None = None,
        motor_names: tuple[str, ...] | None = None, scanned_motor_names: tuple[str, ...] | None = None) -> PreparedSourceExecutionGraph:
    if type(selection) is not SourceSpec or type(execution_source) is not SourceSpec: raise TypeError("source graph requires exact SourceSpec values")
    source = SourceSpec(execution_source.uri, execution_source.kind,
        metadata_uri=execution_source.metadata_uri, entry=execution_source.entry,
        options=_frozen_options(dict(execution_source.options)))
    stamp = SourceExecutionStamp(file, str(adapter_id), int(frame_count), int(first_label),
        tuple(members), tuple(external_members), tuple(dependency_files),
        tuple(admitted_motor_values), tuple(metadata_sources))
    paths = () if descriptor is None else tuple(dict.fromkeys(value for value in
        (descriptor.dataset_path, *descriptor.segment_paths) if value))
    return PreparedSourceExecutionGraph(source, _absolute(str(source_path)), stamp,
        descriptor, str(group_key), motor_names, scanned_motor_names, paths,
        detector_shape, None if native_dtype is None else np.dtype(native_dtype).str,
        reader_binding)
def _cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled(): raise InterruptedError("source qualification cancelled")
def _same(actual: Any, expected: Any, message: str) -> None:
    if actual != expected: raise SourceRevisionChanged(message)
def _finite_motors(values: dict[str, Any]) -> dict[str, float]:
    from xrd_tools.core.metadata import numeric_metadata
    filtered = {key: value for key, value in values.items()
                if not isinstance(value, (bool, np.bool_))}
    return {key: value for key, value in numeric_metadata(filtered).items()
            if key and "roi" not in key.casefold() and "pd" not in key.casefold()}
def _source_policy(source: SourceSpec, options: dict[str, Any]) -> tuple:
    excluded = {"files", "admitted_motor_values", "average_compact_series_v1"}
    return (str(source.uri), source.kind, None if source.metadata_uri is None else str(source.metadata_uri),
            source.entry, _frozen_options({key: value for key, value in options.items() if key not in excluded}))
def _qualify_tiff(source: SourceSpec, *, selected_motor: str | None, reader_binding: str | None,
                   cancelled: Callable[[], bool] | None,
                   expected: PreparedSourceExecutionGraph | None = None,
                   force_compact: bool = False) -> PreparedSourceExecutionGraph | None:
    options = dict(source.options); marker = options.pop("average_compact_series_v1", None)
    if marker is not None and marker is not True: raise ValueError("compact Average TIFF marker is invalid")
    if marker is True and ({"files", "admitted_motor_values"} & set(options)):
        raise ValueError("compact Average TIFF source contains expanded facts")
    selected = Path(str(options.get("selected_file") or (expected.source_path if expected is not None else source.uri))).expanduser()
    compact = marker is True or force_compact
    if compact:
        from xrd_tools.sources.selection import image_series_spec, single_image_spec
        expanded = (single_image_spec(selected, metadata_format=options.get("metadata_format"))
                    if options.get("selection_mode") == "single_image"
                    else image_series_spec(selected, metadata_format=options.get("metadata_format")))
        qualified = dict(expanded.options)
        for key in ("meta_dir", "detector", "detector_shape", "raw_dtype", "raw_header_skip"):
            if key in options: qualified[key] = options[key]
    else: qualified = options
    files = tuple(str(Path(value).expanduser()) for value in qualified.get("files", ()))
    if not files and selected.is_file(): files = (str(selected),)
    if not files or compact and sum(Path(value) == selected for value in files) != 1:
        raise ValueError("TIFF source enumeration is empty or ambiguous")
    qualified.pop("admitted_motor_values", None)
    members: list[SourceFileState] = []; metadata_sources: list[AdmittedMetadataSource] = []
    motor_rows: list[AdmittedMotorValue] = []; common: list[str] | None = None; layout = None
    metadata_format = qualified.get("metadata_format", "auto")
    from xrd_tools.io.metadata import ImageMetadataRead, read_image_metadata_observed
    for index, value in enumerate(files):
        _cancelled(cancelled); path = Path(value); before = SourceFileState.capture(path)
        current = None
        if compact:
            current = read_detector_image_layout(path,
                detector_shape=qualified.get("detector_shape"), detector=qualified.get("detector"),
                raw_dtype=str(qualified.get("raw_dtype", "int32")),
                raw_header_skip=int(qualified.get("raw_header_skip", 0)))
            if current.frame_count != 1: raise ValueError("Average TIFF member must contain one frame")
            if layout is not None and (current.shape, current.dtype) != (layout.shape, layout.dtype):
                raise ValueError("Average TIFF member layouts differ")
            layout = current
        first = (ImageMetadataRead({}, None) if metadata_format is None else
                 read_image_metadata_observed(path, metadata_format, meta_dir=qualified.get("meta_dir"),
                    max_input_bytes=(1 << 16) if reader_binding else None))
        first_state = None if first.source_path is None else SourceFileState.capture(first.source_path)
        second = (ImageMetadataRead({}, None) if metadata_format is None else
                  read_image_metadata_observed(path, metadata_format, meta_dir=qualified.get("meta_dir"),
                    max_input_bytes=(1 << 16) if reader_binding else None))
        second_state = None if second.source_path is None else SourceFileState.capture(second.source_path)
        if ((first.source_path is None) != (second.source_path is None)
                or first.source_path is not None and _path_key(_absolute(str(first.source_path)))
                != _path_key(_absolute(str(second.source_path))) or first_state != second_state):
            raise SourceRevisionChanged("TIFF metadata source changed during qualification")
        metadata = AdmittedMetadataSource(before.path, second_state); motors = _finite_motors(dict(second.values))
        common = list(motors) if common is None else [name for name in common if name in motors]
        motor = None
        if selected_motor is not None:
            if selected_motor not in motors: raise ValueError("selected motor is absent from TIFF metadata")
            motor = AdmittedMotorValue(before.path, selected_motor, float(motors[selected_motor]))
        after = SourceFileState.capture(path)
        if before != after or _path_key(_resolved(before)) != _path_key(_resolved(after)):
            raise SourceRevisionChanged("TIFF source changed during qualification")
        if expected is None:
            members.append(before); metadata_sources.append(metadata)
            if motor is not None: motor_rows.append(motor)
        else:
            if index >= len(expected.stamp.members) or index >= len(expected.stamp.metadata_sources):
                raise SourceRevisionChanged("TIFF source graph gained a member")
            _same(before, expected.stamp.members[index], "TIFF member changed")
            _same(metadata, expected.stamp.metadata_sources[index], "TIFF metadata source changed")
            if motor is not None:
                if index >= len(expected.stamp.admitted_motor_values):
                    raise SourceRevisionChanged("TIFF selected motor graph changed")
                _same(motor, expected.stamp.admitted_motor_values[index], "TIFF selected motor changed")
    names = tuple(common or ()); scanned = names if reader_binding else None
    group = str(qualified.get("scan_name") or selected.stem)
    detector_shape = None if layout is None else tuple(layout.shape)
    native_dtype = None if layout is None else np.dtype(layout.dtype).str
    if expected is not None:
        stamp = expected.stamp
        if len(stamp.members) != len(files) or len(stamp.metadata_sources) != len(files) \
                or len(stamp.admitted_motor_values) != (len(files) if selected_motor else 0):
            raise SourceRevisionChanged("TIFF source graph cardinality changed")
        _same((stamp.adapter_id, stamp.frame_count, stamp.first_label), ("tiff_series", len(files), 1), "TIFF source scalar facts changed")
        _same((expected.source_path, expected.group_key, expected.motor_names,
               expected.scanned_motor_names, expected.detector_shape, expected.native_dtype),
              (_absolute(str(selected)), group, names, scanned, detector_shape, native_dtype),
              "TIFF source graph changed")
        actual_policy = _source_policy(SourceSpec(source.uri, SourceKind.TIFF_SERIES,
            metadata_uri=source.metadata_uri, entry=source.entry), qualified)
        _same(actual_policy, _source_policy(expected.execution_source,
            dict(expected.execution_source.options)), "TIFF source read policy changed")
        return None
    qualified["files"] = tuple(value.path for value in members)
    qualified["admitted_motor_values"] = tuple((value.source_path, value.motor, value.value) for value in motor_rows)
    execution = SourceSpec(source.uri, SourceKind.TIFF_SERIES,
        metadata_uri=source.metadata_uri, entry=source.entry, options=qualified)
    return freeze_source_execution_graph(source, execution, source_path=selected, group_key=group,
        reader_binding=reader_binding, file=members[0], adapter_id="tiff_series",
        frame_count=len(members), first_label=1, detector_shape=detector_shape,
        native_dtype=native_dtype, members=tuple(members), admitted_motor_values=tuple(motor_rows),
        metadata_sources=tuple(metadata_sources), motor_names=names, scanned_motor_names=scanned)
def _drain_average_qualification(
    handle: Any, slot: Any, completion: Callable[[], Any],
) -> Any:
    owner = _QualificationCleanupOwner(handle, slot, completion)
    owner.close()
    return owner.take()


def _drain_qualification(handle: Any, slot: Any) -> bool:
    retried = False; owner = slot.owner
    if owner is not None:
        delayed = bool(getattr(owner, "_cleanup_failed", False)); retried = delayed
        while owner.state != "CLOSED":
            if delayed: time.sleep(0.05)
            try: owner.close(); delayed = False
            except BaseException: retried = delayed = True
        slot.owner = None
    delayed = False
    while handle.id.valid:
        if delayed: time.sleep(0.05)
        try: handle.close(); delayed = False
        except BaseException: retried = delayed = True
    return retried


def _qualify_container(source: SourceSpec, *, selected_motor: str | None, reader_binding: str | None,
                       cancelled: Callable[[], bool] | None,
                       expected: PreparedSourceExecutionGraph | None = None,
                       selected: SelectedContainerInput | None = None) -> PreparedSourceExecutionGraph | None:
    from xrd_tools.io.bluesky_nexus import resolve_nxentry, validate_average_container_metadata_inputs
    from xrd_tools.io.nexus import _NexusDatasetOwnerSlot, _ResolvedNexusStack
    from xrd_tools.sources.adapters import candidate_owner
    from xrd_tools.sources.descriptor import _describe_container_from_open_with_binding
    from xrd_tools.sources.probe import ProbeState
    path = Path(source.uri).expanduser(); entry_name = source.entry or "entry"
    while True:
        before = SourceFileState.capture(path); _cancelled(cancelled)
        owner = None if reader_binding else candidate_owner(path); owner_id = None if owner is None else owner.id
        if selected is not None and (
            before != selected.file
            or _path_key(_resolved(before)) != _path_key(_resolved(selected.file))
            or owner_id != selected.adapter_id
        ):
            raise SourceRevisionChanged("selected container changed before qualification")
        handle = _HDF5_FILE_OPEN(path, "r", **({"rdcc_nbytes": 1 << 20} if reader_binding else {}))
        slot = _NexusDatasetOwnerSlot(); error = None; descriptor = None
        external_members: list[ExternalSourceState] = []; dependencies: list[SourceFileState] = []
        external_index = dependency_index = 0; pair: tuple[Any, Any] = (None, None)
        def emit_external(value: ExternalSourceState) -> None:
            nonlocal external_index
            if expected is not None and external_index >= len(expected.stamp.external_members):
                raise SourceRevisionChanged("container external graph gained a member")
            if expected is not None: _same(value, expected.stamp.external_members[external_index], "container external member changed")
            external_members.append(value)
            external_index += 1
        def emit_dependency(value: SourceFileState) -> None:
            nonlocal dependency_index
            if expected is not None and dependency_index >= len(expected.stamp.dependency_files):
                raise SourceRevisionChanged("container dependency graph gained a member")
            if expected is not None: _same(value, expected.stamp.dependency_files[dependency_index], "container dependency changed")
            dependencies.append(value)
            dependency_index += 1
        try:
            entry_group = resolve_nxentry(handle, entry_name, exact_hint=bool(reader_binding))
            if entry_group is None: raise ValueError(f"requested entry {entry_name!r} is unavailable")
            if reader_binding:
                pair = validate_average_container_metadata_inputs(entry_group, policy="average_bounded_v1")
                if expected is not None: _same((tuple(pair[0] or ()), tuple(pair[1] or ())), (expected.scanned_motor_names, expected.motor_names), "container motor catalog changed")
                descriptor, binding = _describe_container_from_open_with_binding(handle, path=path,
                    entry=entry_name, resolved_entry_group=entry_group,
                    prevalidated_motor_names=pair, owner_slot=slot)
            else:
                if owner is None or source.kind not in owner.kinds:
                    raise ValueError(f"selected container has no compatible owner: {path}")
                descriptor = describe_container_from_open(handle, path=path, entry=entry_name,
                    size=before.size, mtime_ns=before.mtime_ns, adapter_id=owner.id)
                entry_group = resolve_nxentry(
                    handle, descriptor.resolved_entry or entry_name, exact_hint=True,
                )
                if entry_group is None: raise ValueError("container resolved entry disappeared")
                binding = _ResolvedNexusStack(entry_group); slot.owner = binding
                selectors = descriptor.segment_paths or ((descriptor.dataset_path,) if descriptor.dataset_path else ())
                for selector in selectors:
                    value = handle.get(selector)
                    if not isinstance(value, h5py.Dataset): raise ValueError("container has no detector dataset")
                    binding.append(selector, value)
            if descriptor.state is not ProbeState.READY or descriptor.kind is SourceKind.PROCESSED_NEXUS \
                    or descriptor.frame_count < 1 or descriptor.dataset_path is None:
                raise ValueError("container is not a READY raw detector source")
            if selected is not None and descriptor != selected.descriptor:
                raise SourceRevisionChanged("selected container descriptor changed during qualification")
            if reader_binding:
                # Average's bounded reader retains its one captured binding.
                _capture_bound_container_dependencies(binding, descriptor, before, cancelled=cancelled,
                    emit_external=emit_external, emit_dependency=emit_dependency)
            else:
                is_cancelled = _not_cancelled if cancelled is None else cancelled
                for member in _external_members(path, before, descriptor, cancelled=is_cancelled):
                    emit_external(member)
                for dependency in _selected_dependency_files(
                    path, before, descriptor, tuple(external_members), cancelled=is_cancelled,
                ):
                    emit_dependency(dependency)
        except BaseException as caught: error = caught

        def complete() -> PreparedSourceExecutionGraph | None:
            if error is not None:
                if isinstance(error, RuntimeError) and error.args == ("admission cancelled",):
                    raise InterruptedError("source qualification cancelled") from error
                if isinstance(error, KeyError) and error.args == ("captured entry has no detector dataset",):
                    raise ValueError("container has no detector dataset") from error
                raise error
            after = SourceFileState.capture(path); current_owner = None if reader_binding else candidate_owner(path)
            if before != after or _path_key(_resolved(before)) != _path_key(_resolved(after)) \
                    or not reader_binding and (current_owner is None or current_owner.id != owner_id):
                raise SourceRevisionChanged("container changed during qualification")
            if reader_binding and any(not state.matches_disk() for state in (
                    *(item.file for item in external_members), *dependencies)):
                raise SourceRevisionChanged("container dependency changed during qualification")
            all_motors = tuple(pair[1] or ()) if reader_binding else descriptor.motor_names
            scanned = tuple(pair[0] or ()) if reader_binding else None
            if selected_motor is not None and selected_motor not in (all_motors or ()):
                raise ValueError("selected motor is absent from the container")
            group = descriptor.scan_name or path.stem.removesuffix("_master")
            execution = SourceSpec(path, descriptor.kind, metadata_uri=source.metadata_uri,
                entry=descriptor.resolved_entry or entry_name, options=dict(source.options))
            adapter_id = "nexus_hdf5" if reader_binding else owner_id
            if expected is not None:
                if external_index != len(expected.stamp.external_members) or dependency_index != len(expected.stamp.dependency_files):
                    raise SourceRevisionChanged("container dependency graph cardinality changed")
                _same(before, expected.stamp.file, "container source changed")
                _same((expected.stamp.adapter_id, expected.stamp.frame_count, expected.stamp.first_label,
                       expected.descriptor, expected.source_path, expected.group_key, expected.motor_names,
                       expected.scanned_motor_names, expected.detector_shape, expected.native_dtype),
                      (adapter_id, descriptor.frame_count, 0, descriptor, _absolute(str(path)), group,
                       None if all_motors is None else tuple(all_motors), scanned, tuple(descriptor.frame_shape),
                       np.dtype(descriptor.dtype).str), "container source graph changed")
                _same(_source_policy(execution, dict(execution.options)),
                      _source_policy(expected.execution_source, dict(expected.execution_source.options)),
                      "container source read policy changed")
                return None
            return freeze_source_execution_graph(source, execution, source_path=path,
                group_key=group, reader_binding=reader_binding, file=before,
                adapter_id=adapter_id, frame_count=descriptor.frame_count, first_label=0,
                detector_shape=tuple(descriptor.frame_shape), native_dtype=np.dtype(descriptor.dtype).str,
                descriptor=descriptor, external_members=tuple(external_members),
                dependency_files=tuple(dependencies), motor_names=None if all_motors is None else tuple(all_motors),
                scanned_motor_names=scanned)

        if reader_binding:
            return _drain_average_qualification(handle, slot, complete)
        if _drain_qualification(handle, slot):
            continue
        return complete()
def qualify_source_execution_graph(source: SourceSpec, *, selected_motor: str | None = None,
        reader_binding: str | None = None,
        selected_container: SelectedContainerInput | None = None,
        cancelled: Callable[[], bool] | None = None) -> PreparedSourceExecutionGraph:
    if type(source) is not SourceSpec: raise TypeError("source must be an exact SourceSpec")
    if reader_binding not in (None, "average_closed_v1"): raise ValueError("source reader binding is unsupported")
    if selected_container is not None and (
        type(selected_container) is not SelectedContainerInput
        or reader_binding is not None
        or source.kind not in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER)
    ):
        raise TypeError("selected container facts require ordinary container qualification")
    if source.kind is SourceKind.TIFF_SERIES:
        return _qualify_tiff(source, selected_motor=selected_motor,
            reader_binding=reader_binding, cancelled=cancelled)
    if source.kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
        return _qualify_container(source, selected_motor=selected_motor,
                                  reader_binding=reader_binding, cancelled=cancelled,
                                  selected=selected_container)
    raise ValueError(f"source kind {source.kind.value} is unsupported for Average")
def requalify_source_execution_graph(source: SourceSpec, expected: PreparedSourceExecutionGraph,
        *, selected_motor: str | None = None, reader_binding: str | None = None,
        cancelled: Callable[[], bool] | None = None) -> PreparedSourceExecutionGraph:
    if type(expected) is not PreparedSourceExecutionGraph:
        raise TypeError("expected source graph must be exact")
    if reader_binding != expected.reader_binding or source.kind is not expected.execution_source.kind:
        raise SourceRevisionChanged("source reader binding or kind changed")
    validate_source_aliases(expected.stamp, cancelled=cancelled, check_candidate_owner=reader_binding is None)
    if source.kind is SourceKind.TIFF_SERIES:
        _qualify_tiff(source, selected_motor=selected_motor, reader_binding=reader_binding,
            cancelled=cancelled, expected=expected)
    elif source.kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
        _qualify_container(source, selected_motor=selected_motor, reader_binding=reader_binding,
            cancelled=cancelled, expected=expected)
    else: raise ValueError(f"source kind {source.kind.value} is unsupported for Average")
    return expected
def validate_source_execution_graph(value: PreparedSourceExecutionGraph, *,
        cancelled: Callable[[], bool] | None = None) -> None:
    if type(value) is not PreparedSourceExecutionGraph:
        raise TypeError("source graph must be exact")
    validate_source_aliases(value.stamp, cancelled=cancelled, check_candidate_owner=value.reader_binding is None)
    source = value.execution_source
    if source.kind is SourceKind.TIFF_SERIES:
        rows = value.stamp.admitted_motor_values
        selected_motor = rows[0].motor if rows else None
        _qualify_tiff(source, selected_motor=selected_motor, reader_binding=value.reader_binding,
            cancelled=cancelled, expected=value,
            force_compact=value.reader_binding == "average_closed_v1" and value.detector_shape is not None)
    elif source.kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
        _qualify_container(source, selected_motor=None, reader_binding=value.reader_binding,
            cancelled=cancelled, expected=value)
    else: raise SourceRevisionChanged("source graph kind is unsupported")
def source_execution_projection(value: PreparedSourceExecutionGraph) -> dict:
    return value.stamp.as_dict()
def source_execution_identity_v1_projection(value: PreparedSourceExecutionGraph) -> dict:
    return value.stamp.execution_identity_v1.as_dict()
def source_snapshots_projection(value: PreparedSourceExecutionGraph, *, writer: bool) -> dict:
    stamp = value.stamp
    if stamp.members:
        snapshots = {state.path: {"adapter_id": stamp.adapter_id, **state.as_dict(),
            "frame_count": 1, "self_contained": True} for state in stamp.members}
        for metadata in stamp.metadata_sources:
            state = metadata.metadata_file
            if state is not None:
                if state.path in snapshots and snapshots[state.path] != state.as_dict():
                    raise ValueError("conflicting source snapshot state")
                snapshots[state.path] = {"adapter_id": "image_metadata", **state.as_dict(),
                    "frame_count": 0, "self_contained": True,
                    "source_role": "image_metadata"}
    else:
        snapshot = {"adapter_id": stamp.adapter_id, **stamp.file.as_dict(),
                    "frame_count": stamp.frame_count}
        if value.descriptor is not None:
            snapshot.update(dataset_path=value.descriptor.dataset_path,
                            self_contained=value.descriptor.self_contained)
        snapshots = {stamp.file.path: snapshot}
        for external in stamp.external_members:
            state = external.file
            snapshots[state.path] = {
                "adapter_id": stamp.adapter_id, **state.as_dict(),
                "frame_count": external.stop - external.first,
                "dataset_path": external.dataset, "self_contained": True,
            }
        for state in stamp.dependency_files:
            snapshots[state.path] = {
                "adapter_id": "hdf5_dependency", **state.as_dict(),
                "frame_count": 0, "self_contained": True,
                "source_role": "detector_dependency",
            }
    if not writer: return snapshots
    allowed = {"adapter_id", "size", "mtime_ns", "frame_count", "dataset_path", "self_contained"}
    return {path: {key: item for key, item in snapshot.items()
                   if key in allowed and item is not None}
            for path, snapshot in snapshots.items()}
def stable_lineage_projection(value: PreparedSourceExecutionGraph, *, target: str | Path) -> tuple:
    identity = value.stamp.execution_identity_v1
    aliases = tuple((item.raw_path, item.resolved_path, item.target_id, item.roles,
                     item.candidate_owner_id) for item in identity.aliases
                    if "source_file" in item.roles)
    ids = {item.target_id for item in identity.aliases if "source_file" in item.roles}
    targets = tuple((item.target_id, item.resolved_path, item.roles)
                    for item in identity.targets if item.target_id in ids)
    return (value.stamp.adapter_id, value.group_key,
            os.path.normcase(os.path.realpath(target)), aliases, targets)
def append_source_from_execution_graph(value: PreparedSourceExecutionGraph, *, generation: int) -> AppendSource:
    stamp = value.stamp
    if stamp.members and len(stamp.members) != stamp.frame_count:
        raise ValueError("flat-series Append requires one exact member per frame")
    images = tuple(AppendImageMember(item.path, item.size, item.mtime_ns,
        index, index + 1, index) for index, item in enumerate(stamp.members))
    externals = tuple(AppendExternalMember(item.file.path, item.dataset,
        item.file.size, item.file.mtime_ns, item.first, item.stop, item.epoch)
        for item in stamp.external_members)
    if value.descriptor is not None and value.descriptor.kind is SourceKind.EIGER_MASTER and not externals:
        raise ValueError("Eiger Append requires exact external dataset facts")
    return AppendSource(stamp.path, stamp.adapter_id, stamp.size, stamp.mtime_ns,
        stamp.frame_count, science_fingerprint(stamp.execution_identity_v1.as_dict()),
        value.dataset_paths, images, externals, int(generation))
def source_graph_payload(value: PreparedSourceExecutionGraph) -> dict:
    policy_options = {
        key: item for key, item in dict(value.execution_source.options).items()
        if key not in {"files", "admitted_motor_values", "average_compact_series_v1"}
    }
    return {
        "schema_version": 1,
        "reader_binding": value.reader_binding,
        "group_key": value.group_key,
        "motor_names": value.motor_names,
        "scanned_motor_names": value.scanned_motor_names,
        "dataset_paths": value.dataset_paths,
        "detector_layout": (None if value.detector_shape is None else {
            "shape": value.detector_shape, "dtype": value.native_dtype,
        }),
        "source_read_policy": {
            "uri": str(value.execution_source.uri),
            "kind": value.execution_source.kind.value,
            "metadata_uri": (None if value.execution_source.metadata_uri is None
                             else str(value.execution_source.metadata_uri)),
            "entry": value.execution_source.entry,
            "options": _frozen_options(policy_options),
        },
        "source_execution": source_execution_projection(value),
        "execution_identity_v1": source_execution_identity_v1_projection(value),
        "source_snapshots": source_snapshots_projection(value, writer=True),
    }
def source_graph_digest(value: PreparedSourceExecutionGraph) -> str:
    digest = hashlib.sha256(b"xdart.source-execution-graph.v1\0")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False)
    for fragment in encoder.iterencode(source_graph_payload(value)):
        digest.update(fragment.encode())
    return digest.hexdigest()


def _source_state_key(path: str | Path) -> str:
    return _raw_source_key(path)


def _same_source_revision(
    left: SourceFileState,
    right: SourceFileState,
) -> bool:
    return (
        _raw_source_key(left.path) == _raw_source_key(right.path)
        and (
            left.size,
            left.mtime_ns,
            left.ctime_ns,
            left.device,
            left.inode,
        )
        == (
            right.size,
            right.mtime_ns,
            right.ctime_ns,
            right.device,
            right.inode,
        )
    )


def _same_followed_source_revision(
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


def _topology_from_captured_state(
    state: SourceFileState,
    *,
    candidate_owner_id: str | None = None,
) -> _CapturedSourceTopology:
    """Lift an actual capture without consulting the filesystem again."""

    captured = state._resolved_path_at_capture
    if type(captured) is not str:
        raise SourceRevisionChanged(
            "source state has no captured resolved topology: "
            f"{state.path}"
        )
    return _CapturedSourceTopology(
        _raw_source_path(state.path),
        _raw_source_path(captured),
        state,
        candidate_owner_id,
    )


def _capture_source_topology(
    path: Path,
    *,
    cancelled: Callable[[], bool],
    candidate_owner_id: str | None = None,
) -> _CapturedSourceTopology:
    if cancelled():
        raise RuntimeError("admission cancelled")
    selected = Path(_raw_source_path(path))
    return _topology_from_captured_state(
        SourceFileState.capture(selected),
        candidate_owner_id=candidate_owner_id,
    )


def _remember_source_state(
    path: Path,
    states: dict[str, _CapturedSourceTopology],
    *,
    cancelled: Callable[[], bool],
    candidate_owner_id: str | None = None,
) -> _CapturedSourceTopology:
    current = _capture_source_topology(
        path,
        cancelled=cancelled,
        candidate_owner_id=candidate_owner_id,
    )
    if cancelled():
        raise RuntimeError("admission cancelled")
    key = _source_state_key(current.raw_path)
    accepted = states.get(key)
    if accepted is not None:
        if (
            _resolved_source_key(accepted.resolved_path)
            != _resolved_source_key(current.resolved_path)
            or not _same_followed_source_revision(
                accepted.followed_state,
                current.followed_state,
            )
            or (
                candidate_owner_id is not None
                and accepted.candidate_owner_id not in {
                    None,
                    candidate_owner_id,
                }
            )
        ):
            raise SourceRevisionChanged(
                "HDF5 dependency changed during admission: "
                f"{current.raw_path}"
            )
        if (
            accepted.candidate_owner_id is None
            and candidate_owner_id is not None
        ):
            accepted = replace(
                accepted,
                candidate_owner_id=candidate_owner_id,
            )
            states[key] = accepted
        return accepted
    states[key] = current
    return current


def _verify_source_states(
    states: dict[str, _CapturedSourceTopology],
    *,
    cancelled: Callable[[], bool],
) -> None:
    """Prove every captured raw binding twice without rebasing its topology."""

    for sweep in range(2):
        for topology in states.values():
            if cancelled():
                raise RuntimeError("admission cancelled")
            try:
                resolved = _resolve_source_alias(topology.raw_path)
            except (OSError, RuntimeError) as error:
                raise SourceRevisionChanged(
                    f"source alias is unavailable: {topology.raw_path}"
                ) from error
            if _resolved_source_key(resolved) != _resolved_source_key(
                topology.resolved_path
            ):
                raise SourceRevisionChanged(
                    "source alias retargeted during admission: "
                    f"{topology.raw_path}"
                )
            try:
                current = _capture_canonical_source_target(resolved)
            except OSError as error:
                raise SourceRevisionChanged(
                    "source target is unavailable: "
                    f"{topology.resolved_path}"
                ) from error
            if not _same_followed_source_revision(
                topology.followed_state,
                current,
            ):
                raise SourceRevisionChanged(
                    "HDF5 dependency changed during admission: "
                    f"{topology.raw_path}"
                )
            if sweep == 0 and topology.candidate_owner_id is not None:
                if (
                    _candidate_owner_id(topology.raw_path)
                    != topology.candidate_owner_id
                ):
                    raise SourceRevisionChanged(
                        "source candidate owner changed during admission: "
                        f"{topology.raw_path}"
                    )
    if cancelled():
        raise RuntimeError("admission cancelled")


@contextmanager
def _open_stable_hdf5_dependency(
    path: Path,
    states: dict[str, _CapturedSourceTopology],
    *,
    cancelled: Callable[[], bool],
):
    """Open one HDF5 file only while its strong source state stays exact."""

    import h5py

    selected = Path(_raw_source_path(path))
    try:
        before = _remember_source_state(
            selected,
            states,
            cancelled=cancelled,
        )
    except FileNotFoundError:
        if cancelled():
            raise RuntimeError("admission cancelled")
        yield None
        return
    try:
        handle = h5py.File(selected, "r")
    except OSError as error:
        if cancelled():
            raise RuntimeError("admission cancelled") from error
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        if cancelled():
            raise RuntimeError("admission cancelled") from error
        raise ValueError(
            f"HDF5 dependency could not be inspected: {selected}: {error}"
        ) from error
    body_failure: BaseException | None = None
    body_traceback = None
    try:
        yield handle
    except BaseException as error:
        body_failure = error
        body_traceback = error.__traceback__

    close_failure: BaseException | None = None
    close_traceback = None
    try:
        handle.close()
    except BaseException as error:
        close_failure = error
        close_traceback = error.__traceback__

    if body_failure is not None and close_failure is not None:
        if (
            body_failure.__cause__ is None
            and close_failure.__context__ is body_failure
        ):
            close_failure.__context__ = None
        pending = [close_failure]
        seen: set[int] = set()
        close_reaches_body = False
        while pending:
            linked = pending.pop()
            if linked is body_failure:
                close_reaches_body = True
                break
            marker = id(linked)
            if marker in seen:
                continue
            seen.add(marker)
            if linked.__cause__ is not None:
                pending.append(linked.__cause__)
            if linked.__context__ is not None:
                pending.append(linked.__context__)
        if body_failure.__cause__ is None and not close_reaches_body:
            body_failure.__cause__ = close_failure
            body_failure.__suppress_context__ = True
        else:
            note = (
                "secondary HDF5 close failure: "
                f"{type(close_failure).__name__}: {close_failure}"
            )
            if note not in getattr(body_failure, "__notes__", ()):
                body_failure.add_note(note)

    if body_failure is not None and not isinstance(body_failure, Exception):
        raise body_failure.with_traceback(body_traceback)
    if (
        isinstance(body_failure, SourceRevisionChanged)
    ):
        raise body_failure.with_traceback(body_traceback)
    if (
        isinstance(body_failure, RuntimeError)
        and body_failure.args == ("admission cancelled",)
    ):
        raise body_failure.with_traceback(body_traceback)
    primary_failure = (
        body_failure if body_failure is not None else close_failure
    )
    if cancelled():
        raise RuntimeError("admission cancelled") from primary_failure
    try:
        after = _capture_source_topology(
            selected,
            cancelled=cancelled,
        )
    except RuntimeError as error:
        if error.args != ("admission cancelled",):
            raise
        if primary_failure is not None:
            raise error from primary_failure
        raise
    except OSError as error:
        cause = primary_failure if primary_failure is not None else error
        raise SourceRevisionChanged(
            f"HDF5 dependency changed during admission: {selected}"
        ) from cause
    if (
        _resolved_source_key(after.resolved_path)
        != _resolved_source_key(before.resolved_path)
        or not _same_followed_source_revision(
            after.followed_state,
            before.followed_state,
        )
    ):
        drift = SourceRevisionChanged(
            f"HDF5 dependency changed during admission: {selected}"
        )
        if primary_failure is not None:
            raise drift from primary_failure
        raise drift
    if cancelled():
        raise RuntimeError("admission cancelled") from primary_failure
    if body_failure is not None:
        raise body_failure.with_traceback(body_traceback)
    if close_failure is not None:
        raise close_failure.with_traceback(close_traceback)


def _hdf5_link_file(parent: object, filename: object) -> Path:
    """Return the absolute lexical dependency named by its declaring file."""

    base = Path(os.fsdecode(parent.file.filename)).parent
    return Path(_raw_source_path(base / os.fsdecode(filename)))


def _hdf5_object_path(value: object) -> str:
    raw = os.fsdecode(value)
    return "/" + posixpath.normpath("/" + raw.lstrip("/")).lstrip("/")


def _not_cancelled() -> bool:
    return False


def _trace_hdf5_object_dependencies(
    file_path: Path,
    object_path: str,
    *,
    paths: list[Path],
    seen: set[tuple[str, str]],
    cancelled: Callable[[], bool],
    required: bool,
    states: dict[str, _CapturedSourceTopology],
) -> None:
    """Close ExternalLink/soft-link/VDS chains for one HDF5 object path."""

    import h5py

    selected_file = Path(_raw_source_path(file_path))
    selected_object = _hdf5_object_path(object_path)
    if cancelled():
        raise RuntimeError("admission cancelled")
    try:
        topology = _remember_source_state(
            selected_file,
            states,
            cancelled=cancelled,
        )
    except FileNotFoundError:
        # The declaring link/VDS caller already retained this lexical path.
        # Let target-collision adjudication see it before the deferred freeze
        # turns a genuinely missing required dependency into typed pending.
        return
    except OSError as error:
        if required:
            raise SourceRevisionChanged(
                "required HDF5 dependency capture is unverifiable: "
                f"{selected_file}"
            ) from error
        raise
    identity = (
        _resolved_source_key(topology.resolved_path),
        selected_object,
    )
    if identity in seen:
        return
    seen.add(identity)
    with _open_stable_hdf5_dependency(
        selected_file,
        states,
        cancelled=cancelled,
    ) as handle:
        if handle is None:
            # The declaring link/VDS source path is already in ``paths``.  The
            # caller freezes it and emits the finite-landing refusal.
            return
        value: object = handle
        components = tuple(
            component
            for component in selected_object.strip("/").split("/")
            if component
        )
        for offset, component in enumerate(components):
            if cancelled():
                raise RuntimeError("admission cancelled")
            if not isinstance(value, h5py.Group):
                if required:
                    raise ValueError(
                        "selected HDF5 dependency path is incomplete: "
                        f"{selected_file}:{selected_object}"
                    )
                return
            try:
                link = value.get(component, getlink=True)
            except (KeyError, OSError, RuntimeError):
                if required:
                    raise ValueError(
                        "selected HDF5 dependency path is unavailable: "
                        f"{selected_file}:{selected_object}"
                    )
                return
            remainder = components[offset + 1 :]
            if isinstance(link, h5py.ExternalLink):
                dependency = _hdf5_link_file(value, link.filename)
                paths.append(dependency)
                target = _hdf5_object_path(link.path)
                if remainder:
                    target = _hdf5_object_path(
                        posixpath.join(target, *remainder)
                    )
                _trace_hdf5_object_dependencies(
                    dependency,
                    target,
                    paths=paths,
                    seen=seen,
                    cancelled=cancelled,
                    required=required,
                    states=states,
                )
                return
            if isinstance(link, h5py.SoftLink):
                target = os.fsdecode(link.path)
                if not target.startswith("/"):
                    target = posixpath.join(value.name, target)
                if remainder:
                    target = posixpath.join(target, *remainder)
                _trace_hdf5_object_dependencies(
                    Path(os.fsdecode(value.file.filename)),
                    _hdf5_object_path(target),
                    paths=paths,
                    seen=seen,
                    cancelled=cancelled,
                    required=required,
                    states=states,
                )
                return
            try:
                next_value = value.get(component)
            except (KeyError, OSError, RuntimeError):
                if required:
                    raise ValueError(
                        "selected HDF5 dependency path is unavailable: "
                        f"{selected_file}:{selected_object}"
                    )
                return
            if next_value is None:
                if required:
                    raise ValueError(
                        "selected HDF5 dependency path is unavailable: "
                        f"{selected_file}:{selected_object}"
                    )
                return
            value = next_value
        if isinstance(value, h5py.Dataset):
            _extend_hdf5_dataset_dependency_paths(
                value,
                paths=paths,
                seen=seen,
                cancelled=cancelled,
                required=required,
                states=states,
            )
        elif required:
            raise ValueError(
                "selected HDF5 dependency is not a dataset: "
                f"{selected_file}:{selected_object}"
            )


def _extend_hdf5_dataset_dependency_paths(
    dataset: object,
    *,
    paths: list[Path],
    seen: set[tuple[str, str]],
    cancelled: Callable[[], bool],
    required: bool,
    states: dict[str, _CapturedSourceTopology],
) -> None:
    """Add external-storage and recursively closed VDS dependencies."""

    current_file = Path(_raw_source_path(dataset.file.filename))
    base = current_file.parent
    for value in dataset.external or ():
        if cancelled():
            raise RuntimeError("admission cancelled")
        dependency = Path(
            _raw_source_path(base / os.fsdecode(value[0]))
        )
        try:
            _remember_source_state(
                dependency,
                states,
                cancelled=cancelled,
            )
        except FileNotFoundError as error:
            raise SourceRevisionChanged(
                "external HDF5 storage is still landing: "
                f"{dependency}"
            ) from error
        except OSError as error:
            if required:
                raise SourceRevisionChanged(
                    "external HDF5 storage capture is unverifiable: "
                    f"{dependency}"
                ) from error
            raise
        paths.append(dependency)
    if not bool(dataset.is_virtual):
        return
    for source in dataset.virtual_sources():
        if cancelled():
            raise RuntimeError("admission cancelled")
        filename = os.fsdecode(source.file_name)
        dependency = (
            current_file
            if filename in {"", "."}
            else Path(_raw_source_path(base / filename))
        )
        if dependency != current_file:
            paths.append(dependency)
        _trace_hdf5_object_dependencies(
            dependency,
            os.fsdecode(source.dset_name),
            paths=paths,
            seen=seen,
            cancelled=cancelled,
            required=required,
            states=states,
        )


def _external_members(
    master: Path,
    master_state: SourceFileState,
    descriptor: ContainerDescriptor,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[ExternalSourceState, ...]:
    # A HardLink/SoftLink leaf can resolve through an external ancestor. Match
    # the core qualifier's dataset owner, not the syntax of the final link.
    segments = descriptor.segment_paths or (
        (descriptor.dataset_path,) if descriptor.dataset_path else ()
    )
    if not segments:
        raise ValueError("external container has no member-qualified proof")
    import h5py
    from xrd_tools.io.nexus import _selected_link_owner_selector
    master_topology = _topology_from_captured_state(master_state)
    states = {
        _source_state_key(master_topology.raw_path): master_topology
    }
    values: list[ExternalSourceState] = []
    first = 0

    def member_state(path: Path) -> _CapturedSourceTopology:
        try:
            return _remember_source_state(path, states, cancelled=cancelled)
        except FileNotFoundError as error:
            raise SourceRevisionChanged(
                f"external detector member disappeared during admission: {path}"
            ) from error
        except OSError as error:
            raise SourceRevisionChanged(
                f"external detector member capture is unverifiable: {path}"
            ) from error

    try:
        with _open_stable_hdf5_dependency(
            master,
            states,
            cancelled=cancelled,
        ) as handle:
            if handle is None:
                raise SourceRevisionChanged(
                    f"external container disappeared during admission: {master}"
                )
            for epoch, segment in enumerate(segments):
                if cancelled():
                    raise RuntimeError("admission cancelled")
                parent = handle.get(posixpath.dirname(segment) or "/")
                if not isinstance(parent, h5py.Group):
                    raise ValueError("external container lost its exact dataset")
                leaf = posixpath.basename(segment)
                link = parent.get(leaf, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    # Retain direct-link capture before dereferencing it.
                    member_state(_hdf5_link_file(parent, link.filename))
                dataset = parent.get(leaf)
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim not in {2, 3}:
                    raise ValueError("external container lost its exact dataset")
                owner_path, owner_selector = _selected_link_owner_selector(
                    handle, segment, dataset,
                )
                path = Path(_raw_source_path(owner_path))
                topology = member_state(path)
                count = 1 if dataset.ndim == 2 else int(dataset.shape[0])
                frame_shape = (
                    tuple(int(value) for value in dataset.shape)
                    if dataset.ndim == 2
                    else tuple(int(value) for value in dataset.shape[1:])
                )
                if (
                    frame_shape != descriptor.frame_shape
                    or np.dtype(dataset.dtype) != descriptor.dtype
                ):
                    raise ValueError(
                        "external detector layout changed during admission: "
                        f"{path}:{dataset.name}"
                    )
                stop = first + count
                if _resolved_source_key(topology.resolved_path) != _resolved_source_key(
                    master_topology.resolved_path
                ):
                    values.append(
                        ExternalSourceState(
                            topology.followed_state,
                            owner_selector,
                            first,
                            stop,
                            epoch,
                        )
                    )
                first = stop
                if cancelled():
                    raise RuntimeError("admission cancelled")
        if first != descriptor.frame_count:
            raise ValueError(
                "external detector frame count changed during admission: "
                f"descriptor={descriptor.frame_count}, members={first}"
            )
    except SourceRevisionChanged:
        raise
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise
    except (OSError, ValueError) as error:
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise
    _verify_source_states(states, cancelled=cancelled)
    return tuple(values)


def _selected_dependency_files(
    master: Path,
    master_state: SourceFileState,
    descriptor: ContainerDescriptor,
    external_members: tuple[ExternalSourceState, ...],
    *,
    cancelled: Callable[[], bool],
) -> tuple[SourceFileState, ...]:
    """Freeze non-leaf-link and dataset-storage files actually selected."""

    selectors = descriptor.segment_paths or (
        (descriptor.dataset_path,) if descriptor.dataset_path else ()
    )
    if not selectors:
        return ()
    import h5py

    paths: list[Path] = []
    seen: set[tuple[str, str]] = set()
    master_topology = _topology_from_captured_state(master_state)
    states = {
        _source_state_key(master_topology.raw_path): master_topology
    }
    for external in external_members:
        key = _source_state_key(external.file.path)
        accepted = states.get(key)
        if accepted is not None and not _same_source_revision(
            accepted.followed_state,
            external.file,
        ):
            raise SourceRevisionChanged(
                "external detector member changed during admission: "
                f"{external.file.path}"
            )
        states.setdefault(
            key,
            _topology_from_captured_state(external.file),
        )
    try:
        for selector in selectors:
            if cancelled():
                raise RuntimeError("admission cancelled")
            _trace_hdf5_object_dependencies(
                master,
                selector,
                paths=paths,
                seen=seen,
                cancelled=cancelled,
                required=True,
                states=states,
            )
    except SourceRevisionChanged:
        raise
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise
    except (OSError, ValueError) as error:
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise

    for path in dict.fromkeys(paths):
        key = _source_state_key(path)
        if key in states:
            continue
        try:
            _remember_source_state(
                path,
                states,
                cancelled=cancelled,
            )
        except FileNotFoundError as error:
            raise SourceRevisionChanged(
                f"required HDF5 dependency is still landing: {path}"
            ) from error
        except OSError as error:
            raise SourceRevisionChanged(
                "required HDF5 dependency capture is unverifiable: "
                f"{path}"
            ) from error

    frame_count = 0
    try:
        with _open_stable_hdf5_dependency(
            master,
            states,
            cancelled=cancelled,
        ) as handle:
            if handle is None:
                raise SourceRevisionChanged(
                    f"selected detector disappeared during admission: {master}"
                )
            for selector in selectors:
                value = handle.get(selector)
                if (
                    not isinstance(value, h5py.Dataset)
                    or value.ndim not in {2, 3}
                ):
                    raise ValueError(
                        f"selected detector dependency is unavailable: "
                        f"{master}:{selector}"
                    )
                count = 1 if value.ndim == 2 else int(value.shape[0])
                shape = (
                    tuple(int(item) for item in value.shape)
                    if value.ndim == 2
                    else tuple(int(item) for item in value.shape[1:])
                )
                if (
                    shape != descriptor.frame_shape
                    or np.dtype(value.dtype) != descriptor.dtype
                ):
                    raise ValueError(
                        "selected detector layout changed during admission: "
                        f"{master}:{selector}"
                    )
                frame_count += count
        if frame_count != descriptor.frame_count:
            raise ValueError(
                "selected detector frame count changed during admission: "
                f"descriptor={descriptor.frame_count}, "
                f"selected={frame_count}"
            )
    except SourceRevisionChanged:
        raise
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise
    except (OSError, ValueError) as error:
        try:
            _verify_source_states(states, cancelled=cancelled)
        except SourceRevisionChanged as drift:
            raise drift from error
        raise
    excluded = {
        _source_state_key(master),
        *(
            _source_state_key(value.file.path)
            for value in external_members
        ),
    }
    selected: list[SourceFileState] = []
    for path in dict.fromkeys(paths):
        key = _source_state_key(path)
        if key in excluded or any(
            _source_state_key(value.path) == key for value in selected
        ):
            continue
        state = states.get(key)
        if state is None:  # pragma: no cover - pre-freeze owns every path
            raise SourceRevisionChanged(
                f"required HDF5 dependency escaped its freeze: {path}"
            )
        selected.append(state.followed_state)
    _verify_source_states(states, cancelled=cancelled)
    return tuple(selected)


def _capture_bound_container_dependencies(
    binding: Any,
    descriptor: ContainerDescriptor,
    master: SourceFileState,
    *,
    cancelled: Callable[[], bool] | None,
    emit_external: Callable[[ExternalSourceState], None],
    emit_dependency: Callable[[SourceFileState], None],
) -> None:
    first = 0; master_path = Path(master.path).resolve()
    def detached(dataset):
        parent = Path(dataset.file.filename).resolve(); values = []
        for selector in dataset.virtual_sources():
            file_name = os.fsdecode(selector.file_name)
            candidate = parent if file_name in {"", "."} else Path(file_name)
            if not candidate.is_absolute(): candidate = parent.parent / candidate
            values.append((str(candidate.resolve()), "/" + os.fsdecode(selector.dset_name).lstrip("/")))
        return values
    def identity(pair):
        return (os.path.normcase(os.path.normpath(pair[0])), os.path.normpath(pair[1]))
    from xrd_tools.io.nexus import _selected_link_owner_selector
    seen: set[tuple[str, str]] = set()
    pending = []; dependency_paths: set[str] = set()
    for epoch, dataset in enumerate(binding._datasets):
        _cancelled(cancelled)
        try:
            logical_selector = binding.paths[epoch]
        except IndexError as error:
            raise ValueError("container detector binding lost its selector") from error
        owner_path, owner_selector = _selected_link_owner_selector(
            binding.entry_group, logical_selector, dataset,
        )
        seen.add(identity((str(owner_path), owner_selector)))
        before = SourceFileState.capture(owner_path)
        if dataset.ndim not in {2, 3}: raise ValueError("container detector rank changed")
        extent = 1 if dataset.ndim == 2 else int(dataset.shape[0])
        shape = tuple(dataset.shape if dataset.ndim == 2 else dataset.shape[1:])
        if shape != tuple(descriptor.frame_shape) or np.dtype(dataset.dtype) != descriptor.dtype:
            raise SourceRevisionChanged("container detector layout changed")
        if owner_path != master_path:
            emit_external(ExternalSourceState(
                before, owner_selector, first, first + extent, epoch,
            ))
        first += extent
        if owner_path == master_path and bool(dataset.is_virtual): pending.extend(detached(dataset))
        after = SourceFileState.capture(owner_path)
        if before != after or _path_key(_resolved(before)) != _path_key(_resolved(after)):
            raise SourceRevisionChanged("container detector owner changed")
    if first != descriptor.frame_count: raise SourceRevisionChanged("container detector extent changed")
    cursor = 0
    while cursor < len(pending):
        _cancelled(cancelled); selector = pending[cursor]; cursor += 1
        key = identity(selector)
        if key in seen: continue
        seen.add(key); dependency = binding.open_dependency_dataset(selector)
        try:
            dependency_path = Path(dependency.file.filename).resolve()
            before = SourceFileState.capture(dependency_path)
            dependency_key = _path_key(str(dependency_path))
            if dependency_path != master_path and dependency_key not in dependency_paths:
                dependency_paths.add(dependency_key)
                emit_dependency(before)
            if bool(dependency.is_virtual): pending.extend(detached(dependency))
            after = SourceFileState.capture(dependency_path)
            if before != after or _path_key(_resolved(before)) != _path_key(_resolved(after)):
                raise SourceRevisionChanged("container dependency changed")
        finally:
            binding.close_dependency_dataset()
class _AverageSourceReadWindow:
    def __init__(self, value: PreparedSourceExecutionGraph, *, cancelled=None,
                 direct_chunk_policy=None, prevalidated=False):
        self._graph = value; self._cancelled = cancelled; self._cursor = None
        self._tiff = None
        self._prevalidated = bool(prevalidated)
        self._direct_chunk_policy = direct_chunk_policy
        self._direct_chunk_state = None; self._direct_iterator = None
        self._terminal_cleanup_error = None
        self._closed = False; self._entered = False
    @property
    def extent(self) -> int: return self._graph.stamp.frame_count
    @property
    def closed(self) -> bool: return self._closed
    def __enter__(self) -> "_AverageSourceReadWindow":
        if self._entered: return self
        if not self._prevalidated:
            validate_source_execution_graph(self._graph, cancelled=self._cancelled)
        if self._graph.stamp.adapter_id == "nexus_hdf5":
            from xrd_tools.sources.cursor import ContainerCursor
            self._cursor = ContainerCursor(self._graph.source_path,
                entry=self._graph.execution_source.entry or "entry",
                metadata_input_policy="average_bounded_v1",
                expected_scanned_motor_names=self._graph.scanned_motor_names,
                expected_all_motor_names=self._graph.motor_names)
            self._cursor.open()
            if self._direct_chunk_policy is not None:
                from xrd_tools.sources.eiger_direct_chunk import EigerDirectChunkState
                from xrd_tools.sources.read_plan import plan_reads
                descriptor = self._cursor.descriptor
                policy = self._direct_chunk_policy
                plan = plan_reads(
                    descriptor.frame_count, descriptor.frame_shape,
                    descriptor.dtype, descriptor.chunks, policy.frame_bytes,
                    requested_block_frames=1, two_d=descriptor.is_2d,
                )
                self._direct_chunk_state = EigerDirectChunkState(policy)
                self._direct_iterator = self._cursor.iter_eiger_direct_blocks(
                    plan, policy, cancelled=self._cancelled or (lambda: False),
                    state=self._direct_chunk_state,
                )
        else:
            from xrd_tools.sources.image import TiffSeriesSource
            options = dict(self._graph.execution_source.options)
            self._tiff = TiffSeriesSource(
                tuple(item.path for item in self._graph.stamp.members),
                name=self._graph.group_key,
                metadata_format=options.get("metadata_format"),
                meta_dir=options.get("meta_dir"),
                detector_shape=options.get("detector_shape"),
                detector=options.get("detector"),
                raw_dtype=str(options.get("raw_dtype", "int32")),
                raw_header_skip=int(options.get("raw_header_skip", 0)),
                admitted_motor_values=tuple(
                    (item.source_path, item.motor, item.value)
                    for item in self._graph.stamp.admitted_motor_values
                ),
            )
        self._entered = True
        return self
    def read_native(self, logical_index: int) -> np.ndarray:
        _cancelled(self._cancelled); index = int(logical_index)
        if not 0 <= index < self.extent: raise IndexError(index)
        if self._direct_iterator is not None:
            try:
                block = next(self._direct_iterator)
            except StopIteration as error:
                _cancelled(self._cancelled)
                raise RuntimeError(
                    "Average direct source ended before its admitted extent"
                ) from error
            if (block.start, block.stop) != (index, index + 1):
                raise RuntimeError(
                    "Average direct source violated exact frame order"
                )
            array = np.asarray(block.array)
            if array.shape != (1, *self._graph.detector_shape):
                raise RuntimeError(
                    "Average direct source changed its detector layout"
                )
            return array[0]
        if self._cursor is not None: return self._cursor.read_frame(index)
        return np.asarray(self._tiff.load_frame(index + 1))
    def complete_metadata_for(self, logical_index: int) -> dict[str, Any]:
        _cancelled(self._cancelled); index = int(logical_index)
        if self._cursor is not None:
            return dict(self._cursor.metadata_provider().complete_metadata_for(index))
        return dict(getattr(self._tiff, "metadata_for")(
            index + 1, max_input_bytes=1 << 16,
        ))
    def eiger_direct_chunk_layout(self):
        if self._cursor is None:
            return None, "direct decode requires a container source"
        return self._cursor.eiger_direct_chunk_layout()
    def close(self) -> None:
        if self._closed: return
        iterator = self._direct_iterator
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    self._terminal_cleanup_error = SourceCleanupFailed(
                        f"AVERAGE_SOURCE_CLEANUP_FAILED({error})"
                    )
            self._direct_iterator = None
        if self._cursor is not None:
            try:
                self._cursor.close()
            except BaseException as error:
                raise SourceCleanupPending(
                    self, f"AVERAGE_SOURCE_CLEANUP_PENDING: {error}",
                ) from error
            if not self._cursor.closed:
                raise SourceCleanupPending(
                    self, "AVERAGE_SOURCE_CLEANUP_PENDING",
                )
        self._closed = True
        if self._terminal_cleanup_error is not None:
            error, self._terminal_cleanup_error = (
                self._terminal_cleanup_error, None,
            )
            raise error
    @property
    def direct_chunk_fact(self):
        state = self._direct_chunk_state
        return None if state is None else state.fact(self._graph.source_path)
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
def open_source_execution_graph(value: PreparedSourceExecutionGraph, *,
        cancelled: Callable[[], bool] | None = None,
        direct_chunk_policy=None, prevalidated: bool = False,
) -> _AverageSourceReadWindow:
    if value.reader_binding != "average_closed_v1":
        raise ValueError("Average source opening requires average_closed_v1")
    return _AverageSourceReadWindow(
        value, cancelled=cancelled,
        direct_chunk_policy=direct_chunk_policy,
        prevalidated=prevalidated,
    )
__all__ = [
    "AdmittedMetadataSource", "AdmittedMotorValue", "CanonicalSourceTarget",
    "ExternalSourceState", "PreparedSourceExecutionGraph", "SourceAliasBinding",
    "SelectedContainerInput", "SourceCleanupFailed", "SourceCleanupPending",
    "SourceExecutionIdentityV1", "SourceExecutionStamp", "SourceFileState",
    "SourceRevisionChanged", "append_source_from_execution_graph",
    "freeze_source_execution_graph", "open_source_execution_graph",
    "qualify_source_execution_graph", "requalify_source_execution_graph",
    "source_execution_identity_v1_projection", "source_execution_projection",
    "source_graph_digest", "source_graph_payload", "source_snapshots_projection",
    "stable_lineage_projection", "validate_source_execution_graph",
    "validate_source_state_sweep",
]
