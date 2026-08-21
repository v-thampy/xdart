from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import posixpath
from pathlib import Path
from typing import Any, Callable
import numpy as np
from xrd_tools.core.filters import compile_filter
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration import load_detector_calibration
from xrd_tools.io import AppendDisposition, AppendRefused, load_mask
from xrd_tools.io.output_path import OVERWRITE_MODE, resolve_output_target
from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
)
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent
from xrd_tools.sources.adapters import candidate_owner, get_adapter
from xrd_tools.sources.descriptor import ContainerDescriptor
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.directory_index import StaleCandidateError
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.run_plan import RunCandidatePlan
from xrd_tools.sources.selection import DirectorySourceSpec
from .contracts import (
    AcceptedScientificAssets, AdmittedMetadataSource,
    AdmittedMotorValue, AdmittedOutput, AdmissionReceipt,
    ExternalSourceState, OutputDisposition, OutputFact, PlannedOutput,
    SourceExecutionStamp, SourceFileState, StartCapture,
    threshold_pair_is_canonical,
)
from .source_metadata import (
    ordered_motor_intersection,
    read_image_motor_metadata,
)

load_poni = load_detector_calibration


class SourceRevisionChanged(ValueError):
    """Exact source bytes/dependencies drifted during one admitted attempt."""


@dataclass(frozen=True, slots=True)
class ValidatedSourceAliases:
    """Immutable result of the immediate two-sweep source proof."""

    identity: object
    targets: tuple[object, ...]


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
    owner = candidate_owner(Path(path))
    return None if owner is None else owner.id


def validate_source_aliases(
    stamp: SourceExecutionStamp,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ValidatedSourceAliases:
    """Prove every raw alias and target twice in deterministic order."""

    if type(stamp) is not SourceExecutionStamp:
        raise TypeError("source alias validation requires an execution stamp")
    is_cancelled = _not_cancelled if cancelled is None else cancelled
    targets = stamp.canonical_targets
    for sweep in range(2):
        for binding in stamp.source_aliases:
            if is_cancelled():
                raise RuntimeError("admission cancelled")
            try:
                resolved = _resolve_source_alias(binding.raw_path)
            except (OSError, RuntimeError) as error:
                raise SourceRevisionChanged(
                    f"source alias is unavailable: {binding.raw_path}"
                ) from error
            if _resolved_source_key(resolved) != _resolved_source_key(
                binding.resolved_path
            ):
                raise SourceRevisionChanged(
                    (
                        "source candidate changed after admission: "
                        if binding.candidate_owner_id is not None
                        else "source alias retargeted after admission: "
                    )
                    + binding.raw_path
                )
            expected = targets[binding.target_id]
            try:
                current = _capture_canonical_source_target(resolved)
            except OSError as error:
                raise SourceRevisionChanged(
                    f"source target is unavailable: {binding.resolved_path}"
                ) from error
            if current != expected.state:
                raise SourceRevisionChanged(
                    "source target changed after admission: "
                    f"{binding.resolved_path}"
                )
            if sweep == 0 and binding.candidate_owner_id is not None:
                if _candidate_owner_id(binding.raw_path) != binding.candidate_owner_id:
                    raise SourceRevisionChanged(
                        "source candidate owner changed after admission: "
                        f"{binding.raw_path}"
                    )
    return ValidatedSourceAliases(stamp.execution_identity_v1, targets)


@dataclass(frozen=True, slots=True)
class OutputCandidate:
    source: SourceSpec | DirectorySourceSpec
    poni_file: str
    mask_file: str
    save_path: str
    processing_json: str
    fingerprint: str
    _configuration: FrozenRunConfiguration | None = field(
        default=None, repr=False, compare=False,
    )
    def __post_init__(self) -> None:
        texts = (
            self.poni_file, self.mask_file, self.save_path,
            self.processing_json, self.fingerprint,
        )
        if (
            type(self.source) not in {SourceSpec, DirectorySourceSpec}
            or not all(type(value) is str for value in texts)
            or type(json.loads(self.processing_json)) is not dict
        ):
            raise ValueError("output candidate is invalid")
    @classmethod
    def from_start_capture(
        cls, capture: StartCapture,
        assets: AcceptedScientificAssets | None = None,
        gi_motor_choices: tuple[str, ...] | None = None,
    ) -> "OutputCandidate":
        intent = capture.intent_snapshot.thaw()
        if not threshold_pair_is_canonical(intent.threshold):
            raise ValueError(
                "degenerate threshold identity at admission (apply_threshold "
                "== mask_saturation); the start capture canonicalizes this "
                "pair — refusing to sign a configuration that would not "
                "describe its own execution"
            )
        if (
            str(intent.output_mode).strip().lower() == "append"
            and intent.processing_mode == "Int 1D (XYE)"
        ):
            raise ValueError(
                "XYE-only Append has no persisted lineage owner"
            )
        source = intent.source_spec
        if type(source) not in {SourceSpec, DirectorySourceSpec}:
            raise ValueError("output admission requires a supported source")
        accepted = assets or _load_scientific_assets(intent)
        frozen = intent.freeze(gi_motor_choices=gi_motor_choices)
        processing = frozen.as_provenance()
        processing["accepted_scientific_assets"] = {
            "poni_values": (
                None if accepted.poni is None else accepted.poni.to_dict()
            ),
            "poni_detector_config_json": accepted.poni_detector_config_json,
            "poni_sha256": accepted.poni_sha256,
            "mask_sha256": accepted.mask_sha256,
        }
        return cls(
            source, str(intent.poni_file), str(intent.mask_file), str(intent.save_path),
            json.dumps(
                processing, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ),
            frozen.fingerprint,
            frozen,
        )
    def processing_mapping(self) -> dict[str, Any]:
        return json.loads(self.processing_json)
    def matches(self, configuration: FrozenRunConfiguration) -> bool:
        return (
            configuration.thaw_source_spec() == self.source
            and configuration.save_path == self.save_path
            and configuration.fingerprint == self.fingerprint
        )


@dataclass(frozen=True, slots=True)
class DeferredDirectoryEntry:
    """One name/stat-qualified output group awaiting content admission."""

    candidates: tuple[Candidate, ...]
    target: Path
    fact: OutputFact
    physical_paths: tuple[Path, ...]
    protected_states: tuple[SourceFileState, ...]
    skip_reason: str = ""
    _protected_topology: tuple[_CapturedSourceTopology, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not (
            type(self.candidates) is tuple
            and self.candidates
            and all(type(value) is Candidate for value in self.candidates)
            and isinstance(self.target, Path)
            and type(self.fact) is OutputFact
            and type(self.physical_paths) is tuple
            and self.physical_paths
            and all(
                isinstance(value, Path) and value.is_absolute()
                for value in self.physical_paths
            )
            and len(set(self.physical_paths)) == len(self.physical_paths)
            and type(self.protected_states) is tuple
            and all(
                type(value) is SourceFileState
                for value in self.protected_states
            )
            and len({value.path for value in self.protected_states})
            == len(self.protected_states)
            and type(self.skip_reason) is str
            and type(self._protected_topology) is tuple
            and all(
                type(value) is _CapturedSourceTopology
                for value in self._protected_topology
            )
            and (
                not self._protected_topology
                or tuple(
                    value.followed_state
                    for value in self._protected_topology
                ) == self.protected_states
            )
        ):
            raise TypeError("deferred directory entry is invalid")


@dataclass(frozen=True, slots=True)
class DeferredDirectoryPlan:
    """Frozen directory names/targets consumed one output group at a time."""

    candidates: RunCandidatePlan
    entries: tuple[DeferredDirectoryEntry, ...]
    discovered_paths: tuple[Path, ...]
    live: bool = False

    def __post_init__(self) -> None:
        flattened = tuple(
            candidate
            for entry in self.entries
            for candidate in entry.candidates
        )
        physical = tuple(
            path for entry in self.entries for path in entry.physical_paths
        )
        common = (
            type(self.candidates) is RunCandidatePlan
            and type(self.entries) is tuple
            and all(type(value) is DeferredDirectoryEntry for value in self.entries)
            and type(self.discovered_paths) is tuple
            and all(
                isinstance(value, Path) and value.is_absolute()
                for value in self.discovered_paths
            )
            and len(set(self.discovered_paths)) == len(self.discovered_paths)
            and type(self.live) is bool
        )
        finite = (
            bool(self.entries)
            and len(set(physical)) == len(physical)
            and set(physical).issubset(self.discovered_paths)
            and len({candidate.path for candidate in flattened})
            == len(flattened)
            and set(flattened) == set(self.candidates.candidates)
        )
        watching = (
            not self.entries
            and set(self.candidates.paths) == set(self.discovered_paths)
        )
        if not (common and (watching if self.live else finite)):
            raise TypeError("deferred directory plan is invalid")

    @property
    def discovered_file_count(self) -> int:
        return len(self.discovered_paths)

    @property
    def targets(self) -> tuple[Path, ...]:
        return tuple(entry.target for entry in self.entries)


@dataclass(frozen=True, slots=True)
class LiveDirectoryGroup:
    """One exact name/stat revision observed by a directory Live owner."""

    plan: RunCandidatePlan
    target: Path
    physical_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not (
            type(self.plan) is RunCandidatePlan
            and bool(self.plan.candidates)
            and isinstance(self.target, Path)
            and type(self.physical_paths) is tuple
            and self.physical_paths
            and self.physical_paths == self.plan.paths
            and all(path.is_absolute() for path in self.physical_paths)
        ):
            raise TypeError("live directory group is invalid")

    @property
    def revision(self) -> tuple[Candidate, ...]:
        return self.plan.candidates


@dataclass(frozen=True, slots=True)
class LiveDirectoryAttempt:
    """Typed result of one exact, Stop-aware Live group probe."""

    group: LiveDirectoryGroup
    state: ProbeState
    decision: AdmittedOutput | None = None
    reason: str = ""
    revision_changed: bool = False

    def __post_init__(self) -> None:
        if not (
            type(self.group) is LiveDirectoryGroup
            and type(self.state) is ProbeState
            and (
                self.decision is None
                or type(self.decision) is AdmittedOutput
            )
            and (self.state is ProbeState.READY)
            == (self.decision is not None)
            and type(self.reason) is str
            and type(self.revision_changed) is bool
            and (
                not self.revision_changed
                or self.state is ProbeState.IN_PROGRESS
            )
        ):
            raise TypeError("live directory attempt is invalid")


def _validate_exact_tiff_gi_motor(
    intent: RunIntent,
    items: tuple[PlannedOutput, ...],
) -> None:
    """Require the selected metadata motor in every admitted TIFF group."""

    if not intent.gi.enabled:
        return
    raw = intent.gi.incidence_motor
    if raw == "Manual":
        return
    for item in items:
        if item.source_spec.kind is not SourceKind.TIFF_SERIES:
            continue
        motors = item.group.motor_names
        stamp = item.source_stamp
        values = stamp.admitted_motor_values
        if (
            motors is None
            or raw not in motors
            or stamp.adapter_id != "tiff_series"
            or not stamp.members
            or len(values) != len(stamp.members)
            or len(values) != stamp.frame_count
            or any(value.motor != raw for value in values)
        ):
            raise ValueError(
                f"GI metadata motor '{raw}' must have a finite value in "
                "every admitted TIFF; select Manual deliberately for one "
                "fixed angle across the whole run, or choose a "
                "frame-complete metadata motor."
            )


def _directory_matching_paths(
    source: DirectorySourceSpec,
    *,
    cancelled: Callable[[], bool],
) -> tuple[Path, ...]:
    """Freeze the physical suffix/filter universe shown by Run counters."""

    root = Path(source.root).expanduser().absolute()
    suffixes = tuple(str(value).casefold() for value in source.suffixes)
    name_ok = compile_filter(source.name_filter)
    iterator = root.rglob("*") if source.recursive else root.iterdir()
    paths: list[Path] = []
    for path in iterator:
        if cancelled():
            raise RuntimeError("admission cancelled")
        if not path.is_file():
            continue
        try:
            if len(path.absolute().relative_to(root).parts) > 2:
                continue
        except ValueError:
            continue
        low = path.name.casefold()
        suffix = next(
            (value for value in suffixes if low.endswith(value)),
            "",
        )
        if suffixes and not suffix:
            continue
        stem = path.name[:-len(suffix)] if suffix else path.name
        if name_ok(stem):
            paths.append(path.absolute())
    return tuple(sorted(dict.fromkeys(paths)))


def _bounded_directory_plan(plan: RunCandidatePlan) -> RunCandidatePlan:
    """Keep matching files at the selected root plus one immediate level."""

    aligned = tuple(
        (candidate, descriptor)
        for candidate, descriptor in zip(
            plan.candidates, plan.descriptors, strict=True,
        )
        if len(candidate.path.absolute().relative_to(
            plan.root.absolute()
        ).parts) <= 2
    )
    return RunCandidatePlan(
        plan.generation,
        tuple(value[0] for value in aligned),
        plan.root,
        plan.recursive,
        plan.name_filter,
        tuple(value[1] for value in aligned),
    )


def prepare_output(
    capture: StartCapture, *, cancelled: Callable[[], bool],
    session_owner: Callable[[DirectoryIndexSession | None], None],
    targets_owner: Callable[[tuple[Path, ...]], None] | None = None,
) -> AdmissionReceipt:
    if (
        type(capture) is not StartCapture
        or type(capture.intent_snapshot) is not RunIntentSnapshot
    ):
        raise TypeError("admission requires typed StartCapture")
    snapshot = capture.intent_snapshot
    intent = snapshot.thaw()
    assets = _load_scientific_assets(intent)
    candidate = OutputCandidate.from_start_capture(capture, assets)
    source, choices, deferred = candidate.source, None, None
    directory_discovered_paths: tuple[Path, ...] = ()
    if cancelled():
        raise RuntimeError("admission cancelled")
    if type(source) is DirectorySourceSpec:
        session = DirectoryIndexSession(
            retry_deadline=(float("inf") if intent.live_mode else None),
            probe_candidates=False,
        )
        session_owner(session)
        session.configure(
            source.root, recursive=source.recursive,
            name_filter=source.name_filter, suffixes=source.suffixes,
        )
        observation = session.observe(refresh=True)
        discovered = observation.discovered_snapshot
        complete_plan = RunCandidatePlan.from_snapshot(discovered)
        bounded_plan = _bounded_directory_plan(complete_plan)
        excluded = tuple(
            candidate
            for candidate in complete_plan.candidates
            if candidate not in bounded_plan.candidates
        )
        if intent.live_mode:
            directory_discovered_paths = tuple(
                candidate.path.absolute()
                for candidate in bounded_plan.candidates
            )
            deferred = DeferredDirectoryPlan(
                bounded_plan,
                (),
                directory_discovered_paths,
                live=True,
            )
            items = ()
        else:
            directory_discovered_paths = _directory_matching_paths(
                source,
                cancelled=cancelled,
            )
        if (
            not intent.gi.enabled
            and not intent.live_mode
        ):
            deferred = _deferred_directory_plan(
                candidate,
                bounded_plan,
                directory_discovered_paths,
                cancelled=cancelled,
            )
            items = ()
        elif not intent.live_mode:
            session.enable_probes(exclude=excluded)
            while observation.unprobed_count:
                if cancelled():
                    raise RuntimeError("admission cancelled")
                observation = session.observe(refresh=False)
            plan = RunCandidatePlan.from_observation(observation)
            items = _directory_items(candidate, plan, cancelled=cancelled)
            _validate_exact_tiff_gi_motor(intent, items)
            choices = _motor_choices(items)
            candidate = OutputCandidate.from_start_capture(
                capture, assets, choices
            )
    else:
        items = (_series_item(candidate, source, cancelled=cancelled),)
        _validate_exact_tiff_gi_motor(snapshot.thaw(), items)
        if source.kind is SourceKind.TIFF_SERIES:
            choices = _motor_choices(items)
            candidate = OutputCandidate.from_start_capture(
                capture, assets, choices
            )
    if deferred is None:
        _validate_targets(
            candidate,
            source,
            items,
            cancelled=cancelled,
        )
    if targets_owner is not None:
        targets = (
            deferred.targets
            if deferred is not None
            else tuple(dict.fromkeys(item.target for item in items))
        )
        if targets:
            targets_owner(targets)
    outputs = tuple(inspect_output(item, candidate) for item in items)
    if cancelled() or not outputs and deferred is None:
        raise RuntimeError(
            "admission cancelled" if cancelled()
            else "source has no READY candidates"
        )
    return AdmissionReceipt(
        capture.request_id, snapshot.revision, capture.source_capture,
        candidate, outputs, assets, choices,
        deferred_directory=deferred,
        directory_discovered_file_count=len(directory_discovered_paths),
        directory_discovered_paths=directory_discovered_paths,
    )
def inspect_output(
    item: PlannedOutput,
    configuration: FrozenRunConfiguration | OutputCandidate,
    fact: OutputFact | None = None,
) -> AdmittedOutput:
    from .adapters.dynamic_output import _supported_lineage_frame_count

    start = item.source_stamp.first_label
    count = _supported_lineage_frame_count(item.source_stamp)
    accepted = fact if type(fact) is OutputFact else OutputFact(_path_state(item.target))
    labels = tuple(range(start, start + count))
    frozen = (
        configuration._configuration
        if type(configuration) is OutputCandidate
        else configuration
    )
    if frozen is not None and frozen.output_mode == "Append":
        from .adapters.dynamic_output import preview_append_decision

        if type(configuration) is not OutputCandidate:
            raise TypeError(
                "Append inspection requires the signed output candidate"
            )
        append = preview_append_decision(
            frozen,
            item,
            configuration.processing_mapping(),
        )
        if append.disposition is AppendDisposition.REFUSE:
            raise AppendRefused(append)
        labels = tuple(append.write_labels)
    return AdmittedOutput(
        item, OutputDisposition.WRITE, labels, accepted,
        "" if frozen is None or frozen.output_mode != "Append" else append.reason,
    )


def _directory_candidate_groups(
    plan: RunCandidatePlan,
) -> tuple[tuple[Candidate, ...], ...]:
    groups: list[tuple[Candidate, ...]] = []
    consumed: set[Path] = set()
    for candidate in plan.candidates:
        if candidate.path in consumed:
            continue
        adapter = get_adapter(candidate.adapter_id)
        if adapter is None:
            raise ValueError(
                f"directory candidate lost adapter {candidate.adapter_id!r}"
            )
        if SourceKind.IMAGE_FILE in adapter.kinds:
            name = adapter.scan_name(candidate.path)
            members = tuple(
                value
                for value in plan.candidates
                if (
                    value.adapter_id == candidate.adapter_id
                    and value.path.parent == candidate.path.parent
                    and adapter.scan_name(value.path) == name
                )
            )
        else:
            members = (candidate,)
        consumed.update(value.path for value in members)
        groups.append(members)
    return tuple(groups)


def _directory_target(
    configuration: FrozenRunConfiguration | OutputCandidate,
    plan: RunCandidatePlan,
    candidate: Candidate,
    name: str,
) -> Path:
    output_root = Path(configuration.save_path)
    output_request = configuration.save_path
    if not output_root.suffix:
        try:
            relative_parent = candidate.path.parent.relative_to(plan.root)
        except ValueError as error:
            raise ValueError(
                "directory candidate escaped admitted root: "
                f"{candidate.path}"
            ) from error
        output_directory = output_root / relative_parent
        try:
            resolved_root = output_root.resolve(strict=False)
            resolved_output = output_directory.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"directory output could not be resolved: {output_directory}"
            ) from error
        if not resolved_output.is_relative_to(resolved_root):
            raise ValueError(
                "directory output escaped selected root: "
                f"{output_directory}"
            )
        output_request = str(output_directory)
    return _resolved_generated_target(output_request, name)


def _candidate_still_exact(candidate: Candidate) -> bool:
    try:
        stat = candidate.path.stat()
    except OSError:
        return False
    owner = candidate_owner(candidate.path)
    return (
        owner is not None
        and owner.id == candidate.adapter_id
        and (int(stat.st_size), int(stat.st_mtime_ns))
        == candidate.version_stamp
    )


def _validate_exact_candidate_state(
    candidate: Candidate,
    state: SourceFileState,
    *,
    require_current: bool = True,
) -> None:
    """Bind one cheap Candidate to one still-current strong file state."""

    owner = candidate_owner(candidate.path)
    if (
        _source_state_key(state.path)
        != _source_state_key(candidate.path)
        or owner is None
        or owner.id != candidate.adapter_id
        or (state.size, state.mtime_ns) != candidate.version_stamp
        or (require_current and not state.matches_disk())
    ):
        raise SourceRevisionChanged(
            f"source candidate changed during admission: {candidate.path}"
        )


def _capture_exact_candidate_state(
    candidate: Candidate,
) -> SourceFileState:
    try:
        state = SourceFileState.capture(candidate.path)
    except OSError as error:
        raise SourceRevisionChanged(
            f"source candidate disappeared during admission: {candidate.path}"
        ) from error
    # This is the one strong capture required by the binding operation.  The
    # caller owns the immediate Stop boundary before any later current-state
    # revalidation is allowed to perform an equality-equivalent recapture.
    _validate_exact_candidate_state(
        candidate,
        state,
        require_current=False,
    )
    return state


def _classify_inventory_failure(
    candidate: Candidate,
    states: dict[str, _CapturedSourceTopology],
    error: BaseException,
    *,
    cancelled: Callable[[], bool],
) -> None:
    """Upgrade a failed HDF5 inspection only when exact drift is proven."""

    try:
        accepted = states.get(_source_state_key(candidate.path))
        if (
            accepted is None
            or accepted.candidate_owner_id != candidate.adapter_id
        ):
            raise SourceRevisionChanged(
                "source candidate lost its captured topology during admission: "
                f"{candidate.path}"
            )
        _verify_source_states(states, cancelled=cancelled)
    except SourceRevisionChanged as drift:
        raise drift from error


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


def _hdf5_dataset_dependency_paths(
    dataset: object,
    *,
    cancelled: Callable[[], bool],
    required: bool = False,
    states: dict[str, _CapturedSourceTopology] | None = None,
) -> tuple[Path, ...]:
    """Return the full external-storage/VDS closure for one dataset."""

    current_file = Path(_raw_source_path(dataset.file.filename))
    seen = {
        (
            os.path.normcase(os.path.realpath(current_file)),
            _hdf5_object_path(dataset.name),
        )
    }
    paths: list[Path] = []
    accepted_states = {} if states is None else states
    _extend_hdf5_dataset_dependency_paths(
        dataset,
        paths=paths,
        seen=seen,
        cancelled=cancelled,
        required=required,
        states=accepted_states,
    )
    return tuple(dict.fromkeys(paths))


def _scan_all_external_links(
    root: object,
    *,
    cancelled: Callable[[], bool],
    states: dict[str, _CapturedSourceTopology],
) -> tuple[Path, ...]:
    """Walk hard-linked groups and collect links without opening datasets."""

    import h5py

    external: list[Path] = []
    seen: set[tuple[str, int]] = set()
    traced: set[tuple[str, str]] = set()

    def visit(group: object) -> None:
        if cancelled():
            raise RuntimeError("admission cancelled")
        try:
            address = int(h5py.h5o.get_info(group.id).addr)
        except Exception:
            address = hash(group.id)
        file_name = os.path.normcase(os.path.realpath(group.file.filename))
        identity = (file_name, address)
        if identity in seen:
            return
        seen.add(identity)
        for raw_name in group:
            if cancelled():
                raise RuntimeError("admission cancelled")
            name = str(raw_name)
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                dependency = _hdf5_link_file(group, link.filename)
                external.append(dependency)
                _trace_hdf5_object_dependencies(
                    dependency,
                    os.fsdecode(link.path),
                    paths=external,
                    seen=traced,
                    cancelled=cancelled,
                    required=False,
                    states=states,
                )
                value = group.get(name)
                if isinstance(value, h5py.Group):
                    visit(value)
                elif isinstance(value, h5py.Dataset):
                    external.extend(
                        _hdf5_dataset_dependency_paths(
                            value,
                            cancelled=cancelled,
                            states=states,
                        )
                    )
                continue
            if isinstance(link, h5py.SoftLink):
                target = os.fsdecode(link.path)
                if not target.startswith("/"):
                    target = posixpath.join(group.name, target)
                _trace_hdf5_object_dependencies(
                    Path(os.fsdecode(group.file.filename)),
                    target,
                    paths=external,
                    seen=traced,
                    cancelled=cancelled,
                    required=False,
                    states=states,
                )
                try:
                    value = group.get(name)
                except (KeyError, OSError, RuntimeError):
                    value = None
                if isinstance(value, h5py.Group):
                    visit(value)
                elif isinstance(value, h5py.Dataset):
                    external.extend(
                        _hdf5_dataset_dependency_paths(
                            value,
                            cancelled=cancelled,
                            states=states,
                        )
                    )
                continue
            try:
                kind = group.get(name, getclass=True)
            except (KeyError, OSError, RuntimeError):
                continue
            if kind is h5py.Group:
                child = group.get(name)
                if child is not None:
                    visit(child)
            elif kind is h5py.Dataset:
                dataset = group.get(name)
                if dataset is not None:
                    external.extend(
                        _hdf5_dataset_dependency_paths(
                            dataset,
                            cancelled=cancelled,
                            states=states,
                        )
                    )

    visit(root)
    return tuple(dict.fromkeys(external))


def _external_link_inventory(
    candidate: Candidate,
    *,
    candidate_states: tuple[_CapturedSourceTopology, ...],
    cancelled: Callable[[], bool],
) -> tuple[
    tuple[Path, ...],
    tuple[_CapturedSourceTopology, ...],
    str,
]:
    """Inventory pre-write HDF5 dependencies without eager frame counting.

    A canonical 3-D detector dataset or Eiger link wins before the reader's
    recursive fallbacks, so its direct link neighborhood is a complete and
    cheap proof.  A non-canonical layout may be resolved by a recursive
    detector marker, NXdata/NXdetector group, or largest-dataset fallback; only
    those uncommon files receive a complete link-only tree walk.  Dataset
    pixels and frame counts remain just-in-time in both cases.

    This split preserves global output safety: a generated target is checked
    against every ExternalLink that the eventual resolver could consume before
    any earlier output is written.  Soft-linked canonical paths deliberately
    take the complete-walk branch because their final link target cannot be
    proved from the canonical leaf alone.
    """

    states = {
        _source_state_key(topology.raw_path): topology
        for topology in candidate_states
    }
    accepted_candidate = states.get(_source_state_key(candidate.path))
    if accepted_candidate is None:
        raise TypeError("HDF5 inventory lost its exact candidate capture")
    _validate_exact_candidate_state(
        candidate,
        accepted_candidate.followed_state,
    )
    adapter = get_adapter(candidate.adapter_id)
    if adapter is None:
        raise ValueError(f"candidate {candidate.path} lost its adapter")
    if not any(
        kind in {
            SourceKind.NEXUS_STACK,
            SourceKind.EIGER_MASTER,
            SourceKind.PROCESSED_NEXUS,
        }
        for kind in adapter.kinds
    ):
        return (), (), ""
    if cancelled():
        raise RuntimeError("admission cancelled")
    try:
        import h5py
        _remember_source_state(
            candidate.path,
            states,
            cancelled=cancelled,
            candidate_owner_id=candidate.adapter_id,
        )
        is_hdf5 = bool(h5py.is_hdf5(candidate.path))
        _verify_source_states(states, cancelled=cancelled)
    except SourceRevisionChanged:
        raise
    except Exception as error:
        _classify_inventory_failure(
            candidate,
            states,
            error,
            cancelled=cancelled,
        )
        raise ValueError(
            "source dependency type could not be established: "
            f"{candidate.path}: {type(error).__name__}: {error}"
        ) from error
    if not is_hdf5:
        if not _candidate_still_exact(candidate):
            raise SourceRevisionChanged(
                f"source candidate changed during admission: {candidate.path}"
            )
        return (
            (),
            tuple(states.values()),
            "not a readable HDF5 container",
        )

    external: list[Path] = []
    traced: set[tuple[str, str]] = set()

    def record_external(
        parent: object,
        link: object,
        *,
        required: bool = False,
    ) -> None:
        if not isinstance(link, h5py.ExternalLink):
            return
        dependency = _hdf5_link_file(parent, link.filename)
        external.append(dependency)
        _trace_hdf5_object_dependencies(
            dependency,
            os.fsdecode(link.path),
            paths=external,
            seen=traced,
            cancelled=cancelled,
            required=required,
            states=states,
        )

    def trace_indirect_link(
        parent: object,
        link: object,
        *,
        required: bool = False,
        selected_paths: list[Path] | None = None,
        selected_seen: set[tuple[str, str]] | None = None,
        selected_states: dict[str, _CapturedSourceTopology] | None = None,
    ) -> None:
        """Freeze an indirect child before any caller dereferences it."""

        accepted_paths = external if selected_paths is None else selected_paths
        accepted_seen = traced if selected_seen is None else selected_seen
        accepted_states = states if selected_states is None else selected_states
        if isinstance(link, h5py.ExternalLink):
            dependency = _hdf5_link_file(parent, link.filename)
            accepted_paths.append(dependency)
            _trace_hdf5_object_dependencies(
                dependency,
                os.fsdecode(link.path),
                paths=accepted_paths,
                seen=accepted_seen,
                cancelled=cancelled,
                required=required,
                states=accepted_states,
            )
        elif isinstance(link, h5py.SoftLink):
            target = os.fsdecode(link.path)
            if not target.startswith("/"):
                target = posixpath.join(parent.name, target)
            _trace_hdf5_object_dependencies(
                Path(os.fsdecode(parent.file.filename)),
                target,
                paths=accepted_paths,
                seen=accepted_seen,
                cancelled=cancelled,
                required=required,
                states=accepted_states,
            )

    def direct_3d_dataset(group: object, name: str) -> bool:
        """Whether one non-soft canonical leaf resolves to a 3-D dataset."""

        link = group.get(name, getlink=True)
        if isinstance(link, h5py.SoftLink):
            return False
        record_external(group, link, required=True)
        try:
            value = group.get(name)
        except (KeyError, OSError, RuntimeError):
            return False
        if not isinstance(value, h5py.Dataset) or value.ndim != 3:
            return False
        external.extend(
            _hdf5_dataset_dependency_paths(
                value,
                cancelled=cancelled,
                required=True,
                states=states,
            )
        )
        return True

    def has_apstools_flat_contract(entry: object | None) -> bool:
        if entry is None:
            return False
        from xrd_tools.sources.descriptor import (
            _apstools_flat_nxdata_contract,
            _apstools_flat_stack_paths,
        )

        trial_paths: list[Path] = []
        trial_seen = set(traced)
        trial_states = dict(states)
        entry_file = Path(os.fsdecode(entry.file.filename))
        selectors: tuple[str, ...] = ()
        with _open_stable_hdf5_dependency(
            entry_file,
            trial_states,
            cancelled=cancelled,
        ) as trial_handle:
            if trial_handle is None:
                return False
            try:
                trial_entry = trial_handle.get(entry.name)
            except (KeyError, OSError, RuntimeError):
                trial_entry = None
            if not isinstance(trial_entry, h5py.Group):
                return False
            for selector in (
                posixpath.join(trial_entry.name, "data"),
                posixpath.join(
                    trial_entry.name, "instrument", "bluesky"
                ),
            ):
                _trace_hdf5_object_dependencies(
                    entry_file,
                    selector,
                    paths=trial_paths,
                    seen=trial_seen,
                    cancelled=cancelled,
                    required=False,
                    states=trial_states,
                )
            if not _apstools_flat_nxdata_contract(
                trial_handle,
                trial_entry,
            ):
                return False
            data_group = trial_entry.get("data")
            child_dependencies: dict[str, list[Path]] = {}
            inspection_states = dict(trial_states)
            if isinstance(data_group, h5py.Group):
                for raw_name in data_group:
                    name = str(raw_name)
                    try:
                        link = data_group.get(name, getlink=True)
                    except (KeyError, OSError, RuntimeError):
                        continue
                    if not isinstance(
                        link,
                        (h5py.ExternalLink, h5py.SoftLink),
                    ):
                        continue
                    child_paths: list[Path] = []
                    child_seen = set(trial_seen)
                    trace_indirect_link(
                        data_group,
                        link,
                        selected_paths=child_paths,
                        selected_seen=child_seen,
                        selected_states=inspection_states,
                    )
                    child_dependencies[
                        _hdf5_object_path(
                            posixpath.join(data_group.name, name)
                        )
                    ] = child_paths
            selectors = tuple(_apstools_flat_stack_paths(trial_entry))
            for selector in selectors:
                child = child_dependencies.get(
                    _hdf5_object_path(selector)
                )
                if child is not None:
                    trial_paths.extend(child)
                    for path in child:
                        key = _source_state_key(path)
                        child_state = inspection_states.get(key)
                        if child_state is not None:
                            trial_states.setdefault(key, child_state)
                _trace_hdf5_object_dependencies(
                    entry_file,
                    selector,
                    paths=trial_paths,
                    seen=trial_seen,
                    cancelled=cancelled,
                    required=True,
                    states=trial_states,
                )
        _verify_source_states(trial_states, cancelled=cancelled)
        external.extend(trial_paths)
        traced.clear()
        traced.update(trial_seen)
        states.clear()
        states.update(trial_states)
        return True

    def has_decisive_detector(entry: object | None) -> bool:
        """Match only resolver arms that precede every recursive fallback."""

        if entry is None:
            return False
        try:
            from xrd_tools.io.processed_scan_id import is_processed_xdart_file
            entry_name = entry.name.strip("/").split("/")[-1]
            if is_processed_xdart_file(entry.file, entry_name):
                return True
        except Exception:
            pass
        if has_apstools_flat_contract(entry):
            return True
        try:
            data_link = entry.get("data", getlink=True)
        except (KeyError, OSError, RuntimeError):
            data_link = None
        trace_indirect_link(entry, data_link)
        try:
            data = entry.get("data")
        except (KeyError, OSError, RuntimeError):
            data = None
        if not isinstance(data_link, h5py.SoftLink) and isinstance(
            data, h5py.Group
        ):
            # The descriptor treats every direct ExternalLink in entry/data as
            # an Eiger segment and gives that set absolute precedence.
            eiger_links = False
            for name in data:
                link = data.get(str(name), getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    record_external(data, link, required=True)
                    eiger_links = True
                    value = data.get(str(name))
                    if isinstance(value, h5py.Dataset):
                        external.extend(
                            _hdf5_dataset_dependency_paths(
                                value,
                                cancelled=cancelled,
                                required=True,
                                states=states,
                            )
                        )
            if eiger_links:
                return True
            if direct_3d_dataset(data, "data"):
                return True

        try:
            instrument_link = entry.get("instrument", getlink=True)
        except (KeyError, OSError, RuntimeError):
            instrument_link = None
        trace_indirect_link(entry, instrument_link)
        try:
            instrument = entry.get("instrument")
        except (KeyError, OSError, RuntimeError):
            instrument = None
        if isinstance(instrument_link, h5py.SoftLink) or not isinstance(
            instrument, h5py.Group
        ):
            return False
        try:
            detector_link = instrument.get("detector", getlink=True)
        except (KeyError, OSError, RuntimeError):
            detector_link = None
        trace_indirect_link(instrument, detector_link)
        try:
            detector = instrument.get("detector")
        except (KeyError, OSError, RuntimeError):
            detector = None
        if (
            not isinstance(detector_link, h5py.SoftLink)
            and isinstance(detector, h5py.Group)
            and direct_3d_dataset(detector, "data")
        ):
            return True
        for name in instrument:
            link = instrument.get(str(name), getlink=True)
            if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                record_external(instrument, link)
                # Resolver precedence follows instrument child order.  Once
                # an indirect child is encountered, a later local child cannot
                # prove which landed detector wins without following the full
                # dependency tree.
                return False
            try:
                group = instrument.get(str(name))
            except (KeyError, OSError, RuntimeError):
                continue
            if isinstance(group, h5py.Group) and direct_3d_dataset(
                group, "data"
            ):
                return True
        return False

    try:
        with _open_stable_hdf5_dependency(
            candidate.path,
            states,
            cancelled=cancelled,
        ) as handle:
            if handle is None:
                raise SourceRevisionChanged(
                    f"source candidate disappeared during admission: "
                    f"{candidate.path}"
                )
            root_names = tuple(str(raw_name) for raw_name in handle)
            inspected: dict[
                str,
                tuple[
                    bool,
                    str,
                    list[Path],
                    set[tuple[str, str]],
                    dict[str, _CapturedSourceTopology],
                ],
            ] = {}

            def root_group_facts(value: object) -> tuple[bool, str]:
                if not isinstance(value, h5py.Group):
                    return False, ""
                raw_class = value.attrs.get("NX_class", "")
                if isinstance(raw_class, bytes):
                    return True, raw_class.decode(
                        "utf-8", errors="replace"
                    )
                if isinstance(raw_class, np.ndarray):
                    first = raw_class.ravel()[0] if raw_class.size else ""
                    return True, (
                        first.decode("utf-8", errors="replace")
                        if isinstance(first, bytes)
                        else str(first)
                    )
                return True, str(raw_class or "")

            def inspect_root_group(name: str) -> tuple[
                bool,
                str,
                list[Path],
                set[tuple[str, str]],
                dict[str, _CapturedSourceTopology],
            ]:
                cached = inspected.get(name)
                if cached is not None:
                    return cached
                trial_paths: list[Path] = []
                trial_seen = set(traced)
                trial_states = dict(states)
                is_group = False
                nx_class = ""
                try:
                    root_link = handle.get(name, getlink=True)
                except (KeyError, OSError, RuntimeError):
                    root_link = None
                if not isinstance(
                    root_link,
                    (h5py.ExternalLink, h5py.SoftLink),
                ):
                    # A hard-linked root object belongs wholly to the already
                    # frozen master.  Inspect it in-place so the common layout
                    # still costs only one HDF5 open per candidate.
                    try:
                        value = handle.get(name)
                    except (KeyError, OSError, RuntimeError):
                        value = None
                    is_group, nx_class = root_group_facts(value)
                    result = (
                        is_group,
                        nx_class,
                        trial_paths,
                        trial_seen,
                        trial_states,
                    )
                    inspected[name] = result
                    return result
                with _open_stable_hdf5_dependency(
                    candidate.path,
                    trial_states,
                    cancelled=cancelled,
                ) as trial_root:
                    if trial_root is not None:
                        try:
                            link = trial_root.get(name, getlink=True)
                        except (KeyError, OSError, RuntimeError):
                            link = None
                        trace_indirect_link(
                            trial_root,
                            link,
                            selected_paths=trial_paths,
                            selected_seen=trial_seen,
                            selected_states=trial_states,
                        )
                        try:
                            value = trial_root.get(name)
                        except (KeyError, OSError, RuntimeError):
                            value = None
                        is_group, nx_class = root_group_facts(value)
                _verify_source_states(
                    trial_states,
                    cancelled=cancelled,
                )
                result = (
                    is_group,
                    nx_class,
                    trial_paths,
                    trial_seen,
                    trial_states,
                )
                inspected[name] = result
                return result

            selected_root = None
            hint = inspect_root_group("entry")
            if hint[0] and hint[1] in {"NXentry", ""}:
                selected_root = "entry"
            else:
                for name in root_names:
                    value = inspect_root_group(name)
                    if value[0] and value[1] == "NXentry":
                        selected_root = name
                        break
                if selected_root is None and hint[0]:
                    selected_root = "entry"

            selected_entry = None
            if selected_root is not None:
                _, _, selected_paths, selected_seen, selected_states = (
                    inspect_root_group(selected_root)
                )
                external.extend(selected_paths)
                traced.clear()
                traced.update(selected_seen)
                states.clear()
                states.update(selected_states)
                try:
                    value = handle.get(selected_root)
                except (KeyError, OSError, RuntimeError):
                    value = None
                if isinstance(value, h5py.Group):
                    selected_entry = value
            if not has_decisive_detector(selected_entry):
                external.extend(_scan_all_external_links(
                    handle,
                    cancelled=cancelled,
                    states=states,
                ))
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        _classify_inventory_failure(
            candidate,
            states,
            error,
            cancelled=cancelled,
        )
        raise ValueError(
            "source dependency inventory could not be established: "
            f"{candidate.path}: {type(error).__name__}: {error}"
        ) from error
    except SourceRevisionChanged:
        raise
    except Exception as error:
        _classify_inventory_failure(
            candidate,
            states,
            error,
            cancelled=cancelled,
        )
        raise ValueError(
            "source dependency inventory could not be established: "
            f"{candidate.path}: {type(error).__name__}: {error}"
        ) from error
    if cancelled():
        raise RuntimeError("admission cancelled")
    _validate_exact_candidate_state(
        candidate,
        accepted_candidate.followed_state,
    )
    _verify_source_states(states, cancelled=cancelled)
    return (
        tuple(dict.fromkeys(external)),
        tuple(states.values()),
        "",
    )


def _deferred_directory_plan(
    configuration: OutputCandidate,
    plan: RunCandidatePlan,
    discovered_paths: tuple[Path, ...],
    *,
    cancelled: Callable[[], bool],
) -> DeferredDirectoryPlan:
    if not plan.candidates:
        raise RuntimeError("source has no matching candidates")
    staged: list[
        tuple[
            tuple[Candidate, ...],
            tuple[_CapturedSourceTopology, ...],
            Path,
            tuple[Path, ...],
            tuple[_CapturedSourceTopology, ...],
            str,
        ]
    ] = []
    for group in _directory_candidate_groups(plan):
        if cancelled():
            raise RuntimeError("admission cancelled")
        captured_topologies: list[_CapturedSourceTopology] = []
        for source_candidate in group:
            if cancelled():
                raise RuntimeError("admission cancelled")
            captured_topologies.append(
                _topology_from_captured_state(
                    _capture_exact_candidate_state(source_candidate),
                    candidate_owner_id=source_candidate.adapter_id,
                )
            )
            if cancelled():
                raise RuntimeError("admission cancelled")
        candidate_states = tuple(captured_topologies)
        representative = group[0]
        adapter = get_adapter(representative.adapter_id)
        if adapter is None:
            raise ValueError(
                f"candidate {representative.path} lost its adapter"
            )
        name = adapter.scan_name(representative.path)
        target = _directory_target(configuration, plan, representative, name)
        # Directory admission remains name/stat-only. Exact container and
        # external-dependency inspection belongs to the one JIT materializer.
        external_paths = ()
        inventory_states = candidate_states
        skip_reason = ""
        staged.append(
            (
                group,
                candidate_states,
                target,
                external_paths,
                inventory_states,
                skip_reason,
            )
        )

    _validate_deferred_targets(
        configuration,
        plan,
        tuple(
            target
            for (
                _group,
                _candidate_states,
                target,
                _external,
                _states,
                _reason,
            ) in staged
        ),
        tuple(
            path
            for (
                _group,
                _candidate_states,
                _target,
                paths,
                _states,
                _reason,
            ) in staged
            for path in paths
        ),
    )
    entries: list[DeferredDirectoryEntry] = []
    for (
        group,
        candidate_states,
        target,
        external_paths,
        inventory_states,
        skip_reason,
    ) in staged:
        keys = {
            os.path.normcase(os.path.abspath(path))
            for path in (candidate.path for candidate in group)
        }
        physical = tuple(
            path
            for path in discovered_paths
            if os.path.normcase(os.path.abspath(path)) in keys
        )
        if not physical:
            raise ValueError(
                f"directory candidate escaped physical file census: {group[0].path}"
            )
        protected: list[_CapturedSourceTopology] = []
        frozen = {
            _source_state_key(value.raw_path): value
            for value in inventory_states
        }
        reason = skip_reason
        for source_candidate, topology in zip(
            group,
            candidate_states,
            strict=True,
        ):
            if cancelled():
                raise RuntimeError("admission cancelled")
            _validate_exact_candidate_state(
                source_candidate,
                topology.followed_state,
            )
            if cancelled():
                raise RuntimeError("admission cancelled")
            protected.append(topology)
        for path in external_paths:
            if cancelled():
                raise RuntimeError("admission cancelled")
            key = _source_state_key(path)
            if any(
                _source_state_key(value.raw_path) == key
                for value in protected
            ):
                continue
            try:
                topology = frozen.get(key) or _capture_source_topology(
                    path,
                    cancelled=cancelled,
                )
            except FileNotFoundError as error:
                raise SourceRevisionChanged(
                    "source dependency is still landing; finite Standard "
                    f"admission cannot safely start: {path}"
                ) from error
            except OSError as error:
                raise ValueError(
                    f"source dependency could not be frozen: {path}: {error}"
                ) from error
            if cancelled():
                raise RuntimeError("admission cancelled")
            protected.append(topology)
        protected_map = {
            _source_state_key(value.raw_path): value
            for value in protected
        }
        _verify_source_states(protected_map, cancelled=cancelled)
        entries.append(DeferredDirectoryEntry(
            group,
            target,
            OutputFact(_path_state(target)),
            physical,
            tuple(value.followed_state for value in protected),
            reason,
            _protected_topology=tuple(protected),
        ))
    return DeferredDirectoryPlan(
        plan,
        tuple(entries),
        discovered_paths,
    )


def _validate_deferred_targets(
    configuration: OutputCandidate,
    plan: RunCandidatePlan,
    targets: tuple[Path, ...],
    external_paths: tuple[Path, ...],
) -> None:
    source = configuration.source
    if type(source) is not DirectorySourceSpec:
        raise TypeError("deferred targets require a directory source")
    raw_paths = (
        tuple(candidate.path for candidate in plan.candidates)
        + external_paths
    )
    raw_norms = {
        os.path.normcase(os.path.realpath(path)) for path in raw_paths
    }
    raw_inodes: set[tuple[int, int]] = set()
    for path in raw_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        raw_inodes.add((int(stat.st_dev), int(stat.st_ino)))
    protected = tuple(
        Path(value)
        for value in (configuration.poni_file, configuration.mask_file)
        if value
    )
    seen: set[str] = set()
    for target in targets:
        normalized = os.path.normcase(os.path.realpath(target))
        if normalized in seen:
            raise ValueError(f"duplicate output target {target}")
        seen.add(normalized)
        target_inode = None
        try:
            stat = target.stat()
        except OSError:
            pass
        else:
            target_inode = (int(stat.st_dev), int(stat.st_ino))
        if normalized in raw_norms or (
            target_inode is not None and target_inode in raw_inodes
        ):
            raise OutputCollisionError(
                f"Reduction output '{target}' is the same file as a raw "
                "directory input; choose a separate Save Path."
            )
        check_output_not_source(
            target,
            input_files=protected,
            watched_dirs=(source.root,),
            recursive=source.recursive,
            container_directory_mode=True,
        )


def live_directory_groups(
    receipt: AdmissionReceipt,
    observation: object,
) -> tuple[LiveDirectoryGroup, ...]:
    """Project current name/stat revisions into output-shaped Live groups.

    This step opens no candidate content.  Exact content readiness remains a
    separate, Stop-aware JIT operation in
    :func:`materialize_live_directory_group`.
    """

    deferred = receipt.deferred_directory
    discovered = getattr(observation, "discovered_snapshot", None)
    if (
        type(deferred) is not DeferredDirectoryPlan
        or not deferred.live
        or discovered is None
        or not deferred.candidates.matches_config(discovered)
    ):
        raise TypeError("live directory observation lost its admitted owner")
    current = _bounded_directory_plan(
        RunCandidatePlan.from_snapshot(discovered)
    )
    groups: list[LiveDirectoryGroup] = []
    for candidates in _directory_candidate_groups(current):
        adapter = get_adapter(candidates[0].adapter_id)
        if adapter is None:
            raise ValueError(
                f"candidate {candidates[0].path} lost its adapter"
            )
        subset = RunCandidatePlan(
            current.generation,
            candidates,
            current.root,
            current.recursive,
            current.name_filter,
        )
        groups.append(LiveDirectoryGroup(
            subset,
            _directory_target(
                receipt.candidate,
                current,
                candidates[0],
                adapter.scan_name(candidates[0].path),
            ),
            subset.paths,
        ))
    return tuple(groups)


def materialize_live_directory_group(
    receipt: AdmissionReceipt,
    configuration: FrozenRunConfiguration,
    session: DirectoryIndexSession,
    group: LiveDirectoryGroup,
    *,
    cancelled: Callable[[], bool],
    reprobe: bool = False,
) -> LiveDirectoryAttempt:
    """Probe and JIT-admit one exact Live group revision.

    Provisional or concurrently changing bytes produce a typed retry-later
    result.  Output collision/configuration failures remain hard failures.
    """

    observations = []
    for candidate in group.plan.candidates:
        if cancelled():
            raise RuntimeError("admission cancelled")
        try:
            observed = (
                session.reprobe_candidate(candidate, refresh=False)
                if reprobe
                else session.probe_candidate(candidate, refresh=False)
            )
        except StaleCandidateError as error:
            return LiveDirectoryAttempt(
                group,
                ProbeState.IN_PROGRESS,
                reason=str(error),
            )
        if cancelled():
            raise RuntimeError("admission cancelled")
        observations.append(observed)
        if observed.result.state is not ProbeState.READY:
            return LiveDirectoryAttempt(
                group,
                observed.result.state,
                reason=observed.result.reason,
            )

    ready = RunCandidatePlan(
        group.plan.generation,
        tuple(value.candidate for value in observations),
        group.plan.root,
        group.plan.recursive,
        group.plan.name_filter,
        tuple(value.descriptor for value in observations),
    )
    try:
        deferred = _deferred_directory_plan(
            receipt.candidate,
            ready,
            group.physical_paths,
            cancelled=cancelled,
        )
        transient = replace(
            receipt,
            deferred_directory=deferred,
            directory_discovered_file_count=len(group.physical_paths),
            directory_discovered_paths=group.physical_paths,
        )
        decision, _ready_files, _skipped_files = materialize_deferred_output(
            transient,
            session,
            deferred.entries[0],
            cancelled=cancelled,
        )
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        raise
    except SourceRevisionChanged as error:
        return LiveDirectoryAttempt(
            group,
            ProbeState.IN_PROGRESS,
            reason=str(error),
            revision_changed=True,
        )
    if decision is None:
        return LiveDirectoryAttempt(
            group,
            ProbeState.INVALID,
            reason="source revision has no READY detector frames",
        )
    _validate_exact_tiff_gi_motor(
        RunIntent.from_frozen(configuration),
        (decision.item,),
    )
    return LiveDirectoryAttempt(
        group,
        ProbeState.READY,
        decision=decision,
    )


def materialize_deferred_output(
    receipt: AdmissionReceipt,
    session: DirectoryIndexSession,
    entry: DeferredDirectoryEntry,
    *,
    cancelled: Callable[[], bool],
) -> tuple[AdmittedOutput | None, int, int]:
    deferred = receipt.deferred_directory
    if (
        type(deferred) is not DeferredDirectoryPlan
        or type(entry) is not DeferredDirectoryEntry
        or not any(value is entry for value in deferred.entries)
    ):
        raise TypeError("deferred output is not owned by this receipt")
    _validate_deferred_entry_states(entry, cancelled=cancelled)
    if entry.skip_reason:
        return None, 0, len(entry.physical_paths)

    ready: list[Candidate] = []
    descriptors: list[ContainerDescriptor | None] = []
    skipped = 0
    for candidate in entry.candidates:
        if cancelled():
            raise RuntimeError("admission cancelled")
        # ``validate_admitted_receipt`` refreshes and reconciles the complete
        # name/stat snapshot once before execution.  Refreshing again here
        # would rescan the whole directory for every member (O(N^2)); the
        # exact-one probe itself revalidates this candidate after opening it.
        observed = session.probe_candidate(candidate, refresh=False)
        descriptor = observed.descriptor
        if observed.result.state is ProbeState.READY and (
            descriptor is not None
            or observed.result.kind is SourceKind.IMAGE_FILE
        ):
            ready.append(observed.candidate)
            descriptors.append(descriptor)
        else:
            skipped += 1
    if not ready:
        return None, 0, len(entry.physical_paths)
    plan = deferred.candidates
    subset = RunCandidatePlan(
        plan.generation,
        tuple(ready),
        plan.root,
        plan.recursive,
        plan.name_filter,
        tuple(descriptors),
    )
    try:
        items = _directory_items(
            receipt.candidate,
            subset,
            cancelled=cancelled,
        )
    except (ValueError, OSError):
        # A semantic/schema failure remains terminal when every admitted
        # source state is still exact.  If the same failure followed a real
        # byte/dependency drift, this exact revalidation raises the typed Live
        # retry signal instead.
        _validate_deferred_entry_states(entry, cancelled=cancelled)
        raise
    if len(items) != 1:
        raise ValueError("deferred source group did not produce one output")
    item = items[0]
    if item.target != entry.target:
        raise ValueError("deferred source target changed during materialization")
    _validate_deferred_entry_states(
        entry,
        item=item,
        cancelled=cancelled,
    )
    _validate_targets(
        receipt.candidate,
        receipt.candidate.source,
        (item,),
        cancelled=cancelled,
    )
    decision = inspect_output(item, receipt.candidate, entry.fact)
    if item.source_spec.kind is SourceKind.TIFF_SERIES:
        return decision, len(ready), skipped
    return decision, len(entry.physical_paths), 0


def _validate_deferred_entry_states(
    entry: DeferredDirectoryEntry,
    *,
    item: PlannedOutput | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Require one deferred cursor to remain inside its admitted revision."""

    is_cancelled = _not_cancelled if cancelled is None else cancelled
    if item is not None:
        # The complete JIT stamp is the first owner capable of proving every
        # raw alias binding.  Run that proof before the older deferred-state
        # fences so no alias drift can be classified without its frozen
        # raw-to-resolved identity reaching the central validator.
        validate_source_aliases(item.source_stamp, cancelled=is_cancelled)
        _validate_tiff_metadata_selection(item, cancelled=is_cancelled)
    topologies = entry._protected_topology or tuple(
        _topology_from_captured_state(value)
        for value in entry.protected_states
    )
    topology_map = {
        _source_state_key(value.raw_path): value
        for value in topologies
    }
    _verify_source_states(topology_map, cancelled=is_cancelled)
    frozen = {
        _source_state_key(value.raw_path): value.followed_state
        for value in topologies
    }
    if item is None:
        return
    stamp = item.source_stamp
    occurrences = (
        ((stamp.file, "source_file"),)
        + tuple((value, "source_member") for value in stamp.members)
        + tuple(
            (value.file, "external_member")
            for value in stamp.external_members
        )
        + tuple(
            (value, "detector_dependency")
            for value in stamp.dependency_files
        )
        + tuple(
            (value.metadata_file, "image_metadata")
            for value in stamp.metadata_sources
            if value.metadata_file is not None
        )
    )
    for state, role in occurrences:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        accepted = frozen.get(_source_state_key(state.path))
        if accepted is None:
            # TIFF sidecars and selected HDF5 dependencies are discovered and
            # guarded JIT while the source group is materialized, so they are
            # not present in the cheap name/stat candidate freeze.  They are
            # valid only while that exact JIT-captured revision remains current.
            if role in {
                "external_member",
                "detector_dependency",
                "image_metadata",
            } and state.matches_disk():
                continue
            raise SourceRevisionChanged(
                "source dependency escaped admitted revision: "
                f"{state.path}"
            )
        if not _same_source_revision(accepted, state):
            raise SourceRevisionChanged(
                "source dependency escaped admitted revision: "
                f"{state.path}"
            )


def _validate_tiff_metadata_selection(
    item: PlannedOutput,
    *,
    cancelled: Callable[[], bool],
) -> None:
    """Keep nullable/selected TIFF metadata authoritative until decision."""

    stamp = item.source_stamp
    if not stamp.metadata_sources:
        return
    options = dict(item.source_spec.options)
    metadata_format = options.get("metadata_format", "auto")
    meta_dir = options.get("meta_dir")
    for metadata in stamp.metadata_sources:
        if cancelled():
            raise RuntimeError("admission cancelled")
        observed = read_image_motor_metadata(
            metadata.source_path,
            metadata_format,
            meta_dir=meta_dir,
        )
        if cancelled():
            raise RuntimeError("admission cancelled")
        current = observed.source_path
        expected = metadata.metadata_file
        if (
            (expected is None) != (current is None)
            or (
                expected is not None
                and current is not None
                and _raw_source_key(expected.path)
                != _raw_source_key(current)
            )
        ):
            raise SourceRevisionChanged(
                "TIFF metadata source changed before output decision: "
                f"{metadata.source_path}"
            )


def validate_admitted_receipt(
    receipt: AdmissionReceipt,
    session: DirectoryIndexSession | None,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    is_cancelled = _not_cancelled if cancelled is None else cancelled
    if is_cancelled():
        raise RuntimeError("admission cancelled")
    outputs = receipt.outputs
    deferred = receipt.deferred_directory
    if deferred is not None:
        if (
            type(deferred) is not DeferredDirectoryPlan
            or session is None
        ):
            raise RuntimeError("deferred directory admission lost its session")
        if deferred.live:
            # Directory Live owns a persistent source session.  Its baseline is
            # intentionally allowed to gain or revise candidates between the
            # Start click and worker launch; the worker exact-probes the current
            # Candidate revision before every JIT materialization.
            return
        observation = session.observe(refresh=True)
        reconciliation = deferred.candidates.reconcile(
            observation.discovered_snapshot
        )
        if not reconciliation.baseline_current:
            raise ValueError("source group changed after admission")
        for entry in deferred.entries:
            if is_cancelled():
                raise RuntimeError("admission cancelled")
            _validate_deferred_entry_states(
                entry,
                cancelled=is_cancelled,
            )
            if _path_state(entry.target) != entry.fact.target_state:
                raise RuntimeError(
                    f"output target changed after admission: {entry.target}"
                )
        return
    if session is None:
        source = receipt.candidate.source
        if type(source) is SourceSpec and source.kind is SourceKind.TIFF_SERIES:
            _validate_motor_knowledge(
                receipt,
                (_series_item(
                    receipt.candidate,
                    source,
                    cancelled=is_cancelled,
                ),),
            )
        for output in outputs:
            if is_cancelled():
                raise RuntimeError("admission cancelled")
            validate_planned_source(
                output.item,
                cancelled=is_cancelled,
            )
        return
    observation = session.observe(refresh=True)
    if is_cancelled():
        raise RuntimeError("admission cancelled")
    while observation.unprobed_count:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        observation = session.observe(refresh=False)
    current_items = _directory_items(
        receipt.candidate,
        RunCandidatePlan.from_observation(observation),
        cancelled=is_cancelled,
    )
    accepted = _groups(output.item for output in outputs)
    current = _groups(current_items)
    _validate_motor_knowledge(receipt, current_items)
    for output in outputs:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        validate_planned_source(
            output.item,
            cancelled=is_cancelled,
        )
    if accepted != current:
        raise ValueError("source group changed after admission")
def target_state_matches(output: AdmittedOutput) -> bool:
    return _path_state(output.item.target) == output.fact.target_state
def _groups(items: Any) -> dict[tuple[str, str, Path], Any]:
    return {item.group.key: item.group for item in items}
def _validate_motor_knowledge(
    receipt: AdmissionReceipt,
    current_items: tuple[PlannedOutput, ...],
) -> None:
    accepted = _groups(output.item for output in receipt.outputs)
    current = _groups(current_items)
    if (
        {key: value.motor_names for key, value in accepted.items()}
        != {key: value.motor_names for key, value in current.items()}
        or {
            key: value.source_stamp.admitted_motor_values
            for key, value in accepted.items()
        }
        != {
            key: value.source_stamp.admitted_motor_values
            for key, value in current.items()
        }
        or {
            key: value.source_stamp.metadata_sources
            for key, value in accepted.items()
        }
        != {
            key: value.source_stamp.metadata_sources
            for key, value in current.items()
        }
        or _motor_choices(current_items) != receipt.gi_motor_choices
    ):
        raise ValueError("authoritative motor knowledge changed after admission")
def _motor_choices(items: tuple[PlannedOutput, ...]) -> tuple[str, ...] | None:
    return ordered_motor_intersection(
        item.group.motor_names for item in items
    )


def _selected_tiff_gi_motor(
    configuration: FrozenRunConfiguration | OutputCandidate,
) -> str | None:
    if type(configuration) is FrozenRunConfiguration:
        gi = configuration.gi
        return (
            gi.effective_motor
            if gi.enabled and gi.effective_motor != "Manual"
            else None
        )
    values = configuration.processing_mapping().get("gi", {})
    if type(values) is not dict or not bool(values.get("enabled")):
        return None
    motor = str(
        values.get("resolved_motor")
        or values.get("incidence_motor")
        or ""
    )
    return motor if motor and motor != "Manual" else None


def _resolved_generated_target(save_path: str, scan_name: str) -> Path:
    """Delegate vNext's one generated-output naming decision to the shared owner.

    vNext admission is Overwrite-only.  A suffix-shaped requested path is the
    operator's explicit target and is preserved byte-for-byte; a directory
    request generates ``<scan>.nexus`` (P4/OUT-1).  This helper only decides
    how the captured ``save_path`` is supplied to the shared API — suffix,
    collision, writer and transaction policy stay with their owners.
    """
    requested = Path(save_path)
    return Path(resolve_output_target(
        requested.parent if requested.suffix else requested,
        scan_name,
        mode=OVERWRITE_MODE,
        explicit_target=requested if requested.suffix else None,
    ))


def _not_cancelled() -> bool:
    return False


def _capture_source_states(
    paths: tuple[Path, ...],
    cancelled: Callable[[], bool],
) -> tuple[SourceFileState, ...]:
    """Capture an exact member stamp without delaying cancellation.

    File-state capture can block on beamline storage.  Check both sides of
    every member so cancellation raised during one capture cannot trigger a
    sweep of all remaining TIFFs before admission notices it.
    """

    states: list[SourceFileState] = []
    for path in paths:
        if cancelled():
            raise RuntimeError("admission cancelled")
        try:
            state = SourceFileState.capture(path)
        except FileNotFoundError as error:
            raise SourceRevisionChanged(
                f"source candidate disappeared during admission: {path}"
            ) from error
        if cancelled():
            raise RuntimeError("admission cancelled")
        states.append(state)
    return tuple(states)


def _tiff_motor_knowledge(
    members: tuple[Path, ...],
    states: tuple[SourceFileState, ...],
    metadata_format: str | None,
    selected_motor: str | None,
    *,
    meta_dir: Path | str | None = None,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[
    tuple[str, ...] | None,
    tuple[AdmittedMotorValue, ...],
    tuple[AdmittedMetadataSource, ...],
]:
    if len(members) != len(states):
        raise ValueError("TIFF metadata knowledge lost member alignment")
    metadata: list[dict[str, float]] = []
    metadata_sources: list[AdmittedMetadataSource] = []
    for member, state in zip(members, states, strict=True):
        if cancelled():
            raise RuntimeError("admission cancelled")
        discovered = read_image_motor_metadata(
            member,
            metadata_format,
            meta_dir=meta_dir,
        )
        if cancelled():
            raise RuntimeError("admission cancelled")
        discovered_path = discovered.source_path
        before = (
            None
            if discovered_path is None
            else SourceFileState.capture(discovered_path)
        )
        if cancelled():
            raise RuntimeError("admission cancelled")
        # The first read discovers the exact sidecar candidate.  Accept values
        # only from a guarded second read so a writer cannot replace V0 with V1
        # between parsing and the stamp and leave V0 values falsely bound to
        # V1's file identity.  The nullable no-sidecar result is guarded too:
        # a companion that appears during admission changes the source fact.
        observed = read_image_motor_metadata(
            member,
            metadata_format,
            meta_dir=meta_dir,
        )
        if cancelled():
            raise RuntimeError("admission cancelled")
        observed_path = observed.source_path
        if (
            (discovered_path is None) != (observed_path is None)
            or (
                discovered_path is not None
                and observed_path is not None
                and Path(discovered_path).resolve(strict=False)
                != Path(observed_path).resolve(strict=False)
            )
        ):
            raise SourceRevisionChanged(
                f"TIFF metadata source changed during admission: {member}"
            )
        metadata_file = (
            None
            if observed_path is None
            else SourceFileState.capture(observed_path)
        )
        if cancelled():
            raise RuntimeError("admission cancelled")
        if before != metadata_file:
            raise SourceRevisionChanged(
                f"TIFF metadata source changed during admission: {member}"
            )
        metadata.append(dict(observed.values))
        metadata_sources.append(
            AdmittedMetadataSource(state.path, metadata_file)
        )
    if cancelled():
        raise RuntimeError("admission cancelled")
    names = ordered_motor_intersection(
        tuple(tuple(value) for value in metadata)
    )
    admitted: tuple[AdmittedMotorValue, ...] = ()
    if (
        selected_motor is not None
        and all(selected_motor in value for value in metadata)
    ):
        admitted = tuple(
            AdmittedMotorValue(
                state.path,
                selected_motor,
                float(value[selected_motor]),
            )
            for state, value in zip(states, metadata)
        )
    if cancelled():
        raise RuntimeError("admission cancelled")
    return names, admitted, tuple(metadata_sources)


def _execution_tiff_source(
    source: SourceSpec,
    states: tuple[SourceFileState, ...],
    admitted: tuple[AdmittedMotorValue, ...],
) -> SourceSpec:
    options = dict(source.options)
    options["files"] = tuple(value.path for value in states)
    options["admitted_motor_values"] = tuple(
        (value.source_path, value.motor, value.value)
        for value in admitted
    )
    return SourceSpec(
        source.uri,
        source.kind,
        metadata_uri=source.metadata_uri,
        entry=source.entry,
        options=options,
    )


def _series_item(
    configuration: FrozenRunConfiguration | OutputCandidate, source: SourceSpec,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> PlannedOutput:
    if source.kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
        return _container_item(configuration, source, cancelled=cancelled)
    options = dict(source.options)
    members = tuple(Path(value) for value in options.get("files", ()))
    members = members or (Path(options.get("selected_file") or source.uri),)
    states = _capture_source_states(members, cancelled)
    name = str(options.get("scan_name") or members[0].stem)
    target = _resolved_generated_target(configuration.save_path, name)
    motor_names = None
    admitted: tuple[AdmittedMotorValue, ...] = ()
    metadata_sources: tuple[AdmittedMetadataSource, ...] = ()
    if source.kind is SourceKind.TIFF_SERIES:
        metadata_format = options.get("metadata_format", "auto")
        motor_names, admitted, metadata_sources = _tiff_motor_knowledge(
            members,
            states,
            metadata_format,
            _selected_tiff_gi_motor(configuration),
            meta_dir=options.get("meta_dir"),
            cancelled=cancelled,
        )
        source = _execution_tiff_source(source, states, admitted)
    stamp = SourceExecutionStamp(
        states[0],
        "tiff_series",
        len(states),
        1,
        states,
        admitted_motor_values=admitted,
        metadata_sources=metadata_sources,
    )
    return PlannedOutput(
        source,
        members[0],
        target,
        stamp,
        motor_names=motor_names,
    )


def _container_item(
    configuration: FrozenRunConfiguration | OutputCandidate,
    source: SourceSpec,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> PlannedOutput:
    path = Path(source.uri).expanduser()
    state = _capture_source_states((path,), cancelled)[0]
    owner = candidate_owner(path)
    if owner is None or source.kind not in owner.kinds:
        raise ValueError(f"selected container has no compatible owner: {path}")
    result = owner.probe(path)
    current_owner = candidate_owner(path)
    if (
        not state.matches_disk()
        or current_owner is None
        or current_owner.id != owner.id
    ):
        raise SourceRevisionChanged(
            f"source candidate changed during admission: {path}"
        )
    descriptor = result.descriptor
    if (
        result.state is not ProbeState.READY
        or descriptor is None
        or descriptor.frame_count < 1
        or descriptor.kind is SourceKind.PROCESSED_NEXUS
    ):
        reason = result.reason or result.state.value
        raise ValueError(f"selected container is not ready: {path}: {reason}")
    normalized = SourceSpec(
        path,
        descriptor.kind,
        entry=descriptor.resolved_entry or descriptor.requested_entry,
    )
    name = descriptor.scan_name or path.stem.removesuffix("_master")
    target = _resolved_generated_target(configuration.save_path, name)
    external_members = _external_members(
        path,
        state,
        descriptor,
        cancelled=cancelled,
    )
    stamp = SourceExecutionStamp(
        state,
        owner.id,
        descriptor.frame_count,
        0,
        external_members=external_members,
        dependency_files=_selected_dependency_files(
            path,
            state,
            descriptor,
            external_members,
            cancelled=cancelled,
        ),
    )
    return PlannedOutput(
        normalized,
        path,
        target,
        stamp,
        descriptor=descriptor,
    )


def _uses_eager_directory_descriptors(
    configuration: FrozenRunConfiguration | OutputCandidate,
) -> bool:
    if type(configuration) is FrozenRunConfiguration:
        return bool(configuration.gi.enabled or configuration.live_mode)
    values = configuration.processing_mapping()
    gi = values.get("gi", {})
    return bool(
        values.get("live_mode")
        or (type(gi) is dict and gi.get("enabled"))
    )


def _directory_items(
    configuration: FrozenRunConfiguration | OutputCandidate,
    plan: RunCandidatePlan,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[PlannedOutput, ...]:
    output_root, items, consumed = Path(configuration.save_path), [], set()
    resolved_output_root = None
    if not output_root.suffix:
        try:
            resolved_output_root = output_root.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"directory output could not be resolved: {output_root}"
            ) from error
    selected_source = (
        configuration.thaw_source_spec()
        if type(configuration) is FrozenRunConfiguration
        else configuration.source
    )
    metadata_format = (
        selected_source.metadata_format
        if type(selected_source) is DirectorySourceSpec
        else "auto"
    )
    for candidate in plan.candidates:
        if cancelled():
            raise RuntimeError("admission cancelled")
        if candidate.path in consumed:
            continue
        descriptor = plan.descriptor_for(candidate)
        motor_names = None
        if descriptor is None:
            adapter = get_adapter(candidate.adapter_id)
            if adapter is None or SourceKind.IMAGE_FILE not in adapter.kinds:
                raise ValueError(f"candidate {candidate.path} lost its descriptor")
            name = adapter.scan_name(candidate.path)
            grouped = []
            for value in plan.candidates:
                if cancelled():
                    raise RuntimeError("admission cancelled")
                if (
                    value.adapter_id == candidate.adapter_id
                    and value.path.parent == candidate.path.parent
                    and adapter.scan_name(value.path) == name
                ):
                    grouped.append(value)
            members = tuple(grouped)
            consumed.update(value.path for value in members)
            states = _capture_source_states(
                tuple(value.path for value in members),
                cancelled,
            )
            spec = SourceSpec(
                candidate.path.parent, SourceKind.TIFF_SERIES,
                options={
                    "files": tuple(value.path for value in states),
                    "scan_name": name,
                    "metadata_format": metadata_format,
                },
            )
            member_paths = tuple(value.path for value in members)
            motor_names, admitted, metadata_sources = _tiff_motor_knowledge(
                member_paths,
                states,
                metadata_format,
                _selected_tiff_gi_motor(configuration),
                cancelled=cancelled,
            )
            spec = _execution_tiff_source(spec, states, admitted)
            stamp = SourceExecutionStamp(
                states[0],
                "tiff_series",
                len(states),
                1,
                states,
                admitted_motor_values=admitted,
                metadata_sources=metadata_sources,
            )
        else:
            consumed.add(candidate.path)
            if _uses_eager_directory_descriptors(configuration):
                state = _capture_source_states(
                    (candidate.path,),
                    cancelled,
                )[0]
                adapter = get_adapter(candidate.adapter_id)
                if adapter is None:
                    raise ValueError(
                        f"candidate {candidate.path} lost its adapter"
                    )
                refreshed = adapter.probe(candidate.path)
                if not state.matches_disk():
                    raise SourceRevisionChanged(
                        "container changed during eager admission: "
                        f"{candidate.path}"
                    )
                if (
                    refreshed.state is not ProbeState.READY
                    or refreshed.descriptor is None
                ):
                    raise ValueError(
                        "container lost READY descriptor during eager "
                        f"admission: {candidate.path}"
                    )
                descriptor = refreshed.descriptor
            else:
                state = _capture_source_states(
                    (candidate.path,),
                    cancelled,
                )[0]
            if descriptor.kind is SourceKind.PROCESSED_NEXUS:
                raise ValueError("processed output cannot be raw input")
            if descriptor.frame_count < 1:
                continue
            name = descriptor.scan_name or candidate.path.stem.removesuffix("_master")
            spec = SourceSpec(
                candidate.path, descriptor.kind,
                entry=descriptor.resolved_entry or descriptor.requested_entry,
            )
            try:
                external_members = _external_members(
                    candidate.path,
                    state,
                    descriptor,
                    cancelled=cancelled,
                )
            except OSError as error:
                # Deferred materialization owns the accepted strong-state set.
                # Preserve this as a hard read failure here; its outer
                # classifier upgrades it to SourceRevisionChanged only when
                # exact master/member revalidation proves drift.
                raise ValueError(
                    "external source members could not be inspected: "
                    f"{candidate.path}: {error}"
                ) from error
            dependency_files = _selected_dependency_files(
                candidate.path,
                state,
                descriptor,
                external_members,
                cancelled=cancelled,
            )
            if cancelled():
                raise RuntimeError("admission cancelled")
            stamp = SourceExecutionStamp(
                state,
                candidate.adapter_id, descriptor.frame_count, 0,
                external_members=external_members,
                dependency_files=dependency_files,
            )
        output_request = configuration.save_path
        if not output_root.suffix:
            try:
                relative_parent = candidate.path.parent.relative_to(plan.root)
            except ValueError as error:
                raise ValueError(
                    "directory candidate escaped admitted root: "
                    f"{candidate.path}"
                ) from error
            # Preserve each recursive source parent beneath the selected
            # output directory. Direct children keep the existing flat shape;
            # the shared owner still makes the filename/suffix decision.
            output_directory = output_root / relative_parent
            try:
                resolved_output = output_directory.resolve(strict=False)
            except (OSError, RuntimeError) as error:
                raise ValueError(
                    "directory output could not be resolved: "
                    f"{output_directory}"
                ) from error
            if (
                resolved_output_root is None
                or not resolved_output.is_relative_to(resolved_output_root)
            ):
                raise ValueError(
                    "directory output escaped selected root: "
                    f"{output_directory}"
                )
            output_request = str(output_directory)
        items.append(PlannedOutput(
            spec,
            candidate.path,
            _resolved_generated_target(output_request, name),
            stamp,
            candidate,
            descriptor,
            motor_names,
        ))
    return tuple(items)
def validate_planned_source(
    item: PlannedOutput,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    is_cancelled = _not_cancelled if cancelled is None else cancelled
    validate_source_aliases(item.source_stamp, cancelled=is_cancelled)
    _validate_tiff_metadata_selection(item, cancelled=is_cancelled)
    import h5py
    for external in item.source_stamp.external_members:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        state = external.file
        path = Path(state.path)
        try:
            with h5py.File(path, "r") as handle:
                if external.dataset not in handle:
                    raise ValueError(
                        "external source dataset changed: "
                        f"{path}:{external.dataset}"
                    )
        except SourceRevisionChanged:
            raise
        except OSError as error:
            try:
                validate_source_aliases(
                    item.source_stamp,
                    cancelled=is_cancelled,
                )
            except SourceRevisionChanged as drift:
                raise drift from error
            raise
        if is_cancelled():
            raise RuntimeError("admission cancelled")
def source_snapshots(item: PlannedOutput) -> dict[str, dict[str, Any]]:
    stamp = item.source_stamp
    if stamp.members:
        values = {
            state.path: {
                "adapter_id": stamp.adapter_id,
                **state.as_dict(),
                "frame_count": 1,
                "self_contained": True,
            }
            for state in stamp.members
        }
        for metadata in stamp.metadata_sources:
            state = metadata.metadata_file
            if state is not None:
                values[state.path] = {
                    "adapter_id": "image_metadata",
                    **state.as_dict(),
                    "frame_count": 0,
                    "self_contained": True,
                    "source_role": "image_metadata",
                }
        return values

    value: dict[str, Any] = {
        "adapter_id": stamp.adapter_id,
        **stamp.file.as_dict(),
        "frame_count": stamp.frame_count,
    }
    if item.descriptor is not None:
        value.update(
            dataset_path=item.descriptor.dataset_path,
            self_contained=item.descriptor.self_contained,
        )
    values = {str(item.source_path): value}
    for external in stamp.external_members:
        state = external.file
        values[state.path] = {
            "adapter_id": stamp.adapter_id,
            **state.as_dict(),
            "frame_count": external.stop - external.first,
            "dataset_path": external.dataset,
            "self_contained": True,
        }
    for state in stamp.dependency_files:
        values[state.path] = {
            "adapter_id": "hdf5_dependency",
            **state.as_dict(),
            "frame_count": 0,
            "self_contained": True,
            "source_role": "detector_dependency",
        }
    return values
def execution_plan_values(
    configuration: FrozenRunConfiguration, detector_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    threshold, gi = configuration.threshold, configuration.gi
    if not threshold_pair_is_canonical(threshold):
        raise ValueError(
            "degenerate threshold identity reached execution "
            f"(apply_threshold={threshold.apply_threshold}, "
            f"mask_saturation={threshold.mask_saturation}); the start "
            "capture canonicalizes this pair — refusing to execute a "
            "configuration that does not describe its own run"
        )
    if detector_mask is None and configuration.mask_file:
        path = Path(configuration.mask_file)
        detector_mask = np.load(path) if path.suffix == ".npy" else load_mask(path)
    one, two = configuration.bai_1d_args, configuration.bai_2d_args
    if gi.enabled:
        one["gi_mode_1d"], two["gi_mode_2d"] = gi.mode_1d, gi.mode_2d
    manual = gi.enabled and gi.effective_motor == "Manual"
    return one, two, {
        "gi_enabled": gi.enabled,
        "gi_incident_angle": gi.th_val if manual else None,
        "incidence_motor": gi.effective_motor if gi.enabled and not manual else None,
        "tilt_angle": gi.tilt_angle, "sample_orientation": gi.sample_orientation,
        "integrate_1d": True,
        "integrate_2d": not (
            "Viewer" not in configuration.processing_mode
            and "1D" in configuration.processing_mode
            and "2D" not in configuration.processing_mode
        ),
        "threshold_min": (
            threshold.threshold_min if threshold.apply_threshold else None
        ),
        "threshold_max": (
            threshold.threshold_max if threshold.apply_threshold else None
        ),
        "mask_saturation": threshold.mask_saturation,
        "detector_mask": detector_mask,
    }
def _validate_targets(
    configuration: FrozenRunConfiguration | OutputCandidate,
    source: SourceSpec | DirectorySourceSpec,
    items: tuple[PlannedOutput, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    raw: tuple[Path, ...] = ()
    for item in items:
        identity = validate_source_aliases(
            item.source_stamp,
            cancelled=cancelled,
        )
        raw += tuple(Path(binding.raw_path) for binding in identity.identity.aliases)
        raw += tuple(Path(target.resolved_path) for target in identity.targets)
    protected = tuple(
        Path(value) for value in (
            configuration.poni_file, configuration.mask_file,
            source.metadata_uri if type(source) is SourceSpec else None,
        ) if value
    )
    targets: set[str] = set()
    for item in items:
        target = os.path.normcase(os.path.realpath(item.target))
        if target in targets:
            raise ValueError(f"duplicate output target {item.target}")
        targets.add(target)
        check_output_not_source(
            item.target, input_files=(*raw, *protected),
            watched_dirs=(source.root,) if type(source) is DirectorySourceSpec else (),
            recursive=bool(getattr(source, "recursive", False)),
            container_directory_mode=type(source) is DirectorySourceSpec,
        )
def _load_scientific_assets(intent: RunIntent) -> AcceptedScientificAssets:
    calibration, poni_digest = _load_stable_asset(
        intent.poni_file,
        lambda path, data: load_poni(path, data=data)
        if load_poni is load_detector_calibration else load_poni(path),
        max_bytes=1 << 20,
    )
    mask, mask_digest = _load_stable_asset(
        intent.mask_file,
        lambda path, _data: np.load(path) if path.suffix == ".npy" else load_mask(path),
    )
    poni = None if calibration is None else calibration.poni
    poni_values = None if poni is None else (
        float(poni.dist), float(poni.poni1), float(poni.poni2), float(poni.rot1),
        float(poni.rot2), float(poni.rot3), float(poni.wavelength),
        str(poni.detector),
    )
    accepted = None if mask is None else np.ascontiguousarray(mask)
    config_json = None if calibration is None else json.dumps(
        dict(calibration.detector_config), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return AcceptedScientificAssets(
        poni_values, None if accepted is None else accepted.dtype.str,
        None if accepted is None else tuple(accepted.shape),
        None if accepted is None else accepted.tobytes(),
        poni_digest, mask_digest, config_json,
    )
def _load_stable_asset(
    path_text: object, loader: Callable[[Path, bytes], object],
    *, max_bytes: int | None = None,
) -> tuple[object | None, str | None]:
    if type(path_text) is not str or not path_text:
        return None, None
    path = Path(path_text)
    def observation() -> tuple[tuple[int, ...], bytes]:
        state = path.stat()
        with path.open("rb") as stream:
            payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"scientific asset exceeds {max_bytes} bytes: {path}")
        return (
            state.st_size, state.st_mtime_ns, state.st_ctime_ns,
            state.st_dev, state.st_ino,
        ), payload
    try:
        before_state, before = observation()
    except FileNotFoundError:
        return None, None
    value = loader(path, before)
    try:
        after_state, after = observation()
    except FileNotFoundError as exc:
        raise ValueError(f"scientific asset changed while admitted: {path}") from exc
    if after_state != before_state or after != before:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    return value, hashlib.sha256(before).hexdigest()
def _external_members(
    master: Path,
    master_state: SourceFileState,
    descriptor: ContainerDescriptor,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[ExternalSourceState, ...]:
    if descriptor.self_contained is not False:
        return ()
    segments = descriptor.segment_paths or (
        (descriptor.dataset_path,) if descriptor.dataset_path else ()
    )
    if not segments:
        raise ValueError("external container has no member-qualified proof")
    import h5py
    master_topology = _topology_from_captured_state(master_state)
    states = {
        _source_state_key(master_topology.raw_path): master_topology
    }
    values: list[ExternalSourceState] = []
    first = 0
    member_qualified = True
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
                components = tuple(
                    value
                    for value in segment.strip("/").split("/")
                    if value
                )
                parent: object = handle
                for component in components[:-1]:
                    if not isinstance(parent, h5py.Group):
                        member_qualified = False
                        break
                    parent = parent.get(component)
                if (
                    not member_qualified
                    or not components
                    or not isinstance(parent, h5py.Group)
                ):
                    member_qualified = False
                    break
                link = parent.get(components[-1], getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    # SoftLink, ancestor-ExternalLink and VDS layouts are
                    # frozen by the selected dependency closure below.
                    member_qualified = False
                    break
                path = _hdf5_link_file(parent, link.filename)
                try:
                    topology = _remember_source_state(
                        path,
                        states,
                        cancelled=cancelled,
                    )
                except FileNotFoundError as error:
                    raise SourceRevisionChanged(
                        "external detector member disappeared during "
                        f"admission: {path}"
                    ) from error
                except OSError as error:
                    raise SourceRevisionChanged(
                        "external detector member capture is unverifiable: "
                        f"{path}"
                    ) from error
                dataset = parent.get(components[-1])
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(
                        "external container lost its exact dataset"
                    )
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
                        f"{path}:{link.path}"
                    )
                stop = first + count
                values.append(
                    ExternalSourceState(
                        topology.followed_state,
                        link.path,
                        first,
                        stop,
                        epoch,
                    )
                )
                first = stop
                if cancelled():
                    raise RuntimeError("admission cancelled")
        if values and first != descriptor.frame_count:
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
    if not member_qualified:
        return ()
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


def _path_state(path: Path) -> tuple[int, int, int, int, int] | bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return (
        int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns),
        int(stat.st_dev), int(stat.st_ino),
    )
__all__ = [
    "DeferredDirectoryEntry", "DeferredDirectoryPlan", "LiveDirectoryAttempt",
    "LiveDirectoryGroup", "OutputCandidate", "SourceRevisionChanged",
    "ValidatedSourceAliases",
    "execution_plan_values", "inspect_output", "materialize_deferred_output",
    "live_directory_groups", "materialize_live_directory_group",
    "OutputFact", "prepare_output", "source_snapshots",
    "target_state_matches", "validate_admitted_receipt",
    "validate_planned_source", "validate_source_aliases",
]
