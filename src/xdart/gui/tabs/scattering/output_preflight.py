from __future__ import annotations
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import numpy as np
from xrd_tools.core.filters import compile_filter
from xrd_tools.corrections.detector import (
    DEFAULT_MIN_FLAT_FRACTION,
    DetectorImageCheck,
    check_detector_image,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration import (
    apply_sensor_parallax,
    load_detector_calibration,
)
from xrd_tools.io import AppendDisposition, AppendRefused, load_mask
from xrd_tools.io.image import read_detector_image_layout, read_image
from xrd_tools.io.output_path import (
    FINITE_OPERATION_SLOTS,
    OVERWRITE_MODE,
    generated_artifact_family,
    is_artifact_family,
    resolve_output_target,
)
from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
)
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent
from xrd_tools.io.stat_identity import identity_ctime_ns
from xrd_tools.sources.adapters import candidate_owner, get_adapter, registry_identity
from xrd_tools.sources.descriptor import ContainerDescriptor
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.directory_index import StaleCandidateError
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.run_plan import RunCandidatePlan
from xrd_tools.sources.selection import DirectorySourceSpec, live_metadata_format, live_selection
from xrd_tools.sources.point_images import PointImagePolicy, PointImagesRequired, inspect_point_images
from xrd_tools.sources.execution_graph import (
    _CapturedSourceTopology,
    _capture_source_topology,
    _held,
    _same_source_revision,
    _source_state_key,
    _topology_from_captured_state,
    _verify_source_states,
    LivePrefixProof,
    observation_key,
    PreparedSourceExecutionGraph,
    SelectedContainerInput,
    SourceRevisionChanged,
    admitted_metadata_current,
    proven_states,
    qualify_source_execution_graph,
    source_snapshots_projection,
    validate_source_execution_graph,
    validate_source_aliases as _validate_source_aliases_shared,
)
from .contracts import (
    AcceptedScientificAssets, AdmittedOutput, AdmissionReceipt,
    ExternalSourceState, OutputDisposition, OutputFact, PlannedOutput,
    SourceExecutionStamp, SourceFileState, StartCapture,
)
from .source_metadata import ordered_motor_intersection

load_poni = load_detector_calibration
_LOG = logging.getLogger(__name__)

_SCIENTIFIC_ASSET_STREAM_CHUNK_BYTES = 1 << 16
_SCIENTIFIC_MASK_COMPATIBILITY_FLOOR_BYTES = 64 << 20
_SCIENTIFIC_MASK_HEADER_ALLOWANCE_BYTES = 1 << 20
_SCIENTIFIC_MASK_MAX_DECODED_BYTES = 256 << 20


@dataclass(frozen=True, slots=True)
class ValidatedSourceAliases:
    """Immutable result of the immediate two-sweep source proof."""

    identity: object
    targets: tuple[object, ...]




def validate_source_aliases(
    stamp: SourceExecutionStamp,
    *,
    cancelled: Callable[[], bool] | None = None,
    proven: dict[str, SourceFileState] | None = None,
) -> ValidatedSourceAliases:
    """Prove every raw alias and target twice in deterministic order."""

    identity, targets = _validate_source_aliases_shared(stamp, cancelled=cancelled, proven=proven)
    return ValidatedSourceAliases(identity, targets)


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
        saturation: int | None = None,
    ) -> "OutputCandidate":
        intent = capture.intent_snapshot.thaw()
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
        signed = accepted.signed_poni_values
        if signed is not None:
            intent.poni_values = signed
        frozen = intent.freeze(gi_motor_choices=gi_motor_choices, saturation=saturation)
        processing = frozen.as_provenance()
        processing["accepted_scientific_assets"] = {
            "poni_values": accepted.poni_projection,
            "poni_detector_config_json": accepted.poni_detector_config_json,
            "poni_sha256": accepted.poni_sha256,
            "mask_sha256": accepted.mask_sha256,
        }
        for key in ("flat_sha256", "dark_sha256"):
            if getattr(accepted, key) is not None:
                processing["accepted_scientific_assets"][key] = getattr(accepted, key)
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
    # Every output target of this run, keyed lexically and (when the target
    # already exists) by stat identity, so one JIT materialization can refuse
    # to read another group's output as raw data without rescanning the plan.
    _target_keys: frozenset[str] = field(init=False, repr=False, compare=False)
    _target_identities: frozenset[tuple[int, int]] = field(
        init=False, repr=False, compare=False,
    )

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
        object.__setattr__(self, "_target_keys", frozenset(
            _lexical_key(entry.target) for entry in self.entries
        ))
        object.__setattr__(self, "_target_identities", frozenset(
            (entry.fact.target_state[3], entry.fact.target_state[4])
            for entry in self.entries
            if type(entry.fact.target_state) is tuple
        ))

    @property
    def discovered_file_count(self) -> int:
        return len(self.discovered_paths)

    @property
    def targets(self) -> tuple[Path, ...]:
        return tuple(entry.target for entry in self.entries)

    def owns_target(self, path: Path) -> bool:
        """True when *path* is one of this run's output targets.

        Lexical (normcase/abspath) and resolved (realpath) keys catch the
        plain and symlinked spellings.  A hard link is only visible through
        stat identity: the admission-time identities cover targets that
        already existed then, and a dependency carrying more than one link
        is compared against the targets' *current* identities, so a link
        made after admission to an output this run has since written is
        refused as well.  A single-linked dependency (the normal case) is
        settled by its own stat alone.
        """
        if (
            _lexical_key(path) in self._target_keys
            or os.path.normcase(os.path.realpath(path)) in self._target_keys
        ):
            return True
        try:
            state = os.stat(path)
        except OSError:
            return False
        identity = (int(state.st_dev), int(state.st_ino))
        if identity in self._target_identities:
            return True
        if int(state.st_nlink) < 2:
            return False
        for entry in self.entries:
            try:
                current = os.stat(entry.target)
            except OSError:
                continue
            if (int(current.st_dev), int(current.st_ino)) == identity:
                return True
        return False


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
    # READY for only the group's leading members: a later one is still waiting
    # for its metadata, so the group must not settle on this attempt.
    partial: bool = False
    # READY: the proof of what this poll admitted (the members after a prefix
    # proved in it), which construction re-checks instead of validating again.
    proof: LivePrefixProof | None = field(default=None, compare=False)

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
            and type(self.partial) is bool
            and (not self.partial or self.state is ProbeState.READY)
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
    if type(intent.source_spec) is SourceSpec:
        detection = inspect_point_images(intent.source_spec)
        if detection is not None:
            if intent.live_mode:
                raise ValueError('Multiple images per point are unavailable in Live; choose a finite run.')
            raw_policy = intent.source_spec.options.get('point_images')
            policy = None if raw_policy is None else PointImagePolicy.from_dict(raw_policy)
            if policy is None or policy.images_per_point != detection.images_per_point:
                raise PointImagesRequired(detection)
    assets = _load_scientific_assets(intent, cancelled=cancelled)
    candidate = OutputCandidate.from_start_capture(capture, assets)
    source, choices, deferred = candidate.source, None, None
    directory_discovered_paths: tuple[Path, ...] = ()
    if cancelled():
        raise RuntimeError("admission cancelled")
    # A Folder, or in Live an Image Series bound to its members, is watched by one session.
    if (watch := live_selection(source)) is not None and (type(source) is DirectorySourceSpec or intent.live_mode):
        session = DirectoryIndexSession(
            retry_deadline=(float("inf") if intent.live_mode else None),
            probe_candidates=False,
        )
        session_owner(session)
        session.configure(watch.root, recursive=watch.recursive, name_filter=watch.name_filter,
                          suffixes=watch.suffixes, series=watch.series)
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
    automatic = intent.threshold.mask_saturation and not intent.threshold.apply_threshold
    saturation = _admitted_saturation(source, items, assets) if automatic else None
    if saturation is not None:
        # signed with the resolved ceiling before any Append inspection (decision 39)
        candidate = OutputCandidate.from_start_capture(
            capture, assets, choices, saturation=saturation,
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
        saturation=saturation,
    )


def _admitted_saturation(
    source: SourceSpec | DirectorySourceSpec,
    items: tuple[PlannedOutput, ...],
    assets: AcceptedScientificAssets,
) -> int | None:
    """The run's resolved automatic ceiling (decision 39), bound before signing.

    A single source takes the value it records (its container, or its first
    image's header: one scan's images share it), else the calibration detector's
    documented counter limit for integer data; a Live Image Series, admitted only
    after it starts, takes its selected container's or image's. A Directory run
    shares one science identity and qualifies its files only after it starts, so
    it takes the documented limit alone. None: the data type's rule, as before.
    """
    from xrd_tools.core.invalid import resolve_detector_saturation
    from xrd_tools.sources.execution_graph import graph_saturation, selected_saturation

    detector = None if assets.poni_values is None else assets.poni_values[7]
    if type(source) is SourceSpec:
        resolved = (graph_saturation(items[0].graph, detector=detector) if items
                    else selected_saturation(source, detector=detector))
    else:
        resolved = resolve_detector_saturation(detector=detector)
    return None if resolved is None or resolved.source == "dtype" else resolved.value
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


from xrd_tools.reduction.background import background_frame_fact as _background_frame_fact


def _merge_background_binding(bindings, binding, *, limit: int):
    """Insert one immutable binding with exact cumulative retained charge."""
    label, fact, descriptor, fingerprint = binding
    if hashlib.sha256(descriptor).hexdigest() != fingerprint or len(descriptor) > 262_144:
        raise ValueError("Background dependency is malformed")
    prior = next((value for value in bindings if value[0] == label), None)
    if prior is not None:
        if prior != binding: raise ValueError("Background dependency changed for an admitted label")
        return bindings
    charge = sum(1024 + len(json.dumps(value[1], ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()) + len(value[2]) + 64 for value in bindings)
    charge += 1024 + len(json.dumps(fact, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()) + len(descriptor) + 64
    if charge > limit: raise ValueError("Background binding map exceeds its allocation")
    return tuple(sorted((*bindings, binding), key=lambda value: value[0]))


from xrd_tools.reduction.background import background_resource_terms as _background_resource_terms


from xrd_tools.sources.run_plan import directory_candidate_groups as _directory_candidate_groups


def _save_path_is_directory(path: Path) -> bool:
    """An existing folder stays a folder even when its name contains dots."""
    return not path.suffix or path.is_dir()


def _directory_target(
    configuration: FrozenRunConfiguration | OutputCandidate,
    plan: RunCandidatePlan,
    candidate: Candidate,
    name: str,
) -> Path:
    output_root = Path(configuration.save_path)
    output_directory: Path | None = None
    if _save_path_is_directory(output_root):
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
    # Hand the directory DOWN as a directory (F1).  Passing it as a request
    # string let a dotted sub-folder read as a file and split the naming.
    directory, family = _run_output_naming(configuration, name, output_directory)
    return _generated_target_in(
        directory, family, _run_output_slot(configuration),
    )


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


def _deferred_directory_plan(
    configuration: OutputCandidate,
    plan: RunCandidatePlan,
    discovered_paths: tuple[Path, ...],
    *,
    cancelled: Callable[[], bool],
    prefix: PreparedSourceExecutionGraph | None = None,
) -> DeferredDirectoryPlan:
    """``prefix``: a live published graph proved this poll, whose members keep their admitted states."""
    if not plan.candidates:
        raise RuntimeError("source has no matching candidates")
    proven = proven_states(prefix)
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
    for group, name in _output_groups(configuration, plan):
        if cancelled():
            raise RuntimeError("admission cancelled")
        captured_topologies: list[_CapturedSourceTopology] = []
        for source_candidate in group:
            if cancelled():
                raise RuntimeError("admission cancelled")
            state = proven.get(str(source_candidate.path))
            if state is None or (state.size, state.mtime_ns) != source_candidate.version_stamp:
                state = _capture_exact_candidate_state(source_candidate)
            captured_topologies.append(
                _topology_from_captured_state(
                    state, candidate_owner_id=source_candidate.adapter_id,
                )
            )
            if cancelled():
                raise RuntimeError("admission cancelled")
        candidate_states = tuple(captured_topologies)
        representative = group[0]
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
        proven,
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
            if proven.get(str(source_candidate.path)) is not topology.followed_state:
                _validate_exact_candidate_state(source_candidate, topology.followed_state)
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
            if proven.get(value.followed_state.path) is not value.followed_state
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


def _correction_paths(configuration: object) -> tuple[str, ...]:
    """The flat-field and dark files an output must never overwrite.

    Admission guards receive the signed ``OutputCandidate``, which carries them
    only in its processing mapping; a frozen configuration has them as fields.
    """
    if type(configuration) is OutputCandidate:
        values = configuration.processing_mapping()
    else:
        values = {key: getattr(configuration, key, "") for key in ("flat_file", "dark_file")}
    return tuple(str(values[key]) for key in ("flat_file", "dark_file") if values.get(key))


def _validate_deferred_targets(
    configuration: OutputCandidate,
    plan: RunCandidatePlan,
    targets: tuple[Path, ...],
    external_paths: tuple[Path, ...],
    proven: dict[str, SourceFileState] | None = None,
) -> None:
    source = live_selection(configuration.source)
    if source is None:
        raise TypeError("deferred targets require a directory source")
    raw_paths = (
        tuple(candidate.path for candidate in plan.candidates)
        + external_paths
    )
    # A proved prefix member is its admitted state: its resolution and identity.
    held = [(proven or {}).get(str(path)) for path in raw_paths]
    raw_norms = {os.path.normcase(os.path.realpath(path) if state is None else state._resolved_path_at_capture)
                 for path, state in zip(raw_paths, held)}
    raw_inodes: set[tuple[int, int]] = {(state.device, state.inode) for state in held if state is not None}
    for path, state in zip(raw_paths, held):
        if state is not None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        raw_inodes.add((int(stat.st_dev), int(stat.st_ino)))
    protected = tuple(
        Path(value)
        for value in (configuration.poni_file, configuration.mask_file,
                      *_correction_paths(configuration))
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
            container_directory_mode=source.series is None,  # a series' bound already keeps outputs out
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
    for candidates, name in _output_groups(receipt.candidate, current):
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
                name,
            ),
            subset.paths,
        ))
    return tuple(groups)


def _output_groups(configuration: FrozenRunConfiguration | OutputCandidate,
                   plan: RunCandidatePlan) -> tuple[tuple[tuple[Candidate, ...], str], ...]:
    """The plan's output groups and scan names: a Live numbered Image Series is one group in its own members'
    order under its own name; otherwise each adapter's scan name groups the files of one folder."""
    selected = (configuration.thaw_source_spec() if type(configuration) is FrozenRunConfiguration
                else configuration.source)
    watch = live_selection(selected) if _is_live(configuration) else None
    series = None if watch is None else watch.series
    if series is not None and series.stem is not None:
        return ((tuple(sorted(plan.candidates, key=lambda value: series.key(value.path.name))), series.stem),) \
            if plan.candidates else ()
    groups = []
    for group in _directory_candidate_groups(plan):
        adapter = get_adapter(group[0].adapter_id)
        if adapter is None:
            raise ValueError(f"candidate {group[0].path} lost its adapter")
        groups.append((group, adapter.scan_name(group[0].path)))
    return tuple(groups)


def _live_metadata_ready(
    configuration: FrozenRunConfiguration,
    group: LiveDirectoryGroup,
    observations: dict[str, tuple[tuple[str, object], ...]],
    *,
    published_members: int = 0,
    observe_from: int = 0,
    spec_cache: dict | None = None,
) -> tuple[int, str]:
    """How many leading members of a Live group have their metadata ready.

    A member is ready when its selected sidecar (an explicit format) or its
    Auto candidates look the same on two observations a poll apart, and an
    explicit sidecar exists, so a new frame waits for a sidecar still being
    written or about to land.  Under Auto, no sidecar on both observations
    still means no metadata.  Under SPEC a new member is ready once its own
    SPEC row is complete: a row is final once written, so one look suffices
    (qualification then proves it), and no other point's row is ever borrowed;
    published members are not re-gated.  Metadata-off Runs do not wait.
    ``observations`` is the Run's own record, one entry per image, replaced at
    every look. Published members may have lost their sidecar: after the
    observation settles they must reach the published-metadata drift fence,
    rather than wait forever as if they were new frames.
    Also returns why the first member that is not ready waits.
    """

    candidates = group.plan.candidates
    meta_format = live_metadata_format(configuration.thaw_source_spec())
    if meta_format is None:
        return len(candidates), ""
    if meta_format == "spec":
        from xrd_tools.io.metadata import spec_rows_ready

        exempt = min(published_members, len(candidates))
        ready = exempt + spec_rows_ready(tuple(Path(value.path) for value in candidates[exempt:]),
                                         cache=spec_cache)
        if ready == len(candidates):
            return ready, ""
        return ready, f"waiting for the SPEC row of {Path(candidates[ready].path).name}"
    if SourceKind.IMAGE_FILE not in getattr(get_adapter(candidates[0].adapter_id), "kinds", (SourceKind.IMAGE_FILE,)):
        return len(candidates), ""  # a container's own metadata: sidecars are an image policy
    from xrd_tools.io.metadata import live_sidecar_observation

    ready, reason, listings = len(candidates), "", {}
    for index, candidate in enumerate(candidates):
        if index < observe_from:
            continue  # a published member its output's current proof covers
        image = Path(candidate.path)
        seen = live_sidecar_observation(image, meta_format, listings=listings)
        key = observation_key(image)
        previous = observations.get(key)
        observations[key] = seen
        if ready == len(candidates) and (
            previous != seen
            or any(revision is None for _path, revision in seen)
            or (meta_format != "auto" and not seen and index >= published_members)
        ):
            ready, reason = index, f"waiting for the metadata of {image.name}"
    return ready, reason


def observe_live_members(configuration: FrozenRunConfiguration, paths) -> dict[str, object]:
    """Each image's sidecar observation now, from fresh listings, by record key."""
    from xrd_tools.io.metadata import live_sidecar_observation

    fmt = live_metadata_format(configuration.thaw_source_spec())
    listings: dict = {}
    return {} if fmt in (None, "spec") else {
        observation_key(path): live_sidecar_observation(path, fmt, listings=listings) for path in paths}


def _prefix_current_rebased(configuration: FrozenRunConfiguration, graph: PreparedSourceExecutionGraph,
                            proof: LivePrefixProof | None, cancelled: Callable[[], bool]) -> bool:
    """The strict published-prefix check; a pass rebases the proof on what was observed before it."""
    proof = proof if proof is not None and proof.graph is graph else None
    seen = None if proof is None else observe_live_members(
        configuration, [member.path for member in graph.stamp.members])
    registry = registry_identity()
    if not _published_prefix_current(graph, cancelled):
        return False
    if proof is not None:
        proof.rebase(seen, registry)
    return True


def _published_prefix_current(
    graph: PreparedSourceExecutionGraph, cancelled: Callable[[], bool],
) -> bool:
    """Whether the published frames' admitted graph still validates exactly.

    :func:`validate_source_execution_graph` proves its aliases and candidate
    owners, every raw member's exact state, every metadata dependency read
    again (SPEC rows strictly, their extent extended), the admitted motor
    values, read policy and cardinality.  A Stop inside it is admission's
    cancellation; any other failure means "not current" and is left to
    admission, so its existing policy (the published-metadata fence, the
    Append guard) decides.
    """
    try:
        validate_source_execution_graph(graph, cancelled=cancelled)
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        return False
    except InterruptedError as error:
        if error.args == ("source qualification cancelled",):
            raise RuntimeError("admission cancelled") from error
        return False
    except Exception:
        return False
    return True


def materialize_live_directory_group(
    receipt: AdmissionReceipt,
    configuration: FrozenRunConfiguration,
    session: DirectoryIndexSession,
    group: LiveDirectoryGroup,
    *,
    cancelled: Callable[[], bool],
    reprobe: bool = False,
    sidecar_observations: dict[str, tuple[tuple[str, object], ...]] | None = None,
    published_members: int = 0,
    published_graph: PreparedSourceExecutionGraph | None = None,
    proof: LivePrefixProof | None = None,
    spec_cache: dict | None = None,
) -> LiveDirectoryAttempt:
    """Probe and JIT-admit one exact Live group revision.

    Provisional or concurrently changing bytes produce a typed retry-later
    result.  Output collision/configuration failures remain hard failures.
    With ``sidecar_observations`` (the live Run's record), only the group's
    leading members whose metadata is ready are admitted
    (:func:`_live_metadata_ready`); a later ready member waits behind an
    earlier pending one, and such an attempt is ``partial``.  When nothing
    beyond the published frames is ready, the attempt waits without admission
    while ``published_graph`` (their admitted graph) still validates exactly;
    any other outcome is admitted as before.
    """

    fresh = getattr(session, "fresh", frozenset())  # first seen by this poll's listing
    try:
        observations = list(session.probe_candidates(
            group.plan.candidates, refresh=False, reprobe=reprobe, cancelled=cancelled))
    except StaleCandidateError as error:
        return LiveDirectoryAttempt(group, ProbeState.IN_PROGRESS, reason=str(error))
    if cancelled():
        raise RuntimeError("admission cancelled")
    if observations and observations[-1].result.state is not ProbeState.READY:
        return LiveDirectoryAttempt(group, observations[-1].result.state, reason=observations[-1].result.reason)
    physical_paths = group.physical_paths
    prefix = None
    if sidecar_observations is not None:
        # A current proof covers the published members (each output is proved
        # again within LivePrefixProof.PERIOD_S): only later members are looked at.
        checked = None if proof is None else proof.verified_at
        covered = published_members if proof is not None and proof.graph is published_graph \
            and proof.current(cancelled=cancelled) else 0
        count, waiting = _live_metadata_ready(
            configuration, group, sidecar_observations,
            published_members=published_members,
            observe_from=covered, spec_cache=spec_cache,
        )
        if covered and count > covered and proof.verified_at == checked and not proof.holds(cancelled=cancelled):
            # A publication proves its prefix in its own poll; one that changed
            # is looked at again in full, as an uncovered poll would.
            covered, (count, waiting) = 0, _live_metadata_ready(
                configuration, group, sidecar_observations, published_members=published_members, spec_cache=spec_cache)
        # A file only this listing has seen waits for another: one that ran during writes can show
        # a later file and miss an earlier one, and frame labels follow the members' order.
        if (cut := next((index for index, value in enumerate(group.plan.candidates[published_members:count],
                                                             published_members) if value.path in fresh), None)) is not None:
            count, waiting = cut, f"confirming {Path(group.plan.candidates[cut].path).name} in another listing"
        if count == 0:
            return LiveDirectoryAttempt(
                group, ProbeState.IN_PROGRESS, reason=waiting,
            )
        if (
            published_graph is not None
            and count < len(group.plan.candidates)
            and count <= len(published_graph.stamp.members)
            and (covered or _prefix_current_rebased(configuration, published_graph, proof, cancelled))
        ):
            # A frame waits and the published frames are exactly as admitted:
            # re-admitting them would only repeat their admission (and, in
            # Append mode, its same-extent preview against a grown SPEC file).
            return LiveDirectoryAttempt(
                group, ProbeState.IN_PROGRESS, reason=waiting,
            )
        observations = observations[:count]
        physical_paths = physical_paths[:count]
        if getattr(session, "listing_deferred", False):
            # An admission needs a real listing in its own poll (D1): the next one.
            session.force_listing()
            return LiveDirectoryAttempt(group, ProbeState.IN_PROGRESS, reason="listing the directory before admission")
        # The prefix proved in this poll: only the members after it are read,
        # captured and proved strictly.
        prefix = published_graph if covered else None

    ready = RunCandidatePlan(
        group.plan.generation,
        tuple(value.candidate for value in observations),
        group.plan.root,
        group.plan.recursive,
        group.plan.name_filter,
        tuple(value.descriptor for value in observations),
    )
    registry, admitted = registry_identity(), []
    fmt = live_metadata_format(configuration.thaw_source_spec())
    # The discovery the in-poll proof compares with is taken before qualification.
    record = sidecar_observations if sidecar_observations is not None else observe_live_members(
        configuration, ready.paths)

    def recheck(graph: PreparedSourceExecutionGraph) -> bool:
        admitted.append(LivePrefixProof(graph, record=record, registry=registry, metadata_format=fmt,
                                        start=0 if prefix is None else len(prefix.stamp.members)))
        return admitted[-1].holds(cancelled=cancelled)

    try:
        deferred = _deferred_directory_plan(
            receipt.candidate,
            ready,
            physical_paths,
            cancelled=cancelled,
            prefix=prefix,
        )
        transient = replace(
            receipt,
            deferred_directory=deferred,
            directory_discovered_file_count=len(physical_paths),
            directory_discovered_paths=physical_paths,
        )
        decision, _ready_files, _skipped_files = materialize_deferred_output(
            transient,
            session,
            deferred.entries[0],
            cancelled=cancelled,  # a container's published frames: its SPEC rows' fence (R5b)
            published_members=published_members if published_graph is None or published_graph.descriptor is None
            else published_graph.stamp.frame_count,
            prefix=prefix, spec_cache=spec_cache, recheck=recheck,
        )
    except RuntimeError as error:
        if error.args == ("admission cancelled",):
            raise
        raise
    except InterruptedError as error:
        # JIT source revalidation has the same Stop token as qualification.
        if error.args != ("source qualification cancelled",):
            raise
        raise RuntimeError("admission cancelled") from error
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
        partial=len(physical_paths) < len(group.physical_paths),
        proof=admitted[-1],
    )


def already_processed_append(receipt: AdmissionReceipt, entry: DeferredDirectoryEntry):
    """A finite directory Append's scan that its output already holds, recognized from the Run's directory listing
    and the output's committed record, without reading the scan again: its ``CommittedAppendCoverage``, or None to
    send it the full way (maintainer, 2026-10-23: "If something has already been processed, it should be skipped
    anyway"; Codex's design review DESIGN-append-smb-125). TIFF images only; Live, other formats, new or renamed
    files and a file whose size or time changed all go the full way."""
    configuration = receipt.candidate._configuration
    if (
        configuration is None
        or configuration.output_mode != "Append"
        or configuration.live_mode
        or configuration.background.mode != "None"
        or entry.skip_reason
        or any(candidate.adapter_id != "image_file" for candidate in entry.candidates)
    ):
        return None
    from xrd_tools.io.append import committed_append_coverage, science_fingerprint
    from .adapters.dynamic_output import _mode_tokens, _science_projection
    return committed_append_coverage(
        entry.target,
        entry="entry",
        source_base=configuration.project_root or "",
        science_fingerprint=science_fingerprint(
            _science_projection(configuration, receipt.candidate.processing_mapping())
        ),
        modes=_mode_tokens(configuration),
        members=tuple(
            (candidate.path, candidate.size, candidate.mtime_ns)
            for candidate in entry.candidates
        ),
    )


def materialize_deferred_output(
    receipt: AdmissionReceipt,
    session: DirectoryIndexSession,
    entry: DeferredDirectoryEntry,
    *,
    cancelled: Callable[[], bool],
    published_members: int = 0,
    prefix: PreparedSourceExecutionGraph | None = None, spec_cache: dict | None = None,
    recheck: Callable[[PreparedSourceExecutionGraph], bool] | None = None,
) -> tuple[AdmittedOutput | None, int, int]:
    """``recheck`` (a live admission's in-poll proof) is a TIFF series' second look, instead of a
    second validation (a container keeps it); ``prefix``/``spec_cache``: as in _directory_items."""
    deferred = receipt.deferred_directory
    if (
        type(deferred) is not DeferredDirectoryPlan
        or type(entry) is not DeferredDirectoryEntry
        or not any(value is entry for value in deferred.entries)
    ):
        raise TypeError("deferred output is not owned by this receipt")
    proven = proven_states(prefix)
    _validate_deferred_entry_states(entry, cancelled=cancelled, proven=proven)
    if entry.skip_reason:
        return None, 0, len(entry.physical_paths)

    ready: list[Candidate] = []
    descriptors: list[ContainerDescriptor | None] = []
    skipped = 0
    # ``validate_admitted_receipt`` refreshes and reconciles the complete
    # name/stat snapshot once before execution.  Refreshing again here would
    # rescan the whole directory for every member (O(N^2)); each exact probe
    # revalidates its candidate, all on one owner call.
    for observed in session.probe_candidates(
        entry.candidates, refresh=False, stop_unready=False, cancelled=cancelled,
    ):
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
            published_members=published_members,
            prefix=prefix, spec_cache=spec_cache,
        )
    except (ValueError, OSError):
        # A semantic/schema failure remains terminal when every admitted
        # source state is still exact.  If the same failure followed a real
        # byte/dependency drift, this exact revalidation raises the typed Live
        # retry signal instead.
        _validate_deferred_entry_states(entry, cancelled=cancelled, proven=proven)
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
        proven=proven, revalidate=recheck is None or item.descriptor is not None,
    )
    if recheck is not None and not recheck(item.graph):
        raise SourceRevisionChanged(f"live source changed during admission: {item.target.name}")
    _validate_targets(
        receipt.candidate,
        receipt.candidate.source,
        (item,),
        cancelled=cancelled,
        run_plan=deferred,
        proven=proven,
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
    proven: dict[str, SourceFileState] | None = None, revalidate: bool = True,
) -> None:
    """Require one deferred cursor to remain inside its admitted revision; ``proven`` states (a live
    prefix proved this poll) are not proved again, and ``revalidate=False`` skips the graph's validation."""

    is_cancelled = _not_cancelled if cancelled is None else cancelled
    proven = proven or {}
    if item is not None and revalidate:
        # The complete JIT stamp is the first owner capable of proving every
        # raw alias binding.  Run that proof before the older deferred-state
        # fences so no alias drift can be classified without its frozen
        # raw-to-resolved identity reaching the central validator.
        validate_source_execution_graph(
            item.graph, cancelled=is_cancelled,
        )
    topologies = entry._protected_topology or tuple(
        _topology_from_captured_state(value)
        for value in entry.protected_states
    )
    topology_map = {
        _source_state_key(value.raw_path): value
        for value in topologies
    }
    _verify_source_states({key: value for key, value in topology_map.items()
                           if proven.get(value.followed_state.path) is not value.followed_state}, cancelled=is_cancelled)
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
    checked: set[int] = set()  # a SPEC file is every member's metadata file: one state, checked once
    for state, role in occurrences:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        if proven.get(state.path) is state or id(state) in checked:
            continue
        checked.add(id(state))
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
            if role == "image_metadata" and admitted_metadata_current(stamp, state):
                continue  # a SPEC file that only grew past its members' rows
            raise SourceRevisionChanged(
                "source dependency escaped admitted revision: "
                f"{state.path}"
            )
        if not _same_source_revision(accepted, state):
            raise SourceRevisionChanged(
                "source dependency escaped admitted revision: "
                f"{state.path}"
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


#: GUI run mode -> the SHARED operation token.  Deliberately NOT a table of
#: suffixes: `FINITE_OPERATION_SLOTS` owns the one suffix vocabulary, and a
#: second copy here is how `_int2d` in one place and `_int2d` in another drift
#: apart.  This maps only the GUI's own spelling onto that vocabulary.
#:
#: Only the three native run modes appear; anything else is not a run mode and
#: must not silently pick a slot.  ``Int 1D (XYE)`` writes NO ``.nexus`` at all
#: (`dynamic_output.py` guards the whole sink behind ``if not xye_only:``), but
#: a target is still resolved for it as a grouping key, so it maps to the token
#: its science would have used rather than raising.
_RUN_MODE_OPERATIONS = {
    "Int 1D": "int-1d",
    "Int 2D": "int-2d",
    "Int 1D (XYE)": "int-1d",
}

#: Derived, never restated.  A token this GUI names that the shared vocabulary
#: does not define is a programming error, not an operator one.
_RUN_MODE_SLOTS = {
    mode: FINITE_OPERATION_SLOTS[token]
    for mode, token in _RUN_MODE_OPERATIONS.items()
}


def _run_output_slot(
    configuration: "FrozenRunConfiguration | OutputCandidate",
) -> str:
    """The stable public slot this run publishes into.

    Refuses an unrecognised mode rather than defaulting, and an ABSENT mode is
    unrecognised.  A wrong slot here is a run writing to the wrong filename,
    which is worse than a loud failure.

    Fable F2 on `4fe073e8`.  An empty mode used to return NO slot on the
    argument that `from_start_capture` signs every candidate.  That named the
    wrong guard: the branch fires on the FROZEN configuration's
    `processing_mode`, and `from_start_capture` only rejects XYE+Append.  What
    actually stops an empty mode in the GUI is the run strip, which sets a
    mode blocker and disables Start (`run_mode_projection.py:76-80`) -- but
    `RunIntent.__post_init__` and `freeze()` both accept `""`, so any
    programmatic or persisted intent reached admission and planned an
    UN-SLOTTED public name.  It now raises like any other unknown mode.
    """
    frozen = (
        configuration._configuration
        if type(configuration) is OutputCandidate
        else configuration
    )
    mode = None if frozen is None else str(frozen.processing_mode)
    if mode is None and type(configuration) is OutputCandidate:
        # A candidate signed by `from_start_capture` always carries the frozen
        # configuration; a hand-built one may not, but it still carries the
        # provenance mapping the mode was frozen into.
        declared = configuration.processing_mapping().get("processing_mode")
        mode = None if declared is None else str(declared)
    if mode is None:
        # NOTHING declared a mode anywhere -- no frozen configuration AND no
        # provenance entry.  That is a source-shaped double naming a target it
        # will never reduce, not a run; `_directory_items` is exercised that way
        # directly.  Take NO slot rather than inventing one.  This is NARROWER
        # than the branch F2 removed, which also swallowed a DECLARED empty mode
        # on a real frozen configuration -- the reachable case.
        return ""
    slot = _RUN_MODE_SLOTS.get(mode)
    if slot is None:
        # Either no mode reached here at all, or one did and this policy does
        # not know it.  Both are the dangerous case: continuing would publish
        # under some other mode's slot, or under no slot at all.
        raise ValueError(
            f"run processing mode has no stable output slot: {mode!r}"
        )
    return slot


def _run_output_naming(
    configuration: "FrozenRunConfiguration | OutputCandidate",
    scan_name: str,
    output_directory: Path | None = None,
) -> tuple[Path, str]:
    """Decide ONCE where a run writes and which family it publishes into.

    Returns ``(directory, family)``.  The public name is
    ``<family><slot>.nexus`` inside *directory*, and *family* is the value
    persisted as ``@artifact_family_v1``, so the filename and the recorded
    family CANNOT disagree -- they are the same string, used twice.

    Fable F1 on `4fe073e8`.  They used to be decided by two functions reading
    two DIFFERENT inputs: the filename came from a per-candidate
    ``output_request`` that re-inferred file-vs-directory from
    ``Path(...).suffix``, while the family came from ``configuration.save_path``.
    A recursive raw sub-folder whose name contains a dot (`2026.09.04/`,
    `run.001/`, `sample1.5V/`) reads as a suffix, so the sub-folder was treated
    as an explicit FILE request: the run published
    `processed/2026.09_int2d.nexus` while recording the family `scan_0001`.  A
    later Reintegrate then resolved `processed/scan_0001_reintegrate1d.nexus` --
    a slot in a family whose Run file does not exist.

    *output_directory* is the already-validated per-candidate directory for a
    directory-shaped request (recursive sources keep their parent beneath the
    selected root).  It is used as a DIRECTORY and never re-inspected for a
    suffix, which is what makes the dotted sub-folder safe.
    """
    requested = Path(configuration.save_path)
    if not _save_path_is_directory(requested):
        # An explicit FILE request: its parent is the directory and its stem is
        # the family.  A per-candidate directory does not apply.
        directory, family = requested.parent, requested.stem
    else:
        # An AUTOMATIC name.  The grazing-incidence marker is decided here, once,
        # as part of the family, so the persisted family and every filename
        # derived from it agree and a later operation cannot repeat it.
        directory, family = (
            requested if output_directory is None else output_directory,
            generated_artifact_family(
                scan_name, grazing_incidence=_gi_enabled(configuration),
            ),
        )
    if not is_artifact_family(family):
        # REFUSE NOW, ruled 2026-09-04.  This family is what gets written into
        # the artifact for every later operation to consume, and one that
        # breaks the rule cannot resolve `<family><slot>.nexus` at all -- so an
        # Average or Reintegrate on this run's output would refuse afterwards,
        # with a message naming nothing the operator can act on.  Better to say
        # so before the run than after it.
        raise ValueError(
            f"cannot use {family!r} as a processed-result name: it may not "
            "start with '.', '-', '_' or a space, may not end with a space or "
            "'.', may not contain any of / \\ : * ? \" < > |, and is at most "
            "80 characters. Rename the scan, or choose a different output name."
        )
    return directory, family


def _run_artifact_family(
    configuration: "FrozenRunConfiguration | OutputCandidate", scan_name: str,
) -> str:
    """The ROOT family a run publishes into.

    Thin view onto :func:`_run_output_naming`, which decides it alongside the
    directory.  Kept as a name because the family is the thing every LATER
    operation consumes, and it reads better at the call site than a tuple index.
    """
    return _run_output_naming(configuration, scan_name)[1]


def _generated_target_in(directory: Path | str, family: str, slot: str) -> Path:
    """``<family><slot>.nexus`` inside *directory*, via the shared owner.

    *directory* is ALWAYS a directory.  Nothing here re-decides file-vs-
    directory; that decision belongs to :func:`_run_output_naming` and is made
    once, from the requested save path.
    """
    return Path(resolve_output_target(
        directory,
        f"{family}{slot}",
        mode=OVERWRITE_MODE,
        explicit_target=None,
    ))


def _resolved_generated_target(
    save_path: str, scan_name: str, slot: str = "",
    *, grazing_incidence: bool = False,
) -> Path:
    """Delegate vNext's one generated-output naming decision to the shared owner.

    vNext admission is Overwrite-only.  A suffix-shaped requested path keeps
    its directory/stem but is normalized to ``.nexus``; a directory request
    generates ``<scan><slot>.nexus`` (P4/OUT-1, ADR-0010).

    *slot* is empty for the Average NAMING ANCHOR, which is not a written file:
    it supplies the directory and the root FAMILY, so appending a run slot to it
    would make Average derive `<scan>_int2d` as its family and publish the
    chained `<scan>_int2d_int2d_average.nexus` (a 2-D Average's own slot is
    `_int2d_average`).  Callers that name a real run target
    pass the slot from :func:`_run_output_slot`.

    *grazing_incidence* marks an automatic family exactly as
    :func:`_run_output_naming` does, so a GI Average lands beside the GI run.

    This is the NO-per-candidate-directory case (a whole-request target, and
    `page.py`'s Average anchor).  Directory-shaped runs that place each source
    beneath its own parent call :func:`_run_output_naming` with that directory.
    """
    requested = Path(save_path)
    directory, family = (
        (requested.parent, requested.stem)
        if not _save_path_is_directory(requested)
        else (requested, generated_artifact_family(
            scan_name, grazing_incidence=grazing_incidence,
        ))
    )
    return _generated_target_in(directory, family, slot)


def _not_cancelled() -> bool:
    return False


def _capture_source_states(
    paths: tuple[Path, ...],
    cancelled: Callable[[], bool],
) -> tuple[SourceFileState, ...]:
    """Capture an exact member stamp without delaying cancellation.

    File-state capture can block on beamline storage.  Check both sides of
    every member so cancellation raised during one capture cannot trigger a
    sweep of all remaining sources before admission notices it.
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


def _qualify_for_admission(
    source: SourceSpec,
    *,
    selected_motor: str | None = None,
    selected_container: SelectedContainerInput | None = None,
    cancelled: Callable[[], bool],
    infer_point_images: bool = True,
    published_members: int = 0,
    prefix: PreparedSourceExecutionGraph | None = None,
    spec_cache: dict | None = None,
) -> PreparedSourceExecutionGraph:
    try:
        return qualify_source_execution_graph(
            source, selected_motor=selected_motor,
            selected_container=selected_container, cancelled=cancelled,
            infer_point_images=infer_point_images,
            published_members=published_members,
            prefix=prefix, spec_cache=spec_cache,
        )
    except InterruptedError as error:
        if error.args != ("source qualification cancelled",):
            raise
        raise RuntimeError("admission cancelled") from error


def _series_item(
    configuration: FrozenRunConfiguration | OutputCandidate, source: SourceSpec,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> PlannedOutput:
    graph = _qualify_for_admission(
        source, selected_motor=_selected_tiff_gi_motor(configuration),
        cancelled=cancelled,
    )
    directory, family = _run_output_naming(configuration, graph.group_key)
    return PlannedOutput(graph,
        _generated_target_in(directory, family, _run_output_slot(configuration)),
        artifact_family=family)


def _gi_enabled(
    configuration: FrozenRunConfiguration | OutputCandidate,
) -> bool:
    if type(configuration) is not OutputCandidate:
        return bool(configuration.gi.enabled)
    gi = configuration.processing_mapping().get("gi", {})
    return bool(type(gi) is dict and gi.get("enabled"))


def _is_live(configuration: FrozenRunConfiguration | OutputCandidate) -> bool:
    if type(configuration) is FrozenRunConfiguration:
        return bool(configuration.live_mode)
    return bool(configuration.processing_mapping().get("live_mode"))


def _uses_eager_directory_descriptors(
    configuration: FrozenRunConfiguration | OutputCandidate,
) -> bool:
    if type(configuration) is FrozenRunConfiguration:
        return bool(configuration.gi.enabled or configuration.live_mode)
    return bool(
        configuration.processing_mapping().get("live_mode")
        or _gi_enabled(configuration)
    )


def _directory_items(
    configuration: FrozenRunConfiguration | OutputCandidate,
    plan: RunCandidatePlan,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
    published_members: int = 0,
    prefix: PreparedSourceExecutionGraph | None = None,
    spec_cache: dict | None = None,
) -> tuple[PlannedOutput, ...]:
    output_root, items, consumed = Path(configuration.save_path), [], set()
    resolved_output_root = None
    if _save_path_is_directory(output_root):
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
    watch = live_selection(selected_source)
    metadata_format = "auto" if watch is None else watch.metadata_format
    # A Live Image Series keeps its own selection: options, name and members' order.
    series = None if watch is None or not _is_live(configuration) else watch.series
    groups = {value.path: group for group in _output_groups(configuration, plan) for value in group[0]}
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
            members, name = groups[candidate.path]
            consumed.update(value.path for value in members)
            spec = SourceSpec(
                candidate.path.parent, SourceKind.TIFF_SERIES,
                options={
                    "selected_file": str(candidate.path),
                    "scan_name": name,
                    "metadata_format": metadata_format,
                    **({} if series is None else dict(selected_source.options)),
                    "files": tuple(value.path for value in members),
                },
            )
            graph = _qualify_for_admission(
                spec,
                selected_motor=_selected_tiff_gi_motor(configuration),
                cancelled=cancelled,
                # Live never infers images per point: an image waits for its row.
                infer_point_images=not _is_live(configuration),
                published_members=published_members,
                prefix=prefix, spec_cache=spec_cache,
            )
        else:
            consumed.add(candidate.path)
            if _uses_eager_directory_descriptors(configuration):
                state = _capture_source_states(
                    (candidate.path,), cancelled,
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
                    (candidate.path,), cancelled,
                )[0]
            if descriptor.kind is SourceKind.PROCESSED_NEXUS:
                raise ValueError("processed output cannot be raw input")
            if descriptor.frame_count < 1:
                continue
            name = descriptor.scan_name or candidate.path.stem.removesuffix("_master")
            spec = SourceSpec(
                candidate.path, descriptor.kind,
                entry=descriptor.resolved_entry or descriptor.requested_entry,
                # A Live series keeps its selection; a Live SPEC run binds each frame to its row (R5b).
                options=dict(selected_source.options) if series is not None else
                {"metadata_format": "spec"} if _is_live(configuration) and metadata_format == "spec" else {},
            )
            graph = _qualify_for_admission(
                spec,
                selected_container=SelectedContainerInput(
                    state, candidate.adapter_id, descriptor,
                ),
                cancelled=cancelled, infer_point_images=not _is_live(configuration),
                published_members=published_members, spec_cache=spec_cache,  # a container's published frames
            )
        output_directory: Path | None = None
        if _save_path_is_directory(output_root):
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
        # ONE naming decision feeds both the filename and the recorded family,
        # so they cannot diverge for a dotted sub-folder (Fable F1).
        directory, family = _run_output_naming(
            configuration, name, output_directory,
        )
        items.append(PlannedOutput(
            graph,
            _generated_target_in(
                directory, family, _run_output_slot(configuration),
            ),
            candidate,
            family,
        ))
    return tuple(items)
def validate_planned_source(
    item: PlannedOutput,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    is_cancelled = _not_cancelled if cancelled is None else cancelled
    try:
        validate_source_execution_graph(
            item.graph, cancelled=is_cancelled,
        )
    except InterruptedError as error:
        # The same cancellation token as admission (_qualify_for_admission),
        # so a Stop during a live re-check stops the Run instead of failing it.
        if error.args != ("source qualification cancelled",):
            raise
        raise RuntimeError("admission cancelled") from error
def source_snapshots(item: PlannedOutput) -> dict[str, dict[str, Any]]:
    return source_snapshots_projection(
        item.graph, writer=False
    )
from xrd_tools.session.readiness import execution_plan_values


from xrd_tools.session.readiness import native_int_reduction_plan
def _validate_targets(
    configuration: FrozenRunConfiguration | OutputCandidate,
    source: SourceSpec | DirectorySourceSpec,
    items: tuple[PlannedOutput, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
    run_plan: DeferredDirectoryPlan | None = None,
    proven: dict[str, SourceFileState] | None = None,
) -> None:
    raw: tuple[Path, ...] = ()
    proven = proven or {}
    for item in items:
        identity = validate_source_aliases(
            item.source_stamp,
            cancelled=cancelled,
            proven=proven,
        )
        # A proved prefix input keeps its admitted resolution and identity, so
        # only the output could have become it: that is checked once below.
        targets = identity.targets
        raw += tuple(Path(binding.raw_path) for binding in identity.identity.aliases
                     if not _held(proven, binding.raw_path, targets[binding.target_id].state))
        raw += tuple(Path(target.resolved_path) for target in targets
                     if not _held(proven, target.resolved_path, target.state))
        output = _path_state(item.target) if proven else None
        resolved = os.path.normcase(os.path.realpath(item.target)) if proven else None
        if any(type(output) is tuple and output[3:5] == (held.device, held.inode)
               or os.path.normcase(held._resolved_path_at_capture or held.path) == resolved
               for held in proven.values()):
            raise OutputCollisionError(
                f"Reduction output '{item.target}' is the same file as a raw input; choose a separate Save Path.")
    if run_plan is not None:
        # Directory JIT: this group's dependencies are in hand, so refuse to
        # read ANY output target of the run as raw data (a raw container that
        # links into the Save Path).  The item's own target is covered below.
        for dependency in raw:
            if run_plan.owns_target(dependency):
                raise OutputCollisionError(
                    f"Reduction output '{dependency}' is the same file as a raw "
                    f"directory input that '{items[0].target.name}' depends on; "
                    "choose a separate Save Path."
                )
    protected = tuple(
        Path(value) for value in (
            configuration.poni_file, configuration.mask_file,
            *_correction_paths(configuration),
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
def _load_scientific_assets(intent, *, cancelled=_not_cancelled):
    # Preserve this public preflight seam, including traced calibration loaders.
    from xrd_tools.session.scientific_assets import load_scientific_assets
    return load_scientific_assets(intent, cancelled=cancelled, calibration_loader=load_poni)
from xrd_tools.session.scientific_assets import _scientific_mask_layout
from xrd_tools.session.scientific_assets import _scientific_mask_file_limit
from xrd_tools.session.scientific_assets import _asset_state
from xrd_tools.session.scientific_assets import _asset_descriptor_state
from xrd_tools.session.scientific_assets import _stream_asset
from xrd_tools.session.scientific_assets import _load_stable_mask_asset
from xrd_tools.session.scientific_assets import _detector_facts


from xrd_tools.session.scientific_assets import _read_correction_image


from xrd_tools.session.scientific_assets import inspect_detector_correction


from xrd_tools.session.scientific_assets import _load_stable_correction_asset


from xrd_tools.session.scientific_assets import _load_stable_asset
def _lexical_key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


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
    "execution_plan_values", "native_int_reduction_plan", "inspect_output",
    "materialize_deferred_output",
    "live_directory_groups", "materialize_live_directory_group", "observe_live_members",
    "OutputFact", "prepare_output", "source_snapshots",
    "target_state_matches", "validate_admitted_receipt",
    "validate_planned_source", "validate_source_aliases",
]
