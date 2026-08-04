from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
import numpy as np
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration import load_poni
from xrd_tools.io import load_mask
from xrd_tools.io.output_path import OVERWRITE_MODE, resolve_output_target
from xrd_tools.io.output_safety import check_output_not_source
from xrd_tools.session.intent_store import RunIntentSnapshot
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent
from xrd_tools.sources.adapters import candidate_owner, get_adapter
from xrd_tools.sources.descriptor import ContainerDescriptor
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
from . import output_values
from .source_metadata import (
    ordered_motor_intersection,
    read_image_motor_metadata,
)
@dataclass(frozen=True, slots=True)
class OutputCandidate:
    source: SourceSpec | DirectorySourceSpec
    poni_file: str
    mask_file: str
    save_path: str
    processing_json: str
    fingerprint: str
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
        if str(intent.output_mode).strip().lower() != "overwrite":
            raise ValueError(output_values.APPEND_UNAVAILABLE)
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
        )
    def processing_mapping(self) -> dict[str, Any]:
        return json.loads(self.processing_json)
    def matches(self, configuration: FrozenRunConfiguration) -> bool:
        return (
            configuration.thaw_source_spec() == self.source
            and configuration.output_mode == "Overwrite"
            and configuration.save_path == self.save_path
            and configuration.fingerprint == self.fingerprint
        )


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
    assets = _load_scientific_assets(snapshot.thaw())
    candidate = OutputCandidate.from_start_capture(capture, assets)
    source, choices = candidate.source, None
    if cancelled():
        raise RuntimeError("admission cancelled")
    if type(source) is DirectorySourceSpec:
        session = DirectoryIndexSession(probe_candidates=False)
        session_owner(session)
        session.configure(
            source.root, recursive=source.recursive,
            name_filter=source.name_filter, suffixes=source.suffixes,
        )
        observation = session.observe(refresh=True)
        session.enable_probes(exclude=())
        while observation.unprobed_count:
            if cancelled():
                raise RuntimeError("admission cancelled")
            observation = session.observe(refresh=False)
        plan = RunCandidatePlan.from_observation(observation)
        items = _directory_items(candidate, plan, cancelled=cancelled)
        _validate_exact_tiff_gi_motor(snapshot.thaw(), items)
        choices = _motor_choices(items)
        candidate = OutputCandidate.from_start_capture(capture, assets, choices)
    else:
        items = (_series_item(candidate, source, cancelled=cancelled),)
        _validate_exact_tiff_gi_motor(snapshot.thaw(), items)
        if source.kind is SourceKind.TIFF_SERIES:
            choices = _motor_choices(items)
            candidate = OutputCandidate.from_start_capture(
                capture, assets, choices
            )
    _validate_targets(candidate, source, items)
    if targets_owner is not None:
        targets_owner(tuple(dict.fromkeys(item.target for item in items)))
    outputs = tuple(inspect_output(item, candidate) for item in items)
    if cancelled() or not outputs:
        raise RuntimeError(
            "admission cancelled" if cancelled()
            else "source has no READY candidates"
        )
    return AdmissionReceipt(
        capture.request_id, snapshot.revision, capture.source_capture,
        candidate, outputs, assets, choices,
    )
def inspect_output(
    item: PlannedOutput,
    configuration: FrozenRunConfiguration | OutputCandidate,
    fact: OutputFact | None = None,
) -> AdmittedOutput:
    accepted = fact if type(fact) is OutputFact else OutputFact(_path_state(item.target))
    start, count = item.source_stamp.first_label, item.source_stamp.frame_count
    return AdmittedOutput(
        item, OutputDisposition.WRITE, tuple(range(start, start + count)), accepted
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
        state = SourceFileState.capture(path)
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
            raise ValueError(
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
            raise ValueError(
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
        raise ValueError(f"source candidate changed during admission: {path}")
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
    stamp = SourceExecutionStamp(
        state,
        owner.id,
        descriptor.frame_count,
        0,
        external_members=_external_members(
            path,
            descriptor,
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


def _directory_items(
    configuration: FrozenRunConfiguration | OutputCandidate,
    plan: RunCandidatePlan,
    *,
    cancelled: Callable[[], bool] = _not_cancelled,
) -> tuple[PlannedOutput, ...]:
    root, items, consumed = Path(configuration.save_path), [], set()
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
            if descriptor.kind is SourceKind.PROCESSED_NEXUS:
                raise ValueError("processed output cannot be raw input")
            if descriptor.frame_count < 1:
                continue
            name = descriptor.scan_name or candidate.path.stem.removesuffix("_master")
            spec = SourceSpec(
                candidate.path, descriptor.kind,
                entry=descriptor.resolved_entry or descriptor.requested_entry,
            )
            state = _capture_source_states((candidate.path,), cancelled)[0]
            external_members = _external_members(
                candidate.path,
                descriptor,
                cancelled=cancelled,
            )
            if cancelled():
                raise RuntimeError("admission cancelled")
            stamp = SourceExecutionStamp(
                state,
                candidate.adapter_id, descriptor.frame_count, 0,
                external_members=external_members,
            )
        items.append(PlannedOutput(
            spec,
            candidate.path,
            _resolved_generated_target(configuration.save_path, name),
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
    owners = {item.source_stamp.adapter_id}
    if "tiff_series" in owners:
        owners.add("image_file")
    states = item.source_stamp.members or (item.source_stamp.file,)
    for state in states:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        path = Path(state.path)
        owner = candidate_owner(path)
        if (
            not state.matches_disk()
            or owner is None or owner.id not in owners
        ):
            raise ValueError(f"source candidate changed before open: {path}")
    for metadata in item.source_stamp.metadata_sources:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        state = metadata.metadata_file
        if state is not None and not state.matches_disk():
            raise ValueError(
                "TIFF metadata source changed before open: "
                f"{state.path}"
            )
    import h5py
    for external in item.source_stamp.external_members:
        if is_cancelled():
            raise RuntimeError("admission cancelled")
        path = Path(external.file.path)
        if not external.file.matches_disk():
            raise ValueError(
                f"external source member changed before open: {path}"
            )
        with h5py.File(path, "r") as handle:
            if external.dataset not in handle:
                raise ValueError(
                    f"external source dataset changed: {path}:{external.dataset}"
                )
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
            if state is None:
                continue
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
        values[external.file.path] = {
            "adapter_id": stamp.adapter_id,
            **external.file.as_dict(),
            "frame_count": external.stop - external.first,
            "dataset_path": external.dataset,
            "self_contained": True,
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
) -> None:
    raw = tuple(
        Path(member.path) if item.source_stamp.members else item.source_path
        for item in items
        for member in item.source_stamp.members or (item.source_stamp.file,)
    ) + tuple(
        Path(external.file.path)
        for item in items
        for external in item.source_stamp.external_members
    ) + tuple(
        Path(metadata.metadata_file.path)
        for item in items
        for metadata in item.source_stamp.metadata_sources
        if metadata.metadata_file is not None
    )
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
    poni, poni_digest = _load_stable_asset(intent.poni_file, load_poni)
    mask, mask_digest = _load_stable_asset(
        intent.mask_file,
        lambda path: np.load(path) if path.suffix == ".npy" else load_mask(path),
    )
    poni_values = None if poni is None else (
        float(poni.dist), float(poni.poni1), float(poni.poni2), float(poni.rot1),
        float(poni.rot2), float(poni.rot3), float(poni.wavelength),
        str(poni.detector),
    )
    accepted = None if mask is None else np.ascontiguousarray(mask)
    return AcceptedScientificAssets(
        poni_values, None if accepted is None else accepted.dtype.str,
        None if accepted is None else tuple(accepted.shape),
        None if accepted is None else accepted.tobytes(),
        poni_digest, mask_digest,
    )
def _load_stable_asset(
    path_text: object, loader: Callable[[Path], object],
) -> tuple[object | None, str | None]:
    if type(path_text) is not str or not path_text:
        return None, None
    path = Path(path_text)
    try:
        before = path.read_bytes()
    except FileNotFoundError:
        return None, None
    value = loader(path)
    if path.read_bytes() != before:
        raise ValueError(f"scientific asset changed while admitted: {path}")
    return value, hashlib.sha256(before).hexdigest()
def _external_members(
    master: Path, descriptor: ContainerDescriptor,
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
    values, first = [], 0
    with h5py.File(master, "r") as handle:
        for epoch, segment in enumerate(segments):
            if cancelled():
                raise RuntimeError("admission cancelled")
            link = handle.get(segment, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                raise ValueError("external container lost its exact link")
            path = (master.parent / link.filename).resolve(strict=True)
            with h5py.File(path, "r") as member:
                stop = first + int(member[link.path].shape[0])
            values.append(
                ExternalSourceState(
                    SourceFileState.capture(path),
                    link.path,
                    first,
                    stop,
                    epoch,
                )
            )
            first = stop
            if cancelled():
                raise RuntimeError("admission cancelled")
    return tuple(values)
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
    "OutputCandidate", "execution_plan_values",
    "inspect_output", "OutputFact", "prepare_output", "source_snapshots",
    "target_state_matches", "validate_admitted_receipt",
    "validate_planned_source",
]
