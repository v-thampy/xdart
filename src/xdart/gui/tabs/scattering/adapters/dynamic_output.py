from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Any

from xdart.modules.reduction import open_headless_scan_session
from xrd_tools.io import (
    AppendDisposition,
    AppendExternalMember,
    AppendImageMember,
    AppendIntent,
    AppendPreflightCleanupError,
    AppendPreflightState,
    AppendRefused,
    AppendSource,
    begin_same_run_lineage,
    extend_same_run_lineage,
    prepare_append_preflight,
    qualify_append,
    science_fingerprint,
)
from xrd_tools.io.append import seal_append_epoch
from xrd_tools.reduction import (
    CompositeSink,
    NexusSink,
    TransactionalXYESink,
)
from xrd_tools.session import (
    DynamicAccountingLimits,
    DynamicFrameIdentity,
    DynamicRunAccounting,
    StageLedger,
    required_result_modes,
)
from xrd_tools.session.display_logic import xye_prefix_for_unit

from ..contracts import AdmittedOutput, PlannedOutput, SourceExecutionStamp
from ..output_preflight import source_snapshots


_MISSING = object()
_SUPPORTED_LINEAGE_FRAME_CEILING = 1_000_000


def _supported_lineage_frame_count(stamp: SourceExecutionStamp) -> int:
    count = int(stamp.frame_count)
    if count > _SUPPORTED_LINEAGE_FRAME_CEILING:
        raise ValueError(
            "dynamic output lineage exceeds the supported "
            f"{_SUPPORTED_LINEAGE_FRAME_CEILING}-frame ceiling"
        )
    return count


def _target_key(path: Path | str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _stable_lineage(item: PlannedOutput) -> tuple[object, ...]:
    identity = item.source_stamp.execution_identity_v1
    aliases = tuple(
        (
            value.raw_path,
            value.resolved_path,
            value.target_id,
            value.roles,
            value.candidate_owner_id,
        )
        for value in identity.aliases
        if "source_file" in value.roles
    )
    primary_targets = {value.target_id for value in identity.aliases
                       if "source_file" in value.roles}
    targets = tuple(
        (
            value.target_id,
            value.resolved_path,
            value.roles,
        )
        for value in identity.targets
        if value.target_id in primary_targets
    )
    return (
        item.group.adapter_key,
        item.group.group_key,
        _target_key(item.group.target),
        aliases,
        targets,
    )


def _append_source(
    stamp: SourceExecutionStamp,
    item: PlannedOutput,
    *,
    generation: int,
) -> AppendSource:
    image_members = ()
    if stamp.members:
        if len(stamp.members) != stamp.frame_count:
            raise ValueError(
                "flat-series Append requires one exact member per frame"
            )
        image_members = tuple(
            AppendImageMember(
                path=value.path,
                size=value.size,
                mtime_ns=value.mtime_ns,
                source_start=ordinal,
                source_stop=ordinal + 1,
                ordinal=ordinal,
            )
            for ordinal, value in enumerate(stamp.members)
        )
    external_members = tuple(
        AppendExternalMember(
            path=value.file.path,
            dataset_path=value.dataset,
            size=value.file.size,
            mtime_ns=value.file.mtime_ns,
            source_start=value.first,
            source_stop=value.stop,
            ordinal=value.epoch,
        )
        for value in stamp.external_members
    )
    descriptor = item.descriptor
    dataset_paths = ()
    if descriptor is not None and not external_members:
        dataset_paths = tuple(dict.fromkeys(
            value
            for value in (
                descriptor.dataset_path,
                *descriptor.segment_paths,
            )
            if value
        ))
    if (
        descriptor is not None
        and descriptor.kind.value == "eiger_master"
        and not external_members
    ):
        raise ValueError(
            "Eiger Append requires exact external dataset/range facts"
        )
    return AppendSource(
        path=stamp.path,
        adapter_id=stamp.adapter_id,
        size=stamp.size,
        mtime_ns=stamp.mtime_ns,
        extent=stamp.frame_count,
        digest=science_fingerprint(stamp.execution_identity_v1.as_dict()),
        dataset_paths=dataset_paths,
        image_members=image_members,
        external_members=external_members,
        generation=int(generation),
    )


def _mode_tokens(configuration) -> tuple[str, ...]:
    one_key = (
        configuration.gi.mode_1d
        if configuration.gi.enabled else "default"
    )
    two_key = (
        configuration.gi.mode_2d
        if configuration.gi.enabled else "default"
    )
    modes = [f"1d:{one_key}"]
    if not configuration.skip_2d:
        modes.append(f"2d:{two_key}")
    return tuple(modes)


def _science_projection(
    configuration,
    signed: object,
) -> dict[str, Any]:
    if type(signed) is not dict:
        raise TypeError("Append requires the exact signed science mapping")
    values = dict(signed)
    assets = values.pop("accepted_scientific_assets", None)
    if values != configuration.as_provenance():
        raise ValueError(
            "signed science mapping differs from the frozen run configuration"
        )
    if type(assets) is not dict or set(assets) != {
        "poni_values", "poni_sha256", "mask_sha256",
    }:
        raise TypeError("accepted scientific asset identity is malformed")
    if assets["poni_values"] is not None and type(
        assets["poni_values"]
    ) is not dict:
        raise TypeError("accepted PONI values are malformed")
    for key in ("poni_sha256", "mask_sha256"):
        digest = assets[key]
        if digest is not None and (
            type(digest) is not str
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
        ):
            raise TypeError(f"accepted {key} is malformed")
    required = {
        "processing_mode", "bai_1d_args", "bai_2d_args", "gi",
        "threshold", "poni_values",
    }
    if not required.issubset(values):
        raise TypeError("signed science mapping is incomplete")
    if (
        type(values["processing_mode"]) is not str
        or type(values["bai_1d_args"]) is not dict
        or type(values["bai_2d_args"]) is not dict
        or type(values["gi"]) is not dict
        or type(values["threshold"]) is not dict
        or values["poni_values"] is not None
        and type(values["poni_values"]) is not dict
    ):
        raise TypeError("signed numerical science mapping is malformed")
    return {
        "processing_mode": values["processing_mode"],
        "bai_1d_args": values["bai_1d_args"],
        "bai_2d_args": values["bai_2d_args"],
        "gi": values["gi"],
        "threshold": values["threshold"],
        "poni_values": values["poni_values"],
        "accepted_scientific_assets": assets,
    }


def _append_intent(
    configuration,
    item: PlannedOutput,
    *,
    generation: int,
    science_identity: str,
) -> AppendIntent:
    frame_count = _supported_lineage_frame_count(item.source_stamp)
    return AppendIntent(
        entry=item.source_spec.entry or "entry",
        source_base=configuration.project_root or "",
        source_identity=science_fingerprint(_stable_lineage(item)),
        science_fingerprint=science_identity,
        modes=_mode_tokens(configuration),
        source=_append_source(
            item.source_stamp, item, generation=generation,
        ),
        labels=tuple(range(
            item.source_stamp.first_label,
            item.source_stamp.first_label + frame_count,
        )),
    )


def preview_append_decision(
    configuration,
    item: PlannedOutput,
    signed_science: dict[str, Any],
):
    science_identity = science_fingerprint(
        _science_projection(configuration, signed_science)
    )
    return qualify_append(
        item.target,
        _append_intent(
            configuration,
            item,
            generation=1,
            science_identity=science_identity,
        ),
    )


def _writer_source_snapshots(
    item: PlannedOutput,
) -> dict[str, dict[str, Any]]:
    allowed = {
        "adapter_id", "size", "mtime_ns", "frame_count",
        "dataset_path", "self_contained",
    }
    return {
        path: {
            key: value
            for key, value in snapshot.items()
            if key in allowed and value is not None
        }
        for path, snapshot in source_snapshots(item).items()
    }


class DynamicOutputAdapter:
    def __init__(self, configuration) -> None:
        self.configuration = configuration
        self._graphs: dict[str, dict[str, Any]] = {}
        self._pending_preflights: list[Any] = []
        self._pending_nexus: list[NexusSink] = []
        self._pending_xye: list[TransactionalXYESink] = []
        self._science_identity: str | None = None
        self._current: dict[str, Any] | None = None
        self._command_lock = RLock()
        self._activating = False
        self._stop_requested = False

    @staticmethod
    def _discard_identity(values: list[Any], owner: Any) -> None:
        values[:] = [value for value in values if value is not owner]

    def _release_construction_custody(
        self, *, preflight=None, nexus=None, xye=None,
    ) -> None:
        if preflight is not None:
            self._discard_identity(self._pending_preflights, preflight)
        if nexus is not None:
            self._discard_identity(self._pending_nexus, nexus)
        if xye is not None:
            self._discard_identity(self._pending_xye, xye)

    def _settle_construction_custody(self) -> None:
        primary: BaseException | None = None
        for xye in tuple(self._pending_xye):
            try:
                xye.abort(None)
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(self._pending_xye, xye)
        for nexus in tuple(self._pending_nexus):
            try:
                nexus.abort(None)
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(self._pending_nexus, nexus)
                preflight = nexus.append_preflight
                if preflight is not None and preflight.snapshot.state in {
                    AppendPreflightState.NOOP,
                    AppendPreflightState.COMMITTED,
                    AppendPreflightState.ABORTED,
                }:
                    self._discard_identity(
                        self._pending_preflights, preflight,
                    )
        bound = {
            id(nexus.append_preflight)
            for nexus in self._pending_nexus
            if nexus.append_preflight is not None
        }
        for preflight in tuple(self._pending_preflights):
            if id(preflight) in bound:
                continue
            try:
                state = preflight.snapshot.state
                if state is AppendPreflightState.RESERVED:
                    state = preflight.abort().state
                elif state is AppendPreflightState.RETRYABLE:
                    state = preflight.retry_cleanup().state
                if state not in {
                    AppendPreflightState.NOOP,
                    AppendPreflightState.COMMITTED,
                    AppendPreflightState.ABORTED,
                }:
                    raise RuntimeError(
                        "Append construction cleanup remains unsettled: "
                        f"{state.value}"
                    )
            except BaseException as error:
                primary = primary or error
            else:
                self._discard_identity(
                    self._pending_preflights, preflight,
                )
        if primary is not None:
            raise primary

    @property
    def session(self):
        return None if self._current is None else self._current["session"]

    @property
    def sink(self):
        return None if self._current is None else self._current["sink"]

    @property
    def accounting(self):
        return None if self._current is None else self._current["accounting"]

    @property
    def write_labels(self) -> tuple[int, ...]:
        return (
            ()
            if self._current is None
            else tuple(self._current["write_labels"])
        )

    def _revision(self, graph: dict[str, Any], item: PlannedOutput) -> int:
        exact = item.source_stamp.execution_identity_v1
        revisions = graph["revisions"]
        value = revisions.get(exact)
        if value is None:
            value = len(revisions) + 1
            revisions[exact] = value
        return int(value)

    def activate(self, *args, cancelled=lambda: False, **kwargs):
        with self._command_lock:
            if self._activating:
                raise RuntimeError("dynamic output activation is already active")
            if self._stop_requested or cancelled():
                raise RuntimeError("admission cancelled")
            self._activating = True
        value = error = None
        try:
            value = self._activate_owned(
                *args,
                cancelled=lambda: self._stop_requested or cancelled(),
                **kwargs,
            )
            if self._stop_requested or cancelled():
                raise RuntimeError("admission cancelled")
        except BaseException as primary:
            error = primary
            try:
                self._settle_construction_custody()
            except BaseException as cleanup:
                raise primary from cleanup
        finally:
            with self._command_lock:
                self._activating = False
                stop = self._stop_requested or cancelled()
                self._stop_requested = self._stop_requested or stop
        if stop:
            try:
                self.stop()
            except BaseException as cleanup:
                if error is not None:
                    raise error.with_traceback(error.__traceback__) from cleanup
                raise
        if error is not None:
            raise error.with_traceback(error.__traceback__)
        if stop:
            raise RuntimeError("admission cancelled")
        return value

    def _activate_owned(
        self,
        scan,
        plan,
        item: PlannedOutput,
        decision: AdmittedOutput,
        *,
        record_store,
        run_provenance,
        cancelled,
    ):
        if type(run_provenance) is not dict:
            raise TypeError("dynamic output requires exact run provenance")
        science_identity = science_fingerprint(_science_projection(
            self.configuration,
            run_provenance.get("scientific_signature"),
        ))
        frame_count = _supported_lineage_frame_count(item.source_stamp)
        if cancelled():
            raise RuntimeError("admission cancelled")
        if self._science_identity is None:
            self._science_identity = science_identity
        elif self._science_identity != science_identity:
            raise ValueError(
                "dynamic output science identity changed within one run"
            )
        target = Path(item.target)
        key = _target_key(target)
        lineage = _stable_lineage(item)
        labels = tuple(range(
            item.source_stamp.first_label,
            item.source_stamp.first_label + frame_count,
        ))
        if tuple(int(frame.index) for frame in scan.frames) != labels:
            raise ValueError("dynamic output requires the complete exact Scan")
        graph = self._graphs.get(key)
        if graph is not None:
            if cancelled():
                raise RuntimeError("admission cancelled")
            if graph["lineage"] != lineage:
                raise ValueError(
                    "same target cannot change its exact output lineage"
                )
            revision = self._revision(graph, item)
            intent = _append_intent(
                self.configuration,
                item,
                generation=revision,
                science_identity=science_identity,
            )
            if graph["intent"] != intent:
                nexus = graph["nexus"]
                if nexus is not None:
                    extension = graph["session"].extend_live(intent)
                    write_labels = tuple(extension.write_labels)
                else:
                    extension = extend_same_run_lineage(
                        graph["xye_lineage"], graph["intent"], intent,
                    )
                    if extension.disposition is AppendDisposition.REFUSE:
                        raise AppendRefused(extension)
                    write_labels = tuple(extension.write_labels)
                    graph["xye_pending"] = extension
                graph["intent"] = intent
            else:
                write_labels = ()
            graph["scan"] = scan
            graph["item"] = item
            graph["write_labels"] = write_labels
            self._arm(graph, write_labels, revision)
            self._current = graph
            if cancelled():
                raise RuntimeError("admission cancelled")
            return graph["session"], False

        revision = 1
        intent = _append_intent(
            self.configuration,
            item,
            generation=revision,
            science_identity=science_identity,
        )
        modes = required_result_modes(plan)
        if tuple(f"{mode.kind}:{mode.key}" for mode in modes) != intent.modes:
            raise ValueError("Append intent modes differ from the reduction plan")
        xye_only = self.configuration.processing_mode == "Int 1D (XYE)"
        if xye_only and self.configuration.output_mode == "Append":
            raise ValueError("XYE-only Append has no persisted lineage owner")

        preflight = None
        if not xye_only and self.configuration.output_mode == "Append":
            if cancelled():
                raise RuntimeError("admission cancelled")
            try:
                preflight = prepare_append_preflight(target, intent)
            except AppendPreflightCleanupError as error:
                self._pending_preflights.append(error.owner)
                raise
            self._pending_preflights.append(preflight)
            if preflight.snapshot.disposition is AppendDisposition.SKIP:
                preflight.complete_noop()
                self._release_construction_custody(
                    preflight=preflight,
                )
                self._current = {
                    "session": None,
                    "sink": None,
                    "accounting": None,
                    "write_labels": (),
                }
                return None, False

        xye = None
        nexus = None
        try:
            if cancelled():
                raise RuntimeError("admission cancelled")
            integration_1d = getattr(plan, "integration_1d", None)
            if integration_1d is not None:
                xye = TransactionalXYESink(
                    target.parent / str(scan.name),
                    prefix=xye_prefix_for_unit(integration_1d.unit),
                )
                self._pending_xye.append(xye)

            if not xye_only:
                sink_values = dict(
                    overwrite=self.configuration.output_mode == "Overwrite",
                    flush_every=None,
                    atomic=False,
                    incremental_finalization=True,
                    source_base=self.configuration.project_root or None,
                    run_configuration_provenance=run_provenance,
                    source_execution_provenance=item.source_stamp.as_dict(),
                    source_snapshots_provenance=(
                        _writer_source_snapshots(item)
                    ),
                )
                if self.configuration.output_mode == "Append":
                    nexus = NexusSink(
                        target, append_preflight=preflight, **sink_values,
                    )
                else:
                    nexus = NexusSink(
                        target, same_run_intent=intent, **sink_values,
                    )
                self._pending_nexus.append(nexus)
            if nexus is None:
                sink = xye
            elif xye is None:
                sink = nexus
            else:
                sink = CompositeSink((nexus, xye))
            if sink is None:
                raise ValueError("dynamic output graph has no applicable sink")

            targets_by_mode = {}
            for mode in modes:
                targets = []
                if nexus is not None:
                    targets.append(f"nexus:{nexus.path}")
                if xye is not None and mode.kind == "1d":
                    targets.append(xye.canonical_target)
                targets_by_mode[mode] = tuple(targets)
            ledger = StageLedger(
                required_modes=modes,
                targets_by_mode=targets_by_mode,
            )
            accounting = DynamicRunAccounting(
                ledger,
                run_generation=int(self.configuration.generation),
                limits=DynamicAccountingLimits(
                    1, 32, _SUPPORTED_LINEAGE_FRAME_CEILING,
                ),
            )
            xye_lineage = (
                begin_same_run_lineage(intent) if xye_only else None
            )
            if (
                xye_lineage is not None
                and xye_lineage.disposition is not AppendDisposition.WRITE
            ):
                raise AppendRefused(xye_lineage)
            session = open_headless_scan_session(
                scan,
                plan,
                sink=sink,
                executor=self.configuration.max_cores,
                inflight_max=max(1, self.configuration.max_cores),
                gi_freeze_mode=(
                    "scout_union" if self.configuration.gi.enabled else None
                ),
                record_store=record_store,
                record_store_persisted_on_write=False,
                nexus_target=(
                    None if nexus is None else f"nexus:{nexus.path}"
                ),
                xye_target=(
                    None if xye is None else xye.canonical_target
                ),
                accounting=accounting,
                xye_receipt_boundary=xye,
            )
            if nexus is not None and nexus.append_preflight is not None:
                write_labels = tuple(
                    nexus.append_preflight.snapshot.write_labels
                )
            else:
                write_labels = labels
            graph = {
                "target": key,
                "lineage": lineage,
                "scan": scan,
                "item": item,
                "intent": intent,
                "session": session,
                "sink": sink,
                "nexus": nexus,
                "xye": xye,
                "accounting": accounting,
                "record_store": record_store,
                "revisions": {
                    item.source_stamp.execution_identity_v1: revision
                },
                "keys": {},
                "source_revisions": {},
                "armed_labels": frozenset(),
                "write_labels": write_labels,
                "settled_labels": set(),
                "xye_lineage": xye_lineage,
                "xye_pending": None,
                "transition": None,
                "stop_requested": False,
                "projection_pending": False,
            }
            self._graphs[key] = graph
            self._current = graph
            self._release_construction_custody(
                preflight=preflight, nexus=nexus, xye=xye,
            )
            self._arm(graph, write_labels, revision)
            if cancelled():
                raise RuntimeError("admission cancelled")
            return session, True
        except BaseException:
            raise

    def _arm(
        self,
        graph: dict[str, Any],
        labels: tuple[int, ...],
        source_revision: int,
    ) -> None:
        accounting = graph["accounting"]
        frames = {
            int(frame.index): frame for frame in graph["scan"].frames
        }
        armed_labels = []
        for label in labels:
            if label not in frames:
                raise ValueError(f"dynamic output label {label} has no frame")
            key = graph["keys"].get(label)
            if key is None:
                key = DynamicFrameIdentity(
                    graph["lineage"],
                    (graph["target"], int(label)),
                )
                accounting.discover(
                    key,
                    group=graph["lineage"],
                    ordinal=len(graph["keys"]),
                    output_label=int(label),
                )
                graph["keys"][label] = key
            label = int(label)
            graph["source_revisions"][label] = int(source_revision)
            armed_labels.append(label)
        graph["armed_labels"] = frozenset(armed_labels)

    def submit(self, frame, image: Any = _MISSING) -> bool:
        with self._command_lock:
            graph = self._current
            if graph is None:
                raise RuntimeError("dynamic output graph is not active")
            if self._stop_requested or graph["stop_requested"]:
                return False
            label = int(frame.index)
            if label not in graph["armed_labels"]:
                return True
            token = graph["accounting"].begin_attempt(
                graph["keys"][label],
                source_revision=graph["source_revisions"][label],
            )
            graph["accounting"].record_enqueued(token)
            session = graph["session"]
        if image is _MISSING:
            return bool(session.submit(frame, attempt_token=token))
        return bool(session.submit(frame, image, attempt_token=token))

    def _epoch_transition(self, graph: dict[str, Any]):
        with self._command_lock:
            if graph["transition"] not in {None, "epoch"}:
                raise RuntimeError("dynamic output has a pending finish")
            if graph["stop_requested"] and graph["transition"] is None:
                raise RuntimeError("admission cancelled")
            graph["transition"] = "epoch"
        value = graph["session"].commit_epoch()
        pending = graph.get("xye_pending") or graph.get("xye_lineage")
        if graph["nexus"] is None and pending is not None:
            graph["xye_lineage"] = seal_append_epoch(pending)
            graph["xye_pending"] = None
        with self._command_lock:
            graph["transition"] = None
            graph["projection_pending"] = True
            stop = graph["stop_requested"]
        if stop and graph["session"].is_running:
            graph["session"].stop()
        return value, stop

    def commit_epoch(self):
        graph = self._current
        if graph is None:
            raise RuntimeError("dynamic output graph is not active")
        return self._epoch_transition(graph)[0]

    def _finish_transition(self, graph: dict[str, Any], *, stopped=False,
                           stop_already_applied=False):
        with self._command_lock:
            if graph["transition"] not in {None, "finish"}:
                raise RuntimeError("dynamic output has a pending epoch")
            retrying = graph["transition"] == "finish"
            graph["transition"] = "finish"
            graph["stop_requested"] = graph["stop_requested"] or stopped
            stop = graph["stop_requested"]
        if (
            stop and not retrying and not stop_already_applied
            and graph["session"].is_running
        ):
            graph["session"].stop()
        result = graph["session"].finish(raise_on_failure=False)
        with self._command_lock:
            graph["transition"] = None
            graph["projection_pending"] = True
        return result

    def finish_current(self, *, stopped: bool = False):
        graph = self._current
        return None if graph is None else self._finish_transition(graph, stopped=stopped)

    def finish_all(self, *, stopped: bool = False) -> tuple[object, ...]:
        results = []
        primary = None
        try:
            self._settle_construction_custody()
        except BaseException as error:
            primary = error
        for graph in tuple(self._graphs.values()):
            try:
                with self._command_lock:
                    graph["stop_requested"] = (
                        graph["stop_requested"] or stopped
                    )
                    transition = graph["transition"]
                if transition == "epoch":
                    try:
                        _, stop_applied = self._epoch_transition(graph)
                    except BaseException as retry:
                        try:
                            result = graph["session"].finish(
                                raise_on_failure=False,
                            )
                        except BaseException as fallback:
                            raise fallback from retry
                        with self._command_lock:
                            graph["transition"] = None
                            graph["projection_pending"] = True
                        results.append(result)
                        continue
                results.append(self._finish_transition(
                    graph, stop_already_applied=(
                        transition == "epoch" and stop_applied
                    ),
                ))
            except BaseException as error:
                if primary is None:
                    primary = error
        if primary is not None:
            raise primary
        return tuple(results)

    def stop(self) -> None:
        with self._command_lock:
            self._stop_requested = True
            for graph in tuple(self._graphs.values()):
                graph["stop_requested"] = True

    def abort(self, _result=None) -> None:
        self.stop()
        self.finish_all(stopped=True)

    def project_new_durable(self, apply) -> None:
        for graph in tuple(self._graphs.values()):
            current = tuple(self._durable_labels_for(graph))
            with self._command_lock:
                prior = graph["settled_labels"]
                delta = tuple(sorted(set(current) - prior))
                pending = graph["projection_pending"]
            if not pending and not delta:
                continue
            apply(str(graph["item"].target), current, delta)
            with self._command_lock:
                graph["projection_pending"] = False
                prior.update(delta)

    @staticmethod
    def _durable_labels_for(graph: dict[str, Any]) -> tuple[int, ...]:
        accounting = graph["accounting"]
        snapshot = accounting.snapshot()
        required = accounting.ledger.targets_by_mode
        return tuple(sorted(
            int(key.logical_frame_identity[1])
            for key in snapshot.discovered
            if all(
                (key, mode, target) in snapshot.durable
                for mode, targets in required.items()
                for target in targets
            )
        ))

__all__ = ["DynamicOutputAdapter"]
