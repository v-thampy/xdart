"""One finite worker slot for later externally owned operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
import json
import math
import os, stat; from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable

from ..events import CleanupStatus, detached_exception_strings
from ..experiment_authoring import (
    AssetValidationRequest, AssetValidationResult, CalibrationRequest,
    MaskRequest, mask_terminal_result_valid, resolve_mask_executable,
    run_calibration, run_mask, validate_authored_asset,
)
from ..operation_values import (
    OperationCleanupReceipt, OperationContextStamp, OperationIdentity,
    OperationPending, OperationProgress, OperationTerminal, OperationTerminalStatus,
    OperationUpdate,
)
from ..presentation_background import PresentationBackgroundOwner
from xrd_tools.reduction.background import DisplayBackgroundPlan
from xrd_tools.io.bounded_json import BoundedJsonError, bounded_json_snapshot
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    TargetSnapshot,
    stream_terminal_object_revision,
)
from xrd_tools.reduction import ReintegratePlan, ReintegrateProgress, ReintegrateResult, run_reintegrate
from xrd_tools.reduction.reintegrate import ReintegrateCancelled
from xrd_tools.session.run_configuration import FrozenRunConfiguration

def _average_scan_recipe(*args, **kwargs): from xrd_tools.reduction.average import AverageScanRecipe as owner; return owner(*args, **kwargs)
def _average_scan_runner(*args, **kwargs): from xrd_tools.reduction.average import AverageScanRunner as owner; return owner(*args, **kwargs)
AverageScanRecipe = _average_scan_recipe; AverageScanRunner = _average_scan_runner
def detector_calibration_to_integrator(*args, **kwargs): from xrd_tools.integrate.calibration import detector_calibration_to_integrator as owner; return owner(*args, **kwargs)
class _BackgroundPreterminalAbort(RuntimeError): pass
@dataclass(frozen=True, slots=True)
class _ReintegrateRequest:
    target: str
    entry: str
    source_root: str
    expected_target_snapshot: TargetSnapshot
    expected_terminal_identity: StreamTerminal | None
    expected_labels: tuple[int, ...]
    dimension: str
    preparation_json: str
@dataclass(frozen=True, slots=True)
class _ReintegrateSuccessorRequest:
    source_artifact: str
    entry: str
    source_root: str | None
    expected_target_snapshot: TargetSnapshot
    expected_terminal_identity: StreamTerminal | None
    expected_labels: tuple[int, ...]
    dimension: str
    preparation_json: str
    prepared_offer: object
@dataclass(frozen=True, slots=True)
class _AverageRequest:
    configuration: FrozenRunConfiguration
    target: str
    entry: str
    numeric_metadata_keys: tuple[str, ...] | None
    invariant_metadata_keys: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class _ScanPlotRequest:
    plan: object
    table: object
    roi_result: object | None
def _average_calibration(configuration: FrozenRunConfiguration, cancelled=None):
    from .. import output_preflight
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, MaskState, PoniValues
    assets = output_preflight._load_scientific_assets(
        configuration,
        cancelled=(lambda: False) if cancelled is None else cancelled.is_set,
    )
    if configuration.poni_file and assets.poni_values is None: raise ValueError("AVERAGE_CALIBRATION_UNAVAILABLE")
    if configuration.mask_file and assets.mask_bytes is None: raise ValueError("AVERAGE_MASK_UNAVAILABLE")
    config = None
    if assets.poni_values is not None:
        text = assets.poni_detector_config_json
        try:
            if type(text) is not str or len(text.encode("utf-8")) > 65_536: raise ValueError
            config = json.loads(text)
            if (type(config) is not dict
                    or json.dumps(config, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False) != text
                    or type(config.get("orientation")) is not int
                    or config["orientation"] not in range(1, 5)): raise ValueError
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE") from error
    values = None
    if assets.poni_values is not None:
        try:
            values = PoniValues(*assets.poni_values[:7])
        except (TypeError, ValueError, OverflowError) as error: raise ValueError("AVERAGE_CALIBRATION_UNREPRESENTABLE") from error
    mask_state = MaskState.absent()
    if assets.mask_bytes is not None:
        try:
            import numpy as np
            dtype = np.dtype(assets.mask_dtype)
            shape = assets.mask_shape
            if (type(shape) is not tuple or not shape
                    or any(type(item) is not int or item <= 0 for item in shape)
                    or math.prod(shape) * dtype.itemsize != len(assets.mask_bytes)): raise ValueError
            np.frombuffer(assets.mask_bytes, dtype=dtype).reshape(shape)
            mask_state = MaskState(configuration.mask_file, assets.mask_sha256 or "", dtype.str, shape, FactStatus.PRESENT)
        except (TypeError, ValueError, OverflowError) as error: raise ValueError("AVERAGE_MASK_UNREPRESENTABLE") from error
    state = (CalibrationState(mask=mask_state) if values is None else
        CalibrationState(values, assets.poni_values[7], config, "",
            assets.poni_sha256 or "", configuration.poni_file, mask_state,
            FactStatus.PRESENT, assets.poni_parallax))
    if values is None: return state
    try:
        from xrd_tools.core import PONI
        from xrd_tools.core.geometry import DetectorCalibration
        accepted = assets.detector_calibration
        reconstructed = DetectorCalibration(PONI(values.dist, values.poni1, values.poni2,
            values.rot1, values.rot2, values.rot3, values.wavelength_m, state.detector_id),
            dict(state.detector_config), parallax=state.parallax)
        def truth(calibration):
            detector = detector_calibration_to_integrator(calibration).detector
            reported = detector.get_config(); shape = tuple(detector.shape); maximum = tuple(detector.max_shape)
            normalized = DetectorCalibration(
                calibration.poni, reported, parallax=calibration.parallax,
            ).to_json()
            return (type(detector), shape, maximum, float(detector.pixel1),
                float(detector.pixel2), int(detector.orientation), normalized)
        accepted_truth = truth(accepted); reconstructed_truth = truth(reconstructed)
        expected_max_shape = tuple(
            config["max_shape"]
            if "max_shape" in config
            else accepted_truth[2]
        )
        valid = (accepted_truth == reconstructed_truth
            and accepted_truth[0].__module__.startswith("pyFAI.detectors")
            and accepted_truth[0].__name__.casefold() == state.detector_id.casefold()
            and len(accepted_truth[1]) == len(expected_max_shape) == 2
            and all(type(item) is int and item > 0
                    for item in (*accepted_truth[1], *expected_max_shape))
            and all(current <= maximum for current, maximum in
                    zip(accepted_truth[1], expected_max_shape, strict=True))
            and accepted_truth[2] == expected_max_shape
            and accepted_truth[3] > 0 and accepted_truth[4] > 0
            and all(math.isfinite(value) for value in accepted_truth[3:5])
            and accepted_truth[5] == config["orientation"])
    except BaseException as error: raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE") from error
    if not valid: raise ValueError("AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE")
    return state
class OperationSlot:
    """Own at most one joinable worker and its detached latest state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._closed = False
        self._next_serial = 1
        self._identity: OperationIdentity | None = None
        self._stamp: OperationContextStamp | None = None
        self._frozen: object | None = None
        self._worker: Thread | None = None
        self._worker_started = False
        self._worker_identity: int | None = None
        self._cancel_event: Event | None = None
        self._progress: OperationProgress | None = None
        self._progress_delivered_revision = 0
        self._pending: OperationPending | None = None
        self._pending_revision = 0
        self._pending_delivered_revision = 0
        self._pending_commanded_revision = 0
        self._deferred_pending_command: str | None = None
        self._command_queue: Queue[str] | None = None
        self._terminal: OperationTerminal | None = None
        self._stale = False
        self._close_cancel_accepted = False
        self._cancel_sealed = False
        self._clean_receipt: OperationCleanupReceipt | None = None
        self._finalize_hook: Callable[[str], None] | None = None; self._finalized = False
        self._terminal_committed = False
        self._abort_fact: tuple[str, str, str, str] | None = None

    @property
    def owned(self) -> bool:
        with self._lock:
            return self._identity is not None

    @property
    def current_identity(self) -> OperationIdentity | None:
        with self._lock:
            return self._identity

    def begin_calibrate(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        if type(request) is not CalibrationRequest:
            return None
        try:
            request.__post_init__()
        except ValueError:
            return None
        return self._begin(request, stamp, self._run_calibrate)

    def _run_calibrate(self, request, identity, cancelled, publish):
        return run_calibration(
            request, identity, cancelled, publish, self._seal_publication
        )

    def begin_mask(
        self, request: object, stamp: OperationContextStamp
    ) -> OperationIdentity | None:
        if type(request) is not MaskRequest:
            return None
        try:
            request.__post_init__(); source = Path(request.source_path)
            state = source.lstat(); valid = (
                source.resolve(strict=True) == source
                and stat.S_ISREG(state.st_mode) and os.access(source, os.R_OK)
                and not os.path.lexists(request.final_path)
                and Path(request.executable).stem.casefold() == "pyfai-drawmask" and resolve_mask_executable(request.executable) == request.executable
            )
        except (OSError, ValueError):
            return None
        return self._begin(request, stamp, self._run_mask) if valid else None

    def _run_mask(self, request, identity, cancelled, publish):
        terminal = run_mask(
            request, identity, cancelled, publish, self._seal_publication,
        )
        if not mask_terminal_result_valid(terminal, request):
            raise ValueError("mask operation returned an invalid terminal")
        return terminal

    def begin_asset_validation(
        self, request: object, stamp: OperationContextStamp,
    ) -> OperationIdentity | None:
        if type(request) is not AssetValidationRequest:
            return None
        try:
            request.__post_init__()
        except ValueError:
            return None
        return self._begin(request, stamp, self._run_asset_validation)

    @staticmethod
    def _run_asset_validation(request, identity, cancelled, publish):
        if cancelled.is_set():
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
        publish("validate", 0, 1)
        result = validate_authored_asset(request)
        if type(result) is not AssetValidationResult:
            raise TypeError("asset validator returned an invalid result")
        result.__post_init__()
        if result.request is not request:
            raise ValueError("asset validator returned an inexact request")
        if cancelled.is_set():
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
        publish("validate", 1, 1)
        return OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        )

    def begin_background(self, plan: object, stamp: OperationContextStamp,
                         owner: object, reservation: object) -> OperationIdentity | None:
        if (type(plan) is not DisplayBackgroundPlan
                or type(owner) is not PresentationBackgroundOwner
                or type(reservation) is not int or reservation < 1):
            return None
        def body(_plan, identity, cancelled, publish):
            publish("aggregate", 0, len(plan.contributor_ids))
            try:
                receipt = owner.run_and_stage(reservation, cancelled.is_set)
            except InterruptedError:
                return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
            publish("aggregate", len(plan.contributor_ids), len(plan.contributor_ids))
            try:
                return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                         payload=receipt)
            except BaseException as error:
                raise _BackgroundPreterminalAbort(str(error)) from error
        return self._begin(plan, stamp, body,
            finalize=lambda outcome: owner.finalize(reservation, outcome))

    def begin_reintegrate(self, *, target: str, entry: str, source_root: str,
                          expected_target_snapshot: TargetSnapshot,
                          expected_labels: tuple[int, ...], dimension: str,
                          preparation_values: Mapping[str, object],
                          stamp: OperationContextStamp,
                          expected_terminal_identity: StreamTerminal | None = None,
                          ) -> OperationIdentity | None:
        valid = (type(target) is str and bool(target) and type(entry) is str and bool(entry)
                 and type(source_root) is str and bool(source_root)
                 and os.path.isabs(source_root)
                 and os.path.normcase(os.path.normpath(source_root)) == source_root
                 and type(expected_target_snapshot) is TargetSnapshot and expected_target_snapshot.exists
                 and (expected_terminal_identity is None
                      or type(expected_terminal_identity) is StreamTerminal)
                 and type(expected_labels) is tuple and bool(expected_labels)
                 and expected_labels == tuple(sorted(set(expected_labels)))
                 and all(type(value) is int and value >= 0 for value in expected_labels)
                 and type(dimension) is str and dimension in {"1d", "2d"}
                 and isinstance(preparation_values, Mapping))
        if not valid: return None
        try:
            preparation_json = json.dumps(preparation_values, sort_keys=True,
                separators=(",", ":"), allow_nan=False)
            detached = json.loads(preparation_json)
            if type(detached) is not dict or detached != preparation_values: return None
        except (TypeError, ValueError, OverflowError): return None
        request = _ReintegrateRequest(
            target, entry, source_root, expected_target_snapshot,
            expected_terminal_identity, expected_labels, dimension,
            preparation_json,
        )
        return self._begin(request, stamp, self._run_reintegrate_request)

    def _run_reintegrate_request(self, request, identity, cancelled, publish):
        publish("prepare", 0, 1)
        try:
            plan = ReintegratePlan.from_artifact(request.target, entry=request.entry,
                dimension=request.dimension, preparation=json.loads(request.preparation_json),
                source_root=request.source_root,
                expected_target_snapshot=request.expected_target_snapshot,
                expected_terminal_identity=request.expected_terminal_identity,
                expected_labels=request.expected_labels, cancel_token=cancelled)
        except ReintegrateCancelled:
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
        publish("prepare", 1, 1); headless_revision = 0
        def progress(value):
            nonlocal headless_revision
            try:
                if (type(value) is not ReintegrateProgress
                        or value.operation_identity != plan.operation_identity
                        or value.revision <= headless_revision): return
                headless_revision = value.revision
                publish(value.stage, value.completed, value.total)
            except BaseException: return
        result = run_reintegrate(plan, cancel_token=cancelled, progress_cb=progress)
        if type(result) is not ReintegrateResult:
            raise TypeError("reintegration runner returned an invalid result")
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                 payload=result)

    def begin_reintegrate_successor(
        self,
        *,
        source_artifact: str,
        entry: str,
        source_root: str | None,
        expected_target_snapshot: TargetSnapshot,
        expected_labels: tuple[int, ...],
        dimension: str,
        preparation_values: Mapping[str, object],
        prepared_offer: object,
        stamp: OperationContextStamp,
        expected_terminal_identity: StreamTerminal | None = None,
    ) -> OperationIdentity | None:
        offer_type = type(prepared_offer)
        offer_valid = (
            offer_type.__module__
            == "xrd_tools.reduction.reintegrate_prepared"
            and offer_type.__name__ == "PreparedReintegrateOffer"
            and self._is_frozen_dataclass(prepared_offer)
        )
        valid = (
            type(source_artifact) is str
            and bool(source_artifact)
            and type(entry) is str
            and bool(entry)
            and (
                source_root is None
                or (
                    type(source_root) is str
                    and bool(source_root)
                    and os.path.isabs(source_root)
                    and os.path.normcase(os.path.normpath(source_root))
                    == source_root
                )
            )
            and type(expected_target_snapshot) is TargetSnapshot
            and expected_target_snapshot.exists
            and (
                expected_terminal_identity is None
                or type(expected_terminal_identity) is StreamTerminal
            )
            and type(expected_labels) is tuple
            and bool(expected_labels)
            and expected_labels == tuple(sorted(set(expected_labels)))
            and all(
                type(value) is int and value >= 0
                for value in expected_labels
            )
            and dimension in {"1d", "2d"}
            and isinstance(preparation_values, Mapping)
            and offer_valid
        )
        if not valid:
            return None
        try:
            detached, _encoded_bytes = bounded_json_snapshot(
                preparation_values,
                role="GUI reintegration preparation",
                max_encoded_bytes=64 * 1024 * 1024,
                max_key_bytes=8 * 1024 * 1024,
                max_string_bytes=8 * 1024 * 1024,
                max_depth=24,
                max_nodes=262_144,
                max_children=16_384,
            )
            preparation_json = json.dumps(
                detached,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            replay = json.loads(preparation_json)
            if type(detached) is not dict or replay != detached:
                return None
        except (BoundedJsonError, TypeError, ValueError, OverflowError):
            return None
        request = _ReintegrateSuccessorRequest(
            source_artifact,
            entry,
            source_root,
            expected_target_snapshot,
            expected_terminal_identity,
            expected_labels,
            dimension,
            preparation_json,
            prepared_offer,
        )
        return self._begin(
            request, stamp, self._run_reintegrate_successor_request,
        )

    @staticmethod
    def _valid_reintegrate_successor_result(result, plan, result_type) -> bool:
        if type(result) is not result_type:
            return False
        committed = result.disposition in {"COMMITTED", "ALREADY_COMMITTED"}
        committed_labels = result.committed_labels
        dropped_labels = result.publication_dropped_labels
        labels_are_partition = bool(
            type(committed_labels) is tuple
            and committed_labels == tuple(sorted(set(committed_labels)))
            and all(
                type(label) is int and label >= 0
                for label in committed_labels
            )
            and type(dropped_labels) is tuple
            and dropped_labels == tuple(sorted(set(dropped_labels)))
            and all(
                type(label) is int and label >= 0
                for label in dropped_labels
            )
            and not set(committed_labels).intersection(dropped_labels)
            and set((*committed_labels, *dropped_labels)).issubset(
                plan.labels
            )
            and (
                (
                    committed
                    and bool(committed_labels)
                    and tuple(sorted((
                        *committed_labels, *dropped_labels,
                    ))) == plan.labels
                )
                or (not committed and committed_labels == ())
            )
        )
        audit_identity = result.audit_identity
        valid_audit_identity = bool(
            (
                committed
                and type(audit_identity) is str
                and len(audit_identity) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in audit_identity
                )
            )
            or (
                not committed
                and audit_identity is None
            )
        )
        terminal = result.terminal
        diagnostics = result.diagnostics
        hidden_orphan = result.hidden_orphan
        return bool(
            result.source_artifact == plan.source_artifact
            and result.output_artifact == plan.output_artifact
            and type(result.input_labels) is tuple
            and all(type(label) is int for label in result.input_labels)
            and result.input_labels == plan.labels
            and labels_are_partition
            and valid_audit_identity
            and result.science_identity == plan.science_identity
            and result.version_identity == plan.version_identity
            and result.publication_identity == plan.publication_identity
            and result.operation_identity == plan.operation_identity
            and result.disposition in {
                "COMMITTED", "ALREADY_COMMITTED", "ABORTED",
            }
            and type(diagnostics) is tuple
            and len(diagnostics) <= 16
            and all(
                type(value) is str
                and len(value.encode("utf-8")) <= 4096
                for value in diagnostics
            )
            and (
                hidden_orphan is None
                or type(hidden_orphan) is str
                and bool(hidden_orphan)
                and os.path.normcase(os.path.abspath(os.path.expanduser(
                    hidden_orphan
                ))) == hidden_orphan
            )
            and (not committed or hidden_orphan is None)
            and committed == (type(terminal) is StreamTerminal)
            and (
                not committed
                or stream_terminal_object_revision(terminal) is not None
            )
            and committed
            == (
                type(result.commit_identity) is str
                and len(result.commit_identity) == 64
                and all(character in "0123456789abcdef"
                        for character in result.commit_identity)
            )
            and (
                not committed
                or terminal.target == result.output_artifact
            )
            and (
                committed
                or (
                    result.terminal is None
                    and result.commit_identity is None
                )
            )
        )

    def _run_reintegrate_successor_request(
        self, request, identity, cancelled, publish,
    ):
        # These imports intentionally occur on scattering-operation-* only.
        from xrd_tools.reduction import (
            ReintegrateSuccessorPlan,
            ReintegrateSuccessorProgress,
            ReintegrateSuccessorResult,
            run_reintegrate_successor,
        )
        from xrd_tools.reduction.reintegrate import ReintegrateCancelled

        publish("prepare", 0, 1)
        try:
            plan = ReintegrateSuccessorPlan.from_prepared_or_artifact(
                request.prepared_offer,
                request.source_artifact,
                entry=request.entry,
                dimension=request.dimension,
                preparation=json.loads(request.preparation_json),
                source_root=request.source_root,
                expected_target_snapshot=request.expected_target_snapshot,
                expected_terminal_identity=request.expected_terminal_identity,
                expected_labels=request.expected_labels,
                cancel_token=cancelled,
            )
        except ReintegrateCancelled:
            return OperationTerminal(
                identity, OperationTerminalStatus.CANCELLED,
            )
        publish("prepare", 1, 1)
        headless_revision = 0

        def progress(value):
            nonlocal headless_revision
            try:
                if (
                    type(value) is not ReintegrateSuccessorProgress
                    or value.operation_identity != plan.operation_identity
                    or value.revision <= headless_revision
                ):
                    return
                headless_revision = value.revision
                if value.stage == "qualify":
                    return
                if value.stage == "publish" and value.completed == 0:
                    return
                publish(value.stage, value.completed, value.total)
            except BaseException:
                return

        result = run_reintegrate_successor(
            plan, cancel_token=cancelled, progress_cb=progress,
        )
        if not self._valid_reintegrate_successor_result(
            result, plan, ReintegrateSuccessorResult,
        ):
            raise TypeError(
                "immutable reintegration runner returned an invalid result"
            )
        return OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        )

    def begin_average(self, configuration: object, target: object, *,
                      entry: str = "entry",
                      numeric_metadata_keys: tuple[str, ...] | None = None,
                      invariant_metadata_keys: tuple[str, ...] = (),
                      stamp: OperationContextStamp) -> OperationIdentity | None:
        pathlike = lambda value: isinstance(value, (str, Path)) and bool(str(value))
        if (type(configuration) is not FrozenRunConfiguration
                or not pathlike(target)
                or type(entry) is not str or not entry
                or numeric_metadata_keys is not None and type(numeric_metadata_keys) is not tuple
                or type(invariant_metadata_keys) is not tuple):
            return None
        if (numeric_metadata_keys is not None
                and len(numeric_metadata_keys) > 64
                or len(invariant_metadata_keys) > 64
                or any(type(value) is not str or not value for value in (numeric_metadata_keys or ()))
                or any(type(value) is not str or not value for value in invariant_metadata_keys)):
            return None
        request = _AverageRequest(
            configuration, str(target), entry,
            numeric_metadata_keys, invariant_metadata_keys,
        )
        return self._begin(request, stamp, self._run_average_request)

    def _run_average_request(self, request, identity, cancelled, publish):
        publish("prepare", 0, 1)
        try:
            calibration = _average_calibration(request.configuration, cancelled)
        except RuntimeError as error:
            if (
                cancelled.is_set()
                and error.args == ("admission cancelled",)
            ):
                return OperationTerminal(
                    identity, OperationTerminalStatus.CANCELLED,
                )
            raise
        configuration = request.configuration
        source = configuration.thaw_source_spec()
        from ..output_preflight import native_int_reduction_plan
        reduction = native_int_reduction_plan(configuration)
        recipe = AverageScanRecipe(
            source, request.target,
            reduction, entry=request.entry,
            source_base=configuration.project_root or None,
            output_mode=configuration.output_mode,
            live_mode=configuration.live_mode,
            save_xye=configuration.processing_mode == "Int 1D (XYE)",
            batch_mode=configuration.batch_mode,
            calibration=calibration, background=configuration.background,
            numeric_metadata_keys=request.numeric_metadata_keys,
            invariant_metadata_keys=request.invariant_metadata_keys,
            resource_requests={"workers": configuration.max_cores},
        )
        publish("prepare", 1, 1)
        from xrd_tools.reduction.average import (
            AverageCommand, AverageScanPending, AverageScanProgress,
            AverageScanResult, _committed_average_mismatch,
        )
        headless_revision = -1; headless_identity = None
        def progress(value):
            nonlocal headless_revision, headless_identity
            try:
                if (type(value) is not AverageScanProgress or headless_identity is not None
                        and value.operation_identity != headless_identity or value.revision <= headless_revision): return
                headless_identity = value.operation_identity
                headless_revision = value.revision
                publish(value.stage, value.completed, value.total)
            except BaseException: return
        runner = AverageScanRunner(recipe)
        outcome = runner.start(
            cancel_token=cancelled, progress_cb=progress,
            publication_gate=lambda: self._seal_publication(identity),
        )
        while type(outcome) is AverageScanPending:
            command = self._await_average_command(identity, outcome)
            outcome = runner.command(AverageCommand(command), outcome)
        result = outcome
        runner.close()
        if type(result) is not AverageScanResult or headless_identity is not None and result.operation_identity != headless_identity:
            raise TypeError("Average runner returned an invalid result")
        if result.disposition == "COMMITTED":
            mismatch = _committed_average_mismatch(result, request.target, request.entry)
            if mismatch is not None:
                return OperationTerminal(
                    identity, OperationTerminalStatus.FAILED,
                    f"AVERAGE_COMMIT_VERIFICATION_FAILED: {mismatch}",
                    payload=result,
                )
        if result.disposition in {"COMMITTED", "REFUSED"}:
            return OperationTerminal(identity, OperationTerminalStatus.RETURNED,
                                     payload=result)
        if result.disposition == "CANCELLED":
            return OperationTerminal(identity, OperationTerminalStatus.CANCELLED,
                                     payload=result)
        if result.disposition == "ABORTED":
            return OperationTerminal(
                identity, OperationTerminalStatus.FAILED,
                f"{result.diagnostic_code}: {result.diagnostic}", payload=result,
            )
        raise RuntimeError("Average runner retained settlement unexpectedly")

    @staticmethod
    def _analysis_terminal(identity, result, expected):
        if type(result) is not expected:
            raise TypeError("analysis runner returned an invalid result")
        disposition = getattr(result.disposition, "value", None)
        if disposition in {"completed", "refused"}:
            status = OperationTerminalStatus.RETURNED
        elif disposition == "cancelled":
            status = OperationTerminalStatus.CANCELLED
        else:
            raise TypeError("analysis runner retained settlement unexpectedly")
        return OperationTerminal(identity, status, payload=result)

    def begin_metadata(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import MetadataTablePlan
        return (self._begin(plan, stamp, self._run_metadata)
                if type(plan) is MetadataTablePlan else None)

    def _run_metadata(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import (
            MetadataTableResult, run_metadata_table,
        )
        result = run_metadata_table(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("metadata", done, total))
        return self._analysis_terminal(identity, result, MetadataTableResult)

    def begin_metadata_requalification(
        self, plan: object, stamp: OperationContextStamp,
    ) -> OperationIdentity | None:
        from xrd_tools.analysis.scan_operations import (
            MetadataTableRequalificationPlan,
        )
        return (
            self._begin(plan, stamp, self._run_metadata_requalification)
            if type(plan) is MetadataTableRequalificationPlan else None
        )

    def _run_metadata_requalification(
        self, plan, identity, cancelled, publish,
    ):
        from xrd_tools.analysis.scan_operations import (
            MetadataTableRequalificationResult,
            run_metadata_table_requalification,
        )
        result = run_metadata_table_requalification(
            plan,
            cancel_token=cancelled,
            progress_callback=lambda done, total: publish(
                "metadata_requalification", done, total
            ),
        )
        return self._analysis_terminal(
            identity, result, MetadataTableRequalificationResult
        )

    def begin_scan_plot(self, plan: object, table: object, roi_result: object,
                        stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import (
            RoiScanResult, ScanPlotPlan, MetadataTableResult,
        )
        if (type(plan) is not ScanPlotPlan or type(table) is not MetadataTableResult
                or roi_result is not None and type(roi_result) is not RoiScanResult):
            return None
        return self._begin(_ScanPlotRequest(plan, table, roi_result), stamp,
                           self._run_scan_plot)

    def _run_scan_plot(self, request, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import ScanPlotResult, run_scan_plot
        result = run_scan_plot(
            request.plan, request.table, roi_result=request.roi_result,
            cancel_token=cancelled,
            progress_callback=lambda done, total: publish("scan_plot", done, total))
        return self._analysis_terminal(identity, result, ScanPlotResult)

    def begin_roi_preview(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import RoiPreviewPlan
        return (self._begin(plan, stamp, self._run_roi_preview)
                if type(plan) is RoiPreviewPlan else None)

    def _run_roi_preview(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import RoiPreviewResult, run_roi_preview
        result = run_roi_preview(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("roi_preview", done, total))
        return self._analysis_terminal(identity, result, RoiPreviewResult)

    def begin_roi_scan(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.scan_operations import RoiScanPlan
        return (self._begin(plan, stamp, self._run_roi_scan)
                if type(plan) is RoiScanPlan else None)

    def _run_roi_scan(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.scan_operations import RoiScanResult, run_roi_scan
        result = run_roi_scan(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("roi_scan", done, total))
        return self._analysis_terminal(identity, result, RoiScanResult)

    def begin_peak_fit(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan
        return (self._begin(plan, stamp, self._run_peak_fit)
                if type(plan) is DisplayedPeakFitPlan else None)

    def _run_peak_fit(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.display_fit_operations import (
            DisplayedPeakFitResult, run_displayed_peak_fit,
        )
        result = run_displayed_peak_fit(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("peak_fit", done, total))
        return self._analysis_terminal(identity, result, DisplayedPeakFitResult)

    def begin_phase_fit(self, plan: object, stamp: OperationContextStamp):
        from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan
        return (self._begin(plan, stamp, self._run_phase_fit)
                if type(plan) is DisplayedPhaseFitPlan else None)

    def _run_phase_fit(self, plan, identity, cancelled, publish):
        from xrd_tools.analysis.display_fit_operations import (
            DisplayedPhaseFitResult, run_displayed_phase_fit,
        )
        result = run_displayed_phase_fit(
            plan, cancel_token=cancelled,
            progress_callback=lambda done, total: publish("phase_fit", done, total))
        return self._analysis_terminal(identity, result, DisplayedPhaseFitResult)

    def _seal_publication(self, identity: OperationIdentity) -> bool:
        with self._lock:
            event = self._cancel_event
            if (
                self._identity is not identity or self._terminal is not None
                or self._cancel_sealed or event is None or event.is_set()
            ):
                return False
            self._cancel_sealed = True
            return True

    def _begin(self, frozen: object, stamp: OperationContextStamp,
               body: Callable[..., object], finalize: Callable[[str], None] | None = None
               ) -> OperationIdentity | None:
        if (
            not self._is_frozen_dataclass(frozen)
            or type(stamp) is not OperationContextStamp
            or not callable(body) or finalize is not None and not callable(finalize)
        ):
            return None
        try:
            stamp.__post_init__()
        except BaseException:
            return None

        with self._lock:
            if self._closed or self._identity is not None:
                return None
            identity = OperationIdentity(self._next_serial)
            self._next_serial += 1
            cancel_event = Event()
            worker = Thread(target=self._run,
                args=(identity, frozen, cancel_event, body),
                name=f"scattering-operation-{identity.serial}",
                daemon=False)
            self._identity = identity
            self._stamp = stamp
            self._frozen = frozen
            self._worker = worker
            self._worker_started = False
            self._worker_identity = id(worker)
            self._cancel_event = cancel_event
            self._progress = None
            self._progress_delivered_revision = 0
            self._pending = None
            self._pending_revision = 0
            self._pending_delivered_revision = 0
            self._pending_commanded_revision = 0
            self._deferred_pending_command = None
            self._command_queue = Queue()
            self._terminal = None
            self._stale = False
            self._close_cancel_accepted = False
            self._cancel_sealed = False
            self._clean_receipt = None
            self._finalize_hook = finalize; self._finalized = False
            self._terminal_committed = False; self._abort_fact = None
            try:
                worker.start()
            except BaseException as error:
                self._terminal = self._failed(identity, error)
            else:
                self._worker_started = True
            hook = self._take_finalize_locked("START_FAILED") if not self._worker_started else None
        self._invoke_finalize(hook, "START_FAILED")
        return identity

    def observe_stamp(self, stamp: object) -> None:
        valid = type(stamp) is OperationContextStamp
        if valid:
            try:
                stamp.__post_init__()
            except BaseException:
                valid = False
        with self._lock:
            if self._identity is None or self._stale:
                return
            try:
                mismatch = not valid or stamp != self._stamp
            except BaseException:
                mismatch = True
            if mismatch:
                self._stale = True

    def poll(self, identity: object) -> OperationUpdate | None:
        with self._lock:
            if self._identity is not identity:
                return None
            worker = self._worker
            started = self._worker_started
        if worker is None:
            return None
        joined, alive = self._join_state(worker, started)
        if not joined:
            return None

        with self._lock:
            if self._identity is not identity or self._worker is not worker:
                return None
            progress = self._undelivered_progress_locked()
            pending = self._undelivered_pending_locked()
            if alive:
                if progress is None and pending is None:
                    return None
                return OperationUpdate(
                    identity, progress=progress, pending=pending,
                    stale=self._stale,
                )
            terminal = self._terminal
            if terminal is None:
                if self._abort_fact is None:
                    return None
                hook = self._take_finalize_locked("ABORTED_WITHOUT_TERMINAL")
                self._retire_locked()
                update = None
            else:
                update = OperationUpdate(
                    identity,
                    progress=progress,
                    pending=pending,
                    terminal=terminal,
                    stale=self._stale,
                )
                outcome = ("STALE" if self._stale else
                           "TRANSFERRED" if terminal.status is OperationTerminalStatus.RETURNED
                           else terminal.status.value.upper())
                hook = self._take_finalize_locked(outcome)
                self._retire_locked()
        self._invoke_finalize(hook, "ABORTED_WITHOUT_TERMINAL" if update is None else outcome)
        return update

    def cancel(self, identity: object) -> bool:
        with self._lock:
            event = self._cancel_event
            pending = self._pending
            queue = self._command_queue
            pending_command = (
                pending is not None and queue is not None
                and self._pending_commanded_revision < pending.revision
            )
            if (
                self._identity is not identity
                or self._terminal is not None
                or self._cancel_sealed
                or event is None
                or event.is_set() and not pending_command
            ):
                return False
            if not event.is_set():
                event.set()
            if pending_command:
                self._pending_commanded_revision = pending.revision
                queue.put("cancel")
            elif queue is not None:
                self._deferred_pending_command = "cancel"
            return True

    def retry_average(self, identity: object, pending: object) -> bool:
        with self._lock:
            queue = self._command_queue
            if (
                self._identity is not identity
                or type(pending) is not OperationPending
                or self._pending is not pending
                or self._terminal is not None
                or queue is None
                or self._pending_commanded_revision >= pending.revision
            ):
                return False
            self._pending_commanded_revision = pending.revision
            queue.put("retry")
            return True

    def close(self) -> OperationCleanupReceipt:
        with self._lock:
            if self._clean_receipt is not None:
                return self._clean_receipt
            self._closed = True
            identity = self._identity
            if identity is None:
                receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                self._clean_receipt = receipt
                return receipt
            event = self._cancel_event
            if (
                self._terminal is None
                and not self._cancel_sealed
                and event is not None
                and not event.is_set()
            ):
                event.set()
                self._close_cancel_accepted = True
            pending = self._pending
            queue = self._command_queue
            if (pending is not None and queue is not None
                    and self._pending_commanded_revision < pending.revision):
                self._pending_commanded_revision = pending.revision
                queue.put("close")
            elif (pending is None and self._terminal is None
                  and queue is not None
                  and self._deferred_pending_command is None):
                self._deferred_pending_command = "close"
            worker = self._worker
            started = self._worker_started
            worker_identity = self._worker_identity
        if worker is None or worker_identity is None:
            raise RuntimeError("operation slot lost its worker")
        joined, alive = self._join_state(worker, started)

        with self._lock:
            if self._identity is not identity or self._worker is not worker:
                receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                self._clean_receipt = receipt
                return receipt
            if not joined or alive:
                return OperationCleanupReceipt(identity,
                    CleanupStatus.CLEANUP_PENDING,
                    self._close_cancel_accepted, worker_identity,
                    stale=self._stale)
            terminal = self._terminal
            if terminal is None:
                if self._abort_fact is not None:
                    hook = self._take_finalize_locked("ABORTED_WITHOUT_TERMINAL")
                    self._retire_locked()
                    receipt = OperationCleanupReceipt(None, CleanupStatus.CLEANED)
                    self._clean_receipt = receipt
                else:
                    return OperationCleanupReceipt(identity,
                        CleanupStatus.CLEANUP_PENDING,
                        self._close_cancel_accepted, worker_identity,
                        stale=self._stale)
            else:
                receipt = OperationCleanupReceipt(identity, CleanupStatus.CLEANED,
                    self._close_cancel_accepted, worker_identity, terminal,
                    self._stale)
                outcome = ("STALE" if self._stale else
                           "TRANSFERRED" if terminal.status is OperationTerminalStatus.RETURNED
                           else terminal.status.value.upper())
                hook = self._take_finalize_locked(outcome)
                self._retire_locked()
                self._clean_receipt = receipt
        self._invoke_finalize(hook, "ABORTED_WITHOUT_TERMINAL" if terminal is None else outcome)
        return receipt

    def _run(self, identity: OperationIdentity, frozen: object,
             cancel_event: Event, body: Callable[..., object]) -> None:
        def publish(stage: object, completed: object, total: object) -> None:
            self._publish(identity, stage, completed, total)

        terminal = None
        try:
            candidate = body(frozen, identity, cancel_event, publish)
        except BaseException as error:
            if type(error) is _BackgroundPreterminalAbort:
                self._record_abort_without_terminal(identity, error); return
            try:
                terminal = self._failed(identity, error)
            except BaseException as terminal_error:
                self._record_abort_without_terminal(identity, terminal_error)
                return
        else:
            try:
                if (
                    type(candidate) is not OperationTerminal
                    or candidate.identity is not identity
                ):
                    raise ValueError("operation body returned an invalid terminal")
                candidate.__post_init__()
                terminal = candidate
            except BaseException as error:
                self._record_abort_without_terminal(identity, error)
                return
        try:
            with self._lock:
                if self._identity is identity and self._terminal is None:
                    self._terminal = terminal
                    self._terminal_committed = True
        except BaseException as error:
            self._record_abort_without_terminal(identity, error)

    def _record_abort_without_terminal(self, identity: OperationIdentity,
                                       error: BaseException) -> None:
        module, name, message = detached_exception_strings(error)
        try:
            with self._lock:
                if self._identity is identity and not self._terminal_committed:
                    self._abort_fact = (
                        "ABORTED_WITHOUT_TERMINAL", module, name, message)
                    self._terminal = None
        except BaseException:
            pass

    def _publish(self, identity: OperationIdentity, stage: object,
                 completed: object, total: object) -> None:
        try:
            with self._lock:
                if self._identity is not identity or self._terminal is not None:
                    return
                prior = self._progress
                revision = 1 if prior is None else prior.revision + 1
                candidate = OperationProgress(
                    identity, stage, completed, total, revision
                )
                if (
                    prior is not None
                    and candidate.stage == prior.stage
                    and candidate.completed < prior.completed
                ):
                    return
                self._progress = candidate
        except BaseException:
            return

    def _await_average_command(self, identity, value) -> str:
        try:
            phase = value.phase.value
            diagnostic = value.diagnostic
        except BaseException as error:
            raise ValueError("Average runner returned an invalid pending token") from error
        with self._lock:
            if self._identity is not identity or self._terminal is not None:
                raise RuntimeError("Average operation lost pending ownership")
            self._pending_revision += 1
            revision = self._pending_revision
            pending = OperationPending(identity, revision, phase, diagnostic)
            self._pending = pending
            queue = self._command_queue
            if queue is None:
                raise RuntimeError("Average operation lost its command queue")
            immediate = self._deferred_pending_command
            if immediate is not None:
                self._deferred_pending_command = None
                self._pending_commanded_revision = revision
        command = immediate if immediate is not None else queue.get()
        with self._lock:
            if self._identity is not identity or self._pending is not pending:
                raise RuntimeError("Average pending command became stale")
            self._pending = None
        return command

    def _undelivered_progress_locked(self) -> OperationProgress | None:
        progress = self._progress
        if (
            progress is None
            or progress.revision <= self._progress_delivered_revision
        ):
            return None
        self._progress_delivered_revision = progress.revision
        return progress

    def _undelivered_pending_locked(self) -> OperationPending | None:
        pending = self._pending
        if (pending is None
                or pending.revision <= self._pending_delivered_revision):
            return None
        self._pending_delivered_revision = pending.revision
        return pending

    def _retire_locked(self) -> None:
        self._identity = None
        self._stamp = None
        self._frozen = None
        self._worker = None
        self._worker_started = False
        self._worker_identity = None
        self._cancel_event = None
        self._progress = None
        self._progress_delivered_revision = 0
        self._pending = None
        self._pending_revision = 0
        self._pending_delivered_revision = 0
        self._pending_commanded_revision = 0
        self._deferred_pending_command = None
        self._command_queue = None
        self._terminal = None
        self._stale = False
        self._close_cancel_accepted = False
        self._cancel_sealed = False
        self._finalize_hook = None; self._finalized = False
        self._terminal_committed = False; self._abort_fact = None

    def _take_finalize_locked(self, outcome: str):
        if self._finalized or self._finalize_hook is None:
            return None
        self._finalized = True; return self._finalize_hook

    @staticmethod
    def _invoke_finalize(hook, outcome: str) -> None:
        if hook is None:
            return
        try:
            hook(outcome)
        except BaseException:
            return

    @staticmethod
    def _join_state(worker: Thread, started: bool) -> tuple[bool, bool]:
        if not started:
            return True, False
        try:
            worker.join(timeout=0.0)
            return True, worker.is_alive()
        except RuntimeError:
            return False, True

    @staticmethod
    def _failed(identity: OperationIdentity,
                error: BaseException) -> OperationTerminal:
        module, name, message = detached_exception_strings(error)
        return OperationTerminal(identity, OperationTerminalStatus.FAILED,
                                 f"{module}.{name}: {message}")

    @staticmethod
    def _is_frozen_dataclass(value: object) -> bool:
        try:
            parameters = vars(type(value)).get("__dataclass_params__")
            return (
                is_dataclass(value)
                and not isinstance(value, type)
                and parameters is not None
                and parameters.frozen is True
            )
        except BaseException:
            return False
__all__ = ["OperationSlot"]
