"""One finite worker slot for later externally owned operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
import json
import os, stat; from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable

from ..events import CleanupStatus, detached_exception_strings
from ..experiment_authoring import (
    CalibrationRequest, prepare_calibration_request, run_calibration,
    MaskRequest, resolve_mask_executable, run_mask,
)
from ..operation_values import (
    OperationCleanupReceipt, OperationContextStamp, OperationIdentity,
    OperationProgress, OperationTerminal, OperationTerminalStatus,
    OperationUpdate,
)
from ..presentation_background import PresentationBackgroundOwner
from xrd_tools.reduction.background import DisplayBackgroundPlan
from xrd_tools.io.output_transaction import TargetSnapshot
from xrd_tools.reduction import ReintegratePlan, ReintegrateProgress, ReintegrateResult, run_reintegrate
from xrd_tools.reduction.reintegrate import ReintegrateCancelled

class _BackgroundPreterminalAbort(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class _ReintegrateRequest:
    target: str
    entry: str
    expected_target_snapshot: TargetSnapshot
    expected_labels: tuple[int, ...]
    dimension: str
    preparation_json: str

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
            fresh = prepare_calibration_request(request.final_path)
        except (OSError, ValueError):
            return None
        if fresh != request:
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
        return run_mask(request, identity, cancelled, publish, self._seal_publication)

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

    def begin_reintegrate(self, *, target: str, entry: str,
                          expected_target_snapshot: TargetSnapshot,
                          expected_labels: tuple[int, ...], dimension: str,
                          preparation_values: Mapping[str, object],
                          stamp: OperationContextStamp) -> OperationIdentity | None:
        valid = (type(target) is str and bool(target) and type(entry) is str and bool(entry)
                 and type(expected_target_snapshot) is TargetSnapshot and expected_target_snapshot.exists
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
        request = _ReintegrateRequest(target, entry, expected_target_snapshot,
            expected_labels, dimension, preparation_json)
        return self._begin(request, stamp, self._run_reintegrate_request)

    def _run_reintegrate_request(self, request, identity, cancelled, publish):
        publish("prepare", 0, 1)
        try:
            plan = ReintegratePlan.from_artifact(request.target, entry=request.entry,
                dimension=request.dimension, preparation=json.loads(request.preparation_json),
                expected_target_snapshot=request.expected_target_snapshot,
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
            if alive:
                if progress is None:
                    return None
                return OperationUpdate(identity, progress=progress, stale=self._stale)
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
            if (
                self._identity is not identity
                or self._terminal is not None
                or self._cancel_sealed
                or event is None
                or event.is_set()
            ):
                return False
            event.set()
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

    def _undelivered_progress_locked(self) -> OperationProgress | None:
        progress = self._progress
        if (
            progress is None
            or progress.revision <= self._progress_delivered_revision
        ):
            return None
        self._progress_delivered_revision = progress.revision
        return progress

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
