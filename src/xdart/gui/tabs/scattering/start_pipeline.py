from __future__ import annotations

from xrd_tools.session.intent_store import (IntentRecaptureRequired,
                                            RunIntentSnapshot, RunIntentStore)
from xrd_tools.session.run_configuration import FrozenRunConfiguration, RunIntent

from .contracts import (AdmissionReceipt, AdmissionToken, RunExecutorPort,
                        SourceCapture, SourcePort, admitted_capture)
from .controls_editing import (commit_canonical_threshold,
                               commit_canonical_threshold_once)
from .coordinator import ScatteringCoordinator
from .events import (CleanupStatus, ExecutorAccepted, ExecutorClosed, ExecutorStartFailed, LifecycleError,
                     LifecycleResult, LifecycleStatus, OwnersClosed, PreflightAccepted,
                     PreflightRefused, RequestId, RunIdentity)
from .start_outcomes import (RecoveryFailure, StartCapture, StartClosed, StartFailed,
                             StartFailureKind, StartLaunched, StartRecaptureCause,
                             StartRecaptureRequired, StartRefusal, StartRefused, StartRejected,
                             StartRejection, commit_result_is_valid, exception_detail,
                             executor_closed_is_valid, executor_event_is_valid, freeze_configuration, freeze_result_is_valid,
                             lifecycle_is_applied, lifecycle_promotion_is_valid,
                             lifecycle_result_is_valid, recapture_result_is_valid, request_id_is_valid,
                             run_identity_is_valid,
                             snapshot_is_valid, source_capture_error,
                             inert_lifecycle_rejection, invalid_result)
from .state_machine import RunPhase


RecaptureOutcome = StartCapture | StartRefused | StartFailed | StartRecaptureRequired | StartRejected


class StartPipeline:
    def __init__(self, *, intents: RunIntentStore, lifecycle: ScatteringCoordinator,
                 sources: SourcePort, executor: RunExecutorPort) -> None:
        self._intents = intents
        self._lifecycle = lifecycle
        self._sources = sources
        self._executor = executor
        self._capture: StartCapture | None = None
        self._recapture: StartRecaptureRequired | None = None
        self._source_request: RequestId | None = None
        self._close_source_request: RequestId | None = None
        self._source_qualification: SourceCapture | None = None
        self._capture_sequence = 0

    def _valid_capture(self, capture: object) -> bool:
        return (type(capture) is StartCapture and capture is self._capture
                and capture.request_id is self._source_request)

    def _valid_recapture(self, requirement: StartRecaptureRequired) -> bool: return requirement is self._recapture and requirement.request_id is self._source_request

    def _is_preparing_for(self, request_id: RequestId) -> bool:
        try:
            return self._lifecycle.phase is RunPhase.PREPARING and self._lifecycle.request_id is request_id
        except Exception:
            return False

    def _invalidate_tokens(self) -> None:
        self._capture = self._recapture = None

    def _release_source(
        self,
        failures: list[RecoveryFailure],
        *,
        retain_failed: bool = False,
    ) -> None:
        request_id = self._source_request
        self._source_request = None
        self._source_qualification = None
        if request_id is None:
            return
        self._cancel_source_request(
            request_id, failures, retain_failed=retain_failed
        )

    def _cancel_source_request(
        self,
        request_id: RequestId,
        failures: list[RecoveryFailure],
        *,
        retain_failed: bool,
    ) -> None:
        try:
            self._sources.cancel(request_id)
        except Exception as error:
            if (
                retain_failed
                and self._source_request is None
                and self._close_source_request is None
            ):
                self._close_source_request = request_id
            failures.append(RecoveryFailure("source.cancel", exception_detail(error)))

    def _release_executor(
        self, identity: RunIdentity | None, failures: list[RecoveryFailure],
    ) -> ExecutorClosed | None:
        if identity is None:
            return None
        try:
            receipt = self._executor.close(identity)
        except Exception as error:
            failures.append(RecoveryFailure("executor.close", exception_detail(error)))
            return None
        if executor_closed_is_valid(receipt, identity):
            return receipt
        failures.append(invalid_result("executor.close"))
        return None

    def _lifecycle_sequence(self) -> int | None:
        try:
            sequence = self._lifecycle.event_sequence
        except Exception:
            return None
        return (
            sequence
            if type(sequence) is int and sequence >= 0
            else None
        )

    def _effect_after_exception(
        self,
        sequence: int | None,
        *,
        stopping_identity: RunIdentity | None = None,
        closed_identity: RunIdentity | None = None,
    ) -> LifecycleResult | None:
        """Recover only one synchronously observed lifecycle transition."""

        try:
            applied = (
                sequence is not None
                and self._lifecycle.event_sequence == sequence + 1
                and self._lifecycle.closed is True
                and self._lifecycle.request_id is None
                and self._lifecycle.active_run_identity is None
                and self._lifecycle.attempt_run_identity is None
            )
            phase = self._lifecycle.phase
        except Exception:
            return None
        if not applied:
            return None
        if phase is RunPhase.CLOSED:
            return LifecycleResult(
                LifecycleStatus.APPLIED,
                RunPhase.CLOSED,
                run_identity=closed_identity,
            )
        if (
            phase is RunPhase.STOPPING
            and run_identity_is_valid(stopping_identity)
        ):
            return LifecycleResult(
                LifecycleStatus.APPLIED,
                RunPhase.STOPPING,
                run_identity=stopping_identity,
            )
        return None

    def _close_result(self, identity: RunIdentity | None,
                      failures: list[RecoveryFailure]) -> LifecycleResult:
        sequence = self._lifecycle_sequence()
        try:
            result = self._lifecycle.close()
        except Exception as error:
            failures.append(RecoveryFailure("lifecycle.close", exception_detail(error)))
            applied = self._effect_after_exception(
                sequence, stopping_identity=identity
            )
            if applied is not None:
                return applied
        else:
            if lifecycle_is_applied(result, RunPhase.CLOSED, None, None):
                return result
            if (run_identity_is_valid(identity)
                    and lifecycle_is_applied(result, RunPhase.STOPPING, None, identity)):
                return result
            if (identity is None and lifecycle_result_is_valid(result)
                    and result.status is LifecycleStatus.SUPERSEDED
                    and result.error is LifecycleError.SUPERSEDED
                    and result.phase in (RunPhase.STOPPING, RunPhase.CLOSED)
                    and result.request_id is None and result.run_identity is None):
                return result
            failures.append(invalid_result("lifecycle.close"))
        return LifecycleResult(LifecycleStatus.REJECTED, RunPhase.CLOSED)

    def _closed_invariant(self, error: Exception, *,
                          reason: StartFailureKind = StartFailureKind.CANONICAL_INVARIANT,
                          configuration: FrozenRunConfiguration | None = None,
                          run_identity: RunIdentity | None = None,
                          cleanup_status: CleanupStatus = CleanupStatus.NOT_STARTED,
                          recovery_failures: tuple[RecoveryFailure, ...] = ()) -> StartFailed:
        diagnostic = exception_detail(error)
        failures = list(recovery_failures)
        closed = self._close_result(run_identity, failures)
        self._invalidate_tokens()
        self._release_source(failures)
        return StartFailed(reason, configuration, run_identity, closed, cleanup_status,
                           diagnostic.message, diagnostic, tuple(failures))

    def _refuse(self, request_id: RequestId, reason: StartRefusal, *,
                error: Exception | None = None, detail: str = "") -> StartRefused | StartFailed:
        diagnostic = None if error is None else exception_detail(error)
        try:
            result = self._lifecycle.preflight_refused(PreflightRefused(request_id))
        except Exception as lifecycle_error:
            return self._closed_invariant(lifecycle_error)
        if not lifecycle_is_applied(result, RunPhase.IDLE, request_id, None):
            return self._closed_invariant(ValueError("lifecycle rejected preflight refusal"),
                                          reason=StartFailureKind.LIFECYCLE_INVARIANT,
                                          recovery_failures=(invalid_result("lifecycle.preflight_refused"),))
        self._invalidate_tokens()
        failures: list[RecoveryFailure] = []
        self._release_source(failures)
        return StartRefused(request_id, reason, result,
                            diagnostic.message if diagnostic is not None else detail,
                            diagnostic, tuple(failures))

    def _snapshot_or_failure(self) -> RunIntentSnapshot | StartFailed:
        try:
            snapshot = self._intents.snapshot()
        except Exception as error:
            return self._closed_invariant(error)
        if not snapshot_is_valid(snapshot):
            return self._closed_invariant(TypeError("RunIntentStore.snapshot() returned an invalid result"))
        return snapshot

    def _recapture_required(self, capture: StartCapture | StartRecaptureRequired, cause: StartRecaptureCause,
                            snapshot: RunIntentSnapshot, minimum_source_epoch: int = 0) -> StartRecaptureRequired:
        self._capture = None
        sequence = capture.capture_sequence if type(capture) is StartCapture else capture.superseded_capture_sequence
        requirement = StartRecaptureRequired(capture.request_id, sequence, cause,
                                             snapshot, minimum_source_epoch)
        self._recapture = requirement
        return requirement

    def _capture_from_snapshot(self, snapshot: RunIntentSnapshot, request_id: RequestId,
                               minimum_source_epoch: int = 0) -> StartCapture | StartRefused | StartFailed:
        if not snapshot_is_valid(snapshot):
            return self._closed_invariant(TypeError("RunIntentStore.snapshot() returned an invalid result"))
        try:
            snapshot = self._canonical_threshold_snapshot(snapshot)
        except Exception as error:
            return self._closed_invariant(error)
        if snapshot is None:
            return self._refuse(
                request_id,
                StartRefusal.FREEZE_VALIDATION,
                detail=(
                    "threshold canonicalization lost three consecutive "
                    "intent revisions; retry the start"
                ),
            )
        if not snapshot_is_valid(snapshot):
            return self._closed_invariant(TypeError("threshold canonicalization returned an invalid snapshot"))
        try:
            source = snapshot.thaw().source_spec
        except Exception as error:
            return self._closed_invariant(error)
        if source is None:
            return self._refuse(request_id, StartRefusal.MISSING_SOURCE)
        if self._source_request is None:
            self._source_request = request_id
        try:
            source_capture = self._sources.capture(source, request_id)
        except Exception as error:
            return self._refuse(request_id, StartRefusal.SOURCE_CAPTURE_EXCEPTION, error=error)
        invalid = source_capture_error(source_capture, request_id, source,
                                       self._source_qualification, minimum_source_epoch)
        if invalid is not None:
            return self._refuse(request_id, StartRefusal.SOURCE_CAPTURE_INVALID, detail=invalid)
        self._capture_sequence += 1
        self._source_qualification = source_capture
        capture = StartCapture(request_id, self._capture_sequence, snapshot, source_capture)
        self._capture = capture
        self._recapture = None
        return capture

    def _canonical_threshold_snapshot(
        self, snapshot: RunIntentSnapshot
    ) -> RunIntentSnapshot | None:
        """Run THE shared threshold canonicalizer THROUGH the store at the
        start/capture boundary (frozen design checkpoint, 2026-08-04).

        Both run freezes — the admission candidate's local freeze and the
        store freeze that produces the executed configuration — derive from
        this one revisioned state, so committing the canonical identity here
        is the single point where display, fingerprint, execution values and
        persisted provenance are all made to agree.  It runs on EVERY
        capture, not only for degenerate boolean pairs: manual mode must
        also MATERIALIZE the displayed default bounds (a cleared bound
        stores None while the panel shows the substituted default).  The
        CAS loop itself is hosted by the editing module
        (:func:`commit_canonical_threshold`) so the pipeline never handles
        a raw thawed intent — this seam receives and returns snapshots
        only.  Exhaustion returns ``None`` and the caller refuses the
        start with a typed outcome — a raw non-canonical capture is never
        produced.
        """
        return commit_canonical_threshold(self._intents, snapshot)

    def begin(self) -> StartCapture | StartRefused | StartRejected | StartFailed:
        try:
            result = self._lifecycle.begin_start()
        except Exception as error:
            return self._closed_invariant(error, reason=StartFailureKind.LIFECYCLE_INVARIANT)
        if lifecycle_is_applied(result, RunPhase.PREPARING, run_identity=None):
            request_id = result.request_id
            try:
                owned_request = self._lifecycle.request_id
            except Exception as error:
                return self._closed_invariant(
                    error,
                    reason=StartFailureKind.LIFECYCLE_INVARIANT,
                )
            if not request_id_is_valid(owned_request) or request_id is not owned_request:
                return self._closed_invariant(
                    ValueError("lifecycle begin_start returned a foreign request identity"),
                    reason=StartFailureKind.LIFECYCLE_INVARIANT,
                )
        elif lifecycle_result_is_valid(result) and result.status is not LifecycleStatus.APPLIED:
            return StartRejected(StartRejection.LIFECYCLE, result)
        else:
            return self._closed_invariant(
                TypeError("lifecycle begin_start() returned an invalid result"),
                reason=StartFailureKind.LIFECYCLE_INVARIANT,
                recovery_failures=(invalid_result("lifecycle.begin_start"),),
            )
        self._source_qualification = None
        self._capture_sequence = 0
        snapshot = self._snapshot_or_failure()
        return snapshot if isinstance(snapshot, StartFailed) else self._capture_from_snapshot(snapshot, request_id)

    def apply_operator_decision(
        self,
        capture: StartCapture | AdmissionReceipt,
        candidate: RunIntent,
    ) -> RecaptureOutcome:
        if type(capture) is AdmissionReceipt:
            capture = admitted_capture(capture, self._capture)
        if not self._valid_capture(capture):
            return StartRejected(StartRejection.STALE_CAPTURE)
        expected = capture.intent_snapshot.revision
        try:
            result = self._intents.commit(candidate, expected_revision=expected)
        except Exception as error:
            return self._closed_invariant(error)
        if recapture_result_is_valid(result, expected):
            return self._recapture_required(capture, StartRecaptureCause.COMMIT_RACE, result.snapshot)
        if commit_result_is_valid(result, expected):
            self._capture = None
            # Design checkpoint 2026-08-04: every StartCapture creation path
            # routes through the shared canonicalizer.  The operator
            # candidate commits first (the store commit is the only seam
            # that may receive a raw intent); the ACCEPTED snapshot is then
            # canonicalized SINGLE-SHOT — this path owns one specific
            # accepted revision, so an intervening edit is a COMMIT_RACE to
            # surface with the exact current snapshot, never a revision to
            # consume into a replacement capture or a source recapture
            # (bounded rework, 2026-08-04).  Already-canonical candidates
            # take no extra revision.
            try:
                canonical = commit_canonical_threshold_once(self._intents, result.snapshot)
            except Exception as error:
                return self._closed_invariant(error)
            if type(canonical) is IntentRecaptureRequired:
                if not recapture_result_is_valid(canonical, result.snapshot.revision):
                    return self._closed_invariant(TypeError("RunIntentStore.commit() returned an invalid result"))
                return self._recapture_required(capture, StartRecaptureCause.COMMIT_RACE, canonical.snapshot)
            if not snapshot_is_valid(canonical):
                return self._closed_invariant(TypeError("threshold canonicalization returned an invalid snapshot"))
            if canonical.thaw().source_spec == capture.source_capture.source:
                self._capture_sequence += 1
                replacement = StartCapture(
                    capture.request_id, self._capture_sequence,
                    canonical, capture.source_capture,
                )
                self._capture = replacement
                return replacement
            return self._capture_from_snapshot(canonical, capture.request_id)
        return self._closed_invariant(TypeError("RunIntentStore.commit() returned an invalid result"))

    def source_observation_changed(
        self, request_id: RequestId, source_epoch: int
    ) -> StartRecaptureRequired | StartRejected | StartFailed:
        if type(request_id) is not RequestId or type(source_epoch) is not int or source_epoch < 0:
            return StartRejected(StartRejection.SOURCE_OBSERVATION)
        capture = self._capture
        if (
            capture is None
            or request_id is not self._source_request
            or request_id is not capture.request_id
            or not self._is_preparing_for(request_id)
            or source_epoch <= capture.source_capture.source_epoch
        ):
            return StartRejected(StartRejection.SOURCE_OBSERVATION)
        snapshot = self._snapshot_or_failure()
        if isinstance(snapshot, StartFailed):
            return snapshot
        return self._recapture_required(capture, StartRecaptureCause.SOURCE_OBSERVATION, snapshot, source_epoch)

    def recapture(self, requirement: StartRecaptureRequired) -> RecaptureOutcome:
        if (
            not self._valid_recapture(requirement)
            or not self._is_preparing_for(requirement.request_id)
        ):
            return StartRejected(StartRejection.STALE_RECAPTURE)
        current = self._snapshot_or_failure()
        if isinstance(current, StartFailed):
            return current
        if current.revision != requirement.current_snapshot.revision:
            return self._recapture_required(requirement, StartRecaptureCause.RECAPTURE_ADVANCED,
                                            current, requirement.minimum_source_epoch)
        return self._capture_from_snapshot(requirement.current_snapshot, requirement.request_id,
                                           requirement.minimum_source_epoch)

    def _postfreeze_closed(self, error: Exception, configuration: FrozenRunConfiguration,
                           identity: RunIdentity | None,
                           recovery_failures: tuple[RecoveryFailure, ...] = ()) -> StartFailed:
        return self._closed_invariant(error, reason=StartFailureKind.LIFECYCLE_INVARIANT,
                                      configuration=configuration, run_identity=identity,
                                      recovery_failures=recovery_failures)

    def _rejected_containment_result(
        self, identity: RunIdentity, failures: list[RecoveryFailure]
    ) -> LifecycleResult:
        try:
            phase = self._lifecycle.phase
        except Exception as error:
            failures.append(RecoveryFailure("lifecycle.phase", exception_detail(error)))
            phase = RunPhase.FAILED
        if type(phase) is not RunPhase:
            failures.append(invalid_result("lifecycle.phase"))
            phase = RunPhase.FAILED
        return LifecycleResult(LifecycleStatus.REJECTED, phase, run_identity=identity,
                               error=LifecycleError.ILLEGAL_TRANSITION)

    def _post_executor_failure(self, error: Exception | None, configuration: FrozenRunConfiguration,
                               identity: RunIdentity, *,
                               reason: StartFailureKind = StartFailureKind.EXECUTOR_INVARIANT,
                               detail: str = "", recovery_failures: tuple[RecoveryFailure, ...] = ()) -> StartFailed:
        diagnostic = None if error is None else exception_detail(error)
        failures = list(recovery_failures)
        try:
            result = self._lifecycle.contain_executor_failure(identity)
        except Exception as containment_error:
            failures.append(RecoveryFailure("lifecycle.contain_executor_failure", exception_detail(containment_error)))
            result = self._rejected_containment_result(identity, failures)
        if not lifecycle_is_applied(result, RunPhase.FAILED, None, identity):
            failures.append(invalid_result("lifecycle.contain_executor_failure"))
            result = self._rejected_containment_result(identity, failures)
        self._invalidate_tokens()
        self._release_source(failures)
        self._release_executor(identity, failures)
        return StartFailed(reason, configuration, identity, result, CleanupStatus.CLEANUP_PENDING,
                           diagnostic.message if diagnostic is not None else detail, diagnostic, tuple(failures))

    def _executor_invariant(self, error: Exception, configuration: FrozenRunConfiguration,
                            identity: RunIdentity) -> StartFailed:
        failures: list[RecoveryFailure] = []
        try:
            result = self._lifecycle.executor_start_failed(
                ExecutorStartFailed(identity, CleanupStatus.CLEANUP_PENDING)
            )
        except Exception as lifecycle_error:
            failures.append(RecoveryFailure("lifecycle.executor_start_failed", exception_detail(lifecycle_error)))
        else:
            if not lifecycle_is_applied(result, RunPhase.FAILED, None, identity):
                failures.append(invalid_result("lifecycle.executor_start_failed"))
        return self._post_executor_failure(error, configuration, identity, recovery_failures=tuple(failures))

    def start(self, admission: AdmissionReceipt) -> StartLaunched | RecaptureOutcome:
        capture = admitted_capture(admission, self._capture)
        if not self._valid_capture(capture):
            return StartRejected(StartRejection.STALE_CAPTURE)
        expected_revision = capture.intent_snapshot.revision
        try:
            result = self._intents.freeze(expected_revision=expected_revision,
                                          gi_motor_choices=admission.gi_motor_choices)
        except (ValueError, TypeError) as error:
            return self._refuse(capture.request_id, StartRefusal.FREEZE_VALIDATION, error=error)
        except Exception as error:
            return self._closed_invariant(error)
        if recapture_result_is_valid(result, expected_revision):
            return self._recapture_required(capture, StartRecaptureCause.FREEZE_RACE, result.snapshot)
        configuration = freeze_configuration(result)
        if configuration is not None and not freeze_result_is_valid(result, expected_revision):
            try:
                identity = RunIdentity.from_configuration(configuration)
            except Exception:
                identity = None
            return self._postfreeze_closed(
                ValueError("accepted freeze result disagrees with the capture"), configuration, identity)
        if not freeze_result_is_valid(result, expected_revision):
            return self._closed_invariant(TypeError("RunIntentStore.freeze() returned an invalid result"))
        try:
            expected_identity = RunIdentity.from_configuration(configuration)
        except Exception as error:
            return self._postfreeze_closed(error, configuration, None)
        try:
            promoted = self._lifecycle.preflight_accepted(PreflightAccepted(capture.request_id, configuration))
        except Exception as error:
            return self._postfreeze_closed(error, configuration, expected_identity)
        try:
            owned_identity = self._lifecycle.attempt_run_identity
        except Exception as error:
            return self._postfreeze_closed(error, configuration, expected_identity)
        if (
            not lifecycle_promotion_is_valid(promoted, capture.request_id, expected_identity)
            or promoted.run_identity is not owned_identity
        ):
            return self._postfreeze_closed(
                ValueError("lifecycle rejected accepted preflight"),
                configuration,
                expected_identity,
                (invalid_result("lifecycle.preflight_accepted"),),
            )
        identity = owned_identity
        try:
            event = self._executor.start(configuration, capture.source_capture, identity, admission)
        except Exception as error:
            return self._executor_invariant(error, configuration, identity)
        if not executor_event_is_valid(event, identity):
            return self._executor_invariant(
                ValueError("executor returned an unknown or foreign run identity"),
                configuration,
                identity,
            )
        if type(event) is ExecutorAccepted:
            try:
                accepted = self._lifecycle.executor_accepted(event)
            except Exception as error:
                return self._post_executor_failure(
                    error,
                    configuration,
                    identity,
                    reason=StartFailureKind.LIFECYCLE_INVARIANT,
                )
            if not lifecycle_is_applied(accepted, RunPhase.RUNNING, None, identity):
                return self._post_executor_failure(
                    ValueError("lifecycle rejected matching executor acceptance"),
                    configuration,
                    identity,
                    reason=StartFailureKind.LIFECYCLE_INVARIANT,
                    recovery_failures=(invalid_result("lifecycle.executor_accepted"),),
                )
            self._invalidate_tokens()
            failures: list[RecoveryFailure] = []
            self._release_source(failures)
            return StartLaunched(configuration, capture.source_capture, identity, accepted, tuple(failures))
        return self._executor_failed(event, configuration, identity)

    def _executor_failed(
        self, event: ExecutorStartFailed, configuration: FrozenRunConfiguration, identity: RunIdentity
    ) -> StartFailed:
        failures: list[RecoveryFailure] = []
        try:
            failed = self._lifecycle.executor_start_failed(event)
        except Exception as error:
            failures.append(RecoveryFailure("lifecycle.executor_start_failed", exception_detail(error)))
        else:
            if lifecycle_is_applied(failed, RunPhase.FAILED, None, identity):
                self._invalidate_tokens()
                self._release_source(failures)
                if event.cleanup_status is not CleanupStatus.CLEANED:
                    return StartFailed(
                        StartFailureKind.EXECUTOR_START_FAILED, configuration, identity, failed,
                        event.cleanup_status, event.error.value, recovery_failures=tuple(failures),
                    )
                try:
                    closed = self._lifecycle.owners_closed(OwnersClosed(identity))
                except Exception as error:
                    failures.append(RecoveryFailure("lifecycle.owners_closed", exception_detail(error)))
                else:
                    if lifecycle_is_applied(closed, RunPhase.FAILED, None, identity):
                        return StartFailed(
                            StartFailureKind.EXECUTOR_START_FAILED, configuration, identity, failed,
                            event.cleanup_status, event.error.value, recovery_failures=tuple(failures),
                        )
                    failures.append(invalid_result("lifecycle.owners_closed"))
            else:
                failures.append(invalid_result("lifecycle.executor_start_failed"))
        return self._post_executor_failure(
            None, configuration, identity, reason=StartFailureKind.EXECUTOR_START_FAILED,
            detail=event.error.value, recovery_failures=tuple(failures),
        )

    def refuse(
        self,
        capture: StartCapture | AdmissionReceipt | AdmissionToken,
        reason: StartRefusal,
    ) -> StartRefused | StartRejected | StartFailed:
        if type(capture) in {AdmissionReceipt, AdmissionToken}:
            capture = admitted_capture(capture, self._capture)
        if not self._valid_capture(capture):
            return StartRejected(StartRejection.STALE_CAPTURE)
        return self._refuse(capture.request_id, reason)

    def owners_closed(self, event: OwnersClosed) -> LifecycleResult:
        if type(event) is not OwnersClosed or not run_identity_is_valid(event.run_identity):
            return inert_lifecycle_rejection()
        sequence = self._lifecycle_sequence()
        try:
            result = self._lifecycle.owners_closed(event)
        except Exception:
            result = self._effect_after_exception(
                sequence, closed_identity=event.run_identity
            )
            if result is None:
                return inert_lifecycle_rejection()
        if (
            lifecycle_is_applied(result, RunPhase.FAILED, None, event.run_identity)
            or lifecycle_is_applied(result, RunPhase.CLOSED, None, event.run_identity)
        ):
            return result
        return inert_lifecycle_rejection()

    def retry_close_lifecycle(
        self, identity: RunIdentity | None
    ) -> StartClosed:
        """Retry only the lifecycle half of an already-owned close."""

        failures: list[RecoveryFailure] = []
        result = self._close_result(identity, failures)
        return StartClosed(
            result,
            tuple(failures),
            CleanupStatus.CLEANUP_PENDING,
            cleanup_identity=identity,
        )

    def retry_close_source(
        self,
    ) -> tuple[CleanupStatus, tuple[RecoveryFailure, ...]]:
        """Retry only the exact source request retained by ``close``."""

        request_id = self._close_source_request
        if request_id is None:
            return CleanupStatus.CLEANUP_PENDING, ()
        self._close_source_request = None
        failures: list[RecoveryFailure] = []
        self._cancel_source_request(
            request_id, failures, retain_failed=True
        )
        return (
            (
                CleanupStatus.CLEANUP_PENDING
                if failures
                else CleanupStatus.CLEANED
            ),
            tuple(failures),
        )

    def close(self) -> StartClosed:
        failures: list[RecoveryFailure] = []
        try:
            identity = self._lifecycle.active_run_identity
            if identity is None:
                identity = self._lifecycle.attempt_run_identity
        except Exception as error:
            identity = None
            failures.append(RecoveryFailure("lifecycle.identity", exception_detail(error)))
        if identity is not None and not run_identity_is_valid(identity):
            failures.append(invalid_result("lifecycle.identity"))
            identity = None
        result = self._close_result(identity, failures)
        self._invalidate_tokens()
        self._release_source(failures, retain_failed=True)
        receipt = self._release_executor(identity, failures)
        if receipt is None:
            if (
                identity is None
                and not failures
                and lifecycle_result_is_valid(result)
                and result.phase is RunPhase.CLOSED
            ):
                return StartClosed(
                    result,
                    cleanup_status=CleanupStatus.CLEANED,
                    cleanup_identity=None,
                )
            return StartClosed(
                result,
                tuple(failures),
                CleanupStatus.CLEANUP_PENDING,
                cleanup_identity=identity,
            )
        return StartClosed(result, tuple(failures), receipt.cleanup_status,
                           receipt.primary, receipt.cleanup_failures, identity)


__all__ = ["StartPipeline"]
