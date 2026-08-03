"""Qt-free command owner for acquisition and processed browse contexts."""

from __future__ import annotations

from xdart.modules.display_context import (
    AcquisitionContext, BrowseContext, ContextKind, DisplaySelection,
    new_context_token,
)

from .acquisition_runtime import CommandCompensationFailure
from .browse_hydration import _BrowseHydrationOwner
from .browse_preview import (
    browse_preview_polling_needed,
    browse_preview_repaint_ready,
    release_browse,
)
from .browse_values import (
    BrowseCleanupReceipt, BrowseLoadOutcome, BrowseLoadRequest,
    BrowseLoadStatus,
)
from .context_projection import ContextProjection, ProjectionRequest
from .context_runtime import _ContextRuntime
from .display_retirement import DisplayRetirementReceipt
from .display_values import (
    DisplayFrameKey, DisplayNavigationDelta, StandardDisplayPayload,
    StandardRunEvent,
)
from .events import (
    CleanupStatus, DurablePaused, FatalExecution, PauseFailed, PauseRequested,
    ResumeFailed, ResumeRequested, Resumed, RunIdentity, StopRequested,
    detach_exception,
)
from .state_machine import RunPhase
from .shell_projection import ScientificPreferences
from .shell_values import FrameNavigationProjection


class ContextController:
    """Coordinate commands around one exact context/navigation runtime."""

    def __init__(self, *, lifecycle, executor, browse_loader,
                 projection: ContextProjection) -> None:
        self._lifecycle = lifecycle
        self._executor = executor
        self._browse_loader = browse_loader
        self._projection = projection
        self._runtime = _ContextRuntime()
        self._browse_request: BrowseLoadRequest | None = None
        self._browse_hydration_owner: _BrowseHydrationOwner | None = None
        self._cleanup_receipt: BrowseCleanupReceipt | None = None
        self._load_generation = 0
        self._close: BrowseCleanupReceipt | None = None

    @property
    def _closed(self) -> bool:
        return self._close is not None and self._close.cleanup_status is CleanupStatus.CLEANED

    @property
    def acquisition_context(self) -> AcquisitionContext | None:
        return self._runtime.acquisition_context

    @property
    def browse_context(self) -> BrowseContext | None:
        return self._runtime.browse_context

    @property
    def selection(self) -> DisplaySelection | None:
        return self._runtime.selection

    @property
    def retained_contexts(self) -> tuple:
        return self._runtime.retained_contexts

    @property
    def projectable_contexts(self) -> tuple:
        return self._runtime.projectable_contexts

    @property
    def run_identity(self) -> RunIdentity | None:
        return self._runtime.run_identity

    @property
    def browse_pending(self) -> bool:
        return self._browse_request is not None or self._cleanup_receipt is not None

    @property
    def browse_preview_polling_needed(self) -> bool:
        return (
            not self._closed
            and browse_preview_polling_needed(
                self._runtime, self._browse_hydration_owner
            )
        )

    @property
    def frame_keys(self) -> tuple[DisplayFrameKey, ...]:
        return self._runtime.frame_keys

    @property
    def navigation(self) -> FrameNavigationProjection:
        return self._runtime.navigation

    @property
    def resident_frame_keys(self) -> frozenset[DisplayFrameKey]:
        return self._runtime.resident_frame_keys(
            self._projection, self._browse_hydration_owner
        )

    @property
    def norm_aggregate(self):
        """Borrow the §25.3 runtime-held snapshot for ONE shell refresh."""
        return self._runtime.norm_aggregate

    def owns_frame(self, frame: object) -> bool:
        return self._runtime.owns_frame(frame)

    def poll_browse_preview(self) -> bool:
        consumed = (
            not self._closed
            and browse_preview_repaint_ready(
                self._runtime, self._browse_hydration_owner
            )
        )
        if consumed:
            # A consumed cold repaint invalidates the projection pass: the
            # completed read changed exactly what the pass snapshotted.
            self._runtime.invalidate_browse_pass()
        return consumed

    def adopt_acquisition(self, run_identity: RunIdentity) -> DisplaySelection:
        if self._closed or type(run_identity) is not RunIdentity:
            raise RuntimeError("context controller cannot adopt acquisition")
        context = self._executor.acquisition_context(run_identity)
        return self._runtime.adopt_acquisition(run_identity, context)

    def synchronize_acquisition_scope(self) -> bool:
        """Re-adopt an executor rescope missed by the last event snapshot."""

        if self._closed:
            return False
        identity = self._runtime.run_identity
        context = self._runtime.acquisition_context
        selection = self._runtime.selection
        if (
            identity is None
            or context is None
            or selection is None
            or selection.kind is not ContextKind.ACQUISITION
            or not selection.names(context)
            or selection.owner == context.hydration_owner
        ):
            return False
        current = self._executor.acquisition_context(identity)
        if current is not context:
            return False
        self._runtime.adopt_acquisition(identity, current)
        return True

    def accept_navigation(
        self,
        delta: DisplayNavigationDelta,
        *,
        plot_mode: str = "Single",
        follow_latest: bool = True,
    ) -> bool:
        if self._closed:
            return False
        return self._runtime.accept_navigation(
            delta,
            plot_mode=plot_mode,
            follow_latest=follow_latest,
        )

    def select_latest_navigation(
        self, *, plot_mode: str = "Single"
    ) -> bool:
        if self._closed:
            return False
        return self._runtime.select_latest_navigation(plot_mode=plot_mode)

    def select_navigation(
        self,
        current: DisplayFrameKey | None,
        selected: tuple[DisplayFrameKey, ...],
    ) -> bool:
        if self._closed:
            return False
        return self._runtime.select_navigation(current, selected)

    def apply_display_retirement(self, receipt: DisplayRetirementReceipt) -> bool:
        if not self._runtime.retirement_matches(receipt):
            return False
        if self._cleanup_receipt is not None:
            return False
        browse = self._runtime.browse_context
        if browse is not None:
            released = self._release_browse(browse)
            if (
                type(released) is not BrowseCleanupReceipt
                or released.request is not browse.load_request
                or released.cleanup_status is not CleanupStatus.CLEANED
            ):
                return False
            self._runtime.clear_browse(select_acquisition=False)
        request = self._browse_request
        if request is not None:
            self._invalidate_browse_request()
            if self._cleanup_receipt is not None:
                return False
        identity = self._runtime.run_identity
        if identity is None:
            self._runtime.close_selection()
            return self._runtime.acquisition_context is None
        return self._runtime.apply_display_retirement(receipt)

    def release_acquisition(self, identity: RunIdentity | None) -> bool:
        return self._runtime.release_acquisition(identity)

    def pause(self) -> DurablePaused | PauseFailed:
        identity = self._require_identity()
        requested = self._lifecycle.pause_requested(
            PauseRequested(identity)
        )
        if requested.phase is not RunPhase.PAUSING:
            raise RuntimeError("lifecycle refused Pause")
        try:
            durable = self._executor.pause(identity)
        except BaseException as error:
            return self._command_failure(
                identity,
                error,
                PauseFailed,
                self._lifecycle.pause_failed,
                RunPhase.RUNNING,
                "pause",
            )
        accepted = self._lifecycle.durable_paused(durable)
        if accepted.phase is not RunPhase.PAUSED:
            raise RuntimeError("lifecycle refused durable Pause")
        self._runtime.select_acquisition()
        return durable

    def resume(self) -> DisplaySelection | ResumeFailed:
        identity = self._require_identity()
        requested = self._lifecycle.resume_requested(
            ResumeRequested(identity)
        )
        if requested.phase is not RunPhase.RESUMING:
            raise RuntimeError("lifecycle refused Resume")
        try:
            self._executor.resume(identity)
        except BaseException as error:
            return self._command_failure(
                identity,
                error,
                ResumeFailed,
                self._lifecycle.resume_failed,
                RunPhase.PAUSED,
                "resume",
            )
        accepted = self._lifecycle.resumed(Resumed(identity))
        if accepted.phase is not RunPhase.RUNNING:
            raise RuntimeError("lifecycle refused resumed acquisition")
        finished = self._invalidate_browse_request()
        self._runtime.invalidate_browse()
        return (
            self._runtime.selection
            if finished
            else self._runtime.select_acquisition()
        )

    def stop(self):
        identity = self._runtime.run_identity
        if identity is None:
            identity = self._lifecycle.active_run_identity
        if type(identity) is not RunIdentity:
            raise RuntimeError("no acquisition identity")
        result = self._lifecycle.stop_requested(StopRequested(identity))
        if result.run_identity is identity:
            finished = self._invalidate_browse_request()
            self._runtime.invalidate_browse()
            selection = self._runtime.selection
            if (
                not finished
                and selection is not None
                and selection.kind is ContextKind.BROWSE
            ):
                self._runtime.select_acquisition()
            self._executor.stop(identity)
        return result

    def select_acquisition(self) -> DisplaySelection:
        return self._runtime.select_acquisition()

    def select_browse(self) -> DisplaySelection:
        return self._runtime.select_browse()

    def select_browser_target(self, identifier: str) -> bool:
        return self._runtime.select_browser_target(identifier)

    def project_request(
        self,
        frame: object,
        *,
        require_complete: bool = True,
    ) -> ProjectionRequest:
        if self._closed:
            raise RuntimeError("no display selection")
        return self._runtime.project_request(
            frame,
            require_complete=require_complete,
        )

    def resolve_projection(
        self, request: ProjectionRequest
    ) -> StandardDisplayPayload | None:
        return self._runtime.resolve_projection(
            self._projection, request, self._browse_hydration_owner
        )

    def project(self, frame: object) -> StandardDisplayPayload | None:
        return self.resolve_projection(self.project_request(frame))

    def project_navigation(
        self,
        *,
        preferences: ScientificPreferences | None = None,
        processing_mode: str = "Int 2D",
        live_update: bool = False,
    ) -> tuple[StandardDisplayPayload, ...]:
        if self._closed:
            return ()
        return self._runtime.project_navigation(
            self._projection,
            preferences=preferences,
            processing_mode=processing_mode,
            live_update=live_update,
            browse_hydration_owner=self._browse_hydration_owner,
        )

    def commit_navigation_projection(
        self,
        presented_frames: tuple[DisplayFrameKey, ...],
    ) -> bool:
        if self._closed:
            return False
        return self._runtime.commit_navigation_projection(
            presented_frames
        )

    def qualify_display_event(
        self, event: StandardRunEvent
    ) -> StandardDisplayPayload | None:
        if self._closed:
            return None
        return self._runtime.qualify_display_event(
            self._projection, event, self._browse_hydration_owner
        )

    def begin_browse(self, source_path: str) -> BrowseLoadRequest:
        if self._cleanup_receipt is not None:
            raise RuntimeError("Browse cleanup remains pending")
        if (
            self._closed
            or self._close is not None
            or not self._browse_lifecycle_admissible()
            or type(source_path) is not str
            or not source_path
        ):
            raise RuntimeError("Browse is not allowed in the current lifecycle")
        generation = self._load_generation + 1
        request = BrowseLoadRequest(
            new_context_token(ContextKind.BROWSE),
            generation,
            source_path,
        )
        accepted = self._browse_loader.begin(request)
        if accepted is not request:
            self._retain_cancel(request)
            raise RuntimeError("Browse loader did not accept exact request")
        self._load_generation = generation
        browse = self._runtime.browse_context
        if browse is not None:
            hold_presentation = (
                self._runtime.selection is not None
                and self._runtime.selection.names(browse)
            )
            receipt = self._release_browse(browse)
            if (
                type(receipt) is not BrowseCleanupReceipt
                or receipt.request is not browse.load_request
                or receipt.cleanup_status is not CleanupStatus.CLEANED
            ):
                self._retain_cancel(request)
                raise RuntimeError("previous Browse cleanup is pending")
            if hold_presentation:
                self._runtime.begin_replacement(request)
            else:
                self._runtime.clear_browse(select_acquisition=False)
        elif (
            self._browse_request is not None
            and self._runtime.selection is not None
            and self._runtime.selection.kind is ContextKind.BROWSE
            and not (
            self._runtime.retarget_replacement(
                self._browse_request, request
            )
            )
        ):
            self._retain_cancel(request)
            self._browse_request = None
            raise RuntimeError("Browse replacement identity drifted")
        self._browse_request = request
        return request

    def poll_browse(self):
        if self._close is not None:
            return None
        cleanup = self._cleanup_receipt
        if cleanup is not None:
            request = cleanup.request
            assert request is not None
            receipt = self._retain_cancel(request)
            if receipt.cleanup_status is not CleanupStatus.CLEANED:
                return None
            self._runtime.finish_replacement(request)
            return receipt
        request = self._browse_request
        if request is None:
            return None
        outcome = self._browse_loader.poll(request)
        if (
            type(outcome) is not BrowseLoadOutcome
            or outcome.request is not request
            or not self._browse_loader.owns_outcome(outcome)
        ):
            if (
                outcome is None
                and not self._browse_loader.owns_request(request)
            ):
                self._runtime.finish_replacement(request)
                self._browse_request = None
            return None
        admissible = (
            not self._closed
            and self._close is None
            and self._browse_lifecycle_admissible()
        )
        if not admissible:
            self._invalidate_browse_request()
            return None
        candidate = self._browse_loader.context_for_outcome(outcome)
        if outcome.status is BrowseLoadStatus.READY and (
            type(candidate) is not BrowseContext
            or candidate.operation is not request
            or candidate.load_request is not request
            or not candidate.matches(
                request.token, request.load_generation
            )
        ):
            self._browse_request = None
            receipt = self._retain_cancel(request)
            if receipt.cleanup_status is CleanupStatus.CLEANED:
                self._runtime.finish_replacement(request)
            return None
        prior = self._runtime.browse_context
        if outcome.status is BrowseLoadStatus.READY and prior is not None:
            receipt = self._release_browse(prior)
            if (
                type(receipt) is not BrowseCleanupReceipt
                or receipt.request is not prior.load_request
                or receipt.cleanup_status is not CleanupStatus.CLEANED
            ):
                self._invalidate_browse_request()
                return None
        context = self._browse_loader.consume(outcome)
        if outcome.status is not BrowseLoadStatus.READY:
            self._runtime.finish_replacement(request)
            self._browse_request = None
            return outcome
        if (
            context is not candidate
            or type(context) is not BrowseContext
        ):
            return None
        acquisition = self._runtime.acquisition_context
        display = (
            None if acquisition is None else acquisition.publication_store
        )
        owner = _BrowseHydrationOwner(
            context,
            borrowed_transport=getattr(display, "transport", None),
        )
        self._runtime.adopt_browse(context, request)
        self._browse_hydration_owner = owner
        self._browse_request = None
        return outcome

    def close(self) -> BrowseCleanupReceipt:
        if self._closed:
            assert self._close is not None
            return self._close
        browse = self._runtime.browse_context
        pending = self._close or self._cleanup_receipt
        expected = pending.request if pending is not None else (
            self._browse_request or (None if browse is None else browse.load_request)
        )
        if self._close is None:
            self._set_close(expected)
        failures = ()
        loader_receipt = self._browse_loader.close(expected)
        if (
            type(loader_receipt) is not BrowseCleanupReceipt
            or loader_receipt.cleanup_status is not CleanupStatus.CLEANED
            or loader_receipt.request is not expected
        ):
            if (
                type(loader_receipt) is BrowseCleanupReceipt
                and loader_receipt.request is expected
            ):
                failures = loader_receipt.cleanup_failures
            return self._set_close(expected, failures)
        if browse is not None:
            released = self._release_browse(browse)
            if (
                type(released) is BrowseCleanupReceipt
                and released.request is browse.load_request
                and released.cleanup_status is CleanupStatus.CLEANED
            ):
                self._runtime.clear_browse(select_acquisition=False)
            else:
                if (
                    type(released) is BrowseCleanupReceipt
                    and released.request is browse.load_request
                ):
                    failures = released.cleanup_failures
                return self._set_close(expected, failures)
        self._browse_request = None
        self._cleanup_receipt = None
        self._runtime.close_selection()
        self._close = BrowseCleanupReceipt(
            expected, CleanupStatus.CLEANED
        )
        return self._close

    def _set_close(
        self, request, failures=()
    ) -> BrowseCleanupReceipt:
        self._close = BrowseCleanupReceipt(
            request,
            CleanupStatus.CLEANUP_PENDING,
            failures,
        )
        return self._close

    def _command_failure(
        self,
        identity: RunIdentity,
        error: BaseException,
        event_type,
        recover,
        recovered_phase: RunPhase,
        operation: str,
    ):
        compensation = (
            error.diagnostics
            if type(error) is CommandCompensationFailure
            else None
        )
        failed = event_type(
            identity,
            (
                compensation[0]
                if compensation is not None
                else detach_exception(error, f"context.{operation}")
            ),
            None if compensation is None else compensation[1],
        )
        result = (
            self._lifecycle.fatal(FatalExecution(identity))
            if compensation is not None
            else recover(failed)
        )
        expected = (
            RunPhase.FAILED
            if compensation is not None
            else recovered_phase
        )
        if result.phase is not expected:
            raise RuntimeError(
                f"lifecycle refused failed {operation} recovery"
            )
        return failed

    def _require_identity(self) -> RunIdentity:
        if self._closed:
            raise RuntimeError("no acquisition identity")
        return self._runtime.require_identity()

    def _browse_lifecycle_admissible(self) -> bool:
        phase = self._lifecycle.phase
        return (
            phase in {RunPhase.IDLE, RunPhase.PAUSED}
            or (
                phase is RunPhase.FAILED
                and self._lifecycle.reset_permitted
            )
        )

    def _invalidate_browse_request(self) -> bool:
        request = self._browse_request
        self._browse_request = None
        if request is None:
            return False
        finished = self._runtime.finish_replacement(request)
        self._retain_cancel(request)
        return finished

    def _release_browse(
        self, browse: BrowseContext
    ) -> BrowseCleanupReceipt:
        owner = self._browse_hydration_owner
        receipt = release_browse(self._browse_loader, browse, owner)
        if (
            type(receipt) is BrowseCleanupReceipt
            and receipt.request is browse.load_request
            and receipt.cleanup_status is CleanupStatus.CLEANED
            and type(owner) is _BrowseHydrationOwner
            and owner._owns(browse)
        ):
            self._browse_hydration_owner = None
        return receipt

    def _retain_cancel(self, request: BrowseLoadRequest) -> BrowseCleanupReceipt:
        receipt = self._browse_loader.cancel(request)
        if (
            type(receipt) is not BrowseCleanupReceipt
            or receipt.request is not request
        ):
            receipt = BrowseCleanupReceipt(request, CleanupStatus.CLEANUP_PENDING)
        self._cleanup_receipt = None if receipt.cleanup_status is CleanupStatus.CLEANED else receipt
        return receipt

__all__ = ["ContextController"]
