"""Qt-free command owner for acquisition and processed browse contexts."""

from __future__ import annotations

from dataclasses import replace
import os
from threading import RLock

from xdart.modules.display_context import (
    AcquisitionContext, BrowseContext, ContextKind, DisplaySelection,
    Prepared2DCatalogCommit, Prepared2DFrameCommit,
    Viewer2DCatalogHydrationRequest, Viewer2DCleanupReceipt,
    Viewer2DCleanupState, Viewer2DCommitGate, Viewer2DContext,
    Viewer2DDisposal, Viewer2DFrameHydrationRequest,
    Viewer2DReadActivation, Viewer2DReadBudgetReceipt,
    Viewer2DReceiptPhase, Viewer2DRendererClearReceipt,
    Viewer2DRendererClearRequest, Viewer2DState, new_context_token,
)
from xrd_tools.io.viewer_2d import (
    CATALOG_RESERVATION, Viewer2DFormatPolicy,
    viewer_2d_memory_ledger, viewer_2d_selected_ledger,
)
from xrd_tools.session.hydration import (
    HydrationCompletion, HydrationOutcome, HydrationPurpose,
    HydrationReadKey, HydrationScope, HydrationToken,
)

from .acquisition_runtime import (
    CommandCompensationFailure,
    TerminalPauseFailure,
)
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
from .hydration_transport import HydrationTransport


class _TwoDViewerOwner:
    __slots__ = ("controller", "context", "policy", "provider", "catalog",
                 "frame", "receipt", "request", "request_token", "loading",
                 "diagnostic", "clear_request", "changed", "latest_label",
                 "cleanup_token")

    def __init__(self, controller) -> None:
        self.controller = controller
        self.context = self.policy = self.provider = self.catalog = None
        self.frame = self.receipt = self.request = self.request_token = None
        self.loading = self.changed = False
        self.diagnostic = ""
        self.clear_request = self.latest_label = self.cleanup_token = None

    def _owns(self, request) -> bool:
        context = self.context
        return (context is not None and request is not None and request is self.request
                and request.token is self.request_token and request.port is self
                and request.policy is self.policy
                and request.commit_gate is context.commit_gate
                and request.read_key.scope.context_token == context.context_token)

    def activate(self, request):
        with self.controller._viewer_2d_lock:
            context = self.context
            selection = self.controller._runtime.selection
            catalog_request = type(request) is Viewer2DCatalogHydrationRequest
            frame_request = type(request) is Viewer2DFrameHydrationRequest
            if (not self.loading or not (catalog_request or frame_request)
                    or not self._owns(request)
                    or request.read_key.scope.epoch != context.commit_gate.epoch
                    or context.commit_gate.cancelled
                    or catalog_request and (
                        context.state is not Viewer2DState.CATALOG_LOADING
                        or self.receipt is not None
                        or request.generation != context.generation
                        or request.token.presentation_generation != context.generation)
                    or frame_request and (
                        context.state is not Viewer2DState.FRAME_LOADING
                        or request.catalog is not self.catalog
                        or type(self.receipt) is not Viewer2DReadBudgetReceipt
                        or self.receipt.phase is not Viewer2DReceiptPhase.CATALOG_R
                        or request.receipt_identity is not self.receipt.identity
                        or selection is None or selection.kind is not ContextKind.VIEWER_2D
                        or selection.context_token != context.context_token
                        or request.generation != selection.display_generation
                        or request.token.presentation_generation != selection.display_generation)):
                return Viewer2DReadActivation(request.token, None, None, False,
                                              "stale 2D Viewer request")
            if type(request) is Viewer2DCatalogHydrationRequest:
                receipt = Viewer2DReadBudgetReceipt(
                    object(), viewer_2d_memory_ledger(1, 1).budget,
                    CATALOG_RESERVATION, Viewer2DReceiptPhase.CATALOG_R, request.token)
            else:
                ledger = viewer_2d_selected_ledger(request.catalog, request.label)
                receipt = Viewer2DReadBudgetReceipt(
                    request.receipt_identity, ledger.budget, ledger.admission,
                    Viewer2DReceiptPhase.FRAME_A, request.token)
            self.receipt = receipt
            return Viewer2DReadActivation(request.token, receipt, receipt.phase)

    def dispose(self, disposal):
        with self.controller._viewer_2d_lock:
            if type(disposal) is not Viewer2DDisposal:
                return None
            request = disposal.request
            if not self._owns(request):
                return Viewer2DCleanupReceipt(disposal.token, Viewer2DCleanupState.CLEANED)
            activation = disposal.activation
            frame_request = type(request) is Viewer2DFrameHydrationRequest
            phase = (Viewer2DReceiptPhase.FRAME_A if frame_request else Viewer2DReceiptPhase.CATALOG_R)
            if (frame_request and request.catalog is not self.catalog
                    or self.context.state not in ((Viewer2DState.FRAME_LOADING
                        if frame_request else Viewer2DState.CATALOG_LOADING),
                        Viewer2DState.CLEANUP_PENDING)
                    or not activation.accepted or activation.token is not request.token
                    or type(self.receipt) is not Viewer2DReadBudgetReceipt
                    or activation.receipt is not self.receipt
                    or activation.phase is not phase or self.receipt.phase is not phase
                    or self.receipt.request_token is not request.token
                    or disposal.prepared is not None
                    and (disposal.prepared.request is not request or
                         disposal.prepared.activation is not activation)):
                return Viewer2DCleanupReceipt(
                    disposal.token, Viewer2DCleanupState.CLEANUP_PENDING)
            state = (Viewer2DState.CLEANUP_PENDING if self.context.state
                     is Viewer2DState.CLEANUP_PENDING else Viewer2DState.READY)
            candidate_context = replace(self.context, state=state)
            if not self.controller._runtime.replace_viewer_2d_context(candidate_context):
                return Viewer2DCleanupReceipt(
                    disposal.token, Viewer2DCleanupState.CLEANUP_PENDING)
            if (type(request) is Viewer2DFrameHydrationRequest
                    and self.catalog is not None):
                receipt = disposal.activation.receipt
                candidate_receipt = Viewer2DReadBudgetReceipt(
                    receipt.identity, receipt.capacity, CATALOG_RESERVATION,
                    Viewer2DReceiptPhase.CATALOG_R)
                self.controller._runtime.clear_viewer_2d()
                self.receipt = candidate_receipt
                self.frame = None
            else:
                self.receipt = None
            self.context = candidate_context
            return Viewer2DCleanupReceipt(disposal.token, Viewer2DCleanupState.CLEANED)

    def commit(self, prepared):
        controller = self.controller
        with controller._viewer_2d_lock:
            prepared_type = type(prepared)
            request = getattr(prepared, "request", None)
            context = self.context
            if (not self.loading or prepared_type not in (
                    Prepared2DCatalogCommit, Prepared2DFrameCommit)
                    or not self._owns(request)
                    or request.read_key.scope.epoch != context.commit_gate.epoch
                    or not prepared.activation.accepted
                    or prepared.activation.token is not request.token
                    or type(self.receipt) is not Viewer2DReadBudgetReceipt
                    or prepared.activation.receipt is not self.receipt
                    or prepared.activation.phase is not self.receipt.phase
                    or self.receipt.request_token is not request.token
                    or request.token.presentation_generation != request.generation):
                return HydrationOutcome.OWNER_MISMATCH
            if prepared_type is Prepared2DCatalogCommit:
                if (context.state is not Viewer2DState.CATALOG_LOADING
                        or self.receipt.phase is not Viewer2DReceiptPhase.CATALOG_R):
                    return HydrationOutcome.OWNER_MISMATCH
                candidate = replace(context, state=Viewer2DState.FRAME_LOADING)
                try:
                    controller._runtime.adopt_viewer_2d(candidate, prepared.catalog)
                except Exception:
                    return HydrationOutcome.OWNER_MISMATCH
                self.catalog, self.receipt, self.context = (
                    prepared.catalog, prepared.activation.receipt, candidate)
            else:
                current = controller._runtime.navigation.current
                selection = controller._runtime.selection
                if (context.state is not Viewer2DState.FRAME_LOADING
                        or request.catalog is not self.catalog
                        or self.receipt.phase is not Viewer2DReceiptPhase.FRAME_A
                        or request.receipt_identity is not self.receipt.identity
                        or selection is None or selection.kind is not ContextKind.VIEWER_2D
                        or selection.context_token != context.context_token
                        or request.generation != selection.display_generation
                        or current is None or current.local_frame_label != request.label
                        or self.latest_label is not None and self.latest_label != request.label):
                    return HydrationOutcome.OWNER_MISMATCH
                candidate = replace(context, state=Viewer2DState.READY)
                catalog = controller._runtime._viewer_2d_catalog
                if (catalog is not self.catalog
                        or prepared.frame.catalog_identity != catalog.catalog_identity
                        or prepared.frame.label != current.local_frame_label):
                    return HydrationOutcome.OWNER_MISMATCH
                active = prepared.activation.receipt
                receipt = Viewer2DReadBudgetReceipt(
                    active.identity, active.capacity, active.reserved,
                    Viewer2DReceiptPhase.FRAME_READY_A, prepared.request.token)
                self.frame, self.receipt, self.context = (
                    prepared.frame, receipt, candidate)
                controller._runtime._viewer_2d_frame = prepared.frame
                controller._runtime._viewer_2d = candidate
                self.changed = True
        return HydrationOutcome.HYDRATED

    def complete(self, completion):
        if type(completion) is not HydrationCompletion:
            return
        with self.controller._viewer_2d_lock:
            if completion.token is not self.request_token:
                return
            self.loading = False
            self.diagnostic = (
                "" if completion.outcome in {
                    HydrationOutcome.HYDRATED, HydrationOutcome.ALREADY_RESIDENT}
                else completion.diagnostic or "2D Viewer read failed")
            self.changed = True


class ContextController:
    """Coordinate commands around one exact context/navigation runtime."""

    def __init__(self, *, lifecycle, executor, browse_loader,
                 projection: ContextProjection) -> None:
        self._lifecycle = lifecycle
        self._executor = executor
        self._browse_loader = browse_loader
        self._projection = projection
        self._runtime = _ContextRuntime()
        self._viewer_2d_lock = RLock()
        self._viewer_2d = _TwoDViewerOwner(self)
        self._viewer_2d_standalone: HydrationTransport | None = None
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
    def viewer_2d_context(self):
        return self._viewer_2d.context
    @property
    def viewer_2d_frame(self):
        return self._viewer_2d.frame

    @property
    def viewer_2d_loading(self) -> bool:
        return self._viewer_2d.loading

    @property
    def viewer_2d_diagnostic(self) -> str:
        return self._viewer_2d.diagnostic

    @property
    def viewer_2d_cleanup_pending(self) -> bool:
        owner = self._viewer_2d
        return bool(owner.clear_request is not None or
                    owner.context is not None and
                    owner.context.state is Viewer2DState.CLEANUP_PENDING or
                    owner.cleanup_token is not None or
                    self._viewer_2d_standalone is not None and
                    owner.context is None)

    @property
    def viewer_2d_owned(self) -> bool:
        return self.viewer_2d_context is not None or self.viewer_2d_cleanup_pending

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
        return not self._closed and browse_preview_polling_needed(
            self._runtime, self._browse_hydration_owner
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
        if self._runtime.acquisition_context is None:
            return durable
        self._runtime.select_acquisition()
        return durable

    def resume(self) -> DisplaySelection | ResumeFailed | None:
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
            else (self._runtime.select_acquisition()
                  if self._runtime.acquisition_context is not None else None)
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
                and self._runtime.acquisition_context is not None
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

    def open_viewer_2d(self, path: str):
        if (type(path) is not str or not path or len(os.fsencode(path)) > 4096
                or self._closed or self._close is not None
                or self.viewer_2d_cleanup_pending
                or not self._viewer_2d_admissible()):
            raise RuntimeError("2D Viewer is not allowed in the current lifecycle")
        if self._viewer_2d.context is not None:
            if self._viewer_2d.frame is not None:
                raise RuntimeError("2D Viewer renderer clear is required")
            if not self.close_viewer_2d():
                raise RuntimeError("2D Viewer cleanup remains pending")
        if self._browse_request is not None:
            self._invalidate_browse_request()
        if self._cleanup_receipt is not None:
            cleanup = self._retain_cancel(self._cleanup_receipt.request)
            if cleanup.cleanup_status is not CleanupStatus.CLEANED:
                raise RuntimeError("Browse cleanup remains pending")
        self._release_browse_for_viewer()
        provider = self._viewer_2d_provider()
        generation = self._runtime._display_generation + 1
        context = Viewer2DContext(new_context_token(ContextKind.VIEWER_2D),
                                  generation, path, Viewer2DCommitGate(),
                                  Viewer2DState.CATALOG_LOADING)
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            owner.context, owner.policy, owner.provider = (
                context, Viewer2DFormatPolicy(), provider)
            self._runtime._viewer_2d = context
        return self._submit_viewer_2d_catalog()

    def select_viewer_2d_frame(self, label: int) -> bool:
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if owner.context is None or owner.catalog is None:
                return False
            current = self._runtime.navigation.current
            if (current is not None and current.local_frame_label == label
                    and owner.frame is not None):
                return True
            if owner.frame is not None:
                return False
            if owner.loading:
                accepted = type(label) is int and label in owner.catalog.frame_labels
                if accepted:
                    owner.latest_label = label
                return accepted
            if owner.clear_request is not None:
                return False
            try:
                self._runtime.select_viewer_2d(
                    label, advance=owner.receipt.request_token is None)
            except RuntimeError:
                return False
        return self._submit_viewer_2d_frame(label, preselected=True) is not None

    def poll_viewer_2d(self) -> bool:
        follow = None
        with self._viewer_2d_lock:
            changed = self._viewer_2d.changed
            self._viewer_2d.changed = False
            owner = self._viewer_2d
            if not owner.loading:
                follow = owner.latest_label
                if (follow is None and owner.catalog is not None and
                        type(owner.request) is Viewer2DCatalogHydrationRequest):
                    follow = owner.catalog.frame_labels[0]
                owner.latest_label = None
        if follow is not None:
            if self._viewer_2d.frame is None:
                self._submit_viewer_2d_frame(follow)
            return True
        return changed

    def begin_viewer_2d_renderer_clear(self):
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if owner.clear_request is not None:
                return owner.clear_request
            if (owner.context is None or owner.catalog is None or
                    owner.context.state is Viewer2DState.CLEANUP_PENDING
                    and owner.frame is None):
                return None
            selection = self._runtime.selection
            if (selection is None or selection.kind is not ContextKind.VIEWER_2D
                    or selection.context_token != owner.context.context_token):
                return None
            stamp = self._runtime._display_generation + 1
            request = Viewer2DRendererClearRequest(
                owner.context.context_token, stamp, owner.catalog.catalog_identity,
                None if owner.frame is None else owner.frame.label)
            candidate = replace(owner.context, state=Viewer2DState.CLEANUP_PENDING)
            if not self._runtime.replace_viewer_2d_context(candidate):
                return None
            epoch = owner.context.commit_gate.advance()
            self._runtime._display_generation = stamp
            self._runtime._selection = replace(
                selection, owner=replace(selection.owner, epoch=epoch),
                display_generation=stamp)
            owner.context, owner.clear_request = candidate, request
            return request

    def acknowledge_viewer_2d_renderer_clear(self, receipt) -> bool:
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if (type(receipt) is not Viewer2DRendererClearReceipt or not receipt.cleared
                    or receipt.request is not owner.clear_request):
                return False
            prior = owner.receipt
            if prior.phase is Viewer2DReceiptPhase.FRAME_A:
                candidate_receipt = prior
                state = Viewer2DState.CLEANUP_PENDING
            else:
                candidate_receipt = Viewer2DReadBudgetReceipt(
                    prior.identity, prior.capacity, CATALOG_RESERVATION,
                    Viewer2DReceiptPhase.CATALOG_R, prior.request_token)
                state = Viewer2DState.READY
            candidate = replace(owner.context, state=state)
            self._runtime._viewer_2d_frame = None
            if not self._runtime.replace_viewer_2d_context(candidate):
                return False
            owner.context, owner.receipt = candidate, candidate_receipt
            owner.frame = owner.clear_request = None
            return True

    def close_viewer_2d(self) -> bool:
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            context, provider = owner.context, owner.provider
            if (context is not None and
                    (owner.frame is not None or owner.clear_request is not None)):
                owner.context = replace(context, state=Viewer2DState.CLEANUP_PENDING)
                self._runtime.replace_viewer_2d_context(owner.context)
                return False
            if context is not None:
                context.commit_gate.cancel()
        if context is None:
            return self._retire_viewer_2d_standalone()
        try:
            standalone = provider is self._viewer_2d_standalone
            cancel = provider.cancel_gate if standalone else provider.cancel_viewer_2d
            blocked = (provider.blocked_cleanup_token if standalone
                       else provider.viewer_2d_blocked_cleanup_token)
            retry = (provider.retry_blocked_cleanup if standalone
                     else provider.retry_viewer_2d_blocked_cleanup)
            retains = provider.retains_gate if standalone else provider.viewer_2d_retains_gate
            cancel(context.commit_gate)
            cleanup_token = owner.cleanup_token or blocked(owner.request_token)
            if cleanup_token is not None:
                with self._viewer_2d_lock:
                    owner.cleanup_token = cleanup_token
                cleanup = retry(cleanup_token)
                if getattr(cleanup, "state", None) is not Viewer2DCleanupState.CLEANED:
                    return self._hold_viewer_2d_cleanup()
                with self._viewer_2d_lock:
                    owner.cleanup_token = None
            if retains(context.commit_gate):
                return self._hold_viewer_2d_cleanup()
        except Exception:
            return self._hold_viewer_2d_cleanup()
        if not self._retire_viewer_2d_standalone():
            return self._hold_viewer_2d_cleanup()
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            self._runtime.clear_viewer_2d(release=True)
            owner.context = owner.policy = owner.provider = owner.catalog = None
            owner.frame = owner.receipt = owner.request = owner.request_token = None
            owner.clear_request = owner.latest_label = None
            owner.loading = owner.changed = False
            owner.diagnostic = ""
        return True

    def _hold_viewer_2d_cleanup(self) -> bool:
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if owner.context is not None:
                owner.context = replace(owner.context,
                                        state=Viewer2DState.CLEANUP_PENDING)
                self._runtime.replace_viewer_2d_context(owner.context)
        return False

    def _viewer_2d_admissible(self) -> bool:
        phase = self._lifecycle.phase
        return (self._lifecycle.active_run_identity is None
                and self._lifecycle.attempt_run_identity is None
                and (phase is RunPhase.IDLE or phase is RunPhase.FAILED
                     and self._lifecycle.reset_permitted))

    def _viewer_2d_provider(self):
        acquisition = self._runtime.acquisition_context
        if acquisition is not None:
            provider = acquisition.publication_store
            methods = ("submit_viewer_2d", "cancel_viewer_2d",
                       "viewer_2d_retains_gate", "viewer_2d_blocked_cleanup_token",
                       "retry_viewer_2d_blocked_cleanup")
            if not all(callable(getattr(provider, name, None)) for name in methods):
                raise RuntimeError("acquisition has no 2D Viewer provider")
            return provider
        if self._viewer_2d_standalone is None:
            self._viewer_2d_standalone = HydrationTransport(
                lambda _prepared: HydrationOutcome.FAILED,
                lambda _request: (None, None))
        return self._viewer_2d_standalone

    def _viewer_2d_request(self, label):
        owner, context = self._viewer_2d, self._viewer_2d.context
        scope = HydrationScope(
            context.context_token, "viewer-2d", "viewer-2d", context.commit_gate.epoch)
        selection = self._runtime.selection
        generation = (context.generation if label == "catalog"
                      else selection.display_generation)
        key = HydrationReadKey(scope, "viewer-2d", label, HydrationPurpose.PREVIEW)
        token = HydrationToken(key, generation)
        if label == "catalog":
            return Viewer2DCatalogHydrationRequest(
                context.original_path, owner.policy, generation,
                context.commit_gate, owner, key, token)
        return Viewer2DFrameHydrationRequest(
            label, generation, context.commit_gate, owner, owner.catalog,
            owner.receipt.identity, owner.policy, key, token)

    def _viewer_2d_submit(self, request):
        provider = self._viewer_2d.provider
        try:
            submit = (provider.submit if provider is self._viewer_2d_standalone
                      else provider.submit_viewer_2d)
            token = submit(request)
        except Exception:
            token = None
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if owner.request is not request:
                return None
            if token is not request.token:
                activated = (type(owner.receipt) is Viewer2DReadBudgetReceipt
                             and owner.receipt.request_token is request.token)
                catalog_request = type(request) is Viewer2DCatalogHydrationRequest
                state = (Viewer2DState.CLEANUP_PENDING
                         if activated or catalog_request else Viewer2DState.READY)
                candidate = replace(owner.context, state=state)
                if self._runtime.replace_viewer_2d_context(candidate):
                    owner.context = candidate
                if activated or catalog_request:
                    owner.request = request
                    owner.request_token = request.token
                else:
                    prior = owner.receipt
                    owner.receipt = Viewer2DReadBudgetReceipt(
                        prior.identity, prior.capacity, CATALOG_RESERVATION,
                        Viewer2DReceiptPhase.CATALOG_R)
                    owner.request = owner.request_token = None
                owner.loading = False
                owner.diagnostic = "2D Viewer transport refused request"
                owner.changed = True
                return None
        return request

    def _submit_viewer_2d_catalog(self):
        with self._viewer_2d_lock:
            request = self._viewer_2d_request("catalog")
            self._viewer_2d.request = request
            self._viewer_2d.request_token = request.token
            self._viewer_2d.loading = True
        return self._viewer_2d_submit(request)

    def _submit_viewer_2d_frame(self, label, *, preselected=False):
        with self._viewer_2d_lock:
            owner = self._viewer_2d
            if owner.loading:
                owner.latest_label = label
                return owner.request
            if not preselected:
                try:
                    self._runtime.select_viewer_2d(
                        label, advance=owner.receipt.request_token is None)
                except RuntimeError:
                    return None
            owner.context = replace(owner.context, state=Viewer2DState.FRAME_LOADING)
            if not self._runtime.replace_viewer_2d_context(owner.context):
                return None
            request = self._viewer_2d_request(label)
            owner.request, owner.request_token = request, request.token
            owner.loading, owner.diagnostic = True, ""
        return self._viewer_2d_submit(request)

    def _retire_viewer_2d_standalone(self) -> bool:
        provider = self._viewer_2d_standalone
        try:
            retired = provider is None or provider.retire(join_timeout=0.0)
        except Exception:
            return False
        if retired:
            self._viewer_2d_standalone = None
        return retired

    def _release_browse_for_viewer(self) -> None:
        browse = self._runtime.browse_context
        if browse is None:
            return
        receipt = self._release_browse(browse)
        if (type(receipt) is not BrowseCleanupReceipt
                or receipt.cleanup_status is not CleanupStatus.CLEANED):
            raise RuntimeError("Browse cleanup remains pending")
        self._runtime.clear_browse(select_acquisition=False)

    def begin_browse(self, source_path: str) -> BrowseLoadRequest:
        if self.viewer_2d_owned:
            raise RuntimeError("2D Viewer cleanup remains pending")
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
            receipt = self._release_browse(
                browse, preserve_pending_repaint=True
            )
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
            receipt = self._release_browse(
                prior, preserve_pending_repaint=True
            )
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
        if not self.close_viewer_2d():
            return self._set_close(expected)
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
        terminal = type(error) is TerminalPauseFailure
        failed = event_type(
            identity,
            (
                compensation[0]
                if compensation is not None
                else error.diagnostic
                if terminal
                else detach_exception(error, f"context.{operation}")
            ),
            None if compensation is None else compensation[1],
        )
        result = (
            self._lifecycle.fatal(FatalExecution(identity))
            if compensation is not None or terminal
            else recover(failed)
        )
        expected = (
            RunPhase.FAILED
            if compensation is not None or terminal
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
        identity = self._lifecycle.active_run_identity
        return identity if type(identity) is RunIdentity else self._runtime.require_identity()

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
        self, browse: BrowseContext, *, preserve_pending_repaint=False
    ) -> BrowseCleanupReceipt:
        owner = self._browse_hydration_owner
        receipt = release_browse(
            self._browse_loader, browse, owner,
            preserve_pending_repaint=preserve_pending_repaint,
        )
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
