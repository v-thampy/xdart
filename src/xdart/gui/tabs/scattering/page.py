"""Public composition root for the context-qualified scattering workspace."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import os
from typing import Any, Callable
import weakref

from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
    RunIntentSnapshot,
)
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    is_single_image_spec,
)

from .advanced_editor import AdvancedSettingsDialog
from .adapters.browse_loader import BrowseLoader
from .browser_catalog import (
    BrowserCatalogEntry,
    enumerate_processed_artifacts,
    processed_directory,
)
from .contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    RunExecutorPort,
    SourceCountScope,
    SourceObservation,
    SourceObservationRequest,
    SourcePort,
    SourceSelection,
)
from .context_controller import ContextController
from .context_projection import ContextProjection
from .controls_projection import (
    AdvancedSettingsValues,
    EditNoChange,
    EditRefusal,
    OUTPUT_MODE,
    SOURCE_DIRECTORY,
    SOURCE_FILE,
    SOURCE_TYPE,
    reduce_advanced_settings,
    reduce_control_edit,
    reduce_source_selection,
    project_controls,
    source_mode,
)
from .controls_readiness import (
    ControlsReadinessProjection,
    SectionHeaderProjection,
)
from .coordinator import ScatteringCoordinator
from .detector_projection import detector_summary
from .display_values import (
    DisplayFrameKey,
    StandardEventKind,
    StandardRunEvent,
    standard_event_is_valid,
)
from .display_retirement import DisplayRetirementReceipt
from .events import (
    CleanupStatus,
    DurableFinal,
    ExecutionEnded,
    ExecutorClosed,
    FatalExecution,
    LifecycleStatus,
    OwnersClosed,
    RunIdentity,
    detached_exception_strings,
)
from .output_values import APPEND_UNAVAILABLE
from .shell_projection import (
    ScientificPreferences,
    share_plot_axis_for_image,
)
from .shell_values import (
    ProgressProjection,
    ShellCommand,
    ShellCommandKind,
)
from .shell_widgets import (
    experiment_header_projection,
    processing_header_projection,
    project_header_projection,
)
from .source_view import SourceStatusView
from .start_outcomes import (
    RecoveryFailure,
    StartCapture,
    StartClosed,
    StartFailed,
    StartLaunched,
    StartRecaptureRequired,
    StartRefusal,
    StartRefused,
    executor_closed_is_valid,
)
from .start_pipeline import StartPipeline
from .state_machine import RunPhase
from .workspace_shell import ScatteringWorkspaceShell


_LIVE_EVENT_DRAIN_INTERVAL_MS = 125
_BROWSER_CATALOG_REFRESH_INTERVAL_MS = 1500


@dataclass(frozen=True, slots=True)
class _ObservationOperation:
    request: SourceObservationRequest
    future: Future[object]
    preview: bool = False
    candidate_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class _BrowserCatalogOperation:
    token: int
    directory: str
    future: Future[object]


@dataclass(frozen=True, slots=True)
class _AdmissionPageOwner:
    token: AdmissionToken
    releasing: bool = False
    release_receipt: AdmissionReleased | None = None
    admission_receipt: AdmissionReceipt | None = None
    retirement_receipt: DisplayRetirementReceipt | None = None
    retirement_applied: bool = False


class ScatteringWorkspace(QtWidgets.QWidget):
    """One command owner around one passive shell and one context controller."""

    sourceSelectionRequested = QtCore.Signal(object)
    browseRequested = QtCore.Signal(object)
    noticeChanged = QtCore.Signal(str)
    _observationFinished = QtCore.Signal(object, object)
    _browserCatalogFinished = QtCore.Signal(object, object)

    def __init__(
        self,
        *,
        intents: RunIntentStore,
        lifecycle: ScatteringCoordinator,
        sources: SourcePort,
        executor: RunExecutorPort | None = None,
        browser_directory_chooser: (
            Callable[[str], str | None] | None
        ) = None,
        control_path_chooser: (
            Callable[[tuple[str, ...], str], str | None] | None
        ) = None,
        source_selection_chooser: (
            Callable[
                [SourceSelection | None, str | None],
                SourceSelection | None,
            ]
            | None
        ) = None,
        advanced_settings_editor: (
            Callable[
                [RunIntentSnapshot],
                AdvancedSettingsValues | None,
            ]
            | None
        ) = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("scatteringWorkspace")
        self._intents = intents
        self._lifecycle = lifecycle
        self._sources = sources
        self._run_executor = executor
        self._pipeline = (
            StartPipeline(
                intents=intents,
                lifecycle=lifecycle,
                sources=sources,
                executor=executor,
            )
            if executor is not None
            else None
        )
        self._observation_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1)
        )
        self._observation: _ObservationOperation | None = None
        self._source_observation: SourceObservation | None = None
        self._observation_token = 0
        self._browser_catalog_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1)
        )
        self._browser_catalog_operation: (
            _BrowserCatalogOperation | None
        ) = None
        self._browser_catalog_token = 0
        self._browser_catalog: tuple[BrowserCatalogEntry, ...] = ()
        self._browser_follow_identity: RunIdentity | None = None
        self._browser_seen_artifacts: set[str] = set()
        self._browser_transient_frame: DisplayFrameKey | None = None
        self._browser_transient_clear_token: int | None = None
        self._active_batch_mode = False
        self._run_frame_seen = False
        self._retain_outgoing_display = False
        self._batch_latest_frame: DisplayFrameKey | None = None
        self._browser_directory_chooser = (
            browser_directory_chooser
            if browser_directory_chooser is not None
            else self._choose_directory_dialog
        )
        self._control_path_chooser = control_path_chooser
        self._source_selection_chooser = source_selection_chooser
        self._advanced_dialog: AdvancedSettingsDialog | None = None
        self._advanced_settings_editor = (
            advanced_settings_editor
            if advanced_settings_editor is not None
            else self._show_advanced_settings_dialog
        )
        initial_intent = intents.snapshot().thaw()
        self._source_mode = source_mode(initial_intent.source_spec)
        self._source_history: dict[str, SourceSelection] = {}
        if initial_intent.source_spec is not None:
            self._source_history[self._source_mode] = (
                initial_intent.source_spec
            )
        self._browser_directory = processed_directory(
            initial_intent.save_path
        )
        self._browser_explicit_directory = False
        self._admission_state: _AdmissionPageOwner | None = None
        self._closing = False
        self._closed = False
        self._terminal_close: StartClosed | None = None
        self._close_base: StartClosed | None = None
        self._close_identity: RunIdentity | None = None
        self._close_executor: ExecutorClosed | None = None
        self._close_source_pending = False
        self._close_recovery_failures: tuple[RecoveryFailure, ...] = ()
        self._connections: list[
            tuple[Any, Callable[..., object]]
        ] = []

        self._shell_revision = 0
        self._preferences = ScientificPreferences()
        self._rendered_image_axis = self._preferences.image_axis
        self._detector_summary_key: tuple[str, str] | None = None
        self._detector_summary_text = "not configured"
        self._project_readiness_key: tuple[str, str] | None = None
        self._project_readiness = SectionHeaderProjection("")
        self._experiment_readiness_key: tuple[bool, str] | None = None
        self._experiment_readiness = SectionHeaderProjection("")
        self._controls_readiness = ControlsReadinessProjection()
        self._progress = ProgressProjection()
        self._notice_text = ""
        self._date_sorted = False
        self._auto_last = True

        self._context_projection = ContextProjection()
        self._browse_loader = BrowseLoader()
        self._context_controller = ContextController(
            lifecycle=lifecycle,
            executor=executor,
            browse_loader=self._browse_loader,
            projection=self._context_projection,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._shell = ScatteringWorkspaceShell(self)
        layout.addWidget(self._shell)
        self._source_status = SourceStatusView(self._shell.controls)
        self._shell.controls.set_source_widget(
            self._source_status,
            visible=False,
        )

        self._run_timer = QtCore.QTimer(self)
        self._run_timer.setInterval(_LIVE_EVENT_DRAIN_INTERVAL_MS)
        self._browser_catalog_timer = QtCore.QTimer(self)
        self._browser_catalog_timer.setInterval(
            _BROWSER_CATALOG_REFRESH_INTERVAL_MS
        )
        self._connect(
            self._shell.commandRequested, self._handle_shell_command
        )
        self._connect(
            self._source_status.chooseRequested,
            self._choose_source_selection,
        )
        self._connect(
            self._observationFinished, self._on_observation
        )
        self._connect(
            self._browserCatalogFinished,
            self._on_browser_catalog,
        )
        self._connect(self._run_timer.timeout, self._drain_executor)
        self._connect(
            self._browser_catalog_timer.timeout,
            self._poll_browser_catalog,
        )

        initial = self._intents.snapshot()
        self._refresh_shell()
        self._request_observation(initial)
        self._request_browser_catalog()
        self._browser_catalog_timer.start()

    @property
    def _admission(self) -> AdmissionToken | None:
        state = self._admission_state
        return None if state is None else state.token

    @_admission.setter
    def _admission(self, token: AdmissionToken | None) -> None:
        self._admission_state = (
            None if token is None else _AdmissionPageOwner(token)
        )

    def select_source(self, source: SourceSelection) -> None:
        if self._closing or self._closed:
            return
        snapshot = self._intents.snapshot()
        reduced = reduce_source_selection(snapshot, source)
        if isinstance(reduced, EditRefusal):
            self._notice(reduced.reason)
            self._refresh_shell()
            return
        if isinstance(reduced, EditNoChange):
            self._notice("")
            return
        result = self._intents.commit(
            reduced, expected_revision=snapshot.revision
        )
        if type(result) is IntentCommitAccepted:
            self._source_mode = source_mode(source)
            self._source_history[self._source_mode] = source
        self._notice(
            ""
            if type(result) is IntentCommitAccepted
            else "Edit superseded; review current value."
        )
        self._reconcile_snapshot(snapshot, result.snapshot)

    def close_workspace(self) -> StartClosed:
        terminal = self._terminal_close
        if terminal is not None:
            return terminal

        first = not self._closing
        if first:
            self._closing = True
            self._shell.browser.cancel_pending_frame_selection()
            self._run_timer.stop()
            self._browser_catalog_timer.stop()
            self._observation_token += 1
            self._cancel_observation()
            self._browser_catalog_token += 1
            self._cancel_browser_catalog()
            self._close_identity = (
                self._context_controller.run_identity
                or self._lifecycle.active_run_identity
                or self._lifecycle.attempt_run_identity
            )

        token = self._admission
        if token is not None and self._run_executor is not None:
            self._release_admission(token)
            self._refuse_preparing(token)
        admission_clean = self._admission is None

        browse = self._context_controller.close()
        base = self._close_base
        refresh_base = (
            base is None
            or base.lifecycle_result.status is LifecycleStatus.REJECTED
        )
        base_released_executor = False
        if refresh_base:
            if self._pipeline is None:
                result = self._lifecycle.close()
                base = StartClosed(result)
            elif base is None:
                base = self._pipeline.close()
                base_released_executor = True
            else:
                base = self._pipeline.retry_close_lifecycle(
                    self._close_identity
                )
            self._close_base = base
            if self._close_identity is None:
                self._close_identity = base.cleanup_identity
            recovery_failures = self._close_recovery_failures
            self._close_recovery_failures = recovery_failures + tuple(
                failure
                for failure in base.recovery_failures
                if failure not in recovery_failures
            )
            if any(
                failure.operation == "source.cancel"
                for failure in base.recovery_failures
            ):
                self._close_source_pending = True

        if (
            not first
            and self._close_source_pending
            and self._pipeline is not None
        ):
            source_status, source_failures = (
                self._pipeline.retry_close_source()
            )
            if source_failures:
                recovery_failures = self._close_recovery_failures
                self._close_recovery_failures = (
                    recovery_failures
                    + tuple(
                        failure
                        for failure in source_failures
                        if failure not in recovery_failures
                    )
                )
            if source_status is CleanupStatus.CLEANED:
                self._close_source_pending = False

        identity = self._close_identity
        executor_receipt = self._close_executor
        if (
            base_released_executor
            and identity is not None
            and base.cleanup_identity is identity
        ):
            executor_receipt = ExecutorClosed(
                identity,
                base.cleanup_status,
                base.primary,
                base.cleanup_failures,
            )
        elif identity is not None and (
            executor_receipt is None
            or executor_receipt.cleanup_status
            is not CleanupStatus.CLEANED
        ):
            if self._run_executor is not None:
                try:
                    candidate = self._run_executor.close(identity)
                except Exception:
                    candidate = None
                if executor_closed_is_valid(candidate, identity):
                    executor_receipt = candidate
        self._close_executor = executor_receipt

        executor_clean = (
            identity is None
            or (
                executor_receipt is not None
                and executor_receipt.cleanup_status
                is CleanupStatus.CLEANED
            )
        )
        components_clean = (
            admission_clean
            and browse.cleanup_status is CleanupStatus.CLEANED
            and executor_clean
            and not self._close_source_pending
            and all(
                failure.operation
                in {
                    "lifecycle.close",
                    "executor.close",
                    "source.cancel",
                }
                for failure in self._close_recovery_failures
            )
        )
        lifecycle_result = base.lifecycle_result
        if (
            components_clean
            and identity is not None
            and lifecycle_result.phase is RunPhase.STOPPING
            and self._pipeline is not None
        ):
            lifecycle_result = self._pipeline.owners_closed(
                OwnersClosed(identity)
            )
        all_clean = (
            components_clean
            and lifecycle_result.status
            in {LifecycleStatus.APPLIED, LifecycleStatus.SUPERSEDED}
            and lifecycle_result.phase is RunPhase.CLOSED
        )

        cleanup_failures = tuple(base.cleanup_failures)
        primary = base.primary
        if executor_receipt is not None:
            cleanup_failures += tuple(
                failure
                for failure in executor_receipt.cleanup_failures
                if failure not in cleanup_failures
            )
            if primary is None:
                primary = executor_receipt.primary
        cleanup_failures += tuple(
            failure
            for failure in browse.cleanup_failures
            if failure not in cleanup_failures
        )
        receipt = StartClosed(
            lifecycle_result,
            self._close_recovery_failures,
            (
                CleanupStatus.CLEANED
                if all_clean
                else CleanupStatus.CLEANUP_PENDING
            ),
            primary,
            cleanup_failures,
            identity,
        )
        if not all_clean:
            return receipt

        self._context_controller.release_acquisition(identity)
        self._finalize_close()
        self._terminal_close = receipt
        return receipt

    def _finalize_close(self) -> None:
        if self._closed:
            return
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._connections.clear()
        pool, self._observation_pool = self._observation_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        browser_pool, self._browser_catalog_pool = (
            self._browser_catalog_pool,
            None,
        )
        if browser_pool is not None:
            browser_pool.shutdown(wait=False, cancel_futures=True)
        self._source_status.close()
        self._closed = True

    def _handle_shell_command(self, command: object) -> None:
        if (
            self._closing
            or self._closed
            or type(command) is not ShellCommand
        ):
            return
        kind = command.kind
        if kind is ShellCommandKind.RUN_ACTION:
            self._run_action()
            return
        if kind is ShellCommandKind.STOP:
            self._stop_run()
            return
        if kind is ShellCommandKind.MENU:
            if command.value == "File:Open Folder":
                self._choose_browser_directory()
            return
        if kind is ShellCommandKind.REFRESH_BROWSER:
            self._request_browser_catalog()
            return
        if kind is ShellCommandKind.SELECT_SCAN:
            self._select_scan(command.value)
            return
        if kind in {
            ShellCommandKind.SELECT_FRAME,
            ShellCommandKind.HYDRATE_FRAME,
            ShellCommandKind.SELECT_BROWSER_FRAMES,
        }:
            if kind is not ShellCommandKind.SELECT_BROWSER_FRAMES:
                self._shell.browser.cancel_pending_frame_selection()
            self._select_frames(command)
            return
        if kind is ShellCommandKind.CONTROL_DRAFT:
            self._on_field_draft(command.path, command.value)
            return
        if kind is ShellCommandKind.CONTROL_EDIT:
            self._on_field_value(command.path, command.value)
            return
        if kind is ShellCommandKind.CONTROL_BROWSE:
            self._choose_control_path(command.path)
            return
        if kind is ShellCommandKind.CONTROL_ACTION:
            if command.value == "advanced_processing":
                self._edit_advanced_settings()
                return
            labels = {
                "reintegrate_1d": "Reintegrate 1D",
                "reintegrate_2d": "Reintegrate 2D",
                "calibrate": "Calibration",
                "make_mask": "Mask creation",
                "refine_geometry": "Geometry refinement",
            }
            label = labels.get(str(command.value), "Control action")
            self._notice(
                f"{label} is unavailable: no vNext operation service is "
                "mounted."
            )
            return
        if kind is ShellCommandKind.ANALYSIS_ACTION:
            self._notice(
                "Analysis action is unavailable: no vNext analysis service "
                "is mounted."
            )
            return
        if kind in {
            ShellCommandKind.SET_PROCESSING_MODE,
            ShellCommandKind.SET_BATCH,
            ShellCommandKind.SET_CORES,
            ShellCommandKind.SET_LIVE,
            ShellCommandKind.SET_OUTPUT_POLICY,
        }:
            self._edit_run_strip(kind, command.value)
            return
        if self._edit_scientific_preference(command):
            self._refresh_shell()

    def _run_action(self) -> None:
        phase = self._lifecycle.phase
        if phase is RunPhase.RUNNING:
            try:
                result = self._context_controller.pause()
            except Exception as error:
                self._error_notice("Pause failed", error)
            else:
                diagnostic = getattr(result, "diagnostic", None)
                self._notice(
                    "" if diagnostic is None else diagnostic.message
                )
                self._refresh_shell()
            return
        if phase is RunPhase.PAUSED:
            try:
                result = self._context_controller.resume()
            except Exception as error:
                self._error_notice("Resume failed", error)
            else:
                diagnostic = getattr(result, "diagnostic", None)
                self._notice(
                    "" if diagnostic is None else diagnostic.message
                )
                self._refresh_shell()
                self._ensure_timer()
            return
        if not self._commit_focused_control_edit_for_run():
            return
        self._begin_run()

    def _commit_focused_control_edit_for_run(self) -> bool:
        """Commit the one active editor before capturing the next run.

        ``ControlsPanelV2.current_form_edits()`` is diagnostic only.  The
        focused editor is the sole presentation value that can be newer than
        the revisioned intent when an action arrives before
        ``editingFinished``.
        """

        try:
            edit = self._shell.controls.focused_form_edit()
        except Exception as error:
            self._error_notice("Run edit capture failed", error)
            return False
        if edit is None:
            return True
        snapshot = self._intents.snapshot()
        reduced = reduce_control_edit(snapshot, edit.path, edit.value)
        if isinstance(reduced, EditRefusal):
            self._notice(reduced.reason)
            self._refresh_shell()
            return False
        if isinstance(reduced, EditNoChange):
            self._notice("")
            return True
        try:
            result = self._intents.commit(
                reduced, expected_revision=snapshot.revision
            )
        except Exception as error:
            self._error_notice("Run edit commit failed", error)
            return False
        if type(result) is IntentCommitAccepted:
            self._notice("")
            self._reconcile_snapshot(snapshot, result.snapshot)
            return True
        if type(result) is IntentRecaptureRequired:
            self._notice(
                "Run not started: edit superseded; review current value."
            )
            self._reconcile_snapshot(snapshot, result.snapshot)
        return False

    def _begin_run(self) -> None:
        pipeline = self._pipeline
        if self._closing or self._closed or pipeline is None:
            self._notice("Standard execution is unavailable.")
            return
        intent = self._intents.snapshot().thaw()
        if (
            type(intent.output_mode) is not str
            or intent.output_mode.strip().lower() != "overwrite"
        ):
            self._notice(APPEND_UNAVAILABLE)
            self._refresh_shell()
            return
        if self._admission is not None:
            released = self._release_admission(self._admission)
            if released.cleanup_status is not CleanupStatus.CLEANED:
                self._notice("Output cleanup remains pending.")
                self._refresh_shell()
                return
        if self._lifecycle.phase is RunPhase.FAILED:
            reset = self._lifecycle.reset()
            if reset.phase is not RunPhase.IDLE:
                self._notice("Standard cleanup remains pending.")
                self._refresh_shell()
                return
        if not self._start_permitted()[0]:
            self._refresh_shell()
            return
        capture = pipeline.begin()
        if not isinstance(capture, StartCapture):
            self._render_start_outcome(capture)
            return
        self._begin_admission(capture)

    def _begin_admission(self, capture: StartCapture) -> None:
        pipeline = self._pipeline
        executor = self._run_executor
        if pipeline is None:
            return
        if executor is None:
            self._render_start_outcome(
                pipeline.refuse(
                    capture, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            return
        try:
            token = executor.begin_admission(capture)
        except Exception as error:
            self._render_start_outcome(
                pipeline.refuse(
                    capture, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            self._error_notice("Output admission failed", error)
            return
        self._admission = token
        self._retain_outgoing_display = True
        self._progress = ProgressProjection(
            detail="Checking output targets…"
        )
        # Project the phase/readiness lock immediately, while retaining the
        # outgoing browser and scientific presentation until the new run's
        # first accepted frame.  This keeps Run-click feedback prompt without
        # restoring the retired Run-click display flicker.
        self._refresh_shell(preserve_display=True)
        self._ensure_timer()

    def _poll_admission(self) -> bool:
        state = self._admission_state
        token = None if state is None else state.token
        executor = self._run_executor
        pipeline = self._pipeline
        if token is None or executor is None or pipeline is None:
            return False
        if state is not None and state.releasing:
            self._release_admission(token)
            return True
        result = None if state is None else state.admission_receipt
        if result is None:
            try:
                result = executor.poll_admission(token)
            except Exception as error:
                result = AdmissionFailure(
                    token, f"{type(error).__name__}: {error}"
                )
        if result is None:
            return False
        if type(result) is AdmissionFailure:
            released = self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    token, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            self._notice(f"Output admission failed: {result.reason}")
            if released.cleanup_status is not CleanupStatus.CLEANED:
                self._notice("Output cleanup remains pending.")
            return True
        if type(result) is not AdmissionReceipt:
            self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    token, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            return True
        current = self._context_controller.run_identity
        if not _retirement_receipt_is_current(
            result.display_retirement, current
        ):
            # A clean receipt for "no display" (or for another run) cannot
            # authorize replacing the exact historical display still owned by
            # this page.  Retry is reserved for an exact proof whose Browse
            # cleanup is still finishing; an identity mismatch is terminal for
            # this admission attempt.
            self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    result, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            self._notice(
                "Output admission failed: prior display retirement was not "
                "proved."
            )
            return True
        if not self._context_controller.apply_display_retirement(
            result.display_retirement
        ):
            # Browse/display cleanup may complete one event-loop turn after
            # admission.  Retain the exact receipt and retry internally; this
            # transient must not consume the user's Run click or fail a run.
            self._admission_state = replace(
                state,
                admission_receipt=result,
                retirement_receipt=result.display_retirement,
            )
            self._progress = ProgressProjection(
                detail="Finishing prior display cleanup…"
            )
            self._ensure_timer()
            return False
        if self._intents.snapshot().revision != result.revision:
            self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    result, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            return True
        return not self._launch(token, result)

    def _launch(
        self, token: AdmissionToken, receipt: AdmissionReceipt
    ) -> bool:
        pipeline = self._pipeline
        if pipeline is None:
            return False
        outcome = pipeline.start(receipt)
        if isinstance(outcome, StartLaunched):
            self._admission = None
            self._begin_browser_follow(outcome.run_identity)
            self._active_batch_mode = outcome.configuration.batch_mode
            self._run_frame_seen = False
            self._batch_latest_frame = None
            self._browser_transient_frame = None
            self._browser_transient_clear_token = None
            self._progress = ProgressProjection(
                detail="Run started"
            )
            self._notice("")
            self._refresh_shell(preserve_display=True)
            self._ensure_timer()
            return True
        self._release_admission(token)
        if isinstance(outcome, StartRecaptureRequired):
            replacement = pipeline.recapture(outcome)
            if isinstance(replacement, StartCapture):
                outcome = pipeline.refuse(
                    replacement, StartRefusal.OUTPUT_PREFLIGHT
                )
            else:
                outcome = replacement
        self._render_start_outcome(outcome)
        return False

    def _release_admission(
        self, token: AdmissionToken
    ) -> AdmissionReleased:
        executor = self._run_executor
        if executor is None:
            raise RuntimeError("no admission executor")
        state = self._admission_state
        if state is None or state.token is not token:
            return AdmissionReleased(
                token, CleanupStatus.CLEANUP_PENDING
            )
        self._retain_outgoing_display = False

        released = state.release_receipt
        retirement = state.retirement_receipt
        retirement_applied = state.retirement_applied
        if (
            released is None
            or released.cleanup_status is not CleanupStatus.CLEANED
        ):
            try:
                candidate = executor.release_admission(token)
            except Exception:
                candidate = None
            if _admission_release_is_exact(candidate, token):
                released = candidate
                current = self._context_controller.run_identity
                if (
                    retirement is None
                    or not _retirement_receipt_is_current(
                        retirement, current
                    )
                ):
                    retirement = candidate.display_retirement
            else:
                released = AdmissionReleased(
                    token, CleanupStatus.CLEANUP_PENDING
                )

        if not retirement_applied and retirement is not None:
            retirement_applied = (
                self._context_controller.apply_display_retirement(
                    retirement
                )
            )

        state = _AdmissionPageOwner(
            token=token,
            releasing=True,
            release_receipt=released,
            admission_receipt=state.admission_receipt,
            retirement_receipt=retirement,
            retirement_applied=retirement_applied,
        )
        retirement_pending = (
            not retirement_applied
            and retirement is not None
            and _retirement_receipt_is_current(
                retirement,
                self._context_controller.run_identity,
            )
        )
        if (
            released.cleanup_status is CleanupStatus.CLEANED
            and not retirement_pending
        ):
            self._admission_state = None
        else:
            self._admission_state = state
            if not self._closing and not self._closed:
                self._ensure_timer()
        return released

    def _stop_run(self) -> None:
        if self._closing or self._closed:
            return
        token = self._admission
        if token is not None:
            released = self._release_admission(token)
            self._refuse_preparing(token)
            if released.cleanup_status is not CleanupStatus.CLEANED:
                self._notice("Output cleanup remains pending.")
            self._refresh_shell()
            return
        try:
            self._context_controller.stop()
        except Exception as error:
            self._error_notice("Standard stop dispatch failed", error)
            return
        self._progress = replace(
            self._progress, detail="Stopping run…"
        )
        self._refresh_shell()
        self._ensure_timer()

    def _select_scan(self, value: object) -> None:
        if type(value) is not str or not value:
            return
        if os.path.isdir(value):
            self._set_browser_directory(value, explicit=True)
            return
        if self._context_controller.select_browser_target(value):
            self._notice("")
            self._refresh_shell()
            return
        try:
            self._context_controller.begin_browse(value)
        except Exception as error:
            self._error_notice("Browse refused", error)
            if self._context_controller.browse_pending:
                self._ensure_timer()
            return
        self._notice("")
        self._ensure_timer()

    def _select_frames(self, command: ShellCommand) -> None:
        frame = command.frame
        frames = command.frames
        if (
            frame is not None
            and not self._context_controller.owns_frame(frame)
        ):
            return
        if any(
            not self._context_controller.owns_frame(candidate)
            for candidate in frames
        ):
            return
        if not self._context_controller.select_navigation(
            frame, frames
        ):
            return
        navigation = self._context_controller.navigation
        latest = (
            navigation.frames[-1]
            if navigation.frames
            else None
        )
        if frame is not None and frame is not latest:
            self._auto_last = False
        self._refresh_shell()
        self._ensure_timer()

    def _drain_executor(self) -> None:
        if self._closing or self._closed:
            return
        changed = self._poll_admission()
        if self._context_controller.poll_browse_preview():
            changed = True
        if self._context_controller.browse_pending:
            try:
                outcome = self._context_controller.poll_browse()
            except Exception as error:
                self._error_notice("Browse failed", error)
                outcome = None
            if outcome is not None:
                changed = True
                detail = getattr(outcome, "detail", "")
                self._notice(detail)
                current = self._context_controller.navigation.current
                if current is not None:
                    self._follow_processed_artifact(current)

        executor = self._run_executor
        events: tuple[StandardRunEvent, ...] = ()
        if executor is not None:
            try:
                candidate = executor.drain_events()
            except Exception as error:
                self._error_notice(
                    "Standard event drain failed", error
                )
                candidate = ()
            if type(candidate) is tuple:
                events = candidate
            else:
                self._notice(
                    "Standard event drain returned an invalid result."
                )
        for event in events:
            expected = (
                self._context_controller.run_identity
                or self._lifecycle.active_run_identity
                or self._lifecycle.attempt_run_identity
            )
            if (
                expected is None
                or not standard_event_is_valid(event, expected)
            ):
                continue
            if event.kind is StandardEventKind.DISCOVERY:
                self._progress = ProgressProjection(
                    0, event.total, event.detail
                )
                continue
            if event.kind is StandardEventKind.CONTEXT_READY:
                try:
                    self._context_controller.adopt_acquisition(
                        event.run_identity
                    )
                    changed = True
                except Exception as error:
                    self._error_notice(
                        "Acquisition context refused", error
                    )
                continue
            if (
                event.kind is StandardEventKind.FRAME_READY
                and event.navigation_delta is not None
            ):
                if self._context_controller.accept_navigation(
                    event.navigation_delta,
                    plot_mode=self._preferences.plot_mode,
                    follow_latest=self._auto_last,
                ):
                    frame = event.navigation_delta.appended
                    self._run_frame_seen = True
                    self._batch_latest_frame = frame
                    self._browser_transient_frame = frame
                    self._progress = ProgressProjection(
                        event.completed,
                        event.total,
                        event.detail,
                    )
                    if not self._active_batch_mode:
                        self._follow_processed_artifact(frame)
                        changed = True
                continue
            if event.kind is StandardEventKind.DISPLAY_READY:
                if (
                    self._context_controller.qualify_display_event(
                        event
                    )
                    is not None
                    and (
                        self._run_frame_seen
                        or self._lifecycle.phase is RunPhase.IDLE
                    )
                    and not self._active_batch_mode
                ):
                    changed = True
                continue
            if event.kind in {
                StandardEventKind.FINISHED,
                StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            }:
                self._retain_outgoing_display = False
                if self._batch_latest_frame is not None:
                    self._follow_processed_artifact(
                        self._batch_latest_frame
                    )
                self._accept_terminal_event(event)
                # The writer publishes its atomic final path before emitting
                # the terminal event.  Re-enumerate here so a non-batch run
                # whose FRAME_READY preceded that rename becomes visible.
                self._request_browser_catalog()
                operation = self._browser_catalog_operation
                self._browser_transient_clear_token = (
                    None if operation is None else operation.token
                )
                self._active_batch_mode = False
                self._run_frame_seen = False
                self._batch_latest_frame = None
                changed = True

        if changed:
            self._refresh_shell()
        if not self._polling_needed():
            self._run_timer.stop()

    def _accept_terminal_event(
        self, event: StandardRunEvent
    ) -> None:
        identity = event.run_identity
        stop_was_requested = (
            self._lifecycle.phase is RunPhase.STOPPING
        )
        self._progress = ProgressProjection(
            event.completed,
            event.total,
            event.detail,
        )
        failed = event.kind is StandardEventKind.FAILED
        if failed or event.cleanup_status is not CleanupStatus.CLEANED:
            if self._lifecycle.active_run_identity is identity:
                self._lifecycle.fatal(FatalExecution(identity))
            if (
                event.cleanup_status is CleanupStatus.CLEANED
                and self._pipeline is not None
            ):
                self._pipeline.owners_closed(OwnersClosed(identity))
            self._notice(
                event.detail
                or (
                    "Standard execution failed."
                    if failed
                    else "Standard cleanup remains pending."
                )
            )
            return
        ended = self._lifecycle.execution_ended(
            ExecutionEnded(identity)
        )
        if ended.phase is RunPhase.FINALIZING:
            self._lifecycle.durable_final(DurableFinal(identity))
        self._notice(
            "Standard run stopped."
            if (
                event.kind is StandardEventKind.STOPPED
                or stop_was_requested
            )
            else ""
        )

    def _polling_needed(self) -> bool:
        if (
            self._admission is not None
            or self._context_controller.browse_pending
            or self._context_controller.browse_preview_polling_needed
        ):
            return True
        if self._lifecycle.phase not in {
            RunPhase.IDLE,
            RunPhase.FAILED,
            RunPhase.CLOSED,
        }:
            return True
        context = self._context_controller.acquisition_context
        display = (
            None if context is None else context.publication_store
        )
        worker = getattr(display, "hydration_thread", None)
        return worker is not None and worker.is_alive()

    def _ensure_timer(self) -> None:
        if not self._run_timer.isActive():
            self._run_timer.start()

    def _refresh_shell(self, *, preserve_display: bool = False) -> None:
        # The worker can rescope the one mutable acquisition context after its
        # event queue snapshot but before this GUI refresh.  Catch up only an
        # already-selected exact acquisition owner; Browse remains untouched.
        self._context_controller.synchronize_acquisition_scope()
        snapshot = self._intents.snapshot()
        intent = snapshot.thaw()
        controls = self._project_controls(snapshot)
        permitted, blocker = self._start_permitted()
        navigation = self._context_controller.navigation
        live_update = self._lifecycle.phase in {
            RunPhase.STARTING,
            RunPhase.RUNNING,
            RunPhase.PAUSING,
            RunPhase.PAUSED,
            RunPhase.RESUMING,
            RunPhase.STOPPING,
        }
        payloads = self._context_controller.project_navigation(
            preferences=self._preferences,
            processing_mode=intent.processing_mode,
            live_update=live_update,
        )
        resident_frames = self._context_controller.resident_frame_keys
        current = navigation.current
        replacement_ready = (
            current is not None
            and any(
                payload.frame_key is current for payload in payloads
            )
            and any(frame is current for frame in resident_frames)
        )
        if self._retain_outgoing_display and replacement_ready:
            self._retain_outgoing_display = False
        preserve_display = (
            preserve_display or self._retain_outgoing_display
        )
        observation = self._source_observation
        source = intent.source_spec
        if observation is None or observation.source != source:
            source_count = None
            source_count_is_files = False
            source_count_includes_immediate = False
        else:
            source_count = observation.observed_file_count
            source_count_is_files = (
                type(source) is DirectorySourceSpec
                or is_single_image_spec(source)
            )
            source_count_includes_immediate = (
                observation.file_count_scope
                is SourceCountScope.SELECTED_PLUS_IMMEDIATE
            )
        projection = self._context_projection.build_shell(
            revision=self._shell_revision,
            controls=controls,
            controls_readiness=self._controls_readiness,
            phase=self._lifecycle.phase,
            intent=intent,
            contexts=self._context_controller.projectable_contexts,
            selection=self._context_controller.selection,
            navigation=navigation,
            payloads=payloads,
            resident_frames=resident_frames,
            progress=self._progress,
            preferences=self._preferences,
            browser_directory=self._browser_directory,
            browser_catalog=self._browser_catalog,
            browser_transient_frame=self._browser_transient_frame,
            date_sorted=self._date_sorted,
            auto_last=self._auto_last,
            executor_available=self._run_executor is not None,
            start_permitted=permitted,
            start_blocker=blocker,
            notice=self._notice_text,
            source_count=source_count,
            source_count_is_files=source_count_is_files,
            source_count_includes_immediate=(
                source_count_includes_immediate
            ),
            norm_aggregate=self._context_controller.norm_aggregate,
        )
        try:
            self._shell.apply_state(
                projection,
                preserve_display=preserve_display,
            )
        except Exception as error:
            self._notice(
                "Passive shell render failed: "
                f"{detached_exception_strings(error)[2]}"
            )
            return
        self._context_controller.commit_navigation_projection(
            self._shell.scientific.trace_history_keys
        )
        rendered_axis = self._shell.scientific.rendered_image_axis
        if rendered_axis is not None:
            self._rendered_image_axis = rendered_axis
        self._shell_revision += 1

    def _project_controls(
        self, snapshot: RunIntentSnapshot
    ):
        intent = snapshot.thaw()
        detector_key = (
            str(intent.poni_file or ""),
            str(intent.mask_file or ""),
        )
        if detector_key != self._detector_summary_key:
            self._detector_summary_text = detector_summary(*detector_key)
            self._detector_summary_key = detector_key
        observation = None
        source = intent.source_spec
        if source is not None:
            current = self._source_observation
            observation = (
                current
                if current is not None and current.source == source
                else self._sources.project_motor_knowledge(source, None)
            )
        controls = project_controls(
            snapshot,
            observation,
            self._lifecycle.phase,
            advanced_editor_available=(
                self._advanced_settings_editor is not None
            ),
            source_mode_override=self._source_mode,
            detector_summary_override=self._detector_summary_text,
        )
        project_key = (
            str(intent.project_root or ""),
            str(intent.save_path or ""),
        )
        if project_key != self._project_readiness_key:
            self._project_readiness = project_header_projection(controls)
            self._project_readiness_key = project_key
        experiment_key = (
            bool(intent.gi.enabled),
            str(intent.poni_file or ""),
        )
        if experiment_key != self._experiment_readiness_key:
            self._experiment_readiness = experiment_header_projection(
                controls
            )
            self._experiment_readiness_key = experiment_key
        self._controls_readiness = ControlsReadinessProjection(
            self._project_readiness,
            self._experiment_readiness,
            processing_header_projection(controls),
        )
        return controls

    def _start_permitted(self) -> tuple[bool, str]:
        if self._closing or self._closed:
            return False, "Workspace is closing"
        if self._run_executor is None or self._pipeline is None:
            return False, "Execution is unavailable"
        if self._admission is not None:
            return False, "Output cleanup remains pending"
        if self._context_controller.browse_pending:
            return False, "Browse cleanup remains pending"
        phase = self._lifecycle.phase
        if phase is RunPhase.IDLE:
            return True, ""
        if phase is RunPhase.FAILED and self._lifecycle.reset_permitted:
            return True, ""
        return False, (
            "Standard cleanup remains pending"
            if phase is RunPhase.FAILED
            else f"Run is {phase.value}"
        )

    def _edit_run_strip(
        self, kind: ShellCommandKind, value: object
    ) -> None:
        if kind is ShellCommandKind.SET_OUTPUT_POLICY:
            self._on_field_value(OUTPUT_MODE, value)
            return
        snapshot = self._intents.snapshot()
        candidate = snapshot.thaw()
        if kind is ShellCommandKind.SET_PROCESSING_MODE:
            if type(value) is not str or not value:
                return
            candidate.processing_mode = value
        elif kind is ShellCommandKind.SET_BATCH:
            if type(value) is not bool:
                return
            candidate.batch_mode = value
        elif kind is ShellCommandKind.SET_LIVE:
            if type(value) is not bool:
                return
            candidate.live_mode = value
        elif kind is ShellCommandKind.SET_CORES:
            if type(value) is not int or value < 1:
                return
            candidate.max_cores = value
        else:
            return
        result = self._intents.commit(
            candidate, expected_revision=snapshot.revision
        )
        self._reconcile_snapshot(snapshot, result.snapshot)

    def _edit_scientific_preference(
        self, command: ShellCommand
    ) -> bool:
        kind, value = command.kind, command.value
        updates: dict[str, object] = {}
        if kind is ShellCommandKind.SET_NORM_CHANNEL:
            updates["norm_channel"] = str(value)
        elif kind is ShellCommandKind.SET_COLOR_MAP:
            updates["color_map"] = str(value)
        elif kind is ShellCommandKind.SET_LOG_SCALE:
            updates["log_scale"] = bool(value)
        elif kind is ShellCommandKind.SET_IMAGE_AXIS:
            if type(value) is not str or value not in {
                "Q-Chi",
                "2Th-Chi",
                "Qz-Qxy",
                "qip_qoop",
                "q_chi",
                "exit_angles",
            }:
                return False
            updates["image_axis"] = value
            if self._preferences.share_axis:
                matching_plot_axis = share_plot_axis_for_image(value)
                if matching_plot_axis is None:
                    updates["share_axis"] = False
                else:
                    updates["plot_axis"] = matching_plot_axis
        elif kind is ShellCommandKind.SET_PLOT_AXIS:
            updates["plot_axis"] = str(value)
        elif kind is ShellCommandKind.SET_PLOT_MODE:
            if type(value) is not str or value not in {
                "Single",
                "Overlay",
                "Average",
                "Sum",
                "Waterfall",
            }:
                return False
            prior_mode = self._preferences.plot_mode
            updates["plot_mode"] = value
            if value == "Single" and prior_mode != "Single":
                self._shell.browser.cancel_pending_frame_selection()
                current = self._context_controller.navigation.current
                self._context_controller.select_navigation(
                    current,
                    () if current is None else (current,),
                )
        elif kind is ShellCommandKind.SET_SHARE_AXIS:
            if type(value) is not bool:
                return False
            matching_plot_axis = share_plot_axis_for_image(
                getattr(
                    self,
                    "_rendered_image_axis",
                    self._preferences.image_axis,
                )
            )
            updates["share_axis"] = bool(
                value and matching_plot_axis is not None
            )
            if value:
                if matching_plot_axis is not None:
                    updates["plot_axis"] = matching_plot_axis
        elif kind is ShellCommandKind.SET_SLICE_ENABLED:
            updates["slice_enabled"] = bool(value)
        elif kind is ShellCommandKind.SET_SLICE_CENTER:
            updates["slice_center"] = float(value)
        elif kind is ShellCommandKind.SET_SLICE_WIDTH:
            updates["slice_width"] = float(value)
        elif kind is ShellCommandKind.SET_RANGE:
            if type(value) not in {int, float}:
                return False
            if command.path == ("q", "low"):
                updates["q_range"] = (
                    float(value), self._preferences.q_range[1]
                )
            elif command.path == ("q", "high"):
                updates["q_range"] = (
                    self._preferences.q_range[0], float(value)
                )
            elif command.path == ("chi", "low"):
                updates["chi_range"] = (
                    float(value), self._preferences.chi_range[1]
                )
            elif command.path == ("chi", "high"):
                updates["chi_range"] = (
                    self._preferences.chi_range[0], float(value)
                )
            else:
                return False
        elif kind is ShellCommandKind.SET_PLOT_OPTION:
            options = self._preferences.plot_options
            option_updates: dict[str, object] = {}
            if command.path == ("waterfall", "y_axis"):
                if type(value) is not str or not value:
                    return False
                option_updates["waterfall_y_axis"] = value
            elif command.path == ("waterfall", "start"):
                if type(value) is not int or value < 1:
                    return False
                option_updates["waterfall_start"] = value
            elif command.path == ("waterfall", "stop"):
                if type(value) is not int or value < 0:
                    return False
                option_updates["waterfall_stop"] = value
            elif command.path == ("waterfall", "step"):
                if type(value) is not int or value < 1:
                    return False
                option_updates["waterfall_step"] = value
            elif command.path == ("overlay", "offset"):
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                ):
                    return False
                option_updates["overlay_offset"] = float(value)
            elif command.path == ("other", "legend"):
                if type(value) is not bool:
                    return False
                option_updates["show_legend"] = value
            elif command.path == ("other", "intensity_scale"):
                if type(value) is not str or value not in {
                    "Linear",
                    "Sqrt",
                    "Log",
                }:
                    return False
                option_updates["intensity_scale"] = value
            else:
                return False
            updates["plot_options"] = replace(options, **option_updates)
        elif kind is ShellCommandKind.SET_BACKGROUND:
            updates["background_set"] = (
                not self._preferences.background_set
            )
        elif kind is ShellCommandKind.SET_DATE_SORT:
            self._date_sorted = bool(value)
            return True
        elif kind is ShellCommandKind.SET_AUTO_LAST:
            self._auto_last = bool(value)
            if self._auto_last:
                self._context_controller.select_latest_navigation(
                    plot_mode=self._preferences.plot_mode
                )
            return True
        elif kind is ShellCommandKind.CLEAR_1D:
            self._shell.browser.cancel_pending_frame_selection()
            current = self._context_controller.navigation.current
            self._context_controller.select_navigation(
                current, () if current is None else (current,)
            )
            return True
        else:
            return False
        self._preferences = replace(self._preferences, **updates)
        return True

    def _render_start_outcome(self, outcome: object) -> None:
        self._retain_outgoing_display = False
        if isinstance(outcome, (StartRefused, StartFailed)):
            self._notice(outcome.detail or outcome.reason.value)
        else:
            self._notice("Standard run was not started.")
        self._refresh_shell()

    def closeEvent(self, event: QtCore.QEvent) -> None:
        receipt = self.close_workspace()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            event.accept()
        else:
            event.ignore()

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DeferredDelete:
            receipt = self.close_workspace()
            if receipt.cleanup_status is not CleanupStatus.CLEANED:
                return True
        return super().event(event)

    def _connect(
        self, signal: Any, slot: Callable[..., object]
    ) -> None:
        signal.connect(slot)
        self._connections.append((signal, slot))

    def _on_field_draft(
        self, path: object, value: object
    ) -> None:
        if self._closing or self._closed:
            return
        reduced = reduce_control_edit(
            self._intents.snapshot(), path, value  # type: ignore[arg-type]
        )
        self._notice(
            reduced.reason if isinstance(reduced, EditRefusal) else ""
        )
        self._refresh_shell()

    def _show_advanced_settings_dialog(
        self,
        snapshot: RunIntentSnapshot,
    ) -> AdvancedSettingsValues | None:
        dialog = self._advanced_dialog
        if dialog is None:
            dialog = AdvancedSettingsDialog(self)
            self._advanced_dialog = dialog
        return dialog.edit(snapshot)

    def _edit_advanced_settings(self) -> None:
        if self._closing or self._closed:
            return
        if self._lifecycle.phase not in {
            RunPhase.IDLE,
            RunPhase.FAILED,
        }:
            self._notice("Controls are locked during the active run.")
            self._refresh_shell()
            return
        editor = self._advanced_settings_editor
        if editor is None:
            self._notice(
                "Advanced settings require a native vNext editor."
            )
            self._refresh_shell()
            return
        snapshot = self._intents.snapshot()
        try:
            values = editor(snapshot)
        except Exception as error:
            self._error_notice("Advanced settings failed", error)
            return
        if values is None:
            return
        reduced = reduce_advanced_settings(snapshot, values)
        if isinstance(reduced, EditRefusal):
            self._notice(reduced.reason)
            self._refresh_shell()
            return
        if isinstance(reduced, EditNoChange):
            self._notice("")
            self._refresh_shell()
            return
        try:
            result = self._intents.commit(
                reduced,
                expected_revision=snapshot.revision,
            )
        except Exception as error:
            self._error_notice("Advanced settings commit failed", error)
            return
        if type(result) is IntentCommitAccepted:
            self._notice("")
            self._reconcile_snapshot(snapshot, result.snapshot)
        elif type(result) is IntentRecaptureRequired:
            self._notice(
                "Advanced edit superseded; review current values."
            )
            self._reconcile_snapshot(snapshot, result.snapshot)

    def _on_field_value(
        self, path: object, value: object
    ) -> None:
        if self._closing or self._closed:
            return
        if path == SOURCE_TYPE:
            if type(value) is not str or value not in {
                "Image Series",
                "Image Directory",
                "Single Image",
            }:
                self._notice("Unknown source type.")
                self._refresh_shell()
                return
            self._switch_source_mode(value)
            return
        snapshot = self._intents.snapshot()
        reduced = reduce_control_edit(
            snapshot, path, value  # type: ignore[arg-type]
        )
        if isinstance(reduced, EditRefusal):
            self._notice(reduced.reason)
            self._refresh_shell()
            return
        if isinstance(reduced, EditNoChange):
            self._notice("")
            self._refresh_shell()
            return
        result = self._intents.commit(
            reduced, expected_revision=snapshot.revision
        )
        if type(result) is IntentCommitAccepted:
            self._notice("")
            self._reconcile_snapshot(snapshot, result.snapshot)
        elif type(result) is IntentRecaptureRequired:
            self._notice("Edit superseded; review current value.")
            self._reconcile_snapshot(snapshot, result.snapshot)

    def _request_observation(
        self, snapshot: RunIntentSnapshot
    ) -> None:
        source = snapshot.thaw().source_spec
        if source is None:
            self._source_observation = None
            self._source_status.show_no_source()
            return
        pool = self._observation_pool
        if pool is None or self._closing:
            return
        self._observation_token += 1
        if (
            self._source_observation is not None
            and self._source_observation.source != source
        ):
            self._source_observation = None
        request = SourceObservationRequest(
            self._observation_token, snapshot.revision, source
        )
        self._source_status.show_checking(_source_label(source))
        future = pool.submit(self._sources.observe, request)
        operation = _ObservationOperation(request, future)
        self._observation = operation
        page_ref = weakref.ref(self)
        future.add_done_callback(
            lambda done: ScatteringWorkspace._deliver_observation(
                page_ref, operation, done
            )
        )

    @staticmethod
    def _deliver_observation(
        page_ref: weakref.ReferenceType["ScatteringWorkspace"],
        operation: _ObservationOperation,
        future: Future[object],
    ) -> None:
        page = page_ref()
        if page is None:
            return
        try:
            if page._closing or page._closed:
                return
            page._observationFinished.emit(operation, future)
        except RuntimeError:
            return

    def _on_observation(
        self, operation: object, future: object
    ) -> None:
        if (
            self._closing
            or self._closed
            or operation is not self._observation
        ):
            return
        if (
            type(operation) is not _ObservationOperation
            or future is not operation.future
        ):
            return
        self._observation = None
        try:
            observation: object = operation.future.result()
        except Exception:
            self._source_status.show_unavailable(
                _source_label(operation.request.source)
            )
            return
        if type(observation) is not SourceObservation:
            self._source_status.show_unavailable(
                _source_label(operation.request.source)
            )
            return
        request = operation.request
        snapshot = self._intents.snapshot()
        expected_fingerprint = (
            operation.candidate_fingerprint
            if operation.preview
            else None
        )
        if not observation.qualifies(
            request, expected_fingerprint
        ):
            self._source_status.show_unavailable(
                _source_label(operation.request.source)
            )
            return
        if snapshot.thaw().source_spec != request.source:
            return
        self._source_observation = observation
        if operation.preview:
            self._sources.publish_motor_knowledge(observation)
        self._source_status.render(observation)
        self._refresh_shell()
        if (
            not operation.preview
            and (
                type(request.source) is DirectorySourceSpec
                or (
                    type(request.source) is SourceSpec
                    and request.source.kind is SourceKind.TIFF_SERIES
                )
            )
            and observation.exists
            and observation.candidate_fingerprint
            and self._observation_pool is not None
        ):
            future = self._observation_pool.submit(
                self._sources.preview_motors, request
            )
            preview = _ObservationOperation(
                request,
                future,
                True,
                observation.candidate_fingerprint,
            )
            self._observation = preview
            page_ref = weakref.ref(self)
            future.add_done_callback(
                lambda done: ScatteringWorkspace._deliver_observation(
                    page_ref, preview, done
                )
            )

    def _cancel_observation(self) -> None:
        operation, self._observation = self._observation, None
        if operation is not None and not operation.future.cancel():
            try:
                self._sources.cancel_observation(
                    operation.request.observation_id
                )
            except Exception:
                pass

    def _reconcile_snapshot(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
    ) -> None:
        token = self._admission
        if token is not None and current.revision != prior.revision:
            self._release_admission(token)
            self._refuse_preparing(token)
        prior_intent = prior.thaw()
        current_intent = current.thaw()
        prior_source = prior_intent.source_spec
        current_source = current_intent.source_spec
        if current_source is not None:
            current_mode = source_mode(current_source)
            self._source_mode = current_mode
            self._source_history[current_mode] = current_source
        if (
            prior_intent.save_path != current_intent.save_path
            and not self._browser_explicit_directory
        ):
            self._browser_directory = processed_directory(
                current_intent.save_path
            )
            self._browser_catalog = ()
            self._request_browser_catalog()
        self._refresh_shell()
        if prior_source != current_source:
            self._source_observation = None
            self._cancel_observation()
            self._request_observation(current)

    def _choose_directory_dialog(self, current: str) -> str | None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open processed data folder",
            current,
        )
        return chosen or None

    def _choose_control_path(self, path: object) -> None:
        if type(path) is not tuple or not all(
            type(part) is str for part in path
        ):
            self._notice("Unknown control.")
            return
        if path in {SOURCE_FILE, SOURCE_DIRECTORY}:
            snapshot = self._intents.snapshot()
            state = self._project_controls(snapshot)
            fields = (
                ()
                if state.bound_controls is None
                else state.bound_controls.fields
            )
            source_type = next(
                (
                    str(candidate.value)
                    for candidate in fields
                    if candidate.path == SOURCE_TYPE
                ),
                None,
            )
            self._choose_source_selection(source_type)
            return
        chooser = self._control_path_chooser
        if chooser is None:
            self.browseRequested.emit(path)
            return
        snapshot = self._intents.snapshot()
        state = self._project_controls(snapshot)
        fields = (
            ()
            if state.bound_controls is None
            else state.bound_controls.fields
        )
        field = next(
            (candidate for candidate in fields if candidate.path == path),
            None,
        )
        if field is None:
            self._notice("Unknown control.")
            return
        current = "" if field.value is None else str(field.value)
        try:
            selected = chooser(path, current)
        except Exception as error:
            self._error_notice("Browse failed", error)
            return
        if type(selected) is not str or not selected:
            return
        self._on_field_value(path, selected)

    def _choose_source_selection(
        self,
        desired_mode: str | None = None,
    ) -> None:
        chooser = self._source_selection_chooser
        if chooser is None:
            self.sourceSelectionRequested.emit(desired_mode)
            self._refresh_shell()
            return
        current = self._intents.snapshot().thaw().source_spec
        requested_mode = desired_mode or self._source_mode
        try:
            selected = chooser(current, requested_mode)
        except Exception as error:
            self._error_notice("Choose source failed", error)
            return
        if selected is not None:
            self.select_source(selected)
        else:
            self._refresh_shell()

    def _switch_source_mode(self, desired_mode: str) -> None:
        """Switch editor modes without opening a chooser.

        Each mode remembers its last complete selection for this workspace.
        A first visit has no complete selection, so the revisioned intent is
        cleared and Run remains safely unavailable until the user explicitly
        browses.
        """
        if desired_mode == self._source_mode:
            self._notice("")
            self._refresh_shell()
            return
        snapshot = self._intents.snapshot()
        current = snapshot.thaw().source_spec
        if current is not None:
            self._source_history[source_mode(current)] = current
        candidate = snapshot.thaw()
        candidate.source_spec = self._source_history.get(desired_mode)
        result = self._intents.commit(
            candidate,
            expected_revision=snapshot.revision,
        )
        if type(result) is IntentCommitAccepted:
            self._source_mode = desired_mode
            self._notice("")
        else:
            recovered = result.snapshot.thaw().source_spec
            self._source_mode = source_mode(recovered)
            self._notice("Edit superseded; review current value.")
        self._reconcile_snapshot(snapshot, result.snapshot)

    def _choose_browser_directory(self) -> None:
        try:
            selected = self._browser_directory_chooser(
                self._browser_directory
            )
        except Exception as error:
            self._error_notice("Open folder failed", error)
            return
        if type(selected) is not str or not selected:
            return
        self._set_browser_directory(selected, explicit=True)

    def _set_browser_directory(
        self,
        selected: str,
        *,
        explicit: bool,
    ) -> None:
        directory = os.path.abspath(os.path.expanduser(selected))
        self._browser_explicit_directory = explicit
        self._browser_directory = directory
        self._browser_catalog = ()
        self._notice("")
        self._refresh_shell()
        self._request_browser_catalog()

    def _begin_browser_follow(self, identity: RunIdentity) -> None:
        """Release a prior manual-browse veto for one newly launched run."""

        if type(identity) is not RunIdentity:
            return
        self._browser_explicit_directory = False
        self._browser_follow_identity = identity
        self._browser_seen_artifacts.clear()

    def _follow_processed_artifact(
        self,
        frame: DisplayFrameKey,
    ) -> None:
        if (
            self._closing
            or self._closed
            or type(frame) is not DisplayFrameKey
        ):
            return
        if frame.run_identity is not self._browser_follow_identity:
            self._browser_follow_identity = frame.run_identity
            self._browser_seen_artifacts.clear()
        artifact = os.path.abspath(os.path.expanduser(frame.artifact))
        first_seen = artifact not in self._browser_seen_artifacts
        self._browser_seen_artifacts.add(artifact)
        if self._browser_explicit_directory:
            return
        directory = os.path.dirname(artifact)
        changed = directory != self._browser_directory
        if changed:
            self._browser_directory = directory
            self._browser_catalog = ()
            self._refresh_shell()
        if changed or first_seen:
            self._request_browser_catalog()

    def _request_browser_catalog(self) -> None:
        pool = self._browser_catalog_pool
        if (
            pool is None
            or self._closing
            or self._closed
        ):
            return
        self._browser_catalog_token += 1
        directory = self._browser_directory
        future = pool.submit(
            enumerate_processed_artifacts,
            directory,
        )
        operation = _BrowserCatalogOperation(
            self._browser_catalog_token,
            directory,
            future,
        )
        prior, self._browser_catalog_operation = (
            self._browser_catalog_operation,
            operation,
        )
        if prior is not None:
            prior.future.cancel()
        page_ref = weakref.ref(self)
        future.add_done_callback(
            lambda done: ScatteringWorkspace._deliver_browser_catalog(
                page_ref,
                operation,
                done,
            )
        )

    def _poll_browser_catalog(self) -> None:
        """Observe idle external deletion/recreation without GUI-thread I/O."""
        if self._browser_catalog_operation is None:
            self._request_browser_catalog()

    @staticmethod
    def _deliver_browser_catalog(
        page_ref: weakref.ReferenceType["ScatteringWorkspace"],
        operation: _BrowserCatalogOperation,
        future: Future[object],
    ) -> None:
        page = page_ref()
        if page is None:
            return
        try:
            if page._closing or page._closed:
                return
            page._browserCatalogFinished.emit(operation, future)
        except RuntimeError:
            return

    def _on_browser_catalog(
        self,
        operation: object,
        future: object,
    ) -> None:
        if (
            self._closing
            or self._closed
            or type(operation) is not _BrowserCatalogOperation
            or operation is not self._browser_catalog_operation
            or future is not operation.future
            or operation.token != self._browser_catalog_token
            or operation.directory != self._browser_directory
        ):
            return
        self._browser_catalog_operation = None
        try:
            catalog = operation.future.result()
        except Exception as error:
            self._error_notice("Browser refresh failed", error)
            return
        if (
            type(catalog) is not tuple
            or not all(
                type(entry) is BrowserCatalogEntry
                for entry in catalog
            )
        ):
            self._notice("Browser refresh returned invalid data.")
            self._refresh_shell()
            return
        clear_token = self._browser_transient_clear_token
        transient_cleared = (
            clear_token is not None
            and operation.token >= clear_token
        )
        if transient_cleared:
            self._browser_transient_frame = None
            self._browser_transient_clear_token = None
        if catalog == self._browser_catalog and not transient_cleared:
            return
        self._browser_catalog = catalog
        self._refresh_shell()

    def _cancel_browser_catalog(self) -> None:
        operation, self._browser_catalog_operation = (
            self._browser_catalog_operation,
            None,
        )
        if operation is not None:
            operation.future.cancel()

    def _refuse_preparing(self, token: AdmissionToken) -> None:
        if (
            self._pipeline is not None
            and self._lifecycle.phase is RunPhase.PREPARING
        ):
            self._pipeline.refuse(
                token, StartRefusal.OUTPUT_PREFLIGHT
            )

    def _notice(self, message: str) -> None:
        self._notice_text = message
        self.noticeChanged.emit(message)

    def _error_notice(
        self, prefix: str, error: Exception
    ) -> None:
        self._notice(
            f"{prefix}: {detached_exception_strings(error)[2]}"
        )
        if not self._closed:
            self._refresh_shell()


def _source_label(source: SourceSelection | None) -> str:
    if source is None:
        return ""
    if type(source) is DirectorySourceSpec:
        return source.root.name
    return str(source.uri)


def _admission_release_is_exact(
    value: object, token: AdmissionToken
) -> bool:
    try:
        return (
            type(value) is AdmissionReleased
            and value.token is token
            and value.cleanup_status
            in {
                CleanupStatus.CLEANED,
                CleanupStatus.CLEANUP_PENDING,
            }
        )
    except Exception:
        return False


def _retirement_receipt_is_current(
    receipt: DisplayRetirementReceipt,
    current: RunIdentity | None,
) -> bool:
    return (
        receipt.cleanup_status is CleanupStatus.CLEANED
        and receipt.run_identity is current
    )


__all__ = ["ScatteringWorkspace"]
