"""Public composition root for the context-qualified scattering workspace."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import os
import time
from typing import Any, Callable
import weakref

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.modules.display_context import ContextKind
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.gi_motor import pick_default_gi_motor
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
    RunIntentSnapshot,
)
from xrd_tools.session.run_configuration import heavy_residency_choice
from xrd_tools.reduction import ReintegrateResult
from xrd_tools.reduction.provenance_config import jsonable_run_value
from xrd_tools.session.readiness import Tool, tool_from_mode_text
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
    is_single_image_spec,
    single_image_spec,
)
from xdart.utils.browse import browse_start_dir, remember_browse_path
from .advanced_editor import AdvancedSettingsDialog
from .adapters.browse_loader import BrowseLoader
from .adapters.external_operation import OperationSlot
from .browser_catalog import (
    BrowserCatalogEntry,
    DirectoryModifiedCache,
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
from .controls_inventory import SOURCE_EDIT_PATHS
from .controls_projection import (
    AdvancedSettingsValues,
    EditNoChange,
    EditRefusal,
    EditResult,
    MASK_FILE,
    OUTPUT_MODE,
    PONI_FILE,
    SOURCE_DIRECTORY,
    SOURCE_FILE,
    GI_MOTOR,
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
    StandardQuartileTiming,
    StandardRunEvent,
    StandardTerminalTiming,
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
    LifecycleResult,
    OwnersClosed,
    RunIdentity,
    detached_exception_strings,
)
from .experiment_authoring import (
    CalibrationResult, MaskResult, prepare_calibration_request,
    prepare_mask_request, resolve_calibration_executable, resolve_mask_executable,
)
from .shell_projection import (
    ScientificPreferences,
    share_plot_axis_for_image,
)
from .scientific_axes import (
    resolve_norm_presentation,
    slice_recipe_axes_compatible,
    slice_region_orientation,
)
from .shell_values import (
    ArtifactProgress,
    DirectoryFileProgress,
    FrameSelectionIntent,
    ProgressProjection,
    ShellCommand,
    ShellCommandKind,
    SlicePin,
)
from .shell_widgets import (
    experiment_header_projection,
    processing_header_projection,
    project_header_projection,
)
from .source_view import SourceStatusView
from .performance_diagnostics import (
    PerformanceDiagnosticsDialog,
    PerformanceDiagnosticsValues,
    performance_diagnostics_error,
)
from .operation_values import (
    OperationContextStamp, OperationIdentity, OperationTerminalStatus,
    OperationUpdate,
)
from .presentation_background import (DisplayBackgroundTransferReceipt,
    PresentationBackgroundOwner, prepare_background_plan)
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


_NO_DELIBERATE_MANUAL = object()
_NO_AUTOMATIC_GI_MOTOR = object()
_LIVE_EVENT_DRAIN_INTERVAL_MS = 125
_LIVE_PLOT_INTERVAL_ENV = "XDART_LIVE_PLOT_INTERVAL_MS"
_UNSAFE_UNFUNDED_STAGING_ENV = (
    "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC"
)
_UNSAFE_UNFUNDED_STAGING_KEY = (
    "_post_g2_unfunded_staging_diagnostic_v1"
)
_BROWSER_CATALOG_REFRESH_INTERVAL_MS = 1500
_LIVE_SOURCE_REFRESH_PHASES = frozenset({
    RunPhase.RUNNING,
    RunPhase.PAUSING,
    RunPhase.PAUSED,
    RunPhase.RESUMING,
    RunPhase.STOPPING,
})


def _live_plot_interval_ms() -> int:
    try:
        value = int(os.environ.get(
            _LIVE_PLOT_INTERVAL_ENV,
            str(_LIVE_EVENT_DRAIN_INTERVAL_MS),
        ))
    except (TypeError, ValueError):
        value = _LIVE_EVENT_DRAIN_INTERVAL_MS
    return max(_LIVE_EVENT_DRAIN_INTERVAL_MS, value)


def _slice_pins_for_plot_axis(
    pins: tuple[SlicePin, ...],
    prior_axis: str,
    requested_axis: str,
) -> tuple[SlicePin, ...]:
    """Retarget compatible slice recipes and retire incompatible ones."""

    if requested_axis == prior_axis:
        return pins
    if not slice_recipe_axes_compatible(prior_axis, requested_axis):
        return ()
    return tuple(
        replace(pin, plot_axis=requested_axis)
        for pin in pins
    )


def _linearized_frame_selection(navigation, command: ShellCommand):
    if command.intent is FrameSelectionIntent.EXACT:
        return command.frames
    selected_ids = {id(frame) for frame in navigation.selected}
    operand_ids = {id(frame) for frame in command.frames}
    if command.intent is FrameSelectionIntent.TOGGLE_TRACE:
        selected_ids.symmetric_difference_update(operand_ids)
    elif command.intent is FrameSelectionIntent.REMOVE_TRACE_RANGE:
        selected_ids.difference_update(operand_ids)
    else:
        selected_ids.update(operand_ids)
    return tuple(
        frame for frame in navigation.frames if id(frame) in selected_ids
    )


def _source_selection_path(source: object) -> str:
    """Return the concrete browse pick represented by a source selection."""
    if type(source) is DirectorySourceSpec:
        return str(source.root)
    if type(source) is not SourceSpec:
        return ""
    selected = source.options.get("selected_file")
    return str(selected or source.uri)


def _typed_file_source(
    current: object,
    mode: str,
    value: object,
) -> SourceSelection | EditRefusal:
    """Build one complete image-source value from a committed path edit."""

    if type(value) is not str or not value.strip():
        return EditRefusal("Choose an image file.")
    metadata_format: str | None = "auto"
    if type(current) is SourceSpec:
        candidate = current.options.get("metadata_format", "auto")
        if candidate is None or type(candidate) is str:
            metadata_format = candidate
    try:
        if mode == "Image Series":
            return image_series_spec(
                value,
                metadata_format=metadata_format,
            )
        if mode == "Single Image":
            return single_image_spec(
                value,
                metadata_format=metadata_format,
            )
    except (OSError, ValueError) as error:
        return EditRefusal(str(error))
    return EditRefusal("Choose an image source before editing its file.")


def _directory_file_progress(
    event: StandardRunEvent,
) -> DirectoryFileProgress | None:
    if event.files_discovered <= 0:
        return None
    return DirectoryFileProgress(
        event.files_processed,
        event.files_skipped,
        event.files_pending,
        event.files_discovered,
    )


@dataclass(frozen=True, slots=True)
class _ObservationOperation:
    request: SourceObservationRequest
    future: Future[object]
    preview: bool = False
    candidate_fingerprint: str = ""
    passive_refresh: bool = False
    refresh_identity: RunIdentity | None = None


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
        viewer_file_chooser: Callable[[str], str | None] | None = None,
        browser_directory_chooser: (
            Callable[[str, str], str | None] | None
        ) = None,
        control_path_chooser: (
            Callable[[tuple[str, ...], str, str], str | None] | None
        ) = None,
        source_selection_chooser: (
            Callable[
                [SourceSelection | None, str | None, str],
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
        self._pending_source_refresh: SourceObservationRequest | None = None
        self._live_source_refresh_source: DirectorySourceSpec | None = None
        self._source_observation: SourceObservation | None = None
        # LV-UI-5b: source_spec a DELIBERATE user 'Manual' belongs to
        # (F3 sticky rule).  A sentinel — NOT None — marks "no deliberate
        # Manual": a source_spec can itself be None, and the default state
        # must never compare equal to it.
        self._gi_manual_source = _NO_DELIBERATE_MANUAL
        # An automatic metadata pick is source-scoped.  Keep its provenance
        # outside RunIntent so a later source edit can reset only OUR pick,
        # without erasing an explicit real-motor selection by the user.
        self._gi_auto_motor: object = _NO_AUTOMATIC_GI_MOTOR
        self._observation_token = 0
        self._browser_catalog_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1)
        )
        self._browser_catalog_operation: (
            _BrowserCatalogOperation | None
        ) = None
        self._browser_catalog_token = 0
        self._browser_catalog: tuple[BrowserCatalogEntry, ...] = ()
        self._browser_directory_time_cache = DirectoryModifiedCache()
        self._browser_follow_identity: RunIdentity | None = None
        self._browser_seen_artifacts: set[str] = set()
        self._browser_transient_frame: DisplayFrameKey | None = None
        self._browser_transient_clear_token: int | None = None
        self._active_batch_mode = False
        self._run_frame_seen = False
        self._retain_outgoing_display = False
        self._batch_latest_frame: DisplayFrameKey | None = None
        self._presentation_targets: deque[DisplayFrameKey] = deque(maxlen=1)
        self._presentation_run_identity: RunIdentity | None = None
        self._live_plot_interval_ms = _live_plot_interval_ms()
        self._last_live_plot_at: float | None = None
        self._quartile_refresh_identity: RunIdentity | None = None
        self._quartile_refresh_seconds = [0.0, 0.0, 0.0, 0.0]
        self._browser_directory_chooser = (
            browser_directory_chooser
            if browser_directory_chooser is not None
            else self._choose_directory_dialog
        )
        self._viewer_file_chooser = (
            viewer_file_chooser if viewer_file_chooser is not None
            else self._choose_viewer_2d_dialog
        )
        self._viewer_1d_file_chooser = (
            viewer_file_chooser if viewer_file_chooser is not None
            else self._choose_viewer_1d_dialog)
        self._control_path_chooser = control_path_chooser
        self._source_selection_chooser = source_selection_chooser
        self._advanced_dialog: AdvancedSettingsDialog | None = None
        self._advanced_settings_editor = (
            advanced_settings_editor
            if advanced_settings_editor is not None
            else self._show_advanced_settings_dialog
        )
        self._performance_diagnostics_dialog: (
            PerformanceDiagnosticsDialog | None
        ) = None
        self._performance_diagnostics_editor = (
            self._show_performance_diagnostics_dialog
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
        self._detector_scope_owner = None
        self._detector_demand_frame = None
        self._last_scientific_projection = None
        self._rendered_image_axis = self._preferences.image_axis
        self._detector_summary_key: tuple[str, str] | None = None
        self._detector_summary_text = "not configured"
        self._project_readiness_key: tuple[str, str] | None = None
        self._project_readiness = SectionHeaderProjection("")
        self._experiment_readiness_key: tuple[bool, str] | None = None
        self._experiment_readiness = SectionHeaderProjection("")
        self._controls_readiness = ControlsReadinessProjection()
        self._artifact_progress: dict[str, ArtifactProgress] = {}
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
        self._operation_slot = OperationSlot()
        self._background_owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
        self._background_identity: OperationIdentity | None = None
        self._calibration_identity: OperationIdentity | None = None; self._calibration_revision: int | None = None
        self._mask_identity: OperationIdentity | None = None; self._mask_revision: int | None = None
        self._reintegrate_identity: OperationIdentity | None = None
        self._reintegrate_request: object | None = None
        self._reintegrate_target: str | None = None

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

    def _operation_context_stamp(
        self, revision: int | None = None
    ) -> OperationContextStamp:
        if revision is None:
            revision = self._intents.snapshot().revision
        selection = self._context_controller.selection
        return OperationContextStamp(
            revision,
            None if selection is None else selection.context_token,
            None if selection is None else selection.display_generation,
        )

    def _observe_operation_stamp(self, revision: int | None = None) -> None:
        slot = getattr(self, "_operation_slot", None)
        if slot is None:
            return
        slot.observe_stamp(
            ScatteringWorkspace._operation_context_stamp(self, revision)
        )

    def _begin_operation(
        self, frozen: object, body: Callable[..., object]
    ) -> OperationIdentity | None:
        identity = self._operation_slot._begin(
            frozen, self._operation_context_stamp(), body
        )
        if identity is not None:
            self._ensure_timer()
        return identity

    @staticmethod
    def _background_domain(mode: str) -> str | None:
        return {"Int 1D": "integrated_1d", "Int 2D": "integrated_2d", "1D Viewer": "integrated_1d", "2D Viewer": "raw"}.get(mode)

    def _background_key(self, plan, stamp, mode: str, target_facts) -> tuple[object, ...]:
        return (stamp.context_token, stamp.display_generation, plan.domain, mode,
                "raw" if plan.domain == "raw" else "integrated",
                plan.contributor_ids, plan.value_shapes, plan.axis_shapes, target_facts)

    def _background_action(self) -> None:
        owner, slot = self._background_owner, self._operation_slot
        if owner.phase in {"RESERVED", "STAGED", "ACTIVE", "CLEANUP_PENDING"}:
            self._release_display_background(); self._notice("Clearing display background…")
            self._refresh_shell(); return
        if slot.owned:
            self._notice("Display background is unavailable while another operation is active.")
            self._refresh_shell(); return
        mode = self._intents.snapshot().thaw().processing_mode
        domain = self._background_domain(mode)
        controller = self._context_controller; selected = controller.navigation.selected
        payloads = controller.project_background_contributors(self._preferences.slice_pins)
        norm_channel = resolve_norm_presentation(
            controller.norm_aggregate, self._preferences.norm_channel, selected)[2]
        prepared = None if domain is None else prepare_background_plan(
            payloads, domain, controller.navigation.current,
            contributor_count=len(selected), norm_channel=norm_channel)
        if prepared is None:
            self._notice("Select at least one display frame before setting background.")
            self._refresh_shell(); return
        plan, contributors, keys, targets, target_facts = prepared
        stamp = self._operation_context_stamp()
        active_key = self._background_key(plan, stamp, mode, target_facts)
        reservation = owner.reserve(plan, contributors, stamp=stamp,
            active_key=active_key, projection_keys=keys,
            projection_shapes=tuple(target[0].shape for target in targets))
        if reservation is None:
            self._notice("Display background exceeds its standalone 512 MiB workspace.")
            self._refresh_shell(); return
        identity = slot.begin_background(plan, stamp, owner, reservation)
        if identity is None:
            owner.abort(reservation, "START_REFUSED")
            self._notice("Display background operation was not started."); return
        self._background_identity = identity
        self._notice("Computing display background…"); self._refresh_shell(); self._ensure_timer()

    def _consume_background_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._background_identity:
            return False
        if update.terminal is None:
            if update.progress is not None: self._notice("Display background: aggregate…")
            return True
        self._background_identity = None; terminal = update.terminal; receipt = terminal.payload
        if (update.stale or terminal.status is not OperationTerminalStatus.RETURNED
                or type(receipt) is not DisplayBackgroundTransferReceipt):
            self._notice("Display background cancelled." if terminal.status is OperationTerminalStatus.CANCELLED
                         else "Display background was not adopted."); return True
        mode = self._intents.snapshot().thaw().processing_mode
        controller = self._context_controller; selected = controller.navigation.selected
        norm_channel = resolve_norm_presentation(
            controller.norm_aggregate, self._preferences.norm_channel, selected)[2]
        prepared = prepare_background_plan(
            controller.project_background_contributors(self._preferences.slice_pins),
            receipt.domain, controller.navigation.current,
            contributor_count=len(selected), norm_channel=norm_channel)
        valid = prepared is not None and self._background_key(
            prepared[0], self._operation_context_stamp(), mode, prepared[4]) == receipt.active_key
        targets = () if not valid else prepared[3]
        if not valid or not self._background_owner.promote(receipt, targets):
            self._background_owner.abort(receipt.reservation, "ADOPTION_FAILED")
            self._notice("Display background context changed before adoption."); return True
        self._context_controller.reseed_background_projection()
        self._notice("Display background set."); return True

    def _release_display_background(self) -> bool:
        prior = self._background_owner.phase
        projection = self._background_owner.projection(); receipt = None
        if projection is not None:
            self._last_scientific_projection = None
            receipt = self._shell.scientific.release_display_background(projection[2])
        if self._background_owner.release(receipt):
            if prior not in {"EMPTY", "RELEASED"}: self._context_controller.reseed_background_projection()
            self._background_identity = None; return True
        identity = self._background_identity
        if identity is not None and self._operation_slot.current_identity is identity:
            self._operation_slot.cancel(identity)
        self._ensure_timer(); return False

    def _calibrate_action(self) -> None:
        slot, identity = self._operation_slot, self._calibration_identity
        if identity is not None and slot.current_identity is identity:
            accepted = slot.cancel(identity)
            self._notice("Cancelling calibration…" if accepted else "Calibration cancellation was not accepted.")
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        phase = self._lifecycle.phase
        permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        if self._closing or self._closed or self._admission_state is not None or slot.owned or not permitted:
            self._notice("Calibration is unavailable while another operation is active.")
            self._refresh_shell(); return
        if resolve_calibration_executable() is None:
            self._notice("Calibration is unavailable: pyFAI-calib2 is not on PATH.")
            self._refresh_shell(); return
        snapshot = self._intents.snapshot(); start = browse_start_dir("", fallback=snapshot.thaw().project_root)
        try:
            selected = (self._control_path_chooser(PONI_FILE, "", start) if self._control_path_chooser is not None else QtWidgets.QFileDialog.getSaveFileName(self, "Save detector calibration", start, "PONI calibration (*.poni)")[0])
        except Exception as error:
            self._error_notice("Calibration chooser failed", error); return
        if type(selected) is not str or not selected: return
        phase = self._lifecycle.phase
        if self._closing or self._closed or self._admission_state is not None or slot.owned or phase not in {RunPhase.IDLE, RunPhase.FAILED} or phase is RunPhase.FAILED and not self._lifecycle.reset_permitted:
            self._notice("Calibration context changed while choosing output."); return
        snapshot = self._intents.snapshot(); intent = snapshot.thaw()
        try:
            request = prepare_calibration_request(selected, current_poni=str(intent.poni_file or ""), current_mask=str(intent.mask_file or ""))
        except (OSError, ValueError) as error:
            self._notice(str(error)); self._refresh_shell(); return
        remember_browse_path(request.final_path); self._notice(f"Preparing {os.path.basename(request.final_path)}…")
        identity = slot.begin_calibrate(request, self._operation_context_stamp(snapshot.revision))
        if identity is None:
            self._notice("Calibration operation was not started."); return
        self._calibration_identity, self._calibration_revision = identity, snapshot.revision
        self._notice(f"Calibrating {os.path.basename(request.final_path)}…")
        self._refresh_shell(); self._ensure_timer()
    def _consume_calibration_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._calibration_identity: return False
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Calibration: {update.progress.stage}…")
            return True
        terminal, revision = update.terminal, self._calibration_revision
        self._calibration_identity = self._calibration_revision = None
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self._notice("Calibration cancelled." if terminal.status is OperationTerminalStatus.CANCELLED else f"Calibration failed: {terminal.diagnostic}")
            return True
        result = terminal.payload
        valid = (type(result) is CalibrationResult and result.published and result.proof is not None and result.final_state is not None and os.path.normcase(os.path.normpath(result.final_state.path)) == os.path.normcase(os.path.normpath(result.request.final_path)))
        if not valid:
            self._notice("Calibration returned without exact publication proof."); return True
        snapshot = self._intents.snapshot()
        if update.stale or revision is None or snapshot.revision != revision:
            self._notice("Calibration was published but context changed; PONI was not adopted.")
            return True
        reduced = reduce_control_edit(snapshot, PONI_FILE, result.request.final_path)
        if isinstance(reduced, (EditRefusal, EditNoChange)):
            self._notice("Calibration was published but could not be adopted."); return True
        try: committed = self._intents.commit(reduced, expected_revision=revision)
        except Exception as error:
            self._error_notice("Calibration adoption failed", error); return True
        if type(committed) is IntentCommitAccepted:
            self._reconcile_snapshot(snapshot, committed.snapshot); self._notice(f"Calibration adopted: {result.request.final_path}")
        else: self._notice("Calibration was published but its edit was superseded.")
        return True

    def _mask_action(self) -> None:
        slot, identity = self._operation_slot, self._mask_identity
        if identity is not None and slot.current_identity is identity:
            accepted = slot.cancel(identity); self._notice("Cancelling mask…" if accepted else "Mask cancellation was not accepted.")
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        phase = self._lifecycle.phase
        permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        if self._closing or self._closed or self._admission_state is not None or slot.owned or not permitted:
            self._notice("Mask creation is unavailable while another operation is active."); self._refresh_shell(); return
        if resolve_mask_executable() is None:
            self._notice("Mask creation is unavailable: pyFAI-drawmask is not on PATH."); self._refresh_shell(); return
        snapshot = self._intents.snapshot(); start = browse_start_dir("", fallback=snapshot.thaw().project_root)
        try:
            selected = (self._control_path_chooser(MASK_FILE, "", start) if self._control_path_chooser is not None else QtWidgets.QFileDialog.getOpenFileName(self, "Choose TIFF for mask", start, "TIFF image (*.tif *.tiff)")[0])
        except Exception as error:
            self._error_notice("Mask chooser failed", error); return
        if type(selected) is not str or not selected: return
        phase = self._lifecycle.phase
        if self._closing or self._closed or self._admission_state is not None or slot.owned or phase not in {RunPhase.IDLE, RunPhase.FAILED} or phase is RunPhase.FAILED and not self._lifecycle.reset_permitted:
            self._notice("Mask context changed while choosing input."); return
        snapshot = self._intents.snapshot(); intent = snapshot.thaw()
        try: request = prepare_mask_request(selected, current_poni=str(intent.poni_file or ""), current_mask=str(intent.mask_file or ""))
        except (OSError, ValueError) as error:
            self._notice(str(error)); self._refresh_shell(); return
        remember_browse_path(request.source_path); self._notice(f"Preparing {os.path.basename(request.source_path)}…")
        identity = slot.begin_mask(request, self._operation_context_stamp(snapshot.revision))
        if identity is None:
            self._notice("Mask operation was not started."); return
        self._mask_identity, self._mask_revision = identity, snapshot.revision
        self._notice(f"Making {os.path.basename(request.final_path)}…"); self._refresh_shell(); self._ensure_timer()
    def _consume_mask_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._mask_identity: return False
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Mask: {update.progress.stage}…")
            return True
        terminal, revision = update.terminal, self._mask_revision
        self._mask_identity = self._mask_revision = None
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self._notice("Mask cancelled." if terminal.status is OperationTerminalStatus.CANCELLED else f"Mask failed: {terminal.diagnostic}"); return True
        result = terminal.payload
        valid = (type(result) is MaskResult and result.published and result.proof is not None and result.final_state is not None and os.path.normcase(os.path.normpath(result.final_state.path)) == os.path.normcase(os.path.normpath(result.request.final_path)))
        if not valid:
            self._notice("Mask returned without exact publication proof."); return True
        snapshot = self._intents.snapshot()
        if update.stale or revision is None or snapshot.revision != revision:
            self._notice("Mask was published but context changed; mask was not adopted."); return True
        reduced = reduce_control_edit(snapshot, MASK_FILE, result.request.final_path)
        if isinstance(reduced, (EditRefusal, EditNoChange)):
            self._notice("Mask was published but could not be adopted."); return True
        try: committed = self._intents.commit(reduced, expected_revision=revision)
        except Exception as error:
            self._error_notice("Mask adoption failed", error); return True
        if type(committed) is IntentCommitAccepted:
            self._reconcile_snapshot(snapshot, committed.snapshot); self._notice(f"Mask adopted: {result.request.final_path}")
        else: self._notice("Mask was published but its edit was superseded.")
        return True

    @staticmethod
    def _reintegrate_1d_preparation(intent) -> dict[str, object]:
        bai = jsonable_run_value(intent.bai_1d_args, path="reintegrate.selected_plan.bai_args")
        if type(bai) is not dict: raise ValueError("current 1-D integration settings are malformed")
        bai.pop("gi_mode_1d", None); mode = intent.gi.mode_1d; workers = intent.max_cores
        if type(mode) is not str or mode not in {"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"}: raise ValueError("current 1-D mode is unsupported")
        if type(workers) is not int or workers < 1: raise ValueError("current core request is invalid")
        return {"api_version": 1,
            "selected_plan": {"version": 1, "dimension": "1d", "bai_args": bai, "gi_mode": mode},
            "requested_shared_science": {"version": 1, "kind": "persisted_target"},
            "resource_policy": {"version": 1, "kind": "resolve", "envelope_bytes": None,
                                "requests": {"workers": workers}}}

    def _reload_after_reintegrate(self, request, target) -> None:
        if self._context_controller.reload_reintegrate_browse(request, target) is None:
            self._request_browser_catalog()

    def _reintegrate_1d_action(self) -> None:
        slot, active = self._operation_slot, self._reintegrate_identity
        if active is not None and slot.current_identity is active:
            accepted = slot.cancel(active); self._notice("Cancelling Reintegrate 1-D…" if accepted else "Reintegrate cancellation was not accepted."); self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        phase = self._lifecycle.phase; permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        captured = self._context_controller.capture_reintegrate_browse()
        if self._closing or self._closed or self._admission_state is not None or slot.owned or not permitted or captured is None:
            self._notice("Reintegrate 1-D requires one stable loaded Browse context."); self._refresh_shell(); return
        snapshot = self._intents.snapshot()
        try: preparation = self._reintegrate_1d_preparation(snapshot.thaw())
        except (TypeError, ValueError) as error: self._notice(str(error)); self._refresh_shell(); return
        stamp = self._operation_context_stamp(snapshot.revision); recaptured = self._context_controller.capture_reintegrate_browse(); current = self._intents.snapshot()
        try: current_preparation = self._reintegrate_1d_preparation(current.thaw())
        except (TypeError, ValueError): current_preparation = None
        same_browse = (recaptured is not None and recaptured[0] is captured[0] and recaptured[1] is captured[1] and recaptured[2] is captured[2] and recaptured[3:] == captured[3:])
        if current.revision != snapshot.revision or current_preparation != preparation or not same_browse or not self._context_controller.invalidate_reintegrate_browse(*captured):
            self._notice("Reintegrate context changed before dispatch."); self._refresh_shell(); return
        context, request, _selection, target, entry, target_snapshot, labels = captured
        identity = slot.begin_reintegrate(target=target, entry=entry,
            expected_target_snapshot=target_snapshot, expected_labels=labels,
            dimension="1d", preparation_values=preparation, stamp=stamp)
        if identity is None:
            self._reload_after_reintegrate(request, target); self._notice("Reintegrate 1-D was not started; Browse is reloading."); self._refresh_shell(); return
        self._reintegrate_identity, self._reintegrate_request, self._reintegrate_target = identity, request, target
        self._notice("Reintegrating 1-D from authenticated loaded artifact science…"); self._refresh_shell(); self._ensure_timer()

    def _consume_reintegrate_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._reintegrate_identity: return False
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Reintegrate 1-D: {update.progress.stage} {update.progress.completed}/{update.progress.total}…")
            return True
        terminal, request, target = update.terminal, self._reintegrate_request, self._reintegrate_target
        self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = None
        result = terminal.payload
        if terminal.status is OperationTerminalStatus.RETURNED and type(result) is ReintegrateResult:
            self._notice(f"Reintegrate 1-D {result.disposition.lower()}; reloading persisted results.")
        elif terminal.status is OperationTerminalStatus.CANCELLED: self._notice("Reintegrate 1-D cancelled; reloading persisted results.")
        else: self._notice(f"Reintegrate 1-D failed: {terminal.diagnostic}")
        if request is not None and target is not None: self._reload_after_reintegrate(request, target)
        else: self._request_browser_catalog()
        return True

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
        reduced = reduce_source_selection(
            snapshot,
            source,
            reset_auto_gi_motor=self._auto_gi_motor_matches(snapshot),
        )
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
            ScatteringWorkspace._clear_presentation_targets(self)
            self._clear_live_source_refresh()
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

        operation_slot = getattr(self, "_operation_slot", None)
        try:
            operation_close = (
                None if operation_slot is None else operation_slot.close()
            )
            operation_clean = (
                operation_slot is None
                or operation_close.cleanup_status is CleanupStatus.CLEANED
            )
            if operation_clean:
                self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = None
        except Exception:
            operation_clean = False

        if (not self._clear_viewer_1d_renderer(close=True)
                or not self._clear_viewer_2d_renderer(close=True)
                or not self._shell.scientific.clear_workspace()):
            return StartClosed(
                LifecycleResult(
                    LifecycleStatus.REJECTED,
                    self._lifecycle.phase,
                    run_identity=self._close_identity,
                ),
                (),
                CleanupStatus.CLEANUP_PENDING,
                cleanup_identity=self._close_identity,
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
            and operation_clean
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
        ScatteringWorkspace._observe_operation_stamp(self)
        if (
            self._closing
            or self._closed
            or type(command) is not ShellCommand
        ):
            return
        kind = command.kind
        operation_cancel = kind is ShellCommandKind.CONTROL_ACTION and ((command.value == "calibrate" and self._operation_slot.current_identity is self._calibration_identity) or (command.value == "make_mask" and self._operation_slot.current_identity is self._mask_identity) or (command.value == "reintegrate_1d" and self._operation_slot.current_identity is self._reintegrate_identity))
        operation_locked = kind in {ShellCommandKind.RUN_ACTION, ShellCommandKind.CONTROL_DRAFT, ShellCommandKind.CONTROL_EDIT, ShellCommandKind.CONTROL_BROWSE, ShellCommandKind.CONTROL_ACTION, ShellCommandKind.SET_PROCESSING_MODE, ShellCommandKind.SET_BATCH, ShellCommandKind.SET_CORES, ShellCommandKind.SET_LIVE, ShellCommandKind.SET_OUTPUT_POLICY} or kind is ShellCommandKind.MENU and (command.value == "Config:Performance Diagnostics…" or str(command.value).startswith("Config:Heavy residency:"))
        if self._operation_slot.owned and operation_locked and not operation_cancel: self._notice("Experiment operation is still active."); self._refresh_shell(); return
        if kind is ShellCommandKind.RUN_ACTION:
            self._run_action()
            return
        if kind is ShellCommandKind.STOP:
            self._stop_run()
            return
        if kind is ShellCommandKind.MENU:
            if command.value == "Config:Performance Diagnostics…":
                self._edit_performance_diagnostics()
                return
            prefix = "Config:Heavy residency:"
            if str(command.value).startswith(prefix):
                label = str(command.value)[len(prefix):]
                values = {"Auto": None, "16": 16, "32": 32, "64": 64}
                if label not in values:
                    return
                snapshot = self._intents.snapshot()
                candidate = snapshot.thaw()
                value = values[label]
                if value is None:
                    candidate.run_options.pop("heavy_window", None)
                else:
                    candidate.run_options["heavy_window"] = value
                result = self._intents.commit(
                    candidate, expected_revision=snapshot.revision,
                )
                if isinstance(result, IntentRecaptureRequired):
                    self._notice("Heavy residency edit superseded; review current value.")
                else:
                    active = any(value is not None for value in (
                        self._lifecycle.active_run_identity,
                        self._lifecycle.attempt_run_identity,
                    ))
                    self._notice(
                        f"Heavy residency {label} selected"
                        + (" for the next run." if active else ".")
                    )
                self._refresh_shell()
                return
            if command.value == "File:Open Folder":
                self._choose_browser_directory()
            return
        if kind is ShellCommandKind.REFRESH_BROWSER:
            self._browser_directory_time_cache.clear()
            self._request_browser_catalog()
            return
        if kind is ShellCommandKind.SHOW_ALL:
            ScatteringWorkspace._clear_presentation_targets(self)
            self._shell.browser.cancel_pending_frame_selection()
            navigation = self._context_controller.navigation
            if (
                navigation.frames
                and self._context_controller.select_navigation(
                    navigation.current, navigation.frames
                )
            ):
                self._refresh_shell()
            return
        if kind is ShellCommandKind.SELECT_SCAN:
            ScatteringWorkspace._clear_presentation_targets(self)
            if (getattr(self._context_controller, "viewer_1d_owned", False)
                    and not self._clear_viewer_1d_renderer(close=True)):
                return
            if self._context_controller.viewer_2d_owned:
                if not self._clear_viewer_2d_renderer(close=True):
                    return
            self._select_scan(command.value)
            return
        if kind in {
            ShellCommandKind.SELECT_FRAME,
            ShellCommandKind.HYDRATE_FRAME,
            ShellCommandKind.SELECT_BROWSER_FRAMES,
        }:
            ScatteringWorkspace._clear_presentation_targets(self)
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
            if command.value == "calibrate": self._calibrate_action(); return
            if command.value == "make_mask": self._mask_action(); return
            if command.value == "reintegrate_1d": self._reintegrate_1d_action(); return
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
        if kind is ShellCommandKind.SET_BACKGROUND:
            self._background_action()
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
            ScatteringWorkspace._flush_presentation_target(self)
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
            ScatteringWorkspace._clear_presentation_targets(self)
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
        intent = self._intents.snapshot().thaw()
        tool = tool_from_mode_text(intent.processing_mode)
        if tool is Tool.XYE_VIEWER:
            self._choose_viewer_1d_files()
            return
        if tool is Tool.IMAGE_VIEWER:
            if intent.live_mode and self._context_controller.run_identity is None:
                self._notice("2D Viewer Live requires a retained acquisition session")
                return
            self._choose_viewer_2d_file(
                reload=self._context_controller.viewer_2d_context is not None)
            return
        if (self._context_controller.viewer_2d_owned
                and not self._clear_viewer_2d_renderer(close=True)):
            self._notice("2D Viewer cleanup remains pending")
            return
        if (getattr(self._context_controller, "viewer_1d_owned", False)
                and not self._clear_viewer_1d_renderer(close=True)):
            self._notice("1D Viewer cleanup remains pending")
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
        reduced = self._reduce_page_control_edit(
            snapshot, edit.path, edit.value
        )
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

    def _reduce_page_control_edit(
        self,
        snapshot: RunIntentSnapshot,
        path: object,
        value: object,
    ) -> EditResult:
        """Reduce controls whose complete value is owned by the page."""

        reset_auto_gi_motor = (
            type(path) is tuple
            and (path in SOURCE_EDIT_PATHS or path == SOURCE_FILE)
            and self._auto_gi_motor_matches(snapshot)
        )
        if path == SOURCE_FILE:
            source = _typed_file_source(
                snapshot.thaw().source_spec,
                self._source_mode,
                value,
            )
            if isinstance(source, EditRefusal):
                return source
            return reduce_source_selection(
                snapshot,
                source,
                reset_auto_gi_motor=reset_auto_gi_motor,
            )
        return reduce_control_edit(
            snapshot,
            path,  # type: ignore[arg-type]
            value,
            reset_auto_gi_motor=reset_auto_gi_motor,
        )

    def _begin_run(self) -> None:
        pipeline = self._pipeline
        if self._closing or self._closed or pipeline is None:
            self._notice("Standard execution is unavailable.")
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
            return True
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
            ScatteringWorkspace._clear_presentation_targets(self)
            launched_source = outcome.source_capture.source
            self._pending_source_refresh = None
            self._live_source_refresh_source = (
                launched_source
                if (
                    outcome.configuration.live_mode
                    and type(launched_source) is DirectorySourceSpec
                )
                else None
            )
            self._begin_browser_follow(outcome.run_identity)
            self._active_batch_mode = outcome.configuration.batch_mode
            self._run_frame_seen = False
            self._last_live_plot_at = None
            self._quartile_refresh_identity = (
                outcome.run_identity
                if os.environ.get(
                    "XDART_PERF_QUARTILES", "",
                ).strip() == "1"
                else None
            )
            self._quartile_refresh_seconds = [0.0, 0.0, 0.0, 0.0]
            self._batch_latest_frame = None
            self._browser_transient_frame = None
            self._browser_transient_clear_token = None
            self._artifact_progress.clear()
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
        self._retry_deferred_gi_motor_default()
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
            self._retry_deferred_gi_motor_default()
            self._refresh_shell()
            return
        try:
            ScatteringWorkspace._flush_presentation_target(self)
            self._context_controller.stop()
        except Exception as error:
            self._error_notice("Standard stop dispatch failed", error)
            return
        self._progress = replace(
            self._progress, detail="Stopping run…"
        )
        self._refresh_shell()
        self._ensure_timer()

    def _clear_presentation_targets(self) -> None:
        getattr(self, "_presentation_targets", []).clear()
        self._presentation_run_identity = None
        if hasattr(self, "_background_owner"):
            self._release_display_background()

    def _presentation_pacing_active(self, identity: RunIdentity) -> bool:
        controller = self._context_controller
        context, selection = controller.acquisition_context, controller.selection
        return (
            self._lifecycle.phase is RunPhase.RUNNING
            and self._lifecycle.active_run_identity is identity
            and controller.run_identity is identity
            and not self._active_batch_mode
            and self._preferences.plot_mode == "Single"
            and self._auto_last
            and context is not None
            and selection is not None
            and selection.kind is ContextKind.ACQUISITION
            and selection.names(context)
            and selection.owner == context.hydration_owner
        )

    def _select_presentation_target(
            self, identity: RunIdentity, frame: DisplayFrameKey) -> bool:
        controller = self._context_controller
        if (
            frame.run_identity is not identity
            or controller.run_identity is not identity
            or not controller.owns_frame(frame)
        ):
            return False
        return controller.select_navigation(frame, (frame,))

    def _queue_presentation_target(
            self, identity: RunIdentity, frame: DisplayFrameKey) -> None:
        if self._presentation_run_identity is not identity:
            self._clear_presentation_targets()
            self._presentation_run_identity = identity
        self._presentation_targets.append(frame)

    def _advance_presentation_target(self) -> bool:
        identity = getattr(self, "_presentation_run_identity", None)
        if identity is None or not self._presentation_pacing_active(identity):
            self._clear_presentation_targets()
            return False
        while self._presentation_targets:
            frame = self._presentation_targets.popleft()
            if self._select_presentation_target(identity, frame):
                if not self._presentation_targets:
                    self._presentation_run_identity = None
                return True
        self._clear_presentation_targets()
        return False

    def _flush_presentation_target(self) -> bool:
        identity = getattr(self, "_presentation_run_identity", None)
        targets = tuple(getattr(self, "_presentation_targets", ()))
        ScatteringWorkspace._clear_presentation_targets(self)
        if identity is None:
            return False
        return any(self._select_presentation_target(identity, frame)
                   for frame in reversed(targets))

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
        selection = getattr(self._context_controller, "selection", None)
        if selection is not None and selection.kind is ContextKind.VIEWER_1D:
            if frame is None or not any(
                    frame is item for item in self._context_controller.navigation.frames):
                return
            if self._context_controller.select_viewer_1d_frame(frame):
                self._refresh_shell()
            return
        if selection is not None and selection.kind is ContextKind.VIEWER_2D:
            if frame is None:
                return
            current = self._context_controller.navigation.current
            if (current is not frame
                    and self._context_controller.viewer_2d_frame is not None
                    and not self._clear_viewer_2d_renderer()):
                return
            self._context_controller.select_viewer_2d_frame(
                frame.local_frame_label)
            self._ensure_timer()
            return
        if command.kind is ShellCommandKind.SELECT_BROWSER_FRAMES:
            frames = _linearized_frame_selection(
                self._context_controller.navigation,
                command,
            )
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
        poll_viewer_1d = getattr(self._context_controller, "poll_viewer_1d", None)
        if poll_viewer_1d is not None and poll_viewer_1d():
            changed = True
        if self._context_controller.poll_viewer_2d():
            changed = True
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

        force_scientific = changed

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
                self._record_artifact_progress(event)
                self._progress = ProgressProjection(
                    event.completed,
                    event.total,
                    event.detail,
                    tuple(self._artifact_progress.values()),
                    _directory_file_progress(event),
                )
                self._queue_live_source_refresh(event)
                changed = True
                continue
            if event.kind is StandardEventKind.CONTEXT_READY:
                if (self._context_controller.viewer_2d_owned or
                        getattr(self._context_controller, "viewer_1d_owned", False)):
                    continue
                try:
                    self._context_controller.adopt_acquisition(
                        event.run_identity
                    )
                    changed = True
                    force_scientific = True
                except Exception as error:
                    self._error_notice(
                        "Acquisition context refused", error
                    )
                continue
            if (
                event.kind is StandardEventKind.FRAME_READY
                and event.navigation_delta is not None
            ):
                paced = self._presentation_pacing_active(event.run_identity)
                if self._context_controller.accept_navigation(
                    event.navigation_delta,
                    plot_mode=self._preferences.plot_mode,
                    follow_latest=False if paced else self._auto_last,
                ):
                    frame = event.navigation_delta.appended
                    first_paced_frame = paced and not self._run_frame_seen
                    self._run_frame_seen = True
                    self._batch_latest_frame = frame
                    self._browser_transient_frame = frame
                    self._record_artifact_progress(event)
                    self._progress = ProgressProjection(
                        event.completed,
                        event.total,
                        event.detail,
                        tuple(self._artifact_progress.values()),
                        _directory_file_progress(event),
                    )
                    if first_paced_frame:
                        force_scientific = True
                        changed = (
                            self._select_presentation_target(
                                event.run_identity, frame,
                            )
                            or changed
                        )
                    elif paced:
                        self._queue_presentation_target(
                            event.run_identity, frame,
                        )
                    else:
                        changed = True
                        force_scientific = True
                    if not self._active_batch_mode:
                        self._follow_processed_artifact(frame)
                continue
            if event.kind is StandardEventKind.DISPLAY_READY:
                if (
                    self._context_controller.qualify_display_event(
                        event
                    )
                    is not None
                    and not self._active_batch_mode
                ):
                    changed = True
                    force_scientific = True
                continue
            if event.kind in {
                StandardEventKind.FINISHED,
                StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            }:
                force_scientific = True
                ScatteringWorkspace._flush_presentation_target(self)
                self._retain_outgoing_display = False
                if self._batch_latest_frame is not None:
                    self._follow_processed_artifact(
                        self._batch_latest_frame
                    )
                self._accept_terminal_event(event)
                self._retry_deferred_gi_motor_default()
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

        operation_slot = getattr(self, "_operation_slot", None)
        operation_identity = (
            None
            if operation_slot is None
            else operation_slot.current_identity
        )
        if operation_identity is not None:
            ScatteringWorkspace._observe_operation_stamp(self)
            update = operation_slot.poll(operation_identity)
            if update is not None: changed = self._consume_calibration_update(update) or self._consume_mask_update(update) or self._consume_background_update(update) or self._consume_reintegrate_update(update) or changed
            elif operation_identity is self._background_identity and not operation_slot.owned:
                self._background_identity = None; self._notice("Display background failed before terminal publication."); changed = True
            elif operation_identity is self._reintegrate_identity and not operation_slot.owned:
                request, target = self._reintegrate_request, self._reintegrate_target; self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = None
                if request is not None and target is not None: self._reload_after_reintegrate(request, target)
                self._notice("Reintegrate 1-D failed before terminal publication."); changed = True

        advanced_presentation = (
            ScatteringWorkspace._advance_presentation_target(self)
        )
        if advanced_presentation:
            changed = True
        if changed:
            preserve_scientific = False
            last_plot = self._last_live_plot_at
            if (
                advanced_presentation
                and not force_scientific
                and self._live_plot_interval_ms
                > _LIVE_EVENT_DRAIN_INTERVAL_MS
                and last_plot is not None
                and time.monotonic() - last_plot
                < self._live_plot_interval_ms / 1000.0
            ):
                preserve_scientific = True
            if preserve_scientific:
                self._refresh_event_shell(preserve_scientific=True)
            else:
                self._refresh_event_shell()
        if not self._polling_needed():
            self._run_timer.stop()

    def _accept_terminal_event(
        self, event: StandardRunEvent
    ) -> None:
        identity = event.run_identity
        self._clear_live_source_refresh()
        stop_was_requested = (
            self._lifecycle.phase is RunPhase.STOPPING
        )
        self._record_artifact_progress(event)
        timing = self._terminal_timing_with_gui_refresh(
            event.terminal_timing,
            identity,
        )
        self._quartile_refresh_identity = None
        terminal_state = (
            "Failed"
            if event.kind is StandardEventKind.FAILED
            else "Stopped"
            if event.kind is StandardEventKind.STOPPED
            else "Complete"
        )
        terminal_detail = (
            f"{terminal_state} · {timing.elapsed_seconds:.2f} s"
            if timing is not None
            else event.detail
        )
        self._progress = ProgressProjection(
            event.completed,
            event.total,
            terminal_detail,
            tuple(self._artifact_progress.values()),
            _directory_file_progress(event),
            terminal=True,
            terminal_timing=timing,
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

    def _refresh_event_shell(
        self,
        *,
        preserve_scientific: bool = False,
    ) -> None:
        started = (
            time.monotonic()
            if self._quartile_refresh_identity is not None
            else None
        )
        if preserve_scientific:
            self._refresh_shell(preserve_scientific=True)
        else:
            self._refresh_shell()
        if started is None:
            return
        identity = self._lifecycle.active_run_identity
        if identity is not self._quartile_refresh_identity:
            return
        completed, total = self._progress.completed, self._progress.total
        if total < 4 or completed <= 0 or completed > total:
            return
        thresholds = (
            (total + 3) // 4,
            (total + 1) // 2,
            (3 * total + 3) // 4,
        )
        index = next(
            (
                candidate
                for candidate, threshold in enumerate(thresholds)
                if completed <= threshold
            ),
            3,
        )
        self._quartile_refresh_seconds[index] += max(
            0.0, time.monotonic() - started,
        )

    def _terminal_timing_with_gui_refresh(
        self,
        timing: StandardTerminalTiming | None,
        identity: RunIdentity,
    ) -> StandardTerminalTiming | None:
        if (
            timing is None
            or timing.quartiles is None
            or identity is not self._quartile_refresh_identity
        ):
            return timing
        quartiles = timing.quartiles
        details = tuple(
            item for item in quartiles.details if item[0] != "gui_refresh"
        ) + ((
            "gui_refresh",
            tuple(float(value) for value in self._quartile_refresh_seconds),
        ),)
        return replace(
            timing,
            quartiles=StandardQuartileTiming(
                quartiles.frame_counts,
                details,
                quartiles.compute_counts,
            ),
        )

    def _record_artifact_progress(self, event: StandardRunEvent) -> None:
        artifact = event.artifact
        if not artifact or event.artifact_total <= 0:
            return
        prior = self._artifact_progress.get(artifact)
        completed = event.artifact_completed
        published = 0
        if prior is not None:
            completed = max(completed, prior.completed)
            published = (
                prior.completed
                if prior.published is None
                else prior.published
            )
        if event.kind not in {
            StandardEventKind.FINISHED,
            StandardEventKind.STOPPED,
            StandardEventKind.FAILED,
        }:
            # FRAME_READY and the non-terminal directory folds report the
            # exact displayed prefix.  A terminal event may instead advance
            # durable completion after projection failed; do not mislabel
            # that unwitnessed suffix as navigable.
            published = max(published, event.artifact_completed)
        self._artifact_progress[artifact] = ArtifactProgress(
            artifact,
            min(completed, event.artifact_total),
            event.artifact_total,
            min(published, completed, event.artifact_total),
        )

    def _polling_needed(self) -> bool:
        operation_slot = getattr(self, "_operation_slot", None)
        try:
            if operation_slot is not None and operation_slot.owned:
                return True
        except Exception:
            return True
        if (
            self._admission is not None
            or getattr(self._context_controller, "viewer_1d_loading", False)
            or getattr(self._context_controller, "viewer_1d_cleanup_pending", False)
            or self._context_controller.viewer_2d_loading
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

    def _set_detector_mode(self, mode: str) -> bool:
        if mode == "thumbnail":
            if self._preferences.detector_mode != mode: self._release_display_background()
            self._context_controller.clear_full_raw()
            self._detector_demand_frame = None
            self._preferences = replace(
                self._preferences, detector_mode=mode,
                detector_pending=False, detector_diagnostic="",
            )
            return True
        if mode != "full":
            return False
        available, reason = self._context_controller.full_raw_availability()
        if not available:
            self._notice(reason)
            self._preferences = replace(
                self._preferences, detector_mode="thumbnail",
                detector_available=False, detector_pending=False,
                detector_diagnostic=reason,
            )
            return False
        if self._preferences.detector_mode != mode: self._release_display_background()
        self._detector_demand_frame = None
        self._preferences = replace(
            self._preferences, detector_mode=mode,
            detector_available=True, detector_pending=False,
            detector_diagnostic="",
        )
        return True

    def _sync_detector_demand(self) -> None:
        controller = self._context_controller
        selection = getattr(controller, "selection", None)
        owner = (
            selection.owner
            if selection is not None
            and selection.kind is ContextKind.ACQUISITION
            else None
        )
        if owner != self._detector_scope_owner:
            if self._detector_scope_owner is not None or self._preferences.detector_mode == "full":
                controller.clear_full_raw()
            self._detector_scope_owner = owner
            self._detector_demand_frame = None
            self._preferences = replace(
                self._preferences, detector_mode="thumbnail",
                detector_pending=False, detector_diagnostic="",
            )
        available, reason = controller.full_raw_availability()
        current = controller.navigation.current
        if (self._preferences.detector_mode == "full" and available
                and current is not None and current is not self._detector_demand_frame):
            token = controller.request_full_current()
            self._detector_demand_frame = current
            if token is not None:
                self._ensure_timer()
        resident, pending, diagnostic = controller.full_raw_status()
        if self._preferences.detector_mode == "full" and not resident and not pending:
            diagnostic = diagnostic or "Full Raw detector pixels unavailable."
        current = self._preferences
        state = (available, pending, diagnostic or reason)
        if state != (current.detector_available, current.detector_pending, current.detector_diagnostic):
            self._preferences = replace(current, detector_available=state[0], detector_pending=state[1], detector_diagnostic=state[2])

    def _refresh_shell(
        self,
        *,
        preserve_display: bool = False,
        preserve_scientific: bool = False,
    ) -> None:
        # The worker can rescope the one mutable acquisition context after its
        # event queue snapshot but before this GUI refresh.  Catch up only an
        # already-selected exact acquisition owner; Browse remains untouched.
        if self._context_controller.synchronize_acquisition_scope():
            preserve_scientific = False
        snapshot = self._intents.snapshot()
        intent = snapshot.thaw()
        controls = self._project_controls(snapshot)
        permitted, blocker = self._start_permitted()
        tool = tool_from_mode_text(intent.processing_mode)
        viewer_2d, viewer_1d = tool is Tool.IMAGE_VIEWER, tool is Tool.XYE_VIEWER
        viewer = viewer_1d or viewer_2d
        if viewer:
            cleanup = (self._context_controller.viewer_1d_cleanup_pending
                       if viewer_1d else self._context_controller.viewer_2d_cleanup_pending)
            blocked = (self._closing or self._closed
                       or cleanup
                       or self._operation_slot.owned
                       or self._context_controller.browse_pending
                       or self._lifecycle.active_run_identity is not None
                       or self._lifecycle.attempt_run_identity is not None
                       or viewer_2d and intent.live_mode
                       and self._context_controller.run_identity is None)
            permitted = not blocked and (
                self._lifecycle.phase is RunPhase.IDLE
                or self._lifecycle.phase is RunPhase.FAILED
                and self._lifecycle.reset_permitted)
            blocker = ("2D Viewer Live requires a retained acquisition session"
                       if viewer_2d and intent.live_mode
                       and self._context_controller.run_identity is None
                       else f"{'1D' if viewer_1d else '2D'} Viewer cleanup remains pending" if cleanup
                       else "Browse cleanup remains pending" if self._context_controller.browse_pending
                       else "Workspace is closing" if self._closing or self._closed
                       else "Viewer unavailable during active run" if blocked else "")
        navigation = self._context_controller.navigation
        self._sync_detector_demand()
        if self._preferences.slice_pins:
            retained_pins = tuple(
                pin
                for pin in self._preferences.slice_pins
                if self._context_controller.owns_frame(pin.frame)
            )
            if retained_pins != self._preferences.slice_pins:
                self._preferences = replace(
                    self._preferences,
                    slice_pins=retained_pins,
                )
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
        if (
            not preserve_scientific
            and self._retain_outgoing_display
            and replacement_ready
        ):
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
            notice=((self._context_controller.viewer_1d_diagnostic if viewer_1d
                     else self._context_controller.viewer_2d_diagnostic)
                    if viewer else self._notice_text),
            source_count=source_count,
            source_count_is_files=source_count_is_files,
            source_count_includes_immediate=(
                source_count_includes_immediate
            ),
            norm_aggregate=self._context_controller.norm_aggregate,
            presentation_background=self._background_owner.projection(),
        )
        try:
            choice, _ = heavy_residency_choice(intent.run_options)
            self._shell.browser.reconcile_heavy_residency(
                choice,
                next_run=any(value is not None for value in (
                    self._lifecycle.active_run_identity,
                    self._lifecycle.attempt_run_identity,
                )),
            )
            apply_options = {"preserve_display": preserve_display}
            if preserve_scientific:
                apply_options["preserve_scientific"] = True
            self._shell.scientific.expect_display_background(
                self._background_owner.active_key)
            self._shell.apply_state(projection, **apply_options)
        except Exception as error:
            if viewer:
                self._last_scientific_projection = None
                if viewer_1d: self._clear_viewer_1d_renderer(close=True)
            self._notice(
                "Passive shell render failed: "
                f"{detached_exception_strings(error)[2]}"
            )
            return
        if not preserve_scientific:
            scientific, heavy = (
                projection.scientific,
                projection.scientific.heavy,
            )
            self._last_scientific_projection = (None
                if scientific.processing_mode == "Int 1D" and heavy is not None
                and heavy.detector_source == "full" else scientific)
            self._context_controller.commit_navigation_projection(
                self._shell.scientific.trace_history_keys
            )
            rendered_axis = self._shell.scientific.rendered_image_axis
            if rendered_axis is not None:
                self._rendered_image_axis = rendered_axis
            if not preserve_display:
                self._last_live_plot_at = time.monotonic()
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
        operation_identity = self._operation_slot.current_identity
        operation_active = (operation_identity is not None and not self._closing and not self._closed)
        calibration_active = operation_active and operation_identity is self._calibration_identity
        mask_active = operation_active and operation_identity is self._mask_identity
        reintegrate_active = operation_active and operation_identity is self._reintegrate_identity
        phase = self._lifecycle.phase; calibrate_dependency_available = resolve_calibration_executable() is not None; mask_dependency_available = resolve_mask_executable() is not None
        calibrate_available = (
            not self._closing and not self._closed
            and self._admission_state is None
            and (phase is RunPhase.IDLE or phase is RunPhase.FAILED
                 and self._lifecycle.reset_permitted)
            and calibrate_dependency_available
        )
        mask_available = (not self._closing and not self._closed and self._admission_state is None
            and (phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted)
            and mask_dependency_available)
        controls = project_controls(
            snapshot,
            observation,
            self._lifecycle.phase,
            advanced_editor_available=(
                self._advanced_settings_editor is not None
            ),
            source_mode_override=self._source_mode,
            detector_summary_override=self._detector_summary_text,
            calibrate_available=calibrate_available,
            calibrate_dependency_available=calibrate_dependency_available,
            operation_busy=operation_identity is not None,
            calibration_active=calibration_active,
            mask_available=mask_available,
            mask_dependency_available=mask_dependency_available,
            mask_active=mask_active,
            reintegrate_available=(not self._closing and not self._closed
                and self._admission_state is None
                and (phase is RunPhase.IDLE or phase is RunPhase.FAILED
                     and self._lifecycle.reset_permitted)
                and self._context_controller.capture_reintegrate_browse() is not None),
            reintegrate_active=reintegrate_active,
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
        if self._operation_slot.owned:
            return False, "Experiment operation is still active"
        if self._run_executor is None or self._pipeline is None:
            return False, "Execution is unavailable"
        admission = self._admission_state
        if admission is not None:
            if not admission.releasing:
                return False, (
                    "Finishing prior display cleanup…"
                    if admission.admission_receipt is not None
                    else "Checking output targets…"
                )
            released = admission.release_receipt
            if (
                released is not None
                and released.cleanup_status is CleanupStatus.CLEANED
            ):
                return False, "Finishing prior display cleanup…"
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
            if (self._context_controller.viewer_2d_owned
                    and tool_from_mode_text(value) is not Tool.IMAGE_VIEWER
                    and not self._clear_viewer_2d_renderer(close=True)):
                self._notice("2D Viewer cleanup remains pending")
                return
            if (getattr(self._context_controller, "viewer_1d_owned", False)
                    and tool_from_mode_text(value) is not Tool.XYE_VIEWER
                    and not self._clear_viewer_1d_renderer(close=True)):
                self._notice("1D Viewer cleanup remains pending")
                return
            if value != candidate.processing_mode: self._release_display_background()
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

    def _choose_viewer_1d_files(self) -> None:
        context = self._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
            if not self._clear_viewer_1d_renderer(paths=context.paths):
                self._notice("1D Viewer cleanup remains pending")
            self._ensure_timer(); return
        if (self._context_controller.viewer_2d_owned
                and not self._clear_viewer_2d_renderer(close=True)):
            self._notice("2D Viewer cleanup remains pending"); return
        try:
            chooser = getattr(self, "_viewer_1d_file_chooser", self._viewer_file_chooser)
            selected = chooser(self._viewer_1d_start_directory())
            selected = tuple(selected) if type(selected) in {tuple, list} else ()
            if not selected or any(type(path) is not str or not path for path in selected):
                return
            request = self._context_controller.open_viewer_1d(selected)
        except Exception as error:
            self._error_notice("1D Viewer refused", error); return
        if request is not None: self._notice("")
        self._ensure_timer()

    def _choose_viewer_1d_dialog(self, start: str):
        selected, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open 1D Viewer sources", start, "1-D data (*)")
        return tuple(selected)

    def _viewer_1d_start_directory(self) -> str:
        context = self._context_controller.viewer_1d_context
        return "" if context is None else os.path.dirname(context.paths[0])

    def _clear_viewer_1d_renderer(self, *, paths=None, close=False) -> bool:
        self._last_scientific_projection = None
        request = self._context_controller.begin_viewer_1d_renderer_clear(paths)
        if request is not None and close: self._context_controller.close_viewer_1d()
        if request is None:
            context = self._context_controller.viewer_1d_context
            cleared = context is None or context.state.value != "ready"
        elif getattr(request, "acknowledgement_identity", None) is not None:
            self._context_controller.poll_viewer_1d()
            cleared = not self._context_controller.viewer_1d_cleanup_pending
        else:
            try: receipt = self._shell.scientific.clear_viewer_1d(request)
            except Exception: return False
            try: cleared = self._context_controller.acknowledge_viewer_1d_renderer_clear(receipt)
            except Exception: return False
        return bool(cleared and (not close or self._context_controller.close_viewer_1d()))

    def _choose_viewer_2d_file(self, *, reload: bool) -> None:
        if (getattr(self._context_controller, "viewer_1d_owned", False)
                and not self._clear_viewer_1d_renderer(close=True)):
            self._notice("1D Viewer cleanup remains pending"); return
        context = self._context_controller.viewer_2d_context
        if reload:
            selected = None if context is None else context.original_path
        else:
            try:
                selected = self._viewer_file_chooser(
                    self._viewer_2d_start_directory())
            except Exception as error:
                self._error_notice("2D Viewer chooser failed", error)
                return
            if selected is None or type(selected) is not str or not selected:
                return
        if context is not None and not self._clear_viewer_2d_renderer():
            self._notice("2D Viewer cleanup remains pending")
            return
        try:
            request = self._context_controller.open_viewer_2d(selected)
        except Exception as error:
            self._error_notice("2D Viewer refused", error)
            return
        if request is not None:
            self._notice("")
        self._ensure_timer()

    def _choose_viewer_2d_dialog(self, start: str) -> str | None:
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open 2D Viewer source", start, "All files (*)")
        return selected or None

    def _viewer_2d_start_directory(self) -> str:
        context = self._context_controller.viewer_2d_context
        return "" if context is None else os.path.dirname(context.original_path)

    def _clear_viewer_2d_renderer(self, *, close=False) -> bool:
        self._last_scientific_projection = None
        request = self._context_controller.begin_viewer_2d_renderer_clear()
        if request is None:
            cleared = self._context_controller.viewer_2d_frame is None
        else:
            try:
                receipt = self._shell.scientific.clear_viewer_2d(request)
            except Exception:
                return False
            try:
                cleared = self._context_controller.acknowledge_viewer_2d_renderer_clear(
                    receipt)
            except Exception:
                return False
        return bool(cleared and (not close
                    or self._context_controller.close_viewer_2d()))

    def _edit_scientific_preference(
        self, command: ShellCommand
    ) -> bool:
        kind, value = command.kind, command.value
        controller = getattr(self, "_context_controller", None)
        selection = getattr(controller, "selection", None)
        viewer_1d = (selection is not None
                     and selection.kind is ContextKind.VIEWER_1D)
        if viewer_1d and (kind is ShellCommandKind.PIN_SLICE
                or kind is ShellCommandKind.SET_PLOT_MODE
                and type(value) is str and value in {"Average", "Sum"}):
            return False
        updates: dict[str, object] = {}
        if kind is ShellCommandKind.SET_NORM_CHANNEL:
            updates["norm_channel"] = str(value)
        elif kind is ShellCommandKind.SET_COLOR_MAP:
            updates["color_map"] = str(value)
        elif kind is ShellCommandKind.SET_LOG_SCALE:
            updates["log_scale"] = bool(value)
        elif kind is ShellCommandKind.SET_DETECTOR_MODE:
            return self._set_detector_mode(str(value))
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
                    updates["slice_pins"] = _slice_pins_for_plot_axis(
                        self._preferences.slice_pins,
                        self._preferences.plot_axis,
                        matching_plot_axis,
                    )
        elif kind is ShellCommandKind.SET_PLOT_AXIS:
            requested_axis = str(value)
            updates["plot_axis"] = requested_axis
            updates["slice_pins"] = _slice_pins_for_plot_axis(
                self._preferences.slice_pins,
                self._preferences.plot_axis,
                requested_axis,
            )
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
            ScatteringWorkspace._clear_presentation_targets(self)
            updates["plot_mode"] = value
            if value not in {"Overlay", "Waterfall"}:
                updates["slice_pins"] = ()
            if not viewer_1d and value == "Single" and prior_mode != "Single":
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
                    updates["slice_pins"] = _slice_pins_for_plot_axis(
                        self._preferences.slice_pins,
                        self._preferences.plot_axis,
                        matching_plot_axis,
                    )
        elif kind is ShellCommandKind.SET_SLICE_ENABLED:
            updates["slice_enabled"] = bool(value)
        elif kind is ShellCommandKind.SET_SLICE_CENTER:
            updates["slice_center"] = float(value)
        elif kind is ShellCommandKind.SET_SLICE_WIDTH:
            updates["slice_width"] = float(value)
        elif kind is ShellCommandKind.PIN_SLICE:
            state = self._last_scientific_projection
            navigation = self._context_controller.navigation
            heavy = None if state is None else state.heavy
            if (
                state is None
                or state.plot_mode not in {"Overlay", "Waterfall"}
                or not state.slice_enabled
                or heavy is None
                or heavy.cake is None
                or slice_region_orientation(
                    state.plot_axis,
                    heavy.cake_x,
                    heavy.cake_y,
                ) is None
            ):
                return False
            frame = heavy.frame
            if (
                navigation.current is not frame
                or not self._context_controller.owns_frame(frame)
            ):
                return False
            existing = self._preferences.slice_pins
            existing_ids = {pin.projection_id for pin in existing}
            pin = SlicePin(
                frame,
                state.plot_axis,
                float(state.slice_center),
                float(state.slice_width),
            )
            if pin.projection_id in existing_ids:
                return False
            updates["slice_pins"] = (*existing, pin)
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
        elif kind is ShellCommandKind.SET_DATE_SORT:
            requested = bool(value)
            changed = requested != self._date_sorted
            self._date_sorted = requested
            if changed and requested:
                self._request_browser_catalog()
            return True
        elif kind is ShellCommandKind.SET_AUTO_LAST:
            ScatteringWorkspace._clear_presentation_targets(self)
            self._auto_last = bool(value)
            if self._auto_last:
                self._context_controller.select_latest_navigation(
                    plot_mode=self._preferences.plot_mode
                )
            return True
        elif kind is ShellCommandKind.CLEAR_1D:
            background = self._background_owner.projection()
            if (background is not None and background[0] == "integrated_1d"
                    and not self._release_display_background()):
                return False
            if viewer_1d or getattr(controller, "viewer_1d_owned", False):
                return self._clear_viewer_1d_renderer(close=True)
            self._shell.browser.cancel_pending_frame_selection()
            current = self._context_controller.navigation.current
            self._context_controller.select_navigation(
                current, () if current is None else (current,)
            )
            if self._preferences.slice_pins:
                self._preferences = replace(
                    self._preferences,
                    slice_pins=(),
                )
            return True
        else:
            return False
        background = self._background_owner.projection()
        if (background is not None and background[0] == "integrated_1d"
                and (updates.get("norm_channel", self._preferences.norm_channel)
                     != self._preferences.norm_channel
                     or updates.get("slice_pins", self._preferences.slice_pins)
                     != self._preferences.slice_pins)):
            self._release_display_background()
        self._preferences = replace(self._preferences, **updates)
        return True

    def _render_start_outcome(self, outcome: object) -> None:
        self._retain_outgoing_display = False
        if isinstance(outcome, (StartRefused, StartFailed)):
            self._notice(outcome.detail or outcome.reason.value)
        else:
            self._notice("Standard run was not started.")
        self._retry_deferred_gi_motor_default()
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
        if path == SOURCE_FILE:
            self._notice(
                "Choose an image file."
                if type(value) is not str or not value.strip()
                else ""
            )
            self._refresh_shell()
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

    def _show_performance_diagnostics_dialog(
        self,
        snapshot: RunIntentSnapshot,
        plot_interval_ms: int,
    ) -> PerformanceDiagnosticsValues | None:
        dialog = self._performance_diagnostics_dialog
        if dialog is None:
            dialog = PerformanceDiagnosticsDialog(self)
            self._performance_diagnostics_dialog = dialog
        return dialog.edit(snapshot, plot_interval_ms)

    def _edit_performance_diagnostics(self) -> None:
        snapshot = self._intents.snapshot()
        try:
            values = self._performance_diagnostics_editor(
                snapshot, self._live_plot_interval_ms,
            )
        except Exception as error:
            self._error_notice("Performance diagnostics failed", error)
            return
        if values is None:
            return
        error = performance_diagnostics_error(values)
        if error:
            self._notice(error)
            self._refresh_shell()
            return
        candidate = snapshot.thaw()
        candidate.run_options.pop("_post_g2_pipeline", None)
        candidate.run_options["_post_g2_pipeline_v2"] = (
            values.pipeline_mapping()
        )
        candidate.run_options["_post_g2_output_diagnostics_v1"] = (
            values.output_diagnostics_mapping()
        )
        candidate.run_options.pop(_UNSAFE_UNFUNDED_STAGING_KEY, None)
        unsafe_unfunded_staging = (
            os.environ.get(_UNSAFE_UNFUNDED_STAGING_ENV, "").strip() == "1"
            and (
                values.checkpoint,
                values.staging_frame_cap,
            ) == (10_000, 10_008)
        )
        if unsafe_unfunded_staging:
            candidate.run_options[_UNSAFE_UNFUNDED_STAGING_KEY] = {
                "mode": "UNSAFE_UNFUNDED",
                "checkpoint": 10_000,
                "staging_frame_cap": 10_008,
                "max_frames": 3_621,
            }
        try:
            result = self._intents.commit(
                candidate, expected_revision=snapshot.revision,
            )
        except Exception as error:
            self._error_notice("Performance diagnostics commit failed", error)
            return
        if isinstance(result, IntentRecaptureRequired):
            self._notice(
                "Performance diagnostics edit was superseded; review current values."
            )
            self._refresh_shell()
            return
        self._live_plot_interval_ms = values.plot_interval_ms
        self._last_live_plot_at = None
        os.environ["XDART_PERF"] = "1"
        if values.quartile_telemetry:
            os.environ["XDART_PERF_QUARTILES"] = "1"
        else:
            os.environ.pop("XDART_PERF_QUARTILES", None)
        notice = (
            "Performance diagnostics applied: pipeline and output values "
            "take effect on the next run; plot cadence is active now."
        )
        if not values.durable_fsync:
            notice += (
                " Diagnostic fsync is off: crash or power-loss persistence "
                "is not guaranteed."
            )
        if values.quartile_telemetry:
            notice += (
                " Within-run quartile timing is enabled for the next run."
            )
        if values.staging_frame_cap > 64:
            notice += (
                f" Staging cap {values.staging_frame_cap} can increase memory "
                "with frame count and is subject to exact resource admission."
            )
        if unsafe_unfunded_staging:
            notice += (
                " UNSAFE UNFUNDED staging is active for this process: the "
                "64-frame grant is not enlarged, memory can grow with frame "
                "count, and crash recovery is delayed until terminal verify."
            )
        self._notice(notice)
        self._refresh_shell()

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

    def _maybe_default_gi_motor(self, observation) -> None:
        """LV-UI-5b: adopt the metadata-preferred incidence motor.

        The default POLICY is owned by
        :func:`xrd_tools.session.gi_motor.pick_default_gi_motor`; this seam
        only decides WHEN it may apply: the intent still carries the
        construction default ``'Manual'``, the user has not deliberately
        chosen Manual for this exact source (F3 sticky rule), and the
        observation names motors for the selected source.  The pick commits
        through the one ``_on_field_value`` CAS path.
        """
        choices = getattr(observation, "gi_motor_choices", None)
        if not choices:
            return
        # Observation delivery and Run click are serialized on the GUI
        # thread.  Once start owns PREPARING/admission, an automatic CAS would
        # invalidate the exact captured revision and refuse the user's click.
        # Keep the captured displayed value for this run; the retained
        # observation is reconsidered after a clean terminal transition.
        if (
            self._admission is not None
            or self._lifecycle.phase is not RunPhase.IDLE
        ):
            return
        intent = self._intents.snapshot().thaw()
        if intent.gi.incidence_motor != "Manual":
            return
        if (self._gi_manual_source is not _NO_DELIBERATE_MANUAL
                and self._gi_manual_source == intent.source_spec):
            return
        preferred = pick_default_gi_motor(choices)
        if preferred == "Manual":
            return
        self._on_field_value(GI_MOTOR, preferred, automatic=True)

    def _retry_deferred_gi_motor_default(self) -> None:
        observation = self._source_observation
        if observation is not None:
            self._maybe_default_gi_motor(observation)

    def _auto_gi_motor_matches(
        self,
        snapshot: RunIntentSnapshot,
    ) -> bool:
        marker = self._gi_auto_motor
        if marker is _NO_AUTOMATIC_GI_MOTOR:
            return False
        intent = snapshot.thaw()
        return marker == (intent.source_spec, intent.gi.incidence_motor)

    def _on_field_value(
        self,
        path: object,
        value: object,
        *,
        automatic: bool = False,
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
        reduced = self._reduce_page_control_edit(snapshot, path, value)
        if isinstance(reduced, EditRefusal):
            if not automatic:
                self._notice(reduced.reason)
            self._refresh_shell()
            return
        if isinstance(reduced, EditNoChange):
            if path == GI_MOTOR and not automatic:
                # A same-value activation is still an operator claim over an
                # earlier automatic pick.  No intent revision is needed, but
                # provenance must stop treating that value as disposable on
                # the next source change.
                self._gi_manual_source = (
                    snapshot.thaw().source_spec
                    if value == "Manual" else _NO_DELIBERATE_MANUAL
                )
                self._gi_auto_motor = _NO_AUTOMATIC_GI_MOTOR
            if not automatic:
                self._notice("")
            self._refresh_shell()
            return
        result = self._intents.commit(
            reduced, expected_revision=snapshot.revision
        )
        if type(result) is IntentCommitAccepted:
            if path == GI_MOTOR:
                # F3 sticky rule: only an ACCEPTED deliberate 'Manual' pins
                # Manual for this exact source; any real pick releases it.
                self._gi_manual_source = (
                    snapshot.thaw().source_spec
                    if value == "Manual" else _NO_DELIBERATE_MANUAL
                )
                self._gi_auto_motor = (
                    (snapshot.thaw().source_spec, value)
                    if automatic
                    else _NO_AUTOMATIC_GI_MOTOR
                )
            if not automatic:
                self._notice("")
            self._reconcile_snapshot(
                snapshot,
                result.snapshot,
                preserve_terminal=automatic,
            )
        elif type(result) is IntentRecaptureRequired:
            if not automatic:
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
        self._submit_observation(request)

    def _submit_observation(
        self,
        request: SourceObservationRequest,
        *,
        passive_refresh: bool = False,
        refresh_identity: RunIdentity | None = None,
    ) -> None:
        pool = self._observation_pool
        if pool is None or self._closing:
            return
        future = pool.submit(self._sources.observe, request)
        operation = _ObservationOperation(
            request,
            future,
            passive_refresh=passive_refresh,
            refresh_identity=refresh_identity,
        )
        self._observation = operation
        page_ref = weakref.ref(self)
        future.add_done_callback(
            lambda done: ScatteringWorkspace._deliver_observation(
                page_ref, operation, done
            )
        )

    def _queue_live_source_refresh(
        self, event: StandardRunEvent
    ) -> None:
        """Coalesce one Live discovery into one passive source observation."""

        source = self._live_source_refresh_source
        if source is None:
            return
        identity = event.run_identity
        snapshot = self._intents.snapshot()
        self._observation_token += 1
        pending = SourceObservationRequest(
            self._observation_token,
            snapshot.revision,
            source,
        )
        if not self._live_source_refresh_is_current(identity, pending):
            return
        if self._observation is not None:
            self._pending_source_refresh = pending
            return
        self._submit_observation(
            pending,
            passive_refresh=True,
            refresh_identity=identity,
        )

    def _live_source_refresh_is_current(
        self,
        identity: RunIdentity,
        request: SourceObservationRequest,
    ) -> bool:
        source = self._live_source_refresh_source
        if (
            self._closing
            or self._closed
            or self._observation_pool is None
            or source is None
            or request.source != source
            or self._lifecycle.active_run_identity is not identity
            or self._lifecycle.phase not in _LIVE_SOURCE_REFRESH_PHASES
        ):
            return False
        snapshot = self._intents.snapshot()
        intent = snapshot.thaw()
        return (
            intent.live_mode
            and type(intent.source_spec) is DirectorySourceSpec
            and intent.source_spec == source
        )

    def _launch_pending_source_refresh(self) -> None:
        if self._observation is not None:
            return
        pending, self._pending_source_refresh = (
            self._pending_source_refresh,
            None,
        )
        identity = self._lifecycle.active_run_identity
        if (
            pending is not None
            and identity is not None
            and self._live_source_refresh_is_current(identity, pending)
        ):
            self._submit_observation(
                pending,
                passive_refresh=True,
                refresh_identity=identity,
            )

    def _clear_live_source_refresh(self) -> None:
        self._pending_source_refresh = None
        self._live_source_refresh_source = None

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
            self._settle_observation(operation)
        finally:
            self._launch_pending_source_refresh()

    def _settle_observation(
        self, operation: _ObservationOperation
    ) -> None:
        request = operation.request
        if operation.passive_refresh:
            identity = operation.refresh_identity
            if identity is None or not self._live_source_refresh_is_current(
                identity, request
            ):
                return
        try:
            observation: object = operation.future.result()
        except Exception:
            if not operation.passive_refresh:
                self._source_status.show_unavailable(
                    _source_label(request.source)
                )
            return
        if type(observation) is not SourceObservation:
            if not operation.passive_refresh:
                self._source_status.show_unavailable(
                    _source_label(request.source)
                )
            return
        snapshot = self._intents.snapshot()
        expected_fingerprint = (
            operation.candidate_fingerprint
            if operation.preview
            else None
        )
        if not observation.qualifies(
            request, expected_fingerprint
        ):
            if not operation.passive_refresh:
                self._source_status.show_unavailable(
                    _source_label(request.source)
                )
            return
        if snapshot.thaw().source_spec != request.source:
            return
        if operation.passive_refresh:
            identity = operation.refresh_identity
            if identity is None or not self._live_source_refresh_is_current(
                identity, request
            ):
                return
            observation = self._retain_exact_motor_knowledge(observation)
        prior_observation = self._source_observation
        if (
            self._progress.terminal
            and prior_observation is not None
            and (
                prior_observation.candidate_fingerprint,
                prior_observation.status,
                prior_observation.exists,
                prior_observation.observed_file_count,
            )
            != (
                observation.candidate_fingerprint,
                observation.status,
                observation.exists,
                observation.observed_file_count,
            )
        ):
            self._progress = replace(
                self._progress,
                detail="",
                directory_files=None,
                terminal=False,
            )
        self._source_observation = observation
        self._maybe_default_gi_motor(observation)
        if operation.preview:
            self._sources.publish_motor_knowledge(observation)
        self._source_status.render(observation)
        self._refresh_shell()
        if (
            not operation.preview
            and not operation.passive_refresh
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

    def _retain_exact_motor_knowledge(
        self, observation: SourceObservation
    ) -> SourceObservation:
        fingerprint = observation.candidate_fingerprint
        if not fingerprint or observation.gi_motor_choices is not None:
            return observation
        knowledge = self._sources.project_motor_knowledge(
            observation.source,
            fingerprint,
        )
        if (
            type(knowledge) is SourceObservation
            and knowledge.source == observation.source
            and knowledge.candidate_fingerprint == fingerprint
            and knowledge.gi_motor_choices is not None
        ):
            return replace(
                observation,
                gi_motor_choices=knowledge.gi_motor_choices,
            )
        return observation

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
        *,
        preserve_terminal: bool = False,
    ) -> None:
        if (
            current.revision != prior.revision
            and self._progress.terminal
            and not preserve_terminal
        ):
            self._progress = replace(
                self._progress,
                detail="",
                directory_files=None,
                terminal=False,
            )
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
            self._clear_live_source_refresh()
            self._gi_auto_motor = _NO_AUTOMATIC_GI_MOTOR
            self._source_observation = None
            self._cancel_observation()
            self._request_observation(current)
        self._retry_deferred_gi_motor_default()
        ScatteringWorkspace._observe_operation_stamp(self, current.revision)

    def _choose_directory_dialog(
        self,
        _current: str,
        start_directory: str,
    ) -> str | None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open processed data folder",
            start_directory,
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
        project_root = snapshot.thaw().project_root
        start_directory = browse_start_dir(
            current,
            fallback=project_root,
        )
        try:
            selected = chooser(path, current, start_directory)
        except Exception as error:
            self._error_notice("Browse failed", error)
            return
        if type(selected) is not str or not selected:
            return
        remember_browse_path(selected)
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
        intent = self._intents.snapshot().thaw()
        current = intent.source_spec
        requested_mode = desired_mode or self._source_mode
        start_directory = browse_start_dir(
            _source_selection_path(current),
            fallback=intent.project_root,
        )
        try:
            selected = chooser(
                current,
                requested_mode,
                start_directory,
            )
        except Exception as error:
            self._error_notice("Choose source failed", error)
            return
        if selected is not None:
            remember_browse_path(_source_selection_path(selected))
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
        if self._auto_gi_motor_matches(snapshot):
            candidate.gi.incidence_motor = "Manual"
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
        project_root = self._intents.snapshot().thaw().project_root
        start_directory = browse_start_dir(
            self._browser_directory,
            fallback=project_root,
        )
        try:
            selected = self._browser_directory_chooser(
                self._browser_directory,
                start_directory,
            )
        except Exception as error:
            self._error_notice("Open folder failed", error)
            return
        if type(selected) is not str or not selected:
            return
        remember_browse_path(selected)
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
            inspect_directory_contents=self._date_sorted,
            directory_time_cache=self._browser_directory_time_cache,
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
