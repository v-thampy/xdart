"""Public composition root for the context-qualified scattering workspace."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any, Callable
import weakref

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
    Viewer1DContext,
    Viewer1DState,
    Viewer2DContext,
    Viewer2DState,
)
from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentFreezeAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
    RunIntentSnapshot,
)
from xrd_tools.session.run_intent_profile import (
    dump_run_intent_profile,
    load_run_intent_profile,
)
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    RunIntent,
    heavy_residency_choice,
)
from xrd_tools.reduction.provenance_config import jsonable_run_value
from xrd_tools.io.output_transaction import (
    StreamTerminal,
)
from xrd_tools.io.viewer_2d import Viewer2DArtifactCatalog, _stable_revision
from .results_notebook import results_notebook_text
from xrd_tools.session.readiness import Tool, tool_from_mode_text
from xrd_tools.session.display_logic import xye_prefix_for_unit
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    is_single_image_spec,
)
from xdart.utils.browse import browse_start_dir, remember_browse_path
from .advanced_editor import AdvancedSettingsDialog
from .adapters.browse_loader import BrowseLoader
from .adapters.external_operation import OperationSlot
from .batch_terminal_presentation import (
    FULL_READY,
    FULL_REFUSED,
    FULL_REQUESTED,
    PASS_THROUGH,
    QUALIFY,
    RETIRE,
    SELECT,
    BatchNavigationFacts,
    BatchTerminalDecision,
    BatchTerminalPresentation,
    BatchTerminalPresentationController,
)
from .browse_1d_display import (
    Browse1DDisplayRefusal,
    Browse1DReleaseDebt,
    prepare_browse_1d_display,
)
from .browse_1d_projection import (
    Browse1DBorrowBundle,
    Browse1DProjectionStatus,
)
from .browse_values import (
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
    LoadedBrowseCapture,
)
from .contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    RunExecutorPort,
    SourceCountScope,
    SourceObservation,
    SourcePort,
    SourceSelection,
)
from .context_controller import ContextController
from .context_projection import ContextProjection
from .authored_assets import (
    AuthoredAssetDialogCommand,
    AuthoredAssetDialogEffect,
    AuthoredAssetDialogIdentity,
    AuthoredAssetOwner,
    AuthoredAssetOwnerLifecycle,
    AuthoredAssetPhase,
    AuthoredAssetRefreshEffect,
    AuthoredAssetTransition,
)
from .controls_projection import (
    AdvancedSettingsValues,
    EditNoChange,
    EditRefusal,
    MASK_FILE,
    OUTPUT_MODE,
    PONI_FILE,
    PROJECT_ROOT,
    SOURCE_DIRECTORY,
    SOURCE_FILE,
    SOURCE_TYPE,
    reduce_advanced_settings,
    reduce_control_edit,
    project_controls,
)
from .controls_readiness import (
    ControlsReadinessProjection,
    SectionHeaderProjection,
)
from .coordinator import ScatteringCoordinator
from .detector_projection import detector_summary
from .display_values import (
    DisplayFrameKey,
    StandardDisplayPayload,
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
    prepare_calibration_request, prepare_mask_request,
    resolve_calibration_executable,
    resolve_mask_executable,
)
from .external_tools import (
    ExternalNexusQualification,
    ExternalToolId,
    ExternalToolRegistry,
    external_tool_id,
)
from .shell_projection import (
    ScientificPreferences,
    share_plot_axis_for_image,
)
from .scientific_axes import (
    native_1d_plot_axis,
    resolve_norm_presentation,
    slice_recipe_axes_compatible,
    slice_region_orientation,
)
from .scientific_plot_options import waterfall_should_be_active
from .shell_values import (
    ArtifactProgress,
    DirectoryFileProgress,
    FrameNavigationProjection,
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
    OperationContextStamp, OperationIdentity,
    OperationTerminalStatus, OperationUpdate,
)
from .metadata_operations import (
    MetadataDialogIdentity,
    MetadataDisposition,
    MetadataLifecycle,
    MetadataOperationOwner,
    MetadataProcess,
    MetadataRequest,
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
from .output_preflight import native_int_reduction_plan, _run_artifact_family
from xrd_tools.session.readiness import GI_COMPANION_MODES_2D_ARG
from .state_machine import RunPhase
from .source_selection import (
    SourceSelectionTransition,
    SourceObservationWake,
    SourceRefreshEffect,
    SourceSelectionOwner,
    SourceStatusDirective,
)
from .processed_browser import (
    BrowserCatalogRequest,
    BrowserCatalogWake,
    BrowserRefreshEffect,
    ProcessedBrowserOwner,
    ProcessedBrowserTransition,
    ReintegrateSuccessorAdoption,
    ReintegrateSuccessorDirective,
    ReintegrateSuccessorPhase,
    TerminalBrowseHandoff,
    TerminalBrowsePaintReceipt,
    TerminalBrowsePaintRequest,
    TerminalBrowsePresentation,
    TerminalPaintMode,
    TerminalRebindAuthorization,
)
from .workspace_shell import ScatteringWorkspaceShell
from .workspace_operations import (
    WorkspaceOperationOwner,
    WorkspaceRefreshEffect,
)


_LOG = logging.getLogger(__name__)
_LIVE_EVENT_DRAIN_INTERVAL_MS = 125
_DEFAULT_LIVE_PLOT_INTERVAL_MS = 250
_LIVE_PLOT_INTERVAL_ENV = "XDART_LIVE_PLOT_INTERVAL_MS"
_UNSAFE_UNFUNDED_STAGING_ENV = (
    "XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC"
)
_UNSAFE_UNFUNDED_STAGING_KEY = (
    "_post_g2_unfunded_staging_diagnostic_v1"
)
_NEXUS_ONLY_PERFORMANCE_KEYS = (
    "_post_g2_pipeline_v2",
    "_post_g2_output_diagnostics_v1",
    _UNSAFE_UNFUNDED_STAGING_KEY,
)
_BROWSER_CATALOG_REFRESH_INTERVAL_MS = 1500
_DEFERRED_DELETE_RETRY_INTERVAL_MS = 25


@dataclass(frozen=True, slots=True)
class _DeferredBrowseSelection:
    blocked_by: BrowseLoadRequest
    value: str
    is_directory: bool

    def __post_init__(self) -> None:
        if (
            type(self.blocked_by) is not BrowseLoadRequest
            or type(self.value) is not str
            or not self.value
            or type(self.is_directory) is not bool
        ):
            raise ValueError("deferred Browse selection is invalid")


def _live_plot_interval_ms() -> int:
    try:
        value = int(os.environ.get(
            _LIVE_PLOT_INTERVAL_ENV,
            str(_DEFAULT_LIVE_PLOT_INTERVAL_MS),
        ))
    except (TypeError, ValueError):
        value = _DEFAULT_LIVE_PLOT_INTERVAL_MS
    return max(_LIVE_EVENT_DRAIN_INTERVAL_MS, value)


def _browse_perf_enabled() -> bool:
    return (
        bool(os.environ.get("XDART_PERF"))
        or os.environ.get("XDART_PERF_QUARTILES", "").strip() == "1"
    )


def _browse_1d_cache_retry_needed(
    status: Browse1DProjectionStatus | None,
    *,
    transient: bool,
) -> bool:
    """Only incomplete work or separately-owned transient custody can poll."""

    return bool(
        transient or status is Browse1DProjectionStatus.INCOMPLETE
    )


def _drop_nexus_only_performance_options(intent: object) -> None:
    """Keep XYE-only GUI intents free of private NeXus pipeline controls."""

    run_options = getattr(intent, "run_options", None)
    if type(run_options) is not dict:
        raise TypeError("run options must be one mutable mapping")
    for key in _NEXUS_ONLY_PERFORMANCE_KEYS:
        run_options.pop(key, None)


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


def _terminal_frame_signature(
    frame: DisplayFrameKey,
    canonical_by_artifact: dict[str, str],
) -> tuple[str, int] | None:
    canonical = canonical_by_artifact.get(frame.artifact)
    return (
        None
        if canonical is None
        else (canonical, frame.local_frame_label)
    )


def _native_plot_axis_for_run(
    value: RunIntent | FrozenRunConfiguration,
) -> str | None:
    """Resolve one editable or frozen run's native 1-D plot selector."""

    if type(value) not in {RunIntent, FrozenRunConfiguration}:
        return None
    semantic = (
        value.gi.mode_1d
        if value.gi.enabled
        else value.bai_1d_args.get("unit", "q_A^-1")
    )
    return native_1d_plot_axis(semantic)


@dataclass(frozen=True, slots=True)
class _AdmissionPageOwner:
    token: AdmissionToken
    releasing: bool = False
    release_receipt: AdmissionReleased | None = None
    admission_receipt: AdmissionReceipt | None = None
    retirement_receipt: DisplayRetirementReceipt | None = None
    retirement_applied: bool = False


@dataclass(frozen=True, slots=True)
class _NativePlotAxisTransition:
    origin_axis: str
    target_axis: str
    followed_origin: bool
    run_identity: RunIdentity | None = None


class _AuthoredAssetDialog(QtWidgets.QDialog):
    """One nonblocking confirmation surface shared by PONI and mask authoring."""

    acceptRequested = QtCore.Signal(str)
    chooseRequested = QtCore.Signal()
    cancelRequested = QtCore.Signal()

    def __init__(
        self, asset: str, paths: tuple[str, ...],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("authoredAssetConfirmation")
        self.setWindowTitle(
            "Adopt detector calibration" if asset == "poni"
            else "Adopt detector mask"
        )
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._inert = False
        self._busy = False
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "Select the exact authored PONI to adopt:"
            if asset == "poni" else
            "Select the exact authored mask to adopt:",
            self,
        )
        layout.addWidget(label)
        self.paths = QtWidgets.QComboBox(self)
        self.paths.setObjectName("authoredAssetPath")
        for path in paths:
            self.paths.addItem(path, path)
        if not paths:
            self.paths.addItem("No new valid authored file was found.", None)
        self.paths.setToolTip(paths[0] if paths else "")
        layout.addWidget(self.paths)
        self.full_path = QtWidgets.QLineEdit(self)
        self.full_path.setObjectName("authoredAssetFullPath")
        self.full_path.setReadOnly(True)
        self.full_path.setText(paths[0] if paths else "")
        self.full_path.setToolTip(paths[0] if paths else "")
        self.paths.currentIndexChanged.connect(self._show_selected_path)
        layout.addWidget(self.full_path)
        buttons = QtWidgets.QHBoxLayout()
        self.accept_button = QtWidgets.QPushButton("Accept", self)
        self.accept_button.setObjectName("authoredAssetAccept")
        self.accept_button.setEnabled(bool(paths))
        self.choose_button = QtWidgets.QPushButton("Choose Another…", self)
        self.choose_button.setObjectName("authoredAssetChooseAnother")
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.setObjectName("authoredAssetCancel")
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.choose_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.accept_button.clicked.connect(
            lambda: self.acceptRequested.emit(self.selected_path or "")
        )
        self.choose_button.clicked.connect(self.chooseRequested.emit)
        self.cancel_button.clicked.connect(self.reject)
        self.rejected.connect(self._emit_cancel)

    @property
    def selected_path(self) -> str | None:
        value = self.paths.currentData()
        return value if type(value) is str and value else None

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        enabled = not busy
        self.paths.setEnabled(enabled and self.paths.count() > 1)
        self.accept_button.setEnabled(enabled and self.selected_path is not None)
        self.choose_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled)

    def _show_selected_path(self, _index: int) -> None:
        value = self.selected_path or ""
        self.paths.setToolTip(value)
        self.full_path.setText(value)
        self.full_path.setToolTip(value)

    def close_inert(self) -> None:
        self._inert = True
        self._busy = False
        self.close()

    def reject(self) -> None:
        if self._busy and not self._inert:
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._busy and not self._inert:
            event.ignore()
            return
        super().closeEvent(event)

    def _emit_cancel(self) -> None:
        if not self._inert:
            self.cancelRequested.emit()


class ScatteringWorkspace(QtWidgets.QWidget):
    """One command owner around one passive shell and one context controller."""

    sourceSelectionRequested = QtCore.Signal(object)
    browseRequested = QtCore.Signal(object)
    toolRequested = QtCore.Signal(str)
    noticeChanged = QtCore.Signal(str)
    _observationFinished = QtCore.Signal(object)
    _browserCatalogFinished = QtCore.Signal(object)

    def __init__(
        self,
        *,
        intents: RunIntentStore,
        lifecycle: ScatteringCoordinator,
        sources: SourcePort,
        executor: RunExecutorPort | None = None,
        viewer_1d_file_chooser: (
            Callable[[str], tuple[str, ...] | None] | None
        ) = None,
        viewer_2d_file_chooser: (
            Callable[[str], str | None] | None
        ) = None,
        browser_directory_chooser: (
            Callable[[str, str], str | None] | None
        ) = None,
        control_path_chooser: (
            Callable[[tuple[str, ...], str, str], str | None] | None
        ) = None,
        authoring_source_chooser: (
            Callable[[str, str], str | None] | None
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
        profile_path_chooser: (
            Callable[[str, str], str | None] | None
        ) = None,
        results_notebook_chooser: Callable[[str], str | None] | None = None,
        external_tool_registry: ExternalToolRegistry | None = None,
        browse_clock: Callable[[], float] = time.monotonic,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not callable(browse_clock):
            raise TypeError("terminal Browse clock must be callable")
        if (
            external_tool_registry is not None
            and type(external_tool_registry) is not ExternalToolRegistry
        ):
            raise TypeError("external viewer registry must be exact")
        self.setObjectName("scatteringWorkspace")
        self._intents = intents
        self._lifecycle = lifecycle
        self._run_executor = executor
        self._external_tools = (
            ExternalToolRegistry()
            if external_tool_registry is None
            else external_tool_registry
        )
        self._external_nexus_refusal: (
            tuple[LoadedBrowseCapture, ExternalNexusQualification] | None
        ) = None
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
        self._batch_terminal = BatchTerminalPresentationController()
        self._run_frame_seen = False
        self._retain_outgoing_display = False
        self._presentation_targets: deque[DisplayFrameKey] = deque(maxlen=1)
        self._presentation_run_identity: RunIdentity | None = None
        self._live_plot_interval_ms = _live_plot_interval_ms()
        self._last_live_plot_at: float | None = None
        self._scientific_repaint_pending = False
        self._waterfall_candidate_count = 0
        self._browse_1d_release_debt: Browse1DBorrowBundle | None = None
        self._quartile_refresh_identity: RunIdentity | None = None
        self._quartile_refresh_seconds = [0.0, 0.0, 0.0, 0.0]
        self._browser_directory_chooser = (
            browser_directory_chooser
            if browser_directory_chooser is not None
            else self._choose_directory_dialog
        )
        self._viewer_2d_file_chooser = (
            viewer_2d_file_chooser if viewer_2d_file_chooser is not None
            else self._choose_viewer_2d_dialog
        )
        self._viewer_1d_file_chooser = (
            viewer_1d_file_chooser if viewer_1d_file_chooser is not None
            else self._choose_viewer_1d_dialog
        )
        self._control_path_chooser = control_path_chooser
        self._authoring_source_chooser = (
            authoring_source_chooser
            if authoring_source_chooser is not None
            else self._choose_authoring_source_dialog
        )
        self._source_selection_chooser = source_selection_chooser
        self._profile_path_chooser = (
            profile_path_chooser
            if profile_path_chooser is not None
            else self._choose_profile_path_dialog
        )
        self._results_notebook_chooser = (
            results_notebook_chooser
            if results_notebook_chooser is not None
            else self._choose_results_notebook_dialog
        )
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
        initial = intents.snapshot()
        initial_intent = initial.thaw()
        page_ref = weakref.ref(self)
        self._source_selection = SourceSelectionOwner(
            sources,
            intents,
            lambda wake: ScatteringWorkspace._deliver_observation(
                page_ref, wake
            ),
        )
        self._processed_browser = ProcessedBrowserOwner(
            save_path=initial_intent.save_path,
            processing_mode=initial_intent.processing_mode,
            deliver=lambda wake: ScatteringWorkspace._deliver_browser_catalog(
                page_ref, wake
            ),
            clock=browse_clock,
        )
        self._deferred_browse_selection: _DeferredBrowseSelection | None = None
        self._pending_viewer_2d_path: str | None = None
        self._admission_state: _AdmissionPageOwner | None = None
        self._closing = False
        self._closed = False
        self._deferred_delete_pending = False
        self._deferred_delete_reposted = False
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
        self._native_plot_axis_transition: (
            _NativePlotAxisTransition | None
        ) = None
        self._detector_scope_owner = None
        self._detector_demand_frame = None
        self._last_scientific_projection = None
        self._rendered_image_axis = self._preferences.image_axis
        self._detector_summary_key: tuple[str, str] | None = None
        self._detector_summary_text = "not configured"
        self._authoring_dependency_availability: tuple[bool, bool] | None = None
        self._project_readiness_key: tuple[str, str] | None = None
        self._project_readiness = SectionHeaderProjection("")
        self._experiment_readiness_key: tuple[bool, str] | None = None
        self._experiment_readiness = SectionHeaderProjection("")
        self._controls_readiness = ControlsReadinessProjection()
        self._artifact_progress: dict[str, ArtifactProgress] = {}
        self._progress = ProgressProjection()
        self._notice_text = ""

        self._context_projection = ContextProjection()
        self._browse_loader = BrowseLoader()
        self._context_controller = ContextController(
            lifecycle=lifecycle,
            executor=executor,
            browse_loader=self._browse_loader,
            projection=self._context_projection,
        )
        self._workspace_operations = WorkspaceOperationOwner()
        self._authored_assets = AuthoredAssetOwner(intents)
        self._authored_asset_dialog: _AuthoredAssetDialog | None = None
        self._authored_asset_dialog_identity: (
            AuthoredAssetDialogIdentity | None
        ) = None
        self._analysis_slot = OperationSlot()
        self._metadata_operations = MetadataOperationOwner()
        self._background_owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
        self._background_identity: OperationIdentity | None = None
        self._metadata_dialog = self._scan_roi_dialog = None
        self._peak_dialog = self._phase_dialog = None
        self._scan_roi_generation = 0
        self._peak_generation = self._phase_generation = 0
        self._analysis_identity = None; self._analysis_kind = None
        self._analysis_target = None
        self._analysis_generation = None; self._analysis_anchor = None
        self._analysis_fingerprint = ""; self._analysis_request = None
        self._roi_preview_binding = None
        self._metadata_result = self._scan_roi_result = None
        self._peak_result = self._phase_result = None

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
        self._deferred_delete_retry_timer = QtCore.QTimer(self)
        self._deferred_delete_retry_timer.setSingleShot(True)
        self._deferred_delete_retry_timer.setInterval(
            _DEFERRED_DELETE_RETRY_INTERVAL_MS
        )
        # This connection must survive _finalize_close long enough to repost
        # the one DeferredDelete event that was held for terminal cleanup.
        self._deferred_delete_retry_timer.timeout.connect(
            self._retry_deferred_delete
        )
        self._connect(
            self._shell.commandRequested, self._handle_shell_command
        )
        self._connect(
            self._source_status.chooseRequested,
            self._choose_source_selection,
        )
        self._observationFinished.connect(
            self._on_observation,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._connections.append(
            (self._observationFinished, self._on_observation)
        )
        self._browserCatalogFinished.connect(
            self._on_browser_catalog,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._connections.append(
            (self._browserCatalogFinished, self._on_browser_catalog)
        )
        self._connect(self._run_timer.timeout, self._drain_executor)
        self._connect(
            self._browser_catalog_timer.timeout,
            self._poll_browser_catalog,
        )

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

    def _experiment_operation_busy(self) -> bool:
        return (self._workspace_operations.busy
                or self._processed_browser.busy
                or self._authored_assets.busy
        )

    def _analysis_operation_busy(self) -> bool:
        return bool(
            self._analysis_slot.owned
            or self._analysis_identity is not None
            or self._metadata_operations.busy
        )

    def _mutating_operation_busy(self) -> bool:
        return (
            self._experiment_operation_busy()
            or self._analysis_operation_busy()
        )

    def _observe_operation_stamp(self, revision: int | None = None) -> None:
        operations = getattr(self, "_workspace_operations", None)
        if operations is not None:
            if revision is None:
                revision = self._intents.snapshot().revision
            operations.observe_stamp(
                ScatteringWorkspace._operation_context_stamp(
                    self, revision
                ),
                intent_revision=revision,
            )

        analysis_slot = getattr(self, "_analysis_slot", None)
        metadata = self._metadata_operations.active
        analysis = (
            metadata.identity
            if metadata is not None
            else getattr(self, "_analysis_identity", None)
        )
        if (
            analysis_slot is not None
            and analysis is not None
            and analysis_slot.current_identity is analysis
        ):
            kind = (
                metadata.request.kind
                if metadata is not None
                else getattr(self, "_analysis_kind", None)
            )
            if kind in {
                "metadata", "metadata_requalification", "scan_plot",
                "roi_preview", "roi_scan",
            }:
                if revision is None:
                    revision = self._intents.snapshot().revision
                stamp = OperationContextStamp(revision)
            else:
                stamp = ScatteringWorkspace._operation_context_stamp(
                    self, revision
                )
            analysis_slot.observe_stamp(stamp)

    def _analysis_generation_for(self, kind: str) -> int:
        if kind in {"metadata", "metadata_requalification"}:
            dialog = self._metadata_operations.dialog_identity("metadata")
            return -1 if dialog is None else dialog.generation
        return (
                self._scan_roi_generation if kind in {"scan_roi", "scan_plot", "roi_preview", "roi_scan"} else
                self._peak_generation if kind == "peak" else self._phase_generation
        )

    def _metadata_plan_for_current(self):
        controller = self._context_controller
        navigation = controller.navigation
        current = navigation.current
        if current is None:
            return None
        from xrd_tools.analysis.scan_operations import MetadataTablePlan
        processing_mode = self._intents.snapshot().thaw().processing_mode
        source = current.artifact
        metadata_format = None
        if (
            tool_from_mode_text(processing_mode) is Tool.IMAGE_VIEWER
            and current.source_scan == "viewer-2d"
            and current.artifact == "viewer-2d"
        ):
            context = controller.viewer_2d_context
            selection = controller.selection
            frame = controller.viewer_2d_frame
            original_path = getattr(context, "original_path", None)
            canonical_path = (
                os.path.abspath(os.path.expanduser(original_path))
                if type(original_path) is str and original_path
                else None
            )
            if not (
                controller.viewer_2d_owned
                and context is controller._runtime._viewer_2d
                and selection is not None
                and selection.kind is ContextKind.VIEWER_2D
                and selection.names(context)
                and navigation.current is current
                and controller.owns_frame(current)
                and frame is not None
                and getattr(frame, "label", None)
                == current.local_frame_label
                and canonical_path == original_path
                and Path(original_path).suffix.casefold()
                in {".tif", ".tiff"}
            ):
                return None
            source = original_path
            metadata_format = "auto"
        return MetadataTablePlan(
            source,
            metadata_format=metadata_format,
        )

    def _current_analysis_request(self, kind, target):
        from .analysis_mount import (analysis_request_facts,
            metadata_plan_from_syntax, phase_wavelength_angstrom, scan_plot_plan)
        try:
            if kind in {"metadata", "metadata_requalification"}:
                if target == "metadata":
                    plan = self._metadata_plan_for_current()
                else:
                    dialog = self._scan_roi_dialog
                    if dialog is None: return None
                    plan = metadata_plan_from_syntax(
                        dialog.source_widget.external_source_syntax())
                return None if plan is None else analysis_request_facts(plan)
            dialog = self._scan_roi_dialog
            if kind == "scan_plot" and dialog is not None:
                x, y, norm, right, log, table, roi = dialog.vnext_plot_values()
                if table is None: return None
                return analysis_request_facts(scan_plot_plan(x, y, norm),
                    table=table, roi=roi, render=(right, log))
            if kind in {"roi_preview", "roi_scan"} and dialog is not None:
                table = dialog._vnext_table_result
                if table is None: return None
                if kind == "roi_preview":
                    if not table.labels: return None
                    from xrd_tools.analysis.scan_operations import RoiPreviewPlan
                    return analysis_request_facts(RoiPreviewPlan.from_table(
                        table, label=table.labels[0]))
                picker = dialog._roi_dialog
                binding = self._roi_preview_binding
                if (picker is None or binding is None or binding[0] is not table
                        or binding[-1] is not picker): return None
                from xrd_tools.analysis.scan_operations import RoiScanPlan
                plan = RoiScanPlan.from_table(table,
                    signals=tuple(picker.roi_signals()),
                    mask_saturation=picker.mask_saturated())
                return analysis_request_facts(
                    plan, picker=(*binding[1:-1], id(picker)))
            if kind in {"peak", "phase"}:
                dialog = self._peak_dialog if kind == "peak" else self._phase_dialog
                trace, anchor = self._capture_analysis_trace(kind)
                if dialog is None or trace is None or anchor is None: return None
                values = dialog.vnext_fit_values()
                if kind == "peak":
                    from xrd_tools.analysis import DisplayedPeakFitPlan
                    plan = DisplayedPeakFitPlan(trace, **values)
                else:
                    from xrd_tools.analysis import DisplayedPhaseFitPlan
                    wavelength = phase_wavelength_angstrom(
                        self._context_controller, anchor.frame)
                    plan = DisplayedPhaseFitPlan(
                        trace, wavelength_angstrom=wavelength, **values)
                return analysis_request_facts(plan)
        except (AttributeError, TypeError, ValueError):
            return None
        return None

    def _metadata_dialog_for(
        self, identity: object,
    ) -> object | None:
        if (
            type(identity) is not MetadataDialogIdentity
            or self._metadata_operations.dialog_identity(identity.target)
            is not identity
        ):
            return None
        return getattr(self, f"_{identity.target}_dialog", None)

    def _set_metadata_dialog_status(
        self, identity: object, message: object,
    ) -> None:
        dialog = self._metadata_dialog_for(identity)
        status = None if dialog is None else getattr(dialog, "status", None)
        if status is not None:
            status.setText(str(message or ""))

    def _finish_metadata_refresh(
        self, identity: object, message=None,
    ) -> WorkspaceRefreshEffect:
        if message is not None:
            self._set_metadata_dialog_status(identity, message)
        # Metadata never locks editable fields or repaints science.  Its final
        # transition does, however, release the mutually-exclusive Run and
        # mutating-action affordances through one controls-only projection.
        # A newer deferred request owns that handoff and either starts with its
        # own projection or is dropped with a controls effect by dispatch.
        return (
            WorkspaceRefreshEffect.DIALOG
            if (
                self._analysis_operation_busy()
                or self._metadata_operations.deferred is not None
            )
            else WorkspaceRefreshEffect.CONTROLS
        )

    def _begin_analysis(self, kind, plan, generation, *, target=None,
                        request=None, anchor=None, table=None, roi=None):
        if (
            kind in {"metadata", "metadata_requalification"}
            or self._authored_assets.busy
            or self._analysis_identity is not None
        ):
            return None
        from .analysis_mount import (analysis_request_facts,
            analysis_start_allowed, display_anchor_matches)
        target = kind if target is None else target
        request = analysis_request_facts(plan) if request is None else request
        if (not analysis_start_allowed(self)
                or request != self._current_analysis_request(kind, target)):
            return None
        fit = kind in {"peak", "phase"}; fingerprint = (
            plan.trace.trace_fingerprint if fit else "")
        if fit and not display_anchor_matches(
                self, anchor, fingerprint, generation): return None
        stamp = (self._operation_context_stamp() if fit else
                 OperationContextStamp(self._intents.snapshot().revision))
        names = {"roi_preview": "begin_roi_preview",
                 "roi_scan": "begin_roi_scan", "peak": "begin_peak_fit",
                 "phase": "begin_phase_fit"}
        identity = (self._analysis_slot.begin_scan_plot(plan, table, roi, stamp)
                    if kind == "scan_plot" else
                    getattr(self._analysis_slot, names[kind])(plan, stamp))
        if identity is None: return None
        self._analysis_kind, self._analysis_generation = kind, generation
        self._analysis_target, self._analysis_identity = target, identity
        self._analysis_anchor, self._analysis_fingerprint = anchor, fingerprint
        self._analysis_request = request
        self._ensure_timer()
        message = f"Running {kind.replace('_', ' ')}…"
        self._notice(message)
        self._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        return identity

    def _metadata_context_stamp(self) -> OperationContextStamp:
        return OperationContextStamp(self._intents.snapshot().revision)

    def _classify_metadata_request(
        self, captured: MetadataRequest,
    ) -> MetadataDisposition:
        controller = self._context_controller
        kind = captured.kind
        from .analysis_mount import analysis_start_allowed
        return self._metadata_operations.classify(
            captured,
            current_request=self._current_analysis_request(
                kind, captured.target,
            ),
            blocked=bool(
                self._analysis_slot.owned
                or self._analysis_identity is not None
                or self._authored_assets.busy
                or controller.browse_pending
                or controller.viewer_1d_cleanup_pending
                or controller.viewer_2d_cleanup_pending
            ),
            start_allowed=analysis_start_allowed(self),
            closing=self._closing or self._closed,
        )

    def _start_metadata_request(
        self, captured: MetadataRequest,
    ) -> OperationIdentity | None:
        if self._classify_metadata_request(captured) is not MetadataDisposition.READY:
            return None
        method = (
            self._analysis_slot.begin_metadata_requalification
            if captured.kind == "metadata_requalification"
            else self._analysis_slot.begin_metadata
        )
        context = self._metadata_context_stamp()
        identity = method(captured.plan, context)
        if identity is None:
            return None
        process = self._metadata_operations.start(
            captured, identity, context,
        )
        if process is None:
            self._analysis_slot.cancel(identity)
            return None
        self._ensure_timer()
        message = f"Running {captured.kind.replace('_', ' ')}…"
        self._notice(message)
        self._set_metadata_dialog_status(captured.dialog, message)
        self._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        return identity

    def _submit_metadata(
        self, plan, *, target="metadata", request=None, candidate=None,
    ):
        from .analysis_mount import analysis_request_facts
        request = analysis_request_facts(plan) if request is None else request
        if candidate is not None:
            from xrd_tools.analysis.scan_operations import (
                MetadataTableRequalificationPlan,
            )
            plan = MetadataTableRequalificationPlan(
                candidate.receipt, candidate.table_fingerprint,
            )
        dialog = self._metadata_operations.dialog_identity(target)
        captured = self._metadata_operations.capture(
            plan,
            request,
            target=target,
            dialog=dialog,
            candidate=candidate,
        )
        if captured is None:
            return None
        # Install the newest exact request before classifying it.  A transient
        # Browse/viewer cleanup or analysis-slot race must not make the first
        # click disappear, and a newer cross-target request always supersedes
        # an older one.
        disposition = self._classify_metadata_request(captured)
        if disposition is MetadataDisposition.TRANSIENT:
            self._notice("Metadata queued…")
            self._ensure_timer()
            return None
        if disposition is not MetadataDisposition.READY:
            self._metadata_operations.drop(captured)
            return None
        identity = self._start_metadata_request(captured)
        if identity is not None:
            return identity
        # begin_* owns the final CAS.  If another owner won between the
        # readiness check and that CAS, retain the exact request only for a
        # recognized transient blocker.
        if (
            self._classify_metadata_request(captured)
            is not MetadataDisposition.TRANSIENT
        ):
            self._metadata_operations.drop(captured)
        else:
            self._notice("Metadata queued…")
            self._ensure_timer()
        return None

    def _dispatch_deferred_metadata(self) -> WorkspaceRefreshEffect:
        captured = self._metadata_operations.deferred
        if captured is None:
            return WorkspaceRefreshEffect.NONE
        disposition = self._classify_metadata_request(captured)
        if disposition is MetadataDisposition.TRANSIENT:
            return WorkspaceRefreshEffect.NONE
        if disposition is not MetadataDisposition.READY:
            self._metadata_operations.drop(captured)
            return self._finish_metadata_refresh(
                captured.dialog,
                "Metadata request is no longer current.",
            )
        identity = self._start_metadata_request(captured)
        if identity is not None:
            return WorkspaceRefreshEffect.DIALOG
        if (
            self._classify_metadata_request(captured)
            is not MetadataDisposition.TRANSIENT
        ):
            self._metadata_operations.drop(captured)
            return self._finish_metadata_refresh(
                captured.dialog,
                "Metadata request is no longer current.",
            )
        self._ensure_timer()
        return WorkspaceRefreshEffect.NONE

    def _capture_analysis_trace(self, kind):
        from .analysis_mount import capture_display_anchor, displayed_trace_input
        current = self._context_controller.navigation.current
        projection = self._last_scientific_projection
        if current is None or projection is None: return None, None
        try: trace = displayed_trace_input(projection, current)
        except (TypeError, ValueError): return None, None
        generation = self._analysis_generation_for(kind)
        anchor = capture_display_anchor(self, trace.trace_fingerprint, generation)
        return (trace, anchor) if anchor is not None else (None, None)

    def _analysis_dialog_closed(self, target, dialog):
        field = f"_{target}_dialog"
        if getattr(self, field, None) is not dialog: return
        setattr(self, field, None)
        if target in {"metadata", "scan_roi"}:
            from .analysis_mount import cancel_owned
            token = self._metadata_operations.dialog_identity(target)
            cancel_owned(
                self._analysis_slot,
                self._metadata_operations.close_dialog(token),
            )
        if target != "metadata":
            generation = f"_{target}_generation"
            setattr(self, generation, getattr(self, generation) + 1)
        if self._analysis_target == target:
            from .analysis_mount import cancel_owned
            cancel_owned(self._analysis_slot, self._analysis_identity)
        if target == "scan_roi":
            self._roi_preview_binding = self._scan_roi_result = None
        elif target == "peak": self._peak_result = None
        elif target == "phase": self._phase_result = None

    def _open_analysis_mount(self, target, *, focus_roi=False):
        field = f"_{target}_dialog"; dialog = getattr(self, field)
        created = dialog is None
        if created:
            if target == "metadata":
                from .analysis_mount import MetadataResultDialog
                dialog = MetadataResultDialog(self)
            elif target == "scan_roi":
                from xdart.gui.analysis.scan_plot_dialog import ScanPlotDialog
                dialog = ScanPlotDialog(parent=self, vnext_submit=self._scan_analysis_action)
            elif target == "peak":
                from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
                dialog = PeakFitDialog(parent=self, vnext_submit=self._peak_analysis_action)
            else:
                from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
                dialog = PhaseFitDialog(parent=self, vnext_submit=self._phase_analysis_action)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            setattr(self, field, dialog)
            if target in {"metadata", "scan_roi"}:
                if self._metadata_operations.open_dialog(target) is None:
                    dialog.close()
                    setattr(self, field, None)
                    return
            dialog.finished.connect(lambda _value, t=target, d=dialog:
                                    self._analysis_dialog_closed(t, d))
        dialog.show(); dialog.raise_(); dialog.activateWindow()
        if target in {"peak", "phase"}: self._refresh_fit_trace(target)
        elif target == "metadata":
            from .analysis_mount import analysis_result_matches
            current_request = self._current_analysis_request(
                "metadata", "metadata"
            )
            retained = self._metadata_result
            if (
                retained is not None
                and analysis_result_matches(retained, current_request)
            ):
                dialog.clear_result()
                self._submit_metadata(
                    None, target="metadata",
                    request=current_request, candidate=retained,
                )
            else:
                self._metadata_result = None
                dialog.clear_result()
                self._metadata_from_current()
        elif target == "scan_roi":
            if created and self._metadata_result is not None:
                dialog.set_vnext_metadata(self._metadata_result)
            if focus_roi and dialog._vnext_table_result is None:
                dialog.status.setText("Choose a source, then Plot ROI.")

    def _metadata_from_current(self):
        plan = self._metadata_plan_for_current()
        if plan is None:
            self._notice("Select a retained frame first."); return
        from .analysis_mount import analysis_request_facts
        request = analysis_request_facts(plan)
        self._submit_metadata(
            plan,
            request=request,
        )

    def _scan_analysis_action(self, action, value):
        if action == "close":
            self._analysis_dialog_closed("scan_roi", self._scan_roi_dialog); return
        if action == "metadata":
            from .analysis_mount import analysis_request_facts, metadata_plan_from_syntax
            plan = metadata_plan_from_syntax(value)
            if plan is not None:
                self._submit_metadata(
                    plan, target="scan_roi",
                    request=analysis_request_facts(plan),
                )
            return
        if action == "scan_plot" and value:
            from .analysis_mount import analysis_request_facts, scan_plot_plan
            x, y, norm, right, log, table, roi = value
            plan = scan_plot_plan(x, y, norm)
            request = analysis_request_facts(
                plan, table=table, roi=roi, render=(right, log))
            self._begin_analysis("scan_plot", plan, self._scan_roi_generation,
                target="scan_roi", request=request, table=table, roi=roi); return
        if action == "roi_preview" and value is not None and value.labels:
            from .analysis_mount import analysis_request_facts
            from xrd_tools.analysis.scan_operations import RoiPreviewPlan
            plan = RoiPreviewPlan.from_table(value, label=value.labels[0])
            self._begin_analysis("roi_preview", plan, self._scan_roi_generation,
                target="scan_roi", request=analysis_request_facts(plan))

    def _refresh_fit_trace(self, kind):
        trace, _anchor = self._capture_analysis_trace(kind)
        dialog = self._peak_dialog if kind == "peak" else self._phase_dialog
        if trace is None or dialog is None:
            self._notice("Select one displayed 1-D trace first."); return None
        dialog.set_vnext_trace(trace); return trace

    def _peak_analysis_action(self, action, values):
        if action == "close":
            self._analysis_dialog_closed("peak", self._peak_dialog); return
        if action == "reload": self._refresh_fit_trace("peak"); return
        if action != "fit": return
        trace, anchor = self._capture_analysis_trace("peak")
        if trace is None: return
        from xrd_tools.analysis import DisplayedPeakFitPlan
        from .analysis_mount import analysis_request_facts
        plan = DisplayedPeakFitPlan(trace, **values)
        self._begin_analysis("peak", plan, self._peak_generation, anchor=anchor,
                             request=analysis_request_facts(plan))

    def _phase_analysis_action(self, action, values):
        if action == "close":
            self._analysis_dialog_closed("phase", self._phase_dialog); return
        if action == "reload": self._refresh_fit_trace("phase"); return
        if action != "fit": return
        trace, anchor = self._capture_analysis_trace("phase")
        from .analysis_mount import phase_wavelength_angstrom
        wavelength = (None if anchor is None else phase_wavelength_angstrom(
            self._context_controller, anchor.frame))
        if trace is None: return
        from xrd_tools.analysis import DisplayedPhaseFitPlan
        plan = DisplayedPhaseFitPlan(trace, wavelength_angstrom=wavelength, **values)
        from .analysis_mount import analysis_request_facts
        self._begin_analysis("phase", plan, self._phase_generation, anchor=anchor,
                             request=analysis_request_facts(plan))

    def _roi_analysis_signals(self, table, picker, signals):
        binding = self._roi_preview_binding
        if (not signals or self._scan_roi_dialog is None
                or self._scan_roi_dialog._vnext_table_result is not table
                or self._scan_roi_dialog._roi_dialog is not picker
                or binding is None or binding[0] is not table
                or binding[-1] is not picker): return
        from .analysis_mount import analysis_request_facts
        from xrd_tools.analysis.scan_operations import RoiScanPlan
        plan = RoiScanPlan.from_table(table, signals=tuple(signals),
                                     mask_saturation=picker.mask_saturated())
        self._begin_analysis("roi_scan", plan, self._scan_roi_generation,
            target="scan_roi", request=analysis_request_facts(
                plan, picker=(*binding[1:-1], id(picker))))

    def _consume_metadata_update(
        self, update: OperationUpdate, process: MetadataProcess,
    ) -> WorkspaceRefreshEffect:
        captured = process.request
        dialog = self._metadata_dialog_for(captured.dialog)
        current = self._metadata_operations.process_is_current(
            process,
            current_request=self._current_analysis_request(
                captured.kind, captured.target,
            ),
            current_context=self._metadata_context_stamp(),
        )
        newer = self._metadata_operations.has_newer_request(process)
        if update.terminal is None:
            if (
                update.progress is not None
                and current
                and not update.stale
                and not newer
            ):
                message = (
                    f"Analysis: {update.progress.stage} "
                    f"{update.progress.completed}/{update.progress.total}"
                )
                self._notice(message)
                self._set_metadata_dialog_status(captured.dialog, message)
            return WorkspaceRefreshEffect.DIALOG

        from xrd_tools.analysis.scan_operations import (
            AnalysisDisposition,
            MetadataTableRequalificationResult,
            MetadataTableResult,
        )
        terminal_payload = update.terminal.payload
        if (
            current
            and not update.stale
            and not newer
            and captured.kind == "metadata"
            and captured.target == "scan_roi"
            and type(terminal_payload) is MetadataTableResult
            and terminal_payload.disposition is AnalysisDisposition.REFUSED
            and terminal_payload.code == "SOURCE_SELECTION_REQUIRED"
            and dialog is not None
        ):
            self._metadata_operations.finish(update.identity)
            self._roi_preview_binding = self._scan_roi_result = None
            dialog.clear_vnext_metadata()
            if terminal_payload.candidates:
                dialog.source_widget.set_external_candidates(
                    terminal_payload.candidates,
                )
            message = "Choose one headless-qualified source."
            self._notice(message)
            return self._finish_metadata_refresh(captured.dialog, message)

        from .analysis_mount import (
            analysis_result_matches,
            retention_admission,
            terminal_adoption,
        )
        payload, diagnostic = terminal_adoption(update, current=current)
        candidate = captured.candidate
        requalification = captured.kind == "metadata_requalification"
        if requalification:
            valid = (
                type(payload) is MetadataTableRequalificationResult
                and type(candidate) is MetadataTableResult
                and payload.receipt == candidate.receipt
                and payload.table_fingerprint == candidate.table_fingerprint
                and analysis_result_matches(candidate, captured.request)
            )
            if valid:
                payload = candidate
            else:
                payload, diagnostic = (
                    None,
                    diagnostic
                    or "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH",
                )
        elif payload is not None and not analysis_result_matches(
            payload, captured.request,
        ):
            payload = None
            diagnostic = "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH"

        if self._metadata_operations.finish(update.identity) is not process:
            return WorkspaceRefreshEffect.NONE
        if newer:
            # A later request owns both initial and requalified precedence.
            return WorkspaceRefreshEffect.DIALOG
        if payload is None:
            self._notice(diagnostic)
            return self._finish_metadata_refresh(
                captured.dialog, diagnostic,
            )
        if not requalification:
            identity = self._submit_metadata(
                None,
                target=captured.target,
                request=captured.request,
                candidate=payload,
            )
            if (
                identity is None
                and self._metadata_operations.deferred is None
            ):
                return self._finish_metadata_refresh(captured.dialog)
            return WorkspaceRefreshEffect.DIALOG

        retained = {
            "metadata": self._metadata_result,
            "scan_roi": None,
            "peak": self._peak_result,
            "phase": self._phase_result,
        }
        admitted, reason = retention_admission(
            retained, "metadata", payload,
        )
        if not admitted:
            self._notice(reason)
            return self._finish_metadata_refresh(captured.dialog, reason)
        self._roi_preview_binding = None
        self._metadata_result = payload
        self._scan_roi_result = None
        if captured.target == "metadata" and dialog is not None:
            dialog.adopt_result(payload)
            if self._scan_roi_dialog is not None:
                self._scan_roi_dialog.clear_vnext_metadata()
        elif captured.target == "scan_roi" and dialog is not None:
            dialog.set_vnext_metadata(payload)
        self._notice("")
        return self._finish_metadata_refresh(captured.dialog)

    def _consume_analysis_update(self, update):
        metadata = self._metadata_operations.active
        if (
            type(update) is OperationUpdate
            and metadata is not None
            and update.identity is metadata.identity
        ):
            return self._consume_metadata_update(update, metadata)
        if (
            type(update) is not OperationUpdate
            or update.identity is not self._analysis_identity
        ):
            return WorkspaceRefreshEffect.NONE
        if update.terminal is None:
            if update.progress is not None:
                message = (
                    f"Analysis: {update.progress.stage} "
                    f"{update.progress.completed}/{update.progress.total}"
                )
                self._notice(message)
            return WorkspaceRefreshEffect.FULL

        kind, target = self._analysis_kind, self._analysis_target
        request = self._analysis_request
        current = self._analysis_generation == self._analysis_generation_for(
            target,
        )
        current = bool(
            current
            and request is not None
            and request == self._current_analysis_request(kind, target)
        )
        if kind in {"peak", "phase"}:
            from .analysis_mount import display_anchor_matches
            current = current and display_anchor_matches(
                self,
                self._analysis_anchor,
                self._analysis_fingerprint,
                self._analysis_generation,
            )
        from .analysis_mount import (
            analysis_result_matches,
            retention_admission,
            terminal_adoption,
        )
        payload, diagnostic = terminal_adoption(update, current=current)
        if payload is not None and not analysis_result_matches(payload, request):
            payload = None
            diagnostic = "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH"
        self._analysis_identity = self._analysis_kind = self._analysis_target = None
        self._analysis_generation = self._analysis_anchor = None
        self._analysis_request = None
        self._analysis_fingerprint = ""
        if payload is None:
            self._notice(diagnostic)
            return WorkspaceRefreshEffect.FULL

        retained = {
            "metadata": self._metadata_result,
            "scan_roi": self._scan_roi_result,
            "peak": self._peak_result,
            "phase": self._phase_result,
        }
        replacing = (
            "scan_roi"
            if kind in {"scan_plot", "roi_preview", "roi_scan"}
            else kind
        )
        admitted, reason = retention_admission(retained, replacing, payload)
        if not admitted:
            self._notice(reason)
            return WorkspaceRefreshEffect.FULL
        if kind in {"scan_plot", "roi_preview", "roi_scan"}:
            self._roi_preview_binding = None
            if self._scan_roi_dialog is not None:
                self._scan_roi_dialog._retire_vnext_result()
        if kind == "scan_plot":
            self._scan_roi_result = payload
            if self._scan_roi_dialog is not None: self._scan_roi_dialog.set_vnext_scan_result(payload, request[2])
        elif kind == "roi_preview":
            self._scan_roi_result = payload
            if self._scan_roi_dialog is not None and payload.image is not None:
                from xdart.gui.analysis.roi_select_dialog import RoiSelectDialog
                old = self._scan_roi_dialog._roi_dialog
                if old is not None: old.close(); old.deleteLater()
                table = self._scan_roi_dialog._vnext_table_result
                picker = RoiSelectDialog(payload.image, parent=self._scan_roi_dialog)
                picker.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
                self._scan_roi_dialog._roi_dialog = picker
                self._roi_preview_binding = (table, payload.receipt,
                    payload.table_fingerprint, payload.result_fingerprint,
                    payload.label, tuple(payload.labels), picker)
                picker.sigCompute.connect(lambda signals, t=table, p=picker:
                                          self._roi_analysis_signals(t, p, signals))
                picker.show(); picker.raise_(); picker.activateWindow()
        elif kind == "roi_scan":
            self._scan_roi_result = payload
            if self._scan_roi_dialog is not None:
                self._scan_roi_dialog.set_vnext_roi_result(payload)
        elif kind == "peak":
            self._peak_result = payload
            if self._peak_dialog is not None: self._peak_dialog.set_vnext_result(payload)
        else:
            self._phase_result = payload
            if self._phase_dialog is not None: self._phase_dialog.set_vnext_result(payload)
        self._notice("")
        return WorkspaceRefreshEffect.FULL

    @staticmethod
    def _background_domain(mode: str) -> str | None:
        return {"Int 1D": "integrated_1d", "Int 1D (XYE)": "integrated_1d", "Int 2D": "integrated_2d", "1D Viewer": "integrated_1d", "2D Viewer": "raw"}.get(mode)

    def _background_key(self, plan, stamp, mode: str, target_facts) -> tuple[object, ...]:
        return (stamp.context_token, stamp.display_generation, plan.domain, mode,
                "raw" if plan.domain == "raw" else "integrated",
                plan.contributor_ids, plan.value_shapes, plan.axis_shapes, target_facts)

    def _background_action(self) -> None:
        owner, operations = self._background_owner, self._workspace_operations
        if self._analysis_operation_busy():
            self._notice(
                "Display background is unavailable while analysis is active."
            )
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
            return
        background_cancel = (
            self._background_identity is not None
            and operations.current_identity is self._background_identity
        )
        if self._experiment_operation_busy() and not background_cancel:
            self._notice("Display background is unavailable while another operation is active.")
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
            return
        if owner.phase in {"RESERVED", "STAGED", "ACTIVE", "CLEANUP_PENDING"}:
            self._release_display_background(); self._notice("Clearing display background…")
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
        identity = operations.begin_background(
            plan, stamp, owner, reservation
        )
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
        if identity is not None and self._workspace_operations.current_identity is identity:
            self._workspace_operations.cancel(identity)
        self._ensure_timer(); return False

    def _calibrate_action(self) -> None:
        operations = self._workspace_operations
        authored = self._authored_assets
        identity = authored.operation_identity
        if (authored.phase is AuthoredAssetPhase.RUNNING
                and authored.asset == "poni"
                and identity is not None
                and operations.current_identity is identity):
            accepted = operations.cancel(identity)
            self._notice("Cancelling calibration…" if accepted else "Calibration cancellation was not accepted.")
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        phase = self._lifecycle.phase
        permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or not permitted:
            self._notice("Calibration is unavailable while another operation is active.")
            self._refresh_shell(); return
        if resolve_calibration_executable() is None:
            self._notice("Calibration is unavailable: pyFAI-calib2 is not on PATH.")
            self._refresh_shell(); return
        snapshot = self._intents.snapshot(); start = browse_start_dir("", fallback=snapshot.thaw().project_root)
        try:
            selected = self._authoring_source_chooser("poni", start)
        except Exception as error:
            self._error_notice("Calibration chooser failed", error); return
        if type(selected) is not str or not selected: return
        phase = self._lifecycle.phase
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or phase not in {RunPhase.IDLE, RunPhase.FAILED} or phase is RunPhase.FAILED and not self._lifecycle.reset_permitted:
            self._notice("Calibration context changed while choosing input."); return
        snapshot = self._intents.snapshot()
        try:
            request = prepare_calibration_request(selected)
        except (OSError, ValueError) as error:
            self._authoring_error("Calibration not started", str(error)); return
        remember_browse_path(request.source_path)
        stamp = self._operation_context_stamp(snapshot.revision)
        identity = operations.begin_calibrate(request, stamp)
        if identity is None:
            self._notice("Calibration operation was not started."); return
        transition = authored.adopt_operation("poni", request, stamp, identity)
        if authored.operation_identity is not identity:
            operations.cancel(identity)
            self._notice("Calibration operation lost authored-asset custody.")
            self._ensure_timer(); return
        self._apply_authored_asset_transition(transition)
        self._notice(f"Calibrating from {os.path.basename(request.source_path)}…")

    def _mask_action(self) -> None:
        operations = self._workspace_operations
        authored = self._authored_assets
        identity = authored.operation_identity
        if (authored.phase is AuthoredAssetPhase.RUNNING
                and authored.asset == "mask"
                and identity is not None
                and operations.current_identity is identity):
            accepted = operations.cancel(identity); self._notice("Cancelling mask…" if accepted else "Mask cancellation was not accepted.")
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        phase = self._lifecycle.phase
        permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or not permitted:
            self._notice("Mask creation is unavailable while another operation is active."); self._refresh_shell(); return
        if resolve_mask_executable() is None:
            self._notice("Mask creation is unavailable: pyFAI-drawmask is not on PATH."); self._refresh_shell(); return
        snapshot = self._intents.snapshot(); start = browse_start_dir("", fallback=snapshot.thaw().project_root)
        try:
            selected = self._authoring_source_chooser("mask", start)
        except Exception as error:
            self._error_notice("Mask chooser failed", error); return
        if type(selected) is not str or not selected: return
        phase = self._lifecycle.phase
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or phase not in {RunPhase.IDLE, RunPhase.FAILED} or phase is RunPhase.FAILED and not self._lifecycle.reset_permitted:
            self._notice("Mask context changed while choosing input."); return
        snapshot = self._intents.snapshot(); intent = snapshot.thaw()
        try: request = prepare_mask_request(selected, current_poni=str(intent.poni_file or ""), current_mask=str(intent.mask_file or ""))
        except (OSError, ValueError) as error:
            self._authoring_error("Make Mask not started", str(error)); return
        remember_browse_path(request.source_path); self._notice(f"Preparing {os.path.basename(request.source_path)}…")
        stamp = self._operation_context_stamp(snapshot.revision)
        identity = operations.begin_mask(request, stamp)
        if identity is None:
            self._notice("Mask operation was not started."); return
        transition = authored.adopt_operation("mask", request, stamp, identity)
        if authored.operation_identity is not identity:
            operations.cancel(identity)
            self._notice("Mask operation lost authored-asset custody.")
            self._ensure_timer(); return
        self._apply_authored_asset_transition(transition)
        self._notice(f"Making {os.path.basename(request.final_path)}…")

    def _authoring_error(self, title: str, detail: str) -> None:
        _LOG.warning("%s: %s", title, detail)
        self._notice(f"{title}: {detail}")
        if self._closing or self._closed:
            return
        dialog = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Warning, title, title,
            QtWidgets.QMessageBox.StandardButton.Ok, self,
        )
        dialog.setInformativeText(detail)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.open()
        self._refresh_shell(preserve_display=True, preserve_scientific=True)

    def _apply_authored_asset_transition(
        self, transition: AuthoredAssetTransition,
    ) -> None:
        if type(transition) is not AuthoredAssetTransition:
            return
        adoption = transition.adoption
        if (
            adoption is not None
            and type(adoption.result)
            in {IntentCommitAccepted, IntentRecaptureRequired}
        ):
            self._reconcile_snapshot(
                adoption.before, adoption.result.snapshot,
            )
            if adoption.remember_path:
                remember_browse_path(adoption.path)
        if transition.error:
            self._authoring_error("Experiment tool failed", transition.notice)
        elif transition.notice:
            _LOG.info("%s", transition.notice)
            self._notice(transition.notice)
        if transition.cancel_identity is not None:
            self._workspace_operations.cancel(transition.cancel_identity)
        issue = transition.issue
        if issue is not None:
            dialog = _AuthoredAssetDialog(issue.asset, issue.paths, self)
            self._authored_asset_dialog = dialog
            self._authored_asset_dialog_identity = issue.identity
            identity = issue.identity
            dialog.acceptRequested.connect(
                lambda path, i=identity, d=dialog:
                self._accept_authored_asset(i, d, path)
            )
            dialog.chooseRequested.connect(
                lambda i=identity, d=dialog:
                self._choose_another_authored_asset(i, d)
            )
            dialog.cancelRequested.connect(
                lambda i=identity, d=dialog:
                self._cancel_authored_asset(i, d)
            )
            dialog.destroyed.connect(
                lambda _object=None, i=identity, d=dialog:
                self._authored_asset_dialog_destroyed(i, d)
            )
        if transition.dialog is not None:
            self._apply_authored_asset_dialog_command(transition.dialog)
        if self._closing or self._closed:
            return
        if transition.refresh is AuthoredAssetRefreshEffect.CONTROLS:
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
            self._ensure_timer()
        elif transition.refresh is AuthoredAssetRefreshEffect.DIALOG:
            self._ensure_timer()

    def _apply_authored_asset_dialog_command(
        self, command: AuthoredAssetDialogCommand,
    ) -> None:
        dialog = self._authored_asset_dialog
        if (dialog is None
                or command.identity is not self._authored_asset_dialog_identity):
            return
        if command.effect is AuthoredAssetDialogEffect.OPEN:
            dialog.open()
            (dialog.accept_button if dialog.accept_button.isEnabled()
             else dialog.choose_button).setFocus(
                QtCore.Qt.FocusReason.OtherFocusReason,
            )
            dialog.raise_(); dialog.activateWindow()
        elif command.effect is AuthoredAssetDialogEffect.CLOSE:
            dialog.close_inert()
        elif command.effect is AuthoredAssetDialogEffect.SET_BUSY:
            dialog.set_busy(True)
        elif command.effect is AuthoredAssetDialogEffect.SET_IDLE:
            dialog.set_busy(False)

    def _advance_authored_asset_confirmation(self) -> None:
        authored = self._authored_assets
        stamp = self._operation_context_stamp()
        if authored.phase is AuthoredAssetPhase.TERMINAL_READY:
            if authored.asset == "mask":
                request = authored.saved_mask_validation_request(stamp)
                if request is not None:
                    self._begin_authored_validation(None, None, request)
                else:
                    self._apply_authored_asset_transition(
                        AuthoredAssetTransition(
                            AuthoredAssetRefreshEffect.CONTROLS,
                            "Experiment changed; saved mask was not selected.",
                        )
                    )
                return
            evidence = authored.evidence_identity
            if evidence is not None:
                self._apply_authored_asset_transition(
                    authored.issue_confirmation(evidence, stamp)
                )
            return
        if (authored.phase is AuthoredAssetPhase.CONFIRM_ISSUED
                and authored.dialog_identity
                is self._authored_asset_dialog_identity):
            identity = authored.dialog_identity
            if identity is not None:
                self._apply_authored_asset_transition(
                    authored.present_confirmation(identity, stamp)
                )

    def _authored_asset_dialog_destroyed(
        self, identity: AuthoredAssetDialogIdentity,
        dialog: _AuthoredAssetDialog,
    ) -> None:
        if (identity is not self._authored_asset_dialog_identity
                or dialog is not self._authored_asset_dialog):
            return
        self._authored_asset_dialog = None
        self._authored_asset_dialog_identity = None
        self._apply_authored_asset_transition(
            self._authored_assets.detach_dialog(identity)
        )

    def _cancel_authored_asset(
        self, identity: AuthoredAssetDialogIdentity,
        dialog: _AuthoredAssetDialog,
    ) -> None:
        if (identity is not self._authored_asset_dialog_identity
                or dialog is not self._authored_asset_dialog):
            return
        self._apply_authored_asset_transition(
            self._authored_assets.cancel_confirmation(identity)
        )

    def _accept_authored_asset(
        self, identity: AuthoredAssetDialogIdentity,
        dialog: _AuthoredAssetDialog, path: str,
    ) -> None:
        if (identity is not self._authored_asset_dialog_identity
                or dialog is not self._authored_asset_dialog):
            return
        request = self._authored_assets.validation_request(
            identity, path, self._operation_context_stamp(),
        )
        if request is None:
            self._apply_authored_asset_transition(
                self._authored_assets.cancel_confirmation(identity)
            )
            return
        self._begin_authored_validation(identity, dialog, request)

    def _choose_another_authored_asset(
        self, identity: AuthoredAssetDialogIdentity,
        dialog: _AuthoredAssetDialog,
    ) -> None:
        authored = self._authored_assets
        if (identity is not self._authored_asset_dialog_identity
                or dialog is not self._authored_asset_dialog
                or authored.phase is not AuthoredAssetPhase.CONFIRM_PRESENTED):
            return
        asset = authored.asset
        directory = authored.source_directory
        if asset not in {"poni", "mask"} or directory is None:
            return
        chooser = self._control_path_chooser
        control = PONI_FILE if asset == "poni" else MASK_FILE
        intent = self._intents.snapshot().thaw()
        current_value = intent.poni_file if asset == "poni" else intent.mask_file
        current = "" if current_value is None else str(current_value)
        try:
            selected = (chooser(control, current, directory)
                        if chooser is not None else
                        QtWidgets.QFileDialog.getOpenFileName(
                            self,
                            "Choose existing PONI" if asset == "poni"
                            else "Choose existing detector mask",
                            directory,
                            "PONI files (*.poni);;All files (*)"
                            if asset == "poni" else
                            "Detector masks (*.edf *.npy)",
                        )[0])
        except Exception as error:
            self._error_notice("Asset chooser failed", error); return
        if type(selected) is not str or not selected:
            return
        request = authored.validation_request(
            identity, selected, self._operation_context_stamp(),
        )
        if request is None:
            self._apply_authored_asset_transition(
                authored.cancel_confirmation(identity)
            )
            return
        self._begin_authored_validation(identity, dialog, request)

    def _begin_authored_validation(
        self, identity: AuthoredAssetDialogIdentity | None,
        dialog: _AuthoredAssetDialog | None, request: object,
    ) -> None:
        if (identity is not self._authored_asset_dialog_identity
                or dialog is not self._authored_asset_dialog
                or self._workspace_operations.owned):
            self._notice("Asset validation operation was not started.")
            return
        operation_identity = self._workspace_operations.begin_asset_validation(
            request, self._operation_context_stamp(),
        )
        if operation_identity is None:
            self._notice("Asset validation operation was not started.")
            return
        transition = self._authored_assets.adopt_validation(
            identity, request, operation_identity,
        )
        if self._authored_assets.operation_identity is not operation_identity:
            self._workspace_operations.cancel(operation_identity)
            self._notice("Asset validation lost authored-asset custody.")
            self._ensure_timer(); return
        self._apply_authored_asset_transition(transition)

    def _consume_authored_asset_update(self, update: object) -> bool:
        authored = self._authored_assets
        if (type(update) is not OperationUpdate
                or update.identity is not authored.operation_identity):
            return False
        self._apply_authored_asset_transition(
            authored.consume_operation_update(
                update, self._operation_context_stamp(),
            )
        )
        return True

    @staticmethod
    def _reintegrate_preparation(intent, dimension) -> dict[str, object]:
        bai = jsonable_run_value(getattr(intent, f"bai_{dimension}_args"), path="reintegrate.selected_plan.bai_args") if type(dimension) is str and dimension in {"1d", "2d"} else (_ for _ in ()).throw(ValueError("Reintegrate dimension is unsupported")); (None if type(bai) is dict else (_ for _ in ()).throw(ValueError("current integration settings are malformed")))
        bai.pop(f"gi_mode_{dimension}", None); mode = getattr(intent.gi, f"mode_{dimension}"); workers = intent.max_cores
        # Reintegration produces ONE selected mode; the Run-only companion
        # selection is not part of its science.
        bai.pop(GI_COMPANION_MODES_2D_ARG, None)
        if type(mode) is not str or mode not in ({"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"} if dimension == "1d" else {"qip_qoop", "q_chi", "exit_angles"}): raise ValueError("current integration mode is unsupported")
        if type(workers) is not int or workers < 1: raise ValueError("current core request is invalid")
        return {"api_version": 1,
            "selected_plan": {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": mode},
            "requested_shared_science": {"version": 1, "kind": "persisted_target"},
            "resource_policy": {"version": 1, "kind": "resolve", "envelope_bytes": None,
                                "requests": {"workers": workers}}}

    def _capture_current_loaded_browse(self):
        context = self._context_controller.browse_context
        request = None if context is None else context.load_request
        return (
            None
            if type(request) is not BrowseLoadRequest
            else self._context_controller.capture_loaded_browse(request)
        )

    def _qualify_external_nexus(
        self, *, validate_disk: bool,
    ) -> ExternalNexusQualification:
        """Return only a fresh processed Browse or ready 2-D Viewer target."""

        if type(validate_disk) is not bool:
            raise TypeError("external NeXus validation flag must be exact")
        if self._closing or self._closed:
            self._external_nexus_refusal = None
            return ExternalNexusQualification.refused(
                "The workspace is closing; external viewers are unavailable."
            )
        controller = self._context_controller
        capture = self._capture_current_loaded_browse()
        target = None
        viewer_revision = None
        held = self._external_nexus_refusal
        held_is_current = (
            type(capture) is LoadedBrowseCapture
            and type(held) is tuple
            and len(held) == 2
            and type(held[0]) is LoadedBrowseCapture
            and type(held[1]) is ExternalNexusQualification
            and held[1].target is None
            and held[0].is_exactly(capture)
        )
        if not held_is_current:
            self._external_nexus_refusal = None
            held = None
        if self._workspace_operations.busy:
            return ExternalNexusQualification.refused(
                "A workspace operation is active; wait for it to finish."
            )
        if self._lifecycle.phase not in {RunPhase.IDLE, RunPhase.FAILED}:
            return ExternalNexusQualification.refused(
                "An acquisition writer is active; wait for it to finish."
            )
        if type(capture) is LoadedBrowseCapture:
            target = capture.target
        else:
            context = controller.viewer_2d_context
            catalog = controller._runtime._viewer_2d_catalog
            selection = controller.selection
            current = controller.navigation.current
            frame = controller.viewer_2d_frame
            if (
                controller.viewer_2d_owned
                and type(context) is Viewer2DContext
                and context is controller._runtime._viewer_2d
                and context.state is Viewer2DState.READY
                and type(catalog) is Viewer2DArtifactCatalog
                and catalog.canonical_path == context.original_path
                and catalog.format_name == "hdf5-processed"
                and type(selection) is DisplaySelection
                and selection.kind is ContextKind.VIEWER_2D
                and selection.names(context)
                and current is not None
                and controller.owns_frame(current)
                and frame is not None
                and frame.label == current.local_frame_label
            ):
                target = context.original_path
                viewer_revision = catalog.primary_revision
        if target is None:
            return ExternalNexusQualification.refused(
                "Select one stable current processed .nexus file in Browse or 2D Viewer."
            )
        if held is not None:
            return held[1]
        if Path(target).suffix.casefold() != ".nexus":
            return ExternalNexusQualification.refused(
                "The selected target is not a processed .nexus file."
            )
        if validate_disk:
            if capture is not None:
                try:
                    state = Path(target).stat()
                except FileNotFoundError:
                    refusal = ExternalNexusQualification.refused(
                        "The selected processed .nexus file is no longer available."
                    )
                    self._external_nexus_refusal = (capture, refusal)
                    return refusal
                except OSError:
                    refusal = ExternalNexusQualification.refused(
                        "The selected processed .nexus file could not be revalidated."
                    )
                    self._external_nexus_refusal = (capture, refusal)
                    return refusal
                snapshot = capture.target_snapshot
                if (
                    not stat.S_ISREG(state.st_mode)
                    or int(state.st_size) != snapshot.size
                    or int(state.st_mtime_ns) != snapshot.mtime_ns
                    or int(state.st_dev) != snapshot.device
                    or int(state.st_ino) != snapshot.inode
                ):
                    refusal = ExternalNexusQualification.refused(
                        "The selected processed .nexus file changed after it was "
                        "loaded; refresh Browse before opening it."
                    )
                    self._external_nexus_refusal = (capture, refusal)
                    return refusal
            else:
                try:
                    current_revision = _stable_revision(Path(target))
                except FileNotFoundError:
                    return ExternalNexusQualification.refused(
                        "The selected processed .nexus file is no longer available."
                    )
                except Exception:
                    return ExternalNexusQualification.refused(
                        "The selected processed .nexus file could not be revalidated."
                    )
                if current_revision != viewer_revision:
                    return ExternalNexusQualification.refused(
                        "The selected processed .nexus file changed after it was "
                        "opened in 2D Viewer; reload it before opening it."
                    )
        return ExternalNexusQualification.ready(target)

    def _begin_pending_reintegrate_successor(
        self,
        adoption: ReintegrateSuccessorAdoption | None = None,
    ) -> bool:
        browser = self._processed_browser
        current = browser.pending_reintegrate_successor
        if adoption is None:
            adoption = current
        if (
            type(adoption) is not ReintegrateSuccessorAdoption
            or current is not adoption
            or adoption.phase is not ReintegrateSuccessorPhase.PENDING
            or adoption.request is not None
        ):
            return False
        directive = adoption.directive
        stamp = self._operation_context_stamp()
        if (
            stamp != directive.context_stamp
            or not self._context_controller
            .reintegrate_successor_predecessor_is_current(
                directive.predecessor
            )
        ):
            browser.retire_reintegrate_successor(adoption)
            self._request_browser_catalog()
            self._notice(
                "Reintegrate published a new version, but its display owner "
                "changed; the version was cataloged without switching."
            )
            return True
        if not self._release_browse_1d_debt():
            self._ensure_timer()
            return False
        request = None
        try:
            request = (
                self._context_controller.begin_reintegrate_successor_browse(
                    directive.predecessor,
                    directive.successor_path,
                    directive.terminal_commit_identity,
                    owner=directive,
                    expected_entry=directive.entry,
                    expected_labels=directive.committed_labels,
                )
            )
            browser.begin_reintegrate_successor_load(adoption, request)
        except RuntimeError as error:
            if type(request) is BrowseLoadRequest:
                self._context_controller.cancel_reintegrate_successor_browse(
                    request
                )
            browser.retire_reintegrate_successor(adoption)
            self._request_browser_catalog()
            self._notice(
                "Reintegrate successor could not begin Browse validation: "
                f"{error}. The predecessor remains selected."
            )
            return True
        self._ensure_timer()
        return True

    def _reintegrate_successor_owner_for_poll(self) -> object | None:
        adoption = self._processed_browser.pending_reintegrate_successor
        if adoption is None:
            return None
        if adoption.phase is ReintegrateSuccessorPhase.PENDING:
            self._begin_pending_reintegrate_successor(adoption)
            adoption = self._processed_browser.pending_reintegrate_successor
            if adoption is None or adoption.request is None:
                return None
        if adoption.phase is ReintegrateSuccessorPhase.CANCELLING:
            self._retire_cancelled_reintegrate_successor(adoption)
            return None
        if adoption.phase is not ReintegrateSuccessorPhase.LOADING:
            return None
        directive = adoption.directive
        request = adoption.request
        assert request is not None
        if (
            self._operation_context_stamp() != directive.context_stamp
            or not self._context_controller
            .reintegrate_successor_predecessor_is_current(
                directive.predecessor, request,
            )
        ):
            self._cancel_reintegrate_successor_adoption(adoption)
            self._request_browser_catalog()
            self._notice(
                "Reintegrate successor was cataloged without switching "
                "because its display owner changed."
            )
            return None
        return directive

    def _settle_reintegrate_successor_browse(
        self,
        outcome: BrowseLoadOutcome,
        adoption: ReintegrateSuccessorAdoption,
    ) -> bool:
        request = adoption.request
        if (
            adoption.phase is not ReintegrateSuccessorPhase.LOADING
            or request is None
            or outcome.request is not request
        ):
            return False
        directive = adoption.directive
        self._processed_browser.retire_reintegrate_successor(adoption)
        self._request_browser_catalog()
        if outcome.status is BrowseLoadStatus.READY:
            capture = self._context_controller.capture_loaded_browse(request)
            if (
                capture is None
                or capture.target != directive.successor_path
                or capture.entry != directive.entry
                or capture.labels != directive.committed_labels
                or request.terminal_commit_identity
                is not directive.terminal_commit_identity
            ):
                raise RuntimeError(
                    "Reintegrate successor Browse lost terminal ownership"
                )
            self._notice(
                "Reintegrate successor validated; now viewing "
                f"{directive.successor_path}."
            )
            return True
        detail = outcome.detail.strip()
        suffix = f" ({detail})" if detail else ""
        self._notice(
            "Reintegrate successor was not selected; the predecessor remains "
            f"visible. Successor: {directive.successor_path}{suffix}"
        )
        return True

    def _abandon_reintegrate_successor(self) -> bool:
        adoption = self._processed_browser.pending_reintegrate_successor
        if adoption is None:
            return False
        self._cancel_reintegrate_successor_adoption(adoption)
        self._request_browser_catalog()
        return True

    def _cancel_reintegrate_successor_adoption(
        self,
        adoption: ReintegrateSuccessorAdoption,
    ) -> bool:
        browser = self._processed_browser
        if browser.pending_reintegrate_successor is not adoption:
            return False
        if adoption.phase is ReintegrateSuccessorPhase.PENDING:
            browser.retire_reintegrate_successor(adoption)
            return True
        if adoption.phase is ReintegrateSuccessorPhase.LOADING:
            adoption = browser.mark_reintegrate_successor_cancelling(adoption)
        if adoption.phase is not ReintegrateSuccessorPhase.CANCELLING:
            return False
        request = adoption.request
        assert request is not None
        self._context_controller.cancel_reintegrate_successor_browse(request)
        if not self._retire_cancelled_reintegrate_successor(adoption):
            self._ensure_timer()
        return True

    def _retire_cancelled_reintegrate_successor(
        self,
        adoption: ReintegrateSuccessorAdoption | None = None,
    ) -> bool:
        browser = self._processed_browser
        current = browser.pending_reintegrate_successor
        if adoption is None:
            adoption = current
        if (
            type(adoption) is not ReintegrateSuccessorAdoption
            or current is not adoption
            or adoption.phase is not ReintegrateSuccessorPhase.CANCELLING
            or adoption.request is None
            or self._context_controller.owns_browse_request(adoption.request)
        ):
            return False
        request = adoption.request
        if not browser.retire_reintegrate_successor(adoption):
            return False
        self._request_browser_catalog()
        deferred = self._deferred_browse_selection
        if deferred is not None and deferred.blocked_by is request:
            self._deferred_browse_selection = None
            self._select_scan(
                deferred.value,
                is_directory=deferred.is_directory,
            )
        return True

    def _retire_lost_reintegrate_successor(
        self,
        adoption: ReintegrateSuccessorAdoption | None = None,
    ) -> bool:
        """Retire a load whose controller ownership ended without an outcome."""

        browser = self._processed_browser
        current = browser.pending_reintegrate_successor
        if adoption is None:
            adoption = current
        if (
            type(adoption) is not ReintegrateSuccessorAdoption
            or current is not adoption
            or adoption.phase is not ReintegrateSuccessorPhase.LOADING
            or adoption.request is None
            or self._context_controller.owns_browse_request(adoption.request)
        ):
            return False
        request = adoption.request
        if not browser.retire_reintegrate_successor(adoption):
            return False
        self._request_browser_catalog()
        self._notice(
            "Reintegrate successor validation ended without switching; "
            "the predecessor remains selected."
        )
        deferred = self._deferred_browse_selection
        if deferred is not None and deferred.blocked_by is request:
            self._deferred_browse_selection = None
            self._select_scan(
                deferred.value,
                is_directory=deferred.is_directory,
            )
        return True

    def _abandon_reintegrate_display_owner(self) -> bool:
        changed = False
        operations = self._workspace_operations
        identity = operations.reintegrate_identity
        if identity is not None:
            changed = operations.abandon_reintegrate(identity) or changed
        return self._abandon_reintegrate_successor() or changed

    def _reintegrate_action(self, dimension) -> None:
        operations = self._workspace_operations
        active = operations.reintegrate_identity
        if (
            active is not None
            and operations.current_identity is active
            and operations.reintegrate_dimension != dimension
        ):
            return
        if (
            active is not None
            and operations.current_identity is active
            and operations.reintegrate_dimension == dimension
        ):
            if operations.reintegrate_cancel_accepted:
                return
            accepted = operations.cancel_reintegrate(dimension)
            if not accepted:
                return
            notice = f"Stopping Reintegrate {dimension[0]}-D…"
            self._notice(notice)
            if (
                operations.current_identity is active
                and operations.reintegrate_identity is active
                and operations.reintegrate_cancel_accepted
            ):
                self._shell.scientific.reconcile_operation_status(notice)
            return
        if self._authored_assets.busy:
            self._notice("Reintegrate is unavailable while authored-asset confirmation is pending.")
            self._refresh_shell(); return
        if self._context_controller.browse_pending:
            self._notice(
                "Reintegrate is unavailable while Browse cleanup is pending."
            )
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        snapshot = self._intents.snapshot()
        if snapshot.thaw().processing_mode == "Int 1D (XYE)":
            # XYE cannot satisfy the retained NeXus Browse cache contract.
            # Reconcile controls first so that projection diagnostics cannot
            # replace the action-specific refusal the operator just requested.
            self._refresh_shell()
            self._notice(
                f"Reintegrate {dimension[0]}-D is unavailable for XYE-only output."
            )
            return
        phase = self._lifecycle.phase; permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        captured = self._capture_current_loaded_browse()
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or not permitted or captured is None:
            self._notice(f"Reintegrate {dimension[0]}-D requires one stable loaded Browse context."); self._refresh_shell(); return
        try: preparation = self._reintegrate_preparation(snapshot.thaw(), dimension)
        except (TypeError, ValueError) as error: self._notice(str(error)); self._refresh_shell(); return
        stamp = self._operation_context_stamp(snapshot.revision); recaptured = self._context_controller.capture_loaded_browse(captured.request); current = self._intents.snapshot()
        try: current_preparation = self._reintegrate_preparation(current.thaw(), dimension)
        except (TypeError, ValueError): current_preparation = None
        same_browse = (
            recaptured is not None and captured.is_exactly(recaptured)
        )
        if current.revision != snapshot.revision or current_preparation != preparation or not same_browse:
            self._notice("Reintegrate context changed before dispatch."); self._refresh_shell(); return
        identity = operations.begin_reintegrate(
            captured,
            dimension=dimension,
            preparation_values=preparation,
            stamp=stamp,
        )
        if identity is None:
            self._notice(
                f"Reintegrate {dimension[0]}-D was not started; "
                "the selected artifact is unchanged."
            )
            self._refresh_shell(); return
        self._notice(
            f"Reintegrating {dimension[0]}-D into a replacement candidate…"
        ); self._refresh_shell(); self._ensure_timer()

    def _consume_reintegrate_update(
        self, update: object,
    ) -> WorkspaceRefreshEffect:
        transition = self._workspace_operations.consume_reintegrate_update(
            update
        )
        if transition.notice:
            self._notice(transition.notice)
            if update.terminal is not None:
                # Terminal ownership is already qualified by the operation
                # owner. Controls-only settlement must also retire its footer.
                self._shell.scientific.reconcile_operation_status(transition.notice)
        progress = transition.reintegrate_progress
        state = self._workspace_operations.reintegrate_state
        if (
            progress is not None
            and state is not None
            and state.identity is progress.identity
            and state.progress is progress
            and not state.cancel_accepted
            and self._workspace_operations.current_identity
            is progress.identity
        ):
            self._shell.scientific.reconcile_operation_status(
                transition.notice
            )
        if transition.reintegrate_successor is not None:
            adoption = self._processed_browser.adopt_reintegrate_successor(
                transition.reintegrate_successor
            )
            self._begin_pending_reintegrate_successor(adoption)
        if transition.request_catalog:
            self._request_browser_catalog()
        return transition.effect

    def _consume_average_update(
        self, update: object,
    ) -> WorkspaceRefreshEffect:
        transition = self._workspace_operations.consume_average_update(
            update,
            current_intent_revision=self._intents.revision,
        )
        if transition.notice:
            self._notice(transition.notice)
        if transition.average_reload is not None:
            self._processed_browser.adopt_reload(
                transition.average_reload
            )
            self._retry_pending_average_reload(report_error=True)
        if transition.request_catalog:
            self._request_browser_catalog()
        return transition.effect

    def _retry_pending_average_reload(
        self, *, report_error: bool = False,
    ) -> bool:
        browser = self._processed_browser
        directive = browser.pending_average_reload
        if directive is None:
            return False
        if self._closing or self._closed:
            browser.retire_reload(directive)
            return False
        if self._context_controller.browse_pending:
            self._ensure_timer()
            return False
        if not self._release_browse_1d_debt():
            return False
        try:
            self._context_controller.begin_browse(
                directive.target,
                terminal_commit_identity=(
                    directive.terminal_commit_identity
                ),
                source_root=directive.source_root,
            )
        except Exception as error:
            if report_error:
                self._notice(
                    "Average committed; Browse reload deferred: "
                    f"{detached_exception_strings(error)[2]}"
                )
            self._ensure_timer()
            return False
        self._processed_browser.retire_terminal(force=True)
        return browser.retire_reload(directive)

    def _average_action(self, snapshot: RunIntentSnapshot) -> None:
        if self._authored_assets.busy:
            self._notice("Average is unavailable while authored-asset confirmation is pending.")
            self._refresh_shell(); return
        operations = self._workspace_operations
        intent = snapshot.thaw()
        if getattr(intent.background, "mode", "None") != "None":
            self._notice(
                "Average Scan does not support an active Background; "
                "choose Background: None before averaging."
            )
            self._refresh_shell()
            return
        source = intent.source_spec
        phase = self._lifecycle.phase
        if (self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy()
                or phase not in {RunPhase.IDLE, RunPhase.FAILED}
                or phase is RunPhase.FAILED and not self._lifecycle.reset_permitted
                or type(source) is not SourceSpec or not intent.save_path
                or intent.output_mode != "Overwrite" or intent.live_mode
                or intent.processing_mode == "Int 1D (XYE)"
                or tool_from_mode_text(intent.processing_mode)
                not in {Tool.INT_1D, Tool.INT_2D}):
            self._notice("Average requires one finite image source, Overwrite NeXus output, non-Live execution, an integration mode, and an idle workspace."); self._refresh_shell(); return
        try:
            from pathlib import Path
            from .output_preflight import _resolved_generated_target
            options = source.options
            syntactic_name = options.get("scan_name")
            if syntactic_name is not None and (
                type(syntactic_name) is not str or not syntactic_name
            ):
                raise ValueError("Average source scan name is invalid.")
            name = syntactic_name or Path(str(
                options.get("selected_file") or source.uri
            )).stem
            if not name:
                raise ValueError("Average source scan name is empty.")
            generated = _resolved_generated_target(
                intent.save_path, name,
                grazing_incidence=bool(intent.gi.enabled),
            )
            target = os.path.abspath(os.path.expanduser(str(generated)))
        except (TypeError, ValueError, OverflowError) as error:
            self._notice(str(error)); self._refresh_shell(); return
        observation = self._source_selection.observation
        choices = (
            observation.gi_motor_choices
            if observation is not None
            and observation.intent_revision == snapshot.revision
            and observation.source == source
            else None
        )
        try:
            frozen = self._intents.freeze(
                expected_revision=snapshot.revision,
                gi_motor_choices=choices,
            )
        except (TypeError, ValueError) as error:
            self._notice(str(error)); self._refresh_shell(); return
        except Exception as error:
            self._error_notice("Average configuration freeze failed", error)
            return
        if type(frozen) is IntentRecaptureRequired:
            self._notice("Average context changed before dispatch.")
            self._refresh_shell()
            return
        if (
            type(frozen) is not IntentFreezeAccepted
            or frozen.revision != snapshot.revision
            or type(frozen.configuration) is not FrozenRunConfiguration
        ):
            self._notice("Average configuration freeze returned invalid state.")
            self._refresh_shell()
            return
        identity = operations.begin_average(
            frozen.configuration,
            target,
            revision=snapshot.revision,
        )
        if identity is None:
            self._notice("Average operation was not started."); self._refresh_shell(); return
        self._notice("Averaging the finite source…"); self._refresh_shell(); self._ensure_timer()

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
        self._apply_source_observation_transition(
            self._source_selection.select(source)
        )

    def close_workspace(self) -> StartClosed:
        terminal = self._terminal_close
        if terminal is not None:
            return terminal

        first = not self._closing
        if first:
            self._closing = True
            self._pending_viewer_2d_path = None
            self._deferred_browse_selection = None
            self._abandon_reintegrate_display_owner()
            self._retire_native_plot_axis_transition()
            self._retire_batch_presentation(force=True)
            self._apply_authored_asset_transition(
                self._authored_assets.begin_close()
            )
            self._metadata_operations.begin_close()
            self._processed_browser.retire_terminal(force=True)
            for dialog in (self._metadata_dialog, self._scan_roi_dialog,
                           self._peak_dialog, self._phase_dialog):
                if dialog is not None: dialog.close()
            ScatteringWorkspace._clear_presentation_targets(self)
            self._source_selection.begin_close()
            self._shell.browser.cancel_pending_frame_selection()
            self._run_timer.stop()
            self._browser_catalog_timer.stop()
            self._processed_browser.begin_close()
            self._close_identity = (
                self._context_controller.run_identity
                or self._lifecycle.active_run_identity
                or self._lifecycle.attempt_run_identity
            )

        catalog_clean = self._processed_browser.retry_close()

        operations = getattr(self, "_workspace_operations", None)
        analysis_slot = getattr(self, "_analysis_slot", None)
        try:
            authored_cleanup_identity = self._authored_assets.cleanup_identity
            operation_identity_before_close = (
                None
                if (
                    operations is None
                    or type(authored_cleanup_identity)
                    is not OperationIdentity
                )
                else operations.current_identity
            )
            operation_close = (
                None if operations is None else operations.close()
            )
            operation_clean = (
                operations is None
                or operation_close.cleanup_status is CleanupStatus.CLEANED
            )
            if operation_close is not None:
                anonymous_clean_retirement = (
                    type(authored_cleanup_identity) is OperationIdentity
                    and operation_identity_before_close
                    is authored_cleanup_identity
                    and operation_close.cleanup_status
                    is CleanupStatus.CLEANED
                    and operation_close.identity is None
                    and not operations.owned
                    and operations.current_identity is None
                )
                self._apply_authored_asset_transition(
                    self._authored_assets.operation_lost(
                        authored_cleanup_identity
                    )
                    if anonymous_clean_retirement
                    else self._authored_assets.consume_close_receipt(
                        operation_close
                    )
                )
            operation_clean = (
                operation_clean
                and self._authored_assets.lifecycle
                is AuthoredAssetOwnerLifecycle.CLOSED
            )
        except Exception:
            operation_clean = False

        try:
            metadata_cleanup_identity = (
                self._metadata_operations.cleanup_identity
            )
            analysis_identity_before_close = (
                None
                if analysis_slot is None
                else analysis_slot.current_identity
            )
            analysis_close = (
                None if analysis_slot is None else analysis_slot.close()
            )
            analysis_clean = (
                analysis_slot is None
                or analysis_close.cleanup_status is CleanupStatus.CLEANED
            )
            if (
                analysis_close is not None
                and metadata_cleanup_identity is not None
            ):
                anonymous_clean_retirement = (
                    analysis_identity_before_close
                    is metadata_cleanup_identity
                    and analysis_close.cleanup_status
                    is CleanupStatus.CLEANED
                    and analysis_close.identity is None
                    and not analysis_slot.owned
                    and analysis_slot.current_identity is None
                )
                if anonymous_clean_retirement:
                    self._metadata_operations.lost(
                        metadata_cleanup_identity
                    )
                else:
                    self._metadata_operations.consume_close_receipt(
                        analysis_close
                    )
            analysis_clean = (
                analysis_clean
                and self._metadata_operations.lifecycle
                is MetadataLifecycle.CLOSED
            )
            if analysis_clean:
                self._analysis_identity = None
                self._analysis_kind = None
                self._analysis_target = None
                self._analysis_generation = None
                self._analysis_anchor = None
                self._analysis_request = None
                self._analysis_fingerprint = ""
        except Exception:
            analysis_clean = False

        if (not self._release_browse_1d_debt()
                or not self._clear_viewer_1d_renderer(close=True)
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
            and analysis_clean
            and catalog_clean
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
        self._source_selection.finalize_close()
        self._processed_browser.retry_close()
        self._external_nexus_refusal = None
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
        if kind is ShellCommandKind.LAUNCH_EXTERNAL_VIEWER:
            tool = external_tool_id(command.value)
            if tool is None:
                return
            nexus = (
                self._qualify_external_nexus(validate_disk=True)
                if tool is ExternalToolId.NEXPY_SELECTED
                else None
            )
            receipt = self._external_tools.launch(
                tool, nexus=nexus,
            )
            self._notice(receipt.diagnostic)
            self._shell.tools.reconcile_external_tool(
                self._external_tools.availability(
                    tool, nexus=nexus,
                )
            )
            return
        if kind in {ShellCommandKind.SHOW_METADATA, ShellCommandKind.LAUNCH_TOOL,
                    ShellCommandKind.ANALYSIS_ACTION}:
            from .analysis_mount import mount_target
            target = mount_target(kind, command.value)
            if target is not None:
                self._open_analysis_mount(
                    target, focus_roi=str(command.value) in {"roi_statistics", "roi_stats"})
                return
        analysis_locked = (
            kind in {
                ShellCommandKind.RUN_ACTION,
                ShellCommandKind.SET_BACKGROUND,
            }
            or kind is ShellCommandKind.CONTROL_ACTION
            and command.value in {
                "calibrate",
                "make_mask",
                "reintegrate_1d",
                "reintegrate_2d",
            }
        )
        if self._analysis_operation_busy() and analysis_locked:
            self._notice("Analysis operation is still active.")
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
            return
        operation_cancel = (
            kind is ShellCommandKind.SET_BACKGROUND
            and self._background_identity is not None
            and self._workspace_operations.current_identity
            is self._background_identity
            or kind is ShellCommandKind.CONTROL_ACTION
            and (
                command.value == "calibrate"
                and self._authored_assets.asset == "poni"
                and self._authored_assets.phase is AuthoredAssetPhase.RUNNING
                and self._authored_assets.operation_identity is not None
                and self._workspace_operations.current_identity
                is self._authored_assets.operation_identity
                or command.value == "make_mask"
                and self._authored_assets.asset == "mask"
                and self._authored_assets.phase is AuthoredAssetPhase.RUNNING
                and self._authored_assets.operation_identity is not None
                and self._workspace_operations.current_identity
                is self._authored_assets.operation_identity
                or command.value == f"reintegrate_{self._workspace_operations.reintegrate_dimension}"
                and self._workspace_operations.reintegrate_identity is not None
                and self._workspace_operations.current_identity
                is self._workspace_operations.reintegrate_identity
            )
        )
        operation_locked = kind in {ShellCommandKind.RUN_ACTION, ShellCommandKind.SET_BACKGROUND, ShellCommandKind.CONTROL_DRAFT, ShellCommandKind.CONTROL_EDIT, ShellCommandKind.CONTROL_BROWSE, ShellCommandKind.CONTROL_ACTION, ShellCommandKind.SET_PROCESSING_MODE, ShellCommandKind.SET_BATCH, ShellCommandKind.SET_CORES, ShellCommandKind.SET_LIVE, ShellCommandKind.SET_OUTPUT_POLICY} or kind is ShellCommandKind.MENU and (command.value == "Config:Performance Diagnostics…" or str(command.value).startswith("Config:Heavy residency:"))
        operation_retry = (
            kind is ShellCommandKind.RUN_ACTION
            and self._workspace_operations.average_pending is not None
            and self._workspace_operations.current_identity
            is self._workspace_operations.average_identity
        )
        if (
            self._experiment_operation_busy()
            and operation_locked
            and not operation_cancel
            and not operation_retry
        ):
            self._notice("Experiment operation is still active.")
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
            return
        if kind is ShellCommandKind.RUN_ACTION:
            self._run_action()
            return
        if kind is ShellCommandKind.STOP:
            average = self._workspace_operations.average_identity
            if average is not None and self._workspace_operations.current_identity is average:
                accepted = self._workspace_operations.cancel_average()
                self._notice("Cancelling Average…" if accepted else "Average cancellation was not accepted.")
                self._refresh_shell(); return
            self._stop_run()
            return
        if kind is ShellCommandKind.MENU:
            if command.value == "Config:Save":
                self._save_run_intent_profile()
                return
            if command.value == "Config:Load":
                self._load_run_intent_profile()
                return
            if command.value == "Config:Performance Diagnostics…":
                self._edit_performance_diagnostics()
                return
            if command.value == "Help:Export Analyze Results Notebook":
                self._export_results_notebook()
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
            self._deferred_browse_selection = None
            self._abandon_reintegrate_display_owner()
            self._processed_browser.clear_directory_cache()
            self._request_browser_catalog()
            return
        if kind is ShellCommandKind.SHOW_ALL:
            self._retire_batch_presentation()
            self._retain_outgoing_display = False
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
            value = command.value
            if command.path == ("directory",):
                self._select_scan(value, is_directory=True)
                return
            self._retain_outgoing_display = False
            ScatteringWorkspace._clear_presentation_targets(self)
            if command.path not in {(), ("artifact",)}:
                return
            mode = self._intents.snapshot().thaw().processing_mode
            tool = tool_from_mode_text(mode)
            if (type(value) is str and value and (
                    tool is Tool.XYE_VIEWER
                    or mode == "Int 1D (XYE)" and value.lower().endswith(".xye"))):
                selected = command.artifacts or (value,)
                current_path = value
                if command.intent is not FrameSelectionIntent.EXACT:
                    current, prior = self._context_controller.viewer_1d_artifact_selection
                    if command.intent is FrameSelectionIntent.TOGGLE_TRACE:
                        selected = tuple(
                            path for path in prior if path not in selected
                        ) + tuple(
                            path for path in selected if path not in prior
                        )
                    elif command.intent is FrameSelectionIntent.REMOVE_TRACE_RANGE:
                        selected = tuple(
                            path for path in prior if path not in selected
                        )
                    else:
                        selected = tuple(dict.fromkeys((*prior, *selected)))
                    if not selected:
                        return
                    current_path = (
                        value if value in selected else current
                        if current in selected else selected[-1]
                    )
                self._open_viewer_1d_paths(selected, current_path=current_path)
                return
            if tool is Tool.IMAGE_VIEWER and type(value) is str and value:
                self._open_viewer_2d_path(value)
                return
            if (getattr(self._context_controller, "viewer_1d_owned", False)
                    and not self._clear_viewer_1d_renderer(close=True)):
                return
            if self._context_controller.viewer_2d_owned:
                if not self._clear_viewer_2d_renderer(close=True):
                    return
            self._select_scan(value, is_directory=False)
            return
        if kind in {
            ShellCommandKind.SELECT_FRAME,
            ShellCommandKind.HYDRATE_FRAME,
            ShellCommandKind.SELECT_BROWSER_FRAMES,
        }:
            self._retire_batch_presentation()
            self._retain_outgoing_display = False
            selection = self._context_controller.selection
            if (
                selection is None
                or selection.kind is not ContextKind.VIEWER_1D
            ):
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
            if command.value in {"reintegrate_1d", "reintegrate_2d"}: self._reintegrate_action(command.value[-2:]); return
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
        operations = self._workspace_operations
        pending = operations.average_pending
        average = operations.average_identity
        if (
            pending is not None
            and average is not None
            and operations.current_identity is average
        ):
            accepted = operations.retry_average()
            self._notice(
                "Retrying Average cleanup…"
                if accepted
                else "Average cleanup retry was not accepted."
            )
            self._refresh_shell()
            if accepted:
                self._ensure_timer()
            return
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
        snapshot = self._intents.snapshot()
        intent = snapshot.thaw()
        tool = tool_from_mode_text(intent.processing_mode)
        if tool in {Tool.STITCH, Tool.RSM}:
            self.toolRequested.emit("stitch" if tool is Tool.STITCH else "rsm")
            return
        stored_average = intent.run_options.get(
            "series_average", False
        )
        if type(stored_average) is not bool:
            self._notice("Average Scan must be stored as true or false.")
            self._refresh_shell()
            return
        if (
            stored_average
            and intent.processing_mode != "Int 1D (XYE)"
            and tool in {Tool.INT_1D, Tool.INT_2D}
        ):
            if not self._commit_focused_control_edit_for_run():
                return
            snapshot = self._intents.snapshot()
            intent = snapshot.thaw()
            tool = tool_from_mode_text(intent.processing_mode)
            stored_average = intent.run_options.get(
                "series_average", False
            )
            if type(stored_average) is not bool:
                self._notice("Average Scan must be stored as true or false.")
                self._refresh_shell()
                return
            if (
                stored_average
                and intent.processing_mode != "Int 1D (XYE)"
                and tool in {Tool.INT_1D, Tool.INT_2D}
            ):
                self._average_action(snapshot)
                return
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
        snapshot = self._intents.snapshot()
        average = snapshot.thaw().run_options.get("series_average", False)
        if type(average) is not bool:
            self._notice("Average Scan must be stored as true or false.")
            self._refresh_shell()
            return
        if (
            average
            and snapshot.thaw().processing_mode != "Int 1D (XYE)"
            and tool_from_mode_text(snapshot.thaw().processing_mode)
            in {Tool.INT_1D, Tool.INT_2D}
        ):
            self._average_action(snapshot)
            return
        self._begin_run()

    def _commit_focused_control_edit_for_run(self) -> bool:
        """Commit the one active editor before capturing the next run.

        The focused editor is the sole presentation value that can be newer
        than the revisioned intent when an action arrives before
        ``editingFinished``; the renderer never owns a second form snapshot.
        """

        try:
            edit = self._shell.controls.focused_form_edit()
        except Exception as error:
            self._error_notice("Run edit capture failed", error)
            return False
        if edit is None:
            return True
        if self._source_selection.owns_edit(edit.path):
            try:
                transition = self._source_selection.edit(
                    edit.path,
                    edit.value,
                )
            except Exception as error:
                self._error_notice("Run edit commit failed", error)
                return False
            receipt = transition.intent
            if (
                receipt is not None
                and type(receipt.result) is IntentRecaptureRequired
            ):
                transition = replace(
                    transition,
                    notice=(
                        "Run not started: edit superseded; review current "
                        "value."
                    ),
                )
            self._apply_source_observation_transition(transition)
            return bool(
                transition.notice == ""
                and (
                    receipt is None
                    or type(receipt.result) is IntentCommitAccepted
                )
            )
        snapshot = self._intents.snapshot()
        reduced = reduce_control_edit(
            snapshot,
            edit.path,
            edit.value,
            observation=self._source_selection.project_observation(snapshot),
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

    def _profile_action_permitted(self) -> bool:
        phase = self._lifecycle.phase
        return (
            not self._closing
            and not self._closed
            and self._admission_state is None
            and not self._experiment_operation_busy()
            and (
                phase is RunPhase.IDLE
                or phase is RunPhase.FAILED
                and self._lifecycle.reset_permitted
            )
        )

    def _choose_profile_path_dialog(
        self,
        action: str,
        start_directory: str,
    ) -> str | None:
        file_filter = "XDART profiles (*.json);;All files (*)"
        if action == "save":
            selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save scattering profile",
                start_directory,
                file_filter,
            )
        elif action == "load":
            selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load scattering profile",
                start_directory,
                file_filter,
            )
        else:  # pragma: no cover - private callers are closed above
            raise ValueError("profile action must be 'save' or 'load'")
        return selected or None

    def _choose_results_notebook_dialog(self, start_directory: str) -> str | None:
        selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Analyze Results Notebook",
            start_directory,
            "Jupyter notebooks (*.ipynb);;All files (*)",
        )
        return selected or None

    def _selected_results_notebook_sources(
        self,
    ) -> tuple[str, tuple[str, ...], str]:
        nexus = self._qualify_external_nexus(validate_disk=True)
        if nexus.target is not None:
            return "nexus", (nexus.target,), ""
        if self._closing or self._closed:
            return "", (), "The workspace is closing; results are unavailable."
        if self._workspace_operations.busy:
            return "", (), "A workspace operation is active; wait for it to finish."
        if self._lifecycle.phase not in {RunPhase.IDLE, RunPhase.FAILED}:
            return "", (), "An acquisition writer is active; wait for it to finish."
        controller = self._context_controller
        context = controller.viewer_1d_context
        selection = controller.selection
        navigation = controller.navigation
        if (
            type(context) is not Viewer1DContext
            or context.state is not Viewer1DState.READY
            or selection is None
            or selection.kind is not ContextKind.VIEWER_1D
            or not selection.names(context)
            or len(navigation.frames) != len(context.paths)
            or not navigation.selected
        ):
            return "", (), nexus.reason
        selected_paths = []
        for frame in navigation.selected:
            try:
                index = next(
                    index for index, candidate in enumerate(navigation.frames)
                    if candidate is frame
                )
                path = context.paths[index]
                state = Path(path).stat()
            except (StopIteration, OSError):
                return "", (), "A selected XYE result is no longer available."
            if (
                not os.path.isabs(path)
                or Path(path).suffix.casefold() != ".xye"
                or not stat.S_ISREG(state.st_mode)
            ):
                return "", (), "Select ready XYE result files or one stable processed NeXus file."
            selected_paths.append(path)
        return "xye", tuple(selected_paths), ""

    def _export_results_notebook(self) -> None:
        kind, paths, reason = self._selected_results_notebook_sources()
        if not paths:
            self._notice(reason)
            return
        try:
            selected = self._results_notebook_chooser(str(Path(paths[0]).parent))
        except Exception as error:
            self._error_notice("Results notebook chooser failed", error)
            return
        if type(selected) is not str or not selected:
            return
        destination = Path(selected).expanduser()
        if destination.suffix.casefold() != ".ipynb":
            destination = destination.with_name(f"{destination.name}.ipynb")
        try:
            self._write_profile_atomically(
                destination, results_notebook_text(kind=kind, paths=paths),
            )
        except Exception as error:
            self._error_notice("Results notebook export failed", error)
            return
        remember_browse_path(destination)
        self._notice(f"Results notebook saved: {destination.name}")

    def _choose_run_intent_profile_path(self, action: str) -> Path | None:
        intent = self._intents.snapshot().thaw()
        start_directory = browse_start_dir(
            "",
            fallback=intent.project_root,
        )
        try:
            selected = self._profile_path_chooser(action, start_directory)
        except Exception as error:
            self._error_notice("Profile chooser failed", error)
            return None
        if type(selected) is not str or not selected:
            return None
        path = Path(selected).expanduser()
        if action == "save" and not path.suffix:
            path = path.with_suffix(".json")
        return path

    @staticmethod
    def _write_profile_atomically(path: Path, text: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _save_run_intent_profile(self) -> None:
        if not self._profile_action_permitted():
            self._notice("Profiles can be saved only while the workspace is idle.")
            self._refresh_shell()
            return
        if not self._commit_focused_control_edit_for_run():
            return
        path = self._choose_run_intent_profile_path("save")
        if path is None:
            return
        if not self._profile_action_permitted():
            self._notice("Profile save cancelled because the workspace became active.")
            self._refresh_shell()
            return
        try:
            text = dump_run_intent_profile(
                self._intents.snapshot().thaw()
            )
            self._write_profile_atomically(path, text)
        except Exception as error:
            self._error_notice("Profile save failed", error)
            return
        remember_browse_path(path)
        self._notice(f"Profile saved: {path.name}")
        self._refresh_shell()

    def _load_run_intent_profile(self) -> None:
        if not self._profile_action_permitted():
            self._notice("Profiles can be loaded only while the workspace is idle.")
            self._refresh_shell()
            return
        path = self._choose_run_intent_profile_path("load")
        if path is None:
            return
        if not self._profile_action_permitted():
            self._notice("Profile load cancelled because the workspace became active.")
            self._refresh_shell()
            return
        prior = self._intents.snapshot()
        try:
            candidate = load_run_intent_profile(
                path.read_text(encoding="utf-8")
            )
            if candidate.processing_mode == "Int 1D (XYE)":
                _drop_nexus_only_performance_options(candidate)
        except Exception as error:
            self._error_notice("Profile load failed", error)
            return
        if not self._profile_action_permitted():
            self._notice("Profile load cancelled because the workspace became active.")
            self._refresh_shell()
            return
        try:
            result = self._intents.commit(
                candidate,
                expected_revision=prior.revision,
            )
        except Exception as error:
            self._error_notice("Profile load commit failed", error)
            return
        if type(result) is IntentCommitAccepted:
            remember_browse_path(path)
            self._notice(f"Profile loaded: {path.name}")
        else:
            self._notice("Profile load superseded; review current values.")
        self._reconcile_snapshot(prior, result.snapshot)

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
            _LOG.warning("Output admission failed: %s", result.reason)
            released = self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    token, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            self._notice(f"Output admission failed: {result.reason}")
            if released.cleanup_status is not CleanupStatus.CLEANED:
                self._notice("Output cleanup remains pending.")
            elif result.append_refused:
                self._confirm_append_overwrite(result)
            else:
                dialog = QtWidgets.QMessageBox(
                    QtWidgets.QMessageBox.Icon.Warning,
                    "Run not started", "Run not started",
                    QtWidgets.QMessageBox.StandardButton.Ok, self,
                )
                dialog.setInformativeText(result.reason)
                dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
                dialog.open()
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
        self._admission_state = replace(
            state,
            admission_receipt=result,
            retirement_receipt=result.display_retirement,
            retirement_applied=True,
        )
        if self._intents.snapshot().revision != result.revision:
            self._release_admission(token)
            self._render_start_outcome(
                pipeline.refuse(
                    result, StartRefusal.OUTPUT_PREFLIGHT
                )
            )
            return True
        return not self._launch(token, result)

    def _confirm_append_overwrite(self, failure: AdmissionFailure) -> None:
        snapshot = self._intents.snapshot()
        if (snapshot.revision != failure.token.revision
                or snapshot.thaw().output_mode != "Append"
                or not self._start_permitted()[0]):
            return
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Cannot append to existing output")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText("The existing output is incompatible with this Append run.")
        dialog.setInformativeText(
            "Overwrite the existing output and restart from the beginning with "
            "the selected settings, or cancel to keep the saved results."
        )
        dialog.setDetailedText(failure.reason)
        overwrite = dialog.addButton("Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        def finished(_result: int) -> None:
            if (dialog.clickedButton() is not overwrite
                    or self._intents.snapshot().revision != snapshot.revision
                    or not self._start_permitted()[0]):
                return
            self._edit_run_strip(ShellCommandKind.SET_OUTPUT_POLICY, "Overwrite")
            self._run_action()

        dialog.finished.connect(finished)
        dialog.open()

    def _launch(
        self, token: AdmissionToken, receipt: AdmissionReceipt
    ) -> bool:
        pipeline = self._pipeline
        if pipeline is None:
            return False
        outcome = pipeline.start(receipt)
        if isinstance(outcome, StartLaunched):
            self._admission = None
            self._bind_native_plot_axis_to_run(
                outcome.run_identity,
                outcome.configuration,
            )
            ScatteringWorkspace._clear_presentation_targets(self)
            launched_source = outcome.source_capture.source
            self._source_selection.set_launched_source(
                launched_source,
                outcome.run_identity,
                live_mode=outcome.configuration.live_mode,
            )
            self._processed_browser.begin_follow(outcome.run_identity)
            retirement = self._batch_terminal.begin_run(
                outcome.run_identity,
                batch_mode=outcome.configuration.batch_mode,
                visible_progress=ProgressProjection(detail="Run started"),
            )
            self._apply_batch_retirement(retirement)
            self._run_frame_seen = False
            self._last_live_plot_at = None
            self._scientific_repaint_pending = False
            self._waterfall_candidate_count = 0
            self._quartile_refresh_identity = (
                outcome.run_identity
                if os.environ.get(
                    "XDART_PERF_QUARTILES", "",
                ).strip() == "1"
                else None
            )
            self._quartile_refresh_seconds = [0.0, 0.0, 0.0, 0.0]
            self._processed_browser.set_transient_frame(None)
            self._processed_browser.mark_transient_catalog_barrier(None)
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

        # Applied retirement detaches the visible pixels from all context
        # ownership.  A refused launch must not erase them before another
        # accepted frame is ready to paint.
        self._retain_outgoing_display = retirement_applied

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
        self._release_native_plot_axis_transition_for_retry()
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
        self._batch_terminal.record_stop(self._progress)
        self._refresh_shell()
        self._ensure_timer()

    def _batch_navigation_facts(
        self, frame: DisplayFrameKey | None,
    ) -> BatchNavigationFacts:
        controller = self._context_controller
        navigation = controller.navigation
        selection = controller.selection
        return (
            controller.run_identity,
            navigation.frames[-1] if navigation.frames else None,
            navigation.current,
            navigation.selected,
            bool(frame is not None and controller.owns_frame(frame)),
            None if selection is None else selection.display_generation,
        )

    def _apply_batch_retirement(
        self, effect: BatchTerminalDecision, *, release_display: bool = False,
    ) -> None:
        owner = effect.presentation
        if effect.action == RETIRE and owner is not None and owner.awaiting_full_raw:
            self._context_controller.clear_full_raw()
            self._detector_demand_frame = None
        if effect.action == RETIRE and release_display:
            self._retain_outgoing_display = False

    def _retire_batch_presentation(
        self, *, release_display: bool = True, force: bool = False,
    ) -> None:
        self._apply_batch_retirement(
            self._batch_terminal.retire(
                force=force,
            ),
            release_display=release_display,
        )

    def _resolve_batch_full_raw(
        self, identity: RunIdentity, frame: DisplayFrameKey,
    ) -> tuple[str, str]:
        if self._preferences.detector_mode != "full":
            return FULL_READY, ""
        controller = self._context_controller
        selection = controller.selection
        available, reason = controller.full_raw_availability()
        navigation = controller.navigation
        if not (
            available
            and selection is not None
            and controller.run_identity is identity
            and navigation.current is frame
            and navigation.selected == (frame,)
        ):
            return (
                FULL_REFUSED,
                reason
                or "Batch terminal Full Raw owner was not exact; prior display retained.",
            )
        self._detector_scope_owner = selection.owner
        self._detector_demand_frame = frame
        resident, _pending, diagnostic = controller.full_raw_status()
        self._preferences = replace(
            self._preferences,
            detector_available=True,
            detector_pending=not resident,
            detector_diagnostic="" if resident else diagnostic or "",
        )
        if resident:
            return FULL_READY, ""
        if controller.request_full_current() is not None:
            self._ensure_timer()
            return FULL_REQUESTED, ""
        refused = diagnostic or "Batch terminal Full Raw request was refused."
        self._preferences = replace(
            self._preferences,
            detector_pending=False,
            detector_diagnostic=refused,
        )
        return (
            FULL_REFUSED,
            "Batch terminal Full Raw request was refused; prior display retained.",
        )

    def _advance_batch_terminal(self, event: StandardRunEvent) -> BatchTerminalDecision:
        controller = self._batch_terminal
        latest = controller.latest_frame
        context = self._context_controller
        preparation = controller.begin_terminal(
            event, self._batch_navigation_facts(latest),
        )
        if preparation.notice:
            self._notice(preparation.notice)
        frame = preparation.frame
        if preparation.action != SELECT or frame is None:
            return preparation
        selected = context.select_navigation(frame, (frame,))
        full_raw, diagnostic = (
            self._resolve_batch_full_raw(event.run_identity, frame)
            if selected else (FULL_READY, "")
        )
        effect = controller.complete_terminal(
            frame,
            self._batch_navigation_facts(frame),
            full_raw=full_raw,
            diagnostic=diagnostic,
        )
        if effect.notice:
            self._notice(effect.notice)
        if effect.frame is not None:
            self._follow_processed_artifact(effect.frame)
        return effect

    def _consume_batch_display(self, event: StandardRunEvent) -> tuple[bool, bool]:
        controller = self._context_controller
        owner = self._batch_terminal.presentation
        frame = None if owner is None else owner.frame
        effect = self._batch_terminal.inspect_display_event(
            event, self._batch_navigation_facts(frame),
        )
        if effect.action == PASS_THROUGH:
            return False, False
        owner = effect.presentation
        if effect.action != QUALIFY or owner is None:
            return True, False
        payload = controller.qualify_display_event(event)
        accepted = self._batch_terminal.accept_qualified_display(
            owner,
            event,
            payload.frame_key
            if type(payload) is StandardDisplayPayload
            else None,
        )
        if accepted:
            self._preferences = replace(
                self._preferences,
                detector_available=True,
                detector_pending=False,
                detector_diagnostic="",
            )
        return True, accepted

    def _batch_ready_to_paint(self) -> BatchTerminalPresentation | None:
        owner = self._batch_terminal.presentation
        frame = None if owner is None else owner.frame
        return self._batch_terminal.ready_to_paint(
            self._batch_navigation_facts(frame)
        )

    def _paint_batch_terminal(
        self, owner: BatchTerminalPresentation,
    ) -> bool:
        if self._batch_ready_to_paint() is not owner:
            return False
        self._consume_native_plot_axis_transition(owner.run_identity)
        retained = self._retain_outgoing_display
        self._retain_outgoing_display = False
        revision = self._shell_revision
        self._refresh_event_shell(
            suppress_detector_demand=True,
            allow_batch_terminal_paint=True,
        )
        applied = (
            self._batch_terminal.presentation is owner
            and self._shell_revision > revision
        )
        completed = self._batch_terminal.complete_paint(
            owner, applied=applied,
        )
        if not completed and self._batch_terminal.presentation is owner:
            self._retain_outgoing_display = retained
        return completed

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
            and not self._batch_terminal.active
            and self._preferences.plot_mode == "Single"
            and self._processed_browser.auto_last
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

    def _select_scan(
        self, value: object, *, is_directory: bool = False, reopen: bool = False,
    ) -> None:
        if type(value) is not str or not value:
            return
        if type(is_directory) is not bool or type(reopen) is not bool:
            return
        self._abandon_reintegrate_display_owner()
        successor = self._processed_browser.pending_reintegrate_successor
        if (
            successor is not None
            and successor.phase is ReintegrateSuccessorPhase.CANCELLING
            and successor.request is not None
        ):
            self._deferred_browse_selection = _DeferredBrowseSelection(
                successor.request, value, is_directory,
            )
            self._ensure_timer()
            return
        terminal_request = self._processed_browser.terminal_request
        if is_directory:
            self._set_browser_directory(value, explicit=True)
            return
        self._retire_batch_presentation()
        normalized = os.path.normcase(os.path.abspath(os.path.expanduser(value)))
        same_terminal_target = (
            terminal_request is not None
            and normalized == terminal_request.source_path
        )
        if not self._release_browse_1d_debt():
            self._notice("Browse 1-D cache release remains pending.")
            # QListWidget highlights the clicked row before dispatching this
            # command.  Debt refusal has not changed display custody, so
            # immediately replay the authoritative Browser projection rather
            # than leaving old science under a false new selection.
            self._refresh_shell(preserve_scientific=True)
            self._ensure_timer()
            return
        if (
            not reopen
            and (terminal_request is None or same_terminal_target)
            and self._context_controller.select_browser_target(value)
        ):
            self._notice("")
            self._refresh_shell()
            return
        try:
            request = self._context_controller.begin_browse(
                value,
                source_root=(
                    self._intents.snapshot().thaw().project_root or None
                ),
            )
        except Exception as error:
            self._error_notice("Browse refused", error)
            if self._context_controller.browse_pending:
                self._ensure_timer()
            return
        if terminal_request is not None and request is not terminal_request:
            self._processed_browser.retire_terminal(
                terminal_request
            )
        self._notice("")
        # A submitted Browse request is still only intent.  The prior
        # selection remains authoritative until poll_browse adopts the exact
        # ready context; do not let Qt's optimistic row highlight claim the
        # new artifact early.
        self._refresh_shell(preserve_scientific=True)
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
            frames = _linearized_frame_selection(
                self._context_controller.navigation,
                command,
            )
            if self._context_controller.select_viewer_1d(frame, frames):
                self._refresh_shell()
            return
        if selection is not None and selection.kind is ContextKind.VIEWER_2D:
            if frame is None:
                return
            current = self._context_controller.navigation.current
            if (current is not frame
                    and self._context_controller.viewer_2d_frame is not None
                    and not self._clear_viewer_2d_renderer(preserve_navigation=True)):
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
            self._processed_browser.set_auto_last(False)
        handoff = self._processed_browser.terminal_handoff
        request = None if handoff is None else handoff.request
        terminal_artifact = (
            None if handoff is None else handoff.source_artifact
        )
        current = navigation.current
        selected = navigation.selected
        if (
            request is not None
            and terminal_artifact is not None
            and type(handoff.run_identity) is RunIdentity
            and current is not None
            and current.run_identity is handoff.run_identity
            and current.artifact == terminal_artifact
            and all(
                candidate.run_identity is handoff.run_identity
                and candidate.artifact == terminal_artifact
                for candidate in selected
            )
        ):
            # Preserve an explicit acquisition-frame choice made while the
            # clean terminal artifact is loading.  Browse settlement maps the
            # accepted labels onto the authenticated persisted context.
            self._processed_browser.update_terminal_selection(
                request,
                run_identity=handoff.run_identity,
                source_artifact=terminal_artifact,
                current_label=current.local_frame_label,
                selected_labels=tuple(
                    candidate.local_frame_label for candidate in selected
                ),
            )
        self._refresh_shell()
        self._ensure_timer()

    def _drain_executor(self) -> None:
        if self._closing or self._closed:
            return
        # Cache borrow debt is page-owned, not selected-context-owned.  Settle
        # it before draining any event that could replace Browse with an
        # acquisition context.
        pending_average_reload = (
            self._processed_browser.pending_average_reload is not None
        )
        if not self._settle_browse_1d_before_drain():
            return
        successor = self._processed_browser.pending_reintegrate_successor
        if (
            successor is not None
            and successor.phase is ReintegrateSuccessorPhase.PENDING
        ):
            self._begin_pending_reintegrate_successor(successor)
        pending_average_reload_changed = bool(
            pending_average_reload
            and self._processed_browser.pending_average_reload is None
        )
        defer_terminal_science = False
        terminal_science_complete = False
        reuse_terminal_science = False
        hold_batch_terminal_science = False
        browse_presentation_ready: TerminalBrowsePresentation | None = None
        changed = self._poll_admission()
        poll_viewer_1d = getattr(self._context_controller, "poll_viewer_1d", None)
        if poll_viewer_1d is not None and poll_viewer_1d():
            changed = True
        if self._context_controller.poll_viewer_2d():
            changed = True
        if self._pending_viewer_2d_path is not None:
            self._open_viewer_2d_path(self._pending_viewer_2d_path)
            changed = True
        if self._context_controller.poll_browse_preview():
            changed = True
        if self._context_controller.browse_pending:
            successor_owner = self._reintegrate_successor_owner_for_poll()
            successor_adoption = (
                self._processed_browser.pending_reintegrate_successor
            )
            handoff = self._processed_browser.terminal_handoff
            terminal_request = (
                None if handoff is None else handoff.request
            )
            poll_started = (
                None
                if terminal_request is None
                else self._processed_browser.begin_poll_timing(
                    terminal_request
                )
            )
            try:
                outcome = self._context_controller.poll_browse(
                    reintegrate_successor_owner=successor_owner,
                )
            except Exception as error:
                self._error_notice("Browse failed", error)
                outcome = None
            finally:
                if terminal_request is not None:
                    self._processed_browser.finish_poll_timing(
                        terminal_request, poll_started
                    )
            if outcome is not None:
                load_outcome = (
                    outcome if type(outcome) is BrowseLoadOutcome else None
                )
                successor_settled = bool(
                    load_outcome is not None
                    and type(successor_adoption)
                    is ReintegrateSuccessorAdoption
                    and self._settle_reintegrate_successor_browse(
                        load_outcome, successor_adoption,
                    )
                )
                settle_started = (
                    None
                    if terminal_request is None
                    else self._processed_browser.begin_settle_timing(
                        terminal_request
                    )
                )
                reuse_terminal_science = (
                    False
                    if successor_settled
                    else self._settle_terminal_browse(outcome)
                )
                if (
                    terminal_request is not None
                    and load_outcome is not None
                ):
                    self._processed_browser.finish_settle_timing(
                        terminal_request,
                        settle_started,
                        load_outcome,
                    )
                presentation = self._processed_browser.terminal_presentation
                if (
                    load_outcome is not None
                    and load_outcome.status is BrowseLoadStatus.READY
                    and presentation is not None
                    and presentation.request is load_outcome.request
                    and self._processed_terminal_presentation_is_current(
                        presentation
                    )
                ):
                    browse_presentation_ready = presentation
                changed = True
                if not successor_settled:
                    detail = getattr(outcome, "detail", "")
                    self._notice(detail)
                current = self._context_controller.navigation.current
                if current is not None:
                    self._follow_processed_artifact(current)
            if self._retire_cancelled_reintegrate_successor():
                changed = True
        if self._retire_lost_reintegrate_successor():
            changed = True
        handoff = self._processed_browser.terminal_handoff
        if (
            handoff is not None
            and not self._context_controller.owns_browse_request(
                handoff.request
            )
        ):
            self._processed_browser.retire_terminal(handoff.request)
            changed = True
        presentation = self._processed_browser.terminal_presentation
        if (
            presentation is not None
            and not self._processed_terminal_presentation_is_current(
                presentation
            )
        ):
            self._processed_browser.retire_terminal(
                presentation.request
            )
            changed = True

        force_scientific = changed
        frame_presentation_changed = False
        controls_refresh = pending_average_reload_changed

        executor = self._run_executor
        events: tuple[StandardRunEvent, ...] = ()
        waterfall_was_active = False
        if executor is not None:
            scientific_view = self._shell.scientific
            rendered_trace_count = scientific_view.trace_row_count
            if self._retain_outgoing_display:
                pass
            elif self._scientific_repaint_pending:
                self._waterfall_candidate_count = max(
                    self._waterfall_candidate_count,
                    rendered_trace_count,
                )
            else:
                self._waterfall_candidate_count = rendered_trace_count
            waterfall_was_active = scientific_view.bottom_waterfall_active
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
                    follow_latest=(
                        False if paced else self._processed_browser.auto_last
                    ),
                ):
                    frame = event.navigation_delta.appended
                    first_paced_frame = paced and not self._run_frame_seen
                    self._run_frame_seen = True
                    self._record_artifact_progress(event)
                    self._progress = ProgressProjection(
                        event.completed,
                        event.total,
                        event.detail,
                        tuple(self._artifact_progress.values()),
                        _directory_file_progress(event),
                    )
                    if self._batch_terminal.record_frame(event, frame):
                        # Batch owns the accepted publication internally, but
                        # FRAME_READY is never a GUI projection boundary.  In
                        # particular, do not publish a transient browser row or
                        # wake a shell/status reconciliation for this prefix.
                        continue
                    self._processed_browser.set_transient_frame(frame)
                    prior_waterfall_candidate_count = (
                        self._waterfall_candidate_count
                    )
                    self._waterfall_candidate_count += 1
                    waterfall_boundary_crossed = (
                        not waterfall_was_active
                        and not waterfall_should_be_active(
                            self._preferences.plot_mode,
                            prior_waterfall_candidate_count,
                            was_active=False,
                        )
                        and waterfall_should_be_active(
                            self._preferences.plot_mode,
                            self._waterfall_candidate_count,
                            was_active=False,
                        )
                    )
                    if first_paced_frame:
                        force_scientific = True
                        navigation = self._context_controller.navigation
                        already_selected = (
                            navigation.current is frame
                            and len(navigation.selected) == 1
                            and navigation.selected[0] is frame
                        )
                        changed = (
                            already_selected
                            or self._select_presentation_target(
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
                        frame_presentation_changed = True
                        force_scientific = (
                            waterfall_boundary_crossed
                            or force_scientific
                        )
                    self._follow_processed_artifact(frame)
                continue
            if event.kind is StandardEventKind.DISPLAY_READY:
                handled, became_ready = self._consume_batch_display(event)
                if handled:
                    if became_ready:
                        changed = True
                        force_scientific = True
                    continue
                payload = self._context_controller.qualify_display_event(
                    event
                )
                if payload is not None:
                    self._consume_native_plot_axis_transition(
                        event.run_identity
                    )
                    changed = True
                    force_scientific = True
                continue
            if event.kind in {
                StandardEventKind.FINISHED,
                StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            }:
                force_scientific = True
                was_batch = self._batch_terminal.active
                if was_batch:
                    self._advance_batch_terminal(event)
                else:
                    ScatteringWorkspace._flush_presentation_target(self)
                    self._retain_outgoing_display = False
                batch_owner = self._batch_terminal.presentation
                if (
                    not was_batch
                    or event.kind is not StandardEventKind.FINISHED
                    or event.cleanup_status is not CleanupStatus.CLEANED
                    or batch_owner is None
                    or batch_owner.run_identity is not event.run_identity
                    or batch_owner.frame is None
                ):
                    self._release_native_plot_axis_transition_for_retry(
                        event.run_identity
                    )
                terminal_navigation = self._context_controller.navigation
                batch_notice = (
                    self._notice_text
                    if was_batch
                    and batch_owner is not None
                    and batch_owner.frame is None
                    else ""
                )
                self._accept_terminal_event(event)
                opened_xye = self._begin_terminal_xye_viewer(
                    event, current=terminal_navigation.current,
                )
                defer_terminal_science = (
                    not opened_xye and self._begin_terminal_browse(
                        event,
                        was_batch=was_batch,
                        current=terminal_navigation.current,
                        selected=terminal_navigation.selected,
                    )
                )
                handoff = self._processed_browser.terminal_handoff
                terminal_science_complete = (
                    defer_terminal_science
                    and handoff is not None
                    and self._terminal_scientific_matches(
                        handoff, terminal_navigation,
                    )
                )
                if batch_notice and not opened_xye:
                    self._notice(batch_notice)
                self._retry_deferred_gi_motor_default()
                # The writer publishes its atomic final path before emitting
                # the terminal event.  Re-enumerate here so a non-batch run
                # whose FRAME_READY preceded that rename becomes visible.
                catalog_request = self._request_browser_catalog()
                self._processed_browser.mark_transient_catalog_barrier(
                    catalog_request
                )
                self._run_frame_seen = False
                changed = True

        operations = getattr(self, "_workspace_operations", None)
        operation_identity = (
            None
            if operations is None
            else operations.current_identity
        )
        if operation_identity is not None:
            ScatteringWorkspace._observe_operation_stamp(self)
            update = operations.poll(operation_identity)
            if update is not None:
                if self._consume_authored_asset_update(update):
                    operation_refresh = WorkspaceRefreshEffect.NONE
                else:
                    operation_refresh = self._consume_reintegrate_update(
                        update
                    )
                    if operation_refresh is WorkspaceRefreshEffect.NONE:
                        if self._consume_background_update(update):
                            operation_refresh = WorkspaceRefreshEffect.FULL
                        else:
                            operation_refresh = (
                                self._consume_average_update(update)
                            )
                if operation_refresh is WorkspaceRefreshEffect.CONTROLS:
                    controls_refresh = True
                elif operation_refresh is WorkspaceRefreshEffect.FULL:
                    changed = True
                    force_scientific = True
            elif operation_identity is self._background_identity and not operations.owned:
                self._background_identity = None; self._notice("Display background failed before terminal publication."); changed = True; force_scientific = True
            elif (operation_identity is self._authored_assets.operation_identity
                    and not operations.owned):
                self._apply_authored_asset_transition(
                    self._authored_assets.operation_lost(operation_identity)
                )
            elif not operations.owned:
                transition = operations.consume_lost_owner(
                    operation_identity
                )
                if transition.average_reload is not None:
                    self._processed_browser.adopt_reload(
                        transition.average_reload
                    )
                if transition.notice:
                    self._notice(transition.notice)
                if transition.effect is WorkspaceRefreshEffect.CONTROLS:
                    controls_refresh = True
                elif transition.effect is WorkspaceRefreshEffect.FULL:
                    changed = True
                    force_scientific = True

        analysis_slot = getattr(self, "_analysis_slot", None)
        analysis_identity = (
            None
            if analysis_slot is None
            else analysis_slot.current_identity
        )
        if analysis_identity is not None:
            ScatteringWorkspace._observe_operation_stamp(self)
            update = analysis_slot.poll(analysis_identity)
            if update is not None:
                analysis_refresh = self._consume_analysis_update(update)
                if analysis_refresh is WorkspaceRefreshEffect.CONTROLS:
                    controls_refresh = True
                elif analysis_refresh is WorkspaceRefreshEffect.FULL:
                    changed = True
                    force_scientific = True
            elif (
                self._metadata_operations.active_identity
                is analysis_identity
                and not analysis_slot.owned
            ):
                process = self._metadata_operations.lost(
                    analysis_identity
                )
                if process is not None:
                    message = (
                        "Analysis failed before terminal publication."
                    )
                    self._notice(message)
                    if self._finish_metadata_refresh(
                        process.request.dialog, message,
                    ) is WorkspaceRefreshEffect.CONTROLS:
                        controls_refresh = True
            elif (
                analysis_identity is self._analysis_identity
                and not analysis_slot.owned
            ):
                self._analysis_identity = None
                self._analysis_kind = None
                self._analysis_target = None
                self._analysis_generation = None
                self._analysis_anchor = None
                self._analysis_request = None
                self._analysis_fingerprint = ""
                message = "Analysis failed before terminal publication."
                self._notice(message)
                changed = True
                force_scientific = True

        deferred_refresh = self._dispatch_deferred_metadata()
        if deferred_refresh is WorkspaceRefreshEffect.CONTROLS:
            controls_refresh = True
        elif deferred_refresh is WorkspaceRefreshEffect.FULL:
            changed = True
            force_scientific = True

        advanced_presentation = (
            ScatteringWorkspace._advance_presentation_target(self)
        )
        if advanced_presentation:
            changed = True
            frame_presentation_changed = True
        if self._scientific_repaint_pending:
            last_plot = self._last_live_plot_at
            if (
                last_plot is None
                or time.monotonic() - last_plot
                >= self._live_plot_interval_ms / 1000.0
            ):
                changed = True
                force_scientific = True
        if controls_refresh and not changed:
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )
        batch_ready = self._batch_ready_to_paint()
        if changed and batch_ready is not None:
            self._scientific_repaint_pending = False
            painted = self._paint_batch_terminal(batch_ready)
            if not painted:
                self._refresh_event_shell(
                    preserve_scientific=True,
                    skip_scientific_projection=True,
                    suppress_detector_demand=True,
                )
            # _paint_batch_terminal owns the one permitted scientific refresh;
            # the outer drain must never follow it with a second projection.
            changed = False
        hold_batch_terminal_science = bool(
            self._batch_terminal.active
            and self._batch_ready_to_paint() is None
        )
        if changed:
            handoff = self._processed_browser.terminal_handoff
            hold_complete_terminal_science = (
                terminal_science_complete
                or handoff is not None
                and self._context_controller.owns_browse_request(
                    handoff.request
                )
                and self._terminal_scientific_matches(
                    handoff, self._context_controller.navigation,
                )
            )
            preserve_scientific = (
                hold_batch_terminal_science
                or defer_terminal_science
                or reuse_terminal_science
                or hold_complete_terminal_science
            )
            last_plot = self._last_live_plot_at
            if (
                not preserve_scientific
                and
                frame_presentation_changed
                and not force_scientific
                and self._live_plot_interval_ms
                > _LIVE_EVENT_DRAIN_INTERVAL_MS
                and last_plot is not None
                and time.monotonic() - last_plot
                < self._live_plot_interval_ms / 1000.0
            ):
                preserve_scientific = True
            if preserve_scientific:
                if hold_batch_terminal_science:
                    self._scientific_repaint_pending = False
                    self._refresh_event_shell(
                        preserve_scientific=True,
                        skip_scientific_projection=True,
                        suppress_detector_demand=True,
                    )
                elif reuse_terminal_science:
                    presentation = browse_presentation_ready
                    paint = (
                        None
                        if presentation is None
                        else self._begin_processed_terminal_paint(
                            presentation,
                            reuse_science=True,
                        )
                    )
                    prior_revision = self._shell_revision
                    self._refresh_event_shell(
                        preserve_scientific=True,
                        skip_scientific_projection=True,
                        rebind_scientific_navigation=True,
                        terminal_paint=paint,
                    )
                    if paint is not None:
                        self._complete_processed_terminal_paint(
                            paint,
                            applied=self._shell_revision > prior_revision,
                        )
                else:
                    self._scientific_repaint_pending = not (
                        defer_terminal_science
                        and terminal_science_complete
                        or hold_complete_terminal_science
                    )
                    self._refresh_event_shell(
                        preserve_scientific=True,
                        skip_scientific_projection=True,
                    )
            else:
                presentation_owner = browse_presentation_ready
                if presentation_owner is None:
                    candidate_owner = (
                        self._processed_browser.terminal_presentation
                    )
                    if (
                        candidate_owner is not None
                        and self._processed_terminal_presentation_is_current(
                            candidate_owner
                        )
                    ):
                        presentation_owner = candidate_owner
                paint = (
                    None
                    if presentation_owner is None
                    else self._begin_processed_terminal_paint(
                        presentation_owner,
                        reuse_science=False,
                    )
                )
                prior_revision = self._shell_revision
                self._refresh_event_shell(terminal_paint=paint)
                if paint is not None:
                    self._complete_processed_terminal_paint(
                        paint,
                        applied=self._shell_revision > prior_revision,
                    )
        self._advance_authored_asset_confirmation()
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
        frame_label = "Frame" if event.completed == 1 else "Frames"
        terminal_detail = (
            f"{terminal_state} · {event.completed} {frame_label} · "
            f"{timing.elapsed_seconds:.2f} s"
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

    def _processed_terminal_presentation_is_current(
        self, presentation: TerminalBrowsePresentation,
    ) -> bool:
        try:
            captured = self._context_controller.capture_loaded_browse(
                presentation.request
            )
        except BaseException:
            captured = None
        return self._processed_browser.terminal_presentation_is_current(
            presentation,
            captured,
            owns_request=self._context_controller.owns_browse_request(
                presentation.request
            ),
        )

    def _begin_terminal_xye_viewer(
        self,
        event: StandardRunEvent,
        *,
        current: DisplayFrameKey | None,
    ) -> bool:
        """Show a completed XYE Run's terminal curve and its output folder."""

        controller = self._context_controller
        acquisition = controller.acquisition_context
        if (
            event.kind is not StandardEventKind.FINISHED
            or event.cleanup_status is not CleanupStatus.CLEANED
            or self._lifecycle.phase is not RunPhase.IDLE
            or controller.run_identity is not event.run_identity
            or acquisition is None
            or type(acquisition.run_configuration) is not FrozenRunConfiguration
            or acquisition.run_configuration.processing_mode != "Int 1D (XYE)"
        ):
            return False
        frame = current
        if (frame is None or frame.run_identity is not event.run_identity
                or frame.artifact not in event.artifacts):
            frame = next((item for item in reversed(controller.frame_keys)
                          if item.run_identity is event.run_identity
                          and item.artifact in event.artifacts), None)
        if frame is None:
            return False
        plan = native_int_reduction_plan(acquisition.run_configuration)
        prefix = xye_prefix_for_unit(plan.integration_1d.unit)
        # The event artifacts are planned NeXus slots, not written files in
        # XYE-only mode. Match TransactionalXYESink's output family/index naming for
        # the terminal row; the other generated files remain folder-browsable.
        # Do not turn completion into a multi-file Viewer admission/overlay.
        family = _run_artifact_family(acquisition.run_configuration, frame.source_scan)
        current_path = str(Path(frame.artifact).parent / family /
                           f"{prefix}_{family}_{frame.local_frame_label:04d}.xye")
        # This viewer handoff replaces the terminal batch paint itself.
        self._retire_batch_presentation(force=True)
        self._set_browser_directory(str(Path(current_path).parent), explicit=False)
        self._open_viewer_1d_paths((current_path,), current_path=current_path)
        self._notice(f"Completed {event.completed} XYE files · showing {Path(current_path).name}")
        return True

    def _begin_terminal_browse(
        self,
        event: StandardRunEvent,
        *,
        was_batch: bool,
        current: DisplayFrameKey | None,
        selected: tuple[DisplayFrameKey, ...],
    ) -> bool:
        """Load a clean run's exact published artifact through Browse.

        Reintegration deliberately accepts only a stable authenticated Browse
        context.  A finished run or a stopped run's saved prefix can follow
        the existing asynchronous BrowseLoader without a redundant browser
        click or a weaker external-tool/reintegration qualification.
        """

        controller = self._context_controller
        acquisition = controller.acquisition_context
        selection = controller.selection
        artifact = event.artifact
        run_identity = event.run_identity
        if (
            was_batch
            or event.kind not in {StandardEventKind.FINISHED, StandardEventKind.STOPPED}
            or event.kind is StandardEventKind.STOPPED
            and (event.artifact_completed <= 0 or artifact not in event.artifacts)
            or event.cleanup_status is not CleanupStatus.CLEANED
            or self._lifecycle.phase is not RunPhase.IDLE
            or type(artifact) is not str
            or not artifact
            or acquisition is None
            or type(run_identity) is not RunIdentity
            or controller.run_identity is not run_identity
            or selection is None
            or selection.kind is not ContextKind.ACQUISITION
            or not selection.names(acquisition)
            or type(acquisition.run_configuration)
            is not FrozenRunConfiguration
            or acquisition.run_configuration.processing_mode
            == "Int 1D (XYE)"
            or current is not None
            and current.run_identity is not run_identity
            or any(
                frame.run_identity is not run_identity
                for frame in selected
            )
        ):
            return False
        timing_start = self._processed_browser.begin_terminal_timing(
            enabled=_browse_perf_enabled()
        )
        try:
            if not self._release_browse_1d_debt():
                self._processed_browser.retire_terminal_timing(timing_start)
                return False
            commit_identity = event.terminal_commit_identity
            source_root = acquisition.run_configuration.project_root or None
            request = (
                controller.begin_browse(
                    artifact,
                    source_root=source_root,
                )
                if commit_identity is None
                else controller.begin_browse(
                    artifact,
                    terminal_commit_identity=commit_identity,
                    source_root=source_root,
                )
            )
        except RuntimeError:
            self._processed_browser.retire_terminal_timing(timing_start)
            # The published artifact remains available in the browser.  A
            # transient viewer/cleanup owner must not turn a clean Run terminal
            # into a failure or bypass Browse's normal lifecycle checks.
            return False
        except BaseException:
            self._processed_browser.retire_terminal_timing(timing_start)
            raise
        # Browse owns the resolved absolute source path, while the output
        # contract deliberately preserves an explicit target's spelling.
        # Retain both exact identities so acquisition-frame choices made while
        # Browse is pending compare against their original artifact spelling.
        handoff = self._processed_browser.begin_terminal_handoff(
            request,
            run_identity,
            artifact,
            (
                current.local_frame_label
                if current is not None and current.artifact == artifact
                else None
            ),
            tuple(
                frame.local_frame_label
                for frame in selected
                if frame.artifact == artifact
            ),
            request.terminal_commit_identity,
            timing_start=timing_start,
        )
        if handoff is None:
            self._processed_browser.retire_terminal_timing(timing_start)
            return False
        self._ensure_timer()
        return True

    def _terminal_scientific_matches(
        self,
        handoff: TerminalBrowseHandoff,
        navigation: FrameNavigationProjection,
    ) -> bool:
        """Whether terminal Browse can reuse the exact painted science."""

        preferences = self._preferences
        view = self._shell.scientific
        scientific = self._last_scientific_projection
        current = navigation.current
        painted_current = view.navigation_current_key
        run_identity = handoff.run_identity
        scientific_frames = (
            ()
            if scientific is None
            else (
                *((scientific.heavy.frame,)
                  if scientific.heavy is not None else ()),
                *(trace.frame for trace in scientific.traces),
                *scientific.heavy_available,
            )
        )
        if (
            type(run_identity) is not RunIdentity
            or self._context_controller.run_identity is not run_identity
            or self._context_controller.acquisition_context is None
            or preferences.plot_mode not in {"Single", "Overlay", "Waterfall"}
            or preferences.slice_enabled
            or preferences.slice_pins
            or preferences.norm_channel.strip()
            not in {"", "None", "Norm Channel"}
            or self._background_owner.active_key is not None
            or scientific is None
            or scientific.plot_mode != preferences.plot_mode
            or scientific.plot_options != preferences.plot_options
            or scientific.slice_pins
            or scientific.pinned_traces
            or view.presentation_plot_mode != preferences.plot_mode
            or current is None
            or painted_current is None
        ):
            return False
        artifact = handoff.request.source_path
        canonical_by_artifact = {
            handoff.source_artifact: artifact,
            artifact: artifact,
        }
        compared_frames = (
            current,
            painted_current,
            *navigation.frames,
            *view.navigation_frame_keys,
            *view.navigation_selected_keys,
            *navigation.selected,
            *view.trace_history_keys,
            *view.heavy_available_keys,
            *scientific_frames,
        )
        if any(
            type(frame) is not DisplayFrameKey
            or frame.run_identity is not run_identity
            or frame.artifact not in canonical_by_artifact
            for frame in compared_frames
        ):
            return False
        local_frames = tuple(
            frame
            for frame in navigation.frames
            if _terminal_frame_signature(
                frame, canonical_by_artifact,
            )[0] == artifact
        )
        waterfall_exact = True
        if view.bottom_waterfall_active:
            options = scientific.plot_options
            expected_waterfall = view.trace_history_keys[
                options.waterfall_start - 1:
                options.waterfall_stop or None:
                options.waterfall_step
            ]
            painted_waterfall = view.waterfall_source_frame_keys
            waterfall_exact = (
                len(painted_waterfall) == len(expected_waterfall)
                and all(
                    painted is expected
                    for painted, expected in zip(
                        painted_waterfall,
                        expected_waterfall,
                        strict=True,
                    )
                )
            )
        matches = (
            waterfall_exact
            and _terminal_frame_signature(
                current, canonical_by_artifact,
            )[0] == artifact
            and _terminal_frame_signature(
                painted_current, canonical_by_artifact,
            )
            == _terminal_frame_signature(current, canonical_by_artifact)
            and tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in view.navigation_frame_keys
            )
            == tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in local_frames
            )
            and tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in view.navigation_selected_keys
            )
            == tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in navigation.selected
            )
            and tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in view.trace_history_keys
            )
            == tuple(
                _terminal_frame_signature(frame, canonical_by_artifact)
                for frame in navigation.selected
            )
        )
        if not matches or preferences.plot_mode != "Single":
            return matches
        current_signature = _terminal_frame_signature(
            current, canonical_by_artifact,
        )
        return (
            len(navigation.selected) == 1
            and navigation.selected[0] is current
            and len(view.navigation_selected_keys) == 1
            and view.navigation_selected_keys[0] is painted_current
            and len(view.trace_history_keys) == 1
            and view.trace_history_keys[0] is painted_current
            and len(scientific.traces) == 1
            and scientific.traces[0].frame is painted_current
            and scientific.heavy is not None
            and scientific.heavy.frame is painted_current
            and any(
                frame is painted_current
                for frame in view.heavy_available_keys
            )
            and any(
                frame is painted_current
                for frame in scientific.heavy_available
            )
            and _terminal_frame_signature(
                painted_current, canonical_by_artifact,
            ) == current_signature
        )

    def _rebind_last_scientific_projection(
        self,
        authorization: TerminalRebindAuthorization,
    ) -> bool:
        scientific = self._last_scientific_projection
        if (
            scientific is None
            or type(authorization) is not TerminalRebindAuthorization
        ):
            return False
        navigation = authorization.browse_navigation
        source_by_id = {
            id(old): new for old, new in authorization.frame_pairs
        }
        scientific_frames = (
            *((scientific.heavy.frame,) if scientific.heavy is not None else ()),
            *(trace.frame for trace in scientific.traces),
            *scientific.heavy_available,
        )
        if any(
            type(frame) is not DisplayFrameKey
            or source_by_id.get(id(frame)) is None
            for frame in scientific_frames
        ):
            return False
        heavy_frame = (
            None
            if scientific.heavy is None
            else source_by_id.get(id(scientific.heavy.frame))
        )
        trace_frames = tuple(
            source_by_id.get(id(trace.frame)) for trace in scientific.traces
        )
        available = tuple(
            source_by_id.get(id(frame)) for frame in scientific.heavy_available
        )
        if (
            scientific.heavy is not None and heavy_frame is None
            or any(frame is None for frame in trace_frames)
            or any(frame is None for frame in available)
        ):
            return False
        rebound_available = (
            frozenset((navigation.current,))
            if scientific.plot_mode == "Single"
            and navigation.current is not None
            else frozenset(available)
        )
        owned_by_id = {id(frame): frame for frame in navigation.frames}
        if (
            navigation.current is None
            or not any(
                frame is navigation.current for frame in rebound_available
            )
            or any(
                owned_by_id.get(id(frame)) is not frame
                for frame in rebound_available
            )
        ):
            return False
        self._last_scientific_projection = replace(
            scientific,
            heavy=(
                None
                if scientific.heavy is None
                else replace(scientific.heavy, frame=heavy_frame)
            ),
            traces=tuple(
                replace(trace, frame=frame)
                for trace, frame in zip(
                    scientific.traces, trace_frames, strict=True,
                )
            ),
            heavy_available=rebound_available,
        )
        return True

    def _settle_terminal_browse(
        self,
        outcome: object,
    ) -> bool:
        handoff = self._processed_browser.terminal_handoff
        request = None if handoff is None else handoff.request
        if (
            type(outcome) is not BrowseLoadOutcome
            or request is None
            or outcome.request is not request
        ):
            return False
        captured = (
            self._context_controller.capture_loaded_browse(request)
            if outcome.status is BrowseLoadStatus.READY
            else None
        )
        settlement = self._processed_browser.settle_terminal(
            outcome, captured
        )
        if settlement is None:
            return False
        handoff = settlement.handoff
        navigation = self._context_controller.navigation
        by_label = {
            frame.local_frame_label: frame for frame in navigation.frames
        }
        current = by_label.get(handoff.current_label)
        selected = tuple(
            by_label[label]
            for label in handoff.selected_labels
            if label in by_label
        )
        if self._processed_browser.auto_last and navigation.frames:
            self._context_controller.select_navigation(
                navigation.frames[-1], selected
            )
        elif current is not None:
            self._context_controller.select_navigation(current, selected)
        if not settlement.reuse_seal_authorized:
            # An ordinary terminal Browse may preserve its exact frame-label
            # choice, but only a writer-authenticated seal can authorize
            # zero-copy reuse of the acquisition scientific arrays.
            return False
        view = self._shell.scientific
        try:
            source_navigation = FrameNavigationProjection(
                view.navigation_frame_keys,
                view.navigation_current_key,
                view.navigation_selected_keys,
            )
        except (TypeError, ValueError):
            return False
        authorization = (
            self._processed_browser.build_terminal_rebind_authorization(
                handoff,
                source_navigation,
                self._context_controller.navigation,
            )
        )
        matches = self._terminal_scientific_matches(
            handoff, self._context_controller.navigation,
        )
        if matches and authorization is not None:
            self._processed_browser.authorize_terminal_rebind(
                settlement.presentation,
                authorization,
            )
        return bool(matches and authorization is not None)

    def _begin_processed_terminal_paint(
        self,
        presentation: TerminalBrowsePresentation,
        *,
        reuse_science: bool,
    ) -> TerminalBrowsePaintRequest | None:
        try:
            captured = self._context_controller.capture_loaded_browse(
                presentation.request
            )
        except BaseException:
            captured = None
        return self._processed_browser.begin_terminal_paint(
            presentation,
            captured,
            owns_request=self._context_controller.owns_browse_request(
                presentation.request
            ),
            reuse_science=reuse_science,
        )

    def _complete_processed_terminal_paint(
        self,
        request: TerminalBrowsePaintRequest,
        *,
        applied: bool,
    ) -> None:
        completion = self._processed_browser.complete_terminal_paint(
            TerminalBrowsePaintReceipt(
                request,
                applied,
                self._scientific_repaint_pending,
            )
        )
        if completion.schedule_repaint:
            self._scientific_repaint_pending = True
            self._ensure_timer()
            return
        if completion.retired:
            self._refresh_event_shell(
                preserve_scientific=True,
                skip_scientific_projection=True,
                suppress_detector_demand=True,
            )

    def _refresh_event_shell(
        self,
        *,
        preserve_scientific: bool = False,
        skip_scientific_projection: bool = False,
        rebind_scientific_navigation: bool = False,
        suppress_detector_demand: bool = False,
        allow_batch_terminal_paint: bool = False,
        terminal_paint: TerminalBrowsePaintRequest | None = None,
    ) -> None:
        started = (
            time.monotonic()
            if self._quartile_refresh_identity is not None
            else None
        )
        if preserve_scientific:
            self._refresh_shell(
                preserve_scientific=True,
                skip_scientific_projection=skip_scientific_projection,
                rebind_scientific_navigation=rebind_scientific_navigation,
                suppress_detector_demand=suppress_detector_demand,
                allow_batch_terminal_paint=allow_batch_terminal_paint,
                terminal_paint=terminal_paint,
            )
        else:
            self._refresh_shell(
                suppress_detector_demand=suppress_detector_demand,
                allow_batch_terminal_paint=allow_batch_terminal_paint,
                terminal_paint=terminal_paint,
            )
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
            published = prior.published
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
        operations = getattr(self, "_workspace_operations", None)
        analysis_slot = getattr(self, "_analysis_slot", None)
        try:
            if (
                (operations is not None and operations.owned)
                or (analysis_slot is not None and analysis_slot.owned)
            ):
                return True
        except Exception:
            return True
        authored = getattr(self, "_authored_assets", None)
        if getattr(authored, "phase", None) in {
            AuthoredAssetPhase.TERMINAL_READY,
            AuthoredAssetPhase.CONFIRM_ISSUED,
        }:
            return True
        if (
            self._metadata_operations.polling_needed
            or self._admission is not None
            or getattr(self._context_controller, "viewer_1d_loading", False)
            or getattr(self._context_controller, "viewer_1d_cleanup_pending", False)
            or self._context_controller.viewer_2d_loading
            or self._context_controller.viewer_poll_pending
            or self._pending_viewer_2d_path is not None
            or self._context_controller.browse_pending
            or self._context_controller.browse_preview_polling_needed
            or self._browse_1d_release_debt is not None
            or self._processed_browser.polling_needed
            or self._scientific_repaint_pending
        ):
            return True
        if self._batch_terminal.needs_polling:
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

    def _release_browse_1d_debt(self) -> bool:
        bundle = self._browse_1d_release_debt
        if bundle is None:
            return True
        try:
            bundle.release()
        except BaseException:
            if not self._closing and not self._closed:
                self._ensure_timer()
            return False
        if not bundle.released:
            if not self._closing and not self._closed:
                self._ensure_timer()
            return False
        self._browse_1d_release_debt = None
        return True

    def _settle_browse_1d_before_drain(self) -> bool:
        """Gate all poll-driven context mutation on page-held cache custody."""

        if not self._release_browse_1d_debt():
            return False
        self._retry_pending_average_reload()
        return self._browse_1d_release_debt is None

    def _set_detector_mode(self, mode: str) -> bool:
        if mode == "thumbnail":
            terminal = self._batch_terminal.presentation
            terminal_was_waiting = bool(
                terminal is not None and terminal.awaiting_full_raw
            )
            if self._preferences.detector_mode != mode:
                self._retire_batch_presentation()
            if self._preferences.detector_mode != mode: self._release_display_background()
            if not terminal_was_waiting:
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
        if self._preferences.detector_mode != mode:
            self._retire_batch_presentation()
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

    def _browse_snapshot_is_exact_current(
        self,
        scientific: object,
        requested_contract: object,
    ) -> bool:
        """Authorize transient retention only for the adopted Browse scope."""

        snapshot = getattr(scientific, "browse_trace_snapshot", None)
        captured = self._capture_current_loaded_browse()
        navigation = self._context_controller.navigation
        if (
            snapshot is None
            or type(requested_contract) is not tuple
            or not requested_contract
            or captured is None
            or getattr(scientific, "browse_science_contract", None)
            != requested_contract
            or snapshot.science_contract != requested_contract
            or snapshot.plot_mode != self._preferences.plot_mode
            or getattr(scientific, "plot_mode", None) != snapshot.plot_mode
        ):
            return False
        request = captured.request
        selection = captured.selection
        artifact = captured.target
        if (
            selection is not self._context_controller.selection
            or request.source_path != artifact
            or navigation.current is None
        ):
            return False
        expected_logical = navigation.selected
        logical = snapshot.logical_frames
        display = snapshot.display_frames
        traces = getattr(scientific, "traces", ())
        heavy = getattr(scientific, "heavy", None)
        heavy_available = getattr(scientific, "heavy_available", ())
        return bool(
            len(logical) == len(expected_logical)
            and all(
                retained is current
                and retained.artifact == artifact
                and self._context_controller.owns_frame(retained)
                for retained, current in zip(
                    logical, expected_logical, strict=True,
                )
            )
            and len(traces) == len(display)
            and all(
                trace.frame is frame
                and frame.artifact == artifact
                and self._context_controller.owns_frame(frame)
                for trace, frame in zip(traces, display, strict=True)
            )
            and all(
                frame.artifact == artifact
                and self._context_controller.owns_frame(frame)
                for frame in heavy_available
            )
            and (
                heavy is None
                or heavy.frame is navigation.current
                and heavy.frame.artifact == artifact
                and self._context_controller.owns_frame(heavy.frame)
            )
        )

    def _refresh_shell(
        self,
        *,
        preserve_display: bool = False,
        preserve_scientific: bool = False,
        skip_scientific_projection: bool = False,
        rebind_scientific_navigation: bool = False,
        suppress_detector_demand: bool = False,
        allow_batch_terminal_paint: bool = False,
        terminal_paint: TerminalBrowsePaintRequest | None = None,
    ) -> None:
        explicit_preserve_display = preserve_display
        explicit_preserve_scientific = preserve_scientific
        allow_batch_terminal_paint = bool(
            allow_batch_terminal_paint
            and self._batch_ready_to_paint() is not None
        )
        batch_science_hold = (
            self._batch_terminal.active and not allow_batch_terminal_paint
            or self._retain_outgoing_display and self._progress.terminal
        )
        # The worker can rescope the one mutable acquisition context after its
        # event queue snapshot but before this GUI refresh.  Catch up only an
        # already-selected exact acquisition owner; Browse remains untouched.
        if self._context_controller.synchronize_acquisition_scope():
            if not batch_science_hold:
                preserve_scientific = False
                skip_scientific_projection = False
                rebind_scientific_navigation = False
        if explicit_preserve_display or explicit_preserve_scientific:
            preserve_scientific = True
            skip_scientific_projection = True
        if batch_science_hold:
            preserve_display = True
            preserve_scientific = True
            skip_scientific_projection = True
            rebind_scientific_navigation = False
            suppress_detector_demand = True
        controller = self._context_controller
        browse = controller.browse_context
        selection = controller.selection
        selected_invalidated_browse = (
            type(browse) is BrowseContext
            and browse.invalidated
            and not browse.released
            and type(selection) is DisplaySelection
            and selection.kind is ContextKind.BROWSE
            and selection.names(browse)
        )
        pending_browse_replacement = (
            controller.browse_pending
            and browse is None
            and type(selection) is DisplaySelection
            and selection.kind is ContextKind.BROWSE
        )
        if (
            selected_invalidated_browse
            or self._processed_browser.preserve_science
            or pending_browse_replacement
        ):
            preserve_scientific = True
            skip_scientific_projection = True
            rebind_scientific_navigation = False
            suppress_detector_demand = True
        if (
            (skip_scientific_projection or rebind_scientific_navigation)
            and not preserve_scientific
        ):
            raise ValueError(
                "scientific projection shortcuts require paint preservation"
            )
        snapshot = self._intents.snapshot()
        intent = snapshot.thaw()
        controls = self._project_controls(snapshot)
        permitted, blocker = self._start_permitted()
        tool = tool_from_mode_text(intent.processing_mode)
        viewer_2d = (tool is Tool.IMAGE_VIEWER or selection is not None
                     and selection.kind is ContextKind.VIEWER_2D)
        viewer_1d = (tool is Tool.XYE_VIEWER or selection is not None
                     and selection.kind is ContextKind.VIEWER_1D)
        viewer = viewer_1d or viewer_2d
        if tool in {Tool.XYE_VIEWER, Tool.IMAGE_VIEWER, Tool.STITCH, Tool.RSM}:
            cleanup = (self._context_controller.viewer_1d_cleanup_pending
                       if viewer_1d else self._context_controller.viewer_2d_cleanup_pending)
            mutating_busy = self._mutating_operation_busy()
            blocked = (self._closing or self._closed
                       or cleanup
                       or mutating_busy
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
                       else "Analysis operation is still active" if mutating_busy and self._analysis_operation_busy()
                       else "Experiment operation is still active" if mutating_busy
                       else "Viewer unavailable during active run" if blocked else "")
        navigation = self._context_controller.navigation
        if not suppress_detector_demand:
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
        # One capture owns every projection branch, including the sparse cache
        # route and preserve-only shell refreshes.
        self._context_controller.capture_norm_aggregate_for_refresh()
        cache_trace_snapshot = None
        cache_plan = None
        cache_preferences = None
        cache_current_preview = None
        cache_adoption_missing = False
        cache_terminal_diagnostic = ""
        cache_retry_needed = False
        selection = self._context_controller.selection
        browse_selected = bool(
            selection is not None and selection.kind is ContextKind.BROWSE
        )
        requested_plot_axis = self._preferences.plot_axis
        if self._preferences.share_axis:
            requested_plot_axis = (
                share_plot_axis_for_image(self._preferences.image_axis)
                or requested_plot_axis
            )
        # Sparse 1-D rows can serve only their native axis (or Q/2theta
        # conversion). Every other offered axis comes from the selected cake,
        # including GI chi and Q derived from a qip/qoop-only result.
        cake_axis_requested = requested_plot_axis == "chi"
        if browse_selected and browse is not None and navigation.current is not None:
            catalog = browse.scalar_catalog
            row = None if catalog is None else catalog.row(navigation.current.local_frame_label)
            if row is not None:
                native_axis = next((
                    native_1d_plot_axis(unit)
                    for mode, _label, unit, _log in catalog.axes_1d
                    if mode == row.active_mode_1d
                ), None)
                cake_axis_requested = requested_plot_axis != native_axis and not (
                    {requested_plot_axis, native_axis} <= {"Q", "2theta"}
                )
        browse_cache_supported = bool(
            browse_selected
            and intent.processing_mode in {"Int 1D", "Int 2D"}
            and self._preferences.plot_mode in {
                "Single", "Overlay", "Waterfall",
            }
            and not self._preferences.slice_enabled
            and not self._preferences.slice_pins
        )
        browse_slices = bool(
            browse_selected
            and intent.processing_mode == "Int 2D"
            and self._preferences.plot_mode in {"Single", "Overlay", "Waterfall"}
            and (self._preferences.slice_enabled or self._preferences.slice_pins
                 or cake_axis_requested)
        )
        if not browse_slices:
            self._context_controller.cancel_browse_slices()
        if skip_scientific_projection:
            payloads = ()
        elif browse_slices:
            # Heavy images remain on the exact-current preview lane. Selected
            # cuts read only saved cakes independently, off the GUI thread.
            preview = (
                self._context_controller.request_current_browse_preview()
                if self._release_browse_1d_debt() else None
            )
            payloads = () if preview is None else (preview,)
        elif browse_selected and not browse_cache_supported:
            payloads = ()
            cache_adoption_missing = True
            cache_terminal_diagnostic = (
                "Browse cache display is unavailable for Average/Sum or this operation."
            )
            preserve_scientific = True
            self._scientific_repaint_pending = False
            self._notice(cache_terminal_diagnostic)
        elif browse_cache_supported:
            payloads = ()
            adopted = None
            cache_runtime_status = None
            runtime = None
            cache_preferences = self._preferences
            if self._release_browse_1d_debt():
                # Detector/cake hydration stays on the exact-current preview
                # lane; sparse 1-D reads are admitted only by the target plan.
                cache_current_preview = (
                    self._context_controller.request_current_browse_preview()
                )
                runtime = self._context_controller.project_browse_1d_cache(
                    preferences=self._preferences,
                    was_waterfall_active=(
                        self._shell.scientific.bottom_waterfall_active
                    ),
                )
                if runtime is not None:
                    cache_runtime_status = runtime.status
                    try:
                        adopted = prepare_browse_1d_display(runtime)
                    except Browse1DReleaseDebt as error:
                        self._browse_1d_release_debt = error.bundle
                        cache_retry_needed = True
                    except Browse1DDisplayRefusal as error:
                        cache_terminal_diagnostic = str(error)
                        self._notice(cache_terminal_diagnostic)
            else:
                cache_retry_needed = True
            if adopted is not None and (
                self._preferences is not cache_preferences
                or not self._context_controller.browse_1d_plan_is_current(
                    adopted.plan
                )
            ):
                adopted = None
                cache_retry_needed = True
            if adopted is None:
                # Every refusal is fail-closed, but only an incomplete sparse
                # read, an exact release debt, or detected drift can make
                # progress by polling.  Terminal REFUSED must not spin.
                cache_retry_needed = _browse_1d_cache_retry_needed(
                    cache_runtime_status,
                    transient=(
                        cache_retry_needed
                        or self._browse_1d_release_debt is not None
                    ),
                )
                cache_adoption_missing = True
                if not cache_retry_needed:
                    cache_terminal_diagnostic = (
                        cache_terminal_diagnostic
                        or (
                            runtime.diagnostic
                            if runtime is not None
                            else "Browse 1-D runtime is unavailable."
                        )
                        or "Browse 1-D display was refused."
                    )
                    self._notice(cache_terminal_diagnostic)
                # Compatibility is checked against the fully projected
                # presentation contract below.  Until then, do not expose a
                # payload-free projection that could retain a stale hybrid.
                preserve_scientific = True
                self._scientific_repaint_pending = cache_retry_needed
                if cache_retry_needed:
                    self._ensure_timer()
            else:
                payloads = adopted.payloads
                cache_trace_snapshot = adopted.trace_snapshot
                cache_plan = adopted.plan
                current = navigation.current
                if (
                    type(cache_current_preview) is StandardDisplayPayload
                    and current is not None
                    and cache_current_preview.frame_key is current
                    and not any(
                        payload.frame_key is current for payload in payloads
                    )
                ):
                    # A plot-options-filtered current remains a heavy-only
                    # participant; it is never renumbered into the sparse rows.
                    payloads = (*payloads, cache_current_preview)
                self._scientific_repaint_pending = False
        else:
            payloads = self._context_controller.project_navigation(
                preferences=self._preferences,
                processing_mode=intent.processing_mode,
                live_update=live_update,
            )
        resident_frames = self._context_controller.resident_frame_keys
        current = navigation.current
        replacement_identity = (
            self._lifecycle.active_run_identity
            or self._lifecycle.attempt_run_identity
        )
        replacement_ready = (
            replacement_identity is not None
            and self._context_controller.run_identity is replacement_identity
            and current is not None
            and current.run_identity is replacement_identity
            and any(
                payload.frame_key is current for payload in payloads
            )
            and any(frame is current for frame in resident_frames)
        )
        if (
            not self._batch_terminal.active
            and
            not preserve_scientific
            and self._retain_outgoing_display
            and replacement_ready
        ):
            self._retain_outgoing_display = False
        preserve_display = (
            preserve_display
            or self._retain_outgoing_display
            and not batch_science_hold
            and not explicit_preserve_scientific
        )
        observation = self._source_selection.observation
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
        projection_progress = self._batch_terminal.project_progress(
            self._progress
        )
        browser = self._processed_browser.projection()
        viewer_current, viewer_paths = (
            self._context_controller.viewer_1d_artifact_selection
            if viewer_1d else ("", ()))
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
            progress=projection_progress,
            preferences=self._preferences,
            browser_directory=browser.directory,
            browser_catalog_index=browser.scan_index,
            browser_transient_frame=browser.transient_frame,
            viewer_1d_current_path=viewer_current,
            viewer_1d_selected_paths=viewer_paths,
            viewer_waterfall_active=self._shell.scientific.bottom_waterfall_active,
            date_sorted=browser.date_sorted,
            auto_last=browser.auto_last,
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
        if browse_slices and not skip_scientific_projection:
            science = projection.scientific
            cuts = self._context_controller.project_browse_slices(
                preferences=replace(self._preferences, plot_axis=requested_plot_axis),
                norm_channel=("" if science.norm_channel == "Norm Channel" else science.norm_channel),
            )
            pending = cuts is not None and cuts.pending
            diagnostic = ("Browse slicing is unavailable." if cuts is None
                          else cuts.diagnostic or ("Loading selected slices…" if pending else ""))
            self._scientific_repaint_pending = pending
            if pending:
                self._ensure_timer()
            if diagnostic != self._notice_text:
                self._notice(diagnostic)
            projection = replace(projection, scientific=replace(
                science,
                plot_axis=requested_plot_axis,
                traces=() if cuts is None else cuts.traces,
                pinned_traces=() if cuts is None else cuts.pinned_traces,
                replace_trace_history=True,
                status=diagnostic,
            ))
        projection = replace(
            projection,
            scientific=replace(
                projection.scientific,
                background_enabled=not self._mutating_operation_busy(),
            ),
            external_tools=self._external_tools.project(
                nexus=self._qualify_external_nexus(
                    validate_disk=False,
                ),
            ),
        )
        pending_average = self._workspace_operations.average_pending
        if (
            pending_average is not None
            and self._workspace_operations.current_identity
            is self._workspace_operations.average_identity
        ):
            pending_readiness = (
                "Average cleanup pending · Run retries · Stop cancels"
            )
            projection = replace(
                projection,
                run=replace(
                    projection.run,
                    readiness=pending_readiness,
                    readiness_tooltip=pending_average.diagnostic,
                    ready=True,
                    run_enabled=True,
                    stop_enabled=True,
                ),
            )
        if cache_adoption_missing:
            prior = self._last_scientific_projection
            retention_compatible = self._browse_snapshot_is_exact_current(
                prior,
                projection.scientific.browse_science_contract,
            )
            if cache_terminal_diagnostic or not retention_compatible:
                if (cache_retry_needed and not cache_terminal_diagnostic
                        and prior is not None and (prior.traces or prior.heavy is not None)):
                    # Keep only the existing capped, labeled screen raster;
                    # foreign scientific arrays are still released below.
                    self._shell.scientific._begin_viewer_loading_snapshot(
                        projection.scientific.processing_mode)
                # INCOMPLETE may retain only an exact-current Browse snapshot.
                # A terminal refusal/no-runtime state cannot, and a changed
                # request, navigation, artifact, mode, axis, or options cannot
                # label foreign science as the requested pending presentation.
                diagnostic = (
                    cache_terminal_diagnostic
                    or "Loading Browse 1-D display for changed settings…"
                )
                preserve_scientific = False
                preserve_display = False
                projection = replace(
                    projection,
                    scientific=replace(
                        projection.scientific,
                        heavy_available=frozenset(),
                        traces=(),
                        heavy=None,
                        title="Current",
                        status=diagnostic,
                        retain_display=False,
                        browse_trace_snapshot=None,
                    ),
                )
        if cache_trace_snapshot is not None:
            trace_by_id = {
                id(trace.frame): trace
                for trace in projection.scientific.traces
            }
            display_traces = tuple(
                trace_by_id[id(frame)]
                for frame in cache_trace_snapshot.display_frames
                if (
                    id(frame) in trace_by_id
                    and trace_by_id[id(frame)].frame is frame
                )
            )
            if len(display_traces) != len(
                cache_trace_snapshot.display_frames
            ):
                self._scientific_repaint_pending = True
                self._ensure_timer()
                return
            cache_trace_snapshot = replace(
                cache_trace_snapshot,
                science_contract=(
                    replace(
                        projection.scientific,
                        traces=display_traces,
                    ).browse_science_contract
                ),
            )
            projection = replace(
                projection,
                scientific=replace(
                    projection.scientific,
                    traces=display_traces,
                    browse_trace_snapshot=cache_trace_snapshot,
                ),
            )
        authorization = (
            terminal_paint.authorization
            if (
                rebind_scientific_navigation
                and type(terminal_paint) is TerminalBrowsePaintRequest
                and terminal_paint.mode is TerminalPaintMode.REBIND
            )
            else None
        )
        if cache_trace_snapshot is not None and (
            cache_plan is None
            or self._preferences is not cache_preferences
            or not self._context_controller.browse_1d_plan_is_current(
                cache_plan
            )
            or projection.scientific.browse_science_contract
            != cache_trace_snapshot.science_contract
        ):
            self._scientific_repaint_pending = True
            self._ensure_timer()
            return
        try:
            if not preserve_display:
                self._shell.browser.reconcile_detector_mode(projection.scientific)
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
            if cache_trace_snapshot is not None:
                apply_options["replace_scientific_on_failure"] = True
            self._shell.scientific.expect_display_background(
                self._background_owner.active_key)
            self._shell.apply_state(projection, **apply_options)
            viewer_loading = (
                controller.viewer_1d_loading
                if viewer_1d
                else controller.viewer_2d_loading
                if viewer_2d
                else False
            )
            # Match the shell's paint gate: a completed read can precede the
            # trailing browser gesture, which still postpones scientific paint.
            if (not preserve_display and not preserve_scientific
                    and not self._shell.browser.frame_selection_pending
                    and not viewer_loading
                    and not (cache_adoption_missing and cache_retry_needed)):
                self._shell.scientific.drop_viewer_loading_snapshot()
            if rebind_scientific_navigation:
                prior_scientific = self._last_scientific_projection
                prior_trace_history = (
                    self._shell.scientific.trace_history_projections
                )
                rebound = (
                    type(authorization) is TerminalRebindAuthorization
                    and prior_scientific is not None
                    and prior_scientific.browse_trace_snapshot is None
                    and self._context_controller.terminal_rebind_is_current(
                        authorization,
                    )
                    and self._rebind_last_scientific_projection(
                        authorization,
                    )
                    and self._shell.scientific.rebind_navigation(
                        authorization,
                        heavy_available=(
                            self._last_scientific_projection.heavy_available
                        ),
                    )
                    and self._shell.scientific.reconcile_rebound_trace_axis(
                        prior_trace_history,
                        self._last_scientific_projection,
                        navigation,
                    )
                    and self._context_controller
                    .commit_rebound_navigation_projection(
                        self._shell.scientific.trace_history_keys,
                        preferences=self._preferences,
                        processing_mode=intent.processing_mode,
                    )
                )
                self._scientific_repaint_pending = not rebound
        except Exception as error:
            _LOG.exception("Passive shell render failed")
            self._shell.scientific.drop_viewer_loading_snapshot()
            if cache_trace_snapshot is not None:
                self._last_scientific_projection = None
                self._scientific_repaint_pending = True
                self._ensure_timer()
            if viewer:
                self._last_scientific_projection = None
                if viewer_1d: self._clear_viewer_1d_renderer(close=True)
            self._notice(
                "Passive shell render failed: "
                f"{detached_exception_strings(error)[2]}"
            )
            return
        if not preserve_scientific:
            self._scientific_repaint_pending = bool(
                cache_adoption_missing and cache_retry_needed
            )
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
        if (
            not preserve_display
            and not preserve_scientific
            and not self._shell.browser.frame_selection_pending
        ):
            self._waterfall_candidate_count = (
                self._shell.scientific.trace_row_count
            )
        self._shell_revision += 1

    def _project_folder_edit_permitted(self) -> bool:
        phase = self._lifecycle.phase
        return (
            not self._closing and not self._closed
            and self._admission_state is None
            and not self._experiment_operation_busy()
            and (phase is RunPhase.IDLE
                 or phase is RunPhase.FAILED and self._lifecycle.reset_permitted)
            and not self._context_controller.viewer_1d_cleanup_pending
            and not self._context_controller.viewer_2d_cleanup_pending
        )

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
        observation = self._source_selection.project_observation(snapshot)
        operation_identity = self._workspace_operations.current_identity
        operation_active = (operation_identity is not None and not self._closing and not self._closed)
        operation_busy = self._experiment_operation_busy()
        authored_active = (
            operation_active
            and operation_identity is self._authored_assets.operation_identity
            and self._authored_assets.phase is AuthoredAssetPhase.RUNNING
        )
        calibration_active = authored_active and self._authored_assets.asset == "poni"
        mask_active = authored_active and self._authored_assets.asset == "mask"
        reintegrate_active = (
            operation_active
            and operation_identity
            is self._workspace_operations.reintegrate_identity
        )
        phase = self._lifecycle.phase
        if self._authoring_dependency_availability is None or (
            phase is RunPhase.IDLE
            or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        ):
            # Disabled authoring actions do not need filesystem discovery on
            # every plot refresh. Recheck when eligible; launch also validates.
            self._authoring_dependency_availability = (
                resolve_calibration_executable() is not None,
                resolve_mask_executable() is not None,
            )
        calibrate_dependency_available, mask_dependency_available = (
            self._authoring_dependency_availability
        )
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
            source_mode_override=self._source_selection.mode,
            detector_summary_override=self._detector_summary_text,
            calibrate_available=calibrate_available,
            calibrate_dependency_available=calibrate_dependency_available,
            operation_busy=operation_busy,
            calibration_active=calibration_active,
            mask_available=mask_available,
            mask_dependency_available=mask_dependency_available,
            mask_active=mask_active,
            reintegrate_available=(not self._closing and not self._closed
                and self._admission_state is None
                and not self._context_controller.browse_pending
                and intent.processing_mode != "Int 1D (XYE)"
                and (phase is RunPhase.IDLE or phase is RunPhase.FAILED
                     and self._lifecycle.reset_permitted)
                and self._capture_current_loaded_browse() is not None),
            reintegrate_active=reintegrate_active,
            reintegrate_dimension=(
                self._workspace_operations.reintegrate_dimension
            ),
            reintegrate_stop_accepted=(
                self._workspace_operations.reintegrate_cancel_accepted
            ),
        )
        if not self._project_folder_edit_permitted():
            controls = replace(controls, fields=tuple(
                replace(field, enabled=False,
                        reason="Project Folder is locked during Run or cleanup.")
                if field.path == PROJECT_ROOT else field
                for field in controls.fields
            ))
        if self._analysis_operation_busy():
            conflicting = {
                "calibrate",
                "make_mask",
                "reintegrate_1d",
                "reintegrate_2d",
            }
            section_actions = {
                section: tuple(
                    replace(
                        action,
                        enabled=False,
                        reason="Analysis operation is still active.",
                    )
                    if action.action.value in conflicting
                    else action
                    for action in actions
                )
                for section, actions
                in controls.section_actions.items()
            }
            controls = replace(
                controls,
                section_actions=section_actions,
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
        if self._mutating_operation_busy():
            return False, (
                "Analysis operation is still active"
                if self._analysis_operation_busy()
                else "Experiment operation is still active"
            )
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
            handoff = self._processed_browser.terminal_handoff
            if (
                handoff is not None
                and self._context_controller.owns_browse_request(
                    handoff.request
                )
            ):
                return False, "Loading finalized Browse context…"
            return False, "Browse cleanup remains pending"
        presentation = self._processed_browser.terminal_presentation
        if (
            presentation is not None
            and self._processed_terminal_presentation_is_current(
                presentation
            )
        ):
            return False, "Loading finalized Browse context…"
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
        release_outgoing_display = False
        viewer_source = None
        browse_source = None
        if kind is ShellCommandKind.SET_PROCESSING_MODE:
            if type(value) is not str or not value:
                return
            controller = self._context_controller
            target_tool = tool_from_mode_text(value)
            if os.environ.get("XDART_VIEWER_DEBUG") == "1":
                print("viewer_mode", {"from": candidate.processing_mode, "to": value,
                    "browse": getattr(controller.browse_context, "requested_path", None),
                    "browse_pending": controller.browse_pending,
                    "one_d": getattr(controller.viewer_1d_context, "current_path", None),
                    "one_d_owned": controller.viewer_1d_owned,
                    "two_d": getattr(controller.viewer_2d_context, "original_path", None),
                    "current_artifact": getattr(controller.navigation.current, "artifact", None),
                    "selection": str(controller.selection)}, flush=True)
            if target_tool is Tool.IMAGE_VIEWER:
                selection = controller.selection
                browse = controller.browse_context
                if (
                    browse is not None
                    and selection is not None
                    and selection.names(browse)
                ):
                    viewer_source = browse.requested_path
                else:
                    one_d = controller.viewer_1d_context
                    if (
                        controller.viewer_1d_owned
                        and type(one_d) is Viewer1DContext
                        and one_d is controller._runtime._viewer_1d
                        and selection is not None
                        and selection.kind is ContextKind.VIEWER_1D
                        and selection.names(one_d)
                        and one_d.current_path in one_d.paths
                        and Path(one_d.current_path).suffix.casefold()
                        == ".nexus"
                    ):
                        viewer_source = one_d.current_path
            elif target_tool in {Tool.INT_1D, Tool.INT_2D}:
                # Capture the selected processed path before the viewer clear
                # fence retires its context. Reopen through Browse: entry into
                # a viewer released that owner, even if its row is still shown.
                selection = controller.selection
                one_d = controller.viewer_1d_context
                two_d = controller.viewer_2d_context
                if (controller.viewer_1d_owned and one_d is not None
                        and selection is not None and selection.names(one_d)
                        and one_d.current_path in one_d.paths):
                    browse_source = one_d.current_path
                elif (controller.viewer_2d_owned and two_d is not None
                        and selection is not None and selection.names(two_d)):
                    browse_source = two_d.original_path
                if (browse_source is not None
                        and Path(browse_source).suffix.casefold() != ".nexus"):
                    browse_source = None
            if (controller.viewer_2d_owned
                    and target_tool is not Tool.IMAGE_VIEWER
                    and not self._clear_viewer_2d_renderer(close=True)):
                self._notice("2D Viewer cleanup remains pending")
                return
            if (getattr(controller, "viewer_1d_owned", False)
                    and target_tool is not Tool.XYE_VIEWER
                    and not self._clear_viewer_1d_renderer(close=True)):
                self._notice("1D Viewer cleanup remains pending")
                return
            # Closing XYE restores acquisition navigation. Resolve its source
            # after that clear, just as the browser resolves the restored row.
            if target_tool is Tool.IMAGE_VIEWER and viewer_source is None:
                selection = controller.selection
                acquisition = controller.acquisition_context
                if selection is None and acquisition is not None:
                    selection = controller.select_acquisition()
                frame = controller.navigation.current
                if (acquisition is not None and frame is not None
                        and selection is not None
                        and selection.kind is ContextKind.ACQUISITION
                        and selection.names(acquisition)
                        and controller.owns_frame(frame)
                        and Path(frame.artifact).suffix.casefold() == ".nexus"):
                    viewer_source = frame.artifact
            if value != candidate.processing_mode:
                self._retire_batch_presentation()
                release_outgoing_display = True
                self._retain_outgoing_display = False
                self._release_display_background()
            candidate.processing_mode = value
            if target_tool is not Tool.IMAGE_VIEWER:
                self._pending_viewer_2d_path = None
            if value == "Int 1D (XYE)":
                _drop_nexus_only_performance_options(candidate)
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
        if (viewer_source is not None
                and result.snapshot.thaw().processing_mode == candidate.processing_mode):
            self._open_viewer_2d_path(viewer_source)
        if (browse_source is not None
                and result.snapshot.thaw().processing_mode == candidate.processing_mode):
            self._select_scan(browse_source, reopen=True)
        if kind is ShellCommandKind.SET_PROCESSING_MODE and os.environ.get("XDART_VIEWER_DEBUG") == "1":
            print("viewer_mode_result", {"mode": result.snapshot.thaw().processing_mode,
                "viewer_source": viewer_source, "browse_source": browse_source,
                "two_d_state": getattr(controller.viewer_2d_context, "state", None),
                "two_d_loading": controller.viewer_2d_loading,
                "two_d_diagnostic": controller.viewer_2d_diagnostic,
                "notice": self._notice_text, "selection": str(controller.selection)}, flush=True)
        if release_outgoing_display:
            self._retain_outgoing_display = False
            self._refresh_shell()

    def _choose_viewer_1d_files(self) -> None:
        context = self._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
            if not self._clear_viewer_1d_renderer(
                paths=context.paths,
                current_path=context.current_path,
            ):
                self._notice("1D Viewer cleanup remains pending")
            self._ensure_timer(); return
        try:
            selected = self._viewer_1d_file_chooser(
                self._viewer_1d_start_directory()
            )
            if selected is None:
                return
            if type(selected) is not tuple:
                raise TypeError(
                    "1D Viewer chooser must return a tuple of paths"
                )
        except Exception as error:
            self._error_notice("1D Viewer refused", error); return
        self._open_viewer_1d_paths(selected)

    def _open_viewer_1d_paths(
        self,
        selected: tuple[str, ...],
        *,
        current_path: str | None = None,
    ) -> None:
        if (type(selected) is not tuple or not selected
                or any(type(path) is not str or not path for path in selected)
                or len(set(selected)) != len(selected)):
            return
        current_path = selected[0] if current_path is None else current_path
        if type(current_path) is not str or current_path not in selected:
            return
        self._retire_batch_presentation()
        self._retain_outgoing_display = False
        if (self._context_controller.viewer_2d_owned
                and not self._clear_viewer_2d_renderer(close=True)):
            self._notice("2D Viewer cleanup remains pending"); return
        context = self._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
            navigation = self._context_controller.navigation
            if (len(navigation.frames) == len(context.paths)
                    and all(path in context.paths for path in selected)):
                frame_by_path = dict(zip(context.paths, navigation.frames))
                if self._context_controller.select_viewer_1d(
                    frame_by_path[current_path],
                    tuple(frame_by_path[path] for path in selected),
                ):
                    # Already-owned files are a resident selection edit.
                    # Explicit Reload and new files retain the clear fence.
                    self._notice("")
                    self._refresh_shell()
                    return
            if not self._clear_viewer_1d_renderer(
                paths=selected,
                current_path=current_path,
            ):
                self._notice("1D Viewer cleanup remains pending")
            self._ensure_timer(); return
        try:
            request = self._context_controller.open_viewer_1d(
                selected,
                current_path=current_path,
            )
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

    def _clear_viewer_1d_renderer(
        self,
        *,
        paths=None,
        current_path: str | None = None,
        close=False,
    ) -> bool:
        self._last_scientific_projection = None
        request = self._context_controller.begin_viewer_1d_renderer_clear(
            paths,
            current_path=current_path,
        )
        if request is not None and close: self._context_controller.close_viewer_1d()
        if request is None:
            if close:
                self._shell.scientific.drop_viewer_loading_snapshot()
            context = self._context_controller.viewer_1d_context
            cleared = context is None or context.state.value != "ready"
        elif getattr(request, "acknowledgement_identity", None) is not None:
            self._context_controller.poll_viewer_1d()
            cleared = not self._context_controller.viewer_1d_cleanup_pending
        else:
            try:
                receipt = (
                    self._shell.scientific.clear_viewer_1d(request, preserve_navigation=True)
                    if paths is not None and not close
                    else self._shell.scientific.clear_viewer_1d(request)
                )
            except Exception: return False
            try: cleared = self._context_controller.acknowledge_viewer_1d_renderer_clear(receipt)
            except Exception: return False
        return bool(cleared and (not close or self._context_controller.close_viewer_1d()))

    def _choose_viewer_2d_file(self, *, reload: bool) -> None:
        context = self._context_controller.viewer_2d_context
        if reload:
            selected = None if context is None else context.original_path
        else:
            try:
                selected = self._viewer_2d_file_chooser(
                    self._viewer_2d_start_directory())
            except Exception as error:
                self._error_notice("2D Viewer chooser failed", error)
                return
            if selected is None or type(selected) is not str or not selected:
                return
        self._open_viewer_2d_path(selected)

    def _open_viewer_2d_path(self, selected: str) -> None:
        if type(selected) is not str or not selected:
            return
        self._retire_batch_presentation()
        self._retain_outgoing_display = False
        if (getattr(self._context_controller, "viewer_1d_owned", False)
                and not self._clear_viewer_1d_renderer(close=True)):
            self._notice("1D Viewer cleanup remains pending"); return
        context = self._context_controller.viewer_2d_context
        if context is not None and not self._clear_viewer_2d_renderer(
            close=True, preserve_navigation=True,
        ):
            self._pending_viewer_2d_path = selected
            self._notice("2D Viewer cleanup remains pending")
            self._ensure_timer()
            return
        self._pending_viewer_2d_path = None
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

    def _clear_viewer_2d_renderer(self, *, close=False, preserve_navigation=False) -> bool:
        self._last_scientific_projection = None
        request = self._context_controller.begin_viewer_2d_renderer_clear()
        if request is None:
            if close and not preserve_navigation:
                self._shell.scientific.drop_viewer_loading_snapshot()
            cleared = self._context_controller.viewer_2d_frame is None
        else:
            try:
                receipt = (
                    self._shell.scientific.clear_viewer_2d(request, preserve_navigation=True)
                    if preserve_navigation
                    else self._shell.scientific.clear_viewer_2d(request)
                )
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
            updates["plot_options"] = replace(
                self._preferences.plot_options,
                intensity_scale="Log" if value else "Linear",
            )
        elif kind is ShellCommandKind.SET_DETECTOR_MODE:
            return self._set_detector_mode(str(value))
        elif kind is ShellCommandKind.SET_IMAGE_AXIS:
            if type(value) is not str or value not in {
                "Q-Chi",
                "2Th-Chi",
                "qip_qoop",
                "q_chi",
                "q_chi_derived",
                "exit_angles",
            }:
                return False
            # A display preference only: choosing which available 2-D map the
            # pane shows never edits the run intent or starts an integration.
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
            if value != prior_mode:
                self._retire_batch_presentation()
            if not viewer_1d:
                ScatteringWorkspace._clear_presentation_targets(self)
            updates["plot_mode"] = value
            if value not in {"Overlay", "Waterfall"}:
                updates["slice_pins"] = ()
            if value == "Single" and prior_mode != "Single" and not viewer_1d:
                self._shell.browser.cancel_pending_frame_selection()
                current = self._context_controller.navigation.current
                if current is not None:
                    self._context_controller.select_navigation(
                        current, (current,),
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
            self._processed_browser.set_date_sorted(requested)
            return True
        elif kind is ShellCommandKind.SET_AUTO_LAST:
            requested = bool(value)
            if self._processed_browser.set_auto_last(requested):
                self._retire_batch_presentation()
            ScatteringWorkspace._clear_presentation_targets(self)
            if self._processed_browser.auto_last:
                self._context_controller.select_latest_navigation(
                    plot_mode=self._preferences.plot_mode
                )
            return True
        elif kind is ShellCommandKind.CLEAR_1D:
            self._retire_batch_presentation()
            self._retain_outgoing_display = False
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
        self._release_native_plot_axis_transition_for_retry()
        if isinstance(outcome, (StartRefused, StartFailed)):
            self._notice(outcome.detail or outcome.reason.value)
        else:
            self._notice("Standard run was not started.")
        self._retry_deferred_gi_motor_default()
        self._refresh_shell(
            preserve_display=self._retain_outgoing_display,
        )

    def closeEvent(self, event: QtCore.QEvent) -> None:
        receipt = self.close_workspace()
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            event.accept()
        else:
            event.ignore()

    def _schedule_deferred_delete_retry(self) -> None:
        if (
            self._deferred_delete_pending
            and not self._deferred_delete_reposted
            and not self._deferred_delete_retry_timer.isActive()
        ):
            self._deferred_delete_retry_timer.start()

    def _retry_deferred_delete(self) -> None:
        if (
            not self._deferred_delete_pending
            or self._deferred_delete_reposted
        ):
            return
        receipt = (
            self._terminal_close
            if self._closed
            else self.close_workspace()
        )
        if (
            receipt is None
            or receipt.cleanup_status is not CleanupStatus.CLEANED
        ):
            self._schedule_deferred_delete_retry()
            return
        self._deferred_delete_pending = False
        self._deferred_delete_reposted = True
        QtCore.QCoreApplication.postEvent(
            self, QtCore.QEvent(QtCore.QEvent.Type.DeferredDelete),
        )

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DeferredDelete:
            if self._deferred_delete_pending:
                self._schedule_deferred_delete_retry()
                return True
            receipt = self.close_workspace()
            if receipt.cleanup_status is not CleanupStatus.CLEANED:
                self._deferred_delete_pending = True
                self._schedule_deferred_delete_retry()
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
        snapshot = self._intents.snapshot()
        reduced = reduce_control_edit(
            snapshot, path, value,  # type: ignore[arg-type]
            observation=self._source_selection.project_observation(snapshot),
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
        unsafe_unfunded_staging = False
        if candidate.processing_mode == "Int 1D (XYE)":
            _drop_nexus_only_performance_options(candidate)
        else:
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
        xye_only = candidate.processing_mode == "Int 1D (XYE)"
        notice = (
            "Performance diagnostics applied: plot cadence is active now."
            if xye_only
            else "Performance diagnostics applied: pipeline and output values "
            "take effect on the next run; plot cadence is active now."
        )
        if not xye_only and not values.durable_fsync:
            notice += (
                " Diagnostic fsync is off: crash or power-loss persistence "
                "is not guaranteed."
            )
        if values.quartile_telemetry:
            notice += (
                " Within-run quartile timing is enabled for the next run."
            )
        if not xye_only and values.staging_frame_cap > 64:
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

    def _automatic_motor_permitted(self) -> bool:
        """Return whether an observation may revise the next-run intent."""

        return (
            self._admission is None
            and self._lifecycle.phase is RunPhase.IDLE
        )

    def _retry_deferred_gi_motor_default(self) -> SourceRefreshEffect:
        transition = self._source_selection.commit_deferred_motor_default(
            permitted=self._automatic_motor_permitted(),
        )
        return self._compose_source_transition(transition)

    def _record_native_plot_axis_edit(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
    ) -> None:
        """Remember an axis-following edit without touching outgoing paint."""

        prior_axis = _native_plot_axis_for_run(prior.thaw())
        target_axis = _native_plot_axis_for_run(current.thaw())
        if prior_axis is None or target_axis is None:
            self._native_plot_axis_transition = None
            return
        pending = self._native_plot_axis_transition
        if (
            pending is not None
            and pending.run_identity is None
            and pending.target_axis == prior_axis
        ):
            origin_axis = pending.origin_axis
            followed_origin = pending.followed_origin
        else:
            origin_axis = prior_axis
            followed_origin = self._preferences.plot_axis == prior_axis
        self._native_plot_axis_transition = (
            None
            if target_axis == origin_axis
            else _NativePlotAxisTransition(
                origin_axis,
                target_axis,
                followed_origin,
            )
        )

    def _bind_native_plot_axis_to_run(
        self,
        identity: RunIdentity,
        configuration: FrozenRunConfiguration,
    ) -> None:
        """Bind the pending axis-following receipt to one launched run."""

        pending = self._native_plot_axis_transition
        target_axis = _native_plot_axis_for_run(configuration)
        self._native_plot_axis_transition = (
            replace(pending, run_identity=identity)
            if (
                pending is not None
                and pending.run_identity is None
                and target_axis == pending.target_axis
            )
            else None
        )

    def _retire_native_plot_axis_transition(
        self,
        identity: RunIdentity | None = None,
    ) -> None:
        pending = self._native_plot_axis_transition
        if (
            pending is not None
            and (identity is None or pending.run_identity is identity)
        ):
            self._native_plot_axis_transition = None

    def _release_native_plot_axis_transition_for_retry(
        self,
        identity: RunIdentity | None = None,
    ) -> None:
        """Unbind an unpainted compatible transition for the next attempt."""

        pending = self._native_plot_axis_transition
        if (
            pending is None
            or identity is not None
            and pending.run_identity is not identity
        ):
            return
        current_axis = _native_plot_axis_for_run(
            self._intents.snapshot().thaw()
        )
        self._native_plot_axis_transition = (
            replace(pending, run_identity=None)
            if current_axis == pending.target_axis
            else None
        )

    def _consume_native_plot_axis_transition(
        self,
        identity: RunIdentity,
    ) -> bool:
        """Adopt one exact run's native axis at its first accepted paint."""

        pending = self._native_plot_axis_transition
        if pending is None or pending.run_identity is not identity:
            return False
        self._native_plot_axis_transition = None
        preferences = self._preferences
        if (
            not pending.followed_origin
            or preferences.share_axis
            or preferences.plot_axis != pending.origin_axis
        ):
            return False
        self._preferences = replace(
            preferences,
            plot_axis=pending.target_axis,
            slice_pins=_slice_pins_for_plot_axis(
                preferences.slice_pins,
                preferences.plot_axis,
                pending.target_axis,
            ),
        )
        return True

    def _on_field_value(
        self,
        path: object,
        value: object,
    ) -> None:
        if self._closing or self._closed:
            return
        if path == PROJECT_ROOT and not self._project_folder_edit_permitted():
            self._notice("Project Folder is locked during Run or cleanup.")
            self._refresh_shell()
            return
        if self._source_selection.owns_edit(path):
            self._apply_source_observation_transition(
                self._source_selection.edit(path, value)
            )
            return
        snapshot = self._intents.snapshot()
        reduced = reduce_control_edit(
            snapshot,
            path,  # type: ignore[arg-type]
            value,
            observation=self._source_selection.project_observation(snapshot),
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
            self._reconcile_snapshot(
                snapshot,
                result.snapshot,
            )
        elif type(result) is IntentRecaptureRequired:
            self._notice("Edit superseded; review current value.")
            self._reconcile_snapshot(snapshot, result.snapshot)

    def _request_observation(
        self, snapshot: RunIntentSnapshot
    ) -> None:
        transition = self._source_selection.request_observation(snapshot)
        self._apply_source_observation_transition(transition)

    def _queue_live_source_refresh(
        self, event: StandardRunEvent
    ) -> None:
        """Coalesce one Live discovery into one passive source observation."""
        self._source_selection.queue_live_refresh(
            event.run_identity,
            self._intents.snapshot(),
            active_identity=self._lifecycle.active_run_identity,
            phase=self._lifecycle.phase,
        )

    def _clear_live_source_refresh(self) -> None:
        self._source_selection.clear_live_refresh()

    @staticmethod
    def _deliver_observation(
        page_ref: weakref.ReferenceType["ScatteringWorkspace"],
        wake: SourceObservationWake,
    ) -> None:
        page = page_ref()
        if page is None:
            return
        try:
            if page._closing or page._closed:
                return
            page._observationFinished.emit(wake)
        except RuntimeError:
            return

    def _on_observation(
        self, wake: object
    ) -> None:
        if self._closing or self._closed:
            return
        transition = self._source_selection.consume(
            wake,
            active_identity=self._lifecycle.active_run_identity,
            phase=self._lifecycle.phase,
            automatic_motor_permitted=self._automatic_motor_permitted(),
            terminal_progress=self._progress.terminal,
        )
        self._apply_source_observation_transition(transition)

    def _compose_source_transition(
        self,
        transition: SourceSelectionTransition,
    ) -> SourceRefreshEffect:
        if type(transition) is not SourceSelectionTransition:
            return SourceRefreshEffect.NONE
        if transition.notice is not None:
            self._notice(transition.notice)
        source = self._intents.snapshot().thaw().source_spec
        status = transition.status
        if status is SourceStatusDirective.NO_SOURCE:
            self._source_status.show_no_source()
        elif status is SourceStatusDirective.CHECKING and source is not None:
            self._source_status.show_checking(_source_label(source))
        elif status is SourceStatusDirective.UNAVAILABLE and source is not None:
            self._source_status.show_unavailable(_source_label(source))

        if transition.reset_terminal_progress:
            self._progress = replace(
                self._progress,
                detail="",
                directory_files=None,
                terminal=False,
            )
        effect = transition.refresh
        receipt = transition.intent
        if receipt is not None:
            nested = self._apply_snapshot_effects(
                receipt.before,
                receipt.result.snapshot,
                preserve_terminal=transition.preserve_terminal,
            )
            nested_effect = self._compose_source_transition(nested)
            if nested_effect is SourceRefreshEffect.CONTROLS:
                effect = SourceRefreshEffect.CONTROLS

        observation = transition.observation
        if (
            status is SourceStatusDirective.OBSERVED
            and type(observation) is SourceObservation
            and self._source_selection.observation is observation
        ):
            self._source_status.render(observation)
        return effect

    def _apply_source_observation_transition(
        self,
        transition: SourceSelectionTransition,
    ) -> None:
        effect = self._compose_source_transition(transition)
        if effect is SourceRefreshEffect.CONTROLS:
            self._refresh_shell(
                preserve_display=True,
                preserve_scientific=True,
            )

    def _apply_snapshot_effects(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
        *,
        preserve_terminal: bool = False,
    ) -> SourceSelectionTransition:
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
        if (
            _native_plot_axis_for_run(prior_intent)
            != _native_plot_axis_for_run(current_intent)
        ):
            self._record_native_plot_axis_edit(prior, current)
        pending_axis = self._native_plot_axis_transition
        if (
            pending_axis is not None
            and pending_axis.run_identity is None
            and _native_plot_axis_for_run(current_intent)
            != pending_axis.target_axis
        ):
            self._native_plot_axis_transition = None
        source_transition = self._source_selection.reconcile_source(
            prior,
            current,
        )
        self._processed_browser.reconcile_intent(prior, current)
        ScatteringWorkspace._observe_operation_stamp(
            self,
            current.revision,
        )
        return source_transition

    def _reconcile_snapshot(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
        *,
        preserve_terminal: bool = False,
    ) -> None:
        source_transition = self._apply_snapshot_effects(
            prior,
            current,
            preserve_terminal=preserve_terminal,
        )
        self._compose_source_transition(source_transition)
        default_transition = (
            self._source_selection.commit_deferred_motor_default(
                permitted=self._automatic_motor_permitted(),
            )
        )
        self._compose_source_transition(default_transition)
        self._refresh_shell()

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

    def _choose_authoring_source_dialog(
        self, asset: str, start_directory: str,
    ) -> str | None:
        if asset == "poni":
            title = "Choose calibration source image"
            file_filter = (
                "Calibration sources (*.tif *.tiff *.h5 *.hdf5 *.nxs "
                "*.nexus *.edf *.cbf *.img *.mar3450 *.raw);;All files (*)"
            )
        elif asset == "mask":
            title = "Choose TIFF or HDF5/NeXus source for mask"
            file_filter = (
                "Mask sources (*.tif *.tiff *.h5 *.hdf5 *.nxs *.nexus)"
            )
        else:
            return None
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start_directory, file_filter,
        )
        return selected or None

    def _choose_control_path(self, path: object) -> None:
        if type(path) is not tuple or not all(
            type(part) is str for part in path
        ):
            self._notice("Unknown control.")
            return
        if path == PROJECT_ROOT and not self._project_folder_edit_permitted():
            self._notice("Project Folder is locked during Run or cleanup.")
            self._refresh_shell()
            return
        if path in {SOURCE_FILE, SOURCE_DIRECTORY}:
            snapshot = self._intents.snapshot()
            state = self._project_controls(snapshot)
            source_type = next(
                (
                    str(candidate.value)
                    for candidate in state.fields
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
        field = next(
            (
                candidate
                for candidate in state.fields
                if candidate.path == path
            ),
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
        requested_mode = desired_mode or self._source_selection.mode
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

    def _choose_browser_directory(self) -> None:
        project_root = self._intents.snapshot().thaw().project_root
        directory = self._processed_browser.directory
        start_directory = browse_start_dir(
            directory,
            fallback=project_root,
        )
        try:
            selected = self._browser_directory_chooser(
                directory,
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
        self._apply_processed_browser_transition(
            self._processed_browser.set_directory(
                selected,
                explicit=explicit,
            )
        )

    def _follow_processed_artifact(
        self,
        frame: DisplayFrameKey,
    ) -> None:
        self._apply_processed_browser_transition(
            self._processed_browser.follow_processed_artifact(frame)
        )

    def _request_browser_catalog(self) -> BrowserCatalogRequest | None:
        return self._processed_browser.request_catalog()

    def _poll_browser_catalog(self) -> None:
        """Observe idle external deletion/recreation without GUI-thread I/O."""
        self._apply_processed_browser_transition(
            self._processed_browser.poll_catalog()
        )

    @staticmethod
    def _deliver_browser_catalog(
        page_ref: weakref.ReferenceType["ScatteringWorkspace"],
        wake: BrowserCatalogWake,
    ) -> None:
        page = page_ref()
        if page is None:
            return
        try:
            if page._closing or page._closed:
                return
            page._browserCatalogFinished.emit(wake)
        except RuntimeError:
            return

    def _on_browser_catalog(self, wake: object) -> None:
        self._apply_processed_browser_transition(
            self._processed_browser.consume_catalog(wake)
        )

    def _apply_processed_browser_transition(
        self,
        transition: ProcessedBrowserTransition,
    ) -> None:
        if type(transition) is not ProcessedBrowserTransition:
            return
        if transition.notice is not None:
            self._notice(transition.notice)
        if transition.refresh is BrowserRefreshEffect.FULL:
            self._refresh_shell()
        elif transition.refresh is BrowserRefreshEffect.CATALOG:
            self._refresh_shell(
                preserve_scientific=True,
                skip_scientific_projection=True,
                suppress_detector_demand=True,
            )

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
        _LOG.exception(prefix)
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
