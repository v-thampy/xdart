"""Public composition root for the context-qualified scattering workspace."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Callable
import weakref

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.modules.display_context import BrowseContext, ContextKind, DisplaySelection
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.gi_motor import pick_default_gi_motor
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
from xrd_tools.reduction import ReintegrateResult
from xrd_tools.reduction.provenance_config import jsonable_run_value
from xrd_tools.io.viewer_1d import SUPPORTED_VIEWER_1D_SUFFIXES
from xrd_tools.io.viewer_2d import SUPPORTED_VIEWER_SUFFIXES
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    stream_terminal_object_revision,
)
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
    BrowseLoadTiming,
)
from .contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    RunExecutorPort,
    SourceCountScope,
    SourceFileState,
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
    AssetValidationRequest, AssetValidationResult, AuthoredAssetCandidate,
    CalibrationRequest, CalibrationResult, MaskProof, MaskRequest, MaskResult,
    authoring_source_context_current,
    mask_terminal_result_valid,
    prepare_calibration_request, prepare_mask_request,
    resolve_calibration_executable,
    resolve_mask_executable,
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
    OperationContextStamp, OperationIdentity, OperationPending,
    OperationTerminalStatus, OperationUpdate,
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


_LOG = logging.getLogger(__name__)
_NO_DELIBERATE_MANUAL = object()
_NO_AUTOMATIC_GI_MOTOR = object()
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
class _BrowserCatalogRequest:
    token: int
    directory: str
    accepted_suffixes: frozenset[str] | None


@dataclass(frozen=True, slots=True)
class _BrowserCatalogOperation:
    request: _BrowserCatalogRequest
    cancelled: threading.Event
    future: Future[object]
    @property
    def token(self) -> int:
        return self.request.token
    @property
    def directory(self) -> str:
        return self.request.directory
    @property
    def accepted_suffixes(self) -> frozenset[str] | None:
        return self.request.accepted_suffixes


@dataclass(frozen=True, slots=True)
class _DeferredMetadata:
    plan: object
    request: tuple[object, ...]
    target: str
    generation: int
    dialog: object
    candidate: object | None = None


class _OperationRefresh(Enum):
    NONE = "none"
    DIALOG = "dialog"
    CONTROLS = "controls"
    FULL = "full"

    def __bool__(self) -> bool:
        return self is not _OperationRefresh.NONE


@dataclass(frozen=True, slots=True)
class _TerminalBrowseHandoff:
    request: BrowseLoadRequest
    run_identity: RunIdentity
    artifact: str
    current_label: int | None
    selected_labels: tuple[int, ...]
    commit_identity: StreamTerminal | None = None


@dataclass(frozen=True, slots=True)
class _TerminalBrowsePresentation:
    request: BrowseLoadRequest
    context: object


@dataclass(slots=True)
class _TerminalBrowsePerf:
    request: BrowseLoadRequest
    started_at: float
    poll_adopt_count: int = 0
    poll_adopt_s: float = 0.0
    settle_s: float = 0.0
    presentation_s: float = 0.0
    worker: BrowseLoadTiming | None = None
    fallback_pending: bool = False


@dataclass(frozen=True, slots=True)
class _BatchTerminalPresentation:
    run_identity: RunIdentity
    frame: DisplayFrameKey | None
    awaiting_full_raw: bool = False
    full_request_attempted: bool = False
    painted: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.run_identity) is not RunIdentity
            or self.frame is not None
            and (
                type(self.frame) is not DisplayFrameKey
                or self.frame.run_identity is not self.run_identity
            )
            or type(self.awaiting_full_raw) is not bool
            or type(self.full_request_attempted) is not bool
            or type(self.painted) is not bool
            or self.frame is None
            and (
                self.awaiting_full_raw
                or self.full_request_attempted
                or self.painted
            )
            or self.awaiting_full_raw
            and not self.full_request_attempted
            or self.awaiting_full_raw
            and self.painted
        ):
            raise TypeError("Batch terminal presentation is invalid")


def _browser_suffixes_for_mode(mode: str) -> frozenset[str] | None:
    tool = tool_from_mode_text(mode)
    if tool is Tool.XYE_VIEWER:
        return SUPPORTED_VIEWER_1D_SUFFIXES
    if tool is Tool.IMAGE_VIEWER:
        return SUPPORTED_VIEWER_SUFFIXES
    return None


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


def _terminal_identity_for_target(
    value: object,
    target: str,
) -> StreamTerminal | None:
    """Return an exact writer seal only for its lexical output target."""

    if type(value) is not StreamTerminal or type(target) is not str or not target:
        return None
    if stream_terminal_object_revision(value) is None:
        return None
    normalized = os.path.normcase(os.path.abspath(os.path.expanduser(target)))
    return value if value.target == normalized else None


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


@dataclass(slots=True)
class _AuthoredAssetOwner:
    token: int
    asset: str
    stamp: OperationContextStamp
    source_directory: str
    source_request: CalibrationRequest | MaskRequest
    candidates: tuple[AuthoredAssetCandidate, ...]
    expected_shape: tuple[int, int] | None
    dialog: _AuthoredAssetDialog
    queued: bool = True
    validation_identity: OperationIdentity | None = None
    validation_request: AssetValidationRequest | None = None
    requested_path: str | None = None


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
        browse_clock: Callable[[], float] = time.monotonic,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not callable(browse_clock):
            raise TypeError("terminal Browse clock must be callable")
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
        self._browser_catalog_queued: _BrowserCatalogRequest | None = None
        self._browser_catalog_token = 0
        self._browser_catalog: tuple[BrowserCatalogEntry, ...] = ()
        self._browser_directory_time_cache = DirectoryModifiedCache()
        self._browser_follow_identity: RunIdentity | None = None
        self._browser_seen_artifacts: set[str] = set()
        self._browser_transient_frame: DisplayFrameKey | None = None
        self._browser_transient_clear_token: int | None = None
        self._terminal_browse_handoff: _TerminalBrowseHandoff | None = None
        self._terminal_browse_presentation: (
            _TerminalBrowsePresentation | None
        ) = None
        self._terminal_browse_perf: _TerminalBrowsePerf | None = None
        self._browse_clock = browse_clock
        self._terminal_rebind_artifacts: tuple[
            RunIdentity, str, str
        ] | None = None
        self._active_batch_mode = False
        self._run_frame_seen = False
        self._retain_outgoing_display = False
        self._batch_latest_frame: DisplayFrameKey | None = None
        self._batch_visible_progress: ProgressProjection | None = None
        self._batch_terminal_presentation: (
            _BatchTerminalPresentation | None
        ) = None
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
        self._viewer_file_chooser = (
            viewer_file_chooser if viewer_file_chooser is not None
            else self._choose_viewer_2d_dialog
        )
        self._viewer_1d_file_chooser = (
            viewer_file_chooser if viewer_file_chooser is not None
            else self._choose_viewer_1d_dialog)
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
        self._analysis_slot = OperationSlot()
        self._background_owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
        self._background_identity: OperationIdentity | None = None
        self._calibration_identity: OperationIdentity | None = None; self._calibration_revision: int | None = None
        self._calibration_stamp: OperationContextStamp | None = None
        self._calibration_request: object | None = None
        self._mask_identity: OperationIdentity | None = None; self._mask_revision: int | None = None
        self._mask_stamp: OperationContextStamp | None = None
        self._mask_request: object | None = None
        self._authored_asset_token = 0
        self._authored_asset_owner: _AuthoredAssetOwner | None = None
        self._asset_validation_identity: OperationIdentity | None = None
        self._reintegrate_identity: OperationIdentity | None = None
        self._reintegrate_request: object | None = None
        self._reintegrate_target: str | None = None; self._reintegrate_dimension: str | None = None
        self._pending_reintegrate_reload: tuple[
            BrowseLoadRequest, str, StreamTerminal | None,
        ] | None = None
        self._average_identity: OperationIdentity | None = None
        self._average_pending: OperationPending | None = None
        self._average_revision: int | None = None
        self._average_target: str | None = None
        self._average_entry: str | None = None
        self._metadata_dialog = self._scan_roi_dialog = None
        self._peak_dialog = self._phase_dialog = None
        self._metadata_generation = self._scan_roi_generation = 0
        self._peak_generation = self._phase_generation = 0
        self._analysis_identity = None; self._analysis_kind = None
        self._analysis_target = None
        self._analysis_generation = None; self._analysis_anchor = None
        self._analysis_fingerprint = ""; self._analysis_request = None
        self._analysis_candidate = None
        self._deferred_metadata: _DeferredMetadata | None = None
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

    def _experiment_operation_busy(self) -> bool:
        return (self._operation_slot.owned
                or self._authored_asset_owner is not None
                or self._pending_reintegrate_reload is not None)

    def _analysis_operation_busy(self) -> bool:
        return bool(
            self._analysis_slot.owned
            or self._analysis_identity is not None
        )

    def _mutating_operation_busy(self) -> bool:
        return (
            self._experiment_operation_busy()
            or self._analysis_operation_busy()
        )

    def _observe_operation_stamp(self, revision: int | None = None) -> None:
        slot = getattr(self, "_operation_slot", None)
        average = getattr(self, "_average_identity", None)
        if slot is not None and (
            average is not None and slot.current_identity is average
        ):
            if revision is None:
                revision = self._intents.snapshot().revision
            stamp = OperationContextStamp(revision)
            slot.observe_stamp(stamp)
        elif slot is not None:
            stamp = ScatteringWorkspace._operation_context_stamp(
                self, revision
            )
            slot.observe_stamp(stamp)

        analysis_slot = getattr(self, "_analysis_slot", None)
        analysis = getattr(self, "_analysis_identity", None)
        if (
            analysis_slot is not None
            and analysis is not None
            and analysis_slot.current_identity is analysis
        ):
            if getattr(self, "_analysis_kind", None) in {
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

    def _begin_operation(
        self, frozen: object, body: Callable[..., object]
    ) -> OperationIdentity | None:
        if self._authored_asset_owner is not None:
            return None
        identity = self._operation_slot._begin(
            frozen, self._operation_context_stamp(), body
        )
        if identity is not None:
            self._ensure_timer()
        return identity

    def _analysis_generation_for(self, kind: str) -> int:
        return (self._metadata_generation if kind in {
                    "metadata", "metadata_requalification"} else
                self._scan_roi_generation if kind in {"scan_roi", "scan_plot", "roi_preview", "roi_scan"} else
                self._peak_generation if kind == "peak" else self._phase_generation)

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

    def _set_metadata_dialog_status(self, target, message) -> None:
        dialog = getattr(self, f"_{target}_dialog", None)
        status = None if dialog is None else getattr(dialog, "status", None)
        if status is not None:
            status.setText(str(message or ""))

    def _finish_metadata_refresh(
        self, target, message=None,
    ) -> _OperationRefresh:
        if message is not None:
            self._set_metadata_dialog_status(target, message)
        # Metadata never locks editable fields or repaints science.  Its final
        # transition does, however, release the mutually-exclusive Run and
        # mutating-action affordances through one controls-only projection.
        return (
            _OperationRefresh.DIALOG
            if self._analysis_operation_busy()
            else _OperationRefresh.CONTROLS
        )

    def _begin_analysis(self, kind, plan, generation, *, target=None,
                        request=None, anchor=None, table=None, roi=None,
                        candidate=None):
        if (
            self._authored_asset_owner is not None
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
        names = {"metadata": "begin_metadata",
                 "metadata_requalification": "begin_metadata_requalification",
                 "roi_preview": "begin_roi_preview",
                 "roi_scan": "begin_roi_scan", "peak": "begin_peak_fit",
                 "phase": "begin_phase_fit"}
        identity = (self._analysis_slot.begin_scan_plot(plan, table, roi, stamp)
                    if kind == "scan_plot" else
                    getattr(self._analysis_slot, names[kind])(plan, stamp))
        if identity is None: return None
        self._analysis_kind, self._analysis_generation = kind, generation
        self._analysis_target, self._analysis_identity = target, identity
        self._analysis_anchor, self._analysis_fingerprint = anchor, fingerprint
        self._analysis_request = request; self._analysis_candidate = candidate
        self._ensure_timer()
        message = f"Running {kind.replace('_', ' ')}…"
        self._notice(message)
        if kind in {"metadata", "metadata_requalification"}:
            self._set_metadata_dialog_status(target, message)
        self._refresh_shell(
            preserve_display=True,
            preserve_scientific=True,
        )
        return identity

    def _submit_metadata(
        self, plan, generation, *, target="metadata", request=None,
        candidate=None,
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
        deferred = _DeferredMetadata(
            plan=plan,
            request=request,
            target=target,
            generation=generation,
            dialog=getattr(self, f"_{target}_dialog", None),
            candidate=candidate,
        )
        # Install the newest exact request before classifying it.  A transient
        # Browse/viewer cleanup or analysis-slot race must not make the first
        # click disappear, and a newer cross-target request always supersedes
        # an older one.
        self._deferred_metadata = deferred
        disposition = self._classify_deferred_metadata(deferred)
        if disposition == "transient":
            self._notice("Metadata queued…")
            self._ensure_timer()
            return None
        if disposition != "ready":
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
            return None
        kind = (
            "metadata_requalification" if candidate is not None else "metadata"
        )
        identity = self._begin_analysis(
            kind, plan, generation, target=target, request=request,
            candidate=candidate,
        )
        if identity is not None:
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
            return identity
        # begin_* owns the final CAS.  If another owner won between the
        # readiness check and that CAS, retain the exact request only for a
        # recognized transient blocker.
        if self._classify_deferred_metadata(deferred) != "transient":
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
        else:
            self._notice("Metadata queued…")
            self._ensure_timer()
        return None

    def _classify_deferred_metadata(self, deferred: _DeferredMetadata) -> str:
        kind = (
            "metadata_requalification"
            if deferred.candidate is not None else "metadata"
        )
        dialog = getattr(self, f"_{deferred.target}_dialog", None)
        if (
            self._closing
            or self._closed
            or dialog is not deferred.dialog
            or deferred.generation
            != self._analysis_generation_for(deferred.target)
            or deferred.request
            != self._current_analysis_request(kind, deferred.target)
        ):
            return "permanent"
        controller = self._context_controller
        if (
            self._analysis_slot.owned
            or self._authored_asset_owner is not None
            or controller.browse_pending
            or controller.viewer_1d_cleanup_pending
            or controller.viewer_2d_cleanup_pending
        ):
            return "transient"
        from .analysis_mount import analysis_start_allowed
        return "ready" if analysis_start_allowed(self) else "permanent"

    def _dispatch_deferred_metadata(self) -> _OperationRefresh:
        deferred = self._deferred_metadata
        if deferred is None:
            return _OperationRefresh.NONE
        disposition = self._classify_deferred_metadata(deferred)
        if disposition == "transient":
            return _OperationRefresh.NONE
        if disposition != "ready":
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
            return self._finish_metadata_refresh(
                deferred.target,
                "Metadata request is no longer current.",
            )
        kind = (
            "metadata_requalification"
            if deferred.candidate is not None else "metadata"
        )
        identity = self._begin_analysis(
            kind, deferred.plan, deferred.generation,
            target=deferred.target, request=deferred.request,
            candidate=deferred.candidate,
        )
        if identity is not None:
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
            return _OperationRefresh.DIALOG
        if self._classify_deferred_metadata(deferred) != "transient":
            if self._deferred_metadata is deferred:
                self._deferred_metadata = None
            return self._finish_metadata_refresh(
                deferred.target,
                "Metadata request is no longer current.",
            )
        self._ensure_timer()
        return _OperationRefresh.NONE

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
        generation = f"_{target}_generation"
        closing_generation = getattr(self, generation)
        setattr(self, generation, closing_generation + 1)
        if (
            self._deferred_metadata is not None
            and self._deferred_metadata.target == target
            and self._deferred_metadata.dialog is dialog
            and self._deferred_metadata.generation == closing_generation
        ):
            self._deferred_metadata = None
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
                from xdart.gui.tabs.static_scan.scan_plot_dialog import ScanPlotDialog
                dialog = ScanPlotDialog(parent=self, vnext_submit=self._scan_analysis_action)
            elif target == "peak":
                from xdart.gui.tabs.static_scan.peak_fit_dialog import PeakFitDialog
                dialog = PeakFitDialog(parent=self, vnext_submit=self._peak_analysis_action)
            else:
                from xdart.gui.tabs.static_scan.phase_fit_dialog import PhaseFitDialog
                dialog = PhaseFitDialog(parent=self, vnext_submit=self._phase_analysis_action)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            setattr(self, field, dialog)
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
                    None, self._metadata_generation, target="metadata",
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
            self._metadata_generation,
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
                    plan, self._scan_roi_generation, target="scan_roi",
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

    def _consume_analysis_update(self, update):
        if type(update) is not OperationUpdate or update.identity is not self._analysis_identity:
            return _OperationRefresh.NONE
        if update.terminal is None:
            if update.progress is not None:
                message = f"Analysis: {update.progress.stage} {update.progress.completed}/{update.progress.total}"
                self._notice(message)
                if self._analysis_kind in {
                    "metadata", "metadata_requalification",
                }:
                    self._set_metadata_dialog_status(
                        self._analysis_target, message,
                    )
            return (
                _OperationRefresh.DIALOG
                if self._analysis_kind in {
                    "metadata", "metadata_requalification",
                }
                else _OperationRefresh.FULL
            )
        kind, target = self._analysis_kind, self._analysis_target
        metadata_operation = kind in {
            "metadata", "metadata_requalification",
        }
        request = self._analysis_request
        current = self._analysis_generation == self._analysis_generation_for(target)
        current = bool(current and request is not None
                       and request == self._current_analysis_request(kind, target))
        if kind in {"peak", "phase"}:
            from .analysis_mount import display_anchor_matches
            current = current and display_anchor_matches(
                self, self._analysis_anchor, self._analysis_fingerprint,
                self._analysis_generation)
        terminal_payload = update.terminal.payload
        from xrd_tools.analysis.scan_operations import AnalysisDisposition
        if (current and kind == "metadata" and target == "scan_roi"
                and getattr(terminal_payload, "disposition", None)
                    is AnalysisDisposition.REFUSED
                and getattr(terminal_payload, "code", "") == "SOURCE_SELECTION_REQUIRED"
                and self._scan_roi_dialog is not None):
            candidates = getattr(terminal_payload, "candidates", ())
            self._analysis_identity = self._analysis_kind = self._analysis_target = None
            self._analysis_generation = self._analysis_anchor = self._analysis_request = None
            self._analysis_fingerprint = ""; self._analysis_candidate = None
            self._roi_preview_binding = self._scan_roi_result = None
            self._scan_roi_dialog.clear_vnext_metadata()
            if candidates: self._scan_roi_dialog.source_widget.set_external_candidates(candidates)
            message = "Choose one headless-qualified source."
            self._notice(message)
            return self._finish_metadata_refresh(target, message)
        from .analysis_mount import (analysis_result_matches,
            retention_admission, terminal_adoption)
        payload, diagnostic = terminal_adoption(update, current=current)
        candidate = self._analysis_candidate
        requalification = kind == "metadata_requalification"
        if requalification:
            from xrd_tools.analysis.scan_operations import (
                MetadataTableRequalificationResult, MetadataTableResult,
            )
            valid = (
                type(payload) is MetadataTableRequalificationResult
                and type(candidate) is MetadataTableResult
                and payload.receipt == candidate.receipt
                and payload.table_fingerprint == candidate.table_fingerprint
                and analysis_result_matches(candidate, request)
            )
            if valid:
                payload = candidate
                kind = "metadata"
            else:
                payload, diagnostic = (
                    None, diagnostic or "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH"
                )
        elif payload is not None and not analysis_result_matches(payload, request):
            payload, diagnostic = None, "P3_7_ANALYSIS_RESULT_IDENTITY_MISMATCH"
        generation = self._analysis_generation
        self._analysis_identity = self._analysis_kind = self._analysis_target = None
        self._analysis_generation = self._analysis_anchor = self._analysis_request = None
        self._analysis_fingerprint = ""; self._analysis_candidate = None
        if payload is None:
            self._notice(diagnostic)
            return (
                self._finish_metadata_refresh(target, diagnostic)
                if metadata_operation else _OperationRefresh.FULL
            )
        if kind == "metadata" and self._deferred_metadata is not None:
            # A newer user request already owns latest-only precedence.  Do
            # not let this older terminal candidate erase it, either while
            # chaining or after completing the candidate's requalification.
            return _OperationRefresh.DIALOG
        if kind == "metadata" and not requalification:
            identity = self._submit_metadata(
                None, generation, target=target, request=request,
                candidate=payload,
            )
            if identity is None and self._deferred_metadata is None:
                return self._finish_metadata_refresh(target)
            return _OperationRefresh.DIALOG
        retained = {"metadata": self._metadata_result,
                    "scan_roi": self._scan_roi_result,
                    "peak": self._peak_result, "phase": self._phase_result}
        if kind == "metadata": retained["scan_roi"] = None
        replacing = "metadata" if kind == "metadata" else (
            "scan_roi" if kind in {"scan_plot", "roi_preview", "roi_scan"} else kind)
        admitted, reason = retention_admission(retained, replacing, payload)
        if not admitted:
            self._notice(reason)
            return (
                self._finish_metadata_refresh(target, reason)
                if metadata_operation else _OperationRefresh.FULL
            )
        if kind in {"scan_plot", "roi_preview", "roi_scan"}:
            self._roi_preview_binding = None
            if self._scan_roi_dialog is not None:
                self._scan_roi_dialog._retire_vnext_result()
        if kind == "metadata":
            self._roi_preview_binding = None
            self._metadata_result = payload; self._scan_roi_result = None
            if target == "metadata":
                if self._metadata_dialog is not None: self._metadata_dialog.adopt_result(payload)
                if self._scan_roi_dialog is not None: self._scan_roi_dialog.clear_vnext_metadata()
            elif self._scan_roi_dialog is not None:
                self._scan_roi_dialog.set_vnext_metadata(payload)
        elif kind == "scan_plot":
            self._scan_roi_result = payload
            if self._scan_roi_dialog is not None: self._scan_roi_dialog.set_vnext_scan_result(payload, request[2])
        elif kind == "roi_preview":
            self._scan_roi_result = payload
            if self._scan_roi_dialog is not None and payload.image is not None:
                from xdart.gui.tabs.static_scan.roi_select_dialog import RoiSelectDialog
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
        return (
            self._finish_metadata_refresh(target)
            if metadata_operation else _OperationRefresh.FULL
        )

    @staticmethod
    def _background_domain(mode: str) -> str | None:
        return {"Int 1D": "integrated_1d", "Int 2D": "integrated_2d", "1D Viewer": "integrated_1d", "2D Viewer": "raw"}.get(mode)

    def _background_key(self, plan, stamp, mode: str, target_facts) -> tuple[object, ...]:
        return (stamp.context_token, stamp.display_generation, plan.domain, mode,
                "raw" if plan.domain == "raw" else "integrated",
                plan.contributor_ids, plan.value_shapes, plan.axis_shapes, target_facts)

    def _background_action(self) -> None:
        owner, slot = self._background_owner, self._operation_slot
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
            and slot.current_identity is self._background_identity
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
            self._notice(str(error)); self._refresh_shell(); return
        remember_browse_path(request.source_path)
        stamp = self._operation_context_stamp(snapshot.revision)
        identity = slot.begin_calibrate(request, stamp)
        if identity is None:
            self._notice("Calibration operation was not started."); return
        self._calibration_identity, self._calibration_revision = identity, snapshot.revision
        self._calibration_stamp = stamp
        self._calibration_request = request
        self._notice(f"Calibrating from {os.path.basename(request.source_path)}…")
        self._refresh_shell(); self._ensure_timer()
    def _consume_calibration_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._calibration_identity: return False
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Calibration: {update.progress.stage}…")
            return True
        terminal, stamp, request = (
            update.terminal, self._calibration_stamp, self._calibration_request,
        )
        self._calibration_identity = self._calibration_revision = None
        self._calibration_stamp = self._calibration_request = None
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self._notice("Calibration cancelled." if terminal.status is OperationTerminalStatus.CANCELLED else f"Calibration failed: {terminal.diagnostic}")
            return True
        result = terminal.payload
        try:
            expected_argv = (
                (request.executable, request.source_path)
                if (type(request) is CalibrationRequest
                    and Path(request.source_path).suffix.casefold()
                    not in {".h5", ".hdf5", ".nxs", ".nexus"})
                else (request.executable, request.exact_hdf_url)
                if (type(request) is CalibrationRequest
                    and request.exact_hdf_url is not None)
                else (request.executable,)
                if type(request) is CalibrationRequest else ()
            )
            valid = (type(result) is CalibrationResult
                     and result.request is request and stamp is not None
                     and result.exit_code == 0
                     and result.argv == expected_argv
                     and result.diagnostic == "")
            if valid: result.__post_init__()
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            self._notice("Calibration returned without exact discovery proof."); return True
        if update.stale or self._operation_context_stamp() != stamp:
            self._notice("Calibration finished but context changed; PONI was not adopted.")
            return True
        if not authoring_source_context_current(request):
            self._notice("Calibration source context changed; PONI was not adopted.")
            return True
        candidates = tuple(AuthoredAssetCandidate(
            "poni", candidate.path, candidate.proof, candidate.proof.state,
        ) for candidate in result.candidates)
        self._queue_authored_asset_confirmation(
            "poni", stamp, result.request.monitored_directory,
            result.request, candidates,
            expected_shape=None,
        )
        self._notice(
            "Choose whether to adopt the authored PONI."
            if candidates else
            "No new valid PONI was found; choose an existing PONI or cancel."
        )
        return True

    def _mask_action(self) -> None:
        slot, identity = self._operation_slot, self._mask_identity
        if identity is not None and slot.current_identity is identity:
            accepted = slot.cancel(identity); self._notice("Cancelling mask…" if accepted else "Mask cancellation was not accepted.")
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
            self._notice(str(error)); self._refresh_shell(); return
        remember_browse_path(request.source_path); self._notice(f"Preparing {os.path.basename(request.source_path)}…")
        stamp = self._operation_context_stamp(snapshot.revision)
        identity = slot.begin_mask(request, stamp)
        if identity is None:
            self._notice("Mask operation was not started."); return
        self._mask_identity, self._mask_revision = identity, snapshot.revision
        self._mask_stamp = stamp
        self._mask_request = request
        self._notice(f"Making {os.path.basename(request.final_path)}…"); self._refresh_shell(); self._ensure_timer()
    def _consume_mask_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._mask_identity: return False
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Mask: {update.progress.stage}…")
            return True
        terminal, stamp, request = (
            update.terminal, self._mask_stamp, self._mask_request,
        )
        self._mask_identity = self._mask_revision = None
        self._mask_stamp = self._mask_request = None
        result = terminal.payload
        exact = (type(request) is MaskRequest
                 and mask_terminal_result_valid(terminal, request))
        generic_failure = (terminal.status is OperationTerminalStatus.FAILED
                           and result is None)
        if not exact and not generic_failure:
            self._notice("Mask returned without exact publication proof.")
            return True
        if terminal.status is not OperationTerminalStatus.RETURNED:
            self._notice("Mask cancelled." if terminal.status is OperationTerminalStatus.CANCELLED else f"Mask failed: {terminal.diagnostic}"); return True
        if update.stale or stamp is None or self._operation_context_stamp() != stamp:
            self._notice("Mask was published but context changed; mask was not adopted."); return True
        if not authoring_source_context_current(request):
            self._notice("Mask source context changed; mask was not adopted.")
            return True
        candidate = AuthoredAssetCandidate(
            "mask", result.request.final_path, result.proof, result.final_state,
            result.request.source_path,
        )
        self._queue_authored_asset_confirmation(
            "mask", stamp, str(Path(result.request.source_path).parent),
            result.request, (candidate,), expected_shape=result.proof.shape,
        )
        self._notice("Choose whether to adopt the generated mask.")
        return True

    def _queue_authored_asset_confirmation(
        self, asset: str, stamp: OperationContextStamp,
        source_directory: str,
        source_request: CalibrationRequest | MaskRequest,
        candidates: tuple[AuthoredAssetCandidate, ...], *,
        expected_shape: tuple[int, int] | None,
    ) -> None:
        if self._authored_asset_owner is not None:
            self._notice("Another authored-asset confirmation is pending.")
            return
        try:
            valid = (asset in {"poni", "mask"}
                     and type(stamp) is OperationContextStamp
                     and type(source_directory) is str
                     and os.path.isabs(source_directory)
                     and 0 < len(source_directory.encode("utf-8")) <= 16 << 10
                     and type(candidates) is tuple
                     and len(candidates) <= 256
                     and (asset == "poni") == (expected_shape is None)
                     and (expected_shape is None or type(expected_shape) is tuple
                          and len(expected_shape) == 2
                          and all(type(value) is int and value > 0
                                  for value in expected_shape))
                     and (asset == "poni" or len(candidates) == 1))
            valid = (valid
                     and type(source_request) in {CalibrationRequest, MaskRequest}
                     and (asset == "poni")
                     == (type(source_request) is CalibrationRequest)
                     and source_request.source_path
                     == source_request.source_state.path
                     and Path(source_request.source_path).parent
                     == Path(source_directory)
                     and authoring_source_context_current(source_request))
            if valid:
                stamp.__post_init__()
                for candidate in candidates:
                    if type(candidate) is not AuthoredAssetCandidate:
                        valid = False
                        break
                    candidate.__post_init__()
                    if (candidate.asset != asset
                            or Path(candidate.path).parent
                            != Path(source_directory)
                            or expected_shape is not None
                            and candidate.proof.shape != expected_shape):
                        valid = False
                        break
            if (valid and asset == "poni"
                    and sum(candidate.state.size for candidate in candidates)
                    > 16 << 20):
                valid = False
        except (AttributeError, TypeError, ValueError, OverflowError,
                UnicodeEncodeError):
            valid = False
        if not valid:
            self._notice("Authored asset confirmation proof is invalid.")
            return
        self._authored_asset_token += 1
        token = self._authored_asset_token
        dialog = _AuthoredAssetDialog(
            asset, tuple(candidate.path for candidate in candidates), self,
        )
        owner = _AuthoredAssetOwner(
            token, asset, stamp, source_directory, source_request, candidates,
            expected_shape, dialog,
        )
        self._authored_asset_owner = owner
        dialog.acceptRequested.connect(
            lambda path, t=token, d=dialog:
            self._accept_authored_asset(t, d, path)
        )
        dialog.chooseRequested.connect(
            lambda t=token, d=dialog:
            self._choose_another_authored_asset(t, d)
        )
        dialog.cancelRequested.connect(
            lambda t=token, d=dialog:
            self._cancel_authored_asset(t, d)
        )
        dialog.destroyed.connect(
            lambda _object=None, t=token, d=dialog:
            self._authored_asset_dialog_destroyed(t, d)
        )

    def _authored_asset_context_current(
        self, owner: _AuthoredAssetOwner,
    ) -> bool:
        return (not self._closing and not self._closed
                and self._authored_asset_owner is owner
                and self._operation_context_stamp() == owner.stamp
                and authoring_source_context_current(owner.source_request))

    def _show_queued_authored_asset_confirmation(self) -> None:
        owner = self._authored_asset_owner
        if owner is None or not owner.queued:
            return
        if not self._authored_asset_context_current(owner):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return
        owner.queued = False
        owner.dialog.open()
        (owner.dialog.accept_button
         if owner.dialog.accept_button.isEnabled()
         else owner.dialog.choose_button).setFocus(
            QtCore.Qt.FocusReason.OtherFocusReason,
        )
        owner.dialog.raise_()
        owner.dialog.activateWindow()

    def _retire_authored_asset(
        self, owner: _AuthoredAssetOwner, notice: str = "",
    ) -> None:
        if self._authored_asset_owner is not owner:
            return
        self._authored_asset_owner = None
        if self._asset_validation_identity is owner.validation_identity:
            self._asset_validation_identity = None
        owner.dialog.close_inert()
        if notice:
            self._notice(notice)
        if not self._closing and not self._closed:
            self._refresh_shell(preserve_scientific=True)
            self._ensure_timer()

    def _authored_asset_dialog_destroyed(
        self, token: int, dialog: _AuthoredAssetDialog,
    ) -> None:
        owner = self._authored_asset_owner
        if (owner is None or owner.token != token
                or owner.dialog is not dialog):
            return
        if owner.validation_identity is not None:
            self._operation_slot.cancel(owner.validation_identity)
        self._authored_asset_owner = None
        self._asset_validation_identity = None
        if not self._closing and not self._closed:
            self._notice("Authored asset was not adopted.")
            self._refresh_shell()
            self._ensure_timer()

    def _cancel_authored_asset(
        self, token: int, dialog: _AuthoredAssetDialog,
    ) -> None:
        owner = self._authored_asset_owner
        if (owner is None or owner.token != token
                or owner.dialog is not dialog
                or owner.validation_identity is not None):
            return
        label = "PONI" if owner.asset == "poni" else "mask"
        self._retire_authored_asset(owner, f"{label} was not adopted.")

    def _accept_authored_asset(
        self, token: int, dialog: _AuthoredAssetDialog, path: str,
    ) -> None:
        owner = self._authored_asset_owner
        if (owner is None or owner.token != token
                or owner.dialog is not dialog
                or owner.validation_identity is not None):
            return
        if not self._authored_asset_context_current(owner):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return
        candidate = next(
            (item for item in owner.candidates if item.path == path), None,
        )
        if candidate is None:
            self._notice("The selected authored asset is unavailable.")
            return
        self._begin_authored_asset_validation(
            owner, AssetValidationRequest(
                owner.asset, candidate.path, owner.expected_shape, candidate,
                owner.source_request,
            ),
        )

    def _choose_another_authored_asset(
        self, token: int, dialog: _AuthoredAssetDialog,
    ) -> None:
        owner = self._authored_asset_owner
        if (owner is None or owner.token != token
                or owner.dialog is not dialog
                or owner.validation_identity is not None):
            return
        if not self._authored_asset_context_current(owner):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return
        chooser = self._control_path_chooser
        control = PONI_FILE if owner.asset == "poni" else MASK_FILE
        snapshot = self._intents.snapshot(); intent = snapshot.thaw()
        current_value = (intent.poni_file if owner.asset == "poni"
                         else intent.mask_file)
        current = "" if current_value is None else str(current_value)
        try:
            selected = (chooser(control, current, owner.source_directory)
                        if chooser is not None else
                        QtWidgets.QFileDialog.getOpenFileName(
                            self,
                            "Choose existing PONI" if owner.asset == "poni"
                            else "Choose existing detector mask",
                            owner.source_directory,
                            "PONI files (*.poni);;All files (*)"
                            if owner.asset == "poni" else
                            "Detector masks (*.edf *.tif *.tiff *.npy);;All files (*)",
                        )[0])
        except Exception as error:
            self._error_notice("Asset chooser failed", error); return
        if type(selected) is not str or not selected:
            return
        if (not os.path.isabs(selected)
                or not self._authored_asset_context_current(owner)):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return
        self._begin_authored_asset_validation(
            owner, AssetValidationRequest(
                owner.asset, selected, owner.expected_shape,
                source_request=owner.source_request,
            ),
        )

    def _begin_authored_asset_validation(
        self, owner: _AuthoredAssetOwner,
        request: AssetValidationRequest,
    ) -> None:
        if (not self._authored_asset_context_current(owner)
                or self._operation_slot.owned):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return
        identity = self._operation_slot.begin_asset_validation(
            request, owner.stamp,
        )
        if identity is None:
            self._notice("Asset validation operation was not started.")
            return
        owner.validation_identity = identity
        owner.validation_request = request
        owner.requested_path = request.path
        self._asset_validation_identity = identity
        owner.dialog.set_busy(True)
        self._notice("Validating the selected authored asset…")
        self._refresh_shell(); self._ensure_timer()

    @staticmethod
    def _authored_candidate_state_current(
        candidate: AuthoredAssetCandidate,
    ) -> bool:
        try:
            if type(candidate) is not AuthoredAssetCandidate:
                return False
            candidate.__post_init__()
            path = Path(candidate.path)
            raw = path.lstat()
            state = candidate.state
            return (not stat.S_ISLNK(raw.st_mode)
                    and stat.S_ISREG(raw.st_mode)
                    and (raw.st_dev, raw.st_ino, raw.st_size,
                         raw.st_mtime_ns, raw.st_ctime_ns)
                    == (state.device, state.inode, state.size,
                        state.mtime_ns, state.ctime_ns)
                    and SourceFileState.capture(path) == state)
        except (OSError, ValueError):
            return False

    def _consume_asset_validation_update(self, update: object) -> bool:
        if (type(update) is not OperationUpdate
                or update.identity is not self._asset_validation_identity):
            return False
        owner = self._authored_asset_owner
        if update.terminal is None:
            if update.progress is not None:
                self._notice("Validating the selected authored asset…")
            return True
        self._asset_validation_identity = None
        if owner is None or owner.validation_identity is not update.identity:
            return True
        owner.validation_identity = None
        terminal = update.terminal
        if (update.stale or not self._authored_asset_context_current(owner)
                or terminal.status is OperationTerminalStatus.CANCELLED):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return True
        if terminal.status is not OperationTerminalStatus.RETURNED:
            owner.requested_path = None
            owner.validation_request = None
            owner.dialog.set_busy(False)
            self._notice(f"Asset validation failed: {terminal.diagnostic}")
            return True
        result = terminal.payload
        try:
            valid = (type(result) is AssetValidationResult
                     and result.request is owner.validation_request
                     and result.request.path == owner.requested_path
                     and result.candidate.asset == owner.asset
                     and result.candidate.path == owner.requested_path)
            if valid:
                terminal.__post_init__()
                result.__post_init__()
                valid = self._authored_candidate_state_current(
                    result.candidate,
                )
        except (AttributeError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            owner.requested_path = None
            owner.validation_request = None
            owner.dialog.set_busy(False)
            self._notice("Asset validation returned an inexact result.")
            return True
        if not self._authored_asset_context_current(owner):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return True
        snapshot = self._intents.snapshot()
        control = PONI_FILE if owner.asset == "poni" else MASK_FILE
        reduced = reduce_control_edit(snapshot, control, result.candidate.path)
        if isinstance(reduced, EditRefusal):
            self._retire_authored_asset(
                owner, "The selected authored asset could not be adopted.",
            )
            return True
        if isinstance(reduced, EditNoChange):
            path = result.candidate.path
            self._retire_authored_asset(owner, f"Already selected: {path}")
            return True
        if not self._authored_asset_context_current(owner):
            self._retire_authored_asset(
                owner, "Authored asset context changed; nothing was adopted.",
            )
            return True
        if not self._authored_candidate_state_current(result.candidate):
            self._retire_authored_asset(
                owner, "Authored asset changed before adoption.",
            )
            return True
        try:
            committed = self._intents.commit(
                reduced, expected_revision=owner.stamp.intent_revision,
            )
        except Exception as error:
            self._retire_authored_asset(owner)
            self._error_notice("Authored asset adoption failed", error)
            return True
        path = result.candidate.path
        self._retire_authored_asset(owner)
        if type(committed) is IntentCommitAccepted:
            self._reconcile_snapshot(snapshot, committed.snapshot)
            remember_browse_path(path)
            self._notice(f"{'PONI' if control == PONI_FILE else 'Mask'} adopted: {path}")
        else:
            self._notice("Authored asset adoption was superseded.")
        return True

    @staticmethod
    def _reintegrate_preparation(intent, dimension) -> dict[str, object]:
        bai = jsonable_run_value(getattr(intent, f"bai_{dimension}_args"), path="reintegrate.selected_plan.bai_args") if type(dimension) is str and dimension in {"1d", "2d"} else (_ for _ in ()).throw(ValueError("Reintegrate dimension is unsupported")); (None if type(bai) is dict else (_ for _ in ()).throw(ValueError("current integration settings are malformed")))
        bai.pop(f"gi_mode_{dimension}", None); mode = getattr(intent.gi, f"mode_{dimension}"); workers = intent.max_cores
        if type(mode) is not str or mode not in ({"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"} if dimension == "1d" else {"qip_qoop", "q_chi", "exit_angles"}): raise ValueError("current integration mode is unsupported")
        if type(workers) is not int or workers < 1: raise ValueError("current core request is invalid")
        return {"api_version": 1,
            "selected_plan": {"version": 1, "dimension": dimension, "bai_args": bai, "gi_mode": mode},
            "requested_shared_science": {"version": 1, "kind": "persisted_target"},
            "resource_policy": {"version": 1, "kind": "resolve", "envelope_bytes": None,
                                "requests": {"workers": workers}}}

    def _reload_after_reintegrate(
        self,
        request,
        target,
        terminal_commit_identity: StreamTerminal | None = None,
    ) -> None:
        if type(request) is not BrowseLoadRequest or type(target) is not str:
            self._request_browser_catalog()
            return
        self._pending_reintegrate_reload = (
            request, target, terminal_commit_identity,
        )
        self._retry_pending_reintegrate_reload()

    def _retry_pending_reintegrate_reload(self) -> bool:
        pending = self._pending_reintegrate_reload
        if pending is None:
            return False
        if self._closing or self._closed:
            self._pending_reintegrate_reload = None
            return False
        if not self._release_browse_1d_debt():
            self._ensure_timer()
            return False
        request, target, terminal_commit_identity = pending
        reloaded = (
            self._context_controller.reload_reintegrate_browse(
                request, target,
            )
            if terminal_commit_identity is None
            else self._context_controller.reload_reintegrate_browse(
                request,
                target,
                terminal_commit_identity=terminal_commit_identity,
            )
        )
        if reloaded is None:
            if self._context_controller.reintegrate_reload_retryable(
                request, target,
            ):
                self._ensure_timer()
                return False
            self._pending_reintegrate_reload = None
            self._request_browser_catalog()
            self._notice(
                "Reintegrate Browse reload lost its exact context; "
                "refresh the persisted artifact from the browser."
            )
            return True
        self._pending_reintegrate_reload = None
        return True

    def _reintegrate_action(self, dimension) -> None:
        slot, active = self._operation_slot, self._reintegrate_identity
        if active is not None and slot.current_identity is active and self._reintegrate_dimension != dimension: self._refresh_shell(); return
        if active is not None and slot.current_identity is active and self._reintegrate_dimension == dimension: accepted = slot.cancel(active); self._notice(f"Cancelling Reintegrate {dimension[0]}-D…" if accepted else "Reintegrate cancellation was not accepted."); self._refresh_shell(); return
        if self._authored_asset_owner is not None:
            self._notice("Reintegrate is unavailable while authored-asset confirmation is pending.")
            self._refresh_shell(); return
        if not self._commit_focused_control_edit_for_run(): return
        snapshot = self._intents.snapshot()
        if snapshot.thaw().processing_mode == "Int 1D (XYE)":
            self._notice(
                f"Reintegrate {dimension[0]}-D is unavailable for XYE-only output."
            )
            self._refresh_shell()
            return
        phase = self._lifecycle.phase; permitted = phase is RunPhase.IDLE or phase is RunPhase.FAILED and self._lifecycle.reset_permitted
        captured = self._context_controller.capture_reintegrate_browse()
        if self._closing or self._closed or self._admission_state is not None or self._mutating_operation_busy() or not permitted or captured is None:
            self._notice(f"Reintegrate {dimension[0]}-D requires one stable loaded Browse context."); self._refresh_shell(); return
        try: preparation = self._reintegrate_preparation(snapshot.thaw(), dimension)
        except (TypeError, ValueError) as error: self._notice(str(error)); self._refresh_shell(); return
        stamp = self._operation_context_stamp(snapshot.revision); recaptured = self._context_controller.capture_reintegrate_browse(); current = self._intents.snapshot()
        try: current_preparation = self._reintegrate_preparation(current.thaw(), dimension)
        except (TypeError, ValueError): current_preparation = None
        same_browse = (recaptured is not None and recaptured[0] is captured[0] and recaptured[1] is captured[1] and recaptured[2] is captured[2] and recaptured[3:] == captured[3:])
        if current.revision != snapshot.revision or current_preparation != preparation or not same_browse or not self._context_controller.invalidate_reintegrate_browse(*captured):
            self._notice("Reintegrate context changed before dispatch."); self._refresh_shell(); return
        context, request, _selection, target, entry, target_snapshot, labels = captured
        identity = slot.begin_reintegrate(target=target, entry=entry,
            expected_target_snapshot=target_snapshot, expected_labels=labels,
            dimension=dimension, preparation_values=preparation, stamp=stamp,
            expected_terminal_identity=(
                request.terminal_commit_identity
                if stream_terminal_object_revision(
                    request.terminal_commit_identity,
                ) is not None
                else None
            ))
        if identity is None:
            self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = self._reintegrate_dimension = None; self._reload_after_reintegrate(request, target); self._notice(f"Reintegrate {dimension[0]}-D was not started; Browse is reloading."); self._refresh_shell(); return
        self._reintegrate_identity, self._reintegrate_request, self._reintegrate_target, self._reintegrate_dimension = identity, request, target, dimension
        self._notice(f"Reintegrating {dimension[0]}-D from authenticated loaded artifact science…"); self._refresh_shell(); self._ensure_timer()

    def _consume_reintegrate_update(self, update: object) -> _OperationRefresh:
        if type(update) is not OperationUpdate or update.identity is not self._reintegrate_identity: return _OperationRefresh.NONE
        if update.terminal is None:
            if update.progress is not None: self._notice(f"Reintegrate {self._reintegrate_dimension[0]}-D: {update.progress.stage} {update.progress.completed}/{update.progress.total}…")
            return _OperationRefresh.CONTROLS
        terminal, request, target = update.terminal, self._reintegrate_request, self._reintegrate_target
        shown = {"1d": "1-D", "2d": "2-D"}.get(self._reintegrate_dimension, "operation"); self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = self._reintegrate_dimension = None
        result = terminal.payload
        self._notice(f"Reintegrate {shown} {result.disposition.lower()}; reloading persisted results." if terminal.status is OperationTerminalStatus.RETURNED and type(result) is ReintegrateResult else f"Reintegrate {shown} cancelled; reloading persisted results." if terminal.status is OperationTerminalStatus.CANCELLED else f"Reintegrate {shown} failed: {terminal.diagnostic}")
        committed = (
            terminal.status is OperationTerminalStatus.RETURNED
            and type(result) is ReintegrateResult
            and result.disposition == "COMMITTED"
        )
        commit_identity = (
            _terminal_identity_for_target(result.commit_identity, target)
            if committed and target is not None
            else None
        )
        if committed and commit_identity is None:
            self._notice(
                f"Reintegrate {shown} returned an invalid commit identity; "
                "reloading the persisted artifact without its terminal seal."
            )
            if request is not None and target is not None:
                self._reload_after_reintegrate(request, target)
            else:
                self._request_browser_catalog()
            return _OperationRefresh.FULL
        if request is not None and target is not None: self._reload_after_reintegrate(
            request, target, commit_identity,
        )
        else: self._request_browser_catalog()
        return _OperationRefresh.FULL

    def _consume_average_update(self, update: object) -> bool:
        if type(update) is not OperationUpdate or update.identity is not self._average_identity:
            return False
        if update.pending is not None:
            pending = update.pending
            if type(pending) is not OperationPending:
                self._notice("Average failed: invalid cleanup-pending token")
                return True
            self._average_pending = pending
            phase = pending.phase.replace("-", " ")
            self._notice(
                f"Average {phase} pending; press Run to retry or Stop to cancel."
            )
            return True
        if update.terminal is None:
            if update.progress is not None:
                self._notice(f"Average: {update.progress.stage} {update.progress.completed}/{update.progress.total}…")
            return True
        target, entry, revision = self._average_target, self._average_entry, self._average_revision
        self._average_identity = self._average_pending = self._average_revision = self._average_target = self._average_entry = None
        terminal = update.terminal; result = terminal.payload
        from xrd_tools.reduction.average import AverageScanResult
        valid_result = False
        if type(result) is AverageScanResult:
            try:
                result.__post_init__()
            except (AttributeError, TypeError, ValueError, OverflowError):
                pass
            else:
                valid_result = True
        typed_cancellation = (
            valid_result and result.disposition == "CANCELLED"
        )
        if terminal.status is OperationTerminalStatus.CANCELLED:
            if result is None:
                self._notice("Average cancelled.")
                return True
            if typed_cancellation:
                self._notice("Average cancelled.")
            else:
                self._notice("Average failed: invalid terminal result")
            return True
        if typed_cancellation:
            self._notice("Average failed: invalid terminal result")
            return True
        if not valid_result:
            self._notice(f"Average failed: {terminal.diagnostic or 'invalid terminal result'}"); return True
        stale = update.stale or revision != self._intents.revision
        if terminal.status is OperationTerminalStatus.FAILED:
            shown = (f"{result.diagnostic_code}: {result.diagnostic}"
                     if result.disposition == "ABORTED" else
                     f"Average failed: {terminal.diagnostic}")
            self._notice(shown); return True
        if result.disposition == "REFUSED":
            self._notice(f"{result.diagnostic_code}: {result.diagnostic}"); return True
        if result.disposition != "COMMITTED" or terminal.status is not OperationTerminalStatus.RETURNED:
            self._notice("Average returned an invalid terminal disposition."); return True
        if result.target != target or result.entry != entry:
            self._notice("Average terminal target mismatch; Browse was not reloaded."); return True
        commit_identity = _terminal_identity_for_target(
            result.commit_identity, target,
        )
        if commit_identity is None:
            self._notice(
                "Average terminal commit identity mismatch; Browse was not reloaded."
            )
            return True
        if stale:
            self._notice(
                "Average committed but context changed; Browse was not reloaded."
            )
            self._request_browser_catalog()
            self._refresh_shell()
            return True
        try:
            self._clear_terminal_browse()
            if not self._release_browse_1d_debt():
                raise RuntimeError("Browse 1-D cache release remains pending")
            self._context_controller.begin_browse(
                target,
                terminal_commit_identity=commit_identity,
            )
        except Exception as error:
            self._error_notice(
                "Average committed; Browse reload deferred", error,
            )
            if self._context_controller.browse_pending:
                self._ensure_timer()
        self._request_browser_catalog()
        self._refresh_shell()
        return True

    def _average_action(self, snapshot: RunIntentSnapshot) -> None:
        if self._authored_asset_owner is not None:
            self._notice("Average is unavailable while authored-asset confirmation is pending.")
            self._refresh_shell(); return
        slot = self._operation_slot
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
            generated = _resolved_generated_target(intent.save_path, name)
            target = os.path.abspath(os.path.expanduser(str(generated)))
        except (TypeError, ValueError, OverflowError) as error:
            self._notice(str(error)); self._refresh_shell(); return
        observation = self._source_observation
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
        identity = slot.begin_average(
            frozen.configuration, target,
            stamp=OperationContextStamp(snapshot.revision),
        )
        if identity is None:
            self._notice("Average operation was not started."); self._refresh_shell(); return
        self._average_identity, self._average_revision = identity, snapshot.revision
        self._average_pending = None
        self._average_target, self._average_entry = target, "entry"
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
            self._retire_native_plot_axis_transition()
            self._retire_batch_terminal_presentation(force=True)
            authored = self._authored_asset_owner
            if authored is not None:
                self._retire_authored_asset(authored)
            self._deferred_metadata = None
            self._pending_reintegrate_reload = None
            self._clear_terminal_browse()
            for dialog in (self._metadata_dialog, self._scan_roi_dialog,
                           self._peak_dialog, self._phase_dialog):
                if dialog is not None: dialog.close()
            ScatteringWorkspace._clear_presentation_targets(self)
            self._clear_live_source_refresh()
            self._shell.browser.cancel_pending_frame_selection()
            self._run_timer.stop()
            self._browser_catalog_timer.stop()
            self._observation_token += 1
            self._cancel_observation()
            self._browser_catalog_token += 1
            self._close_identity = (
                self._context_controller.run_identity
                or self._lifecycle.active_run_identity
                or self._lifecycle.attempt_run_identity
            )

        catalog_clean = self._cancel_browser_catalog()

        operation_slot = getattr(self, "_operation_slot", None)
        analysis_slot = getattr(self, "_analysis_slot", None)
        try:
            operation_close = (
                None if operation_slot is None else operation_slot.close()
            )
            operation_clean = (
                operation_slot is None
                or operation_close.cleanup_status is CleanupStatus.CLEANED
            )
            if operation_clean:
                self._calibration_identity = self._calibration_revision = self._calibration_stamp = self._calibration_request = None
                self._mask_identity = self._mask_revision = self._mask_stamp = self._mask_request = None
                self._asset_validation_identity = None
                self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = self._reintegrate_dimension = None
                self._average_identity = self._average_pending = self._average_revision = self._average_target = self._average_entry = None
        except Exception:
            operation_clean = False

        try:
            analysis_close = (
                None if analysis_slot is None else analysis_slot.close()
            )
            analysis_clean = (
                analysis_slot is None
                or analysis_close.cleanup_status is CleanupStatus.CLEANED
            )
            if analysis_clean:
                self._analysis_identity = None
                self._analysis_kind = None
                self._analysis_target = None
                self._analysis_generation = None
                self._analysis_anchor = None
                self._analysis_request = None
                self._analysis_fingerprint = ""
                self._analysis_candidate = None
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
            and self._operation_slot.current_identity
            is self._background_identity
            or kind is ShellCommandKind.CONTROL_ACTION
            and (
                command.value == "calibrate"
                and self._calibration_identity is not None
                and self._operation_slot.current_identity
                is self._calibration_identity
                or command.value == "make_mask"
                and self._mask_identity is not None
                and self._operation_slot.current_identity is self._mask_identity
                or command.value == f"reintegrate_{self._reintegrate_dimension}"
                and self._reintegrate_identity is not None
                and self._operation_slot.current_identity
                is self._reintegrate_identity
            )
        )
        operation_locked = kind in {ShellCommandKind.RUN_ACTION, ShellCommandKind.SET_BACKGROUND, ShellCommandKind.CONTROL_DRAFT, ShellCommandKind.CONTROL_EDIT, ShellCommandKind.CONTROL_BROWSE, ShellCommandKind.CONTROL_ACTION, ShellCommandKind.SET_PROCESSING_MODE, ShellCommandKind.SET_BATCH, ShellCommandKind.SET_CORES, ShellCommandKind.SET_LIVE, ShellCommandKind.SET_OUTPUT_POLICY} or kind is ShellCommandKind.MENU and (command.value == "Config:Performance Diagnostics…" or str(command.value).startswith("Config:Heavy residency:"))
        operation_retry = (
            kind is ShellCommandKind.RUN_ACTION
            and self._average_pending is not None
            and self._operation_slot.current_identity is self._average_identity
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
            average = self._average_identity
            if average is not None and self._operation_slot.current_identity is average:
                accepted = self._operation_slot.cancel(average)
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
            self._retire_batch_terminal_presentation()
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
            if tool is Tool.XYE_VIEWER and type(value) is str and value:
                selected = command.artifacts or (value,)
                self._open_viewer_1d_paths(selected, current_path=value)
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
            self._retire_batch_terminal_presentation()
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
        pending = self._average_pending
        average = self._average_identity
        if (
            pending is not None
            and average is not None
            and self._operation_slot.current_identity is average
        ):
            accepted = self._operation_slot.retry_average(average, pending)
            if accepted:
                self._average_pending = None
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
            self._retire_batch_terminal_presentation(
                release_display=False,
                force=True,
            )
            self._active_batch_mode = outcome.configuration.batch_mode
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
            self._batch_latest_frame = None
            self._browser_transient_frame = None
            self._browser_transient_clear_token = None
            self._artifact_progress.clear()
            self._progress = ProgressProjection(
                detail="Run started"
            )
            self._batch_visible_progress = (
                self._progress if self._active_batch_mode else None
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
        if self._active_batch_mode:
            self._batch_visible_progress = self._progress
        self._refresh_shell()
        self._ensure_timer()

    def _retire_batch_terminal_presentation(
        self, *, release_display: bool = True, force: bool = False,
    ) -> bool:
        terminal = self._batch_terminal_presentation
        # Batch is a frozen execution fact until its exact terminal arrives.
        # Presentation edits during RUNNING may not turn later FRAME_READY
        # events into Live paints or discard the tracked absolute latest.
        if terminal is None:
            return False
        if not force and not terminal.painted:
            # An unpainted terminal is a durable fail-closed fence.  Ordinary
            # plot, Auto Last, Show All, and frame-selection commands may
            # update their intent/internal navigation, but cannot publish the
            # partial Batch acquisition.  Only page close or a newly admitted
            # run may replace this custody explicitly.
            return False
        owned = True
        if terminal is not None and terminal.awaiting_full_raw:
            self._context_controller.clear_full_raw()
            self._detector_demand_frame = None
        self._batch_terminal_presentation = None
        self._active_batch_mode = False
        self._batch_latest_frame = None
        self._batch_visible_progress = None
        if release_display:
            self._retain_outgoing_display = False
        return owned

    def _qualify_batch_terminal_frame(
        self, event: StandardRunEvent,
    ) -> DisplayFrameKey | None:
        latest = self._batch_latest_frame
        controller = self._context_controller
        navigation = controller.navigation
        if not (
            type(latest) is DisplayFrameKey
            and latest.run_identity is event.run_identity
            and 0 < event.completed <= event.total
            and latest.work_ordinal == event.completed
            and latest.artifact == event.artifact
            and controller.run_identity is event.run_identity
            and controller.owns_frame(latest)
            and bool(navigation.frames)
            and navigation.frames[-1] is latest
            and controller.select_navigation(latest, (latest,))
        ):
            return None
        selected = controller.navigation
        return (
            latest
            if (
                selected.current is latest
                and selected.selected == (latest,)
                and controller.owns_frame(latest)
            )
            else None
        )

    def _begin_batch_terminal_presentation(
        self, event: StandardRunEvent,
    ) -> _BatchTerminalPresentation:
        identity = event.run_identity
        if (
            event.kind is not StandardEventKind.FINISHED
            or event.cleanup_status is not CleanupStatus.CLEANED
        ):
            self._notice(
                "Batch did not finish cleanly; prior display retained."
            )
            return _BatchTerminalPresentation(identity, None)
        frame = self._qualify_batch_terminal_frame(event)
        if frame is None:
            self._notice(
                "Batch terminal frame was not exact; prior display retained."
            )
            return _BatchTerminalPresentation(identity, None)
        if self._preferences.detector_mode != "full":
            return _BatchTerminalPresentation(identity, frame)

        controller = self._context_controller
        context = controller.acquisition_context
        selection = controller.selection
        aligned = bool(
            context is not None
            and selection is not None
            and selection.kind is ContextKind.ACQUISITION
            and selection.names(context)
            # AcquisitionContext derives a fresh immutable HydrationOwner
            # value on every access; exact scope custody is its full value,
            # while the context and selection themselves remain identity-bound.
            and selection.owner == context.hydration_owner
            and controller.run_identity is identity
            and controller.navigation.current is frame
            and controller.navigation.selected == (frame,)
            and controller.owns_frame(frame)
        )
        available, reason = controller.full_raw_availability()
        if not aligned or not available:
            self._notice(
                reason
                or "Batch terminal Full Raw owner was not exact; prior display retained."
            )
            return _BatchTerminalPresentation(identity, None)

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
            return _BatchTerminalPresentation(identity, frame)

        token = controller.request_full_current()
        if token is None:
            self._preferences = replace(
                self._preferences,
                detector_pending=False,
                detector_diagnostic=(
                    diagnostic
                    or "Batch terminal Full Raw request was refused."
                ),
            )
            self._notice(
                "Batch terminal Full Raw request was refused; prior display retained."
            )
            return _BatchTerminalPresentation(identity, None)
        self._ensure_timer()
        return _BatchTerminalPresentation(
            identity,
            frame,
            awaiting_full_raw=True,
            full_request_attempted=True,
        )

    def _accept_batch_terminal_display(
        self, event: StandardRunEvent,
    ) -> bool:
        owner = self._batch_terminal_presentation
        controller = self._context_controller
        navigation = controller.navigation
        selection = controller.selection
        if (
            not self._active_batch_mode
            and owner is not None
            and owner.painted
            and owner.frame is not None
            and event.run_identity is owner.run_identity
            and event.frame_key is owner.frame
            and event.artifact == owner.frame.artifact
            and selection is not None
            and event.selection_generation == selection.display_generation
            and controller.run_identity is owner.run_identity
            and navigation.current is owner.frame
            and navigation.selected == (owner.frame,)
            and controller.owns_frame(owner.frame)
        ):
            # The terminal Full Raw event may already be queued more than
            # once.  Retain the immutable painted receipt after Batch becomes
            # inactive so an exact duplicate cannot enter the ordinary
            # DISPLAY_READY path and repaint the final singleton.
            return True
        if (
            not self._active_batch_mode
            or owner is None
            or not owner.awaiting_full_raw
            or owner.painted
            or owner.frame is None
            or event.run_identity is not owner.run_identity
            or event.frame_key is not owner.frame
        ):
            return False
        if (
            controller.run_identity is not owner.run_identity
            or navigation.current is not owner.frame
            or navigation.selected != (owner.frame,)
            or not controller.owns_frame(owner.frame)
        ):
            return False
        payload = controller.qualify_display_event(event)
        if (
            type(payload) is not StandardDisplayPayload
            or payload.frame_key is not owner.frame
        ):
            return False
        if self._batch_terminal_presentation is not owner:
            return False
        self._batch_terminal_presentation = replace(
            owner, awaiting_full_raw=False,
        )
        self._preferences = replace(
            self._preferences,
            detector_available=True,
            detector_pending=False,
            detector_diagnostic="",
        )
        return True

    def _batch_terminal_ready_to_paint(
        self,
    ) -> _BatchTerminalPresentation | None:
        owner = self._batch_terminal_presentation
        controller = self._context_controller
        navigation = controller.navigation
        return (
            owner
            if (
                self._active_batch_mode
                and owner is not None
                and owner.frame is not None
                and not owner.awaiting_full_raw
                and not owner.painted
                and controller.run_identity is owner.run_identity
                and navigation.current is owner.frame
                and navigation.selected == (owner.frame,)
                and controller.owns_frame(owner.frame)
            )
            else None
        )

    def _paint_batch_terminal(
        self, owner: _BatchTerminalPresentation,
    ) -> bool:
        if self._batch_terminal_ready_to_paint() is not owner:
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
            self._batch_terminal_presentation is owner
            and self._shell_revision > revision
        )
        if applied:
            self._batch_terminal_presentation = replace(
                owner, painted=True,
            )
            self._active_batch_mode = False
            self._batch_latest_frame = None
            self._batch_visible_progress = None
        elif self._batch_terminal_presentation is owner:
            self._retain_outgoing_display = retained
        return applied

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

    def _select_scan(
        self, value: object, *, is_directory: bool = False,
    ) -> None:
        if type(value) is not str or not value:
            return
        if type(is_directory) is not bool:
            return
        handoff = self._terminal_browse_handoff
        presentation = self._terminal_browse_presentation
        terminal_request = (
            handoff.request
            if handoff is not None
            else None if presentation is None else presentation.request
        )
        if is_directory:
            self._set_browser_directory(value, explicit=True)
            return
        self._retire_batch_terminal_presentation()
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
            (terminal_request is None or same_terminal_target)
            and self._context_controller.select_browser_target(value)
        ):
            self._notice("")
            self._refresh_shell()
            return
        try:
            request = self._context_controller.begin_browse(value)
        except Exception as error:
            self._error_notice("Browse refused", error)
            if self._context_controller.browse_pending:
                self._ensure_timer()
            return
        if terminal_request is not None and request is not terminal_request:
            self._clear_terminal_browse()
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
            if command.kind is ShellCommandKind.SELECT_BROWSER_FRAMES:
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
        handoff = self._terminal_browse_handoff
        request = None if handoff is None else handoff.request
        terminal_artifact = None if handoff is None else handoff.artifact
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
            self._terminal_browse_handoff = _TerminalBrowseHandoff(
                request, handoff.run_identity, terminal_artifact,
                current.local_frame_label,
                tuple(candidate.local_frame_label for candidate in selected),
                handoff.commit_identity,
            )
        self._refresh_shell()
        self._ensure_timer()

    def _drain_executor(self) -> None:
        if self._closing or self._closed:
            return
        # Cache borrow debt is page-owned, not selected-context-owned.  Settle
        # it before draining any event that could replace Browse with an
        # acquisition context, and retry an exact deferred reintegrate reload
        # immediately after its bundle reaches terminal release.
        pending_reload = self._pending_reintegrate_reload is not None
        if not self._settle_browse_1d_before_drain():
            return
        pending_reload_changed = bool(
            pending_reload and self._pending_reintegrate_reload is None
        )
        defer_terminal_science = False
        terminal_science_complete = False
        reuse_terminal_science = False
        hold_batch_terminal_science = False
        browse_presentation_ready: _TerminalBrowsePresentation | None = None
        browse_perf_ready: _TerminalBrowsePerf | None = None
        changed = self._poll_admission() or pending_reload_changed
        poll_viewer_1d = getattr(self._context_controller, "poll_viewer_1d", None)
        if poll_viewer_1d is not None and poll_viewer_1d():
            changed = True
        if self._context_controller.poll_viewer_2d():
            changed = True
        if self._context_controller.poll_browse_preview():
            changed = True
        if self._context_controller.browse_pending:
            browse_perf = self._terminal_browse_perf
            handoff = self._terminal_browse_handoff
            exact_browse_perf = (
                browse_perf
                if (
                    browse_perf is not None
                    and handoff is not None
                    and browse_perf.request is handoff.request
                )
                else None
            )
            poll_started = (
                None
                if exact_browse_perf is None
                else self._browse_timing_now()
            )
            if exact_browse_perf is not None and poll_started is None:
                self._terminal_browse_perf = None
                exact_browse_perf = None
            try:
                outcome = self._context_controller.poll_browse()
            except Exception as error:
                self._error_notice("Browse failed", error)
                outcome = None
            finally:
                if (
                    exact_browse_perf is not None
                    and poll_started is not None
                    and self._terminal_browse_perf is exact_browse_perf
                ):
                    poll_ended = self._browse_timing_now()
                    if poll_ended is None:
                        self._terminal_browse_perf = None
                        exact_browse_perf = None
                    else:
                        exact_browse_perf.poll_adopt_count += 1
                        exact_browse_perf.poll_adopt_s += max(
                            0.0, poll_ended - poll_started,
                        )
            if outcome is not None:
                load_outcome = (
                    outcome if type(outcome) is BrowseLoadOutcome else None
                )
                settle_started = (
                    None
                    if exact_browse_perf is None
                    else self._browse_timing_now()
                )
                if exact_browse_perf is not None and settle_started is None:
                    self._terminal_browse_perf = None
                    exact_browse_perf = None
                reuse_terminal_science = self._settle_terminal_browse(
                    outcome,
                    preserve_perf=exact_browse_perf is not None,
                )
                if (
                    exact_browse_perf is not None
                    and settle_started is not None
                    and self._terminal_browse_perf is exact_browse_perf
                ):
                    settle_ended = self._browse_timing_now()
                    if settle_ended is None:
                        self._terminal_browse_perf = None
                    else:
                        exact_browse_perf.settle_s += max(
                            0.0, settle_ended - settle_started,
                        )
                        if (
                            load_outcome is not None
                            and load_outcome.status is BrowseLoadStatus.READY
                            and type(load_outcome.timing) is BrowseLoadTiming
                        ):
                            exact_browse_perf.worker = load_outcome.timing
                            browse_perf_ready = exact_browse_perf
                        else:
                            self._terminal_browse_perf = None
                presentation = self._terminal_browse_presentation
                if (
                    load_outcome is not None
                    and load_outcome.status is BrowseLoadStatus.READY
                    and presentation is not None
                    and presentation.request is load_outcome.request
                    and self._terminal_browse_presentation_is_current(
                        presentation
                    )
                ):
                    browse_presentation_ready = presentation
                changed = True
                detail = getattr(outcome, "detail", "")
                self._notice(detail)
                current = self._context_controller.navigation.current
                if current is not None:
                    self._follow_processed_artifact(current)
        handoff = self._terminal_browse_handoff
        if (
            handoff is not None
            and not self._context_controller.owns_browse_request(
                handoff.request
            )
        ):
            self._clear_terminal_browse()
            changed = True
        presentation = self._terminal_browse_presentation
        if (
            presentation is not None
            and not self._terminal_browse_presentation_is_current(
                presentation
            )
        ):
            if self._terminal_browse_presentation is presentation:
                self._terminal_browse_presentation = None
                self._terminal_rebind_artifacts = None
            perf = self._terminal_browse_perf
            if perf is not None and perf.request is presentation.request:
                self._terminal_browse_perf = None
            changed = True

        force_scientific = changed
        frame_presentation_changed = False
        controls_refresh = False

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
                    follow_latest=False if paced else self._auto_last,
                ):
                    frame = event.navigation_delta.appended
                    first_paced_frame = paced and not self._run_frame_seen
                    self._run_frame_seen = True
                    self._batch_latest_frame = frame
                    self._record_artifact_progress(event)
                    self._progress = ProgressProjection(
                        event.completed,
                        event.total,
                        event.detail,
                        tuple(self._artifact_progress.values()),
                        _directory_file_progress(event),
                    )
                    if self._active_batch_mode:
                        # Batch owns the accepted publication internally, but
                        # FRAME_READY is never a GUI projection boundary.  In
                        # particular, do not publish a transient browser row or
                        # wake a shell/status reconciliation for this prefix.
                        continue
                    self._browser_transient_frame = frame
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
                terminal_owner = self._batch_terminal_presentation
                if (
                    self._active_batch_mode
                    or terminal_owner is not None
                    and terminal_owner.painted
                ):
                    if self._accept_batch_terminal_display(event):
                        if self._active_batch_mode:
                            changed = True
                            force_scientific = True
                        continue
                    if self._active_batch_mode:
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
                was_batch = self._active_batch_mode
                if was_batch:
                    owner = self._batch_terminal_presentation
                    first_terminal = owner is None
                    if first_terminal:
                        owner = self._begin_batch_terminal_presentation(event)
                        self._batch_terminal_presentation = owner
                        if owner.frame is not None:
                            self._follow_processed_artifact(owner.frame)
                    elif owner.run_identity is not event.run_identity:
                        owner = _BatchTerminalPresentation(
                            event.run_identity, None,
                        )
                        self._batch_terminal_presentation = owner
                        self._notice(
                            "Batch terminal owner changed; prior display retained."
                        )
                else:
                    ScatteringWorkspace._flush_presentation_target(self)
                    self._retain_outgoing_display = False
                batch_owner = self._batch_terminal_presentation
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
                    and self._batch_terminal_presentation is not None
                    and self._batch_terminal_presentation.frame is None
                    else ""
                )
                self._accept_terminal_event(event)
                defer_terminal_science = self._begin_terminal_browse(
                    event,
                    was_batch=was_batch,
                    current=terminal_navigation.current,
                    selected=terminal_navigation.selected,
                )
                handoff = self._terminal_browse_handoff
                terminal_science_complete = (
                    defer_terminal_science
                    and handoff is not None
                    and self._terminal_scientific_matches(
                        handoff, terminal_navigation,
                    )
                )
                if batch_notice:
                    self._notice(batch_notice)
                self._retry_deferred_gi_motor_default()
                if was_batch:
                    # Terminal status may now replace the frozen Batch-visible
                    # status, independently of whether final science qualifies.
                    self._batch_visible_progress = None
                # The writer publishes its atomic final path before emitting
                # the terminal event.  Re-enumerate here so a non-batch run
                # whose FRAME_READY preceded that rename becomes visible.
                catalog_request = self._request_browser_catalog()
                self._browser_transient_clear_token = (
                    None if catalog_request is None else catalog_request.token
                )
                self._run_frame_seen = False
                if (
                    not was_batch
                    or self._batch_terminal_presentation is None
                    or self._batch_terminal_presentation.frame is None
                ):
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
            if update is not None:
                if self._consume_asset_validation_update(update):
                    operation_refresh = _OperationRefresh.FULL
                else:
                    operation_refresh = self._consume_reintegrate_update(
                        update
                    )
                    if (
                        operation_refresh is _OperationRefresh.NONE
                        and (
                            self._consume_calibration_update(update)
                            or self._consume_mask_update(update)
                            or self._consume_background_update(update)
                            or self._consume_average_update(update)
                        )
                    ):
                        operation_refresh = _OperationRefresh.FULL
                if operation_refresh is _OperationRefresh.CONTROLS:
                    controls_refresh = True
                elif operation_refresh is _OperationRefresh.FULL:
                    changed = True
                    force_scientific = True
            elif operation_identity is self._background_identity and not operation_slot.owned:
                self._background_identity = None; self._notice("Display background failed before terminal publication."); changed = True; force_scientific = True
            elif operation_identity is self._reintegrate_identity and not operation_slot.owned:
                request, target = self._reintegrate_request, self._reintegrate_target; shown = {"1d": "1-D", "2d": "2-D"}.get(self._reintegrate_dimension, "operation"); self._reintegrate_identity = self._reintegrate_request = self._reintegrate_target = self._reintegrate_dimension = None
                if request is not None and target is not None: self._reload_after_reintegrate(request, target)
                self._notice(f"Reintegrate {shown} failed before terminal publication."); changed = True; force_scientific = True
            elif operation_identity is self._average_identity and not operation_slot.owned:
                self._average_identity = self._average_pending = self._average_revision = self._average_target = self._average_entry = None
                self._notice("Average failed before terminal publication."); changed = True; force_scientific = True
            elif operation_identity is self._asset_validation_identity and not operation_slot.owned:
                self._asset_validation_identity = None
                owner = self._authored_asset_owner
                if owner is not None:
                    owner.validation_identity = None
                    owner.validation_request = None
                    owner.requested_path = None
                    owner.dialog.set_busy(False)
                self._notice("Asset validation failed before terminal publication.")
                changed = True

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
                if analysis_refresh is _OperationRefresh.CONTROLS:
                    controls_refresh = True
                elif analysis_refresh is _OperationRefresh.FULL:
                    changed = True
                    force_scientific = True
            elif (
                analysis_identity is self._analysis_identity
                and not analysis_slot.owned
            ):
                kind, target = self._analysis_kind, self._analysis_target
                self._analysis_identity = None
                self._analysis_kind = None
                self._analysis_target = None
                self._analysis_generation = None
                self._analysis_anchor = None
                self._analysis_request = None
                self._analysis_fingerprint = ""
                self._analysis_candidate = None
                message = "Analysis failed before terminal publication."
                self._notice(message)
                if kind in {"metadata", "metadata_requalification"}:
                    if self._finish_metadata_refresh(
                        target, message,
                    ) is _OperationRefresh.CONTROLS:
                        controls_refresh = True
                else:
                    changed = True
                    force_scientific = True

        deferred_refresh = self._dispatch_deferred_metadata()
        if deferred_refresh is _OperationRefresh.CONTROLS:
            controls_refresh = True
        elif deferred_refresh is _OperationRefresh.FULL:
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
        batch_ready = self._batch_terminal_ready_to_paint()
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
            self._active_batch_mode
            and self._batch_terminal_ready_to_paint() is None
        )
        if changed:
            handoff = self._terminal_browse_handoff
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
                    presentation_started = (
                        None
                        if browse_perf_ready is None
                        else self._browse_timing_now()
                    )
                    if (
                        browse_perf_ready is not None
                        and presentation_started is None
                        and self._terminal_browse_perf is browse_perf_ready
                    ):
                        self._terminal_browse_perf = None
                        browse_perf_ready = None
                    owner = browse_presentation_ready
                    if (
                        owner is not None
                        and not self._terminal_browse_presentation_is_current(
                            owner
                        )
                    ):
                        owner = None
                        browse_perf_ready = None
                    prior_revision = self._shell_revision
                    self._refresh_event_shell(
                        preserve_scientific=True,
                        skip_scientific_projection=True,
                        rebind_scientific_navigation=True,
                    )
                    if owner is not None:
                        self._finish_terminal_browse_presentation(
                            owner,
                            browse_perf_ready,
                            started=presentation_started,
                            mode="rebind",
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
                    candidate_owner = self._terminal_browse_presentation
                    if (
                        candidate_owner is not None
                        and self._terminal_browse_presentation_is_current(
                            candidate_owner
                        )
                    ):
                        presentation_owner = candidate_owner
                presentation_perf = browse_perf_ready
                presentation_mode = "repaint"
                if (
                    presentation_perf is None
                    and presentation_owner is not None
                ):
                    candidate = self._terminal_browse_perf
                    if (
                        candidate is not None
                        and candidate.request is presentation_owner.request
                    ):
                        presentation_perf = candidate
                        if candidate.fallback_pending:
                            presentation_mode = "repaint-fallback"
                presentation_started = (
                    None
                    if presentation_perf is None
                    else self._browse_timing_now()
                )
                if (
                    presentation_perf is not None
                    and presentation_started is None
                    and self._terminal_browse_perf is presentation_perf
                ):
                    self._terminal_browse_perf = None
                    presentation_perf = None
                if (
                    presentation_owner is not None
                    and not self._terminal_browse_presentation_is_current(
                        presentation_owner
                    )
                ):
                    presentation_owner = None
                    presentation_perf = None
                prior_revision = self._shell_revision
                self._refresh_event_shell()
                if presentation_owner is not None:
                    self._finish_terminal_browse_presentation(
                        presentation_owner,
                        presentation_perf,
                        started=presentation_started,
                        mode=presentation_mode,
                        applied=self._shell_revision > prior_revision,
                    )
        self._show_queued_authored_asset_confirmation()
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

    def _browse_timing_now(self) -> float | None:
        try:
            value = float(self._browse_clock())
        except BaseException:
            return None
        return value if math.isfinite(value) else None

    def _terminal_browse_presentation_is_current(
        self, owner: _TerminalBrowsePresentation,
    ) -> bool:
        if (
            type(owner) is not _TerminalBrowsePresentation
            or self._terminal_browse_presentation is not owner
            or not self._context_controller.owns_browse_request(
                owner.request
            )
        ):
            return False
        try:
            captured = self._context_controller.capture_reintegrate_browse()
        except BaseException:
            return False
        return bool(
            captured is not None
            and captured[0] is owner.context
            and captured[1] is owner.request
        )

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
        context.  A successful run already published that artifact, so follow
        it through the existing asynchronous BrowseLoader instead of requiring
        a redundant browser click or weakening the reintegration contract.
        """

        controller = self._context_controller
        acquisition = controller.acquisition_context
        selection = controller.selection
        artifact = event.artifact
        run_identity = event.run_identity
        if (
            was_batch
            or event.kind is not StandardEventKind.FINISHED
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
        browse_started = (
            self._browse_timing_now()
            if _browse_perf_enabled()
            else None
        )
        try:
            if not self._release_browse_1d_debt():
                return False
            commit_identity = event.terminal_commit_identity
            request = (
                controller.begin_browse(artifact)
                if commit_identity is None
                else controller.begin_browse(
                    artifact,
                    terminal_commit_identity=commit_identity,
                )
            )
        except RuntimeError:
            # The published artifact remains available in the browser.  A
            # transient viewer/cleanup owner must not turn a clean Run terminal
            # into a failure or bypass Browse's normal lifecycle checks.
            return False
        # Browse owns the resolved absolute source path, while the output
        # contract deliberately preserves an explicit target's spelling.
        # Retain both exact identities so acquisition-frame choices made while
        # Browse is pending compare against their original artifact spelling.
        self._clear_terminal_browse()
        self._terminal_browse_handoff = _TerminalBrowseHandoff(
            request,
            run_identity,
            artifact,
            current.local_frame_label
            if current is not None and current.artifact == artifact
            else None,
            tuple(
                frame.local_frame_label
                for frame in selected
                if frame.artifact == artifact
            ),
            request.terminal_commit_identity,
        )
        self._terminal_browse_perf = (
            None
            if browse_started is None
            else _TerminalBrowsePerf(request, browse_started)
        )
        self._ensure_timer()
        return True

    def _terminal_scientific_matches(
        self,
        handoff: _TerminalBrowseHandoff,
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
            or scientific.slice_pins
            or scientific.pinned_traces
            or view.presentation_plot_mode != preferences.plot_mode
            or current is None
            or painted_current is None
        ):
            return False
        artifact = handoff.request.source_path
        canonical_by_artifact = {
            handoff.artifact: artifact,
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
        matches = (
            _terminal_frame_signature(
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

    def _terminal_scientific_rebind_authorized(
        self,
        navigation: FrameNavigationProjection,
        authorization: tuple[RunIdentity, str, str],
    ) -> bool:
        scientific = self._last_scientific_projection
        view = self._shell.scientific
        if (
            type(navigation) is not FrameNavigationProjection
            or type(authorization) is not tuple
            or len(authorization) != 3
            or type(authorization[0]) is not RunIdentity
            or not all(
                type(value) is str and value
                for value in authorization[1:]
            )
            or scientific is None
        ):
            return False
        run_identity, source_artifact, browse_artifact = authorization
        if (
            self._context_controller.run_identity is not run_identity
            or self._context_controller.acquisition_context is None
        ):
            return False
        canonical_by_artifact = {
            source_artifact: browse_artifact,
            browse_artifact: browse_artifact,
        }
        frames = (
            *((navigation.current,) if navigation.current is not None else ()),
            *navigation.frames,
            *navigation.selected,
            *((view.navigation_current_key,)
              if view.navigation_current_key is not None else ()),
            *view.navigation_frame_keys,
            *view.navigation_selected_keys,
            *view.trace_history_keys,
            *view.heavy_available_keys,
            *((scientific.heavy.frame,)
              if scientific.heavy is not None else ()),
            *(trace.frame for trace in scientific.traces),
            *scientific.heavy_available,
        )
        authorized = bool(frames) and all(
            type(frame) is DisplayFrameKey
            and frame.run_identity is run_identity
            and frame.artifact in canonical_by_artifact
            for frame in frames
        )
        if not authorized or self._preferences.plot_mode != "Single":
            return authorized
        current = navigation.current
        painted_current = view.navigation_current_key
        if current is None or painted_current is None:
            return False
        current_signature = _terminal_frame_signature(
            current, canonical_by_artifact,
        )
        return (
            scientific.plot_mode == "Single"
            and view.presentation_plot_mode == "Single"
            and len(navigation.selected) == 1
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
        navigation: FrameNavigationProjection,
        authorization: tuple[RunIdentity, str, str],
    ) -> bool:
        scientific = self._last_scientific_projection
        if (
            scientific is None
            or type(authorization) is not tuple
            or len(authorization) != 3
            or type(authorization[0]) is not RunIdentity
            or not all(
                type(value) is str and value
                for value in authorization[1:]
            )
        ):
            return False
        run_identity, source_artifact, browse_artifact = authorization
        canonical_by_artifact = {
            source_artifact: browse_artifact,
            browse_artifact: browse_artifact,
        }
        candidate_frames = navigation.frames
        scientific_frames = (
            *((scientific.heavy.frame,) if scientific.heavy is not None else ()),
            *(trace.frame for trace in scientific.traces),
            *scientific.heavy_available,
        )
        if any(
            type(frame) is not DisplayFrameKey
            or frame.run_identity is not run_identity
            or frame.artifact not in canonical_by_artifact
            for frame in (*candidate_frames, *scientific_frames)
        ):
            return False
        by_signature = {
            _terminal_frame_signature(frame, canonical_by_artifact): frame
            for frame in navigation.frames
        }
        if len(by_signature) != len(navigation.frames):
            return False

        def rebound(frame: DisplayFrameKey) -> DisplayFrameKey | None:
            return by_signature.get(_terminal_frame_signature(
                frame, canonical_by_artifact,
            ))

        heavy_frame = (
            None
            if scientific.heavy is None
            else rebound(scientific.heavy.frame)
        )
        trace_frames = tuple(rebound(trace.frame) for trace in scientific.traces)
        available = tuple(rebound(frame) for frame in scientific.heavy_available)
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
        *,
        preserve_perf: bool = False,
    ) -> bool:
        handoff = self._terminal_browse_handoff
        request = None if handoff is None else handoff.request
        if (
            type(outcome) is not BrowseLoadOutcome
            or request is None
            or outcome.request is not request
        ):
            return False
        current_label = handoff.current_label
        selected_labels = handoff.selected_labels
        if outcome.status is not BrowseLoadStatus.READY:
            self._clear_terminal_browse(preserve_perf=preserve_perf)
            return False
        captured = self._context_controller.capture_reintegrate_browse()
        if (
            captured is None
            or captured[1] is not request
        ):
            self._clear_terminal_browse(preserve_perf=preserve_perf)
            return False
        navigation = self._context_controller.navigation
        by_label = {
            frame.local_frame_label: frame for frame in navigation.frames
        }
        current = by_label.get(current_label)
        selected = tuple(
            by_label[label]
            for label in selected_labels
            if label in by_label
        )
        if self._auto_last and navigation.frames:
            self._context_controller.select_navigation(
                navigation.frames[-1], selected
            )
        elif current is not None:
            self._context_controller.select_navigation(current, selected)
        context = captured[0]
        self._clear_terminal_browse(preserve_perf=preserve_perf)
        self._terminal_browse_presentation = _TerminalBrowsePresentation(
            request, context,
        )
        commit_identity = handoff.commit_identity
        if (
            request.terminal_commit_identity is not commit_identity
            or type(commit_identity) is not StreamTerminal
            or captured[5].size != commit_identity.size
            or captured[5].digest != commit_identity.digest
        ):
            # An ordinary terminal Browse may preserve its exact frame-label
            # choice, but only a writer-authenticated seal can authorize
            # zero-copy reuse of the acquisition scientific arrays.
            return False
        matches = self._terminal_scientific_matches(
            handoff, self._context_controller.navigation,
        )
        if matches:
            self._terminal_rebind_artifacts = (
                handoff.run_identity,
                handoff.artifact,
                request.source_path,
            )
        return matches

    def _clear_terminal_browse(
        self, *, preserve_perf: bool = False,
    ) -> None:
        self._terminal_browse_handoff = None
        self._terminal_browse_presentation = None
        self._terminal_rebind_artifacts = None
        if not preserve_perf:
            self._terminal_browse_perf = None

    def _finish_terminal_browse_presentation(
        self,
        owner: _TerminalBrowsePresentation,
        perf: _TerminalBrowsePerf | None,
        *,
        started: float | None,
        mode: str,
        applied: bool,
    ) -> None:
        if not self._terminal_browse_presentation_is_current(owner):
            if self._terminal_browse_presentation is owner:
                self._terminal_browse_presentation = None
                self._terminal_rebind_artifacts = None
            current_perf = self._terminal_browse_perf
            if (
                current_perf is not None
                and current_perf.request is owner.request
            ):
                self._terminal_browse_perf = None
            return
        exact_perf = (
            perf
            if (
                perf is not None
                and self._terminal_browse_perf is perf
                and perf.request is owner.request
            )
            else None
        )
        ended = (
            self._browse_timing_now()
            if exact_perf is not None and started is not None
            else None
        )
        if exact_perf is not None:
            if ended is None:
                self._terminal_browse_perf = None
                exact_perf = None
            else:
                exact_perf.presentation_s += max(0.0, ended - started)
        if not applied:
            self._scientific_repaint_pending = True
            self._ensure_timer()
            return
        if mode == "rebind" and self._scientific_repaint_pending:
            if exact_perf is not None:
                exact_perf.fallback_pending = True
            return
        if mode == "repaint-fallback" and self._scientific_repaint_pending:
            return
        worker = None if exact_perf is None else exact_perf.worker
        if (
            type(worker) is BrowseLoadTiming
            and ended is not None
            and type(exact_perf.started_at) is float
            and math.isfinite(exact_perf.started_at)
        ):
            try:
                # The worker froze this resolved source while qualifying the
                # artifact.  Telemetry must not perform GUI-thread path I/O.
                source = worker.canonical_path
                gui_total = max(0.0, ended - exact_perf.started_at)
                _LOG.info(
                    "[PERF-BROWSE] source=%s token=%s generation=%d "
                    "seal=%s records=%d mode=%s | "
                    "worker=%.3fs initial-seal=%.3fs scan-open=%.3fs "
                    "record-iteration=%.3fs presentation-read=%.3fs "
                    "final-seal=%.3fs context-build=%.3fs | "
                    "gui-total=%.3fs poll/adopt=%.3fs(n=%d) "
                    "settle=%.3fs presentation=%.3fs",
                    source,
                    owner.request.token,
                    owner.request.load_generation,
                    worker.seal_mode,
                    worker.record_count,
                    mode,
                    worker.worker_total_s,
                    worker.initial_seal_s,
                    worker.scan_open_s,
                    worker.record_iteration_s,
                    worker.presentation_read_s,
                    worker.final_seal_s,
                    worker.context_build_s,
                    gui_total,
                    exact_perf.poll_adopt_s,
                    exact_perf.poll_adopt_count,
                    exact_perf.settle_s,
                    exact_perf.presentation_s,
                )
            except BaseException:
                pass
        if self._terminal_browse_presentation is owner:
            self._terminal_browse_presentation = None
            self._terminal_rebind_artifacts = None
        current_perf = self._terminal_browse_perf
        if (
            current_perf is not None
            and current_perf.request is owner.request
        ):
            self._terminal_browse_perf = None
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
            )
        else:
            self._refresh_shell(
                suppress_detector_demand=suppress_detector_demand,
                allow_batch_terminal_paint=allow_batch_terminal_paint,
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
        operation_slot = getattr(self, "_operation_slot", None)
        analysis_slot = getattr(self, "_analysis_slot", None)
        try:
            if (
                (operation_slot is not None and operation_slot.owned)
                or (analysis_slot is not None and analysis_slot.owned)
            ):
                return True
        except Exception:
            return True
        if (
            self._deferred_metadata is not None
            or self._admission is not None
            or getattr(self._context_controller, "viewer_1d_loading", False)
            or getattr(self._context_controller, "viewer_1d_cleanup_pending", False)
            or self._context_controller.viewer_2d_loading
            or self._context_controller.browse_pending
            or self._context_controller.browse_preview_polling_needed
            or self._browse_1d_release_debt is not None
            or self._pending_reintegrate_reload is not None
            or self._scientific_repaint_pending
        ):
            return True
        batch_terminal = self._batch_terminal_presentation
        if (
            batch_terminal is not None
            and batch_terminal.frame is not None
            and not batch_terminal.painted
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
        self._retry_pending_reintegrate_reload()
        return self._browse_1d_release_debt is None

    def _set_detector_mode(self, mode: str) -> bool:
        if mode == "thumbnail":
            terminal = self._batch_terminal_presentation
            terminal_was_waiting = bool(
                terminal is not None and terminal.awaiting_full_raw
            )
            if self._preferences.detector_mode != mode:
                self._retire_batch_terminal_presentation()
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
            self._retire_batch_terminal_presentation()
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
        captured = self._context_controller.capture_reintegrate_browse()
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
        _context, request, selection, artifact = captured[:4]
        if (
            selection is not self._context_controller.selection
            or request.source_path != artifact
            or navigation.current is None
        ):
            return False
        expected_logical = (
            (navigation.current,)
            if snapshot.plot_mode == "Single"
            else navigation.selected
        )
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
    ) -> None:
        explicit_preserve_display = preserve_display
        explicit_preserve_scientific = preserve_scientific
        allow_batch_terminal_paint = bool(
            allow_batch_terminal_paint
            and self._batch_terminal_ready_to_paint() is not None
        )
        batch_science_hold = (
            self._active_batch_mode and not allow_batch_terminal_paint
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
            or self._pending_reintegrate_reload is not None
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
        viewer_2d, viewer_1d = tool is Tool.IMAGE_VIEWER, tool is Tool.XYE_VIEWER
        viewer = viewer_1d or viewer_2d
        if viewer:
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
        browse_cache_supported = bool(
            browse_selected
            and intent.processing_mode in {"Int 1D", "Int 2D"}
            and self._preferences.plot_mode in {
                "Single", "Overlay", "Waterfall",
            }
            and not self._preferences.slice_enabled
            and not self._preferences.slice_pins
        )
        if skip_scientific_projection:
            payloads = ()
        elif browse_selected and not browse_cache_supported:
            payloads = ()
            cache_adoption_missing = True
            cache_terminal_diagnostic = (
                "Browse cache display is unavailable for Average/Sum "
                "and 2-D slice or pin projections."
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
            not self._active_batch_mode
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
        projection_progress = (
            self._batch_visible_progress
            if (
                self._active_batch_mode
                and self._batch_visible_progress is not None
            )
            else self._progress
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
            progress=projection_progress,
            preferences=self._preferences,
            browser_directory=self._browser_directory,
            browser_catalog=self._browser_catalog,
            browser_transient_frame=self._browser_transient_frame,
            viewer_1d_paths=(
                self._context_controller.viewer_1d_context.paths
                if viewer_1d
                and self._context_controller.viewer_1d_context is not None
                else ()
            ),
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
        projection = replace(
            projection,
            scientific=replace(
                projection.scientific,
                background_enabled=not self._mutating_operation_busy(),
            ),
        )
        pending_average = self._average_pending
        if (
            pending_average is not None
            and self._operation_slot.current_identity is self._average_identity
        ):
            pending_readiness = (
                "Average cleanup pending · Run retries · Stop cancels"
            )
            projection = replace(
                projection,
                controls=replace(
                    projection.controls,
                    profile=replace(
                        projection.controls.profile,
                        run_enabled=True,
                        run_blockers=(),
                    ),
                ),
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
        rebind_artifacts = (
            self._terminal_rebind_artifacts
            if rebind_scientific_navigation
            else None
        )
        if rebind_scientific_navigation:
            self._terminal_rebind_artifacts = None
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
            if rebind_scientific_navigation:
                prior_scientific = self._last_scientific_projection
                prior_trace_history = (
                    self._shell.scientific.trace_history_projections
                )
                rebound = (
                    rebind_artifacts is not None
                    and prior_scientific is not None
                    and prior_scientific.browse_trace_snapshot is None
                    and self._terminal_scientific_rebind_authorized(
                        navigation, rebind_artifacts,
                    )
                    and self._rebind_last_scientific_projection(
                        navigation, rebind_artifacts,
                    )
                    and self._shell.scientific.rebind_navigation(
                        navigation,
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
        operation_identity = getattr(
            self._operation_slot, "current_identity", None,
        )
        operation_active = (operation_identity is not None and not self._closing and not self._closed)
        operation_busy = self._experiment_operation_busy()
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
            operation_busy=operation_busy,
            calibration_active=calibration_active,
            mask_available=mask_available,
            mask_dependency_available=mask_dependency_available,
            mask_active=mask_active,
            reintegrate_available=(not self._closing and not self._closed
                and self._admission_state is None
                and intent.processing_mode != "Int 1D (XYE)"
                and (phase is RunPhase.IDLE or phase is RunPhase.FAILED
                     and self._lifecycle.reset_permitted)
                and self._context_controller.capture_reintegrate_browse() is not None),
            reintegrate_active=reintegrate_active, reintegrate_dimension=self._reintegrate_dimension,
        )
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
                in controls.profile.section_actions.items()
            }
            controls = replace(
                controls,
                profile=replace(
                    controls.profile,
                    section_actions=section_actions,
                ),
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
            handoff = self._terminal_browse_handoff
            if (
                handoff is not None
                and self._context_controller.owns_browse_request(
                    handoff.request
                )
            ):
                return False, "Loading finalized Browse context…"
            return False, "Browse cleanup remains pending"
        presentation = self._terminal_browse_presentation
        if (
            presentation is not None
            and self._terminal_browse_presentation_is_current(presentation)
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
            if value != candidate.processing_mode:
                self._retire_batch_terminal_presentation()
                release_outgoing_display = True
                self._retain_outgoing_display = False
                self._release_display_background()
            candidate.processing_mode = value
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
            chooser = getattr(self, "_viewer_1d_file_chooser", self._viewer_file_chooser)
            selected = chooser(self._viewer_1d_start_directory())
            selected = tuple(selected) if type(selected) in {tuple, list} else ()
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
        self._retire_batch_terminal_presentation()
        self._retain_outgoing_display = False
        if (self._context_controller.viewer_2d_owned
                and not self._clear_viewer_2d_renderer(close=True)):
            self._notice("2D Viewer cleanup remains pending"); return
        context = self._context_controller.viewer_1d_context
        if context is not None and context.state.value == "ready":
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
        self._open_viewer_2d_path(selected)

    def _open_viewer_2d_path(self, selected: str) -> None:
        if type(selected) is not str or not selected:
            return
        self._retire_batch_terminal_presentation()
        self._retain_outgoing_display = False
        if (getattr(self._context_controller, "viewer_1d_owned", False)
                and not self._clear_viewer_1d_renderer(close=True)):
            self._notice("1D Viewer cleanup remains pending"); return
        context = self._context_controller.viewer_2d_context
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
            if value != prior_mode:
                self._retire_batch_terminal_presentation()
            if not viewer_1d:
                ScatteringWorkspace._clear_presentation_targets(self)
            updates["plot_mode"] = value
            if value not in {"Overlay", "Waterfall"}:
                updates["slice_pins"] = ()
            if value == "Single" and prior_mode != "Single":
                self._shell.browser.cancel_pending_frame_selection()
                current = self._context_controller.navigation.current
                if current is not None:
                    if viewer_1d:
                        self._context_controller.select_viewer_1d(
                            current, (current,),
                        )
                    else:
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
            changed = requested != self._date_sorted
            self._date_sorted = requested
            if changed and requested:
                self._request_browser_catalog()
            return True
        elif kind is ShellCommandKind.SET_AUTO_LAST:
            requested = bool(value)
            if requested != self._auto_last:
                self._retire_batch_terminal_presentation()
            ScatteringWorkspace._clear_presentation_targets(self)
            self._auto_last = requested
            if self._auto_last:
                self._context_controller.select_latest_navigation(
                    plot_mode=self._preferences.plot_mode
                )
            return True
        elif kind is ShellCommandKind.CLEAR_1D:
            self._retire_batch_terminal_presentation()
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
        prior_source = prior_intent.source_spec
        current_source = current_intent.source_spec
        catalog_policy_changed = (
            _browser_suffixes_for_mode(prior_intent.processing_mode)
            != _browser_suffixes_for_mode(current_intent.processing_mode)
        )
        if current_source is not None:
            current_mode = source_mode(current_source)
            self._source_mode = current_mode
            self._source_history[current_mode] = current_source
        browser_directory_changed = (
            prior_intent.save_path != current_intent.save_path
            and not self._browser_explicit_directory
        )
        if browser_directory_changed:
            self._browser_directory = processed_directory(
                current_intent.save_path
            )
        if browser_directory_changed or catalog_policy_changed:
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
            title = "Choose TIFF for mask"
            file_filter = "TIFF image (*.tif *.tiff)"
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

    def _request_browser_catalog(self) -> _BrowserCatalogRequest | None:
        pool = self._browser_catalog_pool
        if (
            pool is None
            or self._closing
            or self._closed
        ):
            return None
        self._browser_catalog_token += 1
        request = _BrowserCatalogRequest(
            self._browser_catalog_token,
            self._browser_directory,
            _browser_suffixes_for_mode(
                self._intents.snapshot().thaw().processing_mode
            ),
        )
        operation = self._browser_catalog_operation
        if operation is not None:
            operation.cancelled.set()
            operation.future.cancel()
            self._browser_catalog_queued = request
            return request
        self._launch_browser_catalog(request)
        return request

    def _launch_browser_catalog(
        self, request: _BrowserCatalogRequest,
    ) -> None:
        pool = self._browser_catalog_pool
        if (
            pool is None or self._closing or self._closed
            or self._browser_catalog_operation is not None
        ):
            return
        cancelled = threading.Event()
        future = pool.submit(
            enumerate_processed_artifacts,
            request.directory,
            accepted_suffixes=request.accepted_suffixes,
            inspect_directory_contents=self._date_sorted,
            directory_time_cache=self._browser_directory_time_cache,
            cancelled=cancelled,
        )
        operation = _BrowserCatalogOperation(request, cancelled, future)
        self._browser_catalog_operation = operation
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
        operation = self._browser_catalog_operation
        if operation is not None and operation.future.done():
            self._on_browser_catalog(operation, operation.future)
        elif operation is None and self._browser_catalog_queued is not None:
            request, self._browser_catalog_queued = (
                self._browser_catalog_queued, None,
            )
            self._launch_browser_catalog(request)
        elif operation is None:
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
        ):
            return
        request = operation.request
        self._browser_catalog_operation = None
        queued, self._browser_catalog_queued = (
            self._browser_catalog_queued, None,
        )
        current = (
            not operation.cancelled.is_set()
            and request.token == self._browser_catalog_token
            and request.directory == self._browser_directory
            and request.accepted_suffixes == _browser_suffixes_for_mode(
                self._intents.snapshot().thaw().processing_mode
            )
        )
        try:
            catalog = operation.future.result()
        except Exception as error:
            if current:
                self._error_notice("Browser refresh failed", error)
            if queued is not None:
                self._launch_browser_catalog(queued)
            return
        if queued is not None:
            self._launch_browser_catalog(queued)
        if not current:
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
            and request.token >= clear_token
        )
        if transient_cleared:
            self._browser_transient_frame = None
            self._browser_transient_clear_token = None
        if catalog == self._browser_catalog and not transient_cleared:
            return
        self._browser_catalog = catalog
        self._refresh_shell(
            preserve_scientific=True,
            skip_scientific_projection=True,
            suppress_detector_demand=True,
        )

    def _cancel_browser_catalog(self) -> bool:
        self._browser_catalog_queued = None
        operation = self._browser_catalog_operation
        if operation is None:
            return True
        operation.cancelled.set()
        operation.future.cancel()
        if not operation.future.done():
            return False
        self._browser_catalog_operation = None
        try:
            operation.future.result()
        except BaseException:
            pass
        return True

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
