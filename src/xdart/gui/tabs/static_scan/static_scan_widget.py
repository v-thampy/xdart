# -*- coding: utf-8 -*-
"""
@author: walroth, thampy
"""

# Standard library imports
import logging
from queue import Queue
import threading
import copy
import os
import math
from collections import OrderedDict
import gc
import imageio
import pyFAI

logger = logging.getLogger(__name__)

# Transitional live-display mirrors.  The publication store is the normal scan
# display source now; these dicts remain as recent-row caches for legacy
# fallback paths and viewer modes.
_DISPLAY_1D_CACHE_MAX = 512
_DISPLAY_2D_CACHE_MAX = 40

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtWidgets: Any = None
    QtCore: Any = None
else:
    from pyqtgraph.Qt import QtWidgets, QtCore

# This module imports
from xdart.modules.live import LiveFrame, LiveScan
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_error_details,
    publication_from_live_frame,
    publication_has_2d_errors,
)
from xdart.modules.wavelength import normalize_wavelength_m
from xrd_tools.core.energy import wavelength_m_to_energy_eV
from .ui.staticUI import Ui_Form
from .h5viewer import H5Viewer
from .display_frame_widget import displayFrameWidget
from .integrator import (
    GI_LABELS_1D,
    GI_LABELS_2D,
    GI_MODES_1D,
    GI_MODES_2D,
    Units,
    Units_dict,
    Units_dict_inv,
    integratorTree,
)
from .scan_threads import stitchThread
from .metadata import metadataWidget
from .wranglers import imageWrangler, nexusWrangler, wranglerWidget
from .controls_logic import (
    AnalysisTool,
    BOUND_CONTROL_PATHS,
    ControlAction,
    INTEGRATOR_BACKED_CONTROL_PATHS,
    INTEGRATOR_BACKED_CONTROL_SPECS,
    INTEGRATION_CONTROL_PATHS,
    ControlState,
    GeomState,
    MeasMode,
    ResultCaps,
    SourceCaps,
    Tool,
    build_control_panel_state,
    build_native_int_reduction_plan_from_scan,
    coerce_control_edit_value,
    tool_from_mode_text,
)
from xdart.utils.throttle import Coalescer
from xdart.utils._utils import FixSizeOrderedDict, get_fname_dir, get_img_data
from xdart.modules.reduction import ThresholdSaturationConfig

QWidget = QtWidgets.QWidget
QSizePolicy = QtWidgets.QSizePolicy
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
QDialog = QtWidgets.QDialog
QInputDialog = QtWidgets.QInputDialog
QCombo = QtWidgets.QComboBox

wranglers = {
    'Image Files': imageWrangler,
    'NeXus': nexusWrangler,
}


class _XyeOverlayInputFilter(QtCore.QObject):
    """Modifier-free, plotMethod-aware multi-select for the XYE file list (E1/E2).

    Active only in XYE viewer (the list is ``ExtendedSelection``).  Mirrors
    ``h5viewer._AccumulatingClickFilter`` semantics on the file list:

    * **Accumulating plot methods** (Overlay / Waterfall / Sum / Average) build a
      comparison set — a plain left-click toggles a file in/out, and Up/Down
      arrows EXTEND the selection (add the newly-current file without clearing),
      so arrow-browsing accumulates just like clicking.
    * **Single** mode browses one file — a plain click replaces (Qt default) and
      arrows move one row (Qt default).

    Directories / ``..`` keep default click-to-navigate; the filter is inert
    outside XYE mode.  (Vivek's model: the *selection* accumulates and the plot =
    the selected set per plotMethod, so Sum/Average accumulate on arrow too —
    which diverges from Int 1D's plan_overlay where Sum/Average are REPLACE.)
    """

    _ACCUMULATING = ('Overlay', 'Waterfall', 'Sum', 'Average')

    def __init__(self, list_widget, is_active, get_method):
        super().__init__(list_widget)
        self._list = list_widget
        self._is_active = is_active
        self._get_method = get_method

    def _accumulating(self):
        get_method = getattr(self, '_get_method', None)
        if not callable(get_method):
            return False
        try:
            return get_method() in self._ACCUMULATING
        except Exception:
            return False

    @staticmethod
    def _is_data_item(item):
        if item is None:
            return False
        text = item.text()
        return text != '..' and not text.endswith('/')

    def eventFilter(self, obj, event):
        is_active = getattr(self, '_is_active', None)
        if not callable(is_active):
            return False
        try:
            active = is_active()
        except Exception:
            return False
        if not active:
            return False
        etype = event.type()
        if etype == QtCore.QEvent.MouseButtonPress:
            return self._on_click(event)
        if etype == QtCore.QEvent.KeyPress:
            return self._on_key(event)
        return False

    @staticmethod
    def _meaningful_modifiers(event):
        """Return the shift/ctrl/meta bits held, coerced to int.

        Mirrors ``_AccumulatingClickFilter``: raw ``modifiers() != NoModifier``
        comparisons are unreliable under PySide6 (a plain click can carry a
        stray flag, notably on macOS), so coerce to int and mask to the only
        modifiers we care about.  Returns ``(has_shift, has_toggle_mod)``."""
        try:
            mods = int(event.modifiers())
        except (TypeError, ValueError):
            return False, False
        shift_bit = int(QtCore.Qt.ShiftModifier)
        ctrl_bit = int(QtCore.Qt.ControlModifier)
        meta_bit = int(QtCore.Qt.MetaModifier)
        return bool(mods & shift_bit), bool(mods & (ctrl_bit | meta_bit))

    def _on_click(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return False
        list_widget = getattr(self, '_list', None)
        if list_widget is None:
            return False
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if has_shift:
            return False                  # let Qt handle shift range-select
        try:
            pos = event.position().toPoint()
        except AttributeError:            # Qt5 fallback
            pos = event.pos()
        item = list_widget.itemAt(pos)
        if not self._is_data_item(item):
            return False
        if not (has_toggle_mod or self._accumulating()):
            return False                  # Single plain click: Qt replace
        # Accumulating (or explicit ctrl/cmd-toggle): toggle this file in/out of
        # the overlay via the selection model (robust in ExtendedSelection).
        sm = list_widget.selectionModel()
        idx = list_widget.indexFromItem(item)
        sm.select(idx, QtCore.QItemSelectionModel.Toggle)
        sm.setCurrentIndex(idx, QtCore.QItemSelectionModel.NoUpdate)
        return True

    def _on_key(self, event):
        list_widget = getattr(self, '_list', None)
        if list_widget is None:
            return False
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if (not self._accumulating()
                or has_shift or has_toggle_mod
                or event.key() not in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down)):
            return False                  # Single / modified: Qt default browse
        step = -1 if event.key() == QtCore.Qt.Key_Up else 1
        row = list_widget.currentRow() + step
        while 0 <= row < list_widget.count():
            item = list_widget.item(row)
            if self._is_data_item(item):
                # Extend: add the newly-current file without clearing the rest,
                # so arrow-browsing builds the comparison set.
                item.setSelected(True)
                list_widget.setCurrentItem(
                    item, QtCore.QItemSelectionModel.NoUpdate)
                return True
            row += step
        return True                       # at an end / only dirs: consume


def scanlocked(func):
    """Decorator that acquires scan_lock before calling the wrapped method.

    If self.scan is not a LiveScan (e.g. during initialisation),
    the function is called without the lock rather than silently returning None.
    """
    def wrapper(self, *args, **kwargs):
        if isinstance(self.scan, LiveScan):
            with self.scan.scan_lock:
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


class staticWidget(QWidget):
    """Tab for integrating data collected by a scanning area detector.
    As of current version, only handles a single angle (2-theta).
    Displays raw images, stitched Q Chi arrays, and integrated I Q
    arrays. Also displays metadata and exposes parameters for
    controlling integration.

    children:
        displayframe: widget which handles displaying images and
            plotting data.
        h5viewer: Has a file explorer panel for loading scans, and
            a panel which shows images that are associated with the
            loaded scan. Has other file saving and loading functions
            as well as configuration saving and loading functions.
        integrator_thread: Not visible to user, but a sub-thread which
            handles integration to free resources for the gui
        integratorTree: Widget for setting the basic integration
            parameters. Also has buttons for starting integration.
        metawidget: Table wiget which displays metadata either for
            entire scan or individual image.

    attributes:
        frame: LiveFrame, currently loaded frame object
        frame_ids: List of LiveFrame indices currently loaded
        frames: Dictionary of currently loaded LiveFrames
        data_1d: Dictionary object holding all 1D data in memory
        data_2d: Dictionary object holding all 2D data in memory
        command_queue: Queue, used to send commands to wrangler
        dirname: str, absolute path of current directory for scan
        file_lock: mp.Condition, process safe lock
        fname: str, current data file name
        scan: LiveScan, current scan data
        timer: QTimer, currently unused but can be used for periodic
            functions.
        ui: Ui_Form, layout from qtdesigner

    methods:
        bai_1d: Sends signal to thread to start integrating 1d
        bai_2d:  Sends signal to thread to start integrating 2d
        clock: Unimplemented, used for periodic updates
        close: Handles cleanup prior to closing
        enable_integration: Sets enabled status of widgets related to
            integration
        first_frame, latest_frame, next_frame: Handle moving between
            different frames in the overall scan
        load_and_set: Combination of load and set methods. Also governs
            file explorer behavior in h5viewer.
        load_scan:
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_data_objects()
        self._init_ui()
        self._init_child_widgets()
        self._connect_signals()
        self._init_wranglers()
        self._strip_combo_checkmarks()
        self._init_defaults_and_timer()
        self.show()
        self.ui.wranglerFrame.activateWindow()

    def _strip_combo_checkmarks(self):
        """Polish every dropdown in the tab in one sweep:

        * Replace the item delegate with a plain QStyledItemDelegate so the
          popup shows the selection by highlight only — the default
          QComboBox delegate draws a current-item checkmark that QSS
          ``::indicator`` can't remove, and it clipped the longer names.
        * Widen the popup to its longest entry (the combo box itself stays
          compact in the toolbar) so options like "Image Viewer" / "Int 1D
          (XYE)" aren't truncated.

        Covers the mode combo (in wranglerStack), the 1D plot Single/Q-θ
        combos, the top-bar Scale/colormap, the 2D-unit combo, and the
        integrator unit combos — all are descendants here.
        """
        for combo in self.findChildren(QtWidgets.QComboBox):
            combo.setItemDelegate(QtWidgets.QStyledItemDelegate(combo))
            view = combo.view()
            try:
                view.setTextElideMode(QtCore.Qt.ElideNone)
                fm = combo.fontMetrics()
                widest = max(
                    (fm.horizontalAdvance(combo.itemText(i))
                     for i in range(combo.count())),
                    default=0,
                )
                if widest:
                    # + room for item padding and the popup scrollbar.
                    view.setMinimumWidth(widest + 44)
            except Exception:
                logger.debug("combo popup sizing skipped", exc_info=True)

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self):
        """Initialize data containers, file lock, and directory paths."""
        self.file_lock = threading.Condition()
        # Reentrant lock guarding concurrent access to data_1d / data_2d from
        # the GUI thread, integratorThread, and fileHandlerThread. Shared with
        # all child widgets and worker threads. Always the OUTER lock when
        # paired with scan.scan_lock (data_lock → scan_lock).
        self.data_lock = threading.RLock()
        # Scratch directory for working .nxs files (under the user's home).
        self.local_path = get_fname_dir()
        self.dirname = self.local_path

        self.fname = os.path.join(self.dirname, 'default.nxs')
        # J2: share ``file_lock`` with the scan so direct
        # LiveFrameSeries lazy loads use the same lock as the
        # wrangler's save paths.
        self.scan = LiveScan('null_main',
                               data_file=self.fname,
                               static=True,
                               file_lock=self.file_lock)
        self.frame = LiveFrame(static=True, gi=self.scan.gi)
        self.frame_ids = []
        self.frames = OrderedDict()
        self.publication_store = PublicationStore()
        # Live-browse caches.  Both reset at scan/viewer boundaries and are no
        # longer authoritative for normal Int 1D/2D readiness; the
        # PublicationStore is.  The 1D mirror is capped in scan mode to prevent
        # long live runs from accumulating every old IntegrationResult copy.
        # Viewer modes temporarily lift the 1D cap because data_1d is their row
        # table for XYE/NeXus previews, not the scan-display mirror.
        #
        # 2D: bounded at 40.  Each data_2d entry carries the full
        # ``map_raw`` detector image (~18 MB) plus the cake, so the cap is a
        # memory ceiling (~40 x 18 MB ≈ 0.7 GB), not a correctness limit.
        # 40 covers the recent-frame 2D live-browse window for most scans;
        # raising it is a straight RAM tradeoff.  Older-than-window 2D frames
        # are available after the run (or while Paused), not mid-run (the
        # writer-active freeze guard refuses to read the file being appended).
        self.data_1d = FixSizeOrderedDict(max=_DISPLAY_1D_CACHE_MAX)
        self.data_2d = FixSizeOrderedDict(max=_DISPLAY_2D_CACHE_MAX)

    def _set_1d_cache_limit(self, limit: int | None) -> None:
        """Switch the transitional 1D mirror between scan and viewer policy."""
        cache = getattr(self, "data_1d", None)
        if cache is None:
            return
        max_value = 0 if limit is None else int(limit)
        if hasattr(cache, "_max"):
            cache._max = max_value
        elif hasattr(cache, "max"):
            cache.max = max_value
        else:
            return
        if max_value > 0:
            while len(cache) > max_value:
                cache.popitem(False)

    def _init_ui(self):
        """Set up the main UI form and detector dialog."""
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.detector_dialog = QDialog()
        self.detector_widget = QCombo()
        self.detector = None

    def _init_child_widgets(self):
        """Create H5Viewer, DisplayFrame, IntegratorTree, and Metadata widgets."""
        # H5Viewer
        self.h5viewer = H5Viewer(self.file_lock, self.local_path, self.dirname,
                                 self.scan, self.frame, self.frame_ids, self.frames,
                                 self.data_1d, self.data_2d,
                                 self.ui.hdf5Frame, data_lock=self.data_lock,
                                 publication_store=self.publication_store)
        self.ui.hdf5Frame.setLayout(self.h5viewer.layout)
        self.h5viewer.update_scans()

        # DisplayFrame
        self.displayframe = displayFrameWidget(self.scan, self.frame,
                                               self.frame_ids, self.frames,
                                               self.data_1d, self.data_2d,
                                               parent=self.ui.middleFrame,
                                               data_lock=self.data_lock,
                                               publication_store=self.publication_store)
        # Back-ref so h5viewer.data_reset can re-arm display-side caches.
        self.h5viewer.displayframe = self.displayframe
        self.ui.middleFrame.setLayout(self.displayframe.ui.layout)

        # IntegratorTree
        self.integratorTree = integratorTree(
            self.scan, self.frame, self.file_lock,
            self.frames, self.frame_ids, self.data_1d, self.data_2d,
            data_lock=self.data_lock,
            publication_store=self.publication_store)
        # Stitch worker (Stitch 1D / Stitch 2D modes): a one-shot off-thread
        # reduction of the loaded scan, routed through the SAME run-state owner
        # (_enter/_exit_run_state) as a wrangler run or a reintegrate.
        self.stitch_thread = stitchThread(self.scan, parent=self)
        self.stitch_thread.started.connect(self._enter_run_state)
        self.stitch_thread.finished.connect(self.stitch_thread_finished)
        self.stitch_thread.errorSig.connect(self._on_stitch_error)
        # Default panel proportions: middle (image/plot) panels ~10% wider
        # than Qt's hint-based split (Vivek).  Applied via singleShot AFTER
        # the window has real geometry -- setSizes at __init__ ran before the
        # main window's resize() and got redistributed away.
        def _default_split():
            try:
                total = sum(self.ui.mainSplitter.sizes()) or 1000
                # Controls (right) and data-browser (left) columns start at the
                # SAME width; the middle display panels take the rest (Vivek).
                # The side columns are 10% narrower than before (0.28 -> 0.252)
                # so the central display isn't squished; the freed space goes to
                # the middle.  Min/max widths are untouched (set elsewhere) --
                # this only moves the default/initial split.  User-resizable via
                # the splitter (re-asserted only during the first-3s launch storm).
                self.ui.mainSplitter.setSizes(
                    [int(total * f) for f in (0.252, 0.496, 0.252)])
                self.ui.mainSplitter.setStretchFactor(1, 1)
                # Left column: the Tools card is now a compact 3-button panel, so
                # give it only a small share (~18%) and let the data browser take
                # the rest.  Stretch so a window resize grows the browser, not
                # Tools.
                ltotal = sum(self.ui.leftSplitter.sizes()) or 600
                self.ui.leftSplitter.setSizes(
                    [int(ltotal * 0.82), int(ltotal * 0.18)])
                self.ui.leftSplitter.setStretchFactor(0, 1)
                self.ui.leftSplitter.setStretchFactor(1, 0)
            except Exception:
                logger.debug("mainSplitter default sizing failed",
                             exc_info=True)
        # Re-assert through every resize for the first 3s after launch, then
        # never touch it again.  Timers lost to late window-manager resizes,
        # and gating on splitterMoved was unreliable (Qt can emit it from its
        # own redistribution during a native resize, which read as a user
        # drag and disabled the hook).  Time-gating is dumb but bulletproof:
        # no user drags the splitter within 3s of launch.
        import time as _time
        self._split_until = _time.monotonic() + 3.0
        self._apply_default_split = _default_split
        QtCore.QTimer.singleShot(0, _default_split)
        QtCore.QTimer.singleShot(1000, _default_split)
        QtCore.QTimer.singleShot(2500, _default_split)
        self.ui.integratorFrame.setLayout(self.integratorTree.ui.verticalLayout)
        if len(self.scan.frames.index) > 0:
            self.integratorTree.update()
        self.integratorTree.ui.raw_to_tif.hide()
        # TOOLS section: lift the integrator's bottom row (frame_3 = Calibrate /
        # Make Mask; raw_to_tif hidden) into the top tools bar.  Reparent the
        # WHOLE frame_3 as one self-contained widget (NOT its individual buttons
        # — plucking buttons out of frame_3's layout leaves a dangling layout
        # item that double-frees on teardown / segfaults).  The buttons keep
        # their clicked wiring + the _apply_integration_control_state enable refs.
        try:
            self.ui.toolsLayout.addWidget(self.integratorTree.ui.frame_3)
        except Exception:
            logger.debug("could not move Calibrate/Make Mask to the tools bar",
                         exc_info=True)
        # CONTROLS section: the single shared run-controls widget, installed into
        # the bottom controlsFrame.  set_wrangler attaches it to the active
        # wrangler (routing its signals) — it is never reparented on swap.  Its
        # mode-change drives the staticWidget-level reaction (viewer reset +
        # display clear + integration-control state) here, ONCE.
        from .ui.static_controls import StaticControls
        self.controls = StaticControls()
        self.ui.controlsLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.controlsLayout.addWidget(self.controls)
        # Hug the controls' own (snug, uniform-padded) content height and fix it,
        # so the bottom controls bar can't be resized by the splitter and has no
        # excess black space above/below the rows.  RECOMPUTED after profile /
        # run-row show-hide (set_wrangler, mode change) so it can't go stale
        # (slack in viewer modes where the run row is hidden, or clip if content
        # grows) -- see _fit_controls_height.
        self._fit_controls_height()
        self.controls.modeCombo.currentTextChanged.connect(
            self._on_processing_mode_changed)
        # Single owner of the shared Stop button: dispatch to whichever run is
        # active — a reintegrate (integrator thread) takes priority, else the
        # wrangler.  The wranglers no longer connect Stop directly, so a Stop
        # press during a reintegrate can't also trip the idle wrangler's stop()
        # side-effects (unchecking Live, command='stop', button morph).
        self.controls.stopButton.clicked.connect(self._on_stop_clicked)
        self.controls_v2 = None
        self._controls_v2_last_signature = None
        self._controls_v2_batch_refresh_deferred = False
        self._controls_v2_refresh_timer = Coalescer(
            250, mode="throttle", parent=self)
        self._controls_v2_refresh_timer.triggered.connect(
            self._refresh_controls_v2_profile_now)
        self._init_controls_v2_preview()
        self._configure_controls_v2_native_run_plan()
        # Reintegrate reuses the shared Batch toggle: Batch off -> live (per-frame,
        # the default); Batch on -> fast multicore.  The integrator reads it
        # through this provider at click time (no direct controls ref needed).
        self.integratorTree._reintegrate_batch_provider = (
            lambda: self.controls.batchButton.isChecked())
        # Restore the integration panel (units/pts/ranges/Auto flags/GI modes
        # + Advanced params) from the previous session; saved in close().
        try:
            from xdart.utils.session import load_session
            _integ = (load_session() or {}).get('integrator')
            if _integ and not self._controls_v2_enabled():
                self.integratorTree.restore_session_state(_integ)
        except Exception:
            logger.debug("integrator session restore failed", exc_info=True)
        self._restore_controls_v2_int_session_state()

        # Metadata
        self.metawidget = metadataWidget(self.scan, self.frame,
                                         self.frame_ids, self.frames,
                                         data_1d=self.data_1d,
                                         publication_store=self.publication_store,
                                         data_lock=self.data_lock)
        # Stage 4 (Direction A): the metadata table is no longer inline in the
        # bottom-left.  It opens on demand via the "Metadata" button, which
        # reparents this same metawidget into a popup dialog (see
        # _open_metadata_dialog).  The widget keeps every ctor reference
        # (scan/frame/frame_ids/frames/publication_store/data_lock) — they are
        # shared mutable objects, so it still refreshes from frame selection and
        # the publication store exactly as before.  The vacated metaFrame now
        # hosts a Tools placeholder for planned modules.
        self._metadata_dialog = None
        self._peak_fit_dialog = None
        self._phase_fit_dialog = None
        self._scan_plot_dialog = None
        # The fit dialog whose batch run is currently in flight (Peak or Phase).
        self._batch_dialog = None
        # Live analysis preview (analyzer framework Step 3): a latest-wins
        # background worker re-fits the newest frame while the dialog's "Live"
        # toggle is on.  Lazily created on first live request; generation gates
        # stale results.
        self._live_analysis_worker = None
        self._live_fit_gen = 0
        # Batch analysis: one worker fits every frame and streams params into the
        # dialog's embedded vs-frame trend (row 3).
        self._batch_analysis_worker = None
        # Set in close() before tearing the widget down — the analysis slots bail
        # on it so a worker signal queued just before teardown can't touch the
        # (about-to-be-destroyed) peak-fit dialog.
        self._tearing_down = False
        self._build_tools_placeholder()

    @staticmethod
    def _controls_v2_enabled() -> bool:
        value = os.environ.get("XDART_CONTROLS_PANEL_V2", "1")
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _init_controls_v2_preview(self) -> None:
        """Mount the Controls Panel V2 editor.

        The panel is visible by default on the V2 branch.  It renders real
        editable rows backed by native Controls V2 state, while the legacy
        widgets stay alive only for delegated actions and the Advanced inspector.
        Set ``XDART_CONTROLS_PANEL_V2=0`` to compare against the legacy panel.
        """
        if not self._controls_v2_enabled():
            return
        try:
            from .ui.controls_panel_v2 import ControlsPanelV2
            panel = ControlsPanelV2(self.ui.wranglerFrame)
            preview = QtWidgets.QScrollArea(self.ui.wranglerFrame)
            preview.setObjectName("controlsPanelV2Preview")
            preview.setWidgetResizable(True)
            preview.setFrameShape(QtWidgets.QFrame.NoFrame)
            preview.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            preview.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            preview.setMinimumHeight(0)
            preview.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            panel.analysisLaunchRequested.connect(
                self._on_controls_v2_analysis_launch)
            panel.controlActionRequested.connect(
                self._on_controls_v2_action)
            panel.fieldValueChanged.connect(
                self._on_controls_v2_field_changed)
            panel.fieldBrowseRequested.connect(
                self._on_controls_v2_field_browse)
            preview.setWidget(panel)
            self.ui.verticalLayout.insertWidget(0, preview, 1)
            self.ui.verticalLayout.setStretchFactor(preview, 1)
            self.ui.wranglerStack.hide()
            # The legacy integrator Calibrate/Make Mask row (frame_3, lifted into
            # toolsLayout) is redundant under V2 — the V2 producer buttons render
            # in the Experiment section and delegate clicks to these.  Hide it so
            # it doesn't duplicate them (the buttons stay alive for delegation).
            try:
                self.integratorTree.ui.frame_3.hide()
            except Exception:
                pass
            self.controls_v2_preview = preview
            self.controls_v2 = panel
            self._install_controls_v2_native_int_hooks()
            panel.set_processing_widget(self.ui.integratorFrame, visible=False)
            self._refresh_controls_v2_profile(immediate=True)
        except Exception:
            self.controls_v2_preview = None
            self.controls_v2 = None
            logger.debug("Controls Panel V2 preview mount failed",
                         exc_info=True)

    def _install_controls_v2_native_int_hooks(self) -> None:
        """Make V2 native Int state the provider for legacy-owned actions."""

        integrator = getattr(self, "integratorTree", None)
        if integrator is None:
            return
        integrator._controls_v2_native_args = True
        integrator.get_gi_config = self._controls_v2_gi_config
        integrator.get_threshold_config = self._controls_v2_threshold_config
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_hydrate_advanced_from_scan()

    def _on_controls_v2_analysis_launch(self, tool) -> None:
        """Open the existing analysis popup for a V2 launcher intent."""
        if tool == AnalysisTool.PEAK_FIT:
            self._open_peak_fit_dialog()
        elif tool == AnalysisTool.PHASE_FIT:
            self._open_phase_fit_dialog()
        elif tool in (AnalysisTool.SCAN_PLOT, AnalysisTool.ROI_STATS):
            self._open_scan_plot_dialog()
        else:
            QMessageBox.information(
                self, "Tool not ready",
                "This analysis tool is scaffolded but not production-ready yet.")

    def _on_controls_v2_action(self, action) -> None:
        """Route Controls V2 preview actions through existing production hooks."""
        if action == ControlAction.CHOOSE_SOURCE:
            self._controls_v2_choose_source()
        elif action == ControlAction.CHOOSE_PROJECT:
            self._controls_v2_choose_project()
        elif action == ControlAction.CHOOSE_OUTPUT:
            self._controls_v2_choose_output()
        elif action == ControlAction.CALIBRATE:
            import time as _time
            calib_started = _time.time()
            self._controls_v2_click_integrator_button("pyfai_calib")
            # pyFAI-calib2 runs as a BLOCKING external subprocess, so by here it
            # has closed.  It can't report the saved path back, so offer to
            # adopt any .poni it just wrote (confirmation popup).
            self._autofill_poni_after_calibrate(calib_started)
        elif action == ControlAction.MAKE_MASK:
            self._controls_v2_click_integrator_button("get_mask")
        elif action == ControlAction.REINTEGRATE_1D:
            self._controls_v2_click_integrator_button("reintegrate1D")
        elif action == ControlAction.REINTEGRATE_2D:
            self._controls_v2_click_integrator_button("reintegrate2D")
        elif action == ControlAction.ADVANCED_PROCESSING:
            self._commit_controls_v2_pending_edits()
            self._show_integration_advanced()
        elif action == ControlAction.REFINE_GEOMETRY:
            QMessageBox.information(
                self, "Refine geometry",
                "Geometry refinement is scaffolded and will be enabled after "
                "the real-data GUI gate lands.")
        else:
            QMessageBox.information(
                self, "Action not ready",
                "This control is scaffolded but not production-ready yet.")
        self._refresh_controls_v2_profile()

    def _controls_v2_click_integrator_button(self, button_name: str) -> None:
        if button_name in {"reintegrate1D", "reintegrate2D"}:
            self._apply_controls_v2_native_int_state(
                commit_pending=True,
                push_integrator=True,
            )
            self._configure_controls_v2_native_run_plan(commit_pending=False)
        else:
            self._commit_controls_v2_pending_edits()
        button = getattr(getattr(self.integratorTree, "ui", None), button_name, None)
        if button is None:
            return
        click = getattr(button, "click", None)
        if callable(click):
            click()

    def _apply_controls_v2_field_value(self, path, value) -> bool:
        if self._set_controls_v2_native_int_field(path, value):
            return True
        param = self._controls_v2_param(tuple(path))
        if param is None:
            return False
        try:
            current = param.value()
            new_value = coerce_control_edit_value(current, value)
            if current != new_value:
                param.setValue(new_value)
        except Exception:
            logger.debug("Controls Panel V2 field update failed for %s", path,
                         exc_info=True)
        return True

    def _commit_controls_v2_pending_edits(self) -> None:
        panel = getattr(self, "controls_v2", None)
        get_edits = getattr(panel, "current_form_edits", None)
        if not callable(get_edits):
            return
        try:
            edits = get_edits()
        except Exception:
            logger.debug("Controls Panel V2 pending edit harvest failed",
                         exc_info=True)
            return
        for edit in edits:
            self._apply_controls_v2_field_value(edit.path, edit.value)
        if edits:
            self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_param(self, path):
        wrangler = getattr(self, "wrangler", None)
        params = getattr(wrangler, "parameters", None)
        if params is None:
            return None
        try:
            return params.child(*path)
        except Exception:
            return None

    def _controls_v2_field_paths(self):
        return BOUND_CONTROL_PATHS

    def _controls_v2_field_values(self):
        values = {}
        for path in self._controls_v2_field_paths():
            if path in INTEGRATOR_BACKED_CONTROL_PATHS:
                continue
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                values[path] = param.value()
            except Exception:
                pass
        values.update(self._controls_v2_native_int_values())
        return values

    def _controls_v2_field_choices(self):
        choices = {}
        for path in self._controls_v2_field_paths():
            if path in INTEGRATOR_BACKED_CONTROL_PATHS:
                continue
            param = self._controls_v2_param(path)
            if param is None:
                continue
            opts = getattr(param, "opts", {}) or {}
            limits = opts.get("limits", None)
            if limits is None:
                limits = opts.get("values", None)
            if isinstance(limits, dict):
                vals = tuple(str(v) for v in limits.values())
            elif isinstance(limits, (list, tuple, set)):
                vals = tuple(str(v) for v in limits)
            else:
                vals = ()
            if vals:
                choices[path] = vals
        choices.update(self._controls_v2_native_int_choices())
        return choices

    @staticmethod
    def _controls_v2_number_text(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "" if value is None else str(value)
        if number.is_integer():
            return str(int(number))
        return str(number)

    @staticmethod
    def _controls_v2_float(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _controls_v2_int(value, default=0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _controls_v2_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
        return bool(value)

    def _controls_v2_threshold_config(self):
        state = getattr(self, "_controls_v2_threshold_state", None)
        scan = getattr(self, "scan", None)
        if not isinstance(state, dict):
            state = {
                "apply_threshold": bool(getattr(scan, "apply_threshold", False)),
                "threshold_min": self._controls_v2_float(
                    getattr(scan, "threshold_min", 0.0), 0.0),
                "threshold_max": self._controls_v2_float(
                    getattr(scan, "threshold_max", 0.0), 0.0),
                "mask_saturation": bool(getattr(scan, "mask_sentinel", True)),
            }
            self._controls_v2_threshold_state = state
        return ThresholdSaturationConfig(
            apply_threshold=bool(state.get("apply_threshold", False)),
            threshold_min=self._controls_v2_float(
                state.get("threshold_min", 0.0), 0.0),
            threshold_max=self._controls_v2_float(
                state.get("threshold_max", 0.0), 0.0),
            mask_saturation=bool(state.get("mask_saturation", True)),
        )

    def _controls_v2_set_threshold_field(self, path, value) -> None:
        cfg = self._controls_v2_threshold_config()
        state = {
            "apply_threshold": bool(cfg.apply_threshold),
            "threshold_min": cfg.threshold_min,
            "threshold_max": cfg.threshold_max,
            "mask_saturation": bool(cfg.mask_saturation),
        }
        if path == ("Mask", "Threshold"):
            state["apply_threshold"] = self._controls_v2_bool(value)
        elif path == ("Mask", "min"):
            state["threshold_min"] = self._controls_v2_float(value, 0.0)
        elif path == ("Mask", "max"):
            state["threshold_max"] = self._controls_v2_float(value, 0.0)
        elif path == ("MaskSat", "mask_sentinel"):
            state["mask_saturation"] = self._controls_v2_bool(value)
        if path in {("Mask", "min"), ("Mask", "max")} and (
            state["threshold_min"] != 0.0 or state["threshold_max"] != 0.0
        ):
            state["apply_threshold"] = True
        self._controls_v2_threshold_state = state
        scan = getattr(self, "scan", None)
        if scan is not None:
            for attr, key in (
                ("apply_threshold", "apply_threshold"),
                ("threshold_min", "threshold_min"),
                ("threshold_max", "threshold_max"),
                ("mask_sentinel", "mask_saturation"),
            ):
                try:
                    setattr(scan, attr, state[key])
                except Exception:
                    pass

    def _controls_v2_default_gi_motor(self) -> str:
        choices = self._controls_v2_native_int_choices().get(("GI", "th_motor"), ())
        for preferred in ("th", "eta", "theta", "gonth", "halpha"):
            for choice in choices:
                if str(choice).lower() == preferred:
                    return str(choice)
        return str(choices[0]) if choices else "Manual"

    def _controls_v2_gi_config(self) -> dict:
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        gic = dict(getattr(scan, "gi_config", {}) or {})
        gi = bool(getattr(scan, "gi", False)) or bool(gic)
        motor = gic.get("incidence_motor", None)
        if motor is None:
            motor = getattr(scan, "incidence_motor", None)
        th_val = gic.get("th_val", 0.1)
        if motor is None or str(motor) == "":
            motor = self._controls_v2_default_gi_motor()
        else:
            try:
                th_val = float(motor)
                motor = "Manual"
            except (TypeError, ValueError):
                motor = str(motor)
        sample_orientation = gic.get("sample_orientation", None)
        if sample_orientation is None:
            sample_orientation = getattr(scan, "sample_orientation", None)
        if sample_orientation is None:
            sample_orientation = 4
        tilt_angle = gic.get("tilt_angle", None)
        if tilt_angle is None:
            tilt_angle = getattr(scan, "tilt_angle", None)
        if tilt_angle is None:
            tilt_angle = 0.0
        return {
            "gi": gi,
            "sample_orientation": self._controls_v2_int(sample_orientation, 4),
            "tilt_angle": self._controls_v2_float(tilt_angle, 0.0),
            "incidence_motor": str(motor or "Manual"),
            "th_val": self._controls_v2_float(th_val, 0.1),
            "gi_mode_1d": str(a1.get("gi_mode_1d", "q_total")),
            "gi_mode_2d": str(a2.get("gi_mode_2d", "qip_qoop")),
        }

    def _controls_v2_apply_gi_config_to_scan(self, cfg=None) -> None:
        scan = getattr(self, "scan", None)
        if scan is None:
            return
        if cfg is None:
            cfg = self._controls_v2_gi_config()
        scan.gi = bool(cfg["gi"])
        if not cfg["gi"]:
            scan.gi_config = {}
            return
        scan.gi_config = {
            "gi_mode_1d": str(cfg["gi_mode_1d"]),
            "gi_mode_2d": str(cfg["gi_mode_2d"]),
            "incidence_motor": str(cfg["incidence_motor"] or ""),
            "th_val": float(cfg["th_val"] or 0.0),
            "tilt_angle": float(cfg["tilt_angle"] or 0.0),
            "sample_orientation": int(cfg["sample_orientation"] or 4),
        }
        incidence = (
            str(cfg["th_val"])
            if cfg["incidence_motor"] == "Manual"
            else str(cfg["incidence_motor"] or "")
        )
        scan.incidence_motor = incidence
        scan.th_mtr = incidence
        scan.sample_orientation = int(cfg["sample_orientation"] or 4)
        scan.tilt_angle = float(cfg["tilt_angle"] or 0.0)

    def _controls_v2_set_gi_field(self, leaf: str, value) -> None:
        scan = getattr(self, "scan", None)
        if scan is None:
            return
        cfg = self._controls_v2_gi_config()
        if leaf == "Grazing":
            cfg["gi"] = self._controls_v2_bool(value)
        elif leaf == "th_motor":
            cfg["incidence_motor"] = str(value)
        elif leaf == "th_val":
            cfg["th_val"] = self._controls_v2_float(value, cfg.get("th_val", 0.1))
        elif leaf == "sample_orientation":
            cfg["sample_orientation"] = self._controls_v2_int(value, 4)
        elif leaf == "tilt_angle":
            cfg["tilt_angle"] = self._controls_v2_float(value, 0.0)
        scan.gi = bool(cfg["gi"])
        if scan.gi:
            a1, a2 = self._controls_v2_scan_int_args()
            a1.setdefault("gi_mode_1d", "q_total")
            a2.setdefault("gi_mode_2d", "qip_qoop")
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"
        scan.gi_config = {
            "gi_mode_1d": str(cfg["gi_mode_1d"]),
            "gi_mode_2d": str(cfg["gi_mode_2d"]),
            "incidence_motor": str(cfg["incidence_motor"] or ""),
            "th_val": float(cfg["th_val"] or 0.0),
            "tilt_angle": float(cfg["tilt_angle"] or 0.0),
            "sample_orientation": int(cfg["sample_orientation"] or 4),
        } if scan.gi else {}
        self._controls_v2_apply_gi_config_to_scan()

    @staticmethod
    def _controls_v2_apply_native_int_snapshot_to_scan(
        snapshot: dict,
        scan,
    ) -> None:
        if scan is None or not isinstance(snapshot, dict):
            return
        lock = getattr(scan, "scan_lock", None)

        def _apply():
            scan.bai_1d_args = copy.deepcopy(snapshot.get("bai_1d_args", {}) or {})
            scan.bai_2d_args = copy.deepcopy(snapshot.get("bai_2d_args", {}) or {})
            scan.gi = bool(snapshot.get("gi", False))
            scan.gi_config = copy.deepcopy(snapshot.get("gi_config", {}) or {})
            for attr in (
                "incidence_motor",
                "th_mtr",
                "sample_orientation",
                "tilt_angle",
            ):
                if attr in snapshot:
                    setattr(scan, attr, copy.deepcopy(snapshot[attr]))

        if lock is None:
            _apply()
        else:
            with lock:
                _apply()

    def _controls_v2_apply_snapshot_to_scan(self, snapshot: dict, scan=None) -> None:
        scan = scan if scan is not None else getattr(self, "scan", None)
        self._controls_v2_apply_native_int_snapshot_to_scan(snapshot, scan)

    def _controls_v2_push_threshold_to_integrator(self) -> None:
        thread = getattr(getattr(self, "integratorTree", None),
                         "integrator_thread", None)
        if thread is not None:
            thread.threshold_config = self._controls_v2_threshold_config()

    def _apply_controls_v2_native_int_state(
        self,
        *,
        commit_pending: bool = True,
        push_wrangler: bool = False,
        push_integrator: bool = False,
    ) -> None:
        """Apply native V2 Int/GI/threshold state to the scan and consumers."""

        if commit_pending:
            self._commit_controls_v2_pending_edits()
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_apply_gi_config_to_scan()
        if push_wrangler:
            self._push_threshold_to_wrangler()
            self._push_gi_to_wrangler()
        if push_integrator:
            self._controls_v2_push_threshold_to_integrator()

    def _controls_v2_native_int_snapshot(self) -> dict:
        scan = getattr(self, "scan", None)
        if scan is None:
            return {}
        return {
            "bai_1d_args": copy.deepcopy(
                getattr(scan, "bai_1d_args", {}) or {}
            ),
            "bai_2d_args": copy.deepcopy(
                getattr(scan, "bai_2d_args", {}) or {}
            ),
            "gi": bool(getattr(scan, "gi", False)),
            "gi_config": copy.deepcopy(getattr(scan, "gi_config", {}) or {}),
            "incidence_motor": copy.deepcopy(
                getattr(scan, "incidence_motor", None)
            ),
            "th_mtr": copy.deepcopy(getattr(scan, "th_mtr", None)),
            "sample_orientation": copy.deepcopy(
                getattr(scan, "sample_orientation", None)
            ),
            "tilt_angle": copy.deepcopy(getattr(scan, "tilt_angle", None)),
        }

    @staticmethod
    def _controls_v2_native_int_snapshot_key(value):
        if isinstance(value, dict):
            return tuple(
                (str(key), staticWidget._controls_v2_native_int_snapshot_key(val))
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                staticWidget._controls_v2_native_int_snapshot_key(val)
                for val in value
            )
        if isinstance(value, set):
            return tuple(
                sorted(
                    staticWidget._controls_v2_native_int_snapshot_key(val)
                    for val in value
                )
            )
        try:
            hash(value)
        except TypeError:
            tolist = getattr(value, "tolist", None)
            if callable(tolist):
                return staticWidget._controls_v2_native_int_snapshot_key(
                    tolist()
                )
            return repr(value)
        return value

    def _controls_v2_scan_int_args(self):
        scan = getattr(self, "scan", None)
        if scan is None:
            return {}, {}
        lock = getattr(scan, "scan_lock", None)

        def _ensure():
            if not isinstance(getattr(scan, "bai_1d_args", None), dict):
                scan.bai_1d_args = {}
            if not isinstance(getattr(scan, "bai_2d_args", None), dict):
                scan.bai_2d_args = {}
            return scan.bai_1d_args, scan.bai_2d_args

        if lock is None:
            return _ensure()
        with lock:
            return _ensure()

    def _controls_v2_ensure_native_int_defaults(self) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        defaults_1d = {
            "unit": "q_A^-1",
            "numpoints": 3000,
            "radial_range": None,
            "azimuth_range": None,
            "correctSolidAngle": True,
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            "polarization_factor": None,
            "method": "csr",
            "safe": True,
        }
        defaults_2d = {
            "unit": "q_A^-1",
            "npt_rad": 500,
            "npt_azim": 500,
            "radial_range": None,
            "azimuth_range": None,
            "correctSolidAngle": True,
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            "polarization_factor": None,
            "method": "csr",
            "safe": True,
        }
        for key, value in defaults_1d.items():
            a1.setdefault(key, copy.deepcopy(value))
        for key, value in defaults_2d.items():
            a2.setdefault(key, copy.deepcopy(value))
        if bool(getattr(getattr(self, "scan", None), "gi", False)):
            a1.setdefault("gi_mode_1d", "q_total")
            a2.setdefault("gi_mode_2d", "qip_qoop")
            # GI integration is Q-space only in this panel.
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"

    def _controls_v2_unit_display(self, unit: object, *, dim: str = "1d") -> str:
        code = str(unit or "q_A^-1")
        idx = Units_dict_inv.get(code, 0)
        if dim == "2d" and idx >= 2:
            idx = 0
        try:
            return Units[idx]
        except Exception:
            return Units[0]

    def _controls_v2_unit_code(self, text: object, *, dim: str = "1d") -> str:
        value = str(text or "").strip()
        if value in Units_dict:
            code = Units_dict[value]
        elif value in Units_dict_inv:
            code = value
        elif value.startswith("2") or "2θ" in value or "2th" in value.lower():
            code = "2th_deg"
        elif "chi" in value.lower() or "χ" in value:
            code = "chi_deg"
        else:
            code = "q_A^-1"
        if dim == "2d" and code == "chi_deg":
            code = "q_A^-1"
        return code

    def _controls_v2_axis_display(self, root: str) -> str:
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        if bool(getattr(scan, "gi", False)):
            if root == "Int1D":
                mode = a1.get("gi_mode_1d", "q_total")
                return GI_LABELS_1D[GI_MODES_1D.index(mode)] if mode in GI_MODES_1D else GI_LABELS_1D[0]
            mode = a2.get("gi_mode_2d", "qip_qoop")
            return GI_LABELS_2D[GI_MODES_2D.index(mode)] if mode in GI_MODES_2D else GI_LABELS_2D[0]
        if root == "Int1D":
            return self._controls_v2_unit_display(a1.get("unit"), dim="1d")
        return "2θ-χ" if a2.get("unit") == "2th_deg" else "Q-χ"

    def _controls_v2_axis_to_native(self, root: str, value: object) -> None:
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        text = str(value or "")
        if bool(getattr(scan, "gi", False)):
            if root == "Int1D":
                try:
                    a1["gi_mode_1d"] = GI_MODES_1D[GI_LABELS_1D.index(text)]
                except ValueError:
                    a1["gi_mode_1d"] = "q_total"
                if self._controls_v2_npts_oop_visible():
                    a1.setdefault("npt_oop", int(a1.get("numpoints", 3000)))
            else:
                try:
                    a2["gi_mode_2d"] = GI_MODES_2D[GI_LABELS_2D.index(text)]
                except ValueError:
                    a2["gi_mode_2d"] = "qip_qoop"
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"
            return
        if root == "Int1D":
            a1["unit"] = self._controls_v2_unit_code(text, dim="1d")
        else:
            a2["unit"] = "2th_deg" if text.startswith("2") else "q_A^-1"

    def _controls_v2_default_range(self, root: str, axis: str):
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        gi = bool(getattr(scan, "gi", False))
        if root == "Int1D":
            if gi:
                mode = a1.get("gi_mode_1d", "q_total")
                if axis == "radial":
                    return (-5.0, 5.0) if mode == "exit_angle" else (
                        (-10.0, 10.0) if mode in {"q_ip", "q_oop"} else (0.0, 5.0)
                    )
                if mode in {"q_ip", "q_oop"}:
                    return (0.0, 5.0)
                if mode == "exit_angle":
                    return (0.0, 90.0)
                return (-180.0, 180.0)
            if axis == "radial":
                return (0.0, 90.0) if a1.get("unit") == "2th_deg" else (0.0, 5.0)
            return (-180.0, 180.0)
        if gi:
            mode = a2.get("gi_mode_2d", "qip_qoop")
            if axis == "radial":
                return (-5.0, 5.0) if mode == "exit_angles" else (
                    (-10.0, 10.0) if mode == "qip_qoop" else (0.0, 5.0)
                )
            if mode == "qip_qoop":
                return (0.0, 5.0)
            if mode == "exit_angles":
                return (0.0, 90.0)
            return (-180.0, 180.0)
        if axis == "radial":
            return (0.0, 90.0) if a2.get("unit") == "2th_deg" else (0.0, 5.0)
        return (-180.0, 180.0)

    def _controls_v2_range_value(self, root: str, axis: str):
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = args.get(key)
        return value if value is not None else self._controls_v2_default_range(root, axis)

    def _controls_v2_set_range_auto(self, root: str, axis: str, auto: bool) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = None if auto else self._controls_v2_range_value(root, axis)
        args[key] = value
        if root == "Int2D" and bool(getattr(getattr(self, "scan", None), "gi", False)):
            alt = "x_range" if axis == "radial" else "y_range"
            if value is None:
                args.pop(alt, None)
            else:
                args[alt] = value
        if root == "Int1D" and axis == "azimuth" and self._controls_v2_npts_oop_visible():
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_set_range_bound(
        self,
        root: str,
        axis: str,
        bound: str,
        value: object,
    ) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        low, high = self._controls_v2_range_value(root, axis)
        number = self._controls_v2_float(value, low if bound == "low" else high)
        if bound == "low":
            low = number
        else:
            high = number
        args[key] = (float(low), float(high))
        if root == "Int2D" and bool(getattr(getattr(self, "scan", None), "gi", False)):
            args["x_range" if axis == "radial" else "y_range"] = args[key]
        if root == "Int1D" and axis == "azimuth" and self._controls_v2_npts_oop_visible():
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_npts_oop_visible(self) -> bool:
        scan = getattr(self, "scan", None)
        if not bool(getattr(scan, "gi", False)):
            return False
        a1, _ = self._controls_v2_scan_int_args()
        return (
            a1.get("gi_mode_1d", "q_total") != "q_total"
            or a1.get("azimuth_range") is not None
        )

    def _controls_v2_integrator_parameter(self, spec):
        integrator = getattr(self, "integratorTree", None)
        if integrator is None or not getattr(spec, "parameter_name", ""):
            return None
        tree_name = {
            "1d": "bai_1d_pars",
            "2d": "bai_2d_pars",
        }.get(spec.parameter_group)
        tree = getattr(integrator, tree_name, None)
        if tree is None:
            return None
        try:
            return tree.child(spec.parameter_name)
        except Exception:
            return None

    def _controls_v2_advanced_value(self, root: str, leaf: str):
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        if leaf == "apply_polarization":
            return args.get("polarization_factor") is not None
        if leaf == "polarization_factor":
            value = args.get("polarization_factor")
            return 0.0 if value is None else value
        defaults = {
            "correctSolidAngle": True,
            "method": "csr",
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            "safe": True,
        }
        return args.get(leaf, defaults.get(leaf, ""))

    def _controls_v2_native_int_values(self):
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        gi_cfg = self._controls_v2_gi_config()
        threshold = self._controls_v2_threshold_config()
        r1 = self._controls_v2_range_value("Int1D", "radial")
        z1 = self._controls_v2_range_value("Int1D", "azimuth")
        r2 = self._controls_v2_range_value("Int2D", "radial")
        z2 = self._controls_v2_range_value("Int2D", "azimuth")

        values = {
            ("GI", "Grazing"): bool(gi_cfg["gi"]),
            ("GI", "th_motor"): str(gi_cfg["incidence_motor"]),
            ("GI", "th_val"): self._controls_v2_number_text(gi_cfg["th_val"]),
            ("GI", "sample_orientation"): int(gi_cfg["sample_orientation"]),
            ("GI", "tilt_angle"): self._controls_v2_number_text(gi_cfg["tilt_angle"]),
            ("Mask", "Threshold"): bool(threshold.apply_threshold),
            ("Mask", "min"): self._controls_v2_number_text(threshold.threshold_min),
            ("Mask", "max"): self._controls_v2_number_text(threshold.threshold_max),
            ("MaskSat", "mask_sentinel"): bool(threshold.mask_saturation),
            ("Int1D", "unit"): self._controls_v2_unit_display(a1.get("unit"), dim="1d"),
            ("Int1D", "axis"): self._controls_v2_axis_display("Int1D"),
            ("Int1D", "points"): str(int(a1.get("numpoints", 3000))),
            ("Int1D", "radial_auto"): a1.get("radial_range") is None,
            ("Int1D", "radial_low"): self._controls_v2_number_text(r1[0]),
            ("Int1D", "radial_high"): self._controls_v2_number_text(r1[1]),
            ("Int1D", "azim_auto"): a1.get("azimuth_range") is None,
            ("Int1D", "azim_low"): self._controls_v2_number_text(z1[0]),
            ("Int1D", "azim_high"): self._controls_v2_number_text(z1[1]),
            ("Int2D", "unit"): self._controls_v2_unit_display(a2.get("unit"), dim="2d"),
            ("Int2D", "axis"): self._controls_v2_axis_display("Int2D"),
            ("Int2D", "radial_points"): str(int(a2.get("npt_rad", 500))),
            ("Int2D", "azim_points"): str(int(a2.get("npt_azim", 500))),
            ("Int2D", "radial_auto"): a2.get("radial_range") is None,
            ("Int2D", "radial_low"): self._controls_v2_number_text(r2[0]),
            ("Int2D", "radial_high"): self._controls_v2_number_text(r2[1]),
            ("Int2D", "azim_auto"): a2.get("azimuth_range") is None,
            ("Int2D", "azim_low"): self._controls_v2_number_text(z2[0]),
            ("Int2D", "azim_high"): self._controls_v2_number_text(z2[1]),
        }
        if bool(gi_cfg["gi"]):
            values[("Int1D", "gi_mode")] = a1.get("gi_mode_1d", "q_total")
            values[("Int2D", "gi_mode")] = a2.get("gi_mode_2d", "qip_qoop")
        if self._controls_v2_npts_oop_visible():
            values[("Int1D", "points_oop")] = str(
                int(a1.get("npt_oop", a1.get("numpoints", 3000)))
            )
        for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
            if not spec.parameter_name:
                continue
            values[spec.path] = self._controls_v2_advanced_value(
                spec.path[0], spec.path[1]
            )
        return values

    def _controls_v2_native_int_choices(self):
        integrator = getattr(self, "integratorTree", None)
        ui = getattr(integrator, "ui", None)
        gi = bool(getattr(getattr(self, "scan", None), "gi", False))

        def _combo_choices(name):
            combo = getattr(ui, name, None)
            if combo is None:
                return ()
            return tuple(combo.itemText(i) for i in range(combo.count()))

        choices = {
            ("Int1D", "unit"): tuple(Units),
            ("Int2D", "unit"): tuple(Units[:2]),
            ("Int1D", "axis"): tuple(GI_LABELS_1D if gi else Units),
            ("Int2D", "axis"): tuple(GI_LABELS_2D if gi else ("Q-χ", "2θ-χ")),
            ("GI", "th_motor"): _combo_choices("gi_motor") or ("Manual", "th"),
        }
        for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
            if spec.kind.value == "combo" and spec.parameter_name:
                param = self._controls_v2_integrator_parameter(spec)
                opts = getattr(param, "opts", {}) or {}
                vals = opts.get("limits", None)
                if vals is None:
                    vals = opts.get("values", None)
                if isinstance(vals, dict):
                    choices[spec.path] = tuple(str(v) for v in vals.values())
                elif isinstance(vals, (list, tuple, set)):
                    choices[spec.path] = tuple(str(v) for v in vals)
        return {path: vals for path, vals in choices.items() if vals}

    def _controls_v2_sync_advanced_parameter(self, path) -> None:
        spec = next(
            (spec for spec in INTEGRATOR_BACKED_CONTROL_SPECS
             if spec.path == tuple(path) and spec.parameter_name),
            None,
        )
        if spec is None:
            return
        param = self._controls_v2_integrator_parameter(spec)
        if param is None:
            return
        value = self._controls_v2_advanced_value(spec.path[0], spec.path[1])
        if spec.path[1] == "polarization_factor" and value is None:
            value = 0.0
        try:
            if param.value() != value:
                param.setValue(value)
        except Exception:
            logger.debug("Controls Panel V2 advanced mirror failed for %s",
                         path, exc_info=True)

    def _controls_v2_hydrate_advanced_from_scan(self) -> None:
        integrator = getattr(self, "integratorTree", None)
        if integrator is None:
            return
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        keys = {
            "correctSolidAngle",
            "dummy",
            "delta_dummy",
            "chi_offset",
            "polarization_factor",
            "method",
            "safe",
        }
        try:
            integrator._args_to_params(
                {k: v for k, v in a1.items() if k in keys},
                integrator.bai_1d_pars,
                dim="1D",
            )
            integrator._args_to_params(
                {k: v for k, v in a2.items() if k in keys},
                integrator.bai_2d_pars,
                dim="2D",
            )
        except Exception:
            logger.debug("Controls Panel V2 advanced hydrate failed",
                         exc_info=True)

    def _set_controls_v2_native_int_field(self, path, value) -> bool:
        path = tuple(path)
        if path not in INTEGRATOR_BACKED_CONTROL_PATHS:
            return False
        root = path[0]
        leaf = path[1] if len(path) > 1 else ""
        if root == "GI":
            self._controls_v2_set_gi_field(leaf, value)
            return True
        if root in {"Mask", "MaskSat"}:
            self._controls_v2_set_threshold_field(path, value)
            return True
        if root not in {"Int1D", "Int2D"}:
            return True
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        if leaf == "unit":
            args["unit"] = self._controls_v2_unit_code(
                value, dim="1d" if root == "Int1D" else "2d")
        elif leaf == "axis":
            self._controls_v2_axis_to_native(root, value)
        elif leaf == "points":
            args["numpoints"] = self._controls_v2_int(value, 3000)
            if self._controls_v2_npts_oop_visible():
                args.setdefault("npt_oop", args["numpoints"])
        elif leaf == "points_oop":
            args["npt_oop"] = self._controls_v2_int(
                value, args.get("numpoints", 3000))
        elif leaf == "radial_points":
            args["npt_rad"] = self._controls_v2_int(value, 500)
        elif leaf == "azim_points":
            args["npt_azim"] = self._controls_v2_int(value, 500)
        elif leaf == "radial_auto":
            self._controls_v2_set_range_auto(root, "radial", self._controls_v2_bool(value))
        elif leaf == "azim_auto":
            self._controls_v2_set_range_auto(root, "azimuth", self._controls_v2_bool(value))
        elif leaf == "radial_low":
            self._controls_v2_set_range_bound(root, "radial", "low", value)
        elif leaf == "radial_high":
            self._controls_v2_set_range_bound(root, "radial", "high", value)
        elif leaf == "azim_low":
            self._controls_v2_set_range_bound(root, "azimuth", "low", value)
        elif leaf == "azim_high":
            self._controls_v2_set_range_bound(root, "azimuth", "high", value)
        elif leaf == "apply_polarization":
            if self._controls_v2_bool(value):
                args["polarization_factor"] = self._controls_v2_float(
                    args.get("polarization_factor"), 0.0)
            else:
                args["polarization_factor"] = None
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf == "polarization_factor":
            args["polarization_factor"] = self._controls_v2_float(value, 0.0)
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf in {"correctSolidAngle", "safe"}:
            args[leaf] = self._controls_v2_bool(value)
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf in {"dummy", "delta_dummy", "chi_offset"}:
            args[leaf] = self._controls_v2_float(value, args.get(leaf, 0.0))
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf == "method":
            args["method"] = str(value)
            self._controls_v2_sync_advanced_parameter(path)
        return True

    def _apply_controls_v2_run_state(self) -> dict:
        """Apply native Controls V2 Int state and snapshot args for this run."""
        self._apply_controls_v2_native_int_state(
            commit_pending=True,
            push_wrangler=True,
        )
        args = {
            'bai_1d_args': copy.deepcopy(
                getattr(self.scan, 'bai_1d_args', {}) or {}
            ),
            'bai_2d_args': copy.deepcopy(
                getattr(self.scan, 'bai_2d_args', {}) or {}
            ),
        }
        self.wrangler.scan_args = args
        return args

    def _controls_v2_int_session_state(self) -> dict:
        """Native Controls V2 Int state used for run/reintegrate plans."""

        if not getattr(self, "_tearing_down", False):
            self._commit_controls_v2_pending_edits()
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_apply_gi_config_to_scan()

        cfg = self._controls_v2_threshold_config()
        threshold = {
            "apply_threshold": bool(cfg.apply_threshold),
            "threshold_min": cfg.threshold_min,
            "threshold_max": cfg.threshold_max,
            "mask_saturation": bool(cfg.mask_saturation),
        }

        scan = getattr(self, "scan", None)
        return {
            "bai_1d_args": copy.deepcopy(
                getattr(scan, "bai_1d_args", {}) or {}
            ),
            "bai_2d_args": copy.deepcopy(
                getattr(scan, "bai_2d_args", {}) or {}
            ),
            "gi_config": copy.deepcopy(getattr(scan, "gi_config", {}) or {}),
            "gi": bool(getattr(scan, "gi", False)),
            "threshold_config": threshold,
        }

    def _restore_controls_v2_int_session_state(self) -> None:
        """Restore the native Controls V2 Int blob after the legacy fallback."""

        try:
            from xdart.utils.session import load_session
            data = (load_session() or {}).get("controls_v2_int")
        except Exception:
            logger.debug("Controls V2 native Int session load failed",
                         exc_info=True)
            return
        if not isinstance(data, dict):
            return

        scan = getattr(self, "scan", None)
        if scan is None:
            return
        try:
            with scan.scan_lock:
                a1 = data.get("bai_1d_args")
                a2 = data.get("bai_2d_args")
                if isinstance(a1, dict):
                    scan.bai_1d_args = copy.deepcopy(a1)
                if isinstance(a2, dict):
                    scan.bai_2d_args = copy.deepcopy(a2)
                scan.gi = bool(data.get("gi", getattr(scan, "gi", False)))
                gic = data.get("gi_config")
                scan.gi_config = copy.deepcopy(gic) if isinstance(gic, dict) else {}
        except Exception:
            logger.debug("Controls V2 native Int scan restore failed",
                         exc_info=True)

        self._controls_v2_hydrate_advanced_from_scan()

        threshold = data.get("threshold_config")
        if isinstance(threshold, dict):
            self._controls_v2_threshold_state = {
                "apply_threshold": bool(threshold.get("apply_threshold", False)),
                "threshold_min": self._controls_v2_float(
                    threshold.get("threshold_min", 0.0), 0.0),
                "threshold_max": self._controls_v2_float(
                    threshold.get("threshold_max", 0.0), 0.0),
                "mask_saturation": bool(threshold.get("mask_saturation", True)),
            }
            cfg = self._controls_v2_threshold_config()
            for attr, value in (
                ("apply_threshold", cfg.apply_threshold),
                ("threshold_min", cfg.threshold_min),
                ("threshold_max", cfg.threshold_max),
                ("mask_sentinel", cfg.mask_saturation),
            ):
                try:
                    setattr(scan, attr, value)
                except Exception:
                    pass

        self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_native_reduction_plan(
        self,
        *,
        include_threshold: bool = True,
        integrate_1d: bool = True,
        integrate_2d: bool = True,
        commit_pending: bool = True,
    ):
        """Build the native Controls V2 reduction plan used by run/reintegrate."""

        if commit_pending:
            self._commit_controls_v2_pending_edits()
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_apply_gi_config_to_scan()

        scan = getattr(self, "scan", None)

        threshold_min = None
        threshold_max = None
        mask_saturation = False
        if include_threshold:
            cfg = self._controls_v2_threshold_config()
            if cfg.apply_threshold:
                threshold_min = cfg.threshold_min
                threshold_max = cfg.threshold_max
            mask_saturation = bool(cfg.mask_saturation)

        return build_native_int_reduction_plan_from_scan(
            scan,
            integrate_1d=integrate_1d,
            integrate_2d=integrate_2d,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            mask_saturation=mask_saturation,
        )

    @staticmethod
    def _controls_v2_native_run_plan_enabled() -> bool:
        value = os.environ.get("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", "1")
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _controls_v2_native_run_plan_builder(
        self,
        snapshot: dict,
    ):
        snapshot = copy.deepcopy(snapshot or {})
        snapshot_key = self._controls_v2_native_int_snapshot_key(snapshot)
        apply_snapshot = type(self)._controls_v2_apply_native_int_snapshot_to_scan

        def _prepare_scan(scan):
            apply_snapshot(snapshot, scan)

        def _builder(
            scan,
            *,
            integrate_1d: bool = True,
            integrate_2d: bool = True,
        ):
            _prepare_scan(scan)
            return build_native_int_reduction_plan_from_scan(
                scan,
                integrate_1d=integrate_1d,
                integrate_2d=integrate_2d,
            )

        _builder.prepare_scan = _prepare_scan
        _builder.plan_cache_key = ("controls_v2_native_int", snapshot_key)
        return _builder

    def _configure_controls_v2_native_run_plan(
        self,
        *,
        commit_pending: bool = False,
    ) -> None:
        builder = None
        if (
            self._controls_v2_enabled()
            and self._controls_v2_native_run_plan_enabled()
        ):
            self._apply_controls_v2_native_int_state(
                commit_pending=commit_pending,
                push_integrator=True,
            )
            builder = self._controls_v2_native_run_plan_builder(
                self._controls_v2_native_int_snapshot()
            )
        owners = (
            getattr(getattr(self, "wrangler", None), "thread", None),
            getattr(getattr(self, "integratorTree", None), "integrator_thread", None),
        )
        for owner in owners:
            cache = getattr(owner, "_plan_cache", None)
            if cache is not None and hasattr(cache, "plan_builder"):
                cache.plan_builder = builder

    def _on_controls_v2_field_changed(self, path, value) -> None:
        self._apply_controls_v2_field_value(path, value)
        self._refresh_controls_v2_profile(immediate=True)

    def _on_controls_v2_field_browse(self, path) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        handlers = {
            ("Project", "project_folder"): self._controls_v2_choose_project,
            ("Project", "h5_dir"): self._controls_v2_choose_output,
            ("Output", "h5_dir"): self._controls_v2_choose_output,
            ("Signal", "poni_file"): getattr(wrangler, "set_poni_file", None),
            ("Calibration", "poni_file"): getattr(wrangler, "browse_poni", None),
            ("Signal", "File"): getattr(wrangler, "set_img_file", None),
            ("Signal", "img_dir"): getattr(wrangler, "set_img_dir", None),
            ("Signal", "meta_dir"): getattr(wrangler, "set_meta_dir", None),
            ("Signal", "mask_file"): (
                getattr(wrangler, "set_mask_file", None)
                or getattr(wrangler, "browse_mask", None)
            ),
            ("NeXus File", "nexus_file"): getattr(wrangler, "browse_nexus", None),
            ("BG", "File"): getattr(wrangler, "set_bg_file", None),
        }
        handler = handlers.get(tuple(path))
        if callable(handler):
            handler()
        self._refresh_controls_v2_profile(immediate=True)

    def _on_mask_created(self, mask_file) -> None:
        """Auto-populate the Mask File field after Make Mask saves a mask.

        Single source of truth: write the same wrangler param the Mask File
        browse handler sets (``("Signal", "mask_file")``) + mirror the cached
        ``wrangler.mask_file`` attr, then refresh the V2 panel so the new path
        shows.  No-op if the path is empty or the param doesn't exist (e.g. a
        wrangler without a mask_file param).
        """
        if not mask_file:
            return
        param = self._controls_v2_param(("Signal", "mask_file"))
        if param is not None:
            try:
                param.setValue(str(mask_file))
            except Exception:
                logger.debug("Auto-populate mask_file failed", exc_info=True)
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None:
            try:
                wrangler.mask_file = str(mask_file)
            except Exception:
                pass
        self._refresh_controls_v2_profile(immediate=True)

    def _autofill_poni_after_calibrate(self, since_ts) -> None:
        """Offer to adopt a PONI written by the just-closed pyFAI-calib2.

        pyFAI-calib2 is an external program that doesn't tell us where the user
        saved the ``.poni``, so we rglob the project folder (falling back to the
        image directory) for a ``*.poni`` modified at/after the calibration
        launch and, if one is found, populate the Poni field after a
        confirmation popup so the user can double-check.  Picks the newest if
        several match; does nothing if none are found.
        """
        from pathlib import Path
        folder = ""
        for path in (("Project", "project_folder"), ("Signal", "img_dir")):
            param = self._controls_v2_param(path)
            try:
                value = param.value() if param is not None else ""
            except Exception:
                value = ""
            if value and os.path.isdir(value):
                folder = value
                break
        if not folder:
            return
        try:
            recent = [
                p for p in Path(folder).rglob("*.poni")
                if p.stat().st_mtime >= since_ts - 2.0
            ]
        except Exception:
            logger.debug("PONI auto-detect scan failed", exc_info=True)
            return
        if not recent:
            return
        newest = max(recent, key=lambda p: p.stat().st_mtime)
        poni_path = str(newest)
        extra = (f"\n\n(newest of {len(recent)} created during calibration)"
                 if len(recent) > 1 else "")
        answer = QMessageBox.question(
            self, "Calibration complete",
            "A new PONI file was found:\n\n"
            f"{poni_path}{extra}\n\nUse it as the calibration for this scan?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._set_poni_field(poni_path)

    def _set_poni_field(self, poni_path) -> None:
        """Adopt ``poni_path`` as the calibration, mirroring a manual browse.

        Sets the active Poni param — Image: ``("Signal", "poni_file")`` (its
        ``sigValueChanged`` runs ``get_poni_dict``); NeXus: ``("Calibration",
        "poni_file")`` — mirrors the wrangler's cached ``poni_file``, reloads
        the NeXus wrangler's cached ``poni`` (which only reloads on browse /
        session-restore, not a bare ``setValue``), then refreshes the V2 panel.
        """
        poni_path = str(poni_path)
        for path in (("Signal", "poni_file"), ("Calibration", "poni_file")):
            param = self._controls_v2_param(path)
            if param is not None:
                try:
                    param.setValue(poni_path)
                except Exception:
                    logger.debug("Set poni_file param failed for %s", path,
                                 exc_info=True)
                break
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None:
            try:
                wrangler.poni_file = poni_path
            except Exception:
                pass
            if isinstance(wrangler, nexusWrangler):
                try:
                    from xrd_tools.core.containers import PONI
                    if os.path.exists(poni_path):
                        wrangler.poni = PONI.from_poni_file(poni_path)
                except Exception:
                    logger.debug("PONI reload after autofill failed",
                                 exc_info=True)
        self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_choose_source(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            mode_text = ""
        if hasattr(wrangler, "browse_nexus"):
            wrangler.browse_nexus()
            return
        if "directory" in str(getattr(wrangler, "inp_type", "")).lower():
            browse = getattr(wrangler, "set_img_dir", None)
        else:
            browse = getattr(wrangler, "set_img_file", None)
        if mode_text in ("Image Viewer", "XYE Viewer"):
            browse = getattr(wrangler, "set_img_file", None) or browse
        if callable(browse):
            browse()

    def _controls_v2_choose_project(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        browse = getattr(wrangler, "set_project_folder", None)
        if browse is None:
            browse = getattr(wrangler, "browse_project_folder", None)
        if callable(browse):
            browse()

    def _controls_v2_choose_output(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        browse = getattr(wrangler, "set_h5_dir", None)
        if browse is None:
            browse = getattr(wrangler, "browse_h5_dir", None)
        if callable(browse):
            browse()

    def _refresh_controls_v2_profile(self, *, immediate: bool = False) -> None:
        if getattr(self, "_tearing_down", False):
            return
        timer = getattr(self, "_controls_v2_refresh_timer", None)
        if immediate or timer is None:
            self._refresh_controls_v2_profile_now()
        else:
            timer.trigger()

    def _refresh_controls_v2_profile_now(self) -> None:
        panel = getattr(self, "controls_v2", None)
        if panel is None:
            return
        if self._controls_v2_batch_run_active():
            self._controls_v2_batch_refresh_deferred = True
            return
        try:
            self._controls_v2_batch_refresh_deferred = False
            state = self._controls_v2_state()
            values = self._controls_v2_field_values()
            choices = self._controls_v2_field_choices()
            render_state = build_control_panel_state(state, values, choices)
            self._controls_v2_update_run_summary(state, render_state.profile)
            signature = render_state
            if signature == getattr(self, "_controls_v2_last_signature", None):
                return
            # A background rebuild (set_state -> clear_rows) would destroy a line
            # editor the user is mid-way through and silently drop the uncommitted
            # text.  If one is focused, defer until it COMMITS (editingFinished =
            # Enter / focus loss) rather than re-arming the throttle every interval
            # (which would keep a timer waking through a long acquisition).  The
            # rebuild is then SCHEDULED through the throttle, never run inside the
            # editingFinished slot (which would delete the editor mid-emission).
            # Signature is left unstamped so the deferred pass still detects the
            # change and rebuilds.
            editor = panel.focusWidget()
            if isinstance(editor, QtWidgets.QLineEdit):
                self._defer_controls_v2_refresh_until_commit(editor)
                return
            self._cancel_deferred_controls_v2_refresh()
            panel.set_state(render_state)
            self._controls_v2_last_signature = signature
        except Exception:
            logger.debug("Controls Panel V2 profile refresh failed",
                         exc_info=True)

    def _controls_v2_update_run_summary(self, state: ControlState, profile) -> None:
        setter = getattr(getattr(self, "controls", None),
                         "set_readiness_summary", None)
        text, ready, tooltip = self._controls_v2_run_summary(state, profile)
        if callable(setter):
            changed = setter(text, ready=ready, tooltip=tooltip)
            if changed:
                self._fit_controls_height()
        self._controls_v2_sync_run_row(profile)

    def _controls_v2_sync_run_row(self, profile) -> None:
        controls = getattr(self, "controls", None)
        if controls is None or getattr(self, "_run_active", False):
            return
        try:
            viewer = str(getattr(profile.processing_page, "value", "")) == "viewer"
            if viewer or controls.actionRow.isHidden():
                return
            controls.set_run_row_enabled(bool(getattr(profile, "can_run", False)))
        except Exception:
            logger.debug("Controls V2 run-row readiness sync failed",
                         exc_info=True)

    @staticmethod
    def _controls_v2_run_summary(state: ControlState, profile) -> tuple[str, bool, str]:
        if str(getattr(getattr(profile, "processing_page", None), "value", "")) == "viewer":
            return "", False, ""
        ready = bool(getattr(profile, "can_run", False))
        status = "Ready" if ready else "Needs setup"
        mode = str(getattr(state, "processing_mode", "") or "").strip()
        if not mode:
            page = getattr(getattr(profile, "processing_page", None),
                           "value", "")
            mode = str(page).replace("_", " ").title()
        blockers = tuple(getattr(profile, "run_blockers", ()) or ())
        parts = [status]
        if not ready and blockers:
            parts.append(str(blockers[0]).rstrip("."))
        if mode:
            parts.append(mode)
        frame_count = int(getattr(state, "frame_count", 0) or 0)
        if frame_count:
            plural = "" if frame_count == 1 else "s"
            parts.append(f"{frame_count} frame{plural}")
        tooltip = "" if ready else "; ".join(str(b) for b in blockers[:3])
        return " · ".join(parts), ready, tooltip

    def _defer_controls_v2_refresh_until_commit(self, editor) -> None:
        """Arm a one-shot: when ``editor`` finishes editing (Enter / focus loss),
        schedule the deferred Controls V2 rebuild through the throttle.  Replaces
        re-arming the throttle each interval, so a focused field during a long
        acquisition no longer keeps a timer alive."""
        if getattr(self, "_controls_v2_pending_editor", None) is editor:
            return
        self._cancel_deferred_controls_v2_refresh()
        self._controls_v2_pending_editor = editor
        editor.editingFinished.connect(self._on_controls_v2_pending_editor_done)

    def _cancel_deferred_controls_v2_refresh(self) -> None:
        """Drop a pending editingFinished one-shot (editor committed, was
        destroyed by a rebuild, or teardown)."""
        editor = getattr(self, "_controls_v2_pending_editor", None)
        if editor is not None:
            try:
                editor.editingFinished.disconnect(
                    self._on_controls_v2_pending_editor_done)
            except (TypeError, RuntimeError):
                pass
        self._controls_v2_pending_editor = None

    def _on_controls_v2_pending_editor_done(self, *args) -> None:
        """The deferred-on editor committed: schedule (do NOT run) the rebuild
        via the throttle, so it lands after this slot returns and we never delete
        the editor while it is still emitting editingFinished."""
        self._cancel_deferred_controls_v2_refresh()
        self._refresh_controls_v2_profile(immediate=False)

    def _on_gi_motor_options_changed(self, _motors=None) -> None:
        """The wrangler emitted a fresh GI motor-column list (metadata loaded);
        re-render so the inline V2 GI motor combo shows the updated choices."""
        self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_state(self) -> ControlState:
        """Build a lightweight, best-effort Controls V2 state snapshot."""
        mode_text = ""
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            pass
        tool = tool_from_mode_text(mode_text)
        gi_cfg = {}
        try:
            gi_cfg = (
                self._controls_v2_gi_config()
                if self._controls_v2_enabled()
                else self.integratorTree.get_gi_config()
            )
        except Exception:
            gi_cfg = {}
        gi_on = bool(gi_cfg.get("gi", getattr(self.scan, "gi", False)))
        meas_mode = MeasMode.GI if gi_on else MeasMode.STANDARD

        frame_count = 0
        try:
            frame_count = len(getattr(self.scan.frames, "index", ()) or ())
        except Exception:
            frame_count = 0
        source_label = self._controls_v2_source_label()
        project_root = str(getattr(getattr(self, "wrangler", None), "project_folder", "") or "")
        project_root_valid = bool(
            project_root and os.path.isdir(os.path.expanduser(project_root))
        )
        has_scan_data = False
        try:
            has_scan_data = not getattr(self.scan, "scan_data", None).empty
        except Exception:
            has_scan_data = False
        wrangler = getattr(self, "wrangler", None)
        has_motors = bool(getattr(wrangler, "motors", None)) or has_scan_data
        has_raw = bool(frame_count or source_label)
        calibration_energy_eV, source_energy_eV = (
            self._controls_v2_energy_values())
        energy_known = (
            calibration_energy_eV is not None
            or source_energy_eV is not None
        )
        source_caps = SourceCaps(
            has_frames=frame_count > 0,
            has_raw=has_raw,
            raw_reachable=has_raw,
            has_metadata=has_scan_data or bool(getattr(wrangler, "scan_args", None)),
            has_motors=has_motors,
            has_energy=energy_known,
            has_geometry=self._controls_v2_calibrated(),
            has_psi_metadata=has_scan_data,
        )

        has_1d = bool(self.data_1d)
        has_2d = bool(self.data_2d)
        labels = ()
        try:
            labels = self.publication_store.labels()
            recent = labels[-16:] if len(labels) > 16 else labels
            pubs = tuple(self.publication_store.get_many(recent).values())
            has_1d = has_1d or any(getattr(pub.view, "int_1d", None) is not None
                                   for pub in pubs)
            has_2d = has_2d or any(getattr(pub.view, "int_2d", None) is not None
                                   for pub in pubs)
        except Exception:
            labels = ()
        result_caps = ResultCaps(
            has_1d=has_1d,
            has_2d=has_2d,
            has_raw=has_raw,
            raw_reachable=has_raw,
            has_scan_metadata=has_scan_data,
            has_rsm=bool(getattr(self.scan, "rsm_result", None)),
            has_phase_result=False,
            has_psi_metadata=has_scan_data,
        )
        geom = GeomState(
            calibrated=self._controls_v2_calibrated(),
            energy_known=energy_known,
            calibration_energy_eV=calibration_energy_eV,
            source_energy_eV=source_energy_eV,
            gi_enabled=gi_on,
            sample_orientation_known=(
                not gi_on or bool(gi_cfg.get("sample_orientation"))),
            ub_known=bool(getattr(self.scan, "ub_matrix", None)),
            material_known=False,
        )
        run_active = (bool(getattr(self, "_run_active", False))
                      or self._session_run_active())
        display_frame_count = frame_count or len(labels)
        if run_active:
            # During a run the processed-frame count ticks every frame; letting
            # it into the render signature would rebuild the whole controls panel
            # ~5 Hz (clear_rows + recreate every row).  Live progress is shown in
            # the status bar, so freeze the panel's frame count at the run-start
            # value (snapshot once) — it resyncs when the run ends and the
            # snapshot clears.  The panel is locked during a run, so a frozen
            # count loses nothing.
            if getattr(self, "_controls_v2_run_frame_count", None) is None:
                self._controls_v2_run_frame_count = display_frame_count
            display_frame_count = self._controls_v2_run_frame_count
        else:
            self._controls_v2_run_frame_count = None
        return ControlState(
            tool=tool,
            mode=meas_mode,
            source_caps=source_caps,
            result_caps=result_caps,
            geom=geom,
            backend=self._controls_v2_backend(tool),
            project_root=project_root,
            project_root_required=True,
            project_root_valid=project_root_valid,
            source_label=source_label,
            save_path=str(getattr(getattr(self, "wrangler", None), "h5_dir", "") or ""),
            detector_summary=self._controls_v2_detector_summary(),
            frame_count=display_frame_count,
            processing_mode=mode_text,
            real_data_gates=frozenset(),
            controls_locked=run_active,
        )

    def _controls_v2_source_label(self) -> str:
        source_type = ""
        param = self._controls_v2_param(("Signal", "inp_type"))
        if param is not None:
            try:
                source_type = str(param.value() or "")
            except Exception:
                source_type = ""

        source_paths = [("NeXus File", "nexus_file")]
        if source_type == "Image Directory":
            source_paths.append(("Signal", "img_dir"))
        else:
            source_paths.append(("Signal", "File"))
        source_paths.extend((("Signal", "File"), ("Signal", "img_dir")))

        seen = set()
        candidates = []
        for path in source_paths:
            if path in seen:
                continue
            seen.add(path)
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                candidates.append(param.value())
            except Exception:
                pass

        wrangler = getattr(self, "wrangler", None)
        candidates.extend((
            getattr(wrangler, "img_file", None),
            getattr(wrangler, "img_dir", None),
            getattr(wrangler, "nexus_file", None),
            getattr(self.scan, "data_file", None),
        ))
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    def _controls_v2_calibrated(self) -> bool:
        return self._controls_v2_current_poni() is not None

    def _controls_v2_detector_summary(self) -> str:
        """Compact detector/PONI summary for the Experiment subsection header."""
        poni = self._controls_v2_current_poni()
        scan = getattr(self, "scan", None)
        integrator = getattr(scan, "_cached_integrator", None)

        detector = self._controls_v2_detector_name(
            getattr(poni, "detector", None))
        if not detector:
            detector = self._controls_v2_detector_name(
                getattr(getattr(integrator, "detector", None), "name", None))
        if not detector:
            detector = self._controls_v2_detector_name(
                getattr(integrator, "detector", None))

        dist_m = self._controls_v2_positive_float(getattr(poni, "dist", None))
        if dist_m is None:
            dist_m = self._controls_v2_positive_float(
                getattr(integrator, "dist", None))

        parts = []
        if detector:
            parts.append(detector)
        if dist_m is not None:
            parts.append(self._controls_v2_detector_distance_text(dist_m))
        if parts:
            parts.append("fitted")
        return " · ".join(parts)

    def _controls_v2_current_poni(self):
        scan = getattr(self, "scan", None)
        wrangler = getattr(self, "wrangler", None)
        for candidate in (
            getattr(scan, "_cached_poni", None),
            getattr(wrangler, "poni", None),
            getattr(getattr(wrangler, "thread", None), "poni", None),
            getattr(getattr(self, "integratorTree", None), "_cached_poni", None),
        ):
            if candidate is not None:
                return candidate

        poni_path = self._controls_v2_poni_path()
        if not poni_path or not os.path.exists(poni_path):
            return None
        try:
            from xrd_tools.core.containers import PONI
            return PONI.from_poni_file(poni_path)
        except Exception:
            logger.debug("Controls V2 PONI summary load failed for %s",
                         poni_path, exc_info=True)
            return None

    def _controls_v2_poni_path(self) -> str:
        candidates = []
        for path in (("Signal", "poni_file"), ("Calibration", "poni_file")):
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                candidates.append(param.value())
            except Exception:
                pass
        wrangler = getattr(self, "wrangler", None)
        candidates.append(getattr(wrangler, "poni_file", ""))
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    @staticmethod
    def _controls_v2_detector_name(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = (
                getattr(value, "name", None)
                or getattr(value, "alias", None)
                or getattr(value, "__class__", type(value)).__name__
            )
        name = str(value).strip()
        if name.lower() in {"", "none", "detector"}:
            return ""
        return name

    @staticmethod
    def _controls_v2_positive_float(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) and out > 0 else None

    @staticmethod
    def _controls_v2_detector_distance_text(dist_m: float) -> str:
        return f"{dist_m * 1000.0:.1f}mm"

    def _controls_v2_energy_known(self) -> bool:
        calibration_energy_eV, source_energy_eV = self._controls_v2_energy_values()
        return calibration_energy_eV is not None or source_energy_eV is not None

    def _controls_v2_energy_values(self) -> tuple[float | None, float | None]:
        """Return ``(calibration_energy_eV, source_energy_eV)`` for run gating.

        The calibration wavelength is authoritative.  A persisted 1 Å value can
        be real, but the historical ``mg_args['wavelength'] == 1e-10``
        constructor default is only a placeholder and must not satisfy the gate.
        """
        scan = getattr(self, "scan", None)
        calibration_energy_eV = self._controls_v2_calibration_energy_eV(
            scan,
            poni=self._controls_v2_current_poni(),
        )
        source_energy_eV = self._controls_v2_source_energy_eV(scan)
        return calibration_energy_eV, source_energy_eV

    @classmethod
    def _controls_v2_energy_from_wavelength(
            cls, value, *, allow_default_sentinel: bool = False
    ) -> float | None:
        wavelength_m = normalize_wavelength_m(
            value,
            allow_default_sentinel=allow_default_sentinel,
        )
        if wavelength_m is None:
            return None
        try:
            energy = float(wavelength_m_to_energy_eV(wavelength_m))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return energy if energy > 0 else None

    @classmethod
    def _controls_v2_calibration_energy_eV(cls, scan, *, poni=None) -> float | None:
        if scan is None:
            return None
        persisted = cls._controls_v2_energy_from_wavelength(
            getattr(scan, "_persisted_wavelength_m", None),
            allow_default_sentinel=True,
        )
        if persisted is not None:
            return persisted
        from_poni = cls._controls_v2_energy_from_wavelength(
            getattr(poni, "wavelength", None),
            allow_default_sentinel=True,
        )
        if from_poni is not None:
            return from_poni
        integrator = getattr(scan, "_cached_integrator", None)
        from_integrator = cls._controls_v2_energy_from_wavelength(
            getattr(integrator, "wavelength", None))
        if from_integrator is not None:
            return from_integrator
        mg_args = getattr(scan, "mg_args", {}) or {}
        if isinstance(mg_args, dict):
            return cls._controls_v2_energy_from_wavelength(
                mg_args.get("wavelength"))
        return None

    @staticmethod
    def _controls_v2_positive_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number <= 0:
            return None
        if number in (float("inf"), float("-inf")):
            return None
        return number

    @classmethod
    def _controls_v2_extract_energy_eV(cls, mapping) -> float | None:
        if not hasattr(mapping, "items"):
            return None
        try:
            lowered = {str(key).lower(): value for key, value in mapping.items()}
        except Exception:
            return None
        for key in (
            "energy_ev",
            "energyev",
            "beam_energy_ev",
            "source_energy_ev",
            "calibration_energy_ev",
        ):
            energy = cls._controls_v2_positive_float(lowered.get(key))
            if energy is not None:
                return energy
        for key in (
            "energy_kev",
            "energykev",
            "beam_energy_kev",
            "source_energy_kev",
        ):
            energy = cls._controls_v2_positive_float(lowered.get(key))
            if energy is not None:
                return energy * 1000.0
        return None

    @classmethod
    def _controls_v2_source_energy_eV(cls, scan) -> float | None:
        if scan is None:
            return None
        for attr in (
            "source_energy_eV",
            "beam_energy_eV",
            "energy_eV",
        ):
            energy = cls._controls_v2_positive_float(getattr(scan, attr, None))
            if energy is not None:
                return energy
        for attr in ("source_energy_keV", "beam_energy_keV", "energy_keV"):
            energy = cls._controls_v2_positive_float(getattr(scan, attr, None))
            if energy is not None:
                return energy * 1000.0
        for attr in ("metadata", "meta", "scan_info"):
            energy = cls._controls_v2_extract_energy_eV(getattr(scan, attr, None))
            if energy is not None:
                return energy
        return None

    def _controls_v2_batch_run_active(self) -> bool:
        run_active = (
            bool(getattr(self, "_run_active", False))
            or self._session_run_active()
        )
        if not run_active:
            return False
        controls = getattr(self, "controls", None)
        try:
            if controls is not None and controls.current_mode() in (
                    "Image Viewer", "XYE Viewer", "NeXus Viewer"):
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def _controls_v2_backend(tool) -> str | None:
        if tool == Tool.STITCH:
            return "multigeometry"
        if tool == Tool.RSM:
            return "rsm"
        return None

    def _build_tools_placeholder(self):
        """Fill the vacated bottom-left ``metaFrame`` with the compact 'Tools' card.

        Each tool is a full-width button labelled with the tool name; the hover
        tooltip carries the description (the old dot+label+Open rows and the
        wrapped note below took too much vertical space).  Clicking opens the
        tool's dialog.  Reclaims the corner freed by moving the metadata table
        into a popup."""
        lay = QtWidgets.QVBoxLayout(self.ui.metaFrame)
        lay.setContentsMargins(13, 9, 13, 10)
        lay.setSpacing(6)

        header = QtWidgets.QLabel('TOOLS')
        header.setObjectName('toolsHeader')
        lay.addWidget(header)

        card = QtWidgets.QFrame()
        card.setObjectName('toolsPlaceholder')
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(9, 9, 9, 9)
        card_lay.setSpacing(6)
        # (symbol, label, handler-or-None, tooltip).  Handler => active tool; the
        # button opens it.  A None handler is a not-yet-built tool (button
        # disabled).  Symbols are decorative glyphs (standard-font safe).
        tools = [
            ('∧', 'Peak Fitting', self._open_peak_fit_dialog,
             'Structure-agnostic peak fitting — selected frame and across the scan.'),
            ('≈', 'Phase Fitting', self._open_phase_fit_dialog,
             'CIF-based phase fitting — selected frame and across the scan.'),
            ('▤', 'Plot Metadata', self._open_scan_plot_dialog,
             'Plot scan metadata + image-ROI statistics vs frame.'),
        ]
        for symbol, name, handler, tip in tools:
            btn = QtWidgets.QPushButton(f'{symbol}   {name}')
            btn.setObjectName('toolButton')
            btn.setToolTip(tip)
            if handler is not None:
                btn.clicked.connect(handler)
            else:
                btn.setEnabled(False)
            card_lay.addWidget(btn)
        lay.addWidget(card)
        lay.addStretch(1)

    def _pattern_for_frame(self, idx):
        """Return ``(x, y, x_label)`` for ONE frame's 1-D pattern, or ``None``.
        Reads the same data + axis unit the main 1-D plot shows.  Shared by the
        single-frame fit (selected frame) and batch (every frame)."""
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        try:
            ydata, xdata = self.displayframe.get_frames_int_1d([idx], rv='all')
        except Exception:
            logger.exception("peak-fit: get_frames_int_1d failed")
            return None
        if xdata is None or ydata is None:
            return None
        import numpy as np
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        if y.ndim > 1:
            y = y[0]
        if x.size == 0 or y.size == 0:
            return None
        try:
            label = self.displayframe.ui.plotUnit.currentText()
        except Exception:
            label = 'q'
        return x, y, label

    def _current_pattern_for_fit(self):
        """Return ``(x, y, x_label)`` for the SELECTED frame's 1-D pattern, or
        ``None`` — so a fit always matches what the user is looking at."""
        idxs = getattr(self, 'frame_ids', None) or []
        if not idxs:
            return None
        return self._pattern_for_frame(idxs[0])

    def _analysis_context(self):
        """Stable analysis-data contract for popup tools.

        Popups consume this context rather than reading staticWidget,
        displayframe, wrangler, or integrator internals.  The providers still
        point at the current live/reloaded publication path, so live fitting
        continues to update on each processed frame.
        """
        from .analysis_context import AnalysisContext
        return AnalysisContext(
            current_pattern_provider=self._current_pattern_for_fit,
            frame_pattern_provider=self._pattern_for_frame,
            scan_uri_provider=self._current_scan_uri,
            mask_provider=self._scan_plot_mask_provider,
            frame_labels_provider=lambda: tuple(getattr(self, 'frame_ids', ()) or ()),
            metadata_provider=lambda: {})

    def _open_peak_fit_dialog(self):
        """Open (or re-show) the Peak Fitting popup — lazy, single-instance,
        non-modal (so the live scan + frame browsing stay responsive; Reload
        re-grabs the current frame)."""
        if self._peak_fit_dialog is None:
            from .peak_fit_dialog import PeakFitDialog
            self._peak_fit_dialog = PeakFitDialog(
                analysis_context=self._analysis_context(), parent=self)
            # Toggling Live on re-fits the current frame at once (then every new
            # frame, via set_data); off just stops pushing.
            self._peak_fit_dialog.live_check.toggled.connect(
                self._on_live_fit_toggled)
            self._peak_fit_dialog.batch_btn.clicked.connect(
                lambda: self._on_batch_clicked(self._peak_fit_dialog))
        dlg = self._peak_fit_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.refresh_pattern()

    def _on_live_fit_toggled(self, on):
        """Live checkbox flipped — fit the current frame immediately on enable so
        there's no wait for the next frame; disabling just stops the pushes."""
        if on:
            dlg = self._peak_fit_dialog
            if dlg is not None:
                dlg.reset_param_trend()      # fresh vs-frame trend for this run
            self._maybe_live_fit()

    def _ensure_live_analysis_worker(self):
        """Lazily create + start the latest-wins live analysis worker."""
        if self._live_analysis_worker is None:
            from .analysis_worker import LiveAnalysisWorker
            self._live_analysis_worker = LiveAnalysisWorker(self)
            self._live_analysis_worker.sigAnalyzed.connect(self._on_live_analyzed)
            self._live_analysis_worker.start()
        return self._live_analysis_worker

    def _maybe_live_fit(self):
        """If the Peak Fitting dialog is open with Live on, push the current
        frame's pattern to it and request a background re-fit (latest-wins).

        Called from ``set_data`` (per frame) and on Live-toggle.  The analyzer +
        request are built through the dialog's own ``build_fit_request`` so live
        and the manual Fit button behave identically."""
        dlg = self._peak_fit_dialog
        if dlg is None or not dlg.isVisible() or not dlg.live_check.isChecked():
            return
        data = self._analysis_context().current_pattern_tuple()
        if not data:
            return
        x, y, label = data
        dlg.set_live_pattern(x, y, label)   # show the data now; fit overlays async
        req = dlg.build_fit_request()
        if req is None:                      # nothing fittable (status set by dialog)
            return
        inp, analyzer = req
        # Label the analysis by the FRAME index (not the axis unit) so the
        # dialog keys the vs-frame trend by frame; build_fit_request defaults the
        # label to "current".
        idxs = getattr(self, 'frame_ids', None) or []
        inp.label = str(idxs[0]) if idxs else ""
        self._live_fit_gen += 1
        self._ensure_live_analysis_worker().request(
            inp.label, self._live_fit_gen, analyzer, inp)

    def _on_live_analyzed(self, label, generation, outcome):
        """Draw a live fit result — but only if it's still the newest request and
        the dialog is still open + Live (a stale or superseded result is dropped,
        so the overlay never lags behind the displayed frame)."""
        if self._tearing_down:
            return                              # widget closing — dialog may be gone
        if generation != self._live_fit_gen:
            return
        dlg = self._peak_fit_dialog
        if dlg is None or not dlg.isVisible() or not dlg.live_check.isChecked():
            return
        if outcome is not None and outcome.ok:
            dlg._draw_outcome(outcome, auto=dlg.auto_check.isChecked())

    # ---- Batch fit (Peak or Phase — dialog-parameterized) --------------
    def _on_batch_clicked(self, dialog):
        """Batch button: start a batch fit for ``dialog``, or cancel one in
        flight.  Shared by the Peak and Phase fitters."""
        worker = self._batch_analysis_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            return
        self._run_batch_fit(dialog)

    def _run_batch_fit(self, dialog):
        """Fit every frame in the scan with ``dialog``'s current settings, then
        plot the parameters vs frame number.

        The analyzer is fixed ONCE from the current frame (via the dialog's
        ``build_fit_request``) and applied to every frame, so each parameter
        series tracks the same thing across frames."""
        import numpy as np
        from xrd_tools.analysis.runner import AnalysisInput
        dlg = dialog
        if dlg is None:
            return
        if dlg._x is None or dlg._y is None:
            dlg.refresh_pattern()
        req = dlg.build_fit_request()
        if req is None:
            return                              # status set by the dialog
        _, analyzer = req
        lo, hi = dlg.batch_x_range()        # Peak: fit range; Phase: full extent
        try:
            frame_idxs = list(self.scan.frames.index)
        except Exception:
            frame_idxs = []
        if not frame_idxs:
            dlg.status.setText("No frames to batch-fit.")
            return
        x_unit = dlg._x_label
        inputs = []
        ctx = self._analysis_context()
        for idx in frame_idxs:
            data = ctx.pattern_tuple_for_frame(idx)
            if not data:
                continue
            fx, fy, _lbl = data
            fx = np.asarray(fx, dtype=float)
            fy = np.asarray(fy, dtype=float)
            mask = (np.isfinite(fx) & np.isfinite(fy)
                    & (fx >= lo) & (fx <= hi))
            if not np.any(mask):
                continue
            inputs.append(AnalysisInput(label=str(idx), x=fx[mask], y=fy[mask],
                                        x_unit=x_unit))
        if not inputs:
            dlg.status.setText("No fittable frames in the selected range.")
            return
        from .analysis_worker import BatchAnalysisWorker
        if self._batch_analysis_worker is None:
            self._batch_analysis_worker = BatchAnalysisWorker(self)
            self._batch_analysis_worker.sigProgress.connect(self._on_batch_progress)
            self._batch_analysis_worker.sigFrameFit.connect(self._on_batch_frame_fit)
            self._batch_analysis_worker.sigBatchDone.connect(self._on_batch_done)
        dlg.reset_param_trend()             # fresh vs-frame trend for this batch
        self._batch_dialog = dlg            # the slots route results back to it
        self._batch_analysis_worker.configure(analyzer, inputs)
        dlg.set_batch_running(True)
        dlg.set_batch_progress(0, len(inputs))
        self._batch_analysis_worker.start()

    def _on_batch_progress(self, done, total):
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is not None:
            dlg.set_batch_progress(done, total)

    def _on_batch_frame_fit(self, label, params):
        """A batch frame finished: grow the dialog's vs-frame trend (row 3)."""
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is None or not dlg.isVisible():
            return
        try:
            frame_idx = int(label)
        except (TypeError, ValueError):
            return
        dlg._accumulate_frame_params(frame_idx, params)

    def _on_batch_done(self, labels, columns):
        """Batch finished: re-enable the dialog (or report a cancel).  The
        vs-frame trend already filled row 3 incrementally via sigFrameFit."""
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is not None:
            dlg.set_batch_running(False)
        if labels is None:                      # cancelled before completion
            if dlg is not None:
                dlg.status.setText("Batch fit cancelled.")
            return
        if dlg is not None:
            dlg.status.setText(
                f"Batch fit done — {len(labels)} frames. Pick a parameter to "
                "track below; Save CSV to export.")

    def _open_phase_fit_dialog(self):
        """Open (or re-show) the Phase Fitting popup — lazy, single-instance,
        non-modal.  Shares the batch worker + vs-frame trend with Peak Fitting."""
        if self._phase_fit_dialog is None:
            from .phase_fit_dialog import PhaseFitDialog
            self._phase_fit_dialog = PhaseFitDialog(
                analysis_context=self._analysis_context(), parent=self)
            self._phase_fit_dialog.batch_btn.clicked.connect(
                lambda: self._on_batch_clicked(self._phase_fit_dialog))
        dlg = self._phase_fit_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.refresh_pattern()

    def _current_scan_uri(self):
        """Best-effort path to the currently-loaded scan (for Scan Plot's default
        source); None when nothing real is loaded (the dialog starts blank)."""
        import os
        for cand in (getattr(self.scan, 'data_file', None),
                     getattr(self, 'fname', None)):
            try:
                if cand and os.path.exists(str(cand)):
                    return str(cand)
            except (TypeError, ValueError):
                continue
        return None

    def _scan_plot_mask_provider(self, uri):
        """The loaded scan's static detector mask (``scan.global_mask``) — but
        ONLY when the Scan Plot's picked source IS that loaded scan.  An
        arbitrary other source has its own detector/geometry, so the loaded
        scan's mask must not be applied to it."""
        import os
        loaded = self._current_scan_uri()
        try:
            same = bool(loaded and uri and os.path.realpath(str(uri))
                        == os.path.realpath(str(loaded)))
        except (TypeError, ValueError):
            same = False
        return getattr(self.scan, 'global_mask', None) if same else None

    def _open_scan_plot_dialog(self):
        """Open (or re-show) the Scan Plot popup — lazy, single-instance,
        non-modal.  Starts on the currently-loaded scan (or blank)."""
        if self._scan_plot_dialog is None:
            from .scan_plot_dialog import ScanPlotDialog
            ctx = self._analysis_context()
            self._scan_plot_dialog = ScanPlotDialog(
                default_uri=ctx.current_scan_uri(),
                mask_provider=ctx.mask_for_scan_uri, parent=self)
        dlg = self._scan_plot_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_metadata_dialog(self):
        """Open (or re-show) the frame-metadata popup.

        Lazy, single-instance, NON-modal: built once on first click by reparenting
        the live ``self.metawidget`` into a ``QDialog`` so its table becomes
        visible.  Non-modal keeps the live scan + h5viewer frame selection
        responsive, and because the widget holds the shared frame_ids /
        publication store, it refreshes as you browse frames (its ``update()``
        is gated on ``tableview.isVisible()``, which is now exactly 'dialog
        open')."""
        if self._metadata_dialog is None:
            dlg = QDialog(self)
            dlg.setObjectName('metadataDialog')
            dlg.setWindowTitle('Frame metadata')
            dlg.resize(460, 460)
            dlg_lay = QtWidgets.QVBoxLayout(dlg)
            dlg_lay.setContentsMargins(0, 0, 0, 0)
            dlg_lay.addWidget(self.metawidget)
            self._metadata_dialog = dlg
        self._metadata_dialog.show()
        self._metadata_dialog.raise_()
        self._metadata_dialog.activateWindow()
        # The table only renders while visible; refresh now it is shown.
        self.metawidget.update()

    def _connect_signals(self):
        """Wire signal/slot connections for H5Viewer, DisplayFrame, and Integrator."""
        # H5Viewer signals
        self.h5viewer.sigUpdate.connect(self.set_data)
        self.h5viewer.file_thread.sigTaskStarted.connect(self.thread_state_changed)
        self.h5viewer.sigThreadFinished.connect(self.thread_state_changed)
        self.h5viewer.ui.listData.itemClicked.connect(self.disable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.enable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.latest_frame)
        # Stage 4: open the frame-metadata popup (local open-dialog connection).
        self.h5viewer.ui.metadata_btn.clicked.connect(self._open_metadata_dialog)

        # DisplayFrame signals.  (The "Update 2D" toggle was removed — 2D
        # now always renders.  The File ▸ Export menu actions still drive
        # the save_image / save_1D methods even though the in-panel Save
        # buttons are gone.)
        self.h5viewer.actionSaveImage.triggered.connect(self.displayframe.save_image)
        self.h5viewer.actionSaveArray.triggered.connect(self.displayframe.save_1D)
        # Plot method changes drive the H5 data list selection mode so
        # accumulating modes (Overlay/Waterfall/Sum/Average) auto-add
        # clicked points without requiring shift/ctrl.
        self.displayframe.sigPlotMethodChanged.connect(
            self.h5viewer.set_data_selection_mode)
        # Initialize once with the current plot method.
        self.h5viewer.set_data_selection_mode(
            self.displayframe.ui.plotMethod.currentText())
        # Viewer-mode Clear: also drop the file-list selection so the cleared
        # plot, the selection and the title agree (the displayframe reset the
        # title; the selection lives on the H5Viewer).
        self.displayframe.sigCleared.connect(self._on_display_cleared)

        # Integrator signals
        # GI on/off now lives on the integrator panel — route its toggle through
        # the same handler the wrangler's GI checkbox used (sets scan.gi +
        # refreshes the panel axis units/labels).
        self.integratorTree.sigUpdateGI.connect(self.update_scattering_geometry)
        # Make Mask just wrote a mask file — auto-populate the Mask File field.
        self.integratorTree.sigMaskCreated.connect(self._on_mask_created)
        self.integratorTree.integrator_thread.started.connect(self.thread_state_changed)
        # Re-integration is a "run" too: route its START through the single
        # run-state owner (task #68) — keeps the 2D panels persistent AND
        # disables the processing controls (task #71) for its duration; cleared
        # in integrator_thread_finished via _exit_run_state.
        self.integratorTree.integrator_thread.started.connect(self._enter_run_state)
        self.integratorTree.integrator_thread.update.connect(self.integrator_thread_update)
        self.integratorTree.integrator_thread.writeError.connect(
            self._show_reintegration_write_error)
        self.integratorTree.integrator_thread.finished.connect(self.integrator_thread_finished)
        # Advanced (re-homed from the wrangler's button onto the integrator's own
        # Reintegrate row): the single combined 1D+2D advanced-settings dialog.
        # Wired ONCE here — the integratorTree persists across wrangler swaps, so
        # there's no per-wrangler connect/disconnect dance to manage.
        if hasattr(self.integratorTree.ui, 'advanced_int'):
            self.integratorTree.ui.advanced_int.clicked.connect(
                self._show_integration_advanced)
        # Pixel rejection (Intensity Threshold + Mask Saturated) now lives in the
        # integrator panel and is read straight from its own
        # integratorTree.get_threshold_config() — for Reintegrate (the integrator
        # reads itself) and for live runs (injected into the wrangler at
        # run-setup; see _push_threshold_to_wrangler).

    def _show_reintegration_write_error(self, message: str) -> None:
        """Surface reintegration save failures in the same status area as runs."""
        try:
            self.wrangler.showLabel.emit(message)
        except Exception:
            logger.debug("could not surface reintegration write failure",
                         exc_info=True)

    def _push_threshold_to_wrangler(self):
        """Inject the native V2 pixel-rejection policy into the active
        wrangler setup params so live and reintegrate share one policy.

        Called from ``start_wrangler`` BEFORE ``wrangler.setup()`` (which reads
        those params and pushes them to the thread).  Per-field guarded: a
        wrangler without an 'Intensity Threshold' group (e.g. NeXus) just skips
        it, and still receives 'Mask Saturated'.
        """
        try:
            cfg = (
                self._controls_v2_threshold_config()
                if self._controls_v2_enabled()
                else self.integratorTree.get_threshold_config()
            )
        except Exception:
            # Fail LOUD: silently falling back means the LIVE run applies the
            # wrangler's default pixel-rejection instead of the integrator's
            # setting (a quiet live≠reintegrate divergence).
            logger.warning(
                "Could not read integrator threshold config; the live run will "
                "use the wrangler default pixel-rejection, which may differ from "
                "the integrator setting.", exc_info=True)
            return
        if cfg is None:
            return
        params = getattr(self.wrangler, 'parameters', None)
        if params is None:
            return

        def _set(group, child, value):
            try:
                params.child(group).child(child).setValue(value)
            except Exception:
                pass  # wrangler lacks this group (e.g. NeXus has no 'Mask')

        _set('Mask', 'Threshold', bool(cfg.apply_threshold))
        _set('Mask', 'min', cfg.threshold_min)
        _set('Mask', 'max', cfg.threshold_max)
        _set('MaskSat', 'mask_sentinel', bool(cfg.mask_saturation))

    def _push_gi_to_wrangler(self):
        """Inject the native V2 GI geometry into the active wrangler setup params."""
        try:
            cfg = (
                self._controls_v2_gi_config()
                if self._controls_v2_enabled()
                else self.integratorTree.get_gi_config()
            )
        except Exception:
            # Fail LOUD: a silent fallback means the LIVE run uses the wrangler's
            # default GI geometry instead of the integrator's (quiet divergence).
            logger.warning(
                "Could not read integrator GI config; the live run will use the "
                "wrangler default GI geometry, which may differ from the "
                "integrator setting.", exc_info=True)
            return
        params = getattr(self.wrangler, 'parameters', None)
        if params is None or cfg is None:
            return

        def _set(group, child, value):
            try:
                params.child(group).child(child).setValue(value)
            except Exception:
                pass  # wrangler lacks this group/child

        _set('GI', 'Grazing', bool(cfg['gi']))
        _set('GI', 'sample_orientation', int(cfg['sample_orientation']))
        _set('GI', 'tilt_angle', float(cfg['tilt_angle']))
        _set('GI', 'th_motor', str(cfg['incidence_motor']))
        _set('GI', 'th_val', str(cfg['th_val']))

    def _init_wranglers(self):
        """Initialize the wrangler stack and select the default wrangler."""
        self.wrangler = wranglerWidget("uninitialized", threading.Condition())
        for name, w in wranglers.items():
            self.ui.wranglerStack.addWidget(
                w(
                    self.fname, self.file_lock,
                    self.scan, self.data_1d, self.data_2d,
                )
            )
        self.ui.wranglerStack.currentChanged.connect(self.set_wrangler)
        self.command_queue = Queue()
        self.set_wrangler(self.ui.wranglerStack.currentIndex())

    def _init_defaults_and_timer(self):
        """Set up default parameters and the coalescing update timer."""
        # Register all parameter trees with the defaultWidget
        parameters = [self.integratorTree.parameters]
        for i in range(self.ui.wranglerStack.count()):
            w = self.ui.wranglerStack.widget(i)
            parameters.append(w.parameters)
        self.h5viewer.defaultWidget.set_parameters(parameters)

        # Single source of truth for "a wrangler/integrator run is in
        # progress" (task #68).  Flipped only by _enter_run_state /
        # _exit_run_state, which drive the display persist flag AND the
        # processing-control disable (task #71) so the two can never desync.
        self._run_active = False

        # Coalescing timer for wrangler updates: when the wrangler thread
        # processes images faster than the GUI can render, only the most
        # recent update is rendered after a short quiet period (200 ms).
        self._pending_update_idx = None
        # Per-frame display refresh: THROTTLE (not debounce) — a steady
        # live stream must still paint every ~200 ms; the latest index
        # wins via _pending_update_idx (the shared coalescing idiom,
        # xdart.utils.throttle).
        self._update_timer = Coalescer(200, mode="throttle", parent=self)
        self._update_timer.triggered.connect(self._flush_pending_update)
        # Reintegrate gets its OWN throttle: bai_*_all (live, batch=1) fires a
        # per-frame `update` signal, and rendering each one synchronously floods
        # the GUI (esp. the 2D cake at ~hundreds-of-ms each) -> the whole GUI
        # freezes + paints nothing until the run ends.  Coalesce to ~5 Hz so the
        # display tracks progress smoothly, like the wrangler's update_data path.
        self._pending_reint_idx = None
        self._reint_update_timer = Coalescer(200, mode="throttle", parent=self)
        self._reint_update_timer.triggered.connect(self._flush_reintegrate_update)
        # Per-frame work is COALESCED off the GUI event loop: update_data only
        # POPs the freshly-integrated frame (cheap) into _pending_frames; the
        # heavy build/upsert/scan_data runs once per ~200 ms flush over ALL frames
        # stashed since the last flush.  Running it per frame on the GUI thread
        # flooded the event loop (esp. once lz4 removed gzip's accidental
        # write-throttle) and froze the GUI for the whole scan.  _scan_info_rows
        # accumulates metadata rows so scan_data is rebuilt as one DataFrame per
        # flush instead of an O(N^2) per-frame `sd.loc[idx] = ser` enlargement.
        self._pending_frames = {}
        self._scan_info_rows = {}
        self._restore_controls_v2_int_session_state()

    def _fit_controls_height(self):
        """Pin the bottom controls bar to its current content height.

        Recomputed (not set once) because StaticControls shows/hides rows after
        init -- the run row hides in viewer modes, mode-specific widgets toggle
        per profile -- so a height frozen from the initial sizeHint would leave
        slack (run row hidden) or clip (content grown).  Called at init and after
        every profile / mode change."""
        try:
            self.ui.controlsFrame.setFixedHeight(
                self.controls.sizeHint().height()
                + 2 * self.ui.controlsFrame.frameWidth())
        except Exception:
            logger.debug("fit controls height failed", exc_info=True)

    def set_wrangler(self, qint):
        """Sets the wrangler based on the selected item in the dropdown.
        Syncs the wrangler's attributes and wires signals as needed.

        args:
            qint: Qt int, index of the new wrangler
        """
        if 'wrangler' in self.__dict__:
            self.disconnect_wrangler()

        self.wrangler = self.ui.wranglerStack.widget(qint)
        self.wrangler.input_q = self.command_queue
        self.wrangler.fname = self.fname
        self.wrangler.file_lock = self.file_lock
        self.wrangler.sigStart.connect(self.start_wrangler)
        if hasattr(self.wrangler, 'sigStitchRequested'):
            self.wrangler.sigStitchRequested.connect(self.start_stitch)
        self.wrangler.sigUpdateData.connect(self.update_data)
        self.wrangler.sigUpdateFile.connect(self.new_scan)
        # self.wrangler.sigUpdateFrame.connect(self.new_frame)
        self.wrangler.sigUpdateGI.connect(self.update_scattering_geometry)
        # GI move (Stage B): the wrangler hands its available SPEC motor columns
        # to the integrator's GI motor dropdown (the integrator owns selection).
        if hasattr(self.wrangler, 'sigGIMotorOptions'):
            self.wrangler.sigGIMotorOptions.connect(
                self.integratorTree.set_gi_motor_options)
            # Re-render the V2 panel so its inline GI motor combo picks up the
            # freshly-populated choices (it reads them from the integrator combo,
            # which set_gi_motor_options just repopulated).  Signature-gated, so
            # it's a no-op when the motor list is unchanged.
            self.wrangler.sigGIMotorOptions.connect(
                self._on_gi_motor_options_changed)
        self.wrangler.started.connect(self.thread_state_changed)
        self.wrangler.finished.connect(self.wrangler_finished)
        # Pause/Resume (Phase B): lift the freeze guard once paused (frozen at a
        # frame boundary), re-engage it just before resuming.  No-op for
        # wranglers that never emit these (nexus).
        self.wrangler.sigPaused.connect(self._on_run_paused)
        self.wrangler.sigResuming.connect(self._on_run_resuming)
        # CONTROLS: attach the shared run-controls to this wrangler, apply its
        # capability profile (mode items + Live/Batch/cores) and restore its
        # persisted mode with the combo's signals BLOCKED — so item population
        # can't fire a mode-change against a half-attached wrangler — then sync
        # the wrangler's mode flags once.  The staticWidget-level mode reaction
        # (_on_processing_mode_changed) is wired ONCE to the shared combo in
        # _init_child_widgets, so it isn't reconnected per wrangler here.
        prof = self.wrangler.controls_profile()
        self.wrangler.attach_controls(self.controls)
        _combo = self.controls.modeCombo
        _combo.blockSignals(True)
        self.controls.apply_profile(
            modes=prof.get('modes'), live=prof.get('live', False),
            batch=prof.get('batch', False), cores=prof.get('cores', True))
        _cur = prof.get('current')
        if _cur:
            _i = _combo.findText(_cur)
            if _i >= 0:
                _combo.setCurrentIndex(_i)
        _combo.blockSignals(False)
        self.wrangler._on_mode_changed(_combo.currentText())
        if hasattr(self.wrangler, 'sigViewerModeChanged'):
            self.wrangler.sigViewerModeChanged.connect(self._on_viewer_mode_changed)
            # Sync current viewer mode (may have been restored from session).
            # Defer to after show() so the QSplitter layout is established
            # before we collapse panels.
            vm = getattr(self.wrangler, 'viewer_mode', None)
            if vm is not None:
                QtCore.QTimer.singleShot(0, lambda v=vm: self._on_viewer_mode_changed(v))
        if hasattr(self.wrangler, 'sigStitchModeChanged'):
            self.wrangler.sigStitchModeChanged.connect(self._on_stitch_mode_changed)
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            self.wrangler.sigSavePathChanged.connect(self._sync_h5viewer_save_dir)
        # Advanced is now re-homed onto the integrator's Reintegrate row
        # (advanced_int, wired once above) so there's exactly ONE Advanced
        # button.  Hide the wrangler's old advancedButton (kept in the .ui so
        # existing layouts/refs don't break) rather than wiring it.
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            self.wrangler.ui.advancedButton.hide()
        native_gi_cfg = (
            self._controls_v2_gi_config()
            if self._controls_v2_enabled()
            else None
        )
        self.wrangler.setup()
        if native_gi_cfg is not None:
            self._controls_v2_apply_gi_config_to_scan(native_gi_cfg)
            self._push_gi_to_wrangler()
        self._configure_controls_v2_native_run_plan()
        self._sync_h5viewer_save_dir(getattr(self.wrangler, 'h5_dir', None))
        # currentTextChanged (above) only fires on a CHANGE, so seed the control
        # and display state once now.  This is especially important on a fresh
        # startup: the mode combo is restored while signals are blocked, so the
        # display must not wait for a later scan/run to learn whether the current
        # mode is 1D-only or 2D.
        self._on_processing_mode_changed(_combo.currentText())
        # E1/E2: modifier-free, plotMethod-aware overlay build for the XYE file
        # list (active only in xye mode).  Mouse presses go to the viewport, key
        # presses to the list widget — install on both.
        try:
            scans = self.h5viewer.ui.listScans
            # listScans persists across wrangler swaps, so REMOVE the prior
            # filter before installing a new one — otherwise repeated mode/source
            # swaps stack filters on the same widget (duplicate handling + a
            # small memory creep).
            prev = getattr(self, '_xye_input_filter', None)
            if prev is not None:
                try:
                    scans.viewport().removeEventFilter(prev)
                    scans.removeEventFilter(prev)
                except Exception:
                    logger.debug("removing prior XYE input filter failed",
                                 exc_info=True)
            self._xye_input_filter = _XyeOverlayInputFilter(
                scans,
                lambda: getattr(self.h5viewer, 'viewer_mode', None) == 'xye',
                lambda: self.displayframe.ui.plotMethod.currentText(),
            )
            scans.viewport().installEventFilter(self._xye_input_filter)
            scans.installEventFilter(self._xye_input_filter)
        except Exception:
            logger.debug("XYE overlay input filter install failed", exc_info=True)
        self.h5viewer.sigNewFile.connect(self.wrangler.set_fname)
        self.h5viewer.sigNewFile.connect(self.displayframe.set_axes)
        self.h5viewer.sigNewFile.connect(self.h5viewer.data_reset)
        # Stage C (2-way sync): on a .nxs load, populate the integration panel
        # from the saved scan so it shows the saved settings + reintegrate
        # reproduces them.  (Re-added per wrangler swap, like the lines above.)
        self.h5viewer.sigNewFile.connect(self._hydrate_integrator_on_load)
        # self.h5viewer.sigNewFile.connect(self.disable_displayframe_update)
        # The freshly-attached profile may have shown/hidden run rows -> refit the
        # controls bar to the new content height.
        self._fit_controls_height()
        self._refresh_controls_v2_profile(immediate=True)

    def disconnect_wrangler(self):
        """Disconnects all signals attached the the current wrangler
        """
        import warnings
        # These signals belong to the wrangler being torn down — a bare
        # disconnect() is fine (the whole wrangler is going away).
        signals = [self.wrangler.sigStart,
                   self.wrangler.sigUpdateData,
                   self.wrangler.sigUpdateFile,
                   self.wrangler.finished,
                   self.wrangler.sigPaused,
                   self.wrangler.sigResuming]
        if hasattr(self.wrangler, 'sigViewerModeChanged'):
            signals.append(self.wrangler.sigViewerModeChanged)
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            signals.append(self.wrangler.sigSavePathChanged)
        if hasattr(self.wrangler, 'sigUpdateGI'):
            signals.append(self.wrangler.sigUpdateGI)
        if hasattr(self.wrangler, 'sigGIMotorOptions'):
            signals.append(self.wrangler.sigGIMotorOptions)
        # (Advanced is no longer wired to the wrangler button — it lives on the
        # integrator's persistent advanced_int now, so there's nothing per-wrangler
        # to disconnect here.)
        for signal in signals:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    signal.disconnect()
            except (TypeError, RuntimeError, SystemError) as e:
                logger.debug("Failed to disconnect signal: %s", e)
        # h5viewer.sigNewFile is on a PERSISTENT object (the viewer survives
        # wrangler swaps), so a bare .disconnect() would also drop any future /
        # other subscriber.  Disconnect ONLY the slots set_wrangler attached.
        for slot in (getattr(self.wrangler, 'set_fname', None),
                     self.displayframe.set_axes,
                     self.h5viewer.data_reset,
                     self._hydrate_integrator_on_load):
            if slot is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.h5viewer.sigNewFile.disconnect(slot)
            except (TypeError, RuntimeError, SystemError):
                pass
        # Release the shared run-controls so the next wrangler can re-alias them
        # cleanly (drops this wrangler's tracked signal connections).
        try:
            self.wrangler.detach_controls()
        except Exception:
            logger.debug("detach_controls failed", exc_info=True)

    def _on_processing_mode_changed(self, mode_text):
        """staticWidget-level reaction to a processing-mode change.

        Wired ONCE to the shared controls' mode combo (see _init_child_widgets),
        so it survives wrangler swaps.  Per-mode integration-control state +
        forcing the display out of any stuck viewer mode for a non-viewer mode.
        """
        # This slot is connected before the active wrangler's mode handler.  Do
        # not let display geometry depend on the wrangler updating scan.skip_2d
        # first, or a fresh startup can briefly use the previous/default mode and
        # show the opposite panel.  The selected mode text is the source of truth
        # for layout.
        self._sync_processing_mode_to_scan(mode_text)
        # Per-mode integration control enable/dim (C3/C4) — runs for every
        # processing-mode change, including the viewer modes.
        self._apply_integration_control_state()
        # The mode change may show/hide the run row (hidden in viewer modes) —
        # refit the controls bar height.  BEFORE the viewer-mode early return so
        # it runs for viewer modes too (that's exactly when the row hides).
        self._fit_controls_height()
        # Skip the rest when in viewer mode — set_viewer_display_mode controls
        # panels.
        if 'Viewer' in mode_text:
            self._refresh_controls_v2_profile()
            return
        # A non-viewer processing mode (Int 1D/2D, Int 1D (XYE)) must take the
        # display OUT of any viewer mode it is stuck in.  The wrangler's
        # sigViewerModeChanged is guarded by its own _prev_viewer_mode, which
        # misses the case where the display was auto-switched to XYE after an
        # Int 1D (XYE) batch (the wrangler's viewer_mode stayed None, so _prev
        # stays '' and no reset emits).  Force the display reset here so the
        # combo and display can't desync.
        if getattr(self.displayframe, 'viewer_mode', None) is not None:
            self._on_viewer_mode_changed('')
        self.displayframe._apply_1d_only_visibility()
        # Drop any visible/cached content from the previous mode, then reload the
        # current selection for the new processing mode.  Calling update() alone
        # can leave a stale image/cake or curve visible when the new mode needs
        # data that has not been loaded yet.
        self.displayframe.clear_display_state()
        self.displayframe.request_plot_autorange()
        self.h5viewer.data_changed()
        self._refresh_controls_v2_profile()

    @staticmethod
    def _mode_skips_2d(mode_text):
        """Return whether a processing-mode label represents a 1D-only run."""
        text = str(mode_text or '')
        if 'Viewer' in text:
            return False
        return ('1D' in text) and ('2D' not in text)

    def _sync_processing_mode_to_scan(self, mode_text):
        """Synchronize scan/display 1D-only state from the selected mode text."""
        skip_2d = self._mode_skips_2d(mode_text)
        targets = [
            getattr(self, 'scan', None),
            getattr(getattr(self, 'displayframe', None), 'scan', None),
            getattr(getattr(self, 'wrangler', None), 'scan', None),
        ]
        thread = getattr(getattr(self, 'wrangler', None), 'thread', None)
        if thread is not None:
            targets.append(getattr(thread, 'scan', None))
        for target in targets:
            if target is None:
                continue
            try:
                target.skip_2d = skip_2d
            except Exception:
                logger.debug("could not sync skip_2d for %r", target,
                             exc_info=True)

    def _sync_h5viewer_save_dir(self, path, *, refresh=True):
        """Point the Scans browser at the active processed-output directory."""
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(str(path)))
        # If the processed-data dir doesn't exist yet (fresh project, no run
        # has created it), browse the nearest existing ancestor -- typically
        # the project folder -- instead of an empty nonexistent path.  Once
        # the first run creates xdart_processed_data, the next save-path
        # signal re-points the browser at it.
        probe = path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if probe and os.path.isdir(probe):
            path = probe
        self.dirname = path
        self.h5viewer.dirname = path
        if refresh:
            self.h5viewer.update_scans()

    def thread_state_changed(self):
        """Called whenever a thread is started or finished.
        """
        return

    def update_data(self, idx):
        """Called by signal from wrangler when a new frame is processed.

        Instead of rendering immediately (which blocks the main thread
        and causes frame-skipping when the wrangler is faster than the
        GUI), we update the in-memory data structures and schedule a
        throttled display refresh via a short single-shot timer.  The
        timer is started only when it isn't already pending, so during
        a fast scan burst the display refreshes at roughly the timer
        interval (~200 ms) instead of waiting for the burst to settle.
        Each flush renders the most recently received index.

        Special case: ``idx == -1`` is the batch-complete signal —
        trigger a full display refresh without touching the h5 viewer.
        """
        if idx == -1:
            # Batch mode finished — just refresh the display
            self._pending_update_idx = -1
            self._update_timer.trigger()
            return

        # A real frame was processed this run (Append-mode feedback gate).
        self._run_saw_frame = True

        # Per-frame mid-scan refresh.  Append idx to scan.frames.index
        # and bypass the file_thread.load_frame disk read by pulling the
        # freshly-integrated frame directly out of the wrangler's in-memory
        # publication slot (see wrangler_thread._published_frames).
        # The frame contains map_raw, mask, int_1d, int_2d, gi_2d, etc.
        # All we need for the displayframe — no disk hit, no file-lock
        # contention with the wrangler's per-frame write.
        try:
            # Guard the in-memory index mutation with the scan's own
            # lock — the file-thread (load_frames / set_datafile) also
            # touches this list, and h5viewer.update_data reads it on the
            # GUI thread.  This is the GUI scan's lock, distinct from
            # the wrangler scan's, so it never contends with the
            # wrangler's disk writes (no GUI stall).
            with self.scan.scan_lock:
                index = self.scan.frames.index
                if idx not in index:
                    # Common case — frames arrive in order: append without
                    # the O(N log N) re-sort.  Only out-of-order inserts
                    # (rare: reload/replace) pay for a sort, so a long scan
                    # stays O(1) per frame instead of O(N log N).
                    if not index or idx > index[-1]:
                        index.append(idx)
                    else:
                        index.append(idx)
                        index.sort()
        except AttributeError:
            # frames may briefly be None or replaced during set_datafile.
            pass

        # Per-frame: POP the freshly-integrated frame from the wrangler's slot
        # (cheap -- just moves the reference) and stash it for the coalesced flush.
        # The heavy build/upsert/scan_data used to run HERE on the GUI thread for
        # EVERY frame; once lz4 removed gzip's accidental write-throttle that
        # flooded the event loop and froze the GUI for the whole scan.  Now
        # _drain_pending_frames does it at ~5/sec over all stashed frames.  Pop
        # drains the wrangler slot so frames can't leak there.
        published = getattr(self.wrangler, "thread", None)
        if published is not None:
            frame = getattr(published, "_published_frames", {}).pop(idx, None)
            if frame is not None:
                self._pending_frames[idx] = frame

        # P4: per-frame the *only* thing we do is remember the latest
        # idx + restart the coalescing timer.  The heavy list-widget
        # rebuild (``h5viewer.update_data()``) and the cursor advance
        # (``latest_frame()``) both fire from ``_flush_pending_update``
        # after the 200 ms quiet period — running them per-frame made
        # the GUI O(N) per frame (full list clear + insertItems for
        # every new frame in a long scan), which compounded to O(N²)
        # over the run and showed up as visible stutter on slow
        # machines / very long scans.  The latest_idx assignment is
        # cheap and must stay per-frame so the flush handler knows
        # which frame to advance the cursor to.
        self.h5viewer.latest_idx = idx

        # Record the latest index and start the coalescing timer if it
        # isn't already running.  Throttle, not debounce: during a fast
        # scan burst (frame inter-arrival < timer interval) the display
        # must still refresh every ~200 ms, not only after the burst
        # settles (debounce here froze plots until end-of-scan).  The
        # Coalescer is constructed mode="throttle", so trigger() keeps
        # the pending fire instead of restarting the countdown.
        self._pending_update_idx = idx
        self._update_timer.trigger()

    def _drain_pending_frames(self):
        """Build + store publications and refresh scan_data for every frame
        stashed since the last flush.

        This is the heavy per-frame work (mask fold + publication build +
        validation + store upsert + scan_data row) moved OFF the per-event GUI
        path and batched here at ~5/sec, so GUI smoothness no longer tracks the
        frame/write rate (the whole-scan freeze, worsened when lz4 removed gzip's
        accidental write-throttle).  Display-only: the writer persists
        independently, so deferring this never loses data or touches
        persist-before-evict; the builds are stamped with the store's current
        generation (a mode switch bumps it and forces a rebuild anyway)."""
        pending = self._pending_frames
        if not pending:
            return
        self._pending_frames = {}
        import os as _os
        import time as _time
        _perf = bool(_os.environ.get("XDART_PERF"))
        t0 = _time.perf_counter()
        _t_mask = _t_build = _t_upsert = _t_scan = 0.0   # per-leg accumulators

        published = getattr(self.wrangler, "thread", None)
        global_mask = getattr(published, "mask", None) if published is not None else None
        if global_mask is not None:
            # Publish the detector gap mask ONCE per drain for the display.  We do
            # NOT fold it into each frame's own mask any more: the raw panel renders
            # via the raw_image payload, whose full-res path (_apply_detector_mask)
            # AND thumbnail gap-bake (combine_flat_masks) both apply scan.global_mask
            # DIRECTLY -- so the per-frame fold (an O(M log M) setdiff1d over the
            # large gap mask, EVERY frame) was pure redundant work and the dominant
            # drain-runaway cost.  frame.mask stays the per-frame map_raw<0; the
            # display unions scan.global_mask for the gaps.
            self.scan.global_mask = global_mask

        _is_gi = bool(getattr(self.scan, "gi", False))
        skip_2d = getattr(self.scan, "skip_2d", False)
        active_1d = (self.scan.bai_1d_args.get("gi_mode_1d", "q_total")
                     if _is_gi else None)
        active_2d = (self.scan.bai_2d_args.get("gi_mode_2d", "qip_qoop")
                     if _is_gi else None)
        from xdart.modules.ewald.scan import _coerce_scan_info

        new_rows = False
        for idx in sorted(pending):
            frame = pending[idx]
            try:
                _ts = _time.perf_counter() if _perf else 0.0
                # Step 6: key the live record under the real GI mode so a later
                # reintegrate at the same mode folds onto it.  .view is unaffected.
                publication = publication_from_live_frame(
                    frame,
                    generation=self.publication_store.generation,
                    active_mode_1d=active_1d,
                    active_mode_2d=active_2d,
                )
                if not skip_2d and publication_has_2d_errors(publication):
                    logger.warning(
                        "Skipping frame %s 2D publication: %s", idx,
                        publication_error_details(publication, "2d"))
                if _perf:
                    _t1 = _time.perf_counter(); _t_build += _t1 - _ts; _ts = _t1
                self.publication_store.upsert(publication)
                if _perf:
                    _t_upsert += _time.perf_counter() - _ts
            except Exception:
                # Non-fatal — displayframe lazy-loads from disk as fallback.
                logger.debug("In-memory frame hand-off failed for idx=%s", idx,
                             exc_info=True)
            # Accumulate the scan_data row (numeric coerced, non-numeric kept).
            info = getattr(frame, "scan_info", None)
            if info:
                coerced = _coerce_scan_info(info)
                if coerced:
                    self._scan_info_rows[int(idx)] = coerced
                    new_rows = True

        # Rebuild scan_data as ONE DataFrame from the accumulated rows -- O(N) per
        # flush, not the O(N^2) per-frame `sd.loc[idx] = ser` enlargement.  Mirrors
        # LiveScan.add_frame (heterogeneous dtypes; pandas infers per column).
        if new_rows:
            _ts = _time.perf_counter() if _perf else 0.0
            import pandas as pd
            try:
                df = pd.DataFrame.from_dict(self._scan_info_rows, orient="index")
                df.sort_index(inplace=True)
                with self.scan.scan_lock:
                    self.scan.scan_data = df
            except (ValueError, TypeError):
                logger.debug("scan_data rebuild skipped", exc_info=True)
            if _perf:
                _t_scan = _time.perf_counter() - _ts

        if _perf:
            logger.info(
                "[PERF] drain %d frame(s): mask=%.0fms build=%.0fms upsert=%.0fms "
                "scan_data=%.0fms total=%.0fms",
                len(pending), _t_mask * 1000, _t_build * 1000, _t_upsert * 1000,
                _t_scan * 1000, (_time.perf_counter() - t0) * 1000)
        logger.debug("[PERF] drained %d frame(s) in %.1f ms",
                     len(pending), (_time.perf_counter() - t0) * 1000)

    def _flush_pending_update(self):
        """Render the most recently received wrangler update.

        Called by _update_timer at most once per ~200 ms.  Coalesces
        all per-frame GUI work that doesn't have to happen immediately:

        * ``h5viewer.update_data()`` — refresh the listData widget
          (incremental append when possible — see
          :meth:`h5viewer.update_data`).
        * ``latest_frame()`` — advance the auto-last cursor to whatever
          ``latest_idx`` is now (P4: was per-frame, now per-flush so
          we don't rebuild the list widget more than once per timer
          tick).
        * ``h5viewer.data_changed()`` — publish the selected frame
          through the normal ``sigUpdate`` path exactly once.
        """
        if self._pending_update_idx is None:
            return
        self._pending_update_idx = None
        # Optional per-flush profiling: set XDART_PERF=1 in the shell to log the
        # drain / list-widget / render split at INFO so the dominant GUI-thread leg
        # is measured, not guessed.
        import os as _os
        import time as _t
        _perf = bool(_os.environ.get("XDART_PERF"))
        _t0 = _t.perf_counter() if _perf else 0.0
        # Build + store publications + scan_data for every frame stashed since the
        # last flush (the coalesced heavy work, off the per-frame GUI event loop).
        self._drain_pending_frames()
        _t1 = _t.perf_counter() if _perf else 0.0
        # Heavy list-widget refresh first — auto-last cursor needs the
        # list to contain the new index before it can select it.
        self.h5viewer.update_data(emit_update=False)
        if self.h5viewer.auto_last:
            self.latest_frame(emit_update=False)
        _t2 = _t.perf_counter() if _perf else 0.0

        method = ""
        try:
            method = self.displayframe.ui.plotMethod.currentText()
        except Exception:
            method = ""
        if self.h5viewer.auto_last and method in ("Overlay", "Waterfall"):
            # Overlay/Waterfall must show EVERY processed frame, not just the
            # frames that landed in this timer window.  Fast (non-GI) scans
            # produce several frames between ticks; selecting only the tick's
            # pending set dropped earlier curves (visible only in slow GI runs).
            # Re-select the FULL processed set each refresh — idempotent and
            # race-free, no pending-delta bookkeeping.
            with self.scan.scan_lock:
                selected = [str(int(i)) for i in self.scan.frames.index]
            if selected:
                self.h5viewer.frame_ids[:] = selected
                self.h5viewer.data_changed(show_all=True)
            else:
                self.h5viewer.data_changed()
        else:
            self.h5viewer.data_changed()  # → sigUpdate → set_data → metawidget.update()

        if _perf:
            _t3 = _t.perf_counter()
            logger.info(
                "[PERF] flush: drain=%.0fms list=%.0fms render=%.0fms total=%.0fms",
                (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
                (_t3 - _t2) * 1000, (_t3 - _t0) * 1000)

    def disable_auto_last(self, q):
        """
        Parameters
        ----------
        q : Qt.QtWidgets.QListWidgetItem
        """
        self.h5viewer.auto_last = False

    def enable_auto_last(self, q):
        """
        Parameters
        ----------
        q : Qt.QtWidgets.QListWidgetItem
        """
        self.h5viewer.auto_last = True

    def _on_display_cleared(self):
        """Viewer-mode Clear: drop the H5Viewer file-list selection so the
        cleared plot, the (now empty) selection and the title all agree.

        ``data_changed`` clears ``frame_ids`` and then early-returns on the empty
        selection (no re-render), so this won't repaint or restore a stale title.
        """
        try:
            self.h5viewer.ui.listData.clearSelection()
        except Exception:
            logger.debug("clear listData selection on display Clear failed",
                         exc_info=True)

    def set_data(self):
        """Connected to h5viewer, sets the data in displayframe based
        on the selected image or overall data.
        """
        # In viewer mode, always update display (no scan dependency)
        is_viewer = getattr(self.h5viewer, 'viewer_mode', None) is not None
        if is_viewer or self.scan.name != 'null_main':
            # Propagate the Image-Viewer file classification from the H5Viewer
            # (which classifies on file select) to the display widget (which
            # renders).  Without this, displayframe._viewer_is_xdart is always
            # False, so the Image Viewer's raw-preview payload takes the
            # *standalone* branch even for processed xdart .nxs frames and fills
            # their baked NaN mask (the inverse of the intended behaviour).
            self.displayframe._viewer_is_xdart = getattr(
                self.h5viewer, '_viewer_is_xdart', False)
            self.displayframe.update()
            # # if (len(self.frames.keys()) > 0) and (len(self.scan.frames.index) > 0):
            # if ((len(self.data_1d.keys()) > 0) and
            #         (len(self.frame_ids) > 0) and
            #         (self.frame_ids[0] != 'No data') and
            #         (len(self.scan.frames.index) > 0)):

            # Frame availability just changed (a scan finished / was loaded /
            # cleared), so refresh the integration controls — this is what
            # toggles the Reintegrate row on once a processed scan exists.
            # _apply_integration_control_state is the single source of truth
            # (mode + run-state + frames + reachable-raw + skip_2d).
            self._apply_integration_control_state()

            self.metawidget.update()
            # self.integratorTree.update()

            # Live peak-fit preview (no-op unless the dialog is open + Live on).
            self._maybe_live_fit()
            self._refresh_controls_v2_profile()

    def _hydrate_integrator_on_load(self, *args):
        """Stage C: when a ``.nxs`` is loaded, populate the integration panel from
        the saved scan (units/npts/ranges/GI), so the panel shows the saved
        reduction and Reintegrate reproduces it.  Skipped during an active run —
        the wrangler owns the config then, and the scan is mid-write."""
        if getattr(self, '_run_active', False):
            return
        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._controls_v2_hydrate_advanced_from_scan()
            self._refresh_controls_v2_profile(immediate=True)
            return
        try:
            self.integratorTree.hydrate_from_scan()
        except Exception:
            logger.debug("integrator hydrate_from_scan failed", exc_info=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the default 19/57/24 split through the launch-time window-
        # manager resize storm (first 3s); afterwards the splitter is the
        # user's.
        import time as _time
        if _time.monotonic() < getattr(self, '_split_until', 0):
            apply = getattr(self, '_apply_default_split', None)
            if callable(apply):
                apply()

    def enable_async_hydration(self):
        """Turn on off-GUI-thread rehydration of evicted frames (D2, greenfield
        Phase 3).  Called by the live app entry (``_gui_main``) — NOT during
        construction — so headless widget tests keep the synchronous blocking
        reads their assertions expect.  Idempotent + defensive."""
        try:
            df = getattr(self, 'displayframe', None)
            if df is not None and hasattr(df, 'enable_async_hydration'):
                df.enable_async_hydration()
        except Exception:
            logger.debug("enable_async_hydration failed", exc_info=True)

    def close(self):
        """Tries a graceful close.
        """
        # Block the analysis slots first: a worker signal queued just before we
        # stop + destroy must not touch the about-to-be-destroyed dialog.
        self._tearing_down = True
        # Persist the integration panel settings (the wrangler tree saves
        # continuously; the integrator panel saves here at exit).
        try:
            from xdart.utils.session import save_session
            state = {'controls_v2_int': self._controls_v2_int_session_state()}
            if not self._controls_v2_enabled():
                state['integrator'] = self.integratorTree.session_state()
            save_session(state)
        except Exception:
            logger.debug("integrator session save failed", exc_info=True)
        # Pause/Resume (Phase B): a PAUSED run blocks the wrangler thread in its
        # `while command == 'pause'` wait.  Closing the window must break that
        # wait so run() returns and the QThread isn't "destroyed while running".
        # Setting command='stop' (the universal run-end signal) exits the pause
        # wait from any state; bound-wait the thread so teardown is clean.
        try:
            w = getattr(self, 'wrangler', None)
            wt = getattr(w, 'thread', None) if w is not None else None
            if wt is not None and wt.isRunning():
                w.command = 'stop'
                wt.command = 'stop'
                # The run() finally performs the end-of-run session finish
                # (writer join up to 60s) + final .nxs flush -- 5s routinely
                # lost that race and Qt aborted on the still-running thread,
                # killing the very flush that protects the data.  30s covers
                # everything but a wedged NFS write.
                if not wt.wait(30000):
                    logger.warning("wrangler thread still finishing at "
                                   "close after 30s; final flush may be "
                                   "incomplete")
        except Exception:
            logger.debug("stopping wrangler thread on close failed", exc_info=True)
        # Reintegration thread: request a between-batches stop and wait --
        # close() never touched it, so a multi-minute reintegrate-all
        # running at close was destroyed mid-loop (Qt6 qFatal) with its
        # cached reduction session never finished.
        try:
            it = getattr(getattr(self, 'integratorTree', None),
                         'integrator_thread', None)
            if it is not None and it.isRunning():
                it.stop_requested = True
                if not it.wait(15000):
                    logger.warning("integrator thread still running at "
                                   "close after 15s")
        except Exception:
            logger.debug("stopping integrator thread on close failed",
                         exc_info=True)
        # Stop the viewer's long-running background threads BEFORE teardown so
        # the persistent fileHandlerThread / async load worker aren't destroyed
        # while running ("QThread: Destroyed while thread is still running") on
        # tab/app close.  Mirrors the GUI-test fixture teardown.
        try:
            h5v = getattr(self, 'h5viewer', None)
            if h5v is not None and hasattr(h5v, 'shutdown_threads'):
                h5v.shutdown_threads()
        except Exception:
            logger.debug("background-thread shutdown on close failed",
                         exc_info=True)
        # D2 (greenfield Phase 3): stop the off-thread frame-hydration worker
        # before teardown (same "destroyed while running" guard as above).
        try:
            df = getattr(self, 'displayframe', None)
            if df is not None and hasattr(df, 'stop_hydration_worker'):
                df.stop_hydration_worker()
            if df is not None and hasattr(df, 'stop_aggregation_worker'):
                df.stop_aggregation_worker()
        except Exception:
            logger.debug("hydration-worker shutdown on close failed",
                         exc_info=True)
        # Stop the analysis workers (live preview + batch) before teardown.
        try:
            law = getattr(self, '_live_analysis_worker', None)
            if law is not None:
                law.stop()
            baw = getattr(self, '_batch_analysis_worker', None)
            if baw is not None:
                baw.stop()
        except Exception:
            logger.debug("analysis-worker shutdown on close failed",
                         exc_info=True)
        del self.scan
        del self.displayframe.scan
        del self.frame
        del self.displayframe.frame
        super().close()

        gc.collect()

    def _show_integration_advanced(self):
        """Show a combined dialog with the integratorTree's existing
        1D and 2D advanced parameter widgets."""
        self._controls_v2_hydrate_advanced_from_scan()
        if not hasattr(self, '_integ_adv_combined_dlg'):
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Integration \u2014 Advanced Settings')
            dlg.resize(420, 450)
            layout = QtWidgets.QVBoxLayout(dlg)

            lbl1d = QtWidgets.QLabel('<b>Integrate 1D</b>')
            layout.addWidget(lbl1d)
            # Re-parent the existing advancedWidget trees into our dialog
            layout.addWidget(self.integratorTree.advancedWidget1D.tree)

            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(line)

            lbl2d = QtWidgets.QLabel('<b>Integrate 2D</b>')
            layout.addWidget(lbl2d)
            layout.addWidget(self.integratorTree.advancedWidget2D.tree)

            self._integ_adv_combined_dlg = dlg

        self._integ_adv_combined_dlg.show()
        self._integ_adv_combined_dlg.raise_()

    def enable_integration(self, enable=True):
        """Calls the integratorTree setEnabled function.
        """
        self.integratorTree.setEnabled(enable)

    def _apply_integration_control_state(self):
        """Enable/disable the integration controls for the current mode (C3/C4).

        - Int 1D / Int 1D (XYE): the 2-D integration panel is disabled — there
          is no cake in a 1D-only run.
        - Image / XYE / NeXus Viewer: the 1-D and 2-D integration panels are
          disabled (file-browser modes; the wrangler processing params are
          disabled separately via the wrangler ``tree``), but **Calibrate** and
          **Make Mask** stay enabled — they're still useful in a viewer.
        - Int 2D: everything enabled.

        While a run is active (``self._run_active``, task #71) the
        processing-affecting controls are force-disabled *on top of* the
        per-mode state: the 1-D and 2-D panels (whose children are the range
        fields, point counts, Auto toggles, unit + GI-mode combos, and the
        Re-Integrate buttons), plus **Calibrate** and **Make Mask** (they mutate
        the PONI / mask the run depends on).  The running frames use a
        deep-copied arg snapshot, but a mid-run edit would otherwise leak into
        the next scan of a multi-scan run and into a later reintegrate.  The
        frame1D/frame2D contents are plain Qt widgets (checkable ``QPushButton``
        Auto toggles, ``QComboBox`` units/modes), so ``setEnabled(False)`` here
        keeps their checked look — the pyqtgraph readonly-checkbox repaint bug
        only affects the wrangler's ParameterTree, not these.  (The Advanced
        1D/2D dialogs ARE pyqtgraph ParameterTrees; they also feed bai_*_args and
        are locked per-widget in _enter_run_state — not blanket-disabled here.)
        ``Stop`` lives on the wrangler and is left enabled; display/h5viewer
        browsing is untouched.

        Disabled widgets dim via the theme's ``:disabled`` style (D2).  Keyed
        off the processing-mode combo + run-state so it's one source of truth.
        """
        itree = getattr(self, 'integratorTree', None)
        if itree is None or not hasattr(itree, 'ui'):
            return
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            mode_text = ''
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer', 'NeXus Viewer')
        is_1d_only = mode_text in ('Int 1D', 'Int 1D (XYE)')
        # 4d: the streaming session is the authoritative run-state when present,
        # but `_run_active` remains the cache that covers the windows the session
        # can't: the start→first-frame gap (the adapter opens on the first frame)
        # and the reintegrate-via-integratorThread path (no adapter at all).  OR
        # them so controls can only ever be *more* disabled mid-run, never wrongly
        # re-enabled before `_exit_run_state` re-asserts the mode-correct state.
        # (The disk-read-guard timing stays on sigPaused/sigResuming — R7 — never
        # on these reads.)
        run_active = bool(getattr(self, '_run_active', False)) or self._session_run_active()
        ui = itree.ui
        # 2-D integration panel: only in Int 2D, and never during a run.
        frame2d = getattr(ui, 'frame2D', None)
        if frame2d is not None:
            frame2d.setEnabled(not is_viewer and not is_1d_only and not run_active)
        # 1-D integration panel: any Int mode, not viewers, never during a run.
        frame1d = getattr(ui, 'frame1D', None)
        if frame1d is not None:
            frame1d.setEnabled(not is_viewer and not run_active)
        # Calibrate / Make Mask stay enabled everywhere (incl. viewers) EXCEPT
        # during a run — they mutate the PONI / mask the run depends on.
        for name in ('pyfai_calib', 'get_mask'):
            btn = getattr(ui, name, None)
            if btn is not None:
                btn.setEnabled(not run_active)

        # Reintegrate / Advanced row tracks the SAME enable as the integration
        # panels above: available in any non-viewer Int mode, never during a run.
        # Reintegrate 2D follows frame2D exactly — also disabled in Int-1D-only
        # modes (no cake to reintegrate).  We do NOT gate on scan.frames or
        # raw-reachability: bai_1d/bai_2d no-op on an empty scan and pop a clear
        # message when raw is unreachable (R3), and probing here opened the .nxs
        # read-only — the mid-run writer crash.  So enable mirrors the panels;
        # "is there anything to reintegrate / is the raw reachable" is enforced
        # (with feedback) only when the user actually clicks.
        reint1d = getattr(ui, 'reintegrate1D', None)
        if reint1d is not None:
            reint1d.setEnabled(not is_viewer and not run_active)
        reint2d = getattr(ui, 'reintegrate2D', None)
        if reint2d is not None:
            reint2d.setEnabled(not is_viewer and not is_1d_only and not run_active)
        adv = getattr(ui, 'advanced_int', None)
        if adv is not None:
            adv.setEnabled(not is_viewer and not run_active)

        # GI (Fiber) + Threshold rows (added this cycle) follow the SAME rule as
        # the integration panels: disabled in viewer modes and during a run.
        # They were omitted before, so they stayed bright/active while the rest of
        # the integrator was greyed -- now the whole integrator dims together.
        for name in ('gi_frame', 'frame_pixreject'):
            frame = getattr(ui, name, None)
            if frame is not None:
                frame.setEnabled(not is_viewer and not run_active)

        if getattr(self, "controls_v2", None) is not None and run_active:
            self._set_controls_v2_current_fields_enabled(False)
        elif getattr(self, "controls_v2", None) is not None:
            self._refresh_controls_v2_profile(immediate=True)

    def _set_controls_v2_current_fields_enabled(self, enabled: bool) -> None:
        """Lock existing V2 editors without rebuilding the panel."""

        panel = getattr(self, "controls_v2", None)
        if panel is None:
            return
        try:
            from .ui.controls_panel_v2 import (  # local import avoids init-time Qt churn
                FormRow,
                PillRow,
                RangeRow,
                SegmentedControl,
            )
        except Exception:
            logger.debug("Controls Panel V2 lock import failed", exc_info=True)
            return

        enabled = bool(enabled)
        for row in panel.findChildren(FormRow):
            editor = getattr(row, "editor", None)
            if editor is not None:
                editor.setEnabled(enabled)
            browse = getattr(row, "browse_button", None)
            if browse is not None:
                browse.setEnabled(enabled)
        for row in panel.findChildren(RangeRow):
            for editor_name in ("_low", "_high"):
                editor = getattr(row, editor_name, None)
                if editor is not None:
                    editor.setEnabled(enabled)
            toggle = getattr(row, "_toggle", None)
            if toggle is not None:
                toggle[1].setEnabled(enabled)
        for row in panel.findChildren(PillRow):
            for _path, button in getattr(row, "_pills", ()):
                button.setEnabled(enabled)
        for row in panel.findChildren(SegmentedControl):
            for _value, button in getattr(row, "_segments", ()):
                button.setEnabled(enabled)

    def _session_run_active(self):
        """4d: True iff a streaming session is open AND reports it is running.

        Reads the wrangler's ``scan_session`` seam (the ``ScanSessionAdapter``),
        never the private slot.  Returns False when no session is open (so the
        OR with ``_run_active`` falls through to the cache) — robustly guarded so
        a duck/partial wrangler in a test never raises here."""
        wrangler = getattr(self, 'wrangler', None)
        session = getattr(wrangler, 'scan_session', None) if wrangler else None
        if session is None:
            return False
        try:
            return bool(session.is_running)
        except Exception:
            return False

    def _enter_run_state(self):
        """Single owner of run START (task #68): mark a wrangler/integrator run
        in progress.  Idempotent — re-entry while already active is a no-op so
        re-fired ``started`` signals don't double-toggle.

        Drives BOTH the display persist flag (so the 2-D panels keep their last
        content during the run, matching the 1-D plot) AND the processing-control
        disable (task #71), so the two can't desync.  Wired through the paths
        that always fire on a run start: ``start_wrangler`` (wrangler live/batch)
        and the ``integrator_thread.started`` signal (reintegrate).
        """
        if self._run_active:
            return
        self._run_active = True
        # Re-snapshot the frame-count freeze from scratch each run: clear any
        # leftover snapshot so a new run can't freeze at the PREVIOUS run's frame
        # count if no run_active=False refresh happened to clear it in between
        # (F6 — the clear in _controls_v2_state is timing-dependent; this is the
        # authoritative reset at run START).
        self._controls_v2_run_frame_count = None
        # Append-mode feedback: track whether THIS run displayed any frame, so a
        # run that processes 0 new frames (Append over an already-complete scan)
        # can still show the scan's last frame at the end (visual confirmation
        # that something happened).  Reset per run start.
        self._run_saw_frame = False
        self.displayframe.set_processing_active(True)
        # Same run-state, pushed to the h5viewer so the frame-selection disk-load
        # guard (data_changed) and the reader-side hydration guard
        # (_processing_active, just set above) share one source of truth and can't
        # drift across live/batch/reintegrate (the GUI must not read the .nxs the
        # writer is churning — that's the frame-click freeze).
        self.h5viewer.set_run_writing(True)
        self._apply_integration_control_state()   # run_active=True → disable
        # Lock the MODE row (mode combo + Batch + Cores) for the run.  A wrangler
        # run also does this via wrangler.enabled(), but a reintegrate does not —
        # so own it here (the single run-start owner).  The ACTION row stays
        # enabled (Pause/Resume/Stop).
        try:
            self.controls.set_mode_row_enabled(False)
            self.controls.set_run_row_enabled(True)
        except Exception:
            logger.debug("lock mode row on run enter failed", exc_info=True)
        # Enable the shared Stop button for the run.  For a wrangler run this is
        # redundant (the wrangler enables it via the alias); for a reintegrate it
        # is the ONLY thing that makes Stop usable -> abort + retune.
        try:
            self.controls.set_stop_enabled(True)
        except Exception:
            logger.debug("enable Stop on run enter failed", exc_info=True)
        # Reintegrate also LOCKS Start: launching a scan mid-reintegrate starts a
        # wrangler run that rebuilds scan.frames out from under the reintegrate
        # loop (the 'Frame not found' KeyError crash).  A wrangler run instead
        # morphs Start->Pause (still clickable), so only lock it for reintegrate.
        _it = getattr(getattr(self, 'integratorTree', None),
                      'integrator_thread', None)
        if _it is not None and _it.isRunning():
            try:
                self.controls.startButton.setEnabled(False)
            except Exception:
                logger.debug("disable Start on reintegrate failed",
                             exc_info=True)
        # The Advanced 1D/2D parameter dialogs also feed bai_*_args (their
        # sigUpdateArgs → get_args mutates scan.bai_1d/2d_args), so a dialog left
        # open from before the run could leak a mid-run edit into the next scan.
        # Disable them too (per-widget, as reintegrate already does via
        # integratorTree.setEnabled(False)); re-enabled by enable_integration in
        # _exit_run_state.
        itree = getattr(self, 'integratorTree', None)
        for name in ('advancedWidget1D', 'advancedWidget2D'):
            adv = getattr(itree, name, None)
            if adv is not None:
                adv.setEnabled(False)

    def _exit_run_state(self):
        """Single owner of run END (task #68): mark the run finished.
        Idempotent — exiting an already-idle state is a no-op.

        Reached on every end path including Stop and exceptions, because it is
        driven from the ``finished`` handlers (``QThread.finished`` fires
        whenever ``run()`` returns).  Ends the display persist window, then
        re-enables the integration controls and re-asserts the *mode-correct*
        per-mode state (Int 1D vs Int 2D vs viewer) rather than a blanket enable,
        so the controls are right for the current mode after the run.
        """
        if not self._run_active:
            return
        self._run_active = False
        self.displayframe.set_processing_active(False)
        # File is idle again: clear the disk-load guard.  set_run_writing(False)
        # also re-fires the standing frame selection so any frame skipped during
        # the run (evicted + disk-load suppressed) loads now from the idle file.
        self.h5viewer.set_run_writing(False)
        # Re-enable the tree (restores the auto-range field gating) then overlay
        # the mode-correct state (run_active is now False).
        self.enable_integration(True)
        self._apply_integration_control_state()
        # Unlock the mode row (matched with _enter_run_state).  A wrangler end
        # also does this via wrangler.enabled(True); both agree on the idle state.
        try:
            self.controls.set_mode_row_enabled(True)
        except Exception:
            logger.debug("unlock mode row on run exit failed", exc_info=True)
        # Run fully ended: drop the Stop button (reintegrate enabled it in
        # _enter_run_state; a wrangler end agrees on the idle state).
        try:
            self.controls.set_stop_enabled(False)
        except Exception:
            logger.debug("disable Stop on run exit failed", exc_info=True)
        # Re-enable Start (reintegrate disabled it in _enter_run_state; a wrangler
        # end resets it via its own idle morph, so True here is consistent).
        try:
            self.controls.startButton.setEnabled(True)
        except Exception:
            logger.debug("re-enable Start on run exit failed", exc_info=True)
        self._controls_v2_run_frame_count = None
        self._controls_v2_last_signature = None
        if getattr(self, "_controls_v2_batch_refresh_deferred", False):
            self._controls_v2_batch_refresh_deferred = False
        self._refresh_controls_v2_profile(immediate=True)

    def _on_stop_clicked(self):
        """Single owner of the shared Stop button — route to the active run.

        A running **reintegrate** takes priority. Stopped reintegrations roll
        back their shadow write and leave the persisted scan unchanged, so Stop
        asks before discarding the in-progress pass, then sets the integrator
        thread's cooperative ``stop_requested`` (checked between batches; one
        frame in Live mode). The thread unwinds within a frame, fires
        ``finished`` → ``integrator_thread_finished`` → ``_exit_run_state``
        (re-enabling the panel). Otherwise delegate to the active
        **wrangler**'s ``stop()`` (its run-end UI reset)."""
        st = getattr(self, 'stitch_thread', None)
        if st is not None and st.isRunning():
            # Stitch is one MultiGeometry call — not abortable mid-reduction;
            # flag it (honoured only before the heavy work starts) and give
            # immediate Stop-button feedback.  The in-flight reduction completes,
            # then finished → stitch_thread_finished → _exit_run_state.
            st.stop_requested = True
            try:
                self.controls.set_stop_enabled(False)
            except Exception:
                logger.debug("disable Stop after stitch-stop failed",
                             exc_info=True)
            return
        it = getattr(getattr(self, 'integratorTree', None),
                     'integrator_thread', None)
        if it is not None and it.isRunning():
            if not self._confirm_discard_reintegrate():
                return                                    # let it finish
            it.stop_requested = True
            try:
                self.controls.set_stop_enabled(False)   # immediate feedback
            except Exception:
                logger.debug("disable Stop after reintegrate-stop failed",
                             exc_info=True)
            return
        w = getattr(self, 'wrangler', None)
        if w is not None and hasattr(w, 'stop'):
            w.stop()

    def _confirm_discard_reintegrate(self) -> bool:
        """Modal warning before stopping a reintegrate.

        Streaming reintegrate writes into shadow groups and atomically swaps
        them only when every requested frame finishes. Stopping rolls back the
        shadow groups, so the persisted scan stays unchanged. Returns True to
        stop and discard the in-progress pass, False to keep running. Isolated
        so tests can stub it without a live dialog.
        """
        from pyqtgraph import Qt
        mb = Qt.QtWidgets.QMessageBox(self)
        mb.setIcon(Qt.QtWidgets.QMessageBox.Icon.Warning)
        mb.setWindowTitle("Stop reintegration?")
        mb.setText("Stop this reintegration?")
        mb.setInformativeText(
            "Frames processed so far are only staged in a temporary write. If "
            "you stop now, that staged work will be discarded and the saved "
            "scan will remain unchanged.\n\n"
            "Stop and discard the in-progress reintegration, or let it finish "
            "so everything is saved?")
        stop_btn = mb.addButton("Stop && Discard",
                                Qt.QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        keep_btn = mb.addButton("Let it finish",
                                Qt.QtWidgets.QMessageBox.ButtonRole.RejectRole)
        mb.setDefaultButton(keep_btn)
        mb.exec()
        return mb.clickedButton() is stop_btn

    def _on_run_paused(self):
        """Pause (Phase B): the run is FROZEN at a frame boundary (the worker has
        drained the in-flight window + flushed the .nxs and emitted sigPaused).
        LIFT the disk-read freeze guard so the user can browse ANY frame from
        disk while paused -- but the run is still active, so keep ``_run_active``
        True and leave the parameter/integration controls hard-disabled (#72).

        Safe ordering: this runs only AFTER the worker is provably idle (sigPaused
        is emitted post-drain/flush), so a disk read here can't race a write.
        ``set_run_writing(False)`` also re-fires the standing frame selection, so
        a frame skipped during the run now loads from the quiesced file."""
        if not self._run_active:
            return                       # not in a run; nothing to lift
        self.displayframe.set_processing_active(False)
        self.h5viewer.set_run_writing(False)

    def _on_run_resuming(self):
        """Resume (Phase B): RE-ENGAGE the freeze guard BEFORE the worker flips
        the command back to the run state, so a browse read can't overlap the
        restarted writer.  ``set_run_writing(True)`` also cancels any in-flight
        browse load on its rising edge.  Runs synchronously (same GUI thread)
        from the wrangler's sigResuming, ahead of the command flip."""
        if not self._run_active:
            return
        self.h5viewer.set_run_writing(True)
        self.displayframe.set_processing_active(True)

    def update_all(self, idx=None):
        """Updates all data in displays.

        This is the main-thread refresh path for the static scan tab. The
        forced ``gc.collect()`` that used to live here has been removed:
        the GIL-interacting stop-the-world pause was contributing to the
        UI stutter noted in the old TODO.  Cycle collection is left to
        the default GC schedule, which is run by CPython between
        allocation bursts.  If profiling ever shows a leak driven by
        reference cycles here, re-add a scoped ``gc.collect()`` with a
        comment explaining the specific object graph being collected.
        """
        if idx is not None:
            self.h5viewer.latest_idx = idx

        self.h5viewer.update_data()
        if self.h5viewer.auto_last:
            self.latest_frame()

        self.displayframe.update()
        self.metawidget.update()

    def integrator_thread_update(self, idx):
        """Per-frame reintegrate signal — THROTTLED.

        The reintegrate worker emits this once per frame (live = every frame).
        Rendering each synchronously floods the GUI event loop (the 2D cake is
        ~hundreds of ms a frame) so nothing paints until the run ends — the
        "freezes + no live updates" report.  Coalesce to the ~5 Hz timer (like
        the wrangler's update_data) and do the actual refresh in
        ``_flush_reintegrate_update``.  set_open_enabled is cheap + wants to be
        prompt, so it stays here."""
        self.h5viewer.set_open_enabled(True)
        self._pending_reint_idx = idx
        self._reint_update_timer.trigger()

    def _flush_reintegrate_update(self):
        """Coalesced reintegrate display refresh (≤ ~5 Hz).  Advances to the most
        recent reintegrated frame and renders it from the in-memory publication
        store (the run-write disk guard is fine — reintegrate has no concurrent
        writer until the end save)."""
        idx = self._pending_reint_idx
        self._pending_reint_idx = None
        if idx is not None:
            self.h5viewer.latest_idx = idx
        self.h5viewer.update_data()
        # Live reintegrate auto-FOLLOWS each frame as it's reduced — that's the
        # whole point (watch progress + decide whether to retune) — so advance
        # the displayed frame even when Auto-Last is off.
        it = getattr(self.integratorTree, 'integrator_thread', None)
        live_reint = bool(getattr(it, 'reintegrate_live', False))
        if self.h5viewer.auto_last or live_reint:
            self.latest_frame()
        self.displayframe.update()
        self.metawidget.update()

    def integrator_thread_finished(self):
        """Function connected to threadFinished signals for
        integratorThread
        """
        self.thread_state_changed()
        # End the run through the single run-state owner (task #68) BEFORE the
        # final refresh so the 2D panels resume normal blank-on-missing for the
        # final frame.  _exit_run_state re-enables the integration controls and
        # re-asserts the mode-correct per-mode state (it folds in the former
        # enable_integration(True) call).  Only exit if no wrangler run is still
        # in flight: a wrangler can be started while a reintegrate runs, and its
        # frames still need the controls locked — its own finished handler will
        # exit the shared run-state then (mirrors the wrangler-enable guard
        # below).
        if not self.wrangler.thread.isRunning():
            self._exit_run_state()
        self.h5viewer.set_open_enabled(True)
        self.update_all()
        if not self.wrangler.thread.isRunning():
            self.wrangler.enabled(True)

    # ── Stitch (Stitch 1D / Stitch 2D modes) ───────────────────────────
    def _stitch_status(self, msg):
        """Surface a stitch status message in the bottom status bar (via the
        active wrangler's router), falling back to the window status bar."""
        w = getattr(self, 'wrangler', None)
        if w is not None and hasattr(w, '_set_status_text'):
            w._set_status_text(msg)
            return
        try:
            self.window().statusBar().showMessage(msg)
        except Exception:
            logger.debug("stitch status failed", exc_info=True)

    def start_stitch(self, mode):
        """Launch the one-shot stitch worker for the loaded scan (Stitch 1D/2D).

        Diverted here from imageWrangler.start() when a Stitch mode is active.
        Gates on frames + geometry up front (run_stitch raises on the worker
        thread otherwise), reads the stitch params from the integrator's
        existing 1D/2D fields, and starts the worker — whose started/finished
        route through the shared run-state owner (_enter/_exit_run_state)."""
        if self.stitch_thread.isRunning() or getattr(self, '_run_active', False):
            return
        scan = self.scan
        if not getattr(scan, 'frames', None):
            self._stitch_status('Load a scan before stitching.')
            return
        if getattr(scan, 'geometry', None) is None:
            self._stitch_status(
                'Stitch needs a calibration/geometry on the scan.')
            return
        # GI guard: the GUI stitch uses the multigeometry backend, which applies
        # NO GI correction (footprint/Fresnel/refraction).  Running it with GI
        # (Fiber) mode ON would silently produce a *non-GI* merge.  The GI-corrected
        # stitch (pyfai_hist + GISettings) is gated on the real-data GIXSGUI
        # convention check, so block rather than mislead.  Toggle GI off for a
        # standard q-stitch.
        if getattr(scan, 'gi', False):
            self._stitch_status(
                'GI-corrected stitch is pending real-data validation — toggle GI '
                '(Fiber) off to run a standard (non-GI) stitch.')
            return
        try:
            params = self._build_stitch_params(mode)
        except Exception:
            logger.error("build stitch params failed", exc_info=True)
            self._stitch_status('Could not read stitch settings.')
            return
        self.stitch_thread.mode = mode
        self.stitch_thread.params = params
        self.stitch_thread.stop_requested = False
        # Fail-loud UX: a detector mask that can't be applied to the stitch
        # geometry is dropped to None by _flat_mask_as_bool with only a log
        # warning, so the stitch would silently run UNMASKED.  Surface it in
        # the run status rather than letting it pass unseen.
        if getattr(scan, 'global_mask', None) is not None and params.get('mask') is None:
            self._stitch_status(
                f'Detector mask could not be applied — stitching '
                f'({mode.upper()}) UNMASKED…')
        else:
            self._stitch_status(f'Stitching ({mode.upper()})…')
        self.stitch_thread.start()         # started -> _enter_run_state

    def _build_stitch_params(self, mode):
        """run_stitch kwargs from the integrator's existing 1D/2D fields (reused
        — no separate stitch options in Phase 1).

        The wrangler's detector/global mask is stored on the scan as flat indices
        (``scan.global_mask``) with the full-res ``scan.detector_shape``; convert
        it to the 2D boolean (True = exclude) run_stitch → pyFAI expect, reusing
        the canonical fail-soft converter (a mask that doesn't fit the detector is
        dropped with a warning, never crashes the stitch).  No `backend` kwarg:
        run_stitch is MultiGeometry-only today (the GI→histogram backend needs a
        headless change first)."""
        from xdart.modules.reduction import _flat_mask_as_bool
        args = self.scan.bai_1d_args if mode == '1d' else self.scan.bai_2d_args
        mask = _flat_mask_as_bool(
            getattr(self.scan, 'global_mask', None),
            getattr(self.scan, 'detector_shape', None),
        )
        p = dict(
            unit=args.get('unit', 'q_A^-1'),
            method=args.get('method', 'BBox'),
            radial_range=args.get('radial_range'),
            azimuth_range=args.get('azimuth_range'),
            mask=mask,
        )
        if mode == '1d':
            p['npt_1d'] = int(self.scan.bai_1d_args.get('numpoints') or 2000)
        else:
            p['npt_rad_2d'] = int(self.scan.bai_2d_args.get('npt_rad') or 1500)
            p['npt_azim_2d'] = int(self.scan.bai_2d_args.get('npt_azim') or 720)
        return p

    def _on_stitch_mode_changed(self, stitch_mode_str):
        """Route the display to/from the persistent stitch view when the wrangler
        Mode dropdown enters/leaves a Stitch mode (``'1d'``/``'2d'``/``''``).

        Only flips the flag + refreshes; ``displayFrameWidget._active_stitch_mode``
        gates the actual render on a matching ``scan.stitched_*`` result, so
        selecting Stitch before a run leaves the per-frame view untouched and
        leaving Stitch restores it."""
        self.displayframe.stitch_display_mode = stitch_mode_str or None
        self.displayframe._bump_display_generation()
        self.update_all()
        self._refresh_controls_v2_profile()

    def stitch_thread_finished(self):
        """Stitch worker done: end the shared run-state (unless a wrangler run is
        also in flight) and refresh.  On success the result becomes the persistent
        display source (StitchDisplayController) — set the flag + bump generation
        BEFORE the refresh so update_all() routes through it (it now survives
        subsequent update() calls instead of the old one-shot paint)."""
        self.thread_state_changed()
        if not self.wrangler.thread.isRunning():
            self._exit_run_state()
        self.h5viewer.set_open_enabled(True)
        if getattr(self.stitch_thread, 'ok', False):
            self.displayframe.stitch_display_mode = self.stitch_thread.mode
            self.displayframe._bump_display_generation()
            # Surface a partial skip so the merge isn't silently a subset.
            skipped = getattr(self.scan, 'stitch_skipped', None) or []
            suffix = (f' — WARNING: {len(skipped)} frame(s) skipped (no raw data)'
                      if skipped else '')
            self._stitch_status(
                f'Stitch {self.stitch_thread.mode.upper()} complete.{suffix}')
        # else: _on_stitch_error already surfaced the failure — don't overwrite.
        self.update_all()
        if not self.wrangler.thread.isRunning():
            self.wrangler.enabled(True)

    def _on_stitch_error(self, msg):
        """Stitch worker raised (caught in the worker, so the thread survived and
        still fires finished → run-state exit).  Surface the message."""
        self._stitch_status(f'Stitch failed: {msg}')
        logger.error("Stitch error: %s", msg)

    def new_scan(self, name, fname, gi, incidence_motor, single_img,
                 series_average):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
            incidence_motor: str, GI incidence-motor name (J1 rename;
                previously this slot was ``th_mtr``).  Qt signals are
                positional so the rename is purely cosmetic at this
                boundary — the value still flows through unchanged.
        """
        # Eagerly set the scan's name to the new scan's name so the
        # per-frame ``h5viewer.update_data`` path doesn't bail on the
        # ``if scan.name == "null_main": return`` guard.  Without
        # this, the scan name only became real after the async
        # ``file_thread`` processed the queued ``set_datafile``
        # command — by which time the wrangler had often moved on
        # to the next scan.  Visible symptom: in multi-scan Image
        # Directory runs the plots stayed blank during the entire
        # run and only the last scan's data appeared at the end.
        # ``self.h5viewer.set_file`` below still queues the proper
        # ``set_datafile`` so the scan's ``data_file`` and the
        # canonical name resolution (via ``scan.set_datafile``)
        # land correctly; this assignment just unblocks the
        # synchronous render path immediately.
        self.scan.name = name
        # G1/T0-1: a new run is a new data identity — drop the wavelength
        # restored from whatever file was open before, synchronously (the
        # async file-thread set_datafile also clears, but frames can render
        # in the window before it lands).  getattr: tests drive this slot
        # with duck-typed scan stubs.
        _clear_wl = getattr(self.scan, '_clear_persisted_wavelength', None)
        if callable(_clear_wl):
            _clear_wl()
        self._sync_h5viewer_save_dir(os.path.dirname(fname), refresh=False)
        self.h5viewer.set_file(fname, internal=True)   # run's own wiring
        self.scan.gi = gi
        self.scan.incidence_motor = incidence_motor
        self.scan.single_img = single_img
        self.scan.series_average = series_average
        # New scan identity: drop any prior whole-scan stitch result and leave the
        # persistent stitch display.  The scan object is REUSED across scans, so
        # stale stitched_* would otherwise keep an old merge on screen; the
        # result-existence guard then returns the display to the per-frame view.
        self.scan.stitched_1d = None
        self.scan.stitched_2d = None
        self.displayframe.stitch_display_mode = None
        self._refresh_controls_v2_profile()
        # Propagate the wrangler-loaded mask (detector + user Mask File,
        # combined into flat indices) into the main scan so the
        # displayframe can overlay it on the raw image.  Without this,
        # self.scan.global_mask stays None after a scan and no mask
        # overlay is drawn (regression introduced by the v2 refactor).
        # Sync to the wrangler thread's CURRENT mask — including ``None``.
        # ``setup()`` rebuilds ``thread.mask`` every run from (detector mask |
        # Mask File); with no detector mask and the Mask File cleared it is
        # ``None``.  The old ``if ... is not None`` guard only ever SET the
        # mask, so removing the Mask File left the previous run's mask stale on
        # ``scan.global_mask`` and it kept rendering on the raw image (and in
        # the cake payload path).  Assign unconditionally so removal clears it;
        # only skip when there is no wrangler thread at all (test stubs / pre-
        # run), where the mask state is genuinely unknown.
        _wthread = getattr(self.wrangler, 'thread', None)
        if _wthread is not None:
            self.scan.global_mask = getattr(_wthread, 'mask', None)
            # Carry the full-res detector shape too, so the display can map the
            # gap mask into thumbnail coords without a resident full-res frame.
            self.scan.detector_shape = getattr(_wthread, 'detector_shape', None)
        # Also carry the run's intensity-threshold settings so the raw-image
        # preview can show the image AS INTEGRATED (mask + threshold).
        # mask_sentinel gates the always-on uint16-65535 saturation mask on the
        # display the same way it gates it in the integration.
        for _attr in ('apply_threshold', 'threshold_min', 'threshold_max',
                      'mask_sentinel'):
            try:
                setattr(self.scan, _attr, getattr(self.wrangler, _attr))
            except Exception:
                pass

        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._controls_v2_apply_gi_config_to_scan()
        else:
            self.integratorTree.get_args('bai_1d')
            self.integratorTree.get_args('bai_2d')
            self.integratorTree.set_image_units()

        # Flush any throttled update from the *previous* scan before we
        # blow away its in-memory state.  Without this, in non-batch
        # multi-scan runs (Image Directory + Eiger) the per-frame
        # sigUpdate from the previous scan was scheduled but never had
        # a chance to render before this slot wiped the caches and
        # repointed the viewer — so the user only ever saw blanks
        # during the run and the final scan at the end.
        self._update_timer.stop()
        self._flush_pending_update()

        # Clear frame index lists (fast; rebuilt by h5viewer.update_data)
        # but DO NOT clear ``data_1d`` / ``data_2d``.  Pre-fix this
        # blanked the display the instant a new scan started.  Leaving these
        # transition/viewer dicts alone lets the previous scan's last-rendered
        # frame linger visibly until the new scan's first publication replaces
        # it on the next ``_flush_pending_update`` tick.
        self.frames.clear()
        self.frame_ids.clear()
        self.publication_store.clear()
        # Drop any frames stashed-but-not-yet-drained + the scan_data row cache
        # from the previous scan so the new scan's coalesced flush starts clean.
        self._pending_frames = {}
        self._scan_info_rows = {}
        # Reset the Overlay/Waterfall accumulator at the scan boundary so a new
        # scan (or a reprocess of the same scan) plots FRESH.  A new scan may use
        # different integration params / GI / axis, so appending its traces across
        # scans would mix incompatible data -- reset is the consistent, correct
        # choice (same for <15 curves and >15 waterfall).  update_plot also
        # self-heals on a scan-key change for any path that bypasses here.
        try:
            self.displayframe.clear_overlay()
        except Exception:
            logger.debug("clear_overlay on new scan failed", exc_info=True)

        # During a live (non-batch) run the async file-thread set_datafile
        # no longer reloads frames from disk (it would clobber the live
        # in-memory index — see fileHandlerThread.set_datafile).  So reset
        # the new scan's frame index synchronously here: drop the previous
        # scan's indices + cached frames so per-frame sigUpdate appends build
        # this scan up from empty.  Transition/viewer snapshots are
        # intentionally left populated so the prior frame lingers on screen
        # until this scan's first publication replaces it.
        if self.h5viewer.live_run_active:
            try:
                import pandas as pd
                with self.scan.scan_lock:
                    self.scan.frames.index.clear()
                    self.scan.frames._in_memory.clear()
                    # Drop the previous scan's whole-scan metadata table so the
                    # panel resets when a new scan starts; update_data rebuilds
                    # it from this scan's frames.  Batch resets via its reload
                    # path — the live run skips that, so without this the stale
                    # table lingered and metawidget.update() below re-rendered it.
                    self.scan.scan_data = pd.DataFrame()
            except AttributeError:
                pass
            # Multi-scan live runs: keep only the currently rendered frame(s)
            # from the outgoing scan so the previous image can linger until the
            # first new frame lands without dragging the recent-row mirrors
            # across scan boundaries.
            keep = set()
            try:
                df = self.displayframe
                for lst in (df.idxs, df.idxs_1d, df.idxs_2d):
                    keep.update(int(i) for i in (lst or ()))
            except Exception:
                pass
            try:
                with self.data_lock:
                    for cache in (self.data_1d, self.data_2d):
                        for k in [k for k in list(cache.keys())
                                  if int(k) not in keep]:
                            cache.pop(k, None)
                # Frame indices restart per scan: re-arm the raw self-heal
                # negative cache alongside the purge.
                self.displayframe._raw_resolve_failed = set()
                self.displayframe._raw_full_shape = None
            except Exception:
                logger.debug("live-swap cache purge skipped", exc_info=True)

        self.displayframe.set_axes()
        # self.displayframe.auto_last = True

        self.h5viewer.scan_name = name
        self.h5viewer.auto_last = True
        self.h5viewer.latest_idx = 1
        self.h5viewer.update_scans()
        self.h5viewer.update()
        # Refresh the metadata panel when a new scan starts — otherwise it
        # only repopulates on the existing set_data / update_all paths and
        # can stay empty for the first frames of a run.
        self.metawidget.update()

    def update_scattering_geometry(self, gi):
        """Connected to sigUpdateGI from wrangler. Called when scattering
        geometry changes between transmission and GI

        args:
            gi: bool, flag for determining if in Grazing incidence
        """
        scan = getattr(self, "scan", None)
        if scan is not None:
            scan.gi = gi
        # Update the integration-panel options now (the next run integrates in
        # the new geometry).  Do NOT rebuild the *display* axis combos here (C1):
        # the displayed plot is still the old-mode data, so switching the
        # plotUnit/imageUnit combos to GI/non-GI axes immediately is misleading.
        # The display combos rebuild via new_scan -> set_axes once a run actually
        # produces plots in the new mode.
        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._refresh_controls_v2_profile(immediate=True)
        else:
            self.integratorTree.set_image_units()

    def new_frame(self, frame_data):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
        """
        frame = LiveFrame(idx=frame_data['idx'], map_raw=frame_data['map_raw'],
                         mask=frame_data['mask'], scan_info=frame_data['scan_info'],
                         poni_file=frame_data['poni_file'], static=self.scan.static, gi=self.scan.gi)
        frame.int_1d = frame_data['int_1d']
        frame.int_2d = frame_data['int_2d']
        frame.map_norm = frame_data['map_norm']
        # self.data_2d[str(frame.idx)] = frame

    def start_wrangler(self):
        """Sets up wrangler, ensures properly synced args, and starts
        the wrangler.thread main method.
        """
        # i_qChi = np.zeros((1000, 1000), dtype=float)

        self._apply_controls_v2_run_state()
        self.wrangler.enabled(False)
        self.wrangler.setup()
        self._configure_controls_v2_native_run_plan()
        self.h5viewer.auto_last = True

        # Live (non-batch) runs drive the display from the in-memory
        # per-frame hand-off.  Flag the run so the async set_datafile
        # repoints the file without a disk reload and data_reset doesn't
        # wipe the live caches (the multi-scan Eiger blank-plot fix).
        # Batch / XYE-only runs keep the original reload-on-new-file
        # behaviour — their final refresh reads frames from disk.
        live = not getattr(self.wrangler.thread, 'batch_mode', False)
        self.h5viewer.live_run_active = live
        self.h5viewer.file_thread.live_run = live
        # Int 1D (XYE) writes only .xye files (no .nxs); tell the file thread
        # not to try loading a .nxs that will never exist.  Cleared in
        # wrangler_finished so a later normal open still loads from disk.
        self.h5viewer.file_thread.no_nxs = getattr(
            self.wrangler.thread, 'xye_only', False)

        # Mark the run active through the single run-state owner (task #68):
        # the 2D panels keep their last-rendered content (instead of blanking)
        # while the run's frames arrive — matching the 1D plot's persistence —
        # AND the integration controls are disabled for the run (task #71).
        # Called synchronously here (GUI thread) so the controls lock before the
        # thread starts.  Cleared in wrangler_finished.
        self._enter_run_state()

        self.wrangler.thread.start()

    def wrangler_finished(self):
        """Called by the wrangler finished signal. If current scan
        matches the wrangler scan, allows for integration.
        """
        # End the run through the single run-state owner (task #68) BEFORE the
        # final flush so the 2D panels resume normal blank-on-missing for the
        # final frame.  Idempotent: a later integrator_thread_finished() (when
        # the scan matches, below) calls _exit_run_state again as a no-op.
        # Overlap guard (symmetric with integrator_thread_finished): a wrangler
        # can be started while a reintegrate is still running, and if it
        # finishes first the reintegrate's frames still need the controls
        # locked — only exit the shared run-state when the integrator run is
        # also done; its finished handler exits it otherwise.
        # Capture once: a concurrent reintegrate-all that is still WRITING means
        # we must neither exit the shared run-state (controls stay locked) nor
        # force a reload of a possibly half-written file (review finding — the
        # internal=True auto-loads below bypass the _run_writing guard).
        _reintegrate_running = self.integratorTree.integrator_thread.isRunning()
        if not _reintegrate_running:
            self._exit_run_state()

        # Flush any pending coalesced update so the final frame is shown.
        self._update_timer.stop()
        self._flush_pending_update()

        # End the live-run window before the end-of-batch reload below:
        # the auto-load set_file(generated_file) must run the full
        # set_datafile (disk reload) so frames/scan_data come back from
        # the finished file, and data_reset must be free to clear stale
        # caches again.
        self.h5viewer.live_run_active = False
        self.h5viewer.file_thread.live_run = False
        # Clear the XYE-only no-load flag so the end-of-batch auto-load (and any
        # later normal file open) reads the .nxs from disk again.
        self.h5viewer.file_thread.no_nxs = False

        self.thread_state_changed()
        self.wrangler.stop()

        # Auto-load the final file generated from the batch if applicable
        is_batch = getattr(self.wrangler.thread, 'batch_mode', False)
        is_xye_only = getattr(self.wrangler.thread, 'xye_only', False)

        if is_batch and not is_xye_only and not _reintegrate_running:
            # Prefer the thread's fname — it's the source of truth for
            # where data was actually written. The widget-level
            # wrangler.fname is set in setup() before the thread runs
            # and may diverge (e.g. spec strips the ``_master`` suffix
            # from eiger master filenames inside the thread, so the
            # widget's fname ends with ``_master.nxs`` but the actual
            # scan output is ``<stem>.nxs``).
            generated_file = (getattr(self.wrangler.thread, 'fname', None)
                              or getattr(self.wrangler, 'fname', None))
            if generated_file and os.path.exists(generated_file):
                # Update directory display to point at the generated folder natively
                generated_dir = os.path.dirname(generated_file)
                if self.h5viewer.dirname != generated_dir:
                    self.h5viewer.dirname = generated_dir
                    self.h5viewer.update_scans()
                # Inform H5Viewer to load the file and set the flag to auto-select its last point.
                # internal=True: this is the app's own post-run wiring, not a user click —
                # the live run already pointed file_thread.fname at this same output file
                # (new_scan, set_file internal=True), so a non-internal call would hit the
                # same-file dedupe and silently skip the end-of-batch reload + select-last
                # (the "last frame doesn't show after batch" regression).  The run has ended
                # (_exit_run_state + live_run_active=False above), so bypassing the run guard
                # is safe.
                self.h5viewer._auto_select_last_on_finish = True
                self.h5viewer.set_file(generated_file, internal=True)

        # Append-mode feedback: a NON-batch run that processed 0 new frames
        # (Append over an already-complete scan/directory) displayed nothing
        # live.  Load the existing scan file and auto-select its LAST frame so
        # the user gets visual confirmation the run actually ran.  Batch already
        # auto-loads + selects-last above; XYE-only has no .nxs to load.
        if (not is_batch and not is_xye_only and not _reintegrate_running
                and not getattr(self, '_run_saw_frame', True)):
            existing_file = (getattr(self.wrangler.thread, 'fname', None)
                             or getattr(self.wrangler, 'fname', None))
            if existing_file and os.path.exists(existing_file):
                existing_dir = os.path.dirname(existing_file)
                if self.h5viewer.dirname != existing_dir:
                    self.h5viewer.dirname = existing_dir
                    self.h5viewer.update_scans()
                self.h5viewer._auto_select_last_on_finish = True
                # internal=True for the same reason as the batch branch: force the
                # reload past the same-file dedupe (the run wired file_thread.fname
                # to this file) so the last-frame select-last actually fires.
                self.h5viewer.set_file(existing_file, internal=True)

        # Live run (saw frames): the streaming path populated the display caches
        # but NOT self.scan.frames (the lazy series re-integration iterates), so
        # the Reintegrate row would stay disabled post-run.  Now that the writer
        # has closed the file, rebuild ONLY the lazy frame index from it (no
        # display reload, no scan-state reset) so reintegrate works immediately —
        # batch already gets this via its end-of-batch reload above.  Skip when a
        # reintegrate is still running (don't repoint frames mid-reintegrate) or
        # when the run saw 0 frames (the append-feedback branch already reloaded).
        _frames_index = getattr(getattr(self.scan, 'frames', None), 'index', None)
        if (not is_batch and not is_xye_only and not _reintegrate_running
                and getattr(self, '_run_saw_frame', True)
                and _frames_index is not None and len(_frames_index) == 0):
            written = (getattr(self.wrangler.thread, 'fname', None)
                       or getattr(self.wrangler, 'fname', None))
            if written and os.path.exists(written):
                try:
                    n = self.scan.load_frame_index_only(written)
                    logger.info(
                        "post-live: indexed %d frame(s) from %s for reintegrate",
                        n, os.path.basename(written))
                    self._apply_integration_control_state()
                except Exception:
                    logger.warning("post-live frame-index populate failed",
                                   exc_info=True)

        # The scan-matches branch delegates to integrator_thread_finished() to
        # run the post-integration UI enable + exit the run-state.  Skip it when
        # a real reintegrate is still running (the overlap case): calling it
        # would exit the shared run-state and re-enable the controls mid-
        # reintegrate.  In that case just re-enable the wrangler (its run IS
        # done); the integrator's own finished handler exits the run-state when
        # the reintegrate completes.
        if (self.scan.name == self.wrangler.scan_name
                and not self.integratorTree.integrator_thread.isRunning()):
            self.integrator_thread_finished()
        else:
            self.wrangler.enabled(True)

        gc.collect()

        # XYE-only batch (Int 1D (XYE)): there is no .nxs to auto-load, so the
        # block above skipped the end-of-batch reload.  Show the folder of
        # generated iq_/itth_ files (written to <scan_dir>/<scan_name> by
        # save_1d) in XYE Viewer mode so the outputs are actually listed.
        # Done last so integrator_thread_finished()'s refresh doesn't undo it.
        if is_batch and is_xye_only:
            try:
                xye_dir = os.path.join(
                    os.path.dirname(self.scan.data_file), self.scan.name)
                if os.path.isdir(xye_dir):
                    self.h5viewer.dirname = xye_dir
                    # Same path the XYE Viewer combo takes: set viewer_mode,
                    # panels, selection mode, and refresh listScans.
                    self._on_viewer_mode_changed('xye')
                    # Auto-select the most recently *written* file (by mtime),
                    # not the name-last one, so the final pattern from this run
                    # is shown without a manual click.
                    self.h5viewer.select_most_recent_scan_entry()
                else:
                    logger.debug(
                        'XYE-only batch finished but output dir not found: %s',
                        xye_dir)
            except Exception:
                logger.debug(
                    'Could not show XYE output folder after batch',
                    exc_info=True)

    def _on_viewer_mode_changed(self, viewer_mode_str):
        """Enable or disable the integrator panel and update h5viewer for viewer mode.

        Args:
            viewer_mode_str: 'image', 'xye', or '' (normal mode)
        """
        viewer_mode = viewer_mode_str or None  # '' → None
        is_viewer = viewer_mode is not None
        is_file_viewer = viewer_mode in ('image', 'xye')
        from PySide6.QtWidgets import QAbstractItemView

        scans = self.h5viewer.ui.listScans
        prev_suspend = getattr(
            self.h5viewer, '_suspend_scan_selection_loads', False,
        )
        was_blocked = scans.blockSignals(True)
        self.h5viewer._suspend_scan_selection_loads = True
        try:
            self.h5viewer.viewer_mode = viewer_mode
            set_cache_limit = getattr(self, "_set_1d_cache_limit", None)
            if callable(set_cache_limit):
                set_cache_limit(
                    None if is_viewer else _DISPLAY_1D_CACHE_MAX)
            tree = getattr(self.wrangler, 'tree', None)
            if tree is not None:
                # Only the actual file-Viewer *processing* modes disable the
                # wrangler inputs.  Int 1D (XYE) is a processing mode whose
                # display auto-switches to XYE to list the generated files, but
                # its inputs (Image File / mask / …) must stay enabled so the
                # user can keep processing.  Key off the processing-mode combo,
                # not the display viewer_mode; fall back to the display when the
                # combo is unavailable.
                mode_text = ''
                try:
                    mode_text = self.controls.current_mode()
                except Exception:
                    mode_text = ''
                # Tree stays enabled in viewers: processing groups are
                # disabled per-group by the wrangler, while Project Folder /
                # Save Path remain usable (they drive the file browser).
                tree.setEnabled(True)
            # Per-mode integration control enable/dim (C3/C4): disable the 1-D/2-D
            # integration panels in viewers, keep Calibrate / Make Mask enabled.
            self._apply_integration_control_state()
            # Relax the Frames panel width so NeXus dataset labels aren't
            # clipped; restored on exit / other modes.
            self.h5viewer._apply_frames_panel_width(viewer_mode)
            if hasattr(self, "metawidget"):
                self.metawidget.viewer_mode = viewer_mode
            # Give displayframe a reference to the wrangler for mask/threshold
            self.displayframe._wrangler = self.wrangler if is_viewer else None
            # In viewer mode, disable New/Save (keep Open Folder and Export)
            self.h5viewer.actionNewFile.setEnabled(not is_viewer)
            self.h5viewer.actionSaveDataAs.setEnabled(not is_viewer)
            # XYE viewer: ExtendedSelection so arrow keys browse one file at a
            # time with the plot following (shift+arrow / shift+click = range);
            # _XyeOverlayInputFilter layers on modifier-free, plotMethod-aware
            # accumulation (toggle/extend on click+arrow in Overlay/Waterfall/
            # Sum/Average).  Others: single select.  Start clean — show the
            # current row only, never a default overlay.
            if viewer_mode == 'xye':
                scans.setSelectionMode(QAbstractItemView.ExtendedSelection)
                scans.clearSelection()
            else:
                scans.setSelectionMode(QAbstractItemView.SingleSelection)
            # Configure display panels for the viewer mode
            self.displayframe._viewer_is_xdart = False
            self.displayframe.set_viewer_display_mode(viewer_mode)
            if is_viewer:
                save_path = getattr(self.wrangler, 'h5_dir', None)
                current_raw = str(getattr(self.h5viewer, 'dirname', '') or '')
                current_dir = (
                    os.path.abspath(os.path.expanduser(current_raw))
                    if current_raw else ''
                )
                default_dir = os.path.abspath(os.path.expanduser(
                    str(getattr(self, 'local_path', get_fname_dir())),
                ))
                if save_path and (not current_dir or current_dir == default_dir):
                    self._sync_h5viewer_save_dir(save_path, refresh=False)
                self.h5viewer.enter_viewer_mode_cleanup()
            else:
                self.h5viewer.cancel_pending_loads()
                if hasattr(self, 'scan') and hasattr(self.scan, 'global_mask'):
                    self.scan.global_mask = None
                self.displayframe.clear_display_state()
            # Refresh scan list to show/hide appropriate file types
            self.h5viewer.update_scans()
        finally:
            self.h5viewer._suspend_scan_selection_loads = prev_suspend
            scans.blockSignals(was_blocked)
        self._fit_controls_height()
        self._refresh_controls_v2_profile()

    def latest_frame(self, checked=None, *, emit_update=True):
        """Advances to last frame in data list, updates displayframe, and
        set auto_last to True.

        Wraps the cursor advance in ``blockSignals`` and invokes
        ``data_changed()`` explicitly — same pattern as
        ``H5Viewer.update_data``.  Without the block, the
        ``ClearAndSelect`` cursor move would fire
        ``itemSelectionChanged`` → ``data_changed`` → ``sigUpdate`` →
        ``set_data`` → a redundant ``displayframe.update()`` on top of
        whatever the caller does next (every call site that drives
        ``latest_frame`` follows it with its own display refresh).
        """
        self.h5viewer.auto_last = True
        if self.h5viewer.ui.listData.count() <= 1:
            return

        lw = self.h5viewer.ui.listData
        idx = self.h5viewer.latest_idx
        lw.blockSignals(True)
        try:
            if isinstance(idx, int):
                items = lw.findItems(str(idx), QtCore.Qt.MatchExactly)
                for item in items:
                    self.h5viewer.set_current_frame(item)
            else:
                last_row = lw.count() - 1
                if last_row >= 0:
                    item = lw.item(last_row)
                    if item is not None:
                        try:
                            self.h5viewer.latest_idx = int(item.text())
                        except ValueError:
                            self.h5viewer.latest_idx = item.text()
                    self.h5viewer.set_current_frame(last_row)
        finally:
            lw.blockSignals(False)
        if emit_update:
            self.h5viewer.data_changed()

    def raw_to_tiff(self):
        self.popup_detector_options()

    def popup_detector_options(self):
        """
        Popup Qt Window to select options for Waterfall Plot
        Options include Y-axis unit and number of points to skip
        """
        if self.detector_dialog.layout() is None:
            self.setup_detector_options_widget()

        self.detector_dialog.show()

    def setup_detector_options_widget(self):
        """
        Setup y-axis option for Waterfall plot
        Setup first image and step size for wf and overlay plots
        """
        layout = QtWidgets.QGridLayout()
        self.detector_dialog.setLayout(layout)

        self.detector_widget = QCombo()
        accept_button = QtWidgets.QPushButton('Okay')
        cancel_button = QtWidgets.QPushButton('Cancel')

        layout.addWidget(QtWidgets.QLabel('Choose Detector'), 0, 0)
        layout.addWidget(self.detector_widget, 1, 0)
        layout.addWidget(accept_button, 2, 1)
        layout.addWidget(cancel_button, 2, 2)

        detectors = ['Pilatus 1M', 'Pilatus 100k', 'Pilatus 300kw']
        self.detector_widget.addItems(detectors)

        accept_button.clicked.connect(self.set_detector)
        cancel_button.clicked.connect(self.close_detector_popup)

    def close_detector_popup(self):
        self.detector_dialog.close()

    def set_detector(self):
        detector_name = self.detector_widget.currentText()
        self.detector = pyFAI.detector_factory(name=detector_name)
        self.detector_dialog.close()

        rawFile, _ = QFileDialog().getOpenFileName(
            filter='RAW (*.raw)',
            caption='Choose Raw File',
            options=QFileDialog.DontUseNativeDialog
        )

        if os.path.isfile(rawFile):
            img = get_img_data(rawFile, self.detector, return_float=False)
            if img is not None:
                tifFile = os.path.splitext(rawFile)[0] + '.tif'
                imageio.imwrite(tifFile, img)
                message = f'{os.path.basename(tifFile)} saved'
            else:
                message = 'File does not match detector..'
        else:
            message = 'Invalid Raw File'

        out_dialog = QMessageBox()
        out_dialog.setText(message)
        out_dialog.exec()
