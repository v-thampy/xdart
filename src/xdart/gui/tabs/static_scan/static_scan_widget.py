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
from .ui.staticUI import Ui_Form
from .h5viewer import H5Viewer
from .display_frame_widget import displayFrameWidget
from .integrator import integratorTree
from .metadata import metadataWidget
from .wranglers import imageWrangler, nexusWrangler, wranglerWidget
from xdart.utils.throttle import Coalescer
from xdart.utils._utils import FixSizeOrderedDict, get_fname_dir, get_img_data

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
        try:
            return self._get_method() in self._ACCUMULATING
        except Exception:
            return False

    @staticmethod
    def _is_data_item(item):
        if item is None:
            return False
        text = item.text()
        return text != '..' and not text.endswith('/')

    def eventFilter(self, obj, event):
        if not self._is_active():
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
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if has_shift:
            return False                  # let Qt handle shift range-select
        try:
            pos = event.position().toPoint()
        except AttributeError:            # Qt5 fallback
            pos = event.pos()
        item = self._list.itemAt(pos)
        if not self._is_data_item(item):
            return False
        if not (has_toggle_mod or self._accumulating()):
            return False                  # Single plain click: Qt replace
        # Accumulating (or explicit ctrl/cmd-toggle): toggle this file in/out of
        # the overlay via the selection model (robust in ExtendedSelection).
        sm = self._list.selectionModel()
        idx = self._list.indexFromItem(item)
        sm.select(idx, QtCore.QItemSelectionModel.Toggle)
        sm.setCurrentIndex(idx, QtCore.QItemSelectionModel.NoUpdate)
        return True

    def _on_key(self, event):
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if (not self._accumulating()
                or has_shift or has_toggle_mod
                or event.key() not in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down)):
            return False                  # Single / modified: Qt default browse
        step = -1 if event.key() == QtCore.Qt.Key_Up else 1
        row = self._list.currentRow() + step
        while 0 <= row < self._list.count():
            item = self._list.item(row)
            if self._is_data_item(item):
                # Extend: add the newly-current file without clearing the rest,
                # so arrow-browsing builds the comparison set.
                item.setSelected(True)
                self._list.setCurrentItem(
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
        # Default panel proportions: middle (image/plot) panels ~10% wider
        # than Qt's hint-based split (Vivek).  Applied via singleShot AFTER
        # the window has real geometry -- setSizes at __init__ ran before the
        # main window's resize() and got redistributed away.
        def _default_split():
            try:
                total = sum(self.ui.mainSplitter.sizes()) or 1000
                # Right (integration/wrangler) column reduced ~15% (0.24 ->
                # 0.204; it was too wide on startup, esp. on Windows); the
                # freed width goes to the middle display panels (Vivek).
                self.ui.mainSplitter.setSizes(
                    [int(total * f) for f in (0.19, 0.606, 0.204)])
                self.ui.mainSplitter.setStretchFactor(1, 1)
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
            if _integ:
                self.integratorTree.restore_session_state(_integ)
        except Exception:
            logger.debug("integrator session restore failed", exc_info=True)

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
        # Live analysis preview (analyzer framework Step 3): a latest-wins
        # background worker re-fits the newest frame while the dialog's "Live"
        # toggle is on.  Lazily created on first live request; generation gates
        # stale results.
        self._live_analysis_worker = None
        self._live_fit_gen = 0
        # Batch analysis (Step 4): one worker fits every frame; the results
        # popup plots parameters vs frame number.
        self._batch_analysis_worker = None
        self._batch_results_dialog = None
        self._batch_x_unit = ""
        self._build_tools_placeholder()

    def _build_tools_placeholder(self):
        """Fill the vacated bottom-left ``metaFrame`` with the 'Tools' card.

        Reclaims the corner freed by moving the metadata table into a popup.
        Active tools get an 'Open' button; not-yet-built ones show a disabled
        PLANNED chip.  Peak Fitting is wired (``_open_peak_fit_dialog``)."""
        lay = QtWidgets.QVBoxLayout(self.ui.metaFrame)
        lay.setContentsMargins(13, 11, 13, 13)
        lay.setSpacing(8)

        header = QtWidgets.QLabel('TOOLS')
        header.setObjectName('toolsHeader')
        lay.addWidget(header)

        card = QtWidgets.QFrame()
        card.setObjectName('toolsPlaceholder')
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(11, 11, 11, 11)
        card_lay.setSpacing(8)
        # (label, handler-or-None).  Handler => active tool with an Open button.
        tools = [
            ('Peak Fitting', self._open_peak_fit_dialog),
            ('Plot Metadata', None),
        ]
        for name, handler in tools:
            row = QtWidgets.QWidget()
            row_lay = QtWidgets.QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(8)
            dot = QtWidgets.QFrame()
            dot.setObjectName('toolDot')
            dot.setFixedSize(7, 7)
            label = QtWidgets.QLabel(name)
            label.setObjectName('toolLabel')
            row_lay.addWidget(dot)
            row_lay.addWidget(label)
            row_lay.addStretch(1)
            if handler is not None:
                open_btn = QtWidgets.QPushButton('Open')
                open_btn.setObjectName('toolOpen')
                open_btn.clicked.connect(handler)
                row_lay.addWidget(open_btn)
            else:
                chip = QtWidgets.QLabel('PLANNED')
                chip.setObjectName('toolChip')
                row_lay.addWidget(chip)
                row.setEnabled(False)      # reads as inactive (D2 disabled styling)
            card_lay.addWidget(row)
        lay.addWidget(card)

        note = QtWidgets.QLabel(
            'Peak Fitting fits the selected frame’s 1-D pattern. '
            'Plot Metadata is planned.')
        note.setObjectName('toolsNote')
        note.setWordWrap(True)
        lay.addWidget(note)
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

    def _open_peak_fit_dialog(self):
        """Open (or re-show) the Peak Fitting popup — lazy, single-instance,
        non-modal (so the live scan + frame browsing stay responsive; Reload
        re-grabs the current frame)."""
        if self._peak_fit_dialog is None:
            from .peak_fit_dialog import PeakFitDialog
            self._peak_fit_dialog = PeakFitDialog(
                self._current_pattern_for_fit, parent=self)
            # Toggling Live on re-fits the current frame at once (then every new
            # frame, via set_data); off just stops pushing.
            self._peak_fit_dialog.live_check.toggled.connect(
                self._on_live_fit_toggled)
            self._peak_fit_dialog.batch_btn.clicked.connect(self._on_batch_clicked)
        dlg = self._peak_fit_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.refresh_pattern()

    def _on_live_fit_toggled(self, on):
        """Live checkbox flipped — fit the current frame immediately on enable so
        there's no wait for the next frame; disabling just stops the pushes."""
        if on:
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
        data = self._current_pattern_for_fit()
        if not data:
            return
        x, y, label = data
        dlg.set_live_pattern(x, y, label)   # show the data now; fit overlays async
        req = dlg.build_fit_request()
        if req is None:                      # nothing fittable (status set by dialog)
            return
        inp, analyzer = req
        self._live_fit_gen += 1
        self._ensure_live_analysis_worker().request(
            label, self._live_fit_gen, analyzer, inp)

    def _on_live_analyzed(self, label, generation, outcome):
        """Draw a live fit result — but only if it's still the newest request and
        the dialog is still open + Live (a stale or superseded result is dropped,
        so the overlay never lags behind the displayed frame)."""
        if generation != self._live_fit_gen:
            return
        dlg = self._peak_fit_dialog
        if dlg is None or not dlg.isVisible() or not dlg.live_check.isChecked():
            return
        if outcome is not None and outcome.ok:
            dlg._draw_outcome(outcome, auto=dlg.auto_check.isChecked())

    # ---- Batch peak fit (Step 4) ---------------------------------------
    def _on_batch_clicked(self):
        """Batch button: start a batch fit, or cancel one already in flight."""
        worker = self._batch_analysis_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            return
        self._run_batch_fit()

    def _run_batch_fit(self):
        """Fit every frame in the scan with the dialog's current settings, then
        (on completion) plot the parameters vs frame number.

        The fit model is fixed ONCE from the current frame (auto-detect /
        positions / peak count via the dialog's ``build_fit_request``) and the
        SAME analyzer is applied to every frame, so each parameter series tracks
        the same peak across frames.  The range is re-applied per frame."""
        import numpy as np
        from xrd_tools.analysis.runner import AnalysisInput
        dlg = self._peak_fit_dialog
        if dlg is None:
            return
        if dlg._x is None or dlg._y is None:
            dlg.refresh_pattern()
        req = dlg.build_fit_request()
        if req is None:
            return                              # status set by the dialog
        _, analyzer = req
        lo, hi = dlg._fit_range()
        try:
            frame_idxs = list(self.scan.frames.index)
        except Exception:
            frame_idxs = []
        if not frame_idxs:
            dlg.status.setText("No frames to batch-fit.")
            return
        x_unit = dlg._x_label
        inputs = []
        for idx in frame_idxs:
            data = self._pattern_for_frame(idx)
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
            self._batch_analysis_worker.sigBatchDone.connect(self._on_batch_done)
        self._batch_x_unit = x_unit
        self._batch_analysis_worker.configure(analyzer, inputs)
        dlg.set_batch_running(True)
        dlg.set_batch_progress(0, len(inputs))
        self._batch_analysis_worker.start()

    def _on_batch_progress(self, done, total):
        dlg = self._peak_fit_dialog
        if dlg is not None:
            dlg.set_batch_progress(done, total)

    def _on_batch_done(self, labels, columns):
        """Batch finished: re-enable the dialog and open the vs-frame results
        popup (or report a cancel)."""
        dlg = self._peak_fit_dialog
        if dlg is not None:
            dlg.set_batch_running(False)
        if labels is None:                      # cancelled before completion
            if dlg is not None:
                dlg.status.setText("Batch fit cancelled.")
            return
        if dlg is not None:
            dlg.status.setText(f"Batch fit done — {len(labels)} frames.")
        if self._batch_results_dialog is None:
            from .batch_fit_results_dialog import BatchFitResultsDialog
            self._batch_results_dialog = BatchFitResultsDialog(self)
        rd = self._batch_results_dialog
        rd.set_results(labels, columns, x_unit=self._batch_x_unit)
        rd.show()
        rd.raise_()
        rd.activateWindow()

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
        """Inject the integrator's CURRENT pixel-rejection policy into the active
        wrangler's (now-hidden) Mask / MaskSat params, so a LIVE run applies the
        SAME Intensity-Threshold / Mask-Saturated rejection that Reintegrate
        does — the integrator is the single source of truth.

        Called from ``start_wrangler`` BEFORE ``wrangler.setup()`` (which reads
        those params and pushes them to the thread).  Per-field guarded: a
        wrangler without an 'Intensity Threshold' group (e.g. NeXus) just skips
        it, and still receives 'Mask Saturated'.
        """
        try:
            cfg = self.integratorTree.get_threshold_config()
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
        """Inject the integrator's CURRENT GI geometry into the active wrangler's
        (now-hidden) GI carrier params, so a LIVE run uses the SAME GI geometry
        the integrator shows (and Reintegrate reads).  Called from
        ``start_wrangler`` BEFORE ``wrangler.setup()`` (which reads the GI params
        into the thread).  Per-field guarded: a wrangler without a 'GI' group
        just skips."""
        try:
            cfg = self.integratorTree.get_gi_config()
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
        self.wrangler.sigUpdateData.connect(self.update_data)
        self.wrangler.sigUpdateFile.connect(self.new_scan)
        # self.wrangler.sigUpdateFrame.connect(self.new_frame)
        self.wrangler.sigUpdateGI.connect(self.update_scattering_geometry)
        # GI move (Stage B): the wrangler hands its available SPEC motor columns
        # to the integrator's GI motor dropdown (the integrator owns selection).
        if hasattr(self.wrangler, 'sigGIMotorOptions'):
            self.wrangler.sigGIMotorOptions.connect(
                self.integratorTree.set_gi_motor_options)
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
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            self.wrangler.sigSavePathChanged.connect(self._sync_h5viewer_save_dir)
        # Advanced is now re-homed onto the integrator's Reintegrate row
        # (advanced_int, wired once above) so there's exactly ONE Advanced
        # button.  Hide the wrangler's old advancedButton (kept in the .ui so
        # existing layouts/refs don't break) rather than wiring it.
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            self.wrangler.ui.advancedButton.hide()
        self.wrangler.setup()
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

    def _hydrate_integrator_on_load(self, *args):
        """Stage C: when a ``.nxs`` is loaded, populate the integration panel from
        the saved scan (units/npts/ranges/GI), so the panel shows the saved
        reduction and Reintegrate reproduces it.  Skipped during an active run —
        the wrangler owns the config then, and the scan is mid-write."""
        if getattr(self, '_run_active', False):
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
        # Persist the integration panel settings (the wrangler tree saves
        # continuously; the integrator panel saves here at exit).
        try:
            from xdart.utils.session import save_session
            save_session({'integrator': self.integratorTree.session_state()})
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

        self.wrangler.enabled(False)

        self.integratorTree.get_args('bai_1d')
        self.integratorTree.get_args('bai_2d')

        args = {'bai_1d_args': self.scan.bai_1d_args,
                'bai_2d_args': self.scan.bai_2d_args}
        self.wrangler.scan_args = copy.deepcopy(args)
        # Pixel rejection is owned by the integrator panel now — push the
        # current Threshold / Mask-Saturated policy into the wrangler BEFORE
        # setup() so the live run applies exactly what Reintegrate would.
        self._push_threshold_to_wrangler()
        # GI geometry is owned by the integrator panel now — push it into the
        # wrangler's hidden GI carrier params BEFORE setup() too, same reason.
        self._push_gi_to_wrangler()
        self.wrangler.setup()
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
