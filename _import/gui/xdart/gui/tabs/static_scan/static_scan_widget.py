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

# Bound the in-memory frame cache so a 1000-frame Eiger scan doesn't
# pile up ~30 GB of map_raw arrays in h5viewer.data_2d.  When we add a
# new entry, evict the oldest non-current frame; user-scrolling back to
# old frames will lazy-load them from disk again via file_thread.
# Keep large enough that recent frames + the auto-last selection don't
# thrash; small enough that peak memory stays sane.
_FRAME_CACHE_MAX = 32

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
        # O4: both 1D and 2D caches are bounded with the same cap.
        # Pre-O4 ``data_1d`` was an unbounded OrderedDict while
        # ``data_2d`` was FixSizeOrderedDict(max=20), and the manual
        # eviction loop keyed off ``len(data_2d) > 32`` could never
        # fire because data_2d auto-capped at 20.  Net effect: data_1d
        # grew without bound on long live runs (a frame's int_1d copy
        # is small but ~10k frames still added up).  Same cap on both
        # keeps the two snapshots roughly in sync — they're separate
        # dicts (the snapshots they hold are different sizes, so a
        # single shared store would lose typing precision), but
        # FixSizeOrderedDict's farthest-from-new-key eviction policy
        # converges on roughly the same active window.
        self.data_1d = FixSizeOrderedDict(max=_FRAME_CACHE_MAX)
        self.data_2d = FixSizeOrderedDict(max=20)

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
        self.ui.middleFrame.setLayout(self.displayframe.ui.layout)

        # IntegratorTree
        self.integratorTree = integratorTree(
            self.scan, self.frame, self.file_lock,
            self.frames, self.frame_ids, self.data_1d, self.data_2d,
            data_lock=self.data_lock,
            publication_store=self.publication_store)
        self.ui.integratorFrame.setLayout(self.integratorTree.ui.verticalLayout)
        if len(self.scan.frames.index) > 0:
            self.integratorTree.update()
        self.integratorTree.ui.raw_to_tif.hide()

        # Metadata
        self.metawidget = metadataWidget(self.scan, self.frame,
                                         self.frame_ids, self.frames,
                                         data_1d=self.data_1d,
                                         publication_store=self.publication_store,
                                         data_lock=self.data_lock)
        self.ui.metaFrame.setLayout(self.metawidget.layout)

    def _connect_signals(self):
        """Wire signal/slot connections for H5Viewer, DisplayFrame, and Integrator."""
        # H5Viewer signals
        self.h5viewer.sigUpdate.connect(self.set_data)
        self.h5viewer.file_thread.sigTaskStarted.connect(self.thread_state_changed)
        self.h5viewer.sigThreadFinished.connect(self.thread_state_changed)
        self.h5viewer.ui.listData.itemClicked.connect(self.disable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.enable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.latest_frame)

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

        # Integrator signals
        self.integratorTree.integrator_thread.started.connect(self.thread_state_changed)
        # Re-integration is a "run" too: route its START through the single
        # run-state owner (task #68) — keeps the 2D panels persistent AND
        # disables the processing controls (task #71) for its duration; cleared
        # in integrator_thread_finished via _exit_run_state.
        self.integratorTree.integrator_thread.started.connect(self._enter_run_state)
        self.integratorTree.integrator_thread.update.connect(self.integrator_thread_update)
        self.integratorTree.integrator_thread.finished.connect(self.integrator_thread_finished)

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
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(200)  # ms
        self._update_timer.timeout.connect(self._flush_pending_update)

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
        self.wrangler.started.connect(self.thread_state_changed)
        self.wrangler.finished.connect(self.wrangler_finished)
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'processingModeCombo'):
            def _on_mode_changed(mode_text):
                # Per-mode integration control enable/dim (C3/C4) — runs for
                # every processing-mode change, including the viewer modes.
                self._apply_integration_control_state()
                # Skip the rest when in viewer mode — set_viewer_display_mode
                # controls panels.
                if 'Viewer' in mode_text:
                    return
                # A non-viewer processing mode (Int 1D/2D, Int 1D (XYE)) must take
                # the display OUT of any viewer mode it is stuck in.  The
                # wrangler's sigViewerModeChanged is guarded by its own
                # _prev_viewer_mode, which misses the case where the display was
                # auto-switched to XYE after an Int 1D (XYE) batch (the wrangler's
                # viewer_mode stayed None, so _prev stays '' and no reset emits).
                # Force the display reset here so the combo and display can't desync.
                if getattr(self.displayframe, 'viewer_mode', None) is not None:
                    self._on_viewer_mode_changed('')
                self.displayframe._apply_1d_only_visibility()
                # Drop any visible/cached content from the previous mode,
                # then reload the current selection for the new processing
                # mode. Calling update() alone can leave a stale image/cake
                # or curve visible when the new mode needs data that has not
                # been loaded yet.
                self.displayframe.clear_display_state()
                self.displayframe.request_plot_autorange()
                self.h5viewer.data_changed()
            self.wrangler.ui.processingModeCombo.currentTextChanged.connect(_on_mode_changed)
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
        # Wire the wrangler's Advanced button to show the integratorTree's
        # existing 1D/2D advanced parameter dialogs in a combined popup.
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            self.wrangler.ui.advancedButton.clicked.connect(
                self._show_integration_advanced)
        self.wrangler.setup()
        self._sync_h5viewer_save_dir(getattr(self.wrangler, 'h5_dir', None))
        # E1/E2: modifier-free, plotMethod-aware overlay build for the XYE file
        # list (active only in xye mode).  Mouse presses go to the viewport, key
        # presses to the list widget — install on both.
        try:
            scans = self.h5viewer.ui.listScans
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
        # self.h5viewer.sigNewFile.connect(self.disable_displayframe_update)

    def disconnect_wrangler(self):
        """Disconnects all signals attached the the current wrangler
        """
        import warnings
        signals = [self.wrangler.sigStart,
                   self.wrangler.sigUpdateData,
                   self.wrangler.sigUpdateFile,
                   self.wrangler.finished,
                   self.h5viewer.sigNewFile]
        if hasattr(self.wrangler, 'sigViewerModeChanged'):
            signals.append(self.wrangler.sigViewerModeChanged)
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            signals.append(self.wrangler.sigSavePathChanged)
        # Disconnect Advanced button from integration popup
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            try:
                self.wrangler.ui.advancedButton.clicked.disconnect(
                    self._show_integration_advanced)
            except (TypeError, RuntimeError) as e:
                logger.debug("Failed to disconnect Advanced button signal: %s", e)
        for signal in signals:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    signal.disconnect()
            except (TypeError, RuntimeError, SystemError) as e:
                logger.debug("Failed to disconnect signal: %s", e)

    def _sync_h5viewer_save_dir(self, path, *, refresh=True):
        """Point the Scans browser at the active processed-output directory."""
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(str(path)))
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
            if not self._update_timer.isActive():
                self._update_timer.start()
            return

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

        # Consume the published frame from the wrangler thread.
        published = getattr(self.wrangler, "thread", None)
        if published is not None:
            frame = getattr(published, "_published_frames", {}).pop(idx, None)
        else:
            frame = None
        if frame is not None:
            try:
                global_mask = getattr(published, "mask", None)
                if global_mask is not None:
                    self.scan.global_mask = global_mask
                publication = publication_from_live_frame(
                    frame,
                    generation=self.publication_store.generation,
                )
                with self.h5viewer.data_lock:
                    # 1D copy (no map_raw / 2D payload) — small object
                    self.h5viewer.data_1d[int(idx)] = frame.copy_for_display(
                        include_2d=False,
                    )
                    # 2D payload as a dict matching what file_thread.load_frames
                    # produces (see scan_threads.load_frames).  Keys are what
                    # displayframe.get_frames_map_raw / get_frames_int_2d
                    # look for.
                    skip_2d = getattr(self.scan, "skip_2d", False)
                    has_2d_errors = publication_has_2d_errors(publication)
                    if not skip_2d and not has_2d_errors:
                        self.h5viewer.data_2d[int(idx)] = {
                            "map_raw": getattr(frame, "map_raw", None),
                            "bg_raw": getattr(frame, "bg_raw", 0),
                            "mask": getattr(frame, "mask", None),
                            "int_2d": getattr(frame, "int_2d", None),
                            "gi_2d": getattr(frame, "gi_2d", {}),
                            "thumbnail": getattr(frame, "thumbnail", None),
                        }
                    elif not skip_2d and has_2d_errors:
                        self.h5viewer.data_2d.pop(int(idx), None)
                        logger.warning(
                            "Skipping frame %s 2D display cache: %s",
                            idx,
                            publication_error_details(publication, "2d"),
                        )
                    # ── Bounded cache eviction ────────────────────────
                    # O4: ``data_1d`` and ``data_2d`` are both bounded
                    # FixSizeOrderedDicts now (see __init__), so their
                    # own ``__setitem__`` already enforces the per-dict
                    # cap.  The previous manual loop here keyed off
                    # ``data_2d > _FRAME_CACHE_MAX=32`` was dead code:
                    # FixSizeOrderedDict capped data_2d at 20 long
                    # before that threshold, while data_1d (then
                    # unbounded) silently grew without bound.  No
                    # explicit eviction needed now; downstream cache
                    # misses fall through to the file_thread lazy load.
                    self.publication_store.upsert(publication)
            except Exception:
                # Cache miss is non-fatal — displayframe will lazy-load
                # from disk via file_thread.load_frame as fallback.
                logger.debug("In-memory frame hand-off failed for idx=%s",
                             idx, exc_info=True)

            # Mirror add_frame's scan_data accumulation: the live hand-off
            # bypasses scan.add_frame, so without this the GUI scan's
            # scan_data stays empty in non-batch and the metadata panel can
            # only fall back to the single selected frame.  Build the
            # whole-scan table here so it shows every frame (as it did
            # before the live fast-path).
            info = getattr(frame, "scan_info", None)
            if info:
                import pandas as pd
                from xdart.modules.ewald.scan import _numeric_scan_info
                # Keep only numeric-coercible fields — a single non-numeric
                # value (sample name, "24.4C" temperature, timestamp) would
                # otherwise make pd.Series(..., dtype="float64") raise and
                # silently skip the whole row, leaving scan_data empty and
                # the metadata panel blank in non-batch.  Mirrors
                # LiveScan.add_frame's _numeric_scan_info filter.
                numeric_info = _numeric_scan_info(info)
                if numeric_info:
                    try:
                        ser = pd.Series(numeric_info, dtype="float64")
                        with self.scan.scan_lock:
                            sd = self.scan.scan_data
                            if list(sd.columns):
                                sd.loc[idx] = ser
                                # In-order fast path: frames usually arrive in
                                # ascending order, so the row just appended is
                                # already last — skip the O(N log N) sort_index
                                # on every frame.  Only an out-of-order insert
                                # (rare: reload/replace) pays for the sort.
                                index = sd.index
                                if len(index) >= 2 and index[-1] < index[-2]:
                                    sd.sort_index(inplace=True)
                            else:
                                self.scan.scan_data = pd.DataFrame(
                                    numeric_info, index=[idx], dtype="float64")
                    except (ValueError, TypeError):
                        logger.debug("scan_data update skipped for idx=%s", idx,
                                     exc_info=True)

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
        # scan burst (frame inter-arrival < timer interval) we want the
        # display to refresh every ~200 ms, not only after the burst
        # settles.  Debounce semantics (.start() unconditionally
        # restarting the countdown) made the display freeze on whatever
        # the last completed flush rendered until the scan finished —
        # visible as "plots only update at end of scan."
        self._pending_update_idx = idx
        if not self._update_timer.isActive():
            self._update_timer.start()

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
        # Heavy list-widget refresh first — auto-last cursor needs the
        # list to contain the new index before it can select it.
        self.h5viewer.update_data(emit_update=False)
        if self.h5viewer.auto_last:
            self.latest_frame(emit_update=False)

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
                return

        self.h5viewer.data_changed()  # → sigUpdate → set_data → metawidget.update()

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

            if not is_viewer:
                if len(self.frame_ids) == 0:
                    self.integratorTree.ui.integrate1D.setEnabled(False)
                    self.integratorTree.ui.integrate2D.setEnabled(False)
                else:
                    self.integratorTree.ui.integrate1D.setEnabled(True)
                    self.integratorTree.ui.integrate2D.setEnabled(True)

            self.metawidget.update()
            # self.integratorTree.update()

    def close(self):
        """Tries a graceful close.
        """
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
            mode_text = self.wrangler.ui.processingModeCombo.currentText()
        except Exception:
            mode_text = ''
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer', 'NeXus Viewer')
        is_1d_only = mode_text in ('Int 1D', 'Int 1D (XYE)')
        run_active = getattr(self, '_run_active', False)
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
        self.displayframe.set_processing_active(True)
        self._apply_integration_control_state()   # run_active=True → disable
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
        # Re-enable the tree (restores the auto-range field gating) then overlay
        # the mode-correct state (run_active is now False).
        self.enable_integration(True)
        self._apply_integration_control_state()

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
        # self.thread_state_changed()
        if idx is not None:
            self.h5viewer.latest_idx = idx

        self.h5viewer.set_open_enabled(True)
        self.h5viewer.update_data()
        
        if self.h5viewer.auto_last:
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
        self._sync_h5viewer_save_dir(os.path.dirname(fname), refresh=False)
        self.h5viewer.set_file(fname)
        self.scan.gi = gi
        self.scan.incidence_motor = incidence_motor
        self.scan.single_img = single_img
        self.scan.series_average = series_average
        # Propagate the wrangler-loaded mask (detector + user Mask File,
        # combined into flat indices) into the main scan so the
        # displayframe can overlay it on the raw image.  Without this,
        # self.scan.global_mask stays None after a scan and no mask
        # overlay is drawn (regression introduced by the v2 refactor).
        wrangler_mask = getattr(getattr(self.wrangler, 'thread', None),
                                'mask', None)
        if wrangler_mask is not None:
            self.scan.global_mask = wrangler_mask

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
        # blanked the display the instant a new scan started and the
        # FixSizeOrderedDict eviction would have handled the memory
        # bound anyway as new frames came in.  Leaving the data dicts
        # alone lets the previous scan's last-rendered frame linger
        # visibly until the new scan's first frame replaces it on
        # the next ``_flush_pending_update`` tick.
        self.frames.clear()
        self.frame_ids.clear()
        self.publication_store.clear()

        # During a live (non-batch) run the async file-thread set_datafile
        # no longer reloads frames from disk (it would clobber the live
        # in-memory index — see fileHandlerThread.set_datafile).  So reset
        # the new scan's frame index synchronously here: drop the previous
        # scan's indices + cached frames so per-frame sigUpdate appends
        # build this scan up from empty.  The data_1d / data_2d snapshots
        # are intentionally left populated so the prior frame lingers on
        # screen until this scan's first frame replaces it.
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
        self.scan.gi = gi
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
        if not self.integratorTree.integrator_thread.isRunning():
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

        if is_batch and not is_xye_only:
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
                # Inform H5Viewer to load the file and set the flag to auto-select its last point
                self.h5viewer._auto_select_last_on_finish = True
                self.h5viewer.set_file(generated_file)

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
                    mode_text = self.wrangler.ui.processingModeCombo.currentText()
                except Exception:
                    mode_text = ''
                viewer_processing = (
                    mode_text in ('Image Viewer', 'XYE Viewer')
                    if mode_text else is_file_viewer
                )
                tree.setEnabled(not viewer_processing)
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
