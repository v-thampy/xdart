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
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

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
                                 self.ui.hdf5Frame, data_lock=self.data_lock)
        self.ui.hdf5Frame.setLayout(self.h5viewer.layout)
        self.h5viewer.update_scans()

        # DisplayFrame
        self.displayframe = displayFrameWidget(self.scan, self.frame,
                                               self.frame_ids, self.frames,
                                               self.data_1d, self.data_2d,
                                               parent=self.ui.middleFrame,
                                               data_lock=self.data_lock)
        self.ui.middleFrame.setLayout(self.displayframe.ui.layout)

        # IntegratorTree
        self.integratorTree = integratorTree(
            self.scan, self.frame, self.file_lock,
            self.frames, self.frame_ids, self.data_1d, self.data_2d,
            data_lock=self.data_lock)
        self.ui.integratorFrame.setLayout(self.integratorTree.ui.verticalLayout)
        if len(self.scan.frames.index) > 0:
            self.integratorTree.update()
        self.integratorTree.ui.raw_to_tif.hide()

        # Metadata
        self.metawidget = metadataWidget(self.scan, self.frame,
                                         self.frame_ids, self.frames,
                                         data_1d=self.data_1d)
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
                # Skip when in viewer mode — set_viewer_display_mode controls panels
                if 'Viewer' in mode_text:
                    return
                self.displayframe._apply_1d_only_visibility()
                # Drop any visible/cached content from the previous mode,
                # then reload the current selection for the new processing
                # mode. Calling update() alone can leave a stale image/cake
                # or curve visible when the new mode needs data that has not
                # been loaded yet.
                self.displayframe.clear_display_state()
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
        # Wire the wrangler's Advanced button to show the integratorTree's
        # existing 1D/2D advanced parameter dialogs in a combined popup.
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            self.wrangler.ui.advancedButton.clicked.connect(
                self._show_integration_advanced)
        self.wrangler.setup()
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
                with self.h5viewer.data_lock:
                    # 1D copy (no map_raw / 2D payload) — small object
                    self.h5viewer.data_1d[int(idx)] = frame.copy_for_display(
                        include_2d=False,
                    )
                    # 2D payload as a dict matching what file_thread.load_frames
                    # produces (see scan_threads.load_frames).  Keys are what
                    # displayframe.get_frames_map_raw / get_frames_int_2d
                    # look for.
                    if not getattr(self.scan, "skip_2d", False):
                        self.h5viewer.data_2d[int(idx)] = {
                            "map_raw": getattr(frame, "map_raw", None),
                            "bg_raw": getattr(frame, "bg_raw", 0),
                            "mask": getattr(frame, "mask", None),
                            "int_2d": getattr(frame, "int_2d", None),
                            "gi_2d": getattr(frame, "gi_2d", {}),
                            "thumbnail": getattr(frame, "thumbnail", None),
                        }
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
        self.h5viewer.data_changed()

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
        self.enable_integration(True)
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
        self.h5viewer.dirname = os.path.dirname(fname)
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
                with self.scan.scan_lock:
                    self.scan.frames.index.clear()
                    self.scan.frames._in_memory.clear()
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
        self.integratorTree.set_image_units()
        self.displayframe.set_axes()

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

        self.wrangler.thread.start()

    def wrangler_finished(self):
        """Called by the wrangler finished signal. If current scan
        matches the wrangler scan, allows for integration.
        """
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

        if self.scan.name == self.wrangler.scan_name:
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
                    # Auto-select the last (most recent) generated file so
                    # the final pattern is shown without a manual click.
                    self.h5viewer.select_last_scan_entry()
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
        from PySide6.QtWidgets import QAbstractItemView

        scans = self.h5viewer.ui.listScans
        prev_suspend = getattr(
            self.h5viewer, '_suspend_scan_selection_loads', False,
        )
        was_blocked = scans.blockSignals(True)
        self.h5viewer._suspend_scan_selection_loads = True
        try:
            # Keep integratorTree enabled so mask/threshold controls remain accessible
            self.h5viewer.viewer_mode = viewer_mode
            # Give displayframe a reference to the wrangler for mask/threshold
            self.displayframe._wrangler = self.wrangler if is_viewer else None
            # In viewer mode, disable New/Save (keep Open Folder and Export)
            self.h5viewer.actionNewFile.setEnabled(not is_viewer)
            self.h5viewer.actionSaveDataAs.setEnabled(not is_viewer)
            # XYE viewer: allow multi-select for overlay; others: single select
            if viewer_mode == 'xye':
                scans.setSelectionMode(QAbstractItemView.ExtendedSelection)
            else:
                scans.setSelectionMode(QAbstractItemView.SingleSelection)
            # Configure display panels for the viewer mode
            self.displayframe.set_viewer_display_mode(viewer_mode)
            if is_viewer:
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
