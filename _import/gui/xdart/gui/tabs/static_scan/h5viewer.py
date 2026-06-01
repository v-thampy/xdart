# -*- coding: utf-8 -*-
"""
@author: walroth
"""
# Standard library imports
import logging
import os
import time

logger = logging.getLogger(__name__)

# This module imports
import re
import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.io.export import read_xye
from ssrl_xrd_tools.io.image import read_image, count_frames
from xdart.utils.session import load_session, save_session
from .ui.h5viewerUI import Ui_Form
from xdart.modules.live import LiveFrame
from .scan_threads import fileHandlerThread
from ...widgets import defaultWidget
from xdart import utils
from xdart.utils import catch_h5py_file as catch
from xdart.utils.h5pool import get_pool

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui


QTreeWidget = QtWidgets.QTreeWidget
QTreeWidgetItem = QtWidgets.QTreeWidgetItem
QWidget = QtWidgets.QWidget
QFileDialog = QtWidgets.QFileDialog
QItemSelectionModel = QtCore.QItemSelectionModel


class _LoadFramesWorker(QtCore.QObject):
    """M1 background worker for ``load_frames_data``.

    Runs ``_load_frame_v2`` for each requested frame on a dedicated
    QThread so the GUI's event loop never blocks on HDF5 reads.

    Usage::

        worker = _LoadFramesWorker(data_file, file_lock, gi, frame_ids, load_2d)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunkLoaded.connect(viewer._absorb_chunk)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    Cancellation: the owner calls ``worker.cancel()`` (sets a
    threading.Event); the run loop checks between every frame and
    bails cleanly.  The currently-in-flight HDF5 read still
    completes — there's no safe way to preempt a libhdf5 call — but
    no further reads start.
    """

    # N1: signals carry the worker's generation number so the GUI
    # can drop stale chunks from a cancelled worker that's still
    # draining its run loop.
    #
    # ``load_2d`` rides along in the signal payload so the receiver
    # slot can be a plain bound method instead of a lambda — see
    # ``H5Viewer.load_frames_data`` for the wiring.  Pre-fix the
    # connection used a lambda to inject ``load_2d``, but Qt can't
    # determine a lambda's thread affinity and fell back to
    # direct-call delivery on the emitter (worker) thread, so the
    # GUI-thread-only ``QTimer.start()`` inside ``_absorb_chunk``
    # raised "Timers cannot be started from another thread".
    chunkLoaded = QtCore.Signal(int, int, object, bool)  # (gen, idx, frame, load_2d)
    finished = QtCore.Signal(int)  # (generation,)
    cancelled = QtCore.Signal(int)  # (generation,)

    def __init__(self, data_file, file_lock, gi, frame_ids, load_2d,
                 generation, parent=None):
        super().__init__(parent)
        self.data_file = data_file
        self.file_lock = file_lock
        self.gi = gi
        self.frame_ids = list(frame_ids)
        self.load_2d = load_2d
        # N1: monotonic generation number assigned by the owning
        # H5Viewer.  Workers whose generation no longer matches
        # ``H5Viewer._load_generation`` have been superseded and
        # their emitted chunks are dropped on the GUI side.
        self.generation = int(generation)
        # Use threading.Event so the run loop (running on the worker
        # thread) and cancel() (called from the GUI thread) can
        # synchronise without going through Qt's signal queue.
        import threading
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Request the worker stop after its current frame read."""
        self._cancel.set()

    def run(self) -> None:
        """Run on the worker thread.  Loads each frame and emits one
        ``chunkLoaded`` signal per success; emits ``finished`` (or
        ``cancelled``) when done."""
        try:
            from xdart.modules.ewald.frame_series import _load_frame_v2
            pool = get_pool()
            # Acquire a borrowed file handle from the pool.  Pre-M1
            # the GUI did the same dance on the main thread; here we
            # do it on the worker thread.  If the writer paused the
            # pool mid-load, pool.get() returns None and we bail
            # cleanly — the GUI will re-fire data_changed once the
            # writer releases.
            # Self-review fix #1: re-acquire the pool handle inside
            # each loop iteration.  Pre-fix the worker grabbed
            # ``file`` once and used it across all reads; if a
            # concurrent save called ``pool.pause()`` the handle got
            # closed under the worker's feet, the subsequent
            # ``_load_frame_v2`` raised a swallowed exception, and
            # the rest of the load silently failed.  Now every
            # iteration goes through ``pool.get()`` so a paused pool
            # returns None and the worker bails cleanly until the
            # next user selection re-fires the load.
            for idx in self.frame_ids:
                if self._cancel.is_set():
                    self.cancelled.emit(self.generation)
                    return
                file = pool.get(self.data_file)
                if file is None:
                    # Writer paused the pool — exit gracefully; the
                    # writer's resume() will trigger a sigUpdate
                    # that re-fires the GUI's data_changed slot.
                    logger.debug(
                        "load worker gen=%s: pool paused at idx=%s; "
                        "stopping",
                        self.generation, idx,
                    )
                    break
                try:
                    with self.file_lock:
                        frame = _load_frame_v2(file, idx, static=True,
                                             gi=self.gi)
                except (KeyError, IndexError, OSError, ValueError) as e:
                    logger.debug("load worker: frame %s skipped: %s",
                                 idx, e)
                    continue
                # Emit on the worker thread; Qt queues the slot
                # invocation back to the GUI thread automatically
                # because the signal target is a bound method on a
                # QObject living on the GUI thread.
                self.chunkLoaded.emit(
                    self.generation, int(idx), frame, bool(self.load_2d),
                )
        except Exception:
            logger.exception("LoadFramesWorker crashed unexpectedly")
        finally:
            self.finished.emit(self.generation)


class _AccumulatingClickFilter(QtCore.QObject):
    """Centralized click handler for the H5Viewer ``listData`` widget.

    listData uses ``ExtendedSelection`` at all times. This filter
    intercepts left-button presses on the viewport and dispatches them
    according to the active plot method and held modifiers. By owning
    the click logic explicitly we sidestep two cross-platform Qt
    quirks at once:

    1. On macOS, Qt swaps Ctrl and Meta by default — so the literal
       Ctrl key reports as ``Qt.MetaModifier`` and is *not* recognized
       by ``ExtendedSelection``'s built-in toggle logic. We treat
       Ctrl, Cmd and Meta interchangeably here so toggling works on
       every platform regardless of which physical key the user
       presses.
    2. PySide6's flag-enum comparisons (``mods != Qt.NoModifier``)
       can be unreliable across versions. We coerce modifiers to
       ``int`` and check bits explicitly.

    Decision matrix (left-button press only — everything else passes
    through):

        ┌─────────────────────┬──────────────────────────┐
        │ Modifier            │ Action                   │
        ├─────────────────────┼──────────────────────────┤
        │ Shift               │ pass through (Qt range)  │
        │ Ctrl / Cmd / Meta   │ toggle clicked item      │
        │ none — accumulating │ toggle clicked item      │
        │ none — Single mode  │ pass through (Qt replace)│
        └─────────────────────┴──────────────────────────┘
    """

    _ACCUMULATING = ('Overlay', 'Waterfall', 'Sum', 'Average')

    def __init__(self, listwidget, get_mode):
        super().__init__(listwidget)
        self._lw = listwidget
        self._get_mode = get_mode

    def _toggle(self, item):
        """Toggle a single item via the selection model and notify
        downstream listeners (data_changed via itemSelectionChanged,
        disable_auto_last via itemClicked)."""
        idx = self._lw.indexFromItem(item)
        sm = self._lw.selectionModel()
        sm.select(idx, QItemSelectionModel.Toggle)
        sm.setCurrentIndex(idx, QItemSelectionModel.NoUpdate)
        try:
            self._lw.itemClicked.emit(item)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if event.type() != QtCore.QEvent.MouseButtonPress:
            return False
        try:
            btn = event.button()
            mods_obj = event.modifiers()
        except AttributeError:
            return False
        if btn != QtCore.Qt.LeftButton:
            return False

        # Coerce modifier flags to int so bitwise checks are immune
        # to PySide6 enum/flag comparison quirks.
        try:
            mods = int(mods_obj)
        except (TypeError, ValueError):
            return False
        ctrl_bit = int(QtCore.Qt.ControlModifier)
        shift_bit = int(QtCore.Qt.ShiftModifier)
        meta_bit = int(QtCore.Qt.MetaModifier)
        has_shift = bool(mods & shift_bit)
        has_toggle_mod = bool(mods & (ctrl_bit | meta_bit))

        # Shift-range select is delegated to Qt — it walks from the
        # current/anchor item through the clicked item and replaces
        # the selection. We don't try to reimplement that.
        if has_shift:
            return False

        try:
            mode = self._get_mode()
        except Exception:
            return False
        accumulating = mode in self._ACCUMULATING

        # Resolve the clicked item.
        try:
            pos = event.position().toPoint()
        except AttributeError:
            pos = event.pos()
        item = self._lw.itemAt(pos)
        if item is None:
            # Click on empty space — let Qt clear selection in Single
            # mode, swallow it in accumulating modes so we don't lose
            # the existing multi-selection.
            return accumulating

        if has_toggle_mod or accumulating:
            self._toggle(item)
            return True

        # Plain click in Single mode → pass through to Qt's default
        # ExtendedSelection handler (which replaces the selection
        # with just the clicked item).
        return False


class H5Viewer(QWidget):
    """Widget for displaying the contents of an LiveScan object and
    a basic file explorer. Also holds menus for more general tasks like
    setting defaults.
    
    attributes:
        (QAction attributes not shown, associated menus are)
        exportMenu: Sub-menu for exporting images and 1d data
        file_lock: Condition, lock governing file access
        fileMenu: Menu for saving files and exporting data
        fname: Current data file name
        layout: ui layout TODO: this can stay with ui
        paramMenu: Menu for saving and loading defaults
        toolbar: QToolBar, holds the menus
        ui: Ui_Form from qtdesigner

    methods:
        set_data: Sets the data in the dataList
        set_open_enabled: Sets the ability to open scans to enabled or
            disables
        update: Updates files in scansList
        TODO: Rename the methods and attributes based on what they
            actually do
    """
    sigNewFile = Qt.QtCore.Signal(str)
    sigUpdate = Qt.QtCore.Signal()
    sigThreadFinished = Qt.QtCore.Signal()

    def __init__(self, file_lock, local_path, dirname,
                 scan, frame, frame_ids, frames,
                 data_1d, data_2d,
                 parent=None, data_lock=None):
        super().__init__(parent)
        import threading as _threading
        self.data_lock = data_lock if data_lock is not None else _threading.RLock()
        self._init_data_objects(file_lock, local_path, dirname,
                                scan, frame, frame_ids, frames,
                                data_1d, data_2d)
        self._init_ui()
        self._init_toolbar()
        self._connect_signals()
        self._init_file_thread()

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self, file_lock, local_path, dirname,
                           scan, frame, frame_ids, frames,
                           data_1d, data_2d):
        """Initialize data references and state flags."""
        self.local_path = local_path
        self.file_lock = file_lock
        self.dirname = dirname
        self.scan = scan
        self.frame = frame
        self.frame_ids = frame_ids
        self.frames = frames
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.new_scan = True
        self.update_2d = True
        self.auto_last = True
        self.latest_idx = None
        self.new_scan_loaded = False
        self.viewer_mode = None
        # True only while a live (non-batch) wrangler run is in progress.
        # Suppresses ``data_reset`` (wired to the async ``sigNewFile``)
        # so the per-frame in-memory caches the live display depends on
        # aren't wiped mid-run.  Toggled by static_scan_widget.
        self.live_run_active = False
        self._displayed_list_count = 0
        self._displayed_last_label = None

    def _init_ui(self):
        """Set up the main UI form and default widget."""
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.layout = self.ui.gridLayout
        self.defaultWidget = defaultWidget()
        self.defaultWidget.sigSetUserDefaults.connect(self.set_user_defaults)

    def _init_toolbar(self):
        """Create toolbar with File and Config menus."""
        self.toolbar = QtWidgets.QToolBar('Tools')

        # Actions
        self.actionOpenFolder = QtGui.QAction()
        self.actionOpenFolder.setText('Open Folder')
        self.actionSetDefaults = QtGui.QAction()
        self.actionSetDefaults.setText('Advanced...')
        self.actionSaveDataAs = QtGui.QAction()
        self.actionSaveDataAs.setText('Save As')
        self.actionNewFile = QtGui.QAction()
        self.actionNewFile.setText('New')

        # Export sub-menu
        self.exportMenu = QtWidgets.QMenu()
        self.exportMenu.setTitle('Export')
        self.actionSaveImage = QtGui.QAction()
        self.actionSaveImage.setText('Current Image')
        self.exportMenu.addAction(self.actionSaveImage)
        self.actionSaveArray = QtGui.QAction()
        self.actionSaveArray.setText('Current 1D Array')
        self.exportMenu.addAction(self.actionSaveArray)

        # Config sub-menu
        self.paramMenu = QtWidgets.QMenu()
        self.paramMenu.setTitle('Config')
        self.actionSaveParams = QtGui.QAction()
        self.actionSaveParams.setText('Save')
        self.actionSaveParams.triggered.connect(self.defaultWidget.save_defaults)
        self.paramMenu.addAction(self.actionSaveParams)
        self.actionLoadParams = QtGui.QAction()
        self.actionLoadParams.setText('Load')
        self.actionLoadParams.triggered.connect(self.defaultWidget.load_defaults)
        self.paramMenu.addAction(self.actionLoadParams)
        self.paramMenu.addAction(self.actionSetDefaults)

        # File menu
        self.fileMenu = QtWidgets.QMenu()
        self.fileMenu.addAction(self.actionOpenFolder)
        self.fileMenu.addAction(self.actionNewFile)
        self.fileMenu.addAction(self.actionSaveDataAs)
        self.fileMenu.addMenu(self.exportMenu)

        # Toolbar buttons
        self.fileButton = QtWidgets.QToolButton()
        self.fileButton.setText('File')
        self.fileButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.fileButton.setMenu(self.fileMenu)
        self.paramButton = QtWidgets.QToolButton()
        self.paramButton.setText('Config')
        self.paramButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.paramButton.setMenu(self.paramMenu)

        self.toolbar.addWidget(self.fileButton)
        self.toolbar.addWidget(self.paramButton)
        self.layout.addWidget(self.toolbar, 0, 0, 1, 2)

    def _connect_signals(self):
        """Wire signal/slot connections for list widgets and menu actions."""
        self.actionSetDefaults.triggered.connect(self.defaultWidget.show)
        self.ui.listScans.itemDoubleClicked.connect(self.scans_clicked)
        self.ui.listScans.itemClicked.connect(self._scans_single_clicked)
        self.ui.listScans.currentItemChanged.connect(self._scans_current_changed)
        self.ui.listScans.itemSelectionChanged.connect(self._scans_selection_changed)
        self.ui.listScans.installEventFilter(self)
        self.ui.listData.itemSelectionChanged.connect(self.data_changed)
        # listData stays in ExtendedSelection at all times so all
        # standard selection behaviors (arrow keys, shift-range,
        # ctrl-toggle) work consistently across modes. The click
        # filter installed below handles plain-click toggling for
        # accumulating plot methods (Overlay/Waterfall/Sum/Average)
        # and works around macOS quirks where the literal Ctrl key
        # maps to Qt.MetaModifier instead of Qt.ControlModifier.
        self.ui.listData.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self._plot_method = 'Single'
        self._list_click_filter = _AccumulatingClickFilter(
            self.ui.listData, lambda: self._plot_method)
        self.ui.listData.viewport().installEventFilter(self._list_click_filter)
        self.ui.show_all.clicked.connect(self.show_all)
        self.actionOpenFolder.triggered.connect(self.open_folder)
        self.actionSaveDataAs.triggered.connect(self.save_data_as)
        self.actionNewFile.triggered.connect(self.new_file)

    def set_data_selection_mode(self, plot_method):
        """Update internal plot-method state and reconcile listData
        selection when switching between modes.

        listData stays in ExtendedSelection at all times. When the
        user enters ``Single`` mode from a multi-selection state,
        this method collapses the selection down to the most recently
        focused item so the next click behaves naturally. The click
        filter consults ``self._plot_method`` to decide how to handle
        plain clicks.
        """
        prev_method = self._plot_method
        self._plot_method = plot_method
        if plot_method == 'Single' and prev_method != 'Single':
            lw = self.ui.listData
            current = lw.currentItem()
            selected = lw.selectedItems()
            if len(selected) > 1:
                lw.blockSignals(True)
                try:
                    lw.clearSelection()
                    keep = current if current in selected else selected[-1]
                    if keep is not None:
                        keep.setSelected(True)
                        lw.setCurrentItem(keep)
                finally:
                    lw.blockSignals(False)
                self.data_changed()

    def _init_file_thread(self):
        """Create and start the background file handler thread."""
        self.file_thread = fileHandlerThread(self.scan, self.frame,
                                             self.file_lock,
                                             frame_ids=self.frame_ids,
                                             frames=self.frames,
                                             data_1d=self.data_1d,
                                             data_2d=self.data_2d,
                                             data_lock=self.data_lock)
        self.file_thread.sigTaskDone.connect(self.thread_finished)
        self.file_thread.sigNewFile.connect(self.sigNewFile.emit)
        self.file_thread.sigUpdate.connect(self.sigUpdate.emit)
        self.file_thread.start(Qt.QtCore.QThread.LowPriority)
        self._h5pool = get_pool()
        # M1: handle for the per-selection LoadFramesWorker.  None
        # when no load is in flight.  Owns a QThread that gets
        # created on demand and reaped between selections.
        self._load_worker = None
        self._load_thread = None
        # N1: monotonic generation counter for load workers.  Every
        # new worker gets a fresh number; _absorb_chunk drops any
        # incoming chunk whose generation no longer matches.  This
        # prevents stale chunks from a cancelled worker (still
        # draining its run loop after cancel()) from polluting the
        # new selection's data dicts.
        self._load_generation = 0
        # O6: coalesce ``sigUpdate`` emits while a chunk burst is
        # streaming in from ``_LoadFramesWorker``.  Without this, a
        # 100-frame selection fires 100 full-display repaints in
        # rapid succession.  With it, the burst is debounced to a
        # single emit ~100 ms after the last chunk lands — and the
        # worker-finished slot forces a final emit so the last
        # paint always reflects the full selection.
        self._update_coalesce_timer = Qt.QtCore.QTimer(self)
        self._update_coalesce_timer.setSingleShot(True)
        self._update_coalesce_timer.setInterval(100)  # ms
        self._update_coalesce_timer.timeout.connect(self.sigUpdate.emit)
        
    def load_starting_defaults(self):
        default_path = os.path.join(self.local_path, "last_defaults.json")
        if os.path.exists(default_path):
            self.defaultWidget.load_defaults(fname=default_path)
        else:
            self.defaultWidget.save_defaults(fname=default_path)

    def set_user_defaults(self):
        default_path = os.path.join(self.local_path, "last_defaults.json")
        self.defaultWidget.save_defaults(fname=default_path)

    def update(self):
        """Calls both update_scans and update_data.
        """
        # self.update_scans()
        self.update_data()

        # Restore session
        session = load_session()
        saved_dir = session.get('data_dir', '')
        if saved_dir and os.path.isdir(saved_dir):
            self.dirname = saved_dir
            self.update_scans()

    # File extensions for viewer modes
    _IMAGE_EXTS = {'.tif', '.tiff', '.raw', '.edf', '.h5', '.hdf5', '.nxs'}
    _XYE_EXTS = {'.xye'}

    @staticmethod
    def _natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', text)]

    def update_scans(self):
        """Populate listScans with files in the current directory.

        In normal mode, shows HDF5 files and directories.
        In image viewer mode, shows image files and directories.
        In xye viewer mode, shows xye files and directories.
        """
        if not os.path.exists(self.dirname):
            return

        self.ui.listScans.clear()
        self.ui.listScans.addItem('..')

        names = sorted(os.listdir(self.dirname), key=self._natural_sort_key)
        for name in names:
            abspath = os.path.join(self.dirname, name)
            if os.path.isdir(abspath):
                self.ui.listScans.addItem(name + '/')
            else:
                ext = os.path.splitext(name)[1].lower()
                if self.viewer_mode == 'image':
                    if ext in self._IMAGE_EXTS:
                        self.ui.listScans.addItem(name)
                elif self.viewer_mode == 'xye':
                    if ext in self._XYE_EXTS:
                        self.ui.listScans.addItem(name)
                else:
                    # Normal mode: only HDF5/NeXus scan files
                    if name.split('.')[-1] in ('h5', 'hdf5', 'nxs'):
                        self.ui.listScans.addItem(name)

    # ── Selection helper ──────────────────────────────────────────────
    def set_current_frame(self, item_or_row):
        """Advance the listData cursor *and narrow the selection* to a
        single entry.

        ``QListWidget.setCurrentItem(item)`` and ``setCurrentRow(row)``
        default to ``QItemSelectionModel.NoUpdate`` when the list is in
        ``ExtendedSelection`` mode — the current item moves but the
        selection is never narrowed.  During a live scan this is
        visible as the entire Frames list lighting up as new frames
        arrive.  Pass ``ClearAndSelect`` so the highlight follows the
        cursor.

        Callers that drive this from inside their own
        ``blockSignals(True/False)`` block (and then invoke
        ``data_changed()`` explicitly) get the same behaviour without
        firing ``itemSelectionChanged``.
        """
        lw = self.ui.listData
        if isinstance(item_or_row, int):
            lw.setCurrentRow(item_or_row, QItemSelectionModel.ClearAndSelect)
        else:
            lw.setCurrentItem(item_or_row, QItemSelectionModel.ClearAndSelect)

    def _remember_displayed_frames(self):
        """Cache enough list state to recognize append-only live updates."""
        lw = self.ui.listData
        count = lw.count()
        self._displayed_list_count = count
        self._displayed_last_label = (
            lw.item(count - 1).text() if count > 0 else None
        )

    def update_data(self, emit_update=True):
        """Updates list with all frame ids.

        Fast paths in order of likelihood for a live scan:

        1. ``_idxs == items`` — list already shows everything the scan
           knows about (timer tick fired but no new frames since last
           flush).  Just refresh the cursor if auto_last is on, done.
        2. **P4 fast path**: ``_idxs`` is exactly ``items + tail`` — the
           scan appended new frames and the existing list is intact.
           ``addItems(tail)`` instead of clearing + re-inserting all
           items, so per-flush cost is O(new) not O(N).  Critical for
           long scans: the previous code rebuilt the entire list every
           time a frame arrived, which compounded to O(N²) over a run.
        3. Slow path — full rebuild.  Used on out-of-order inserts,
           reorderings, file reloads, anything that doesn't match the
           append-only pattern.
        """
        if self.scan.name == "null_main":
            return

        # with self.scan.scan_lock:
        frame_index = self.scan.frames.index

        if len(frame_index) == 0:
            self.ui.listData.clear()
            self._remember_displayed_frames()
            # self.ui.listData.addItem('No Data')
            return

        lw = self.ui.listData

        def _emit_changed():
            if emit_update:
                self.data_changed()

        def _select_latest() -> bool:
            if not (self.auto_last and isinstance(self.latest_idx, int)):
                return False
            target = str(self.latest_idx)
            last_row = lw.count() - 1
            if last_row >= 0 and lw.item(last_row).text() == target:
                self.set_current_frame(last_row)
                return True
            matched = lw.findItems(target, QtCore.Qt.MatchExactly)
            for item in matched:
                self.set_current_frame(item)
            return bool(matched)

        # Common live-scan path: frame ids were appended at the tail since
        # the last GUI flush.  Verify only the cached boundary instead of
        # rebuilding and comparing the full list every 200 ms.
        current_count = lw.count()
        if (current_count >= 1
                and len(frame_index) > current_count
                and not self.new_scan_loaded):
            current_last = lw.item(current_count - 1).text()
            cached_count = getattr(self, "_displayed_list_count", 0)
            cached_last = getattr(self, "_displayed_last_label", None)
            expected_last = str(frame_index[current_count - 1])
            if (current_count == cached_count
                    and current_last == cached_last
                    and current_last == expected_last):
                new_tail = [
                    str(frame_index[pos])
                    for pos in range(current_count, len(frame_index))
                ]
                lw.blockSignals(True)
                lw.addItems(new_tail)
                if self.auto_last:
                    _select_latest()
                lw.blockSignals(False)
                self._remember_displayed_frames()
                if self.auto_last:
                    _emit_changed()
                return

        _idxs = [str(i) for i in frame_index]
        items = [lw.item(x).text() for x in range(lw.count())]

        if _idxs == items:
            if self.new_scan_loaded:
                self.new_scan_loaded = False
                self.ui.listData.setCurrentRow(-1)
                self.frame_ids = []
                self._remember_displayed_frames()
                return
            if self.auto_last and isinstance(self.latest_idx, int) and str(self.latest_idx) in _idxs:
                self.ui.listData.blockSignals(True)
                _select_latest()
                self.ui.listData.blockSignals(False)
                _emit_changed()
            return

        # ── P4 incremental-append fast path ─────────────────────────
        # The common case during a live scan: the scan just gained
        # one or more frames at the tail; everything else is the same.
        # Avoid the full clear + insertItems(0, _idxs) rebuild and
        # just addItems(new_tail), which is O(len(tail)) instead of
        # O(len(_idxs)).  Guarded by:
        #   - existing items are a strict prefix of _idxs (no inserts
        #     in the middle, no relabels);
        #   - we're not in the post-load "clear selection" path
        #     (new_scan_loaded already handled by the equal-list
        #     branch above, but we re-check here defensively);
        #   - the listWidget isn't empty (clear+insert is cheaper for
        #     small N anyway, and avoids fiddling with first-load
        #     selection semantics).
        if (len(items) >= 1
                and len(_idxs) > len(items)
                and _idxs[: len(items)] == items
                and not self.new_scan_loaded):
            new_tail = _idxs[len(items):]
            self.ui.listData.blockSignals(True)
            self.ui.listData.addItems(new_tail)
            if self.auto_last:
                _select_latest()
            self.ui.listData.blockSignals(False)
            self._remember_displayed_frames()
            if self.auto_last:
                _emit_changed()
            return

        previous_loc = self.ui.listData.currentRow()
        previous_sel = [item.text() for item in self.ui.listData.selectedItems()]

        # Block signals while rebuilding the list to prevent spurious
        # itemSelectionChanged → data_changed → sigUpdate cascades.
        self.ui.listData.blockSignals(True)

        self.ui.listData.clear()
        self.ui.listData.insertItems(0, _idxs)
        self._remember_displayed_frames()

        if self.new_scan_loaded:
            self.new_scan_loaded = False
            self.ui.listData.setCurrentRow(-1)
            self.frame_ids.clear()
            self.ui.listData.blockSignals(False)
            return

        if self.auto_last and isinstance(self.latest_idx, int) and (str(self.latest_idx) in _idxs):
            _select_latest()
            self.ui.listData.blockSignals(False)
            _emit_changed()
            return

        if previous_loc > self.ui.listData.count() - 1:
            previous_loc = self.ui.listData.count() - 1

        if len(previous_sel) < 2:
            self.ui.listData.setCurrentRow(previous_loc)
        else:
            for text in previous_sel:
                matched = self.ui.listData.findItems(text, QtCore.Qt.MatchExactly)
                for item in matched:
                    item.setSelected(True)

        self.ui.listData.blockSignals(False)
        _emit_changed()

    def show_all(self):

        if len(self.scan.frames.index) > 0:
            self.frame_ids.clear()
            self.frame_ids += self.scan.frames.index

        self.new_scan = False
        self.data_changed(show_all=True)

    def thread_finished(self, task):
        if task != "load_frame":
            self.update()
            if getattr(self, '_auto_select_last_on_finish', False):
                self._auto_select_last_on_finish = False
                if self.ui.listData.count() > 0:
                    self.set_current_frame(self.ui.listData.count() - 1)
        self.sigThreadFinished.emit()
    
    def _scans_single_clicked(self, q):
        """Handle single click in listScans — only acts in viewer mode.

        XYE mode uses _scans_selection_changed instead to avoid double-firing.
        """
        if self.viewer_mode is not None and self.viewer_mode != 'xye':
            self.scans_clicked(q)

    def _scans_current_changed(self, current, previous):
        """Handle arrow-key navigation in listScans (image viewer only).

        Loads files on selection change so the user can browse with
        arrow keys.  Directories are NOT auto-entered — use click or
        Enter for navigation.

        XYE mode uses _scans_selection_changed instead (fires after
        the selection is fully updated, avoiding off-by-one with
        Shift+arrow).
        """
        if current is None or self.viewer_mode is None:
            return
        # XYE mode: handled by _scans_selection_changed
        if self.viewer_mode == 'xye':
            return
        item_text = current.text()
        # Skip directories and ".." — don't auto-navigate on arrow keys
        if item_text == '..' or item_text.endswith('/'):
            return
        self.scans_clicked(current)

    def _scans_selection_changed(self):
        """Handle selection changes in listScans (XYE viewer mode only).

        Uses itemSelectionChanged which fires after the selection is
        fully updated, so Shift+arrow works correctly for multi-select.
        """
        if self.viewer_mode != 'xye':
            return
        selected = self.ui.listScans.selectedItems()
        if not selected:
            return
        # Skip if only directories/".." are selected
        has_files = any(
            not item.text().endswith('/') and item.text() != '..'
            for item in selected
        )
        if has_files:
            self._load_xye_files()

    def eventFilter(self, obj, event):
        """Handle Enter/Return key on listScans to navigate into folders."""
        if obj is self.ui.listScans and event.type() == event.Type.KeyPress:
            from PySide6.QtCore import Qt
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                current = self.ui.listScans.currentItem()
                if current is not None:
                    item_text = current.text()
                    if item_text == '..' or item_text.endswith('/'):
                        self.scans_clicked(current)
                        return True
        return super().eventFilter(obj, event)

    def scans_clicked(self, q):
        """Handles items being clicked/double-clicked in listScans.

        Normal mode: navigates folders or loads scan data from HDF5.
        Image Viewer: loads a single image or populates listData for multi-frame files.
        XYE Viewer: loads a single xye file as a 1D line.
        """
        try:
            item_text = q.data(0)

            # Navigation: ".." or folder
            if item_text == '..':
                if self.dirname[-1] in ['/', '\\']:
                    up = os.path.dirname(self.dirname[:-1])
                else:
                    up = os.path.dirname(self.dirname)
                if os.path.isdir(up) and os.path.splitdrive(up)[1] != '':
                    self.dirname = up
                    self.update_scans()
                return
            if '/' in item_text:
                dirname = os.path.join(self.dirname, item_text)
                if os.path.isdir(dirname):
                    self.dirname = dirname
                    self.update_scans()
                return

            if item_text == 'No scans':
                return

            fpath = os.path.join(self.dirname, item_text)

            # ── Viewer modes ──────────────────────────────────────────
            if self.viewer_mode == 'xye':
                self._load_xye_files()
                return
            if self.viewer_mode == 'image':
                self._load_image_file(fpath)
                return

            # ── Normal mode: open HDF5 scan ───────────────────────────
            self.set_file(fpath)
            self.new_scan_loaded = True
        except AttributeError:
            pass

    # ── Viewer mode loaders ───────────────────────────────────────────────

    def _load_xye_files(self):
        """Load all selected xye files from listScans for overlay.

        Each file gets a sequential index (1, 2, 3, …) in data_1d.
        listData is populated with filenames and all rows are selected
        so the display frame renders every curve.
        """
        selected = self.ui.listScans.selectedItems()
        if not selected:
            return

        with self.data_lock:
            self.data_1d.clear()
            self.data_2d.clear()
        self.frame_ids.clear()

        idx = 1
        for item in selected:
            item_text = item.text()
            # Skip directories
            if item_text == '..' or item_text.endswith('/'):
                continue
            fpath = os.path.join(self.dirname, item_text)
            try:
                xdata, ydata, sigma = read_xye(fpath)
            except Exception:
                logger.debug("Could not load xye file %s", fpath, exc_info=True)
                continue

            # Guess unit from filename prefix
            fname_lower = os.path.basename(fpath).lower()
            unit = 'q_A^-1' if fname_lower.startswith('iq') else '2th_deg'

            int_1d = IntegrationResult1D(
                radial=xdata, intensity=ydata, sigma=sigma, unit=unit,
            )
            frame = LiveFrame(idx=idx, static=True, gi=False)
            frame.int_1d = int_1d
            frame.scan_info = {'source_file': os.path.basename(fpath)}

            with self.data_lock:
                self.data_1d[idx] = frame
            self.frame_ids.append(str(idx))
            idx += 1

        if len(self.data_1d) == 0:
            return

        # Populate listData with loaded filenames (all selected).
        # Display filename but store numeric index in UserRole so
        # data_changed can map back to data_1d keys.
        self.ui.listData.blockSignals(True)
        self.ui.listData.clear()
        for key in self.data_1d:
            frame = self.data_1d[key]
            fname = frame.scan_info.get('source_file', f'file_{key}')
            display_name = os.path.basename(fname)
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, key)
            self.ui.listData.addItem(item)
        self.ui.listData.selectAll()
        self.ui.listData.blockSignals(False)
        # Keep the live-scan boundary cache in sync with the freshly
        # populated list so a subsequent update_data() doesn't take the
        # append-only fast path against stale cached state.
        self._remember_displayed_frames()

        self.sigUpdate.emit()

    @staticmethod
    def _is_xdart_processed(fpath):
        """True if ``fpath`` is a processed xdart v2 scan file (integrated
        data + per-frame source pointers), not a raw detector file."""
        try:
            import h5py
            with h5py.File(fpath, 'r') as f:
                return ('entry/integrated_1d' in f
                        or 'entry/integrated_2d' in f
                        or 'entry/frames' in f)
        except Exception:
            return False

    def _load_image_file(self, fpath):
        """Load an image file for viewing.

        For multi-frame files (Eiger HDF5 masters, tiff stacks),
        populate listData with frame indices.  For single-frame
        files, display the image directly.
        """
        with self.data_lock:
            self.data_1d.clear()
            self.data_2d.clear()
        self.frame_ids.clear()
        self.ui.listData.clear()

        ext = os.path.splitext(fpath)[1].lower()
        nframes = 1

        # ── Processed xdart v2 scan file ─────────────────────────────
        # No raw images live inside — load each frame's raw via its stored
        # source pointer (get_raw_frame), listed by the file's actual frame
        # labels (which may be 0-based / gapped), not 1..N.
        self._viewer_is_xdart = (
            ext in ('.h5', '.hdf5', '.nxs') and self._is_xdart_processed(fpath)
        )
        if self._viewer_is_xdart:
            from ssrl_xrd_tools.io import get_frames as _get_frames
            try:
                labels = [int(x) for x in _get_frames(fpath)]
            except Exception:
                logger.debug("Failed to read frame labels from %s", fpath, exc_info=True)
                labels = []
            if labels:
                self._viewer_image_path = fpath
                self._viewer_image_nframes = len(labels)
                for lbl in labels:
                    self.ui.listData.addItem(str(lbl))
                self._load_single_frame(fpath, frame_idx=labels[0], frame_id=labels[0])
                self.frame_ids.append(str(labels[0]))
                self.ui.listData.setCurrentRow(0)
                self._remember_displayed_frames()
                self.sigUpdate.emit()
                return
            # No labels found — fall through to the generic handling.

        # Check for multi-frame files
        if ext in ('.h5', '.hdf5', '.nxs'):
            nframes = count_frames(fpath)
            if nframes == 0:
                # count_frames failed — try loading as single frame
                nframes = 1
        elif ext in ('.tif', '.tiff'):
            try:
                import fabio
                img = fabio.open(fpath)
                nframes = img.nframes
                img.close()
            except Exception:
                logger.debug("Failed to detect frame count from TIFF file %s", fpath, exc_info=True)
                nframes = 1

        # HDF5/NeXus files always show frame numbers (even with 1 frame)
        is_hdf5 = ext in ('.h5', '.hdf5', '.nxs')

        if nframes > 1 or (is_hdf5 and nframes >= 1):
            # Multi-frame or HDF5: populate listData with frame numbers
            for i in range(nframes):
                self.ui.listData.addItem(str(i + 1))
            # Store the file path so data_changed can load individual frames
            self._viewer_image_path = fpath
            self._viewer_image_nframes = nframes
            # Load and display first frame
            self._load_single_frame(fpath, frame_idx=0, frame_id=1)
            self.frame_ids.append('1')
            self.ui.listData.setCurrentRow(0)
        else:
            # Single frame (tif, raw, edf): load directly, leave listData blank
            self._viewer_image_path = None
            self._load_single_frame(fpath, frame_idx=0, frame_id=1)
            self.frame_ids.append('1')

        # Sync the live-scan boundary cache with whatever just landed in
        # listData (frame-numbers list, or empty for single-frame files).
        self._remember_displayed_frames()
        self.sigUpdate.emit()

    # Common detector shapes to try for raw binary files (name, shape)
    _RAW_DETECTOR_FALLBACKS = [
        ('Pilatus 100k', (195, 487)),
        ('Pilatus 300k', (619, 487)),
        ('Pilatus 300kw', (195, 1475)),
        ('Pilatus 1M', (1043, 981)),
        ('Rayonix MX225', (3072, 3072)),
        ('Rayonix SX165', (2048, 2048)),
    ]

    def _load_single_frame(self, fpath, frame_idx=0, frame_id=1):
        """Load a single frame from an image file into data_2d.

        ``frame_idx`` is the 0-based offset *within the file* (passed to
        ``read_image``); ``frame_id`` is the 1-based id shown in listData
        and stored in ``frame_ids``.  The data dicts must be keyed by
        ``frame_id`` — ``data_changed``/``get_idxs`` look them up by the
        1-based id from the list, so keying by the 0-based ``frame_idx``
        (the old behaviour) left the viewer unable to find the image and
        showed a blank panel.
        """
        # Processed xdart v2 file: no raw images inside.  Resolve the raw
        # detector frame via the stored per-frame source pointer (falling
        # back to the stored thumbnail).  ``frame_id`` is the frame label.
        if getattr(self, '_viewer_is_xdart', False):
            try:
                from ssrl_xrd_tools.io import get_raw_frame
                img_data = np.asarray(
                    get_raw_frame(fpath, frame=frame_id), dtype=float,
                )
            except Exception:
                logger.warning('Could not load raw frame %s from processed file %s',
                               frame_id, os.path.basename(fpath))
                logger.debug('get_raw_frame failed', exc_info=True)
                return
            with self.data_lock:
                self.data_2d[int(frame_id)] = {
                    'map_raw': img_data,
                    'bg_raw': np.zeros_like(img_data),
                    'mask': None,
                    'int_2d': None,
                    'gi_2d': {},
                    'thumbnail': None,
                }
                frame = LiveFrame(idx=frame_id, static=True, gi=False)
                frame.scan_info = {'source_file': os.path.basename(fpath)}
                self.data_1d[int(frame_id)] = frame
            return

        try:
            img_data = np.asarray(
                read_image(fpath, frame=frame_idx), dtype=float,
            )
        except Exception:
            logger.debug("Failed to load frame %d from %s", frame_idx, fpath, exc_info=True)
            # For raw files, try common detector shapes
            ext = os.path.splitext(fpath)[1].lower()
            if ext == '.raw':
                img_data = self._try_raw_detectors(fpath)
                if img_data is None:
                    logger.warning('Cannot load %s — raw file does not match any '
                                   'known detector shape.', os.path.basename(fpath))
                    return
            else:
                logger.warning('Could not load image %s frame %d', fpath, frame_idx)
                return

        with self.data_lock:
            self.data_2d[int(frame_id)] = {
                'map_raw': img_data,
                'bg_raw': np.zeros_like(img_data),
                'mask': None,
                'int_2d': None,
                'gi_2d': {},
                'thumbnail': None,
            }
            # Minimal data_1d entry so display doesn't crash
            frame = LiveFrame(idx=frame_id, static=True, gi=False)
            frame.scan_info = {'source_file': os.path.basename(fpath)}
            self.data_1d[int(frame_id)] = frame
    
    def _try_raw_detectors(self, fpath):
        """Try reading a raw binary file with common detector shapes."""
        for name, shape in self._RAW_DETECTOR_FALLBACKS:
            try:
                img_data = np.asarray(
                    read_image(fpath, detector_shape=shape), dtype=float,
                )
                logger.debug('Loaded %s as %s (%dx%d)',
                             os.path.basename(fpath), name, shape[0], shape[1])
                return img_data
            except Exception:
                logger.debug("Failed to load %s as %s detector shape %s", fpath, name, shape, exc_info=True)
                continue
        return None

    def set_file(self, fname):
        """Changes the data file.

        args:
            fname: str, absolute path for data file
        """
        if fname != '':
            try:
                # with self.file_lock:
                #     with catch_h5py_file(fname, 'a') as _:
                #         pass

                self.ui.listData.itemSelectionChanged.disconnect(self.data_changed)
                self.ui.listData.clear()
                self.ui.listData.addItem('Loading...')
                # Reset the live-scan boundary cache — the next
                # update_data() must rebuild from the new scan.
                self._remember_displayed_frames()
                # self.set_open_enabled(False)
                self.file_thread.fname = fname
                self.file_thread.queue.put("set_datafile")
                self.ui.listData.itemSelectionChanged.connect(self.data_changed)
                self.new_scan = True
            except Exception:
                logger.exception("Failed to set file: %s", fname)
                return

    def data_changed(self, show_all=False):
        """Connected to itemSelectionChanged signal of listData.

        In viewer image mode with a multi-frame file, loads the
        selected frame on demand.  Otherwise falls through to the
        normal HDF5-based loading.
        """
        if not show_all:
            self.frame_ids.clear()
            items = self.ui.listData.selectedItems()
            if self.viewer_mode == 'xye':
                # XYE viewer stores the int key in UserRole
                self.frame_ids += sorted(
                    [str(item.data(QtCore.Qt.UserRole)) for item in items
                     if item.data(QtCore.Qt.UserRole) is not None])
            else:
                self.frame_ids += sorted([str(item.text()) for item in items])
            idxs = self.frame_ids
        else:
            idxs = self.frame_ids

        if (len(idxs) == 0) or ('No data' in idxs):
            # F1: no sleep on the Qt thread.  Pre-F1 this slept 100 ms
            # on every spurious empty-selection signal; multiplied by
            # many selectionChanged events that fire during list
            # rebuilds, this added visible UI stutter for no
            # functional reason (we return immediately afterwards).
            return

        # ── Image viewer ─────────────────────────────────────────────
        if self.viewer_mode == 'image':
            viewer_path = getattr(self, '_viewer_image_path', None)
            if viewer_path is not None:
                # Multi-frame: load selected frames on demand
                for idx_str in idxs:
                    idx = int(idx_str)
                    if idx not in self.data_2d:
                        self._load_single_frame(
                            viewer_path,
                            frame_idx=idx - 1,  # listData shows 1-based
                            frame_id=idx,
                        )
            # Single-frame: data already loaded by _load_image_file
            self.sigUpdate.emit()
            return

        # ── XYE viewer: data already loaded by scans_clicked ─────────
        if self.viewer_mode == 'xye':
            self.sigUpdate.emit()
            return

        # ── Normal mode: load from HDF5 ──────────────────────────────
        load_2d = self.update_2d

        if len(self.scan.frames.index) > 1:
            if len(idxs) == len(self.scan.frames.index):
                load_2d = False

        if load_2d:
            idxs_memory = [int(idx) for idx in idxs if int(idx) in self.data_2d.keys()]
        else:
            idxs_memory = [int(idx) for idx in idxs if int(idx) in self.data_1d.keys()]

        # Multi-frame combination is now done on demand by
        # get_frames_int_2d / get_frames_map_raw — no shared accumulator
        # state to maintain here. Just figure out which frames still
        # need to be loaded from disk.
        frame_ids = [int(idx) for idx in idxs
                    if int(idx) not in idxs_memory]

        if len(frame_ids) > 0:
            self.load_frames_data(frame_ids, load_2d)

        self.sigUpdate.emit()

    def closeEvent(self, event):
        self._h5pool.close_all()
        super().closeEvent(event)

    def data_reset(self):
        """Resets data in memory (self.frames, self.frame_ids, self.data_..
        """
        # During a live (non-batch) wrangler run the display is driven by
        # the in-memory per-frame hand-off in static_scan_widget.update_data.
        # This slot is wired to ``sigNewFile``, which the async file-thread
        # ``set_datafile`` emits a few ms after new_scan() — clearing the
        # freshly-populated data_1d/data_2d/frames before the throttled
        # refresh can render them.  That is the multi-scan Eiger "plots
        # stay blank" bug.  new_scan() already does the controlled reset
        # the live path needs, so skip the wipe while a run is active.
        if self.live_run_active:
            return
        self._h5pool.close(self.scan.data_file)
        self.frames.clear()
        self.frame_ids.clear()
        with self.data_lock:
            self.data_1d.clear()
            self.data_2d.clear()
        self.new_scan = True

    def open_folder(self):
        """Changes the directory being displayed in the file explorer.
        """
        dirname = QFileDialog().getExistingDirectory(
            caption='Choose Directory',
            dir='',
            options=QFileDialog.ShowDirsOnly
        )
        if os.path.exists(dirname):
            self.dirname = dirname
            save_session({'data_dir': dirname})
            self.frames.clear()
            with self.data_lock:
                self.data_1d.clear()
                self.data_2d.clear()
            self.new_scan = True
            self.update_scans()
    
    def set_open_enabled(self, enable):
        """Sets the save and open actions to enable
        
        args:
            enable: bool, if True actions are enabled
        """
        self.actionSaveDataAs.setEnabled(enable)
        self.paramMenu.setEnabled(enable)
        self.actionOpenFolder.setEnabled(enable)
        self.actionNewFile.setEnabled(enable)
        # self.ui.listScans.setEnabled(enable)
    
    def save_data_as(self):
        """Saves all data to hdf5 file. Also sets fname to be the
        selected file.
        """
        fname, _ = QFileDialog.getSaveFileName()
        with self.file_thread.lock:
            self.file_thread.new_fname = fname
            self.file_thread.queue.put("save_data_as")
        self.set_file(fname)
    
    def new_file(self):
        """Calls file dialog and sets the file name.
        """
        fname, _ = QFileDialog.getSaveFileName()
        self.set_file(fname)

    def load_frames_data(self, frame_ids, load_2d):
        """Dispatch a background ``_LoadFramesWorker`` for the given
        frame_ids and return immediately.

        M1: pre-M1 this method ran ``_load_frame_v2`` for every frame
        on the GUI thread, with a ``QApplication.processEvents()``
        yield every 4 reads (J3) to keep the UI from fully freezing.
        That kept the *event loop* alive but the HDF5 reads still
        blocked the slot — selecting 100 frames was still seconds of
        sluggish input.  Now the reads run on a dedicated worker
        QThread and stream results back via ``chunkLoaded`` signals.

        Cancellation: starting a new load while one is in flight
        cancels the old one — the existing worker's run loop bails
        between reads, and the new selection's worker takes over.
        This is what users expect when they're rapidly scrolling
        through frames; no queued-up stale loads catching up later.

        sigUpdate emits: ``_absorb_chunk`` emits ``sigUpdate`` after
        every batch absorbed, so the display catches up
        incrementally; the caller's own ``sigUpdate.emit()`` after
        this method returns is still useful for "selection-only"
        updates (cache hits) but the bulk of the visible refresh
        comes from the worker.
        """
        if not frame_ids:
            return
        # Cancel any in-flight worker.  We don't wait — the worker's
        # run loop checks the cancel flag between reads and exits
        # cleanly on its own; the new worker can start immediately.
        if self._load_worker is not None:
            try:
                self._load_worker.cancel()
            except (RuntimeError, AttributeError):
                pass

        # Spin up the new worker.  Lives on its own QThread; both get
        # cleaned up after ``finished`` signals via deleteLater.
        # N1: bump the generation counter so any in-flight chunks
        # from the previous worker are dropped.
        self._load_generation += 1
        gen = self._load_generation
        worker = _LoadFramesWorker(
            data_file=self.scan.data_file,
            file_lock=self.file_lock,
            gi=self.scan.gi,
            frame_ids=frame_ids,
            load_2d=load_2d,
            generation=gen,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        # Connect to the bound method directly (no lambda) so Qt can
        # detect the receiver's thread affinity (the H5Viewer lives
        # on the GUI thread) and use QueuedConnection automatically.
        # The lambda-wrapped version of this connect fell back to
        # DirectConnection on the worker thread, which made
        # ``_update_coalesce_timer.start()`` raise on Windows with
        # "Timers cannot be started from another thread" — Mac's Qt
        # build happened to tolerate this but Windows is strict.
        # ``load_2d`` rides in the signal payload (4th arg) now.
        worker.chunkLoaded.connect(self._absorb_chunk)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Drop refs after the thread is done so a later cancel() call
        # on a stale handle doesn't accidentally talk to a deleted
        # QObject.
        thread.finished.connect(self._clear_load_worker_refs)

        self._load_worker = worker
        self._load_thread = thread
        thread.start()

    def _absorb_chunk(self, generation, idx, frame, load_2d) -> None:
        """Slot for ``_LoadFramesWorker.chunkLoaded``.  Runs on the
        GUI thread; writes the loaded frame into the viewer dicts
        under ``data_lock`` and emits ``sigUpdate`` so the display
        repaints incrementally.

        N1: drops the chunk silently when its ``generation`` no
        longer matches ``self._load_generation`` — that means a
        newer load has already been started (via a new selection)
        and the chunk belongs to a cancelled worker whose run loop
        was mid-emit when the cancel was issued.  Queued Qt
        signals don't get clobbered by ``deleteLater``; they
        arrive on the GUI thread after the cancel.  Without this
        check those stale frames would land in ``data_1d`` /
        ``data_2d`` and pollute the current selection.
        """
        if generation != self._load_generation:
            logger.debug(
                "absorb_chunk dropping stale frame %s from gen=%s "
                "(current gen=%s)",
                idx, generation, self._load_generation,
            )
            return
        try:
            with self.data_lock:
                if not load_2d:
                    self.data_1d[int(idx)] = frame.copy(include_2d=False)
                else:
                    self.data_1d[int(idx)] = frame.copy(include_2d=False)
                    self.data_2d[int(idx)] = {
                        'map_raw': frame.map_raw,
                        'bg_raw': frame.bg_raw,
                        'mask': frame.mask,
                        'int_2d': frame.int_2d,
                        'gi_2d': frame.gi_2d,
                        'thumbnail': frame.thumbnail,
                    }
            # O6: coalesce display updates while a chunk burst is
            # streaming in.  Schedule (or restart) a debounced emit
            # rather than firing once per chunk.  ``_on_load_worker_finished``
            # forces a final emit so the burst's last paint is
            # guaranteed even if the timer is still pending.
            self._update_coalesce_timer.start()
        except (AttributeError, RuntimeError) as e:
            logger.debug("absorb_chunk skipped frame %s: %s", idx, e)

    def _clear_load_worker_refs(self) -> None:
        """Drop ``_load_worker`` / ``_load_thread`` once the worker
        signals finished — but ONLY if our handle still points at
        that worker.

        Self-review fix #3: queued ``thread.finished`` slot for
        worker A can arrive AFTER ``load_frames_data`` has already
        assigned worker B to ``self._load_worker``.  Pre-fix this
        slot would null out worker B's handle, leaving the next
        selection unable to ``cancel()`` it.  Identity-gate the
        clear so only the actually-finished worker's slot wins.

        The sender is the QThread of the worker that finished;
        compare to ``self._load_thread`` to identify it.
        """
        sender = self.sender()
        if sender is None or sender is self._load_thread:
            self._load_worker = None
            self._load_thread = None
            # O6: force one final sigUpdate so the burst's final
            # paint always reflects the full selection — otherwise
            # the coalesce timer might still be pending when the
            # last chunk arrived but the worker has now terminated.
            if self._update_coalesce_timer.isActive():
                self._update_coalesce_timer.stop()
            self.sigUpdate.emit()

    # Removed legacy load_frame_data — all reads now go through
    # LiveFrame.load_from_nexus via load_frames_data above.
    #
    # Removed get_frames_sum / _safe_accumulate / _raw_minus_bg and the
    # add_idxs/sub_idxs/sum_int_2d/sum_map_raw machinery: combining 2D
    # data across multiple selected frames is now done on demand by
    # display_data.get_frames_int_2d / get_frames_map_raw, which iterate
    # the current selection straight from data_2d. The old stateful
    # approach was both inconsistent with the 1D path (get_frames_int_1d)
    # and silently dead for sum_map_raw, which was never read anywhere.
