# -*- coding: utf-8 -*-
"""
@author: walroth
"""
# Standard library imports
from dataclasses import dataclass
import logging
import os
import time
import math
import warnings
from queue import Empty

logger = logging.getLogger(__name__)

_ORPHANED_FILE_THREADS = []
_ORPHANED_LOAD_WORKERS = []


def _retain_orphaned_file_thread(thread) -> None:
    """Keep a slow fileHandlerThread alive until it exits.

    Qt aborts if a QThread wrapper is destroyed while the native thread is
    still running.  Close normally waits for the persistent file thread, but a
    slow HDF5 read can outlive that bounded wait.  Retaining the Python wrapper
    prevents teardown from deleting it; the already-queued sentinel lets it exit
    after the current task.
    """
    if thread in _ORPHANED_FILE_THREADS:
        return
    _ORPHANED_FILE_THREADS.append(thread)

    def _forget_thread():
        try:
            _ORPHANED_FILE_THREADS.remove(thread)
        except ValueError:
            pass

    try:
        thread.finished.connect(_forget_thread)
    except Exception:
        pass


def _retain_orphaned_load_worker(worker, thread) -> None:
    """Keep a cancelled load worker/thread alive until Qt finishes it."""
    token = (worker, thread)
    if token in _ORPHANED_LOAD_WORKERS:
        return
    _ORPHANED_LOAD_WORKERS.append(token)

    def _forget_worker():
        try:
            _ORPHANED_LOAD_WORKERS.remove(token)
        except ValueError:
            pass

    try:
        thread.finished.connect(_forget_worker)
    except Exception:
        pass

# This module imports
import re
import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.export import read_xye
from xrd_tools.io.image import read_image, count_frames
from xdart.utils.session import load_session, save_session
from .ui.h5viewerUI import Ui_Form
from xdart.modules.live import LiveFrame
from xdart.utils.throttle import Coalescer
from .viewer_raw_lru import (
    VIEWER_RAW_LIMIT,
    clear_viewer_raw_lru,
    remember_viewer_raw_lru,
)
from .scan_threads import fileHandlerThread
from .browse_debug import browse_debug_log, sequence_summary
from .display_logic import xye_unit_from_filename
from .display_controllers import ImageViewerController
from xrd_tools.io import ImageSourceKind
from xdart.modules.frame_publication import (
    PublicationStore,
    publication_error_details,
    publication_from_frame_view,
    publication_from_live_frame,
    publication_has_2d_errors,
)
from xrd_tools.core import FrameView, numeric_metadata
from ...widgets import defaultWidget
from xdart import utils
from xdart.utils import catch_h5py_file as catch
from xdart.utils.h5pool import get_pool

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

try:
    # shiboken validity check: True only while the C++ half of a QObject is
    # still alive.  Used to avoid touching / GC-deleting a moveToThread'd
    # worker whose ``deleteLater`` has already run (which crashes with
    # "QObject: shared QObject was deleted directly").
    from shiboken6 import isValid as _qt_isvalid
except Exception:  # pragma: no cover - non-PySide6 / headless fallback
    def _qt_isvalid(obj):
        return obj is not None


QTreeWidget = QtWidgets.QTreeWidget
QTreeWidgetItem = QtWidgets.QTreeWidgetItem
QWidget = QtWidgets.QWidget
QFileDialog = QtWidgets.QFileDialog
QItemSelectionModel = QtCore.QItemSelectionModel


def _clear_raw_cache_for(viewer) -> None:
    """Reset hydrated-raw LRU state on real and lightweight test viewers."""
    viewer_rows_2d = getattr(viewer, "viewer_rows_2d", None)
    if viewer_rows_2d is not None:
        clear_viewer_raw_lru(viewer_rows_2d)


def _clear_publication_store_for(viewer) -> None:
    """Reset publication state when present on real or lightweight viewers."""
    store = getattr(viewer, "publication_store", None)
    if store is not None:
        store.clear()


_BROWSE_ONE_SHOT_METHODS = ("Single", "Overlay", "Waterfall", "Sum", "Average")
_BROWSE_ANCHOR_HEAVY_ATTEMPT_LIMIT = 3


def _qt_enum_value(value, default: int = 0) -> int:
    """Return the integer payload for Qt enums/flags across PyQt/PySide."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raw = getattr(value, "value", None)
        while raw is not None and raw is not value:
            try:
                return int(raw)
            except (TypeError, ValueError):
                next_raw = getattr(raw, "value", None)
                if next_raw is raw:
                    break
                raw = next_raw
        try:
            return int(value.__index__())
        except Exception:
            return default


def _qt_has_modifier(modifiers, modifier) -> bool:
    return bool(_qt_enum_value(modifiers) & _qt_enum_value(modifier))


def _frame_label_sort_key(value):
    text = value.text() if hasattr(value, "text") else value
    text = str(text)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


def _browse_one_shot_enabled(viewer) -> bool:
    if getattr(viewer, "viewer_mode", None) in ("image", "xye", "nexus"):
        return False
    if getattr(viewer, "_run_writing", False):
        return False
    return getattr(viewer, "_plot_method", None) in _BROWSE_ONE_SHOT_METHODS


def _browse_bulk_selection_enabled(viewer, selected_ids, *, show_all=False) -> bool:
    """True for browse gestures that should render from one selected-set batch."""
    if not _browse_one_shot_enabled(viewer):
        return False
    try:
        count = len(selected_ids)
    except TypeError:
        count = 0
    if show_all:
        return count > 1
    return count > 1


def _browse_debug_mode(viewer) -> str:
    try:
        return str(viewer.ui.plotMethod.currentText())
    except Exception:
        return str(getattr(viewer, "_plot_method", ""))


def _current_selected_frame_label(viewer, candidates=()):
    candidate_values = []
    for candidate in candidates or ():
        try:
            candidate_values.append(int(candidate))
        except (TypeError, ValueError):
            continue
    candidate_set = set(candidate_values)
    item = None
    try:
        item = viewer.ui.listData.currentItem()
    except Exception:
        item = None
    if item is not None:
        try:
            label = int(item.text())
        except (TypeError, ValueError):
            label = None
        if label is not None and (not candidate_set or label in candidate_set):
            return label
    return candidate_values[-1] if candidate_values else None


@dataclass
class _ViewerRow:
    """Lightweight row used by Image/XYE/NeXus viewer modes.

    These rows are not live detector frames.  Keeping them distinct from
    ``LiveFrame`` prevents browser-only state from accidentally entering
    reduction/display paths that expect real frame caches and integration
    results.
    """

    idx: int
    scan_info: dict
    nexus_preview_payload: dict | None = None


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
        worker.finished.connect(worker.deleteLater)  # before quit (see below)
        worker.finished.connect(thread.quit)
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
                 generation, hydrate_raw=True, parent=None):
        super().__init__(parent)
        self.data_file = data_file
        self.file_lock = file_lock
        self.gi = gi
        self.frame_ids = list(frame_ids)
        self.load_2d = load_2d
        self.hydrate_raw = bool(hydrate_raw)
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
                try:
                    with self.file_lock:
                        try:
                            file = pool.get(self.data_file)
                        except (FileNotFoundError, OSError) as e:
                            # The scan file doesn't exist / can't be opened
                            # (e.g. a default placeholder path before any scan
                            # is saved, or a deleted file). Bail cleanly rather
                            # than letting it reach the top-level "crashed
                            # unexpectedly" handler.
                            logger.debug(
                                "load worker gen=%s: data file unavailable "
                                "(%s); stopping", self.generation, e,
                            )
                            break
                        if file is None:
                            # Writer paused the pool — exit gracefully; the
                            # writer's resume() will trigger a sigUpdate that
                            # re-fires the GUI's data_changed slot.
                            logger.debug(
                                "load worker gen=%s: pool paused at idx=%s; "
                                "stopping",
                                self.generation, idx,
                            )
                            break
                        frame = _load_frame_v2(
                            file,
                            idx,
                            static=True,
                            gi=self.gi,
                            include_2d=bool(self.load_2d),
                            include_thumbnail=bool(self.load_2d),
                        )
                except (KeyError, IndexError, OSError, ValueError) as e:
                    logger.debug("load worker: frame %s skipped: %s",
                                 idx, e)
                    continue
                # Emit on the worker thread; Qt queues the slot
                # invocation back to the GUI thread automatically
                # because the signal target is a bound method on a
                # QObject living on the GUI thread.
                preview = (
                    frame.copy_for_display(include_2d=True)
                    if (self.load_2d and self.hydrate_raw and frame.map_raw is None)
                    else frame
                )
                self.chunkLoaded.emit(
                    self.generation, int(idx), preview, bool(self.load_2d),
                )
                # Publish the lightweight thumbnail-backed frame first, then
                # hydrate the detector source off the GUI thread.  A second
                # chunk replaces the preview when raw data arrives.
                if (self.load_2d and self.hydrate_raw
                        and frame.map_raw is None
                        and not self._cancel.is_set()):
                    try:
                        if frame._lazy_load_raw() and not self._cancel.is_set():
                            self.chunkLoaded.emit(
                                self.generation, int(idx), frame, True,
                            )
                    except Exception:
                        logger.debug("load worker: raw frame %s unavailable",
                                     idx, exc_info=True)
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
       can be unreliable across versions. Modifiers go through the
       shared enum helper used by the keyboard path.

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

        has_shift = _qt_has_modifier(mods_obj, QtCore.Qt.ShiftModifier)
        has_toggle_mod = (
            _qt_has_modifier(mods_obj, QtCore.Qt.ControlModifier)
            or _qt_has_modifier(mods_obj, QtCore.Qt.MetaModifier)
        )

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
                 viewer_rows_1d, viewer_rows_2d,
                 parent=None, data_lock=None, publication_store=None):
        super().__init__(parent)
        import threading as _threading
        self.data_lock = data_lock if data_lock is not None else _threading.RLock()
        self._init_data_objects(file_lock, local_path, dirname,
                                scan, frame, frame_ids, frames,
                                viewer_rows_1d, viewer_rows_2d, publication_store)
        self._init_ui()
        self._init_toolbar()
        self._connect_signals()
        self._init_file_thread()

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self, file_lock, local_path, dirname,
                           scan, frame, frame_ids, frames,
                           viewer_rows_1d, viewer_rows_2d, publication_store):
        """Initialize data references and state flags."""
        self.local_path = local_path
        self.file_lock = file_lock
        self.dirname = dirname
        self.scan = scan
        self.frame = frame
        self.frame_ids = frame_ids
        self.frames = frames
        self.viewer_rows_1d = viewer_rows_1d
        self.viewer_rows_2d = viewer_rows_2d
        self.publication_store = (
            publication_store if publication_store is not None else PublicationStore()
        )
        self.new_scan = True
        self.update_2d = True
        self.auto_last = True
        self.latest_idx = None
        self.new_scan_loaded = False
        # Int 1D/2D only: set True by scans_clicked when a .nxs is manually
        # selected in the browser, so the display-clearing sigNewFile cascade
        # (axes rebuild + bkg clear + cache wipe) is DEFERRED until the user
        # actually clicks a frame — the plots stay as-is on scan-select instead of
        # blanking.  Consumed by staticWidget.set_data on the first frame-click.
        # (Image/XYE/NeXus viewers auto-display frame 0 and never set this.)
        self._browser_scan_reset_pending = False
        self.viewer_mode = None
        # True only while a live (non-batch) wrangler run is in progress.
        # Suppresses ``data_reset`` (wired to the async ``sigNewFile``)
        # so the per-frame in-memory caches the live display depends on
        # aren't wiped mid-run.  Toggled by static_scan_widget.
        self.live_run_active = False
        # True while ANY run (live / batch / reintegrate) is writing the .nxs.
        # Set by the task-#68 run-state owner (static_scan_widget._enter/_exit_
        # run_state) alongside the displayframe's _processing_active, so the
        # frame-selection disk-load guard (data_changed) and the reader-side
        # hydration guard (display_data._hydrate_frame_from_disk) share one
        # source of truth and can't drift.  Distinct from live_run_active, which
        # is live-only and also drives data_reset / the file-thread repoint.
        self._run_writing = False
        self._displayed_list_count = 0
        self._displayed_last_label = None

    def _init_ui(self):
        """Set up the main UI form and default widget."""
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.layout = self.ui.gridLayout
        self._add_refresh_button()
        self.defaultWidget = defaultWidget()
        self.defaultWidget.sigSetUserDefaults.connect(self.set_user_defaults)
        self._apply_frames_panel_width(None)

    def _add_refresh_button(self):
        """Place Refresh in the DATA BROWSER header (top-right), per the redesign.

        Refresh re-reads the current directory listing (``update_scans``) — handy
        in Image/XYE Viewer modes where new files may be written outside xdart
        while a run is in progress.  The mockup puts it on the right of a header
        row above the lists; the bottom row then holds Show All / Auto Last /
        Metadata.
        """
        self.ui.refresh = QtWidgets.QPushButton('Refresh')
        self.ui.refresh.setObjectName('refresh')
        self.ui.refresh.setMaximumSize(QtCore.QSize(16777215, 25))
        # Refresh is placed on the RIGHT of the DATA BROWSER header row (built in
        # _init_toolbar, below the File/Config toolbar).  The bottom row below the
        # lists holds Show All / Auto Last / Metadata.

        # Bottom button row: Show All / Auto Last / Metadata (Refresh removed).
        btn_row = QtWidgets.QWidget()
        # Constrain the row to the button height so it doesn't steal vertical
        # stretch from the lists splitter above (which would leave the buttons
        # floating in the middle of the panel).
        btn_row.setFixedHeight(25)
        btn_row.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(self.ui.gridLayout.horizontalSpacing())
        # Move the existing buttons out of the grid into the row.
        self.ui.gridLayout.removeWidget(self.ui.show_all)
        self.ui.gridLayout.removeWidget(self.ui.auto_last)
        btn_layout.addWidget(self.ui.show_all)
        # Stage 4 (Direction A): the frame-metadata table is no longer inline in
        # the bottom-left — this button opens it as an on-demand popup (wired in
        # staticWidget._connect_signals -> _open_metadata_dialog).
        self.ui.metadata_btn = QtWidgets.QPushButton('Metadata ▾')
        self.ui.metadata_btn.setObjectName('metadata_btn')
        self.ui.metadata_btn.setMaximumSize(QtCore.QSize(16777215, 25))
        # Order: Show All | Metadata | Auto Last (Auto Last rightmost).
        btn_layout.addWidget(self.ui.metadata_btn)
        btn_layout.addWidget(self.ui.auto_last)
        self.ui.gridLayout.addWidget(btn_row, 3, 0, 1, 2)

    def refresh_directory(self):
        """Re-read the current directory's file listing (Refresh button).

        Repopulates ``listScans`` from ``self.dirname``; ``update_scans``
        preserves the XYE-overlay selection so a refresh mid-compare doesn't
        drop the user's selected files.
        """
        self.update_scans()

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

        # Toolbar buttons.  objectName'd so the theme can suppress the oversized
        # QToolButton menu-indicator arrow (themes/dark.py).
        self.fileButton = QtWidgets.QToolButton()
        self.fileButton.setObjectName('fileMenuButton')
        self.fileButton.setText('File')
        self.fileButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.fileButton.setMenu(self.fileMenu)
        self.paramButton = QtWidgets.QToolButton()
        self.paramButton.setObjectName('configMenuButton')
        self.paramButton.setText('Config')
        self.paramButton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.paramButton.setMenu(self.paramMenu)

        self.toolbar.addWidget(self.fileButton)
        self.toolbar.addWidget(self.paramButton)

        # DATA BROWSER header row, sitting just BELOW the File/Config toolbar: a
        # section title on the left, Refresh on the right.  Stacked with the
        # toolbar inside one wrapper so the pair occupies grid row 0 together
        # (no renumbering of the Scans/Data labels + lists below).
        self.dataBrowserBar = QtWidgets.QFrame()
        self.dataBrowserBar.setObjectName('dataBrowserBar')
        _db = QtWidgets.QHBoxLayout(self.dataBrowserBar)
        _db.setContentsMargins(2, 0, 0, 0)
        _db.setSpacing(6)
        self.dataBrowserHeader = QtWidgets.QLabel('DATA BROWSER')
        self.dataBrowserHeader.setObjectName('dataBrowserHeader')
        _db.addWidget(self.dataBrowserHeader)
        _db.addStretch(1)
        if hasattr(self.ui, 'refresh'):
            _db.addWidget(self.ui.refresh)

        header = QtWidgets.QWidget()
        _hdr = QtWidgets.QVBoxLayout(header)
        _hdr.setContentsMargins(0, 0, 0, 0)
        _hdr.setSpacing(2)
        _hdr.addWidget(self.toolbar)
        _hdr.addWidget(self.dataBrowserBar)
        self.layout.addWidget(header, 0, 0, 1, 2)

    def _connect_signals(self):
        """Wire signal/slot connections for list widgets and menu actions."""
        self.actionSetDefaults.triggered.connect(self.defaultWidget.show)
        self.ui.listScans.itemDoubleClicked.connect(self.scans_clicked)
        self.ui.listScans.itemClicked.connect(self._scans_single_clicked)
        self.ui.listScans.currentItemChanged.connect(self._scans_current_changed)
        self.ui.listScans.itemSelectionChanged.connect(self._scans_selection_changed)
        self.ui.listScans.installEventFilter(self)
        self.ui.listData.itemSelectionChanged.connect(self.data_changed)
        self.ui.listData.installEventFilter(self)
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
        self.ui.refresh.clicked.connect(self.refresh_directory)
        self.actionOpenFolder.triggered.connect(self.open_folder)
        self.actionSaveDataAs.triggered.connect(self.save_data_as)
        self.actionNewFile.triggered.connect(self.new_file)

    def set_data_selection_mode(self, plot_method):
        """Update internal plot-method state and reconcile listData
        selection when switching between modes.

        listData stays in ExtendedSelection at all times. When the
        user enters ``Single`` mode from a multi-selection state, the
        selected set is preserved and rendered through the same
        multi-frame path as Overlay. Plain clicks still replace the
        selection in Single mode, while Cmd/Ctrl clicks can deselect
        individual rows. The click
        filter consults ``self._plot_method`` to decide how to handle
        plain clicks.
        """
        prev_method = self._plot_method
        self._plot_method = plot_method
        if plot_method == 'Single' and prev_method != 'Single':
            # Single mode now plots a multi-selection exactly like Overlay; the
            # only Single-specific affordance is that plain clicks replace while
            # Cmd/Ctrl clicks can remove rows.  Keep any existing selection when
            # switching modes and just refresh the display model.
            self.data_changed()

    def _init_file_thread(self):
        """Create the background file handler thread.

        The thread starts lazily on the first queued file operation.  A fresh
        Controls/Viewer widget should not carry an idle QThread while the rest
        of the Qt tree is still being constructed.
        """
        self.file_thread = fileHandlerThread(self.scan, self.frame,
                                             self.file_lock,
                                             frame_ids=self.frame_ids,
                                             frames=self.frames,
                                             data_lock=self.data_lock)
        self.file_thread.sigTaskDone.connect(self.thread_finished)
        self.file_thread.sigNewFile.connect(self.sigNewFile.emit)
        self.file_thread.sigUpdate.connect(self._emit_file_thread_update)
        self._file_thread_shutdown = False
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
        # Keep only a small working set of hydrated detector arrays. Reduced
        # results and thumbnails stay cached for every loaded frame.
        self._raw_cache_limit = 8
        # O6: coalesce ``sigUpdate`` emits while a chunk burst is
        # streaming in from ``_LoadFramesWorker``.  Without this, a
        # 100-frame selection fires 100 full-display repaints in
        # rapid succession.  With it, the burst is debounced to a
        # single emit ~100 ms after the last chunk lands — and the
        # worker-finished slot forces a final emit so the last
        # paint always reflects the full selection.
        # O6: one debounced emit per chunk burst (the shared coalescing
        # idiom — see xdart.utils.throttle for the throttle-vs-debounce
        # contract; _on_load_worker_finished force-flushes the tail).
        self._update_coalesce_timer = Coalescer(100, mode="debounce",
                                                parent=self)
        self._update_coalesce_timer.triggered.connect(
            self._emit_coalesced_update)
        # FREEZE FIX (part 2): debounce the DISK LOAD too.  data_changed runs fully
        # per selection event, and its load_frames_data does a blocking
        # _teardown_load_worker (thread.wait ~2 s) — so a rapid shift/ctrl /
        # arrow-hold burst fires a FLOOD of 2 s waits on the GUI thread -> beachball
        # (the render debounce above alone doesn't cover this).  Coalesce the load to
        # ONE call for the final selection.
        self._pending_load_ids = None
        self._pending_load_2d = True
        self._load_coalesce_timer = Coalescer(100, mode="debounce", parent=self)
        self._load_coalesce_timer.triggered.connect(self._flush_pending_load)
        self._xye_parse_cache = {}
        # Fast shift+arrow selection sweeps can emit itemSelectionChanged dozens
        # of times per second.  Debounce the whole normal-mode body so the O(k)
        # selection parse / cache-probe work runs once for the final selection,
        # not once for every intermediate range.
        self._pending_data_changed = False
        self._selection_coalesce_timer = Coalescer(100, mode="debounce",
                                                  parent=self)
        self._selection_coalesce_timer.triggered.connect(
            self._flush_pending_data_changed)
        self._browse_gesture_active = False
        self._browse_pending_data_changed = False
        self._browse_one_shot_pending_render = False
        self._browse_one_shot_load_generation = None
        self._browse_one_shot_target_labels = ()
        self._browse_one_shot_publications = {}
        self._browse_one_shot_signature = None
        self._browse_one_shot_anchor_label = None
        self._browse_anchor_heavy_after_next_render = None
        self._browse_anchor_heavy_inflight_label = None
        self._browse_anchor_heavy_attempt_key = None
        self._browse_anchor_heavy_attempt_count = 0
        self._browse_anchor_heavy_attempt_logged = False
        self._browse_last_selection_signature = None
        self._overlay_visit_intent_labels = []
        self._overlay_visit_inflight_labels = ()
        self._overlay_hydrated_pending_append_labels = []

    def _ensure_file_thread_running(self) -> None:
        """Start the persistent file loader on first real file operation."""
        ft = getattr(self, "file_thread", None)
        if ft is None or getattr(self, "_file_thread_shutdown", False):
            return
        try:
            if not ft.isRunning():
                ft.start(Qt.QtCore.QThread.LowPriority)
        except RuntimeError:
            logger.debug("file_thread could not be started", exc_info=True)
        
    def load_starting_defaults(self):
        default_path = os.path.join(utils.get_config_dir(), "last_defaults.json")
        if os.path.exists(default_path):
            self.defaultWidget.load_defaults(fname=default_path)
        else:
            self.defaultWidget.save_defaults(fname=default_path)

    def set_user_defaults(self):
        default_path = os.path.join(utils.get_config_dir(), "last_defaults.json")
        self.defaultWidget.save_defaults(fname=default_path)

    def update(self):
        """Refresh the current file/frame selection."""
        self.update_data()

    # File extensions for viewer modes
    _IMAGE_EXTS = {'.tif', '.tiff', '.raw', '.edf', '.h5', '.hdf5', '.nxs'}
    _XYE_EXTS = {'.xye'}
    _NEXUS_EXTS = {'.h5', '.hdf5', '.nxs'}

    @staticmethod
    def _natural_sort_key(text):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', text)]

    def update_scans(self):
        """Populate listScans with files in the current directory.

        In normal mode, shows HDF5 files and directories.
        In image viewer mode, shows image files and directories.
        In xye viewer mode, shows xye files and directories.
        In nexus viewer mode, shows HDF5/NeXus files and directories.
        """
        if not os.path.exists(self.dirname):
            return

        lw = self.ui.listScans
        was_blocked = lw.blockSignals(True)
        try:
            # XYE overlay is built by modifier-free multi-select; during a live
            # run new .xye files keep arriving and repopulate this list.  Capture
            # the current selection by name so we can restore it after the
            # rebuild — otherwise the overlay resets every time a file is written
            # (the crux of the real-time compare/track workflow).
            preserve_selection = self.viewer_mode == 'xye'
            selected_names = (
                {item.text() for item in lw.selectedItems()}
                if preserve_selection else set()
            )
            current_name = (
                lw.currentItem().text()
                if preserve_selection and lw.currentItem() is not None else None
            )

            lw.clear()
            lw.addItem('..')

            # os.scandir exposes d_type from the single readdir, so entry.is_dir()
            # needs no extra stat() per entry (unlike os.path.isdir).  This is the
            # hot path on Refresh / folder navigation / browsing large raw-image
            # dirs, and every saved stat is a network round-trip on the SSRL NFS
            # deployment.  is_dir() follows symlinks by default, matching isdir.
            with os.scandir(self.dirname) as it:
                entries = sorted(it, key=lambda e: self._natural_sort_key(e.name))
            for entry in entries:
                name = entry.name
                if entry.is_dir():
                    lw.addItem(name + '/')
                else:
                    ext = os.path.splitext(name)[1].lower()
                    if self.viewer_mode == 'image':
                        if ext in self._IMAGE_EXTS:
                            lw.addItem(name)
                    elif self.viewer_mode == 'xye':
                        if ext in self._XYE_EXTS:
                            lw.addItem(name)
                    elif self.viewer_mode == 'nexus':
                        if ext in self._NEXUS_EXTS:
                            lw.addItem(name)
                    else:
                        # Normal mode: only HDF5/NeXus scan files
                        if name.split('.')[-1] in ('h5', 'hdf5', 'nxs'):
                            lw.addItem(name)

            # Restore the prior multi-selection by name (signals stay blocked, so
            # this doesn't re-trigger a load — the already-loaded curves remain).
            if selected_names:
                for row in range(lw.count()):
                    item = lw.item(row)
                    if item.text() in selected_names:
                        item.setSelected(True)
                    if item.text() == current_name:
                        lw.setCurrentItem(item, QItemSelectionModel.NoUpdate)
        finally:
            lw.blockSignals(was_blocked)

    def select_last_scan_entry(self):
        """Select the last data-file entry in listScans (the most recent
        output, since the list is naturally sorted), triggering its load via
        itemSelectionChanged.  Skips '..' and directories.  Returns the
        selected row, or -1 if there is no selectable entry."""
        lw = self.ui.listScans
        last_row = -1
        for row in range(lw.count()):
            text = lw.item(row).text()
            if text == '..' or text.endswith('/'):
                continue
            last_row = row
        if last_row >= 0:
            lw.setCurrentRow(last_row, QItemSelectionModel.ClearAndSelect)
        return last_row

    def select_most_recent_scan_entry(self):
        """Select the most recently *modified* data-file entry in listScans.

        ``select_last_scan_entry`` picks the name-last row, but the file that
        was saved last isn't necessarily name-last (mixed prefixes, or files
        left over from earlier runs that sort later).  Pick by mtime so the
        final pattern just written is the one selected; among files with the
        same mtime (a batch flush writes them together) the name-last one
        (highest frame number) wins.  Returns the selected row, or -1."""
        lw = self.ui.listScans
        best_row, best_mtime = -1, None
        for row in range(lw.count()):
            text = lw.item(row).text()
            if text == '..' or text.endswith('/'):
                continue
            try:
                mtime = os.path.getmtime(os.path.join(self.dirname, text))
            except OSError:
                continue
            if best_mtime is None or mtime >= best_mtime:
                best_mtime, best_row = mtime, row
        if best_row >= 0:
            lw.setCurrentRow(best_row, QItemSelectionModel.ClearAndSelect)
        return best_row

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

    def update_data(self, emit_update=True, *, force_rebuild: bool = False):
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

        if os.environ.get("XDART_PERF"):
            logger.info(
                "[PERF] h5viewer.update_data: index_len=%d live_run=%s "
                "new_scan_loaded=%s -> %s",
                len(frame_index), getattr(self, "live_run_active", None),
                getattr(self, "new_scan_loaded", None),
                "CLEAR" if len(frame_index) == 0 else "keep/rebuild")

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
        if (not force_rebuild
                and current_count >= 1
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

        if not force_rebuild and _idxs == items:
            if self.new_scan_loaded:
                self.new_scan_loaded = False
                self.ui.listData.setCurrentRow(-1)
                self.frame_ids.clear()
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
        if (not force_rebuild
                and len(items) >= 1
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

        # Restore the prior selection by frame TEXT (id), not row index.  Under
        # streaming the pool completes frames out of order, so the list above is
        # re-sorted on (nearly) every coalesce tick; a saved row index then
        # points at a DIFFERENT frame, so a single-frame click would jump to a
        # stale neighbour and get re-clobbered each tick (the "shows 47 while 52
        # is selected" bug).  Text lookup is immune to reordering — this is what
        # the multi-select branch always did; the single-select branch used to
        # use the fragile row index.  Fall back to the row only when nothing was
        # selected (or the selected frame no longer exists).
        if previous_sel:
            current_item = None
            for text in previous_sel:
                for item in self.ui.listData.findItems(text, QtCore.Qt.MatchExactly):
                    item.setSelected(True)
                    if current_item is None:
                        current_item = item
            if current_item is not None:
                self.ui.listData.setCurrentItem(current_item)
            else:
                self.ui.listData.setCurrentRow(previous_loc)
        else:
            self.ui.listData.setCurrentRow(previous_loc)

        self.ui.listData.blockSignals(False)
        _emit_changed()

    def show_all(self):

        if len(self.scan.frames.index) > 0:
            self.frame_ids.clear()
            self.frame_ids += [str(idx) for idx in self.scan.frames.index]
            lw = self.ui.listData
            was_blocked = lw.blockSignals(True)
            try:
                lw.selectAll()
            finally:
                lw.blockSignals(was_blocked)
            browse_debug_log(
                logger,
                "gesture_settle",
                trigger_source="Show All",
                mode=_browse_debug_mode(self),
                selected=sequence_summary(self.frame_ids),
            )

        self.new_scan = False
        self.data_changed(show_all=True)

    def thread_finished(self, task):
        self.update()
        if getattr(self, '_auto_select_last_on_finish', False):
            self._auto_select_last_on_finish = False
            if self.ui.listData.count() > 0:
                self.set_current_frame(self.ui.listData.count() - 1)
        self.sigThreadFinished.emit()
    
    def _scans_single_clicked(self, q):
        """Handle single click in listScans — uniform across ALL modes.

        Single click navigates folders and loads files everywhere (Vivek):
        the Image/NeXus viewers and the normal Int 1D/2D modes act directly;
        XYE mode routes FILE loads through _scans_selection_changed (it
        fires after the selection settles, so Shift+arrow multi-select works
        and nothing double-fires) and handles only NAVIGATION here.
        """
        if getattr(self, '_suspend_scan_selection_loads', False):
            return
        if self.viewer_mode == 'xye':
            text = q.text()
            if text == '..' or text.endswith('/'):
                self.scans_clicked(q)
            return
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
        if getattr(self, '_suspend_scan_selection_loads', False):
            return
        if current is None:
            return
        # XYE mode: handled by _scans_selection_changed
        if self.viewer_mode == 'xye':
            return
        item_text = current.text()
        # Skip directories and ".." — don't auto-navigate on arrow keys
        if item_text == '..' or item_text.endswith('/'):
            return
        # Uniform across modes (Vivek): arrow keys load .nxs in the normal
        # Int 1D/2D modes exactly like the viewers (the file-handler queue
        # serializes the loads).
        self.scans_clicked(current)

    def _scans_selection_changed(self):
        """Handle selection changes in listScans (XYE viewer mode only).

        Uses itemSelectionChanged which fires after the selection is
        fully updated, so Shift+arrow works correctly for multi-select.
        """
        if getattr(self, '_suspend_scan_selection_loads', False):
            return
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

    def _browse_bulk_keys(self):
        return {
            _qt_enum_value(QtCore.Qt.Key_Up),
            _qt_enum_value(QtCore.Qt.Key_Down),
            _qt_enum_value(QtCore.Qt.Key_PageUp),
            _qt_enum_value(QtCore.Qt.Key_PageDown),
            _qt_enum_value(QtCore.Qt.Key_Home),
            _qt_enum_value(QtCore.Qt.Key_End),
        }

    def _begin_browse_gesture(self) -> None:
        self._browse_gesture_active = True
        self._browse_pending_data_changed = False
        browse_debug_log(
            logger,
            "gesture_begin",
            trigger_source="arrow-press",
            mode=_browse_debug_mode(self),
            selected=sequence_summary(getattr(self, "frame_ids", ())),
        )
        for attr in (
            "_selection_coalesce_timer",
            "_load_coalesce_timer",
            "_update_coalesce_timer",
        ):
            timer = getattr(self, attr, None)
            if timer is not None and timer.isActive():
                timer.stop()

    def _finish_browse_gesture(self) -> None:
        if not getattr(self, "_browse_gesture_active", False):
            return
        self._browse_gesture_active = False
        self._browse_pending_data_changed = False
        browse_debug_log(
            logger,
            "gesture_settle",
            trigger_source="arrow-release",
            mode=_browse_debug_mode(self),
            selected=sequence_summary(getattr(self, "frame_ids", ())),
        )
        self.data_changed()

    def _handle_browse_key_event(self, event) -> bool:
        """Track held Shift+navigation gestures without doing browse work.

        The QListWidget still owns selection mechanics; this method only
        brackets the burst so itemSelectionChanged signals cannot hydrate or
        repaint until the key is released.
        """
        if not _browse_one_shot_enabled(self):
            return False
        try:
            event_type = event.type()
            key = _qt_enum_value(event.key())
        except AttributeError:
            return False
        key_press = QtCore.QEvent.Type.KeyPress
        key_release = QtCore.QEvent.Type.KeyRelease
        if event_type not in (key_press, key_release):
            return False

        bulk_keys = H5Viewer._browse_bulk_keys(self)
        shift_key = _qt_enum_value(QtCore.Qt.Key_Shift)
        try:
            is_auto_repeat = bool(event.isAutoRepeat())
        except AttributeError:
            is_auto_repeat = False

        if event_type == key_press:
            if key in bulk_keys and _qt_has_modifier(
                    event.modifiers(), QtCore.Qt.ShiftModifier):
                H5Viewer._begin_browse_gesture(self)
                return True
            return False

        if not getattr(self, "_browse_gesture_active", False):
            return False
        if is_auto_repeat:
            return True
        if key in bulk_keys or key == shift_key:
            H5Viewer._finish_browse_gesture(self)
            return True
        return False

    def eventFilter(self, obj, event):
        """Handle Enter/Return key on listScans to navigate into folders."""
        if obj is self.ui.listData:
            H5Viewer._handle_browse_key_event(self, event)
            return False
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
        if getattr(self, '_suspend_scan_selection_loads', False):
            return
        try:
            try:
                item_text = q.data(0)
            except RuntimeError:
                # Double-click after single-click navigation: the first click
                # rebuilt listScans, so the second event can deliver an item
                # whose C++ object is already deleted.  Nothing to act on.
                return

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
            if self.viewer_mode == 'nexus':
                self._load_nexus_file(fpath)
                return

            # ── Normal mode: open HDF5 scan ───────────────────────────
            # Manual browser select in Int 1D/2D: DEFER the display reset until a
            # frame is clicked, so the plots stay as-is on select (the frame list
            # still populates + deselects).  Only reached in normal mode — the
            # viewer branches above return first, keeping their auto-first-frame.
            current_fname = getattr(getattr(self, 'file_thread', None), 'fname', None)
            if (getattr(self, '_run_writing', False)
                    or (fpath and fpath == current_fname)):
                self.set_file(fpath)
                return
            self._browser_scan_reset_pending = True
            self.set_file(fpath)
            self.new_scan_loaded = True
        except AttributeError:
            pass

    # ── Viewer mode loaders ───────────────────────────────────────────────

    def _upsert_viewer_row_publication(self, label, scan_info) -> None:
        """Phase 3c: mirror a viewer-mode row's METADATA (only) into the shared
        publication store, so the metadata panel reads it store-first like every
        other mode.  Metadata-only (no 1D/2D arrays) keeps the integration
        display path from ever rendering a browser row; the row itself stays in
        ``viewer_rows_1d`` for the viewer controller's own plotting/inspection."""
        store = getattr(self, "publication_store", None)
        if store is None:
            return
        try:
            info = dict(scan_info or {})
            view = FrameView.from_results(
                label=label, metadata_raw=info,
                metadata_numeric=numeric_metadata(info))
            store.upsert(publication_from_frame_view(
                view, generation=store.generation,
                source_identity=str(info.get("source_file", label))))
        except Exception:
            logger.debug("viewer-row publication upsert failed for %s", label,
                         exc_info=True)

    def _load_xye_files(self):
        """Load all selected xye files from listScans for overlay.

        Each file gets a sequential index (1, 2, 3, …) in viewer_rows_1d.
        listData is populated with filenames and all rows are selected
        so the display frame renders every curve.
        """
        selected = self.ui.listScans.selectedItems()
        if not selected:
            return

        with self.data_lock:
            self.viewer_rows_1d.clear()
            self.viewer_rows_2d.clear()
            _clear_publication_store_for(self)
            _clear_raw_cache_for(self)
        self.frame_ids.clear()

        idx = 1
        cache = getattr(self, '_xye_parse_cache', None)
        if cache is None:
            cache = {}
            self._xye_parse_cache = cache
        for item in selected:
            item_text = item.text()
            # Skip directories
            if item_text == '..' or item_text.endswith('/'):
                continue
            fpath = os.path.join(self.dirname, item_text)
            abspath = os.path.abspath(fpath)
            try:
                mtime = os.path.getmtime(abspath)
                key = (abspath, mtime)
                cached = cache.get(key)
                if cached is None:
                    cached = read_xye(abspath)
                    cache[key] = cached
                    for old_key in [
                        old for old in cache
                        if old[0] == abspath and old != key
                    ]:
                        cache.pop(old_key, None)
                xdata, ydata, sigma = cached
            except Exception:
                logger.debug("Could not load xye file %s", fpath, exc_info=True)
                continue

            # xdart XYE exports encode the integration unit in the prefix
            # (iq → q_A^-1, itth → 2th_deg); unprefixed files default to Q,
            # never an assumed 2θ.  The single source of truth is the pure
            # xye_unit_from_filename.
            unit = xye_unit_from_filename(fpath)

            int_1d = IntegrationResult1D(
                radial=xdata, intensity=ydata, sigma=sigma, unit=unit,
            )
            frame = LiveFrame(idx=idx, static=True, gi=False)
            frame.int_1d = int_1d
            frame.scan_info = {'source_file': os.path.basename(fpath)}

            with self.data_lock:
                self.viewer_rows_1d[idx] = frame
            _upsert = getattr(self, '_upsert_viewer_row_publication', None)
            if _upsert is not None:
                _upsert(idx, frame.scan_info)
            self.frame_ids.append(str(idx))
            idx += 1

        if len(self.viewer_rows_1d) == 0:
            return

        # Populate listData with loaded filenames (all selected).
        # Display filename but store numeric index in UserRole so
        # data_changed can map back to viewer_rows_1d keys.
        self.ui.listData.blockSignals(True)
        self.ui.listData.clear()
        for key in self.viewer_rows_1d:
            frame = self.viewer_rows_1d[key]
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

    def _load_nexus_file(self, fpath):
        """Inspect a NeXus/HDF5 file without loading large arrays."""
        from xrd_tools.io import inspect_nexus

        with self.data_lock:
            self.viewer_rows_1d.clear()
            self.viewer_rows_2d.clear()
            _clear_publication_store_for(self)
            _clear_raw_cache_for(self)
        self.frame_ids.clear()
        was_blocked = self.ui.listData.blockSignals(True)
        try:
            self.ui.listData.clear()
        finally:
            self.ui.listData.blockSignals(was_blocked)

        try:
            summary = inspect_nexus(fpath, max_depth=4, max_children=300)
        except Exception as exc:
            logger.warning("Could not inspect NeXus file %s: %s", fpath, exc)
            self.ui.labelCurrent.setText(os.path.basename(fpath))
            self._remember_displayed_frames()
            self.sigUpdate.emit()
            return

        rows = self._nexus_summary_rows(summary)
        self._viewer_nexus_path = fpath
        self._viewer_nexus_summary = summary

        nexus_rows = []
        with self.data_lock:
            for row_id, (label, info) in enumerate(rows, start=1):
                scan_info = dict(info)
                scan_info.setdefault("source_file", os.path.basename(fpath))
                self.viewer_rows_1d[row_id] = _ViewerRow(
                    idx=row_id,
                    scan_info=scan_info,
                )
                nexus_rows.append((row_id, scan_info))
        # Phase 3c: mirror each row's metadata into the publication store so the
        # metadata panel reads it store-first (the _ViewerRow stays in viewer_rows_1d
        # for the NeXus inspector controller).
        _upsert = getattr(self, '_upsert_viewer_row_publication', None)
        if _upsert is not None:
            for row_id, scan_info in nexus_rows:
                _upsert(row_id, scan_info)

        was_blocked = self.ui.listData.blockSignals(True)
        try:
            self.ui.listData.clear()
            initial_row = None
            for row, (_label, info) in enumerate(rows):
                if info.get("nexus_preview_kind") in ("plot_1d", "image_2d"):
                    initial_row = row
                    break
            if initial_row is None and rows:
                initial_row = 0

            for row_id, (label, _info) in enumerate(rows, start=1):
                item = QtWidgets.QListWidgetItem(str(label))
                item.setData(QtCore.Qt.UserRole, row_id)
                self.ui.listData.addItem(item)
            if initial_row is not None:
                self.ui.listData.setCurrentRow(initial_row)
                self.frame_ids.append(str(initial_row + 1))
        finally:
            self.ui.listData.blockSignals(was_blocked)

        self._refresh_nexus_selected_preview(self.frame_ids)
        self.ui.labelCurrent.setText(os.path.basename(fpath))
        self._remember_displayed_frames()
        self.sigUpdate.emit()

    def _refresh_nexus_selected_preview(self, idxs):
        """Attach a bounded dataset preview to the selected NeXus row."""
        if not idxs:
            return
        try:
            row_id = int(idxs[0])
        except (TypeError, ValueError):
            return
        viewer_path = getattr(self, "_viewer_nexus_path", None)
        if not viewer_path:
            return
        with self.data_lock:
            frame = self.viewer_rows_1d.get(row_id)
        if frame is None:
            return
        info = dict(getattr(frame, "scan_info", None) or {})
        dataset_path = info.get("dataset_path")
        if not dataset_path:
            return
        frame.nexus_preview_payload = None
        try:
            preview = self._load_nexus_preview_payload(viewer_path, info)
        except Exception as exc:
            info["preview_error"] = str(exc)
        else:
            payload, preview_info = preview
            frame.nexus_preview_payload = payload
            info.update(preview_info)
        frame.scan_info = info
        # Phase 3c: refresh the store mirror with the preview-enriched metadata.
        _upsert = getattr(self, '_upsert_viewer_row_publication', None)
        if _upsert is not None:
            _upsert(row_id, info)

    def _load_nexus_preview_payload(self, viewer_path, info):
        from xrd_tools.io import preview_nexus_dataset, read_nexus_dataset

        dataset_path = info["dataset_path"]
        shape = tuple(info.get("_shape") or ())
        preview_kind = info.get("nexus_preview_kind")
        attrs = dict(info.get("_attrs") or {})
        label = (
            attrs.get("long_name")
            or attrs.get("description")
            or attrs.get("title")
            or os.path.basename(str(dataset_path))
        )

        if preview_kind == "plot_1d":
            data_selection, axis_sel = self._nexus_1d_selection(shape)
            data = read_nexus_dataset(
                viewer_path, dataset_path, selection=data_selection,
            )
            y = np.asarray(data.data, dtype=float).ravel()
            x, x_label, x_unit = self._nexus_axis_values(
                viewer_path,
                info.get("x_axis_path"),
                y.size,
                info.get("x_label") or "index",
                info.get("x_unit") or "",
                axis_selection=axis_sel,
            )
            if x.shape != y.shape:
                x = np.arange(y.size, dtype=float)
                x_label, x_unit = "index", ""
            payload = {
                "kind": "plot_1d",
                "x": x,
                "y": y,
                "x_label": x_label,
                "x_unit": x_unit,
                "y_label": info.get("y_label") or str(label),
                "y_unit": info.get("y_unit") or str(attrs.get("units") or ""),
                "label": str(label),
            }
            preview_info = {
                "preview_selection": data.selection,
                "preview_truncated": bool(
                    self._nexus_selection_truncated(shape, data_selection)
                ),
                "preview": self._nexus_value(data.data, max_len=800),
            }
            return payload, preview_info

        if preview_kind == "image_2d":
            data_selection, row_sel, col_sel = self._nexus_2d_selection(shape)
            data = read_nexus_dataset(
                viewer_path, dataset_path, selection=data_selection,
            )
            image = np.asarray(data.data, dtype=float)
            if image.ndim != 2:
                image = np.squeeze(image)
            if image.ndim != 2:
                raise ValueError(f"Cannot preview {dataset_path}: expected 2D slice")
            x, x_label, x_unit = self._nexus_axis_values(
                viewer_path,
                info.get("x_axis_path"),
                image.shape[1],
                info.get("x_label") or "x",
                info.get("x_unit") or "",
                axis_selection=col_sel,
            )
            y, y_label, y_unit = self._nexus_axis_values(
                viewer_path,
                info.get("y_axis_path"),
                image.shape[0],
                info.get("y_label") or "y",
                info.get("y_unit") or "",
                axis_selection=row_sel,
            )
            payload = {
                "kind": "image_2d",
                "image": image,
                "x": x,
                "y": y,
                "x_label": x_label,
                "x_unit": x_unit,
                "y_label": y_label,
                "y_unit": y_unit,
                "label": str(label),
            }
            preview_info = {
                "preview_selection": data.selection,
                "preview_truncated": bool(self._nexus_selection_truncated(shape, data_selection)),
                "preview": self._nexus_value(data.data, max_len=800),
            }
            return payload, preview_info

        preview = preview_nexus_dataset(viewer_path, dataset_path, max_items=64)
        return None, {
            "preview_selection": preview.selection,
            "preview_truncated": bool(preview.truncated),
            "preview": self._nexus_value(preview.data, max_len=800),
        }

    @staticmethod
    def _nexus_1d_selection(shape, *, max_points=8192):
        """Bounded selection for the 1D GUI preview.

        Strides a long 1D axis down to ``<= max_points`` so a generic NeXus
        file's huge 1D dataset can't freeze the GUI; xdart's ~2000-pt curves
        read whole (stride 1).  Returns ``(data_selection, axis_selection)``
        so the x-axis is strided the same way (matching lengths).  The
        headless ``read_nexus_dataset`` (called without a selection) still
        reads in full — only this GUI preview path downsamples.
        """
        n = int(shape[-1]) if shape else 0
        stride = max(1, math.ceil(n / max_points)) if n > max_points else 1
        # Stride 1 stays a plain ``:`` (full read) so short xdart curves are
        # unchanged; only an oversized axis gets a strided slice.
        axis_sel = slice(None) if stride == 1 else slice(None, None, stride)
        if len(shape) <= 1:
            return axis_sel, axis_sel
        return tuple(0 for _ in range(len(shape) - 1)) + (axis_sel,), axis_sel

    @staticmethod
    def _nexus_2d_selection(shape, *, max_pixels=262144):
        if len(shape) < 2:
            raise ValueError("2D preview needs at least two dataset dimensions")
        rows, cols = int(shape[-2]), int(shape[-1])
        stride = max(1, int(math.ceil(math.sqrt(max(1, rows * cols) / max_pixels))))
        row_sel = slice(None, None, stride)
        col_sel = slice(None, None, stride)
        return (
            tuple(0 for _ in range(len(shape) - 2)) + (row_sel, col_sel),
            row_sel,
            col_sel,
        )

    @staticmethod
    def _nexus_selection_truncated(shape, selection):
        if not isinstance(selection, tuple):
            selection = (selection,)
        if len(selection) != len(shape):
            return True
        for sel, dim in zip(selection, shape):
            if isinstance(sel, slice):
                start, stop, step = sel.indices(dim)
                if start != 0 or stop != dim or step != 1:
                    return True
            elif dim != 1:
                return True
        return False

    def _nexus_axis_values(
        self,
        viewer_path,
        axis_path,
        length,
        label,
        unit,
        *,
        axis_selection=None,
    ):
        if not axis_path:
            return np.arange(length, dtype=float), label, unit
        try:
            from xrd_tools.io import read_nexus_dataset
            selection = axis_selection if axis_selection is not None else np.s_[:]
            axis = read_nexus_dataset(viewer_path, axis_path, selection=selection)
            values = np.asarray(axis.data, dtype=float).ravel()
            attrs = dict(axis.attrs)
            label = (
                attrs.get("long_name")
                or attrs.get("description")
                or attrs.get("title")
                or label
            )
            unit = attrs.get("units") or unit
        except Exception:
            logger.debug("Could not load NeXus axis %s", axis_path, exc_info=True)
            values = np.arange(length, dtype=float)
        return values, str(label), str(unit or "")

    def _nexus_summary_rows(self, summary):
        rows = []
        path = getattr(summary, "path", "")
        xdart = getattr(summary, "xdart", None)
        rows.append((
            "File summary",
            {
                "kind": "file",
                "path": path,
                "entries": ", ".join(getattr(summary, "entries", ()) or ()),
                "processed_xdart": bool(getattr(xdart, "is_processed", False)),
            },
        ))
        if xdart is not None:
            rows.extend(self._nexus_xdart_rows(xdart))
        rows.extend(self._nexus_tree_rows(getattr(summary, "tree", None)))
        return rows

    def _nexus_xdart_rows(self, xdart):
        rows = []
        if xdart.integrated_1d is not None:
            rows.append((
                "Integrated 1D",
                self._nexus_reduced_info(xdart.integrated_1d),
            ))
        if xdart.integrated_2d is not None:
            info = self._nexus_reduced_info(xdart.integrated_2d)
            info["two_d_kind"] = getattr(xdart.integrated_2d.two_d_kind, "value", None)
            rows.append(("Integrated 2D", info))
        rows.append((
            "Scan metadata",
            {
                "kind": "scan_data",
                "columns": ", ".join(xdart.scan_data_columns),
                "frame_labels": self._nexus_value(xdart.frame_labels),
            },
        ))
        if xdart.geometry_columns:
            rows.append((
                "Per-frame geometry",
                {
                    "kind": "per_frame_geometry",
                    "columns": ", ".join(xdart.geometry_columns),
                },
            ))
        if xdart.thumbnail_count or xdart.source_count:
            rows.append((
                "Frame records",
                {
                    "kind": "frames",
                    "frame_count": len(xdart.frame_labels),
                    "thumbnail_count": xdart.thumbnail_count,
                    "source_count": xdart.source_count,
                },
            ))
        if xdart.raw_image_dataset:
            shape = tuple(xdart.raw_image_shape or ())
            rank = len(shape)
            preview_kind = "image_2d" if rank >= 2 else None
            rows.append((
                "Raw detector dataset",
                {
                    "kind": "raw_dataset",
                    "dataset_path": xdart.raw_image_dataset,
                    "_shape": shape,
                    "shape": self._nexus_value(shape),
                    "dtype": xdart.raw_image_dtype or "",
                    "nexus_preview_kind": preview_kind,
                    "x_label": "column",
                    "y_label": "row",
                    "z_label": "Detector intensity",
                },
            ))
        return rows

    def _nexus_reduced_info(self, reduced):
        axes = [
            f"{axis.name} {axis.shape}"
            + (f" [{axis.units}]" if axis.units else "")
            for axis in reduced.axes
        ]
        axis_by_name = {axis.name: axis for axis in reduced.axes}
        intensity_shape = tuple(reduced.intensity_shape or ())
        info = {
            "kind": "reduced_stack",
            "path": reduced.path,
            "dataset_path": f"{reduced.path}/intensity",
            "_shape": intensity_shape,
            "frame_count": reduced.frame_count,
            "frame_labels": self._nexus_value(reduced.frame_labels),
            "intensity_shape": self._nexus_value(intensity_shape),
            "axes": "; ".join(axes),
            "y_label": "Intensity",
        }
        if len(reduced.axes) == 1:
            q_axis = reduced.axes[0]
            info.update({
                "nexus_preview_kind": "plot_1d",
                "x_axis_path": q_axis.path,
                "x_label": q_axis.name,
                "x_unit": q_axis.units or "",
            })
        elif len(reduced.axes) >= 2:
            q_axis = axis_by_name.get("q") or reduced.axes[-1]
            chi_axis = axis_by_name.get("chi") or reduced.axes[-2]
            info.update({
                "nexus_preview_kind": "image_2d",
                "x_axis_path": q_axis.path,
                "x_label": q_axis.name,
                "x_unit": q_axis.units or "",
                "y_axis_path": chi_axis.path,
                "y_label": chi_axis.name,
                "y_unit": chi_axis.units or "",
            })
        return info

    def _nexus_tree_rows(self, root, *, max_nodes=200):
        if root is None:
            return []
        rows = []

        def walk(node, depth=0):
            if len(rows) >= max_nodes:
                return
            if getattr(node, "path", "/") != "/":
                prefix = "  " * max(0, depth - 1)
                shape = getattr(node, "shape", None)
                dtype = getattr(node, "dtype", None)
                suffix = ""
                if shape is not None:
                    suffix = f" {shape}"
                    if dtype:
                        suffix += f" {dtype}"
                label = f"{prefix}{getattr(node, 'path', '')}{suffix}"
                info = {
                    "kind": getattr(node, "kind", ""),
                    "path": getattr(node, "path", ""),
                    "shape": self._nexus_value(shape),
                    "_shape": tuple(shape or ()),
                    "dtype": dtype or "",
                    "nx_class": getattr(node, "nx_class", None) or "",
                    "attrs": self._nexus_value(getattr(node, "attrs", {})),
                    "_attrs": dict(getattr(node, "attrs", {}) or {}),
                }
                if getattr(node, "kind", "") == "dataset":
                    info["dataset_path"] = getattr(node, "path", "")
                    rank = len(tuple(shape or ()))
                    if rank == 1:
                        info["nexus_preview_kind"] = "plot_1d"
                        info["x_label"] = "index"
                        attrs = dict(getattr(node, "attrs", {}) or {})
                        info["y_label"] = (
                            attrs.get("long_name")
                            or attrs.get("description")
                            or attrs.get("title")
                            or getattr(node, "name", "")
                            or "value"
                        )
                        info["y_unit"] = attrs.get("units", "")
                    elif rank >= 2:
                        info["nexus_preview_kind"] = "image_2d"
                        attrs = dict(getattr(node, "attrs", {}) or {})
                        info["x_label"] = "column"
                        info["y_label"] = "row"
                        info["z_label"] = (
                            attrs.get("long_name")
                            or attrs.get("description")
                            or attrs.get("title")
                            or getattr(node, "name", "")
                            or "value"
                        )
                        info["z_unit"] = attrs.get("units", "")
                if getattr(node, "error", None):
                    info["error"] = node.error
                rows.append((label, info))
            for child in getattr(node, "children", ()):
                walk(child, depth + 1)

        walk(root)
        return rows

    @staticmethod
    def _nexus_value(value, *, max_len=240):
        value = "" if value is None else value
        if isinstance(value, dict):
            text = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text

    def _load_image_file(self, fpath):
        """Load an image file for viewing.

        For multi-frame files (Eiger HDF5 masters, tiff stacks),
        populate listData with frame indices.  For single-frame
        files, display the image directly.
        """
        with self.data_lock:
            self.viewer_rows_1d.clear()
            self.viewer_rows_2d.clear()
            _clear_publication_store_for(self)
            _clear_raw_cache_for(self)
        self.frame_ids.clear()
        was_blocked = self.ui.listData.blockSignals(True)
        try:
            self.ui.listData.clear()
        finally:
            self.ui.listData.blockSignals(was_blocked)

        ext = os.path.splitext(fpath)[1].lower()
        nframes = 1

        # ── Processed xdart v2 scan file ─────────────────────────────
        # No raw images live inside — load each frame's raw via its stored
        # source pointer, listed by the file's actual frame labels (which may
        # be 0-based / gapped), not 1..N.  Stage 5: classification goes
        # through the headless ssrl boundary — xdart no longer opens HDF5 to
        # guess what the file is.
        self._viewer_source_info = ImageViewerController.classify(fpath)
        self._viewer_is_xdart = self._viewer_source_info.kind in (
            ImageSourceKind.PROCESSED_XDART, ImageSourceKind.THUMBNAIL_ONLY,
        )
        if self._viewer_is_xdart:
            # Use the classifier's *displayable* frame labels — the frame
            # groups that actually carry a thumbnail/source.  (The old
            # get_frames(union=True) returned the integrated frame_index
            # union, which can list labels with no frame group; loading the
            # first such label returned no image and blanked the viewer.)
            labels = [int(x) for x in self._viewer_source_info.frame_labels]
            if labels:
                self._viewer_image_path = fpath
                self._viewer_image_nframes = len(labels)
                self._populate_image_viewer_rows(labels, labels[0])
                loaded = self._load_single_frame(
                    fpath, frame_idx=labels[0], frame_id=labels[0],
                )
                if loaded is not False:
                    self.frame_ids.append(str(labels[0]))
                self._remember_displayed_frames()
                self.sigUpdate.emit()
                return
            logger.warning(
                '%s is an xdart processed file without v2 raw-frame '
                'source pointers; Image Viewer cannot treat it as a raw '
                'detector image stack.',
                os.path.basename(fpath),
            )
            self._viewer_image_path = None
            self._viewer_image_nframes = 0
            self._remember_displayed_frames()
            self.sigUpdate.emit()
            return

        if (ext in ('.h5', '.hdf5', '.nxs')
                and self._viewer_source_info.kind is ImageSourceKind.UNKNOWN):
            logger.warning(
                '%s is not a viewable image or xdart processed scan.',
                os.path.basename(fpath),
            )
            self._viewer_image_path = None
            self._viewer_image_nframes = 0
            self._remember_displayed_frames()
            self.sigUpdate.emit()
            return

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
            labels = list(range(1, nframes + 1))
            self._populate_image_viewer_rows(labels, 1)
            # Store the file path so data_changed can load individual frames
            self._viewer_image_path = fpath
            self._viewer_image_nframes = nframes
            # Load and display first frame
            loaded = self._load_single_frame(fpath, frame_idx=0, frame_id=1)
            if loaded is not False:
                self.frame_ids.append('1')
        else:
            # Single frame (tif, raw, edf): still publish a synthetic row so
            # Image Viewer has one coherent selection/display state.
            self._viewer_image_path = fpath
            self._viewer_image_nframes = 1
            self._populate_image_viewer_rows([1], 1)
            loaded = self._load_single_frame(fpath, frame_idx=0, frame_id=1)
            if loaded is not False:
                self.frame_ids.append('1')

        # Sync the live-scan boundary cache with whatever just landed in
        # listData (frame-numbers list, or empty for single-frame files).
        self._remember_displayed_frames()
        self.sigUpdate.emit()

    def _populate_image_viewer_rows(self, labels, selected_label=None):
        """Populate Image Viewer frame rows without emitting selection signals."""
        lw = self.ui.listData
        was_blocked = lw.blockSignals(True)
        try:
            lw.clear()
            for label in labels:
                lw.addItem(str(label))
            if selected_label is not None:
                try:
                    row = list(labels).index(selected_label)
                except ValueError:
                    row = -1
                if row >= 0:
                    lw.setCurrentRow(row)
        finally:
            lw.blockSignals(was_blocked)

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
        """Load a single frame from an image file into viewer_rows_2d.

        ``frame_idx`` is the 0-based offset *within the file* (passed to
        ``read_image``); ``frame_id`` is the 1-based id shown in listData
        and stored in ``frame_ids``.  The data dicts must be keyed by
        ``frame_id`` — ``data_changed``/``get_idxs`` look them up by the
        1-based id from the list, so keying by the 0-based ``frame_idx``
        (the old behaviour) left the viewer unable to find the image and
        showed a blank panel.
        """
        # Processed xdart v2 file: no raw images inside.  Resolve the raw
        # detector frame via the stored per-frame source pointer, falling
        # back to the dequantized thumbnail — through the headless ssrl
        # boundary (Stage 5).  ``frame_id`` is the frame label.  The result
        # records which source it returned, so a thumbnail is stored as a
        # thumbnail (its mask is already baked in — never re-masked).
        if getattr(self, '_viewer_is_xdart', False):
            res = ImageViewerController.load_processed_frame(fpath, frame_id)
            if res.image is None:
                logger.warning(
                    'Could not load raw frame or thumbnail %s from '
                    'processed file %s', frame_id, os.path.basename(fpath),
                )
                return False
            img_data = np.asarray(res.image, dtype=float)
            thumb_data = img_data if res.source == 'thumbnail' else None
            scan_info = {'source_file': os.path.basename(fpath)}
            with self.data_lock:
                self.viewer_rows_2d[int(frame_id)] = {
                    'map_raw': img_data,
                    'bg_raw': 0,
                    'mask': None,
                    'int_2d': None,
                    'gi_2d': {},
                    'thumbnail': thumb_data,
                }
                frame = _ViewerRow(idx=int(frame_id), scan_info=scan_info)
                self.viewer_rows_1d[int(frame_id)] = frame
                store = getattr(self, "publication_store", None)
                if store is not None:
                    view = FrameView(
                        label=int(frame_id),
                        thumbnail=thumb_data,
                        mask_baked=thumb_data is not None,
                        metadata_raw=scan_info,
                        metadata_numeric=numeric_metadata(scan_info),
                        source_path=fpath,
                        source_frame_index=frame_id,
                    )
                    store.upsert(
                        publication_from_frame_view(
                            view,
                            generation=store.generation,
                            source_identity=str(fpath),
                            raw_status=(
                                "thumbnail"
                                if thumb_data is not None else "evicted"
                            ),
                        )
                    )
            # Bound the full-res raws (Image Viewer browsing loaded ~18 MB
            # per file with no ceiling; the LRU keeps the intended ~8).
            # getattr: tests drive this on duck holders.
            _trim = getattr(self, '_remember_viewer_raw_lru', None)
            if _trim is not None:
                _trim(int(frame_id))
            logger.debug(
                'Image Viewer loaded processed frame %s from %s via %s: '
                'shape=%s finite=%d',
                frame_id, os.path.basename(fpath), res.source,
                getattr(img_data, 'shape', None),
                int(np.isfinite(img_data).sum()),
            )
            return True

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
                    return False
            else:
                logger.warning('Could not load image %s frame %d', fpath, frame_idx)
                return False

        with self.data_lock:
            scan_info = {'source_file': os.path.basename(fpath)}
            self.viewer_rows_2d[int(frame_id)] = {
                'map_raw': img_data,
                'bg_raw': 0,
                'mask': None,
                'int_2d': None,
                'gi_2d': {},
                'thumbnail': None,
            }
            # Lightweight row marker.  The raw pixels live in viewer_rows_2d and are
            # reloadable from _viewer_image_path, so never park them in
            # viewer_rows_1d/raw_ref.
            frame = _ViewerRow(idx=int(frame_id), scan_info=scan_info)
            self.viewer_rows_1d[int(frame_id)] = frame
            store = getattr(self, "publication_store", None)
            if store is not None:
                view = FrameView(
                    label=int(frame_id),
                    metadata_raw=scan_info,
                    metadata_numeric=numeric_metadata(scan_info),
                    source_path=fpath,
                    source_frame_index=frame_idx,
                )
                store.upsert(
                    publication_from_frame_view(
                        view,
                        generation=store.generation,
                        source_identity=str(fpath),
                        raw_status="evicted",
                    )
                )
        # Bound the full-res raws (Image Viewer browsing loaded ~18 MB per
        # file with no ceiling; the LRU keeps the intended ~8).
        _trim = getattr(self, '_remember_viewer_raw_lru', None)
        if _trim is not None:
            _trim(int(frame_id))
        logger.debug(
            'Image Viewer loaded frame %s from %s: shape=%s finite=%d '
            'min=%s max=%s',
            frame_id, os.path.basename(fpath), getattr(img_data, 'shape', None),
            int(np.isfinite(img_data).sum()),
            np.nanmin(img_data) if np.isfinite(img_data).any() else None,
            np.nanmax(img_data) if np.isfinite(img_data).any() else None,
        )
        return True

    def _try_raw_detectors(self, fpath):
        """Try reading a raw binary file with common detector shapes."""
        fallbacks = getattr(
            self, "_RAW_DETECTOR_FALLBACKS", H5Viewer._RAW_DETECTOR_FALLBACKS,
        )
        for name, shape in fallbacks:
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

    def set_file(self, fname, *, internal=False):
        """Changes the data file.

        args:
            fname: str, absolute path for data file
            internal: True for the app's own wiring (new_scan pointing the
                file thread at the run's output file) -- bypasses the
                user-interaction guards below.
        """
        if not internal:
            # Run guard (the uniform single-click/arrow loading made this
            # reachable mid-run): repointing/reloading the shared scan during
            # an ACTIVE run desyncs the live scan identity (live branch) or
            # reloads a half-written file (batch).  data_changed has the
            # same guard.  new_scan's own per-scan repoint passes
            # internal=True.
            if getattr(self, '_run_writing', False):
                logger.debug("set_file ignored during active run: %s", fname)
                return
            # Same-file dedupe: a fresh single click fires currentItemChanged
            # (press) AND itemClicked (release), both routed here -- without
            # this the .nxs loaded twice per click (three times on a
            # double-click).  Use Refresh to force a reload of the same file.
            if fname and fname == getattr(self.file_thread, 'fname', None):
                return
        if fname != '':
            try:
                self.cancel_pending_loads()
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
                self._ensure_file_thread_running()
                self.file_thread.queue.put("set_datafile")
                self.ui.listData.itemSelectionChanged.connect(self.data_changed)
                self.new_scan = True
            except Exception:
                logger.exception("Failed to set file: %s", fname)
                return

    def set_run_writing(self, active):
        """Single switch (driven by the run-state owner ``_enter``/``_exit_run_
        state``) telling frame selection NOT to read the ``.nxs`` while a run is
        writing it.  On rising edge, cancel any in-flight load so a stale worker
        isn't left churning against the now-active writer; on falling edge,
        re-fire the standing selection so any frame that was skipped (evicted +
        disk-load suppressed during the run) loads now that the file is idle.
        """
        active = bool(active)
        was = self._run_writing
        self._run_writing = active
        if active and not was:
            self.cancel_pending_loads()
        elif was and not active and self.ui.listData.selectedItems():
            # File is idle again — load whatever the user has selected (the
            # disk-load guard in data_changed is now open).
            self.data_changed()

    def _emit_render_update(self, requestor: str, *, generation=None,
                            labels=None, granted=True, suppressed_by=None) -> None:
        browse_debug_log(
            logger,
            "render_request",
            requestor=requestor,
            mode=_browse_debug_mode(self),
            generation=generation,
            load_generation=getattr(self, "_load_generation", None),
            selected=sequence_summary(
                labels if labels is not None else getattr(self, "frame_ids", ())),
            granted=bool(granted),
            suppressed_by=suppressed_by,
        )
        if granted:
            self.sigUpdate.emit()
            H5Viewer._drain_browse_anchor_heavy_after_render(
                self, requestor=requestor)

    def _emit_file_thread_update(self) -> None:
        self._emit_render_update("h5viewer.file_thread")

    def _emit_coalesced_update(self) -> None:
        self._emit_render_update("h5viewer.update_coalesce")

    def _flush_pending_load(self):
        """Fire the debounced disk load for the final selection of a rapid burst
        (see _load_coalesce_timer).  One load -> one blocking _teardown_load_worker
        wait, instead of one per selection event."""
        ids = self._pending_load_ids
        if ids and not getattr(self, '_run_writing', False):
            self._pending_load_ids = None
            browse_debug_log(
                logger,
                "bulk_hydration_flush",
                mode=_browse_debug_mode(self),
                labels=sequence_summary(ids),
                load_2d=bool(getattr(self, "_pending_load_2d", True)),
            )
            self.load_frames_data(ids, self._pending_load_2d)

    def _flush_pending_data_changed(self):
        if not getattr(self, "_pending_data_changed", False):
            return
        self._pending_data_changed = False
        H5Viewer._data_changed_now(self, show_all=False)

    def _clear_browse_one_shot_snapshot(self):
        self._browse_one_shot_target_labels = ()
        self._browse_one_shot_publications = {}
        self._browse_one_shot_signature = None
        self._browse_one_shot_anchor_label = None
        self._browse_anchor_heavy_after_next_render = None
        self._browse_anchor_heavy_inflight_label = None

    def _browse_anchor_heavy_key(self, label):
        try:
            label = int(label)
        except (TypeError, ValueError):
            return None
        signature = getattr(self, "_browse_one_shot_signature", None)
        active_key = getattr(self, "_browse_anchor_heavy_attempt_key", None)
        if signature is None and isinstance(active_key, tuple):
            try:
                if int(active_key[0]) == label:
                    return active_key
            except (TypeError, ValueError, IndexError):
                pass
        return (label, signature)

    def _start_browse_anchor_heavy_attempt_window(self, label, signature) -> None:
        key = H5Viewer._browse_anchor_heavy_key(self, label)
        if key is None:
            return
        # The signature may not have been assigned to the snapshot yet when the
        # data_changed path chooses its anchor.  Use the just-computed signature
        # so one bulk browse gesture gets one stable retry budget.
        key = (key[0], signature)
        if getattr(self, "_browse_anchor_heavy_attempt_key", None) == key:
            return
        self._browse_anchor_heavy_attempt_key = key
        self._browse_anchor_heavy_attempt_count = 0
        self._browse_anchor_heavy_attempt_logged = False

    def _clear_browse_anchor_heavy_attempt_window(self) -> None:
        self._browse_anchor_heavy_attempt_key = None
        self._browse_anchor_heavy_attempt_count = 0
        self._browse_anchor_heavy_attempt_logged = False

    def _claim_browse_anchor_heavy_attempt(self, label, *, requestor: str) -> bool:
        key = H5Viewer._browse_anchor_heavy_key(self, label)
        if key is None:
            return False
        if getattr(self, "_browse_anchor_heavy_attempt_key", None) != key:
            self._browse_anchor_heavy_attempt_key = key
            self._browse_anchor_heavy_attempt_count = 0
            self._browse_anchor_heavy_attempt_logged = False
        try:
            limit = int(getattr(
                self,
                "_browse_anchor_heavy_attempt_limit",
                _BROWSE_ANCHOR_HEAVY_ATTEMPT_LIMIT,
            ) or 0)
        except (TypeError, ValueError):
            limit = _BROWSE_ANCHOR_HEAVY_ATTEMPT_LIMIT
        count = int(getattr(self, "_browse_anchor_heavy_attempt_count", 0) or 0)
        if limit > 0 and count >= limit:
            if not getattr(self, "_browse_anchor_heavy_attempt_logged", False):
                browse_debug_log(
                    logger,
                    "browse_anchor_heavy_suppressed",
                    mode=_browse_debug_mode(self),
                    reason="attempt_limit",
                    requestor=requestor,
                    attempts=count,
                    limit=limit,
                    labels=sequence_summary((label,)),
                )
                self._browse_anchor_heavy_attempt_logged = True
            return False
        self._browse_anchor_heavy_attempt_count = count + 1
        browse_debug_log(
            logger,
            "browse_anchor_heavy_attempt",
            mode=_browse_debug_mode(self),
            requestor=requestor,
            attempt=count + 1,
            limit=limit,
            labels=sequence_summary((label,)),
        )
        return True

    def _prime_browse_one_shot_snapshot(self, labels, store=None):
        target = tuple(int(label) for label in labels)
        publications = {}
        if store is not None:
            try:
                for label, publication in (store.get_many(target) or {}).items():
                    view = getattr(publication, "view", None)
                    if view is not None and getattr(view, "has_1d", False):
                        publications[int(label)] = publication
            except Exception:
                logger.debug("browse one-shot resident snapshot failed",
                             exc_info=True)
        self._browse_one_shot_target_labels = target
        self._browse_one_shot_publications = publications
        return publications

    def _browse_anchor_has_2d_payload(self, label) -> bool:
        try:
            label_key = int(label)
        except (TypeError, ValueError):
            return False
        snapshot = getattr(self, "_browse_one_shot_publications", None) or {}
        publication = snapshot.get(label_key)
        if publication is None:
            store = getattr(self, "publication_store", None)
            get = getattr(store, "get", None)
            if callable(get):
                try:
                    publication = get(label_key)
                except Exception:
                    publication = None
        view = getattr(publication, "view", None)
        if view is None or not getattr(view, "has_2d", False):
            return False
        return (
            getattr(view, "raw", None) is not None
            or getattr(view, "thumbnail", None) is not None
        )

    def _queue_browse_anchor_heavy_after_render(self, *, reason: str) -> None:
        if _browse_debug_mode(self) not in ("Single", "Overlay", "Waterfall"):
            return
        label = getattr(self, "_browse_one_shot_anchor_label", None)
        if label is None:
            return
        try:
            label = int(label)
        except (TypeError, ValueError):
            return
        if H5Viewer._browse_anchor_has_2d_payload(self, label):
            H5Viewer._clear_browse_anchor_heavy_attempt_window(self)
            return
        self._browse_anchor_heavy_after_next_render = label
        browse_debug_log(
            logger,
            "browse_anchor_heavy_queued",
            mode=_browse_debug_mode(self),
            reason=reason,
            labels=sequence_summary((label,)),
        )

    def _drain_browse_anchor_heavy_after_render(self, *, requestor: str) -> None:
        label = getattr(self, "_browse_anchor_heavy_after_next_render", None)
        if label is None:
            return
        self._browse_anchor_heavy_after_next_render = None
        H5Viewer._schedule_browse_anchor_heavy_load(
            self, label, requestor=requestor)

    def _schedule_browse_anchor_heavy_load(self, label, *, requestor: str) -> None:
        try:
            label = int(label)
        except (TypeError, ValueError):
            return
        if getattr(self, "_run_writing", False):
            browse_debug_log(
                logger,
                "browse_anchor_heavy_suppressed",
                mode=_browse_debug_mode(self),
                reason="run_writing",
                labels=sequence_summary((label,)),
            )
            return
        if H5Viewer._browse_anchor_has_2d_payload(self, label):
            H5Viewer._clear_browse_anchor_heavy_attempt_window(self)
            return
        if getattr(self, "_browse_anchor_heavy_inflight_label", None) == label:
            return
        if not H5Viewer._claim_browse_anchor_heavy_attempt(
                self, label, requestor=requestor):
            return
        self._browse_anchor_heavy_inflight_label = label
        browse_debug_log(
            logger,
            "browse_anchor_heavy_scheduled",
            mode=_browse_debug_mode(self),
            requestor=requestor,
            labels=sequence_summary((label,)),
        )
        self.load_frames_data([label], True)

    def _capture_overlay_visit_intent(self) -> None:
        if getattr(self, "viewer_mode", None) in ("image", "xye", "nexus"):
            return
        if getattr(self, "_run_writing", False):
            return
        if getattr(self, "_plot_method", None) not in ("Overlay", "Waterfall"):
            return
        item = None
        try:
            item = self.ui.listData.currentItem()
        except Exception:
            item = None
        if item is None:
            try:
                selected = self.ui.listData.selectedItems()
                item = selected[-1] if selected else None
            except Exception:
                item = None
        if item is None:
            return
        try:
            label = int(item.text())
        except (TypeError, ValueError):
            return
        intents = getattr(self, "_overlay_visit_intent_labels", None)
        if not isinstance(intents, list):
            intents = []
            self._overlay_visit_intent_labels = intents
        if intents and intents[-1] == label:
            return
        intents.append(label)
        browse_debug_log(
            logger,
            "overlay_visit_intent",
            mode=_browse_debug_mode(self),
            labels=sequence_summary((label,)),
            pending_count=len(intents),
        )

    def data_changed(self, show_all=False):
        """Connected to itemSelectionChanged signal of listData.

        In viewer image mode with a multi-frame file, loads the
        selected frame on demand.  Otherwise falls through to the
        normal HDF5-based loading.
        """
        if (not show_all
                and getattr(self, "_browse_gesture_active", False)
                and _browse_one_shot_enabled(self)):
            self._browse_pending_data_changed = True
            browse_debug_log(
                logger,
                "render_request",
                requestor="h5viewer.selection_changed",
                mode=_browse_debug_mode(self),
                generation=getattr(self, "_load_generation", None),
                selected=sequence_summary(getattr(self, "frame_ids", ())),
                granted=False,
                suppressed_by="active_browse_gesture",
            )
            return

        if show_all:
            timer = getattr(self, "_selection_coalesce_timer", None)
            if timer is not None and timer.isActive():
                timer.stop()
            self._pending_data_changed = False
            H5Viewer._data_changed_now(self, show_all=True)
            return

        if getattr(self, "viewer_mode", None) not in ("image", "xye", "nexus"):
            H5Viewer._capture_overlay_visit_intent(self)
            timer = getattr(self, "_selection_coalesce_timer", None)
            if timer is not None:
                self._pending_data_changed = True
                browse_debug_log(
                    logger,
                    "render_request",
                    requestor="h5viewer.selection_changed",
                    mode=_browse_debug_mode(self),
                    selected=sequence_summary(getattr(self, "frame_ids", ())),
                    granted=False,
                    suppressed_by="selection_debounce",
                )
                timer.start()
                return

        H5Viewer._data_changed_now(self, show_all=False)

    def _data_changed_now(self, show_all=False):
        if not show_all:
            self.frame_ids.clear()
            items = self.ui.listData.selectedItems()
            if self.viewer_mode == 'xye':
                # XYE viewer stores the int key in UserRole
                self.frame_ids += sorted(
                    [str(item.data(QtCore.Qt.UserRole)) for item in items
                     if item.data(QtCore.Qt.UserRole) is not None],
                    key=_frame_label_sort_key)
            elif self.viewer_mode == 'nexus':
                # NeXus viewer rows are schema/preview records, not scan
                # frame labels; keep their stable numeric ids in UserRole.
                self.frame_ids += sorted(
                    [str(item.data(QtCore.Qt.UserRole)) for item in items
                     if item.data(QtCore.Qt.UserRole) is not None],
                    key=_frame_label_sort_key)
            else:
                self.frame_ids += sorted(
                    [str(item.text()) for item in items],
                    key=_frame_label_sort_key)
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
                loaded_ids = []
                for idx_str in idxs:
                    idx = int(idx_str)
                    if idx not in self.viewer_rows_2d:
                        loaded = self._load_single_frame(
                            viewer_path,
                            frame_idx=idx - 1,  # generic listData is 1-based
                            frame_id=idx,
                        )
                        if loaded is False:
                            continue
                    loaded_ids.append(str(idx))
                self.frame_ids[:] = loaded_ids
            # Single-frame: data already loaded by _load_image_file
            self.sigUpdate.emit()
            return

        # ── XYE viewer: data already loaded by scans_clicked ─────────
        if self.viewer_mode == 'xye':
            self.sigUpdate.emit()
            return

        # ── NeXus viewer: metadata row already loaded; refresh preview ─
        if self.viewer_mode == 'nexus':
            self._refresh_nexus_selected_preview(idxs)
            self.sigUpdate.emit()
            return

        # ── Normal mode: load from HDF5 ──────────────────────────────
        # Selected labels must be integer scan-frame ids.  During a
        # viewer<->scan mode transition listData can still hold non-integer
        # labels (e.g. xye filenames) before viewer_mode flips and the list
        # is rebuilt; treat those as "nothing to load" instead of crashing on
        # int('..._0001.xye').
        int_idxs = []
        for idx in idxs:
            try:
                int_idxs.append(int(idx))
            except (TypeError, ValueError):
                continue
        browse_debug_log(
            logger,
            "selected_set",
            trigger_source="Show All" if show_all else "debounce",
            mode=_browse_debug_mode(self),
            selected=sequence_summary(int_idxs),
        )
        if not int_idxs:
            emit_render = getattr(self, "_emit_render_update", None)
            if callable(emit_render):
                emit_render("h5viewer.no_valid_integer_selection", labels=idxs)
            else:
                self.sigUpdate.emit()
            return

        load_2d = self.update_2d

        if len(self.scan.frames.index) > 1:
            if len(int_idxs) == len(self.scan.frames.index):
                load_2d = False
        browse_bulk_one_shot = _browse_bulk_selection_enabled(
            self, int_idxs, show_all=show_all)
        browse_one_shot = browse_bulk_one_shot
        if browse_one_shot:
            load_2d = False

        signature = (tuple(int_idxs), bool(load_2d), _browse_debug_mode(self))
        if browse_bulk_one_shot:
            self._browse_one_shot_anchor_label = _current_selected_frame_label(
                self, int_idxs)
            if (not show_all
                    and signature == getattr(
                        self, "_browse_last_selection_signature", None)):
                browse_debug_log(
                    logger,
                    "render_request",
                    requestor="h5viewer.data_changed_now",
                    mode=_browse_debug_mode(self),
                    selected=sequence_summary(int_idxs),
                    granted=False,
                    suppressed_by="duplicate_selection_snapshot",
                )
                return
            self._browse_last_selection_signature = signature
            H5Viewer._start_browse_anchor_heavy_attempt_window(
                self, self._browse_one_shot_anchor_label, signature)
        else:
            self._browse_last_selection_signature = None
            if not browse_one_shot:
                H5Viewer._clear_browse_one_shot_snapshot(self)

        overlay_visit_labels = ()
        if (
            not browse_bulk_one_shot
            and not show_all
            and getattr(self, "_plot_method", None) in ("Overlay", "Waterfall")
        ):
            intents = tuple(
                getattr(self, "_overlay_visit_intent_labels", ()) or ())
            if intents:
                overlay_visit_labels = tuple(dict.fromkeys(intents))
                self._overlay_visit_intent_labels = []
                browse_one_shot = True
                load_2d = False

        read_idxs = tuple(overlay_visit_labels or int_idxs)

        keys = set()
        store = getattr(self, "publication_store", None)
        if store is not None:
            try:
                from .display_publication import publication_availability
                pub_1d, pub_2d, _raw = publication_availability(
                    store, labels=read_idxs)
                keys = set(pub_2d if load_2d else pub_1d)
            except Exception:
                logger.debug("publication availability lookup failed",
                             exc_info=True)
        idxs_memory = [i for i in read_idxs if i in keys]

        # Multi-frame combination is now done on demand by
        # get_frames_int_2d / get_frames_map_raw — no shared accumulator
        # state to maintain here. Just figure out which frames still
        # need to be loaded from disk.
        frame_ids = [i for i in read_idxs if i not in idxs_memory]
        browse_debug_log(
            logger,
            "resident_vs_missing",
            mode=_browse_debug_mode(self),
            selected_count=len(int_idxs),
            resident_1d_count=len(idxs_memory),
            missing_1d_count=len(frame_ids),
            load_2d=bool(load_2d),
            resident=sequence_summary(idxs_memory),
            missing=sequence_summary(frame_ids),
        )

        # While ANY run is writing the .nxs, reading it here contends on
        # file_lock/h5pool with the writer and, worse, drags the GUI thread into
        # load_frames_data -> _teardown_load_worker's thread.wait(2000) on every
        # writer save (each save re-fires sigUpdate -> re-fires this load) -> a
        # multi-minute beachball that only clears at run end.  Serve frame
        # selection from the in-memory caches only while a run is active
        # (``_run_writing`` is set by the task-#68 run-state owner alongside the
        # displayframe's ``_processing_active`` that gates the reader-side
        # hydration, so the two guards can't drift across live/batch/reintegrate).
        # Cached frames display instantly; evicted frames repaint when the run
        # ends (set_run_writing(False) re-fires this handler).
        if browse_one_shot:
            H5Viewer._prime_browse_one_shot_snapshot(self, read_idxs, store=store)
            self._browse_one_shot_signature = signature
        if frame_ids and not getattr(self, '_run_writing', False):
            # Debounce the disk load (see _load_coalesce_timer): coalesce a rapid
            # selection burst to ONE load for the final selection, so the blocking
            # _teardown_load_worker wait can't flood the GUI thread (beachball).
            self._pending_load_ids = frame_ids
            self._pending_load_2d = load_2d
            if browse_one_shot:
                self._browse_one_shot_pending_render = True
                self._browse_one_shot_load_generation = None
                if overlay_visit_labels:
                    self._overlay_visit_inflight_labels = tuple(read_idxs)
                browse_debug_log(
                    logger,
                    "bulk_hydration_scheduled",
                    mode=_browse_debug_mode(self),
                    labels=sequence_summary(frame_ids),
                    load_2d=bool(load_2d),
                )
                self._load_coalesce_timer.start()
                return
            self._load_coalesce_timer.start()
        elif frame_ids:
            logger.debug(
                "Run active: skipping disk load of %d evicted frame(s) %s; "
                "serving cache only until the run ends", len(frame_ids), frame_ids,
            )

        # FREEZE FIX: route the terminal (normal-mode, multi-frame) render through
        # the 100 ms debounce Coalescer instead of a direct synchronous emit — a
        # rapid shift/ctrl multi-select burst then collapses to ONE render of the
        # final selection instead of one heavy O(N) render per selection event (the
        # beachball on a long scan).  Programmatic live flushes bypass the
        # selection debounce via ``_data_changed_now``; this debounce is for
        # user-driven frame-list sweeps only.  Viewer-mode single-frame emits above
        # stay direct (not the freeze driver + want an immediate paint).
        if browse_one_shot:
            self._browse_one_shot_pending_render = False
            self._browse_one_shot_load_generation = None
            if overlay_visit_labels:
                self._overlay_hydrated_pending_append_labels = list(
                    overlay_visit_labels)
            elif browse_bulk_one_shot:
                H5Viewer._queue_browse_anchor_heavy_after_render(
                    self, reason="resident_one_shot")
        browse_debug_log(
            logger,
            "render_request",
            requestor="h5viewer.data_changed_now",
            mode=_browse_debug_mode(self),
            selected=sequence_summary(int_idxs),
            granted=False,
            suppressed_by="update_coalesce_pending",
        )
        self._update_coalesce_timer.start()

    def closeEvent(self, event):
        # Retire any in-flight load worker before teardown so the interpreter
        # doesn't GC-delete a still-running moveToThread'd QObject at shutdown.
        self.shutdown_threads()
        self._h5pool.close_all()
        super().closeEvent(event)

    def data_reset(self):
        """Resets data in memory (self.frames, self.frame_ids, self.data_..
        """
        # During a live (non-batch) wrangler run the display is driven by
        # the in-memory per-frame hand-off in static_scan_widget.update_data.
        # This slot is wired to ``sigNewFile``, which the async file-thread
        # ``set_datafile`` emits a few ms after new_scan() — clearing the
        # freshly-populated viewer_rows_1d/viewer_rows_2d/frames before the throttled
        # refresh can render them.  That is the multi-scan Eiger "plots
        # stay blank" bug.  new_scan() already does the controlled reset
        # the live path needs, so skip the wipe while a run is active.
        if self.live_run_active:
            return
        self.cancel_pending_loads()
        self._h5pool.close(self.scan.data_file)
        self.frames.clear()
        self.frame_ids.clear()
        with self.data_lock:
            self.viewer_rows_1d.clear()
            self.viewer_rows_2d.clear()
            _clear_publication_store_for(self)
            _clear_raw_cache_for(self)
        # Re-arm the raw self-heal: frame indices restart per scan, so a
        # stale negative-cache entry from the previous file suppressed
        # hydration for the SAME idx of the new one.
        df = getattr(self, 'displayframe', None)
        if df is not None:
            df._raw_resolve_failed = set()
            df._raw_full_shape = None       # new file -> possibly a new detector size
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
                self.viewer_rows_1d.clear()
                self.viewer_rows_2d.clear()
                _clear_publication_store_for(self)
                _clear_raw_cache_for(self)
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
            self._ensure_file_thread_running()
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
        data_file = getattr(self.scan, 'data_file', None)
        _df_exists = bool(data_file) and os.path.exists(os.fspath(data_file))
        if not _df_exists:
            logger.debug(
                "Skipping background frame load; data file is unavailable: %s",
                data_file,
            )
            # RL-1 diagnostic: reveal WHY load_frames_data cancels (and thus bumps
            # the generation, invalidating hydration and re-arming the render).  If
            # this fires repeatedly on a completed browsed scan, data_file is
            # missing/None -> the generation-churn driver; if it never fires, the
            # Show All loop is the store-cap (512 < selection) eviction thrash.
            browse_debug_log(
                logger,
                "load_frames_data_cancel",
                data_file=str(data_file),
                data_file_set=bool(data_file),
                exists=_df_exists,
                frame_count=len(frame_ids),
                mode=_browse_debug_mode(self),
            )
            self.cancel_pending_loads()
            return
        # N1: bump the generation counter before retiring the outgoing worker
        # so any chunk queued from the old selection is rejected immediately.
        self._load_generation += 1
        gen = self._load_generation
        browse_debug_log(
            logger,
            "generation_bump",
            cause="load_frames_data",
            generation=gen,
            mode=_browse_debug_mode(self),
            labels=sequence_summary(frame_ids),
        )
        if (getattr(self, "_browse_one_shot_pending_render", False)
                and _browse_one_shot_enabled(self)):
            self._browse_one_shot_load_generation = gen
        else:
            self._browse_one_shot_load_generation = None
        self._retire_load_worker_for_reselection()

        # Spin up the new worker.  Lives on its own QThread; both get
        # cleaned up after ``finished`` signals via deleteLater.
        worker = _LoadFramesWorker(
            data_file=data_file,
            file_lock=self.file_lock,
            gi=self.scan.gi,
            frame_ids=frame_ids,
            load_2d=load_2d,
            generation=gen,
            hydrate_raw=True,
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
        # Order matters: post the worker's deferred delete BEFORE asking the
        # thread to quit, so the deleteLater is queued ahead of the event
        # loop exit and gets processed on the worker thread.  (quit-first
        # could stop the loop before the deferred delete ran, leaving the
        # C++ worker for GC to direct-delete → crash.)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        # Drop refs after the thread is done so a later cancel() call
        # on a stale handle doesn't accidentally talk to a deleted
        # QObject.
        thread.finished.connect(self._clear_load_worker_refs)

        self._load_worker = worker
        self._load_thread = thread
        browse_debug_log(
            logger,
            "bulk_hydration_chunk_issued",
            mode=_browse_debug_mode(self),
            generation=gen,
            labels=sequence_summary(frame_ids),
            load_2d=bool(load_2d),
        )
        thread.start()

    def _absorb_chunk(self, generation, idx, frame, load_2d) -> None:
        """Slot for ``_LoadFramesWorker.chunkLoaded``.  Runs on the
        GUI thread; publishes the loaded scan frame into
        ``publication_store`` and emits ``sigUpdate`` so the display
        repaints incrementally.

        N1: drops the chunk silently when its ``generation`` no
        longer matches ``self._load_generation`` — that means a
        newer load has already been started (via a new selection)
        and the chunk belongs to a cancelled worker whose run loop
        was mid-emit when the cancel was issued.  Queued Qt
        signals don't get clobbered by ``deleteLater``; they
        arrive on the GUI thread after the cancel.  Without this
        check those stale frames would land in the publication store
        and pollute the current selection.
        """
        if generation != self._load_generation:
            logger.debug(
                "absorb_chunk dropping stale frame %s from gen=%s "
                "(current gen=%s)",
                idx, generation, self._load_generation,
            )
            browse_debug_log(
                logger,
                "bulk_hydration_chunk_completed",
                mode=_browse_debug_mode(self),
                generation=generation,
                labels=sequence_summary((idx,)),
                granted=False,
                suppressed_by="stale_load_generation",
                current_generation=getattr(self, "_load_generation", None),
            )
            return
        try:
            store = getattr(self, "publication_store", None)
            publication = publication_from_live_frame(
                frame,
                generation=(store.generation if store is not None else 0),
                include_2d=bool(load_2d),
                include_thumbnail=bool(load_2d),
                retain_raw_ref=bool(load_2d),
            )
            frame_idx = int(idx)
            has_2d_error = publication_has_2d_errors(publication)
            if load_2d and has_2d_error:
                logger.warning(
                    "Skipping frame %s 2D publication: %s",
                    idx,
                    publication_error_details(publication, "2d"),
                )
            with self.data_lock:
                if store is not None:
                    self.viewer_rows_1d.pop(frame_idx, None)
                    self.viewer_rows_2d.pop(frame_idx, None)
                    if not load_2d:
                        store.invalidate([frame_idx])
                    publication = store.upsert(publication)
                if (
                    not load_2d
                    and getattr(self, "_browse_one_shot_load_generation", None)
                    == generation
                ):
                    target = set(getattr(
                        self, "_browse_one_shot_target_labels", ()) or ())
                    if frame_idx in target:
                        snapshot = getattr(
                            self, "_browse_one_shot_publications", None)
                        if not isinstance(snapshot, dict):
                            snapshot = {}
                            self._browse_one_shot_publications = snapshot
                        snapshot[frame_idx] = publication
                elif load_2d:
                    anchor = getattr(self, "_browse_one_shot_anchor_label", None)
                    try:
                        anchor = int(anchor)
                    except (TypeError, ValueError):
                        anchor = None
                    if frame_idx == anchor:
                        snapshot = getattr(
                            self, "_browse_one_shot_publications", None)
                        if not isinstance(snapshot, dict):
                            snapshot = {}
                            self._browse_one_shot_publications = snapshot
                        snapshot[frame_idx] = publication
                        self._browse_anchor_heavy_inflight_label = None
                        if getattr(publication.view, "has_2d", False):
                            H5Viewer._clear_browse_anchor_heavy_attempt_window(
                                self)
            browse_debug_log(
                logger,
                "bulk_hydration_chunk_completed",
                mode=_browse_debug_mode(self),
                generation=generation,
                labels=sequence_summary((idx,)),
                granted=True,
                has_1d=bool(publication.view.has_1d),
                has_2d=bool(publication.view.has_2d),
            )
            # O6: coalesce display updates while a chunk burst is
            # streaming in.  Schedule (or restart) a debounced emit
            # rather than firing once per chunk.  ``_on_load_worker_finished``
            # forces a final emit so the burst's last paint is
            # guaranteed even if the timer is still pending.
            if (getattr(self, "_browse_one_shot_pending_render", False)
                    and getattr(self, "_browse_one_shot_load_generation", None) == generation):
                browse_debug_log(
                    logger,
                    "render_request",
                    requestor="h5viewer.absorb_chunk",
                    mode=_browse_debug_mode(self),
                    generation=generation,
                    selected=sequence_summary(getattr(self, "frame_ids", ())),
                    granted=False,
                    suppressed_by="browse_one_shot_wait_for_worker_finished",
                )
                return
            browse_debug_log(
                logger,
                "render_request",
                requestor="h5viewer.absorb_chunk",
                mode=_browse_debug_mode(self),
                generation=generation,
                selected=sequence_summary(getattr(self, "frame_ids", ())),
                granted=False,
                suppressed_by="update_coalesce_pending",
            )
            self._update_coalesce_timer.start()
        except (AttributeError, RuntimeError) as e:
            logger.debug("absorb_chunk skipped frame %s: %s", idx, e)

    def _remember_viewer_raw_lru(self, idx: int) -> None:
        """Retain a bounded LRU of full detector arrays in ``viewer_rows_2d``.

        D5: the LRU state lives WITH the shared ``viewer_rows_2d`` (see
        ``viewer_raw_lru.py``) so the worker-thread insert paths trim the
        same cache; this method adds the lock the helper requires
        (re-entrant — callers already inside ``data_lock`` are fine).
        """
        limit = max(1, int(getattr(self, "_raw_cache_limit",
                                   VIEWER_RAW_LIMIT)))
        with self.data_lock:
            keep = ()
            if getattr(self, "viewer_mode", None) == "image":
                keep = tuple(
                    int(label) for label in getattr(self, "frame_ids", ())
                    if str(label).lstrip("-").isdigit()
                )
            evicted = remember_viewer_raw_lru(
                self.viewer_rows_2d, idx, limit=limit, keep=keep,
            )
            if getattr(self, "viewer_mode", None) == "image":
                keep_set = set(keep)
                for stale in evicted:
                    if stale in keep_set:
                        continue
                    self.viewer_rows_2d.pop(stale, None)
                    self.viewer_rows_1d.pop(stale, None)
            else:
                viewer_rows_1d = getattr(self, "viewer_rows_1d", None)
                if viewer_rows_1d is None:
                    return
                for stale in evicted:
                    frame = viewer_rows_1d.get(stale)
                    if frame is not None and hasattr(frame, "map_raw"):
                        frame.map_raw = None
                        if hasattr(frame, "bg_raw"):
                            frame.bg_raw = None

    def _clear_raw_cache(self) -> None:
        """Reset the hydrated-raw LRU after viewer_rows_2d is cleared."""
        _clear_raw_cache_for(self)

    def _apply_frames_panel_width(self, viewer_mode) -> None:
        """Relax the Frames (``listData``) max width in NeXus viewer mode.

        The base max width is set in the .ui (h5viewerUI, 60 px — right for
        frame-index labels like "1"/"2").  NeXus rows are dataset-path /
        field labels ("Integrated 1D", "Raw detector dataset") that clip at
        60 px, so you can't tell what you're selecting.  Override the cap at
        runtime here (not in the generated UI): unbounded in NeXus mode so
        the splitter can size it, restored to the .ui default otherwise."""
        lw = getattr(self.ui, "listData", None)
        if lw is None:
            return
        if getattr(self, "_frames_panel_max_width", None) is None:
            # Remember the .ui default once so we can restore it exactly.
            self._frames_panel_max_width = lw.maximumWidth()
        if viewer_mode == "nexus":
            lw.setMaximumWidth(16777215)   # QWIDGETSIZE_MAX — splitter sizes it
        else:
            default = int(self._frames_panel_max_width)
            if 0 < default < 16777215:
                default = max(default, int(round(default * 1.5)))
            lw.setMaximumWidth(default)

    def cancel_pending_loads(self) -> None:
        """Cancel stale background frame hydration and reject queued chunks."""
        # Bump first so any chunk still queued from the outgoing worker is
        # dropped by the generation gate in _absorb_chunk.
        self._load_generation += 1
        browse_debug_log(
            logger,
            "generation_bump",
            cause="cancel_pending_loads",
            generation=self._load_generation,
            mode=_browse_debug_mode(self),
            labels=sequence_summary(getattr(self, "frame_ids", ())),
        )
        # Validity-guarded, deterministic teardown (cancel + quit + wait +
        # null) — never touch or GC-delete a half-deleted moveToThread'd
        # worker.
        self._teardown_load_worker()
        timer = getattr(self, '_update_coalesce_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
        timer = getattr(self, '_load_coalesce_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._pending_load_ids = None
        self._browse_one_shot_pending_render = False
        self._browse_one_shot_load_generation = None
        H5Viewer._clear_browse_one_shot_snapshot(self)
        self._overlay_visit_intent_labels = []
        self._overlay_visit_inflight_labels = ()
        self._overlay_hydrated_pending_append_labels = []
        self._browse_gesture_active = False
        self._browse_pending_data_changed = False
        timer = getattr(self, '_selection_coalesce_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._pending_data_changed = False

    def _retire_load_worker_for_reselection(self) -> None:
        """Cancel the old selection worker without blocking the GUI thread."""
        worker = getattr(self, '_load_worker', None)
        thread = getattr(self, '_load_thread', None)
        if worker is not None and _qt_isvalid(worker):
            try:
                worker.cancel()
            except (RuntimeError, AttributeError):
                pass
        if thread is not None and _qt_isvalid(thread):
            try:
                if thread.isRunning():
                    thread.quit()
            except (RuntimeError, AttributeError):
                pass
            try:
                thread.setParent(None)
            except (RuntimeError, AttributeError):
                pass
            _retain_orphaned_load_worker(worker, thread)
        self._load_worker = None
        self._load_thread = None

    def shutdown_threads(self) -> None:
        """Stop the persistent background threads this viewer owns so they are
        not destroyed while running on tab/app close.

        Without this the long-lived ``fileHandlerThread`` (an infinite
        queue-driven loop in ``run()``) and the async load worker keep running
        after the widget is torn down, which trips Qt's
        "QThread: Destroyed while thread is still running" abort at interpreter
        shutdown.  This is the production version of what the GUI test fixture
        already does on teardown.  Idempotent and exception-safe.
        """
        try:
            self.cancel_pending_loads()          # quit + wait the load worker
        except Exception:
            logger.debug("cancel_pending_loads on shutdown failed",
                         exc_info=True)
        ft = getattr(self, 'file_thread', None)
        if ft is not None:
            try:
                self._file_thread_shutdown = True
                # No stale file loads should run after close.  Drain queued
                # work first, then append one sentinel so the thread exits
                # after its current task (or immediately if idle).
                queue = getattr(ft, 'queue', None)
                if queue is not None:
                    while True:
                        try:
                            queue.get_nowait()
                        except Empty:
                            break
                for signal_name in (
                    "sigTaskStarted", "sigTaskDone", "sigNewFile", "sigUpdate",
                ):
                    signal = getattr(ft, signal_name, None)
                    if signal is not None:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                signal.disconnect()
                        except Exception:
                            pass
                try:
                    ft.live_run = False
                    ft.no_nxs = False
                except Exception:
                    pass
                if queue is not None:
                    queue.put(None)              # sentinel -> run() breaks
                if ft.isRunning():
                    if not ft.wait(10000):       # bounded wait
                        logger.warning(
                            "file_thread still running after shutdown wait; "
                            "keeping QThread handle until it exits")
                        _retain_orphaned_file_thread(ft)
            except Exception:
                logger.debug("file_thread shutdown failed", exc_info=True)

    def enter_viewer_mode_cleanup(self) -> None:
        """Clear scan-frame state before Image/XYE viewer data is loaded."""
        self.cancel_pending_loads()

        with self.data_lock:
            self.viewer_rows_1d.clear()
            self.viewer_rows_2d.clear()
            _clear_publication_store_for(self)
            self._clear_raw_cache()
        self.frame_ids.clear()
        self.latest_idx = None
        self.new_scan_loaded = False

        for attr in ('_viewer_image_path', '_viewer_image_nframes',
                     '_viewer_is_xdart', '_viewer_source_info'):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass

        lw = self.ui.listData
        was_blocked = lw.blockSignals(True)
        try:
            lw.clear()
        finally:
            lw.blockSignals(was_blocked)
        self._remember_displayed_frames()

        scans = self.ui.listScans
        was_blocked = scans.blockSignals(True)
        try:
            scans.clearSelection()
            scans.setCurrentRow(-1)
        finally:
            scans.blockSignals(was_blocked)

    def _teardown_load_worker(self) -> None:
        """Stop + wait for the in-flight load worker/thread, THEN drop refs.

        This is the only safe way to retire a ``moveToThread``'d worker:
        we must let the worker thread's event loop run its
        ``worker.deleteLater`` and exit before the Python reference is
        dropped, otherwise GC tries to delete the C++ QObject directly and
        the process crashes with "shared QObject was deleted directly".

        * Every access is guarded with :func:`_qt_isvalid` so we never
          touch a half-deleted worker.
        * ``thread.quit(); thread.wait(2000)`` blocks the GUI thread
          briefly (bounded, only at scan / selection boundaries) until the
          worker's loop has fully exited — processing the deferred delete.
        """
        worker = getattr(self, '_load_worker', None)
        thread = getattr(self, '_load_thread', None)
        if worker is not None and _qt_isvalid(worker):
            try:
                worker.cancel()
            except (RuntimeError, AttributeError):
                pass
        if thread is not None and _qt_isvalid(thread):
            try:
                if thread.isRunning():
                    thread.quit()
                    # Bounded wait: let the worker's event loop process the
                    # posted deleteLater and exit before we release the ref.
                    if not thread.wait(2000):
                        logger.warning(
                            "load thread did not exit within 2s; retaining "
                            "orphaned worker until it finishes")
                        try:
                            thread.setParent(None)
                        except (RuntimeError, AttributeError):
                            pass
                        _retain_orphaned_load_worker(worker, thread)
                        self._load_worker = None
                        self._load_thread = None
                        return
            except (RuntimeError, AttributeError):
                pass
        self._load_worker = None
        self._load_thread = None

    def _clear_load_worker_refs(self) -> None:
        """Drop ``_load_worker`` / ``_load_thread`` once the worker
        signals finished — but ONLY if our handle still points at
        that thread, and only by nulling (never by calling methods or
        forcing deletion).

        Self-review fix #3: queued ``thread.finished`` slot for
        worker A can arrive AFTER ``load_frames_data`` has already
        assigned worker B to ``self._load_worker``.  Identity-gate the
        clear so only the actually-finished thread's slot wins.

        Race-safety: by the time this fires, ``worker.deleteLater`` has
        already been processed by the worker's event loop (it is connected
        BEFORE ``thread.quit`` — see :meth:`load_frames_data`), so the C++
        objects are gone and nulling the Python handles can't trigger a
        direct-delete of a live QObject.  We do NOT call any method on the
        worker/thread here, and we only act on an exact identity match — a
        stale or already-deleted sender is ignored.
        """
        sender = self.sender()
        if sender is not None and sender is self._load_thread:
            one_shot = (
                getattr(self, "_browse_one_shot_pending_render", False)
                and getattr(self, "_browse_one_shot_load_generation", None)
                == getattr(self, "_load_generation", None)
            )
            self._load_worker = None
            self._load_thread = None
            # O6: force one final sigUpdate so the burst's final
            # paint always reflects the full selection — otherwise
            # the coalesce timer might still be pending when the
            # last chunk arrived but the worker has now terminated.
            if self._update_coalesce_timer.isActive():
                self._update_coalesce_timer.stop()
            if one_shot:
                overlay_labels = tuple(
                    getattr(self, "_overlay_visit_inflight_labels", ()) or ())
                if overlay_labels:
                    self._overlay_hydrated_pending_append_labels = list(
                        overlay_labels)
                    self._overlay_visit_inflight_labels = ()
                else:
                    H5Viewer._queue_browse_anchor_heavy_after_render(
                        self, reason="one_shot_worker_finished")
                self._browse_one_shot_pending_render = False
                self._browse_one_shot_load_generation = None
                if getattr(self, "_browse_gesture_active", False):
                    self._browse_pending_data_changed = True
                    browse_debug_log(
                        logger,
                        "render_request",
                        requestor="h5viewer.load_worker_finished",
                        mode=_browse_debug_mode(self),
                        generation=getattr(self, "_load_generation", None),
                        selected=sequence_summary(getattr(self, "frame_ids", ())),
                        granted=False,
                        suppressed_by="active_browse_gesture",
                    )
                    return
            else:
                self._browse_anchor_heavy_inflight_label = None
            emit_render = getattr(self, "_emit_render_update", None)
            if callable(emit_render):
                emit_render(
                    "h5viewer.load_worker_finished",
                    generation=getattr(self, "_load_generation", None),
                )
            else:
                self.sigUpdate.emit()
                H5Viewer._drain_browse_anchor_heavy_after_render(
                    self, requestor="h5viewer.load_worker_finished")

    # Removed legacy load_frame_data — all reads now go through
    # LiveFrame.load_from_nexus via load_frames_data above.
    #
    # Removed get_frames_sum / _safe_accumulate / _raw_minus_bg and the
    # add_idxs/sub_idxs/sum_int_2d/sum_map_raw machinery: combining 2D
    # data across multiple selected frames is now done on demand by
    # display_data.get_frames_int_2d / get_frames_map_raw, which iterate
    # the current selection straight from viewer_rows_2d. The old stateful
    # approach was both inconsistent with the 1D path (get_frames_int_1d)
    # and silently dead for sum_map_raw, which was never read anywhere.
