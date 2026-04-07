# -*- coding: utf-8 -*-
"""
@author: walroth
"""
# Standard library imports
import logging
import os
import time
import gc

logger = logging.getLogger(__name__)

# This module imports
import re
import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.io.export import read_xye
from ssrl_xrd_tools.io.image import read_image, count_frames
from xdart.utils.session import load_session, save_session
from .ui.h5viewerUI import Ui_Form
from xdart.modules.ewald import EwaldArch
from .sphere_threads import fileHandlerThread
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


class H5Viewer(QWidget):
    """Widget for displaying the contents of an EwaldSphere object and
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
                 sphere, arch, arch_ids, arches,
                 data_1d, data_2d,
                 parent=None, data_lock=None):
        super().__init__(parent)
        import threading as _threading
        self.data_lock = data_lock if data_lock is not None else _threading.RLock()
        self._init_data_objects(file_lock, local_path, dirname,
                                sphere, arch, arch_ids, arches,
                                data_1d, data_2d)
        self._init_ui()
        self._init_toolbar()
        self._connect_signals()
        self._init_file_thread()

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self, file_lock, local_path, dirname,
                           sphere, arch, arch_ids, arches,
                           data_1d, data_2d):
        """Initialize data references and state flags."""
        self.local_path = local_path
        self.file_lock = file_lock
        self.dirname = dirname
        self.sphere = sphere
        self.arch = arch
        self.arch_ids = arch_ids
        self.arches = arches
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.new_scan = True
        self.update_2d = True
        self.auto_last = True
        self.latest_idx = None
        self.new_scan_loaded = False
        self.viewer_mode = None

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
        self.ui.show_all.clicked.connect(self.show_all)
        self.actionOpenFolder.triggered.connect(self.open_folder)
        self.actionSaveDataAs.triggered.connect(self.save_data_as)
        self.actionNewFile.triggered.connect(self.new_file)

    def _init_file_thread(self):
        """Create and start the background file handler thread."""
        self.file_thread = fileHandlerThread(self.sphere, self.arch,
                                             self.file_lock,
                                             arch_ids=self.arch_ids,
                                             arches=self.arches,
                                             data_1d=self.data_1d,
                                             data_2d=self.data_2d,
                                             data_lock=self.data_lock)
        self.file_thread.sigTaskDone.connect(self.thread_finished)
        self.file_thread.sigNewFile.connect(self.sigNewFile.emit)
        self.file_thread.sigUpdate.connect(self.sigUpdate.emit)
        self.file_thread.start(Qt.QtCore.QThread.LowPriority)
        self._h5pool = get_pool()
        
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
    
    def update_data(self):
        """Updates list with all arch ids.
        """
        if self.sphere.name == "null_main":
            return

        # with self.sphere.sphere_lock:
        _idxs = [str(i) for i in list(self.sphere.arches.index)]

        if len(_idxs) == 0:
            self.ui.listData.clear()
            # self.ui.listData.addItem('No Data')
            return

        lw = self.ui.listData
        items = [lw.item(x).text() for x in range(lw.count())]
        eq = _idxs == items

        if (len(_idxs) > 1) and (_idxs == items):
            if self.new_scan_loaded:
                self.new_scan_loaded = False
                self.ui.listData.setCurrentRow(-1)
                self.arch_ids = []
                return
            if self.auto_last and isinstance(self.latest_idx, int) and str(self.latest_idx) in _idxs:
                self.ui.listData.blockSignals(True)
                matched = self.ui.listData.findItems(str(self.latest_idx), QtCore.Qt.MatchExactly)
                for item in matched:
                    self.ui.listData.setCurrentItem(item)
                self.ui.listData.blockSignals(False)
                self.data_changed()
            return
        if (len(_idxs) > 1) and (len(_idxs) == len(items)):
            if self.auto_last and isinstance(self.latest_idx, int):
                self.ui.listData.blockSignals(True)
                matched = self.ui.listData.findItems(str(self.latest_idx), QtCore.Qt.MatchExactly)
                for item in matched:
                    self.ui.listData.setCurrentItem(item)
                self.ui.listData.blockSignals(False)
                self.data_changed()
            return

        previous_loc = self.ui.listData.currentRow()
        previous_sel = [item.text() for item in self.ui.listData.selectedItems()]

        # Block signals while rebuilding the list to prevent spurious
        # itemSelectionChanged → data_changed → sigUpdate cascades.
        self.ui.listData.blockSignals(True)

        self.ui.listData.clear()
        self.ui.listData.insertItems(0, _idxs)

        if self.new_scan_loaded:
            self.new_scan_loaded = False
            self.ui.listData.setCurrentRow(-1)
            self.arch_ids.clear()
            self.ui.listData.blockSignals(False)
            return

        if self.auto_last and isinstance(self.latest_idx, int) and (str(self.latest_idx) in _idxs):
            items = self.ui.listData.findItems(str(self.latest_idx), QtCore.Qt.MatchExactly)
            if len(items):
                for item in items:
                    self.ui.listData.setCurrentItem(item)
            self.ui.listData.blockSignals(False)
            self.data_changed()
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
        self.data_changed()

    def show_all(self):

        if len(self.sphere.arches.index) > 0:
            self.arch_ids.clear()
            self.arch_ids += self.sphere.arches.index

        self.new_scan = False
        self.data_changed(show_all=True)

    def thread_finished(self, task):
        if task != "load_arch":
            self.update()
            if getattr(self, '_auto_select_last_on_finish', False):
                self._auto_select_last_on_finish = False
                if self.ui.listData.count() > 0:
                    self.ui.listData.setCurrentRow(self.ui.listData.count() - 1)
        self.sigThreadFinished.emit()

        gc.collect()
    
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

        Normal mode: navigates folders or loads sphere data from HDF5.
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
        self.arch_ids.clear()

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
            arch = EwaldArch(idx=idx, static=True, gi=False)
            arch.int_1d = int_1d
            arch.scan_info = {'source_file': os.path.basename(fpath)}

            with self.data_lock:
                self.data_1d[idx] = arch
            self.arch_ids.append(str(idx))
            idx += 1

        if len(self.data_1d) == 0:
            return

        # Populate listData with loaded filenames (all selected).
        # Display filename but store numeric index in UserRole so
        # data_changed can map back to data_1d keys.
        self.ui.listData.blockSignals(True)
        self.ui.listData.clear()
        for key in self.data_1d:
            arch = self.data_1d[key]
            fname = arch.scan_info.get('source_file', f'file_{key}')
            display_name = os.path.basename(fname)
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, key)
            self.ui.listData.addItem(item)
        self.ui.listData.selectAll()
        self.ui.listData.blockSignals(False)

        self.sigUpdate.emit()

    def _load_image_file(self, fpath):
        """Load an image file for viewing.

        For multi-frame files (Eiger HDF5 masters, tiff stacks),
        populate listData with frame indices.  For single-frame
        files, display the image directly.
        """
        with self.data_lock:
            self.data_1d.clear()
            self.data_2d.clear()
        self.arch_ids.clear()
        self.ui.listData.clear()

        ext = os.path.splitext(fpath)[1].lower()
        nframes = 1

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
            self._load_single_frame(fpath, frame_idx=0, arch_idx=1)
            self.arch_ids.append('1')
            self.ui.listData.setCurrentRow(0)
        else:
            # Single frame (tif, raw, edf): load directly, leave listData blank
            self._viewer_image_path = None
            self._load_single_frame(fpath, frame_idx=0, arch_idx=1)
            self.arch_ids.append('1')

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

    def _load_single_frame(self, fpath, frame_idx=0, arch_idx=1):
        """Load a single frame from an image file into data_2d."""
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
            self.data_2d[int(arch_idx)] = {
                'map_raw': img_data,
                'bg_raw': np.zeros_like(img_data),
                'mask': None,
                'int_2d': None,
                'gi_2d': {},
                'thumbnail': None,
            }
            # Minimal data_1d entry so display doesn't crash
            arch = EwaldArch(idx=arch_idx, static=True, gi=False)
            arch.scan_info = {'source_file': os.path.basename(fpath)}
            self.data_1d[int(arch_idx)] = arch
    
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
            self.arch_ids.clear()
            items = self.ui.listData.selectedItems()
            if self.viewer_mode == 'xye':
                # XYE viewer stores the int key in UserRole
                self.arch_ids += sorted(
                    [str(item.data(QtCore.Qt.UserRole)) for item in items
                     if item.data(QtCore.Qt.UserRole) is not None])
            else:
                self.arch_ids += sorted([str(item.text()) for item in items])
            idxs = self.arch_ids
        else:
            idxs = self.arch_ids

        if (len(idxs) == 0) or ('No data' in idxs):
            time.sleep(0.1)
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
                            arch_idx=idx,
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

        if len(self.sphere.arches.index) > 1:
            if len(idxs) == len(self.sphere.arches.index):
                load_2d = False

        if load_2d:
            idxs_memory = [int(idx) for idx in idxs if int(idx) in self.data_2d.keys()]
        else:
            idxs_memory = [int(idx) for idx in idxs if int(idx) in self.data_1d.keys()]

        # Multi-arch combination is now done on demand by
        # get_arches_int_2d / get_arches_map_raw — no shared accumulator
        # state to maintain here. Just figure out which arches still
        # need to be loaded from disk.
        arch_ids = [int(idx) for idx in idxs
                    if int(idx) not in idxs_memory]

        if len(arch_ids) > 0:
            self.load_arches_data(arch_ids, load_2d)

        self.sigUpdate.emit()

    gc.collect()

    def closeEvent(self, event):
        self._h5pool.close_all()
        super().closeEvent(event)

    def data_reset(self):
        """Resets data in memory (self.arches, self.arch_ids, self.data_..
        """
        self._h5pool.close(self.sphere.data_file)
        self.arches.clear()
        self.arch_ids.clear()
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
            self.arches.clear()
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

    def load_arches_data(self, arch_ids, load_2d):
        """Loads per-frame data from NeXus HDF5 (``entry/frames/``).

        args:
            arch_ids: list of frame indices to load
            load_2d: bool — whether to load 2D data and thumbnails
        """
        file = self._h5pool.get(self.sphere.data_file)
        if file is None:
            # File is being written to by the wrangler thread — skip this read.
            return

        if "entry" not in file or "frames" not in file["entry"]:
            return
        frames_grp = file["entry/frames"]

        for idx in arch_ids:
            try:
                arch = EwaldArch(idx=idx, static=True, gi=self.sphere.gi)
                arch.load_from_nexus(frames_grp, load_2d=load_2d)

                with self.data_lock:
                    if not load_2d:
                        self.data_1d[int(idx)] = arch.copy(include_2d=False)
                    else:
                        self.data_1d[int(idx)] = arch.copy(include_2d=False)
                        self.data_2d[int(idx)] = {'map_raw': arch.map_raw,
                                                  'bg_raw': arch.bg_raw,
                                                  'mask': arch.mask,
                                                  'int_2d': arch.int_2d,
                                                  'gi_2d': arch.gi_2d,
                                                  'thumbnail': arch.thumbnail}

            except KeyError:
                pass

    # Removed legacy load_arch_data — all reads now go through
    # EwaldArch.load_from_nexus via load_arches_data above.
    #
    # Removed get_arches_sum / _safe_accumulate / _raw_minus_bg and the
    # add_idxs/sub_idxs/sum_int_2d/sum_map_raw machinery: combining 2D
    # data across multiple selected arches is now done on demand by
    # display_data.get_arches_int_2d / get_arches_map_raw, which iterate
    # the current selection straight from data_2d. The old stateful
    # approach was both inconsistent with the 1D path (get_arches_int_1d)
    # and silently dead for sum_map_raw, which was never read anywhere.
