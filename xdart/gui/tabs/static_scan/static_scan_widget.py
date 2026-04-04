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

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtWidgets: Any = None
    QtCore: Any = None
else:
    from pyqtgraph.Qt import QtWidgets, QtCore

# This module imports
from xdart.modules.ewald import EwaldSphere, EwaldArch
from .ui.staticUI import Ui_Form
from .h5viewer import H5Viewer
from .display_frame_widget import displayFrameWidget
from .integrator import integratorTree
from .metadata import metadataWidget
from .wranglers import specWrangler, nexusWrangler, wranglerWidget
from xdart.utils._utils import FixSizeOrderedDict, get_fname_dir, get_img_data

QWidget = QtWidgets.QWidget
QSizePolicy = QtWidgets.QSizePolicy
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
QDialog = QtWidgets.QDialog
QInputDialog = QtWidgets.QInputDialog
QCombo = QtWidgets.QComboBox

wranglers = {
    'SPEC': specWrangler,
    'NeXus': nexusWrangler,
}


def spherelocked(func):
    """Decorator that acquires sphere_lock before calling the wrapped method.

    If self.sphere is not an EwaldSphere (e.g. during initialisation),
    the function is called without the lock rather than silently returning None.
    """
    def wrapper(self, *args, **kwargs):
        if isinstance(self.sphere, EwaldSphere):
            with self.sphere.sphere_lock:
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
        arch: EwaldArch, currently loaded arch object
        arch_ids: List of EwaldArch indices currently loaded
        arches: Dictionary of currently loaded EwaldArches
        data_1d: Dictionary object holding all 1D data in memory
        data_2d: Dictionary object holding all 2D data in memory
        command_queue: Queue, used to send commands to wrangler
        dirname: str, absolute path of current directory for scan
        file_lock: mp.Condition, process safe lock
        fname: str, current data file name
        sphere: EwaldSphere, current scan data
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
        first_arch, latest_arch, next_arch: Handle moving between
            different arches in the overall sphere
        load_and_set: Combination of load and set methods. Also governs
            file explorer behavior in h5viewer.
        load_sphere:
    """

    def __init__(self, local_path=None, parent=None):
        super().__init__(parent)
        self._init_data_objects(local_path)
        self._init_ui()
        self._init_child_widgets()
        self._connect_signals()
        self._init_wranglers()
        self._init_defaults_and_timer()
        self.show()
        self.ui.wranglerFrame.activateWindow()

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self, local_path):
        """Initialize data containers, file lock, and directory paths."""
        self.file_lock = threading.Condition()
        local_path = get_fname_dir()
        self.local_path = local_path
        self.dirname = os.path.join(local_path)
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)

        self.fname = os.path.join(self.dirname, 'default.nxs')
        self.sphere = EwaldSphere('null_main',
                                  data_file=self.fname,
                                  static=True)
        self.arch = EwaldArch(static=True, gi=self.sphere.gi)
        self.arch_ids = []
        self.arches = OrderedDict()
        self.data_1d = OrderedDict()
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
                                 self.sphere, self.arch, self.arch_ids, self.arches,
                                 self.data_1d, self.data_2d,
                                 self.ui.hdf5Frame)
        self.ui.hdf5Frame.setLayout(self.h5viewer.layout)
        self.h5viewer.update_scans()

        # DisplayFrame
        self.displayframe = displayFrameWidget(self.sphere, self.arch,
                                               self.arch_ids, self.arches,
                                               self.data_1d, self.data_2d,
                                               parent=self.ui.middleFrame)
        self.ui.middleFrame.setLayout(self.displayframe.ui.layout)

        # IntegratorTree
        self.integratorTree = integratorTree(
            self.sphere, self.arch, self.file_lock,
            self.arches, self.arch_ids, self.data_1d, self.data_2d)
        self.ui.integratorFrame.setLayout(self.integratorTree.ui.verticalLayout)
        if len(self.sphere.arches.index) > 0:
            self.integratorTree.update()
        self.integratorTree.ui.raw_to_tif.hide()

        # Metadata
        self.metawidget = metadataWidget(self.sphere, self.arch,
                                         self.arch_ids, self.arches)
        self.ui.metaFrame.setLayout(self.metawidget.layout)

    def _connect_signals(self):
        """Wire signal/slot connections for H5Viewer, DisplayFrame, and Integrator."""
        # H5Viewer signals
        self.h5viewer.sigUpdate.connect(self.set_data)
        self.h5viewer.file_thread.sigTaskStarted.connect(self.thread_state_changed)
        self.h5viewer.sigThreadFinished.connect(self.thread_state_changed)
        self.h5viewer.ui.listData.itemClicked.connect(self.disable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.enable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.latest_arch)

        # DisplayFrame signals
        self.displayframe.ui.update2D.stateChanged.connect(self.update_h5_options)
        self.h5viewer.actionSaveImage.triggered.connect(self.displayframe.save_image)
        self.h5viewer.actionSaveArray.triggered.connect(self.displayframe.save_1D)

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
                    self.sphere, self.data_1d, self.data_2d,
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
        # self.wrangler.sigUpdateArch.connect(self.new_arch)
        self.wrangler.sigUpdateGI.connect(self.update_scattering_geometry)
        self.wrangler.started.connect(self.thread_state_changed)
        self.wrangler.finished.connect(self.wrangler_finished)
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'processingModeCombo'):
            def _on_mode_changed(mode_text):
                # Skip when in viewer mode — set_viewer_display_mode controls panels
                if 'Viewer' in mode_text:
                    return
                self.displayframe._apply_1d_only_visibility()
                if '2D' in mode_text:
                    self.displayframe.update()
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
        """Called by signal from wrangler when a new arch is processed.

        Instead of rendering immediately (which blocks the main thread
        and causes frame-skipping when the wrangler is faster than the
        GUI), we update the in-memory data structures and schedule a
        coalesced display refresh via a short single-shot timer.  If
        another sigUpdateData arrives before the timer fires, only the
        latest index is rendered.

        Special case: ``idx == -1`` is the batch-complete signal —
        trigger a full display refresh without touching the h5 viewer.
        """
        if idx == -1:
            # Batch mode finished — just refresh the display
            self._pending_update_idx = -1
            self._update_timer.start()
            return

        self.h5viewer.file_thread.queue.put("update_sphere")

        with self.file_lock:
            self.h5viewer.latest_idx = idx
            if self.h5viewer.auto_last:
                self.latest_arch()
            self.h5viewer.update_data()

        # Record the latest index and (re)start the coalescing timer.
        # If the timer is already running, restart resets the countdown
        # so only one render happens after the burst settles.
        self._pending_update_idx = idx
        self._update_timer.start()

    def _flush_pending_update(self):
        """Render the most recently received wrangler update.

        Called by _update_timer after a short quiet period (200 ms).
        This ensures the display always shows the latest data without
        burning CPU on intermediate frames the user never sees.
        """
        if self._pending_update_idx is None:
            return
        self._pending_update_idx = None
        self.displayframe.update()
        self.metawidget.update()

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
        # In viewer mode, always update display (no sphere dependency)
        is_viewer = getattr(self.h5viewer, 'viewer_mode', None) is not None
        if is_viewer or self.sphere.name != 'null_main':
            self.displayframe.update()
            # # if (len(self.arches.keys()) > 0) and (len(self.sphere.arches.index) > 0):
            # if ((len(self.data_1d.keys()) > 0) and
            #         (len(self.arch_ids) > 0) and
            #         (self.arch_ids[0] != 'No data') and
            #         (len(self.sphere.arches.index) > 0)):

            if not is_viewer:
                if len(self.arch_ids) == 0:
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
        del self.sphere
        del self.displayframe.sphere
        del self.arch
        del self.displayframe.arch
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
        """Updates all data in displays
        TODO: Currently taking the most time for the main gui thread
        """
        if idx is not None:
            self.h5viewer.latest_idx = idx
            
        self.h5viewer.update_data()
        if self.h5viewer.auto_last:
            self.latest_arch()

        self.displayframe.update()
        self.metawidget.update()

        gc.collect()

    def integrator_thread_update(self, idx):
        # self.thread_state_changed()
        if idx is not None:
            self.h5viewer.latest_idx = idx

        self.h5viewer.set_open_enabled(True)
        self.h5viewer.update_data()
        
        if self.h5viewer.auto_last:
            self.latest_arch()
            
        self.displayframe.update()

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

    def new_scan(self, name, fname, gi, th_mtr, single_img, series_average):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
        """
        # if self.sphere.name != name or self.sphere.name == 'null_main':
        self.h5viewer.dirname = os.path.dirname(fname)
        self.h5viewer.set_file(fname)
        self.sphere.gi = gi
        self.sphere.th_mtr = th_mtr
        self.sphere.single_img = single_img
        self.sphere.series_average = series_average

        self.integratorTree.get_args('bai_1d')
        self.integratorTree.get_args('bai_2d')
        self.integratorTree.set_image_units()

        # Clear data objects
        self.data_1d.clear()
        self.data_2d.clear()
        self.arches.clear()
        self.arch_ids.clear()

        self.displayframe.set_axes()
        # self.displayframe.auto_last = True

        self.h5viewer.scan_name = name
        self.h5viewer.auto_last = True
        self.h5viewer.latest_idx = 1
        self.h5viewer.update_scans()
        self.h5viewer.update()

    def update_scattering_geometry(self, gi):
        """Connected to sigUpdateGI from wrangler. Called when scattering
        geometry changes between transmission and GI

        args:
            gi: bool, flag for determining if in Grazing incidence
        """
        self.sphere.gi = gi
        self.integratorTree.set_image_units()
        self.displayframe.set_axes()

    def new_arch(self, arch_data):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
        """
        arch = EwaldArch(idx=arch_data['idx'], map_raw=arch_data['map_raw'],
                         mask=arch_data['mask'], scan_info=arch_data['scan_info'],
                         poni_file=arch_data['poni_file'], static=self.sphere.static, gi=self.sphere.gi)
        arch.int_1d = arch_data['int_1d']
        arch.int_2d = arch_data['int_2d']
        arch.map_norm = arch_data['map_norm']
        # self.data_2d[str(arch.idx)] = arch

    def start_wrangler(self):
        """Sets up wrangler, ensures properly synced args, and starts
        the wrangler.thread main method.
        """
        # i_qChi = np.zeros((1000, 1000), dtype=float)

        self.wrangler.enabled(False)

        self.integratorTree.get_args('bai_1d')
        self.integratorTree.get_args('bai_2d')

        args = {'bai_1d_args': self.sphere.bai_1d_args,
                'bai_2d_args': self.sphere.bai_2d_args}
        self.wrangler.sphere_args = copy.deepcopy(args)
        self.wrangler.setup()
        self.h5viewer.auto_last = True
        self.wrangler.thread.start()

    def wrangler_finished(self):
        """Called by the wrangler finished signal. If current scan
        matches the wrangler scan, allows for integration.
        """
        # Flush any pending coalesced update so the final frame is shown.
        self._update_timer.stop()
        self._flush_pending_update()

        self.thread_state_changed()
        self.wrangler.stop()
        
        # Auto-load the final file generated from the batch if applicable
        is_batch = getattr(self.wrangler.thread, 'batch_mode', False)
        is_xye_only = getattr(self.wrangler.thread, 'xye_only', False)

        if is_batch and not is_xye_only:
            generated_file = getattr(self.wrangler, 'fname', None)
            if generated_file and os.path.exists(generated_file):
                # Update directory display to point at the generated folder natively 
                generated_dir = os.path.dirname(generated_file)
                if self.h5viewer.dirname != generated_dir:
                    self.h5viewer.dirname = generated_dir
                    self.h5viewer.update_scans()
                # Inform H5Viewer to load the file and set the flag to auto-select its last point
                self.h5viewer._auto_select_last_on_finish = True
                self.h5viewer.set_file(generated_file)

        if self.sphere.name == self.wrangler.scan_name:
            self.integrator_thread_finished()
        else:
            self.wrangler.enabled(True)

        gc.collect()

    def update_h5_options(self, state):
        """Changes H5Widget Option to update only 1D or both views
        """
        self.h5viewer.update_2d = state

    def _on_viewer_mode_changed(self, viewer_mode_str):
        """Enable or disable the integrator panel and update h5viewer for viewer mode.

        Args:
            viewer_mode_str: 'image', 'xye', or '' (normal mode)
        """
        viewer_mode = viewer_mode_str or None  # '' → None
        is_viewer = viewer_mode is not None
        # Keep integratorTree enabled so mask/threshold controls remain accessible
        self.h5viewer.viewer_mode = viewer_mode
        # Give displayframe a reference to the wrangler for mask/threshold
        self.displayframe._wrangler = self.wrangler if is_viewer else None
        # In viewer mode, disable New/Save (keep Open Folder and Export)
        self.h5viewer.actionNewFile.setEnabled(not is_viewer)
        self.h5viewer.actionSaveDataAs.setEnabled(not is_viewer)
        # XYE viewer: allow multi-select for overlay; others: single select
        from PySide6.QtWidgets import QAbstractItemView
        if viewer_mode == 'xye':
            self.h5viewer.ui.listScans.setSelectionMode(
                QAbstractItemView.ExtendedSelection)
        else:
            self.h5viewer.ui.listScans.setSelectionMode(
                QAbstractItemView.SingleSelection)
        # Configure display panels for the viewer mode
        self.displayframe.set_viewer_display_mode(viewer_mode)
        # Refresh scan list to show/hide appropriate file types
        self.h5viewer.update_scans()

    def latest_arch(self):
        """Advances to last arch in data list, updates displayframe, and
        set auto_last to True
        """
        self.h5viewer.auto_last = True
        if self.h5viewer.ui.listData.count() <= 1:
            return

        idx = self.h5viewer.latest_idx
        if isinstance(idx, int):
            self.h5viewer.latest_idx = idx

            items = self.h5viewer.ui.listData.findItems(str(idx), QtCore.Qt.MatchExactly)
            if len(items):
                for item in items:
                    self.h5viewer.ui.listData.setCurrentItem(item)
        else:
            last_row = self.h5viewer.ui.listData.count() - 1
            if last_row >= 0:
                item = self.h5viewer.ui.listData.item(last_row)
                if item is not None:
                    try:
                        self.h5viewer.latest_idx = int(item.text())
                    except ValueError:
                        self.h5viewer.latest_idx = item.text()
                self.h5viewer.ui.listData.setCurrentRow(last_row)

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
