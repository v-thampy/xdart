# -*- coding: utf-8 -*-
"""
@author: thampy, walroth
"""

# Standard library imports
import logging
import os
import fnmatch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Qt imports
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

# Project imports
from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.io.metadata import read_image_metadata
from .wrangler_widget import wranglerWidget
from .spec_wrangler_thread import specThread, _get_scan_info  # noqa: F401
from .ui.specUI import Ui_Form
from ....gui_utils import NamedActionParameter
from xdart.utils import get_fname_dir, match_img_detector
from xdart.utils.session import load_session, save_session


QFileDialog = QtWidgets.QFileDialog
QDialog = QtWidgets.QDialog
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton


def_poni_file = ''
def_img_file = ''

params = [
    {'name': 'Calibration', 'type': 'group', 'children': [
        {'name': 'poni_file', 'title': 'PONI File    ', 'type': 'str', 'value': def_poni_file},
        NamedActionParameter(name='poni_file_browse', title='Browse...'),
    ], 'expanded': True},
    {'name': 'Signal', 'type': 'group', 'children': [
        {'name': 'inp_type', 'title': '', 'type': 'list',
         'values': ['Image Series', 'Image Directory', 'Single Image'], 'value': 'Image Series'},
        {'name': 'File', 'title': 'Image File   ', 'type': 'str', 'value': def_img_file},
        NamedActionParameter(name='img_file_browse', title='Browse...'),
        {'name': 'img_dir', 'title': 'Directory', 'type': 'str', 'value': '', 'visible': False},
        NamedActionParameter(name='img_dir_browse', title='Browse...', visible=False),
        {'name': 'include_subdir', 'title': 'Subdirectories', 'type': 'bool', 'value': False, 'visible': False},
        {'name': 'img_ext', 'title': 'File Type  ', 'type': 'list',
         'values': ['tif', 'raw', 'h5', 'nxs', 'mar3450'], 'value': 'tif', 'visible': False},
        {'name': 'series_average', 'title': 'Average Scan', 'type': 'bool', 'value': False, 'visible': True},
        {'name': 'meta_ext', 'title': 'Meta File', 'type': 'list',
         'values': ['None', 'txt', 'pdi', 'SPEC'], 'value': 'txt'},
        {'name': 'Filter', 'type': 'str', 'value': '', 'visible': False},
        {'name': 'write_mode', 'title': 'Write Mode  ', 'type': 'list',
         'values': ['Append', 'Overwrite'], 'value': 'Append'},
        {'name': 'mask_file', 'title': 'Mask File', 'type': 'str', 'value': ''},
        NamedActionParameter(name='mask_file_browse', title='Browse...'),
    ], 'expanded': True, 'visible': False},
    {'name': 'GI', 'title': 'Grazing Incidence', 'type': 'group', 'children': [
        {'name': 'Grazing', 'type': 'bool', 'value': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'list', 'values': ['th'],
         'value': 'th'},
        {'name': 'th_val', 'title': 'Theta', 'type': 'str', 'value': '0.1', 'visible': False},
        {'name': 'sample_orientation', 'title': 'Sample Orientation', 'type': 'int', 'value': 4,
         'limits': (1, 8), 'step': 1,
         'tip': 'EXIF sample orientation (1-8) for pyFAI FiberIntegrator'},
        {'name': 'tilt_angle', 'title': 'Tilt Angle', 'type': 'float', 'value': 0.0,
         'step': 0.1, 'tip': 'Chi offset / tilt angle in degrees for FiberIntegrator'},
        # gi_mode_1d / gi_mode_2d are controlled by the integrator panel
        # (axis1D / axis2D combos).  Kept here for session persistence only.
        {'name': 'gi_mode_1d', 'title': '1D Mode', 'type': 'list',
         'values': {'Q': 'q_total', u'Q\u1D62\u209A': 'q_ip', u'Q\u2092\u2092\u209A': 'q_oop'},
         'value': 'q_total', 'visible': False},
        {'name': 'gi_mode_2d', 'title': '2D Mode', 'type': 'list',
         'values': {u'Q\u1D62\u209A\u2013Q\u2092\u2092\u209A': 'qip_qoop', u'Q-\u03C7': 'q_chi'},
         'value': 'qip_qoop', 'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'Mask', 'title': 'Intensity Threshold', 'type': 'group', 'children': [
        {'name': 'Threshold', 'type': 'bool', 'value': False},
        {'name': 'min', 'title': 'Min', 'type': 'int', 'value': 0},
        {'name': 'max', 'title': 'Max', 'type': 'int', 'value': 0},
    ], 'expanded': False, 'visible': False},
    {'name': 'BG', 'title': 'Background', 'type': 'group', 'children': [
        {'name': 'bg_type', 'title': '', 'type': 'list',
         'values': ['None', 'Single BG File', 'Series Average', 'BG Directory'], 'value': 'None'},
        {'name': 'File', 'title': 'BG File', 'type': 'str', 'value': '', 'visible': False},
        NamedActionParameter(name='bg_file_browse', title='Browse...', visible=False),
        {'name': 'Match', 'title': 'Match Parameter', 'type': 'group', 'children': [
            {'name': 'Parameter', 'type': 'list', 'values': ['None'], 'value': 'None'},
            {'name': 'match_fname', 'title': 'Match File Root', 'type': 'bool', 'value': False},
            {'name': 'bg_dir', 'title': 'Directory', 'type': 'str', 'value': ''},
            NamedActionParameter(name='bg_dir_browse', title='Browse...'),
            {'name': 'Filter', 'type': 'str', 'value': ''},
        ], 'expanded': True, 'visible': False},
        {'name': 'Scale', 'type': 'float', 'value': 1, 'visible': False},
        {'name': 'norm_channel', 'title': 'Normalize', 'type': 'list', 'values': ['bstop'], 'value': 'bstop',
         'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'h5_dir', 'title': 'Save Path', 'type': 'str', 'value': get_fname_dir(), 'enabled': False},
    NamedActionParameter(name='h5_dir_browse', title='Browse...', visible=False),
]

ctr = 1



class specWrangler(wranglerWidget):
    """Widget for integrating data associated with spec file. Can be
    used "live", will continue to poll data folders until image data
    and corresponding spec data are available.

    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
        thread: wranglerThread or subclass, QThread for controlling
            processes
        timeout: int, how long before thread stops looking for new
            data.
        tree: pyqtgraph ParameterTree, stores and organizes parameters
        ui: Ui_Form from qtdesigner

    methods:
        stop: function to pass stop command to thread via command_queue
        enabled: Enables or disables interactivity
        set_image_dir: sets the image directory
        set_poni_file: sets the calibration poni file
        set_spec_file: sets the spec data file
        set_fname: Method to safely change file name
        setup: Syncs thread parameters prior to starting

    signals:
        finished: Connected to thread.finished signal
        sigStart: Tells tthetaWidget to start the thread and prepare
            for new data.
        sigUpdateData: int, signals a new arch has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
        showLabel: str, connected to thread showLabel signal, sets text
            in specLabel
    """
    showLabel = QtCore.Signal(str)

    def __init__(self, fname, file_lock, sphere, data_1d, data_2d, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(fname, file_lock, parent)

        # Scan Parameters
        self.poni = None
        self.scan_parameters = []
        self.counters = []
        self.motors = []
        self.command = None
        self.sphere = sphere
        self.data_1d = data_1d
        self.data_2d = data_2d

        # Setup gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.startButton.clicked.connect(self.start)
        # self.ui.startButton.clicked.connect(self.sigStart.emit)
        self.ui.stopButton.clicked.connect(self.stop)
        self.ui.processingModeCombo.currentTextChanged.connect(self._on_mode_changed)
        self.ui.liveCheckBox.stateChanged.connect(self._on_mode_changed)
        self.ui.batchCheckBox.stateChanged.connect(self._on_mode_changed)
        self.ui.processingModeCombo.currentTextChanged.connect(lambda _: self._save_to_session())
        self.ui.liveCheckBox.stateChanged.connect(lambda _: self._save_to_session())
        self.ui.batchCheckBox.stateChanged.connect(lambda _: self._save_to_session())
        self._on_mode_changed()

        self.showLabel.connect(self.ui.specLabel.setText)

        # Setup parameter tree
        self.tree = ParameterTree()
        self.tree.setMinimumWidth(150)
        self.stylize_ParameterTree()
        self.parameters = Parameter.create(
            name='spec_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        # Squeeze parameter tree columns to reduce panel width
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.resizeSection(0, 130)  # name column
        self.layout = QtWidgets.QVBoxLayout(self.ui.paramFrame)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.tree)

        # Set attributes from Parameter Tree and a couple more
        # Calibration
        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        self.get_poni_dict()

        # Signal
        self.inp_type = self.parameters.child('Signal').child('inp_type').value()
        self.img_file = self.parameters.child('Signal').child('File').value()
        self.img_dir = self.parameters.child('Signal').child('img_dir').value()
        self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
        self.img_ext = self.parameters.child('Signal').child('img_ext').value()
        self.single_img = True if self.inp_type == 'Single Image' else False
        self.file_filter = self.parameters.child('Signal').child('Filter').value()
        self.series_average = self.parameters.child('Signal').child('series_average').value()
        self.meta_ext = self.parameters.child('Signal').child('meta_ext').value()
        if self.meta_ext == 'None':
            self.meta_ext = None

        # Mask
        self.mask_file = self.parameters.child('Signal').child('mask_file').value()

        # Threshold
        self.apply_threshold = self.parameters.child('Mask').child('Threshold').value()
        self.threshold_min = self.parameters.child('Mask').child('min').value()
        self.threshold_max = self.parameters.child('Mask').child('max').value()

        # Write Mode
        self.write_mode = self.parameters.child('Signal').child('write_mode').value()

        # Background
        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        self.bg_file = self.parameters.child('BG').child('File').value()
        self.bg_dir = self.parameters.child('BG').child('Match').child('bg_dir').value()
        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        self.bg_match_fname = self.parameters.child('BG').child('Match').child('match_fname').value()
        self.bg_file_filter = self.parameters.child('BG').child('Match').child('Filter').value()
        self.bg_scale = self.parameters.child('BG').child('Scale').value()
        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()

        # Grazing Incidence
        self.gi = self.parameters.child('GI').child('Grazing').value()
        self.th_mtr = self.parameters.child('GI').child('th_motor').value()
        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        # gi_mode_1d / gi_mode_2d are driven by the integrator panel;
        # default here, actual values set from sphere.bai_*_args at thread start.
        self.gi_mode_1d = self.sphere.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')

        # HDF5 Save Path
        self.h5_dir = self.parameters.child('h5_dir').value()

        # NOTE: Integration Advanced button (self.ui.advancedButton) is wired
        # in static_scan_widget.set_wrangler() to show the integratorTree's
        # existing advancedWidget1D / advancedWidget2D dialogs directly.

        # Wire signals from parameter tree based buttons
        self.parameters.sigTreeStateChanged.connect(self.setup)
        self.parameters.sigTreeStateChanged.connect(self._save_to_session)

        self.parameters.child('Calibration').child('poni_file_browse').sigActivated.connect(
            self.set_poni_file
        )
        self.parameters.child('Calibration').child('poni_file').sigValueChanged.connect(
            self.get_poni_dict
        )
        self.parameters.child('Signal').child('inp_type').sigValueChanged.connect(
            self.set_inp_type
        )
        self.parameters.child('Signal').child('img_file_browse').sigActivated.connect(
            self.set_img_file
        )
        self.parameters.child('Signal').child('img_dir_browse').sigActivated.connect(
            self.set_img_dir
        )
        self.parameters.child('Signal').child('mask_file_browse').sigActivated.connect(
            self.set_mask_file
        )
        self.parameters.child('Signal').child('series_average').sigValueChanged.connect(
            self.set_series_average
        )
        self.parameters.child('Signal').child('meta_ext').sigValueChanged.connect(
            self.set_meta_ext
        )
        self.parameters.child('BG').child('bg_type').sigValueChanged.connect(
            self.set_bg_type
        )
        self.parameters.child('BG').child('bg_file_browse').sigActivated.connect(
            self.set_bg_file
        )
        self.parameters.child('BG').child('Match').child('bg_dir_browse').sigActivated.connect(
            self.set_bg_dir
        )
        self.parameters.child('BG').child('Match').child('Parameter').sigValueChanged.connect(
            self.set_bg_matching_par
        )
        self.parameters.child('BG').child('norm_channel').sigValueChanged.connect(
            self.set_bg_norm_channel
        )
        self.parameters.child('GI').child('th_motor').sigValueChanged.connect(
            self.set_gi_th_motor
        )
        self.parameters.child('GI').child('th_val').sigValueChanged.connect(
            self.set_gi_th_motor
        )
        self.parameters.child('h5_dir_browse').sigActivated.connect(
            self.set_h5_dir
        )

        # Setup thread
        self.thread = specThread(
            self.command_queue,
            self.sphere_args,
            self.file_lock,
            self.fname,
            self.h5_dir,
            self.scan_name,
            self.single_img,
            self.poni,
            self.inp_type,
            self.img_file,
            self.img_dir,
            self.include_subdir,
            self.img_ext,
            self.series_average,
            self.meta_ext,
            self.file_filter,
            self.mask_file,
            self.write_mode,
            self.bg_type,
            self.bg_file,
            self.bg_dir,
            self.bg_matching_par,
            self.bg_match_fname,
            self.bg_file_filter,
            self.bg_scale,
            self.bg_norm_channel,
            self.gi,
            self.th_mtr,
            self.sample_orientation,
            self.tilt_angle,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.command,
            self.sphere,
            self.data_1d,
            self.data_2d,
            live_mode=self.live_mode,
            max_cores=self.ui.maxCoresSpinBox.value(),
            parent=self,
        )

        self.thread.showLabel.connect(self.ui.specLabel.setText)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateArch.connect(self.sigUpdateArch.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

        # Enable/disable buttons initially
        self.ui.stopButton.setEnabled(False)

        self.setup()
        self._restore_from_session()

    # --- Session persistence ---

    # Flat map: (session_key, param_path_tuple, is_path, self_attr)
    # is_path=True  → only restore if the path exists on disk
    # self_attr     → attribute to sync on restore (None = handled by sigValueChanged)
    _SESSION_PARAMS = [
        ('poni_file',      ('Calibration', 'poni_file'),             True,  'poni_file'),
        ('inp_type',       ('Signal', 'inp_type'),                   False, 'inp_type'),
        ('img_file',       ('Signal', 'File'),                       True,  'img_file'),
        ('img_dir',        ('Signal', 'img_dir'),                    True,  'img_dir'),
        ('include_subdir', ('Signal', 'include_subdir'),             False, 'include_subdir'),
        ('img_ext',        ('Signal', 'img_ext'),                    False, 'img_ext'),
        ('series_average', ('Signal', 'series_average'),             False, 'series_average'),
        ('meta_ext',       ('Signal', 'meta_ext'),                   False, None),
        ('file_filter',    ('Signal', 'Filter'),                     False, 'file_filter'),
        ('write_mode',     ('Signal', 'write_mode'),                 False, 'write_mode'),
        ('mask_file',      ('Signal', 'mask_file'),                  True,  'mask_file'),
        ('bg_type',        ('BG', 'bg_type'),                        False, 'bg_type'),
        ('bg_file',        ('BG', 'File'),                           True,  'bg_file'),
        ('bg_dir',         ('BG', 'Match', 'bg_dir'),                True,  'bg_dir'),
        ('bg_match_fname', ('BG', 'Match', 'match_fname'),           False, 'bg_match_fname'),
        ('bg_file_filter', ('BG', 'Match', 'Filter'),                False, 'bg_file_filter'),
        ('bg_scale',       ('BG', 'Scale'),                          False, 'bg_scale'),
        ('gi',             ('GI', 'Grazing'),                        False, 'gi'),
        ('th_mtr',         ('GI', 'th_motor'),                       False, 'th_mtr'),
        ('sample_orientation', ('GI', 'sample_orientation'),         False, 'sample_orientation'),
        ('tilt_angle',     ('GI', 'tilt_angle'),                     False, 'tilt_angle'),
        ('gi_mode_1d',     ('GI', 'gi_mode_1d'),                    False, 'gi_mode_1d'),
        ('gi_mode_2d',     ('GI', 'gi_mode_2d'),                    False, 'gi_mode_2d'),
        ('apply_threshold', ('Mask', 'Threshold'),                   False, 'apply_threshold'),
        ('threshold_min',  ('Mask', 'min'),                          False, 'threshold_min'),
        ('threshold_max',  ('Mask', 'max'),                          False, 'threshold_max'),
        ('h5_dir',         ('h5_dir',),                              True,  'h5_dir'),
    ]

    def _save_to_session(self, *args):
        data = {}
        for key, path, _, _ in self._SESSION_PARAMS:
            try:
                p = self.parameters
                for segment in path:
                    p = p.child(segment)
                data[key] = p.value()
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug("Failed to save session parameter %s: %s", key, e)
        data['processing_mode'] = self.ui.processingModeCombo.currentText()
        data['live_mode'] = self.ui.liveCheckBox.isChecked()
        data['batch_mode'] = self.ui.batchCheckBox.isChecked()
        save_session(data)

    def _restore_from_session(self):
        session = load_session()
        for key, path, is_path, attr in self._SESSION_PARAMS:
            val = session.get(key)
            if val is None:
                continue
            if is_path and not Path(val).exists():
                continue
            try:
                p = self.parameters
                for segment in path:
                    p = p.child(segment)
                p.setValue(val)
                if attr is not None:
                    setattr(self, attr, val)
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                logger.debug("Failed to restore session parameter %s: %s", key, e)
        # Restore processing mode dropdown and checkboxes
        mode = session.get('processing_mode')
        if mode:
            idx = self.ui.processingModeCombo.findText(mode)
            if idx >= 0:
                self.ui.processingModeCombo.setCurrentIndex(idx)
        if 'live_mode' in session:
            self.ui.liveCheckBox.setChecked(session['live_mode'])
        if 'batch_mode' in session:
            self.ui.batchCheckBox.setChecked(session['batch_mode'])
        # meta_ext needs None conversion (sigValueChanged fires set_meta_ext automatically)
        # poni_file needs poni_dict loaded
        if session.get('poni_file') and Path(session['poni_file']).exists():
            self.get_poni_dict()

    # Signal to notify static_scan_widget that viewer mode changed.
    # Emits the viewer_mode string ('image', 'xye') or '' for normal.
    sigViewerModeChanged = QtCore.Signal(str)

    def _on_mode_changed(self, *args):
        """Update all flags from the processing mode dropdown and checkboxes."""
        mode_text = self.ui.processingModeCombo.currentText()
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer')
        is_xye = mode_text == 'Int 1D (XYE)'

        # Pre-process state overrides
        self.ui.liveCheckBox.blockSignals(True)
        self.ui.batchCheckBox.blockSignals(True)

        if is_viewer:
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setEnabled(False)
            self.ui.coresLabel.setEnabled(False)
            self.ui.maxCoresSpinBox.setEnabled(False)
        elif is_xye:
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setChecked(True)
            self.ui.batchCheckBox.setEnabled(False)
            self.ui.coresLabel.setEnabled(True)
            self.ui.maxCoresSpinBox.setEnabled(True)
        else:
            self.ui.liveCheckBox.setEnabled(True)
            self.ui.batchCheckBox.setEnabled(True)

            # Mutual exclusion: when one is checked, uncheck the other
            is_live = self.ui.liveCheckBox.isChecked()
            is_batch = self.ui.batchCheckBox.isChecked()
            if is_live and is_batch:
                # Live was just checked — uncheck batch (or vice versa).
                # Determine which was the trigger by checking the sender.
                # If unclear, prefer live (last action wins).
                self.ui.batchCheckBox.setChecked(False)
                is_batch = False
            if is_live:
                self.ui.batchCheckBox.setEnabled(False)
            if is_batch:
                self.ui.liveCheckBox.setEnabled(False)

            # Sync cores enabled state with batch checkbox
            self.ui.coresLabel.setEnabled(is_batch)
            self.ui.maxCoresSpinBox.setEnabled(is_batch)

        self.ui.liveCheckBox.blockSignals(False)
        self.ui.batchCheckBox.blockSignals(False)

        # Ensure cores are always visible
        self.ui.coresLabel.setVisible(True)
        self.ui.maxCoresSpinBox.setVisible(True)

        self.batch_mode = self.ui.batchCheckBox.isChecked()
        self.live_mode = self.ui.liveCheckBox.isChecked()
        self.xye_only = is_xye
        
        if mode_text == 'Image Viewer':
            self.viewer_mode = 'image'
            self.sphere.skip_2d = False
        elif mode_text == 'XYE Viewer':
            self.viewer_mode = 'xye'
            self.sphere.skip_2d = False
        else:
            self.viewer_mode = None
            self.sphere.skip_2d = '1D' in mode_text

        # Sync to thread
        self.thread.batch_mode = self.batch_mode
        self.thread.xye_only = self.xye_only
        self.thread.live_mode = self.live_mode

        # Gray out integration controls in viewer mode
        self._set_integration_controls_enabled(not is_viewer)
        # Hide start/stop in viewer mode
        self.ui.frame.setVisible(not is_viewer)
        # Notify parent only when viewer mode actually changed (avoids
        # unnecessary layout resets when just toggling Live/Batch).
        new_vm = self.viewer_mode or ''
        if not hasattr(self, '_prev_viewer_mode'):
            self._prev_viewer_mode = ''
        if new_vm != self._prev_viewer_mode:
            self._prev_viewer_mode = new_vm
            self.sigViewerModeChanged.emit(new_vm)

    def _set_integration_controls_enabled(self, enabled):
        """Enable or disable parameter tree groups related to integration."""
        for group_name in ('Calibration', 'BG', 'Mask', 'GI'):
            try:
                grp = self.parameters.child(group_name)
                grp.setOpts(enabled=enabled)
            except (AttributeError, KeyError) as e:
                logger.debug("Failed to set enabled state for %s: %s", group_name, e)
        # Also disable write mode and mask file in Signal group
        for child_name in ('write_mode', 'mask_file', 'mask_file_browse'):
            try:
                self.parameters.child('Signal').child(child_name).setOpts(enabled=enabled)
            except (AttributeError, KeyError) as e:
                logger.debug("Failed to set enabled state for Signal.%s: %s", child_name, e)
        # Disable save path
        try:
            self.parameters.child('h5_dir').setOpts(enabled=enabled)
            self.parameters.child('h5_dir_browse').setOpts(enabled=enabled)
        except (AttributeError, KeyError) as e:
            logger.debug("Failed to set enabled state for h5_dir parameters: %s", e)

    def setup(self):
        """Sets up the child thread, syncs all parameters.
        """
        # Calibration
        global ctr
        ctr += 1

        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        self.thread.poni = self.poni

        # Signal
        self.file_filter = self.parameters.child('Signal').child('Filter').value()
        self.thread.file_filter = self.file_filter

        self.inp_type = self.parameters.child('Signal').child('inp_type').value()
        self.thread.inp_type = self.inp_type

        self.get_img_fname()
        self.thread.img_file = self.img_file

        self.scan_name, _ = _get_scan_info(self.img_file)
        self.thread.scan_name = self.scan_name

        self.thread.single_img = self.single_img
        self.thread.img_dir, self.thread.img_ext = self.img_dir, self.img_ext

        self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
        self.thread.include_subdir = self.include_subdir

        self.thread.series_average = self.series_average
        self.thread.meta_ext = self.meta_ext

        self.thread.h5_dir = self.h5_dir
        self.fname = os.path.join(self.h5_dir, self.scan_name + '.nxs')
        self.thread.fname = self.fname

        self.mask_file = self.parameters.child('Signal').child('mask_file').value()
        self.thread.mask_file = self.mask_file

        # Threshold
        self.apply_threshold = self.parameters.child('Mask').child('Threshold').value()
        self.thread.apply_threshold = self.apply_threshold
        self.threshold_min = self.parameters.child('Mask').child('min').value()
        self.thread.threshold_min = self.threshold_min
        self.threshold_max = self.parameters.child('Mask').child('max').value()
        self.thread.threshold_max = self.threshold_max

        # Write Mode
        self.write_mode = self.parameters.child('Signal').child('write_mode').value()
        self.thread.write_mode = self.write_mode

        # Background
        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        self.thread.bg_type = self.bg_type

        self.bg_file = self.parameters.child('BG').child('File').value()
        self.thread.bg_file = self.bg_file

        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        self.thread.bg_matching_par = self.bg_matching_par

        self.bg_dir = self.parameters.child('BG').child('Match').child('bg_dir').value()
        self.thread.bg_dir = self.bg_dir

        self.bg_match_fname = self.parameters.child('BG').child('Match').child('match_fname').value()
        self.thread.bg_match_fname = self.bg_match_fname

        self.bg_file_filter = self.parameters.child('BG').child('Match').child('Filter').value()
        self.thread.bg_file_filter = self.bg_file_filter

        self.bg_scale = self.parameters.child('BG').child('Scale').value()
        self.thread.bg_scale = self.bg_scale

        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()
        self.thread.bg_norm_channel = self.bg_norm_channel

        # Grazing Incidence
        self.gi = self.parameters.child('GI').child('Grazing').value()
        self.thread.gi = self.gi

        # self.th_mtr = self.parameters.child('GI').child('th_motor').value()
        self.thread.th_mtr = self.th_mtr

        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.thread.sample_orientation = self.sample_orientation

        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        self.thread.tilt_angle = self.tilt_angle

        # GI modes are driven by the integrator panel (axis1D / axis2D),
        # so read them from sphere.bai_*_args which the integrator updates.
        self.gi_mode_1d = self.sphere.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
        self.thread.gi_mode_1d = self.gi_mode_1d
        self.thread.gi_mode_2d = self.gi_mode_2d

        # Notify integrator panel so labels/widgets update immediately.
        self.sigUpdateGI.emit(self.gi)

        # Processing mode flags and parallel cores
        self.thread.live_mode = self.live_mode
        self.thread.batch_mode = self.batch_mode
        self.thread.xye_only = self.xye_only
        self.thread.max_cores = self.ui.maxCoresSpinBox.value()
        self.sphere.max_cores = self.thread.max_cores  # used by sphere_threads

        self.thread.command = self.command

        self.thread.file_lock = self.file_lock
        self.thread.sphere_args = self.sphere_args

        self.thread.sphere = self.sphere
        self.thread.data_1d = self.data_1d
        self.thread.data_2d = self.data_2d

    def start(self):
        self.command = 'start'
        self.thread.command = 'start'
        self.ui.stopButton.setEnabled(True)
        self.sigStart.emit()

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        self.ui.stopButton.setEnabled(False)
        self.ui.specLabel.setText('')

    def set_poni_file(self):
        """Opens file dialogue and sets the calibration file
        """
        fname, _ = QFileDialog().getOpenFileName(
            filter="PONI (*.poni *.PONI)"
        )
        if fname != '':
            self.parameters.child('Calibration').child('poni_file').setValue(fname)
            self.poni_file = fname
            self._save_to_session()

    def get_poni_dict(self):
        """Load the PONI calibration file and store as a PONI object."""
        if not os.path.exists(self.poni_file):
            for child in self.parameters.children():
                child.hide()
            self.parameters.child('Calibration').show()
            return

        try:
            self.poni = PONI.from_poni_file(self.poni_file)
        except (IOError, OSError, ValueError, KeyError) as e:
            logger.debug("Failed to load PONI file %s: %s", self.poni_file, e)
            self.poni = None
        if self.poni is None:
            logger.warning('Invalid Poni File: %s', self.poni_file)
            self.thread.signal_q.put(('message', 'Invalid Poni File'))
            return

        for child in self.parameters.children():
            child.show()

    def set_inp_type(self):
        """Change Parameter Names depending on Input Type
        """
        self.single_img = False
        self.parameters.child('Signal').child('File').show()
        self.parameters.child('Signal').child('img_file_browse').show()
        self.parameters.child('Signal').child('img_dir').hide()
        self.parameters.child('Signal').child('img_dir_browse').hide()
        self.parameters.child('Signal').child('include_subdir').hide()
        self.parameters.child('Signal').child('Filter').hide()
        self.parameters.child('Signal').child('series_average').show()
        self.parameters.child('Signal').child('img_ext').hide()

        inp_type = self.parameters.child('Signal').child('inp_type').value()
        if inp_type == 'Image Directory':
            self.parameters.child('Signal').child('File').hide()
            self.parameters.child('Signal').child('img_file_browse').hide()
            self.parameters.child('Signal').child('img_dir').show()
            self.parameters.child('Signal').child('img_dir_browse').show()
            self.parameters.child('Signal').child('include_subdir').show()
            self.parameters.child('Signal').child('Filter').show()
            self.parameters.child('Signal').child('img_ext').show()

        if inp_type == 'Single Image':
            self.single_img = True
            self.parameters.child('Signal').child('series_average').hide()

        self.inp_type = inp_type
        self.get_img_fname()

    def set_img_file(self):
        """Opens file dialogue and sets the spec data file
        """
        fname, _ = QFileDialog().getOpenFileName(
            filter="Images (*.tiff *.tif *.h5 *.hdf5 *.nxs *.raw *.mar3450)"
        )
        if fname != '':
            self.parameters.child('Signal').child('File').setValue(fname)

    def set_img_dir(self):
        """Opens file dialogue and sets the signal data folder
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Image Directory',
            dir='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            self.parameters.child('Signal').child('img_dir').setValue(path)
        self.img_dir = path

    def get_img_fname(self):
        """Sets file name based on chosen options
        """
        old_fname = self.img_file
        if self.inp_type != 'Image Directory':
            img_file = self.parameters.child('Signal').child('File').value()
            if os.path.exists(img_file):
                self.img_file = img_file
                _p = Path(self.img_file)
                self.img_dir, self.img_ext = str(_p.parent), _p.suffix.lstrip('.')
                self._sync_meta_ext_to_img_ext()
                # Auto-detect metadata sidecar if not already set (skipped for .nxs)
                if not self.meta_ext and self.img_ext.lower() != 'nxs':
                    self.detect_meta_ext(self.img_file)

        else:
            self.img_ext = self.parameters.child('Signal').child('img_ext').value()
            self.img_dir = self.parameters.child('Signal').child('img_dir').value()
            self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
            self._sync_meta_ext_to_img_ext()

            filters = '*' + '*'.join(f for f in self.file_filter.split()) + '*'
            filters = filters if filters != '**' else '*'

            file_found = False
            for idx, (subdir, dirs, files) in enumerate(os.walk(self.img_dir)):
                for file in files:
                    fname = os.path.join(subdir, file)
                    if fnmatch.fnmatch(fname, f'{filters}.{self.img_ext}'):
                        if match_img_detector(fname, self.poni):
                            if self.meta_ext:
                                if self.exists_meta_file(fname):
                                    self.img_file = fname
                                    file_found = True
                                    break
                                else:
                                    continue
                            else:
                                self.img_file = fname
                                # Auto-detect metadata sidecar if not set
                                if not self.meta_ext:
                                    self.detect_meta_ext(fname)
                                break
                if file_found or (not self.include_subdir):
                    break

        if (((self.img_file != old_fname) or (self.img_file and (len(self.scan_parameters) < 1)))
                and self.meta_ext):
            if self.exists_meta_file(self.img_file):
                self.set_pars_from_meta()

    def set_series_average(self):
        self.series_average = self.parameters.child('Signal').child('series_average').value()

    def set_meta_ext(self):
        self.meta_ext = self.parameters.child('Signal').child('meta_ext').value()
        if self.meta_ext == 'None':
            self.meta_ext = None
        self._save_to_session()
        self.get_img_fname()

    def _sync_meta_ext_to_img_ext(self):
        """Force meta_ext='None' and lock it when the image type is NeXus.

        NeXus/.nxs files embed their own metadata (motors, counters, energy)
        inside the HDF5 tree, so no sidecar file is needed.  The parameter is
        made readonly while img_ext == 'nxs' and re-enabled otherwise.
        """
        meta_param = self.parameters.child('Signal').child('meta_ext')
        if (self.img_ext or '').lower() == 'nxs':
            if meta_param.value() != 'None':
                meta_param.setValue('None')    # fires set_meta_ext
            meta_param.setReadonly(True)
        else:
            meta_param.setReadonly(False)

    def exists_meta_file(self, img_file):
        """Checks for existence of meta file for image file"""
        if self.meta_ext != 'SPEC':
            meta_files = [
                f'{os.path.splitext(img_file)[0]}.{self.meta_ext}',
                f'{img_file}.{self.meta_ext}'
            ]
            if os.path.exists(meta_files[0]) or os.path.exists(meta_files[1]):
                return True
        else:
            spec_fname, _, _ = _extract_scan_info(Path(img_file))
            if spec_fname:
                img_fpath = Path(img_file)
                for parent in (img_fpath.parent, img_fpath.parents[1]):
                    if (parent / spec_fname).is_file():
                        return True

        return False

    def detect_meta_ext(self, img_file):
        """Auto-detect metadata sidecar format for *img_file*.

        Probes for ``.txt`` and ``.pdi`` sidecar files.  If found, updates
        the GUI parameter and ``self.meta_ext``.  Returns the detected
        extension string or ``None``.
        """
        base = os.path.splitext(img_file)[0]
        for ext in ('txt', 'pdi'):
            if os.path.exists(f'{base}.{ext}') or os.path.exists(f'{img_file}.{ext}'):
                # Update the GUI dropdown so the user sees the change
                param = self.parameters.child('Signal').child('meta_ext')
                param.setValue(ext)          # fires set_meta_ext automatically
                return ext
        return None

    def set_pars_from_meta(self):
        self.get_scan_parameters()
        self.set_bg_matching_options()
        self.set_gi_motor_options()
        self.set_bg_norm_options()

    def set_mask_file(self):
        """Opens file dialogue and sets the mask file
        """
        fname, _ = QFileDialog().getOpenFileName(
            filter="EDF (*.edf)"
        )
        if fname != '':
            self.parameters.child('Signal').child('mask_file').setValue(fname)
        self.mask_file = fname

    def set_bg_type(self):
        """Change Parameter Names depending on BG Type
        """
        for child in self.parameters.child('BG').children():
            child.hide()
        self.parameters.child('BG').child('bg_type').show()

        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        if self.bg_type == 'None':
            return
        elif self.bg_type != 'BG Directory':
            self.parameters.child('BG').child('File').show()
            if self.bg_type == 'Single BG File':
                opts = {'title': 'File Name'}
            else:
                opts = {'title': 'First File'}
            self.parameters.child('BG').child('File').setOpts(**opts)
            self.parameters.child('BG').child('bg_file_browse').show()
        else:
            self.parameters.child('BG').child('Match').show()

        self.parameters.child('BG').child('Scale').show()
        self.parameters.child('BG').child('norm_channel').show()

    def set_bg_file(self):
        """Opens file dialogue and sets the background file
        """
        fname, _ = QFileDialog().getOpenFileName(
            filter=f"Images (*.{self.img_ext})"
        )
        if fname != '':
            self.parameters.child('BG').child('File').setValue(fname)
        self.bg_file = fname

    def set_bg_dir(self):
        """Opens file dialogue and sets the background folder
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose BG Directory',
            dir='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            self.parameters.child('BG').child('Match').child('bg_dir').setValue(path)
        self.bg_dir = path

    def set_h5_dir(self):
        """Opens file dialogue and sets the path where processed data is stored
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Save Directory',
            directory='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            Path(path).mkdir(parents=True, exist_ok=True)
            self.parameters.child('h5_dir').setValue(path)
            self.h5_dir = path

    def set_bg_matching_options(self):
        """Reads image metadata to populate matching parameters
        """
        pars = [p for p in self.scan_parameters if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        pars.insert(0, 'None')
        if 'TEMP' in pars:
            pars.insert(1, pars.pop(pars.index('TEMP')))

        value = 'None'
        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('BG').child('Match').child('Parameter').setOpts(**opts)

    def set_bg_matching_par(self):
        """Changes bg matching parameter
        """
        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        if self.bg_matching_par == 'None':
            self.bg_matching_par = None

    def set_bg_norm_options(self):
        """Counter Values used to normalize and subtract background
        """
        pars = self.counters
        pars.insert(0, 'None')

        opts = {'values': pars, 'limits': pars, 'value': 'None'}
        self.parameters.child('BG').child('norm_channel').setOpts(**opts)

    def set_bg_norm_channel(self):
        """Changes bg matching parameter
        """
        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()

    def set_gi_motor_options(self):
        """Reads image metadata to populate possible GI theta motor
        """
        pars = [p for p in self.motors if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        if 'th' in pars:
            pars.insert(0, pars.pop(pars.index('th')))
            value = 'th'
        elif 'theta' in pars:
            pars.insert(0, pars.pop(pars.index('theta')))
            value = 'theta'
        else:
            value = 'Theta'

        pars = ['Manual'] + pars

        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('GI').child('th_motor').setOpts(**opts)

    def set_gi_th_motor(self):
        """Update Grazing theta motor"""
        self.th_mtr = self.parameters.child('GI').child('th_motor').value()
        self.parameters.child('GI').child('th_val').hide()
        if self.th_mtr == 'Manual':
            self.parameters.child('GI').child('th_val').show()
            self.th_mtr = self.parameters.child('GI').child('th_val').value()

    def get_scan_parameters(self):
        """ Reads image metadata to populate matching parameters
        """
        if not self.img_file:
            return

        img_meta = read_image_metadata(self.img_file, meta_format=self.meta_ext)
        self.scan_parameters = list(img_meta.keys())
        self.counters = self.scan_parameters
        self.motors = self.scan_parameters

    def enabled(self, enable):
        """Sets tree and start button to enable.

        args:
            enable: bool, True for enabled False for disabled.
        """
        self.tree.setEnabled(enable)
        self.ui.startButton.setEnabled(enable)

    def stylize_ParameterTree(self):
        self.tree.setStyleSheet("""
        QTreeView::item:has-children {
            background-color: rgb(230, 230, 230); 
            color: rgb(30, 30, 30);
        }
            """)


