# -*- coding: utf-8 -*-
"""
@author: thampy, walroth
"""

# Standard library imports
import os
import re
import time
import glob
import fnmatch
import numpy as np
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# pyFAI imports
import fabio

# Qt imports
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

# This module imports
from xdart.modules.ewald import EwaldArch, EwaldSphere
from xdart.modules.ewald.arch import _make_integrator_from_poni_dict
from ssrl_xrd_tools.integrate.gid import create_fiber_integrator
from .wrangler_widget import wranglerWidget, wranglerThread, wranglerProcess
from .ui.specUI import Ui_Form
from ....gui_utils import NamedActionParameter
from xdart.utils import get_img_data
from ssrl_xrd_tools.io.metadata import read_image_metadata
from xdart.utils import split_file_name, get_scan_name, get_img_number, get_fname_dir, get_sname_img_number
from xdart.utils import match_img_detector, get_series_avg, get_specFile, get_mask_array
from xdart.utils import write_xye, write_csv
from xdart.utils.session import load_session, save_session
from xdart.utils.containers.poni import get_poni_dict
from xdart.utils.h5pool import get_pool as _get_h5pool
# from xdart.utils import natural_sort_ints


from icecream import ic; ic.configureOutput(prefix='', includeContext=True)

QFileDialog = QtWidgets.QFileDialog
QDialog = QtWidgets.QDialog
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton

def_poni_file = '' # '/Users/vthampy/SSRL_Data/RDA/static_det_test_data/test_xfc_data/test_xfc.poni'
def_img_file = '' # '/Users/vthampy/SSRL_Data/RDA/static_det_test_data/test_xfc_data/images/images_0005.tif'

if not os.path.exists(def_poni_file):
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
         'values': ['tif', 'raw', 'h5', 'mar3450'], 'value': 'tif', 'visible': False},
        {'name': 'series_average', 'title': 'Average Scan', 'type': 'bool', 'value': False, 'visible': True},
        {'name': 'meta_ext', 'title': 'Meta File', 'type': 'list',
         'values': ['None', 'txt', 'pdi', 'SPEC'], 'value': 'txt'},
        {'name': 'Filter', 'type': 'str', 'value': '', 'visible': False},
        {'name': 'write_mode', 'title': 'Write Mode  ', 'type': 'list',
         'values': ['Append', 'Overwrite'], 'value': 'Append'},
        {'name': 'mask_file', 'title': 'Mask File', 'type': 'str', 'value': ''},
        NamedActionParameter(name='mask_file_browse', title='Browse...'),
    ], 'expanded': True, 'visible': False},
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
    {'name': 'GI', 'title': 'Grazing Incidence', 'type': 'group', 'children': [
        {'name': 'Grazing', 'type': 'bool', 'value': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'list', 'values': ['th'], 'value': 'th'},
        {'name': 'th_val', 'title': 'Theta', 'type': 'str', 'value': '0.1', 'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'h5_dir', 'title': 'Save Path', 'type': 'str', 'value': get_fname_dir(), 'enabled': False},
    NamedActionParameter(name='h5_dir_browse', title='Browse...', visible=False),
]

ctr = 1


def _integrate_frame(img_data, bg_raw, poni_dict, bai_1d_args, bai_2d_args,
                     mask, gi, th_mtr, skip_2d, img_number, scan_info,
                     series_average):
    """Worker function for parallel integration. Runs in a subprocess.

    Each worker process builds its own pyFAI CSR lookup table on the first
    frame it processes; subsequent frames in the same worker reuse the cache.
    Returns the integrated EwaldArch (picklable via __getstate__/__setstate__).
    """
    from xdart.modules.ewald.arch import EwaldArch
    arch = EwaldArch(img_number, img_data, poni_dict=poni_dict,
                     scan_info=scan_info, static=True, gi=gi,
                     th_mtr=th_mtr, bg_raw=bg_raw,
                     series_average=series_average)
    arch.integrate_1d(global_mask=mask, **bai_1d_args)
    if not skip_2d:
        arch.integrate_2d(global_mask=mask, **bai_2d_args)
    return arch


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
        cont, pause, stop: functions to pass continue, pause, and stop
            commands to thread via command_queue
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
    showLabel = Qt.QtCore.Signal(str)

    def __init__(self, fname, file_lock, sphere, data_1d, data_2d, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(fname, file_lock, parent)

        # Scan Parameters
        self.poni_dict = None
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
        self.ui.pauseButton.clicked.connect(self.pause)
        self.ui.stopButton.clicked.connect(self.stop)
        self.ui.continueButton.clicked.connect(self.cont)
        self.ui.skip2dCheckBox.stateChanged.connect(
            lambda _: setattr(self.sphere, 'skip_2d', self.ui.skip2dCheckBox.isChecked())
        )
        self.live_mode = False
        self.ui.liveCheckBox.stateChanged.connect(
            lambda _: self._set_live_mode(self.ui.liveCheckBox.isChecked())
        )
        self.ui.maxCoresSpinBox.valueChanged.connect(
            lambda v: setattr(self.thread, 'max_cores', v)
        )

        self.showLabel.connect(self.ui.specLabel.setText)

        # Setup parameter tree
        self.tree = ParameterTree()
        self.stylize_ParameterTree()
        self.parameters = Parameter.create(
            name='spec_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        self.layout = Qt.QtWidgets.QVBoxLayout(self.ui.paramFrame)
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

        # HDF5 Save Path
        self.h5_dir = self.parameters.child('h5_dir').value()

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
            self.poni_dict,
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
        self.ui.pauseButton.setEnabled(False)
        self.ui.continueButton.setEnabled(False)
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
            except Exception:
                pass
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
            except Exception:
                pass
        # meta_ext needs None conversion (sigValueChanged fires set_meta_ext automatically)
        # poni_file needs poni_dict loaded
        if session.get('poni_file') and Path(session['poni_file']).exists():
            self.get_poni_dict()

    def _set_live_mode(self, enabled):
        """Toggle live directory-watching mode."""
        self.live_mode = enabled
        self.thread.live_mode = enabled

    def setup(self):
        """Sets up the child thread, syncs all parameters.
        """
        # Calibration
        global ctr
        ctr += 1
        # ic(ctr)

        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        self.thread.poni_dict = self.poni_dict

        # Signal
        self.file_filter = self.parameters.child('Signal').child('Filter').value()
        self.thread.file_filter = self.file_filter

        self.inp_type = self.parameters.child('Signal').child('inp_type').value()
        self.thread.inp_type = self.inp_type

        self.get_img_fname()
        self.thread.img_file = self.img_file

        self.scan_name = get_scan_name(self.img_file)
        self.thread.scan_name = self.scan_name

        self.thread.single_img = self.single_img
        self.thread.img_dir, self.thread.img_ext = self.img_dir, self.img_ext

        self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
        self.thread.include_subdir = self.include_subdir

        self.thread.series_average = self.series_average
        self.thread.meta_ext = self.meta_ext

        self.thread.h5_dir = self.h5_dir
        self.fname = os.path.join(self.h5_dir, self.scan_name + '.hdf5')
        self.thread.fname = self.fname

        self.mask_file = self.parameters.child('Signal').child('mask_file').value()
        self.thread.mask_file = self.mask_file

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
        # ic(self.th_mtr)

        # Live mode and parallel cores
        self.thread.live_mode = self.live_mode
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
        self.ui.pauseButton.setEnabled(True)
        self.ui.continueButton.setEnabled(False)
        self.ui.stopButton.setEnabled(True)
        self.sigStart.emit()

    def pause(self):
        self.command = 'pause'
        self.thread.command = 'pause'
        # if self.thread.isRunning():
        self.ui.pauseButton.setEnabled(False)
        self.ui.continueButton.setEnabled(True)

    def cont(self):
        self.command = 'continue'
        self.thread.command = 'continue'
        # if self.thread.isRunning():
        self.ui.pauseButton.setEnabled(True)
        self.ui.continueButton.setEnabled(False)

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        # if self.thread.isRunning():
        self.ui.pauseButton.setEnabled(False)
        self.ui.continueButton.setEnabled(False)
        self.ui.stopButton.setEnabled(False)

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
        """Opens file dialogue and sets the calibration file
        """
        if not os.path.exists(self.poni_file):
            for child in self.parameters.children():
                child.hide()
            self.parameters.child('Calibration').show()
            return

        self.poni_dict = get_poni_dict(self.poni_file)
        if self.poni_dict is None:
            print('Invalid Poni File')
            self.thread.signal_q.put(('message', 'Invalid Poni File'))
            return

        for child in self.parameters.children():
            child.show()

    def set_inp_type(self):
        """Change Parameter Names depending on Input Type
        """
        # ic()
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
        # ic()
        fname, _ = QFileDialog().getOpenFileName(
            filter="Images (*.tiff *.tif *.h5 *.raw *.mar3450)"
        )
        if fname != '':
            self.parameters.child('Signal').child('File').setValue(fname)

    def set_img_dir(self):
        """Opens file dialogue and sets the signal data folder
        """
        # ic()
        path = QFileDialog().getExistingDirectory(
            caption='Choose Image Directory',
            directory='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            self.parameters.child('Signal').child('img_dir').setValue(path)
        self.img_dir = path

    def get_img_fname(self):
        """Sets file name based on chosen options
        """
        # ic()
        old_fname = self.img_file
        if self.inp_type != 'Image Directory':
            img_file = self.parameters.child('Signal').child('File').value()
            if os.path.exists(img_file):
                self.img_file = img_file
                self.img_dir, _, self.img_ext = split_file_name(self.img_file)
                # self.meta_ext = self.get_meta_ext(self.img_file)

        else:
            self.img_ext = self.parameters.child('Signal').child('img_ext').value()
            self.img_dir = self.parameters.child('Signal').child('img_dir').value()
            self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()

            filters = '*' + '*'.join(f for f in self.file_filter.split()) + '*'
            filters = filters if filters != '**' else '*'

            file_found = False
            # ic(file_found, self.img_file)
            for idx, (subdir, dirs, files) in enumerate(os.walk(self.img_dir)):
                for file in files:
                    fname = os.path.join(subdir, file)
                    if fnmatch.fnmatch(fname, f'{filters}.{self.img_ext}'):
                        # ic(fname, self.img_file, self.poni_dict)
                        if match_img_detector(fname, self.poni_dict):
                            # ic(self.img_file, self.meta_ext)
                            if self.meta_ext:
                                if self.exists_meta_file(fname):
                                    self.img_file = fname
                                    file_found = True
                                    break
                                else:
                                    continue
                            else:
                                self.img_file = fname
                                break
                            # self.meta_ext = self.get_meta_ext(fname)
                # ic(self.img_file, file_found, self.include_subdir, idx)
                if file_found or (not self.include_subdir):
                    # ic('breaking')
                    break

        # ic(self.img_file, self.scan_parameters)
        if (((self.img_file != old_fname) or (self.img_file and (len(self.scan_parameters) < 1)))
                and self.meta_ext):
            if self.exists_meta_file(self.img_file):
                self.set_pars_from_meta()

    def set_series_average(self):
        self.series_average = self.parameters.child('Signal').child('series_average').value()

    def set_meta_ext(self):
        # ic()
        self.meta_ext = self.parameters.child('Signal').child('meta_ext').value()
        if self.meta_ext == 'None':
            self.meta_ext = None
        self._save_to_session()
        self.get_img_fname()

    def exists_meta_file(self, img_file):
        """Checks for existence of meta file for image file"""
        # ic()
        if self.meta_ext != 'SPEC':
            meta_files = [
                f'{os.path.splitext(img_file)[0]}.{self.meta_ext}',
                f'{img_file}.{self.meta_ext}'
            ]
            if os.path.exists(meta_files[0]) or os.path.exists(meta_files[1]):
                # ic('returning True \n')
                return True
        else:
            meta_file = get_specFile(img_file)
            if meta_file and os.path.exists(meta_file):
                return True

        return False

    def set_pars_from_meta(self):
        # ic()
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
        # ic()
        fname, _ = QFileDialog().getOpenFileName(
            filter=f"Images (*.{self.img_ext})"
        )
        if fname != '':
            self.parameters.child('BG').child('File').setValue(fname)
        self.bg_file = fname

    def set_bg_dir(self):
        """Opens file dialogue and sets the background folder
        """
        # ic()
        path = QFileDialog().getExistingDirectory(
            caption='Choose BG Directory',
            directory='',
            options=QFileDialog.ShowDirsOnly
            # options =(QFileDialog.ShowDirsOnly)
        )
        if path != '':
            self.parameters.child('BG').child('Match').child('bg_dir').setValue(path)
        self.bg_dir = path

    def set_h5_dir(self):
        """Opens file dialogue and sets the path where processed data is stored
        """
        # ic()
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
        # ic(self.scan_parameters)
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
        # ic()
        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        if self.bg_matching_par == 'None':
            self.bg_matching_par = None

    def set_bg_norm_options(self):
        """Counter Values used to normalize and subtract background
        """
        # ic()
        pars = self.counters
        # ic(self.counters)
        pars.insert(0, 'None')

        opts = {'values': pars, 'limits': pars, 'value': 'None'}
        self.parameters.child('BG').child('norm_channel').setOpts(**opts)

    def set_bg_norm_channel(self):
        """Changes bg matching parameter
        """
        # ic()
        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()

    def set_gi_motor_options(self):
        """Reads image metadata to populate possible GI theta motor
        """
        # ic()
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
        # ic(pars)

        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('GI').child('th_motor').setOpts(**opts)

    def set_gi_th_motor(self):
        """Update Grazing theta motor"""
        self.th_mtr = self.parameters.child('GI').child('th_motor').value()
        self.parameters.child('GI').child('th_val').hide()
        if self.th_mtr == 'Manual':
            self.parameters.child('GI').child('th_val').show()
            self.th_mtr = self.parameters.child('GI').child('th_val').value()
            # ic(self.th_mtr)

    def get_scan_parameters(self):
        """ Reads image metadata to populate matching parameters
        """
        # ic(self.img_file)
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
        # ic()
        self.tree.setEnabled(enable)
        self.ui.startButton.setEnabled(enable)

    def stylize_ParameterTree(self):
        self.tree.setStyleSheet("""
        QTreeView::item:has-children {
            background-color: rgb(230, 230, 230); 
            color: rgb(30, 30, 30);
        }
            """)


class specThread(wranglerThread):
    """Thread for controlling the specProcessor process. Receives
    manages a command and signal queue to pass commands from the main
    thread and communicate back relevant signals

    attributes:
        command_q: mp.Queue, queue to send commands to process
        file_lock: mp.Condition, process safe lock for file access
        scan_name: str, name of current scan
        fname: str, full path to data file.
        h5_dir: str, data file directory.
        img_file: str, path to image file
        img_dir: str, path to image directory
        img_ext : str, extension of image file
        series_average : bool, flag to average over series
        meta_ext : str, extension of metadata file
        poni_dict: str, Poni File name
        detector: str, Detector name
        input_q: mp.Queue, queue for commands sent from parent
        signal_q: mp.Queue, queue for commands sent from process
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
        timeout: float or int, how long to continue checking for new
            data.
        command: command passed to stop, pause, continue etc.
        data_1d/2d: Dictionaries to store processed data for plotting

    signals:
        showLabel: str, sends out text to be used in specLabel

    methods:
        run: Main method, called by start
    """
    showLabel = Qt.QtCore.Signal(str)

    def __init__(
            self,
            command_queue,
            sphere_args,
            file_lock,
            fname,
            h5_dir,
            scan_name,
            single_img,
            poni_dict,
            inp_type,
            img_file,
            img_dir,
            include_subdir,
            img_ext,
            series_average,
            meta_ext,
            file_filter,
            mask_file,
            write_mode,
            bg_type,
            bg_file,
            bg_dir,
            bg_matching_par,
            bg_match_fname,
            bg_file_filter,
            bg_scale,
            bg_norm_channel,
            gi,
            th_mtr,
            command,
            sphere,
            data_1d,
            data_2d,
            live_mode=False,
            max_cores=1,
            parent=None):

        """command_queue: mp.Queue, queue for commands sent from parent
        sphere_args: dict, used as **kwargs in sphere initialization.
            see EwaldSphere.
        fname: str, path to data file.
        h5_dir: str, data file directory.
        file_lock: mp.Condition, process safe lock for file access
        scan_name: str, name of current scan
        single_img: bool, True if there is only one image
        poni_dict: str, poni file name
        detector: str, Detector name
        img_file: str, path to input image file
        img_dir: str, path to image directory
        include_subdir: bool, flag to include subdirectories
        img_ext : str, extension of image file
        series_average : bool, flag to average over series
        meta_ext : str, extension of metadata file
        timeout: float or int, how long to continue checking for new
            data.
        command: command passed to stop, pause, continue etc.
        gi: bool, grazing incidence flag to determine if pyGIX is to be used
        th_mtr: float, incidence angle
        """
        super().__init__(command_queue, sphere_args, fname, file_lock, parent)

        self.h5_dir = h5_dir
        self.scan_name = scan_name
        self.single_img = single_img
        self.poni_dict = poni_dict
        self.inp_type = inp_type
        self.img_file = img_file
        self.img_dir = img_dir
        self.include_subdir = include_subdir
        self.img_ext = img_ext
        self.series_average = series_average
        self.meta_ext = meta_ext
        self.file_filter = file_filter
        self.mask_file = mask_file
        self.write_mode = write_mode
        self.bg_type = bg_type
        self.bg_file = bg_file
        self.bg_dir = bg_dir
        self.bg_matching_par = bg_matching_par
        self.bg_match_fname = bg_match_fname
        self.bg_file_filter = bg_file_filter
        self.bg_scale = bg_scale
        self.bg_norm_channel = bg_norm_channel
        self.gi = gi
        self.th_mtr = th_mtr
        self.live_mode = live_mode
        self.max_cores = max_cores
        self.command = command
        self.sphere = sphere
        self.data_1d = data_1d
        self.data_2d = data_2d

        self.user = None
        self.mask = None
        self.detector = None
        self.img_fnames = []
        self.processed = []
        self.processed_scans = []
        self.sub_label = ''

    def run(self):
        """Initializes specProcess and watches for new commands from
        parent or signals from the process.
        """
        t0 = time.time()
        if (self.poni_dict == '') or (self.img_file == ''):
            return

        self.img_fnames.clear()
        self.processed.clear()
        self.processed_scans.clear()
        self.detector = self.poni_dict['detector']
        self.sub_label = ''
        # self.get_mask()
        self.mask = get_mask_array(self.detector, self.mask_file)
        self._cached_gi_incident_angle = None

        self.process_scan()
        print(f'Total Time: {time.time() - t0:0.2f}')

    def process_scan(self):
        """Batch-integrate all existing images, then optionally watch for new ones (live mode).

        Phase 1 — Collect: drain the current directory glob into a pending list.
        Phase 2 — Process: parallel (ProcessPoolExecutor) when max_cores > 1, else sequential.
        Phase 3 — Watch (live mode only): poll every 2 s for new files; process each immediately.
        """
        sphere = None
        files_processed = 0
        _cached_poni_dict = None

        # ── Phase 1 & 2: collect then process all existing images ─────────────
        pending = []  # [(img_file, img_number, img_data, img_meta, bg_raw)]

        while True:
            if self.command == 'stop':
                break
            if self.command == 'pause':
                self.showLabel.emit('Paused')
                time.sleep(0.5)
                continue

            img_file, scan_name, img_number, img_data, img_meta = self.get_next_image()
            if img_data is None:
                break  # initial glob exhausted — move on to processing
            if img_file is not None:
                fname = os.path.splitext(os.path.basename(img_file))[0]
                self.showLabel.emit(f'Collecting {fname[-30:]}')
            else:
                print(f'Invalid image file, skipping')
                continue

            img_number = 1 if img_number is None else img_number
            self.scan_name = scan_name

            # Flush and switch sphere when scan name changes
            if (sphere is None) or (scan_name != sphere.name):
                if pending:
                    files_processed += self._dispatch_batch(sphere, pending)
                    pending = []
                sphere = self.initialize_sphere()
                _cached_poni_dict = None

            # Rebuild cached AzimuthalIntegrator when poni_dict identity changes
            if self.poni_dict is not _cached_poni_dict:
                sphere._cached_integrator = _make_integrator_from_poni_dict(self.poni_dict)
                sphere._cached_fiber_integrator = None
                _cached_poni_dict = self.poni_dict
                self._cached_gi_incident_angle = None

            if img_number in list(sphere.arches.index):
                if self.single_img:
                    self.sigUpdate.emit(img_number)
                    break
                continue

            bg_raw = self.get_background(img_file, img_number, img_meta)
            pending.append((img_file, img_number, img_data, img_meta, bg_raw))

            if self.single_img:
                break

        # Process whatever is left
        if pending and sphere is not None and self.command != 'stop':
            files_processed += self._dispatch_batch(sphere, pending)

        # ── Phase 3: live watching ────────────────────────────────────────────
        if self.live_mode and self.command != 'stop' and sphere is not None:
            self.showLabel.emit('Watching for new files...')
            while self.command != 'stop':
                if self.command == 'pause':
                    self.showLabel.emit('Paused')
                    time.sleep(0.5)
                    continue

                img_file, scan_name, img_number, img_data, img_meta = self.get_next_image()
                if img_data is None:
                    # Nothing new yet — show watching status and sleep
                    self.showLabel.emit('Watching for new files...')
                    time.sleep(2.0)
                    continue

                img_number = 1 if img_number is None else img_number
                self.scan_name = scan_name

                if (sphere is None) or (scan_name != sphere.name):
                    sphere = self.initialize_sphere()
                    _cached_poni_dict = None

                if self.poni_dict is not _cached_poni_dict:
                    sphere._cached_integrator = _make_integrator_from_poni_dict(self.poni_dict)
                    sphere._cached_fiber_integrator = None
                    _cached_poni_dict = self.poni_dict
                    self._cached_gi_incident_angle = None

                if img_number in list(sphere.arches.index):
                    continue

                bg_raw = self.get_background(img_file, img_number, img_meta)
                # Process immediately — single-threaded for low latency
                self._process_one(sphere, img_file, img_number, img_data, img_meta, bg_raw)
                files_processed += 1

        print(f'\nTotal Files Processed: {files_processed}')

    # ── Batch dispatch helpers ────────────────────────────────────────────────

    def _dispatch_batch(self, sphere, pending):
        """Route a batch to parallel or sequential processing based on max_cores."""
        max_cores = getattr(self, 'max_cores', 1)
        n_workers = min(max_cores, len(pending))
        if n_workers > 1:
            return self._process_batch_parallel(sphere, pending, n_workers)
        else:
            count = 0
            for item in pending:
                if self.command == 'stop':
                    break
                self._process_one(sphere, *item)
                count += 1
            return count

    def _process_one(self, sphere, img_file, img_number, img_data, img_meta, bg_raw):
        """Integrate one image sequentially and save. Includes timing instrumentation."""
        fname = os.path.splitext(os.path.basename(img_file))[0]
        self.showLabel.emit(f'Processing {fname[-30:]}')

        _t1 = time.time()
        arch = EwaldArch(
            img_number, img_data, poni_dict=self.poni_dict,
            scan_info=img_meta, static=True, gi=self.gi,
            th_mtr=self.th_mtr, bg_raw=bg_raw,
            series_average=self.series_average,
            integrator=sphere._cached_integrator,
        )
        _t_arch = time.time() - _t1

        if self.gi:
            _incident_angle = arch._get_incident_angle()
            if (sphere._cached_fiber_integrator is None
                    or _incident_angle != self._cached_gi_incident_angle):
                sphere._cached_fiber_integrator = create_fiber_integrator(
                    arch._make_lib_poni(),
                    incident_angle=_incident_angle,
                    tilt_angle=arch.tilt_angle,
                    angle_unit="deg",
                )
                self._cached_gi_incident_angle = _incident_angle

        _t2 = time.time()
        arch.integrate_1d(
            global_mask=self.mask,
            fiber_integrator=sphere._cached_fiber_integrator,
            **sphere.bai_1d_args,
        )
        _t_1d = time.time() - _t2

        _t3 = time.time()
        if not sphere.skip_2d:
            arch.integrate_2d(
                global_mask=self.mask,
                fiber_integrator=sphere._cached_fiber_integrator,
                **sphere.bai_2d_args,
            )
        _t_2d = time.time() - _t3

        arch.skip_map_raw = sphere.skip_2d
        _get_h5pool().close(sphere.data_file)
        _t4 = time.time()
        with self.file_lock:
            sphere.add_arch(
                arch=arch, calculate=False, update=True,
                get_sd=True, set_mg=False, static=True, gi=self.gi,
                th_mtr=self.th_mtr, series_average=self.series_average
            )
            sphere.save_to_h5(data_only=True, replace=False)
        _t_h5 = time.time() - _t4

        _t5 = time.time()
        self.save_1d(sphere, arch, img_number)
        _t_csv = time.time() - _t5

        _t_total = _t_arch + _t_1d + _t_2d + _t_h5 + _t_csv
        print(
            f'[TIMING] image_{img_number:04d}: '
            f'arch_init={_t_arch:.2f}s int_1d={_t_1d:.2f}s int_2d={_t_2d:.2f}s '
            f'h5_write={_t_h5:.2f}s csv={_t_csv:.2f}s total={_t_total:.2f}s'
        )
        print(f'Processed {fname} {self.sub_label}')
        self.sigUpdate.emit(img_number)

    def _process_batch_parallel(self, sphere, pending, n_workers):
        """Integrate pending images in parallel subprocesses; write H5 serially."""
        future_to_meta = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for (img_file, img_number, img_data, img_meta, bg_raw) in pending:
                future = executor.submit(
                    _integrate_frame,
                    img_data, bg_raw, self.poni_dict,
                    sphere.bai_1d_args, sphere.bai_2d_args,
                    self.mask, self.gi, self.th_mtr,
                    sphere.skip_2d, img_number, img_meta,
                    self.series_average,
                )
                future_to_meta[future] = (img_file, img_number)

            count = 0
            for future in as_completed(future_to_meta):
                if self.command == 'stop':
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                img_file, img_number = future_to_meta[future]
                fname = os.path.splitext(os.path.basename(img_file))[0]
                try:
                    arch = future.result()
                except Exception as exc:
                    print(f'Error integrating image_{img_number:04d}: {exc}')
                    continue

                arch.skip_map_raw = sphere.skip_2d
                _get_h5pool().close(sphere.data_file)
                with self.file_lock:
                    sphere.add_arch(
                        arch=arch, calculate=False, update=True,
                        get_sd=True, set_mg=False, static=True, gi=self.gi,
                        th_mtr=self.th_mtr, series_average=self.series_average
                    )
                    sphere.save_to_h5(data_only=True, replace=False)

                self.save_1d(sphere, arch, img_number)
                print(f'Processed (parallel) {fname} {self.sub_label}')
                if len(fname) > 40:
                    fname = f'{fname[:8]}....{fname[-30:]}'
                self.showLabel.emit(f'{fname}')
                self.sigUpdate.emit(img_number)
                count += 1

        return count

    def get_next_image(self):
        """ Gets next image in image series or in directory to process

        Returns:
            image_name {str}: image file path
            image_number {int}: image file number (if part of series)
            image_data {np.ndarray}: image file data array
        """
        if self.single_img:
            img_data = get_img_data(self.img_file, self.detector, return_float=True)
            meta = read_image_metadata(self.img_file, meta_format=self.meta_ext) if self.meta_ext else {}
            # return self.img_file, get_img_number(self.img_file), img_data
            scan_name, img_number = get_sname_img_number(self.img_file)
            return self.img_file, scan_name, img_number, img_data, meta

        if len(self.img_fnames) == 0:
            if self.inp_type != 'Image Directory':
                first_img = self.img_file
                self.img_fnames = Path(self.img_dir).glob(f'{self.scan_name}_*.{self.img_ext}')
            else:
                first_img = ''
                filters = '*' + '*'.join(f for f in self.file_filter.split()) + '*'
                filters = filters if filters != '**' else '*'
                if self.include_subdir:
                    self.img_fnames = Path(self.img_dir).rglob(f'{filters}.{self.img_ext}')
                else:
                    self.img_fnames = Path(self.img_dir).glob(f'{filters}.{self.img_ext}')

            self.img_fnames = [str(f) for f in self.img_fnames if
                               (str(f) >= first_img) and (str(f) not in self.processed)]

            # self.img_fnames = deque(sorted(self.img_fnames))
            self.img_fnames = deque(natural_sort_ints(self.img_fnames))

        img_file, scan_name, img_number, img_data, img_meta = None, None, 1, None, {}
        n = 0
        while len(self.img_fnames) > 0:
            fname = self.img_fnames[0]
            sname, snumber = get_sname_img_number(fname)

            if (n > 0) and (scan_name != sname):
                break

            self.processed.append(fname)
            self.img_fnames.popleft()

            data = get_img_data(fname, self.detector, return_float=True)
            if data is None:
                continue

            meta = read_image_metadata(fname, meta_format=self.meta_ext) if self.meta_ext else {}
            n += 1

            if (not self.series_average) or (snumber is None):
                return fname, sname, snumber, data, meta
            else:
                if n == 1:
                    img_data = data
                    img_meta = meta
                else:
                    # img_data = img_data*(n-1)/n + data/n
                    img_data += data
                    for (k, v) in meta.items():
                        try:
                            img_meta[k] = float(img_meta[k]) + float(meta[k])
                        except TypeError:
                            pass

                scan_name, img_file = sname, fname
                # ic(sname, scan_name, n)
                # if len(self.img_fnames) == 0:
                #     return img_file, scan_name, 1, img_data, meta

        if n > 1:
            img_data /= n
            for (k, v) in img_meta.items():
                try:
                    img_meta[k] /= n
                except TypeError:
                    pass

        return img_file, scan_name, img_number, img_data, img_meta
        # return None, None, None, None, None

    def get_meta_data(self, img_file):
        return read_image_metadata(img_file, meta_format=self.meta_ext)

    def subtract_bg(self, img_data, img_file, img_number, img_meta):
        bg = self.get_background(img_file, img_number, img_meta)
        try:
            img_data -= bg
            # min_int = img_data.min()
            # if min_int < 0:
            #     img_data -= min_int
        except ValueError:
            pass

    def initialize_sphere(self):
        """ If scan changes, initialize new EwaldSphere object
        If mode is overwrite, replace existing HDF5 file, else append to it
        """
        fname = os.path.join(self.h5_dir, self.scan_name + '.hdf5')
        sphere = EwaldSphere(self.scan_name,
                             data_file=fname,
                             static=True,
                             gi=self.gi,
                             th_mtr=self.th_mtr,
                             series_average=self.series_average,
                             single_img=self.single_img,
                             global_mask=self.mask,
                             **self.sphere_args)
        sphere.skip_2d = self.sphere.skip_2d

        write_mode = self.write_mode
        if not os.path.exists(fname):
            write_mode = 'Overwrite'

        _get_h5pool().close(sphere.data_file)
        with self.file_lock:
            if write_mode == 'Append':
                sphere.load_from_h5(replace=False, mode='a')
                sphere.skip_2d = self.sphere.skip_2d  # re-apply after load may overwrite it
                for (k, v) in self.sphere_args.items():
                    setattr(sphere, k, v)
                existing_arches = sphere.arches.index
                if len(existing_arches) == 0:
                    sphere.save_to_h5(replace=True)
            else:
                sphere.save_to_h5(replace=True)

        self.sigUpdateFile.emit(
            self.scan_name, fname,
            self.gi, self.th_mtr, self.single_img,
            self.series_average
        )
        print(f'\n***** New Scan *****')

        return sphere

    def get_mask(self):
        """Get mask array from mask file
        """
        self.mask = self.detector.calc_mask()
        if self.mask_file and os.path.exists(self.mask_file):
            if self.mask is not None:
                try:
                    self.mask += fabio.open(self.mask_file).data
                except ValueError:
                    print('Mask file not valid for Detector')
                    pass
            else:
                self.mask = fabio.open(self.mask_file).data

        if self.mask is None:
            return None

        if self.mask.shape != self.detector.shape:
            print('Mask file not valid for Detector')
            return None

        self.mask = np.flatnonzero(self.mask)

    def get_background(self, img_file, img_number, img_meta):
        """Subtract background image if bg_file or bg_dir specified
        """
        if self.bg_type == 'None':
            return 0

        bg, bg_file, bg_meta, norm_factor = 0, None, None, 1
        self.sub_label, norm_label, bg_scale_label = '', '', ''

        if self.bg_type == 'Single BG File':
            if self.bg_file:
                bg_file = self.bg_file
                bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext)
        elif self.bg_type == 'Series Average':
            if self.bg_file:
                sname, fnames, bg, bg_meta = get_series_avg(self.bg_file, self.detector, self.meta_ext)
                if sname is None:
                    return 0
        else:
            if self.bg_dir and (self.bg_match_fname or self.bg_matching_par):
                bg_file_filter = 'bg' if not self.bg_file_filter else self.bg_file_filter
                if self.bg_match_fname:
                    bg_file_filter = f'{self.scan_name} {bg_file_filter}'
                filters = '*' + '*'.join(f for f in bg_file_filter.split()) + '*'
                filters = filters if filters != '**' else '*'

                meta_files = sorted(glob.glob(os.path.join(
                    # self.img_dir, f'{filters}[0-9][0-9][0-9][0-9].{self.meta_ext}')))
                    self.img_dir, f'{filters}.{self.meta_ext}')))

                for meta_file in meta_files:
                    bg_file = f'{os.path.splitext(meta_file)[0]}.{self.img_ext}'
                    if bg_file == img_file:
                        bg_file = None
                        continue

                    # bg_meta = get_img_meta(meta_file)
                    bg_meta = read_image_metadata(bg_file, meta_format=self.meta_ext)
                    if self.bg_match_fname:
                        if img_number == get_img_number(meta_file):
                            break
                    else:
                        try:
                            if bg_meta[self.bg_matching_par] == img_meta[self.bg_matching_par]:
                                break
                        except KeyError:
                            bg_file = None
                            continue

        if self.bg_type != 'Series Average':
            if bg_file is None:
                return 0.

            bg = get_img_data(bg_file, self.detector, return_float=True)
            if bg is None:
                return 0.

        if self.bg_scale != 1:
            bg *= self.bg_scale
            bg_scale_label = f'{self.bg_scale:0.2f} [Scale] x '
        if (self.bg_norm_channel != 'None') and (img_meta is not None) and (bg_meta is not None):
            try:
                if ((self.bg_norm_channel in img_meta.keys()) and
                        (self.bg_norm_channel in bg_meta.keys()) and
                        (bg_meta[self.bg_norm_channel] != 0)):
                    norm_factor = (img_meta[self.bg_norm_channel] / bg_meta[self.bg_norm_channel])
                    bg *= norm_factor
                    norm_label = f'{norm_factor:0.2f} [Normalized to Channel - {self.bg_norm_channel}] x '
            except (KeyError, TypeError):
                pass

        if self.bg_type != 'Series Average':
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{os.path.basename(bg_file)}]'
        else:
            self.sub_label = f'[Subtracted {bg_scale_label}{norm_label}{sname}]'

        return bg

    @staticmethod
    def save_1d(sphere, arch, idx):
        """
        Automatically save 1D integrated data
        """
        path = os.path.dirname(sphere.data_file)
        path = os.path.join(path, sphere.name)
        Path(path).mkdir(parents=True, exist_ok=True)

        q, tth, intensity = arch.int_1d.q, arch.int_1d.ttheta, arch.int_1d.norm

        # Write I(q) to xye
        fname = os.path.join(path, f'iq_{sphere.name}_{str(idx).zfill(4)}.xye')
        write_xye(fname, q, intensity, np.sqrt(abs(intensity)))

        # Write I(tth) to xye
        fname = os.path.join(path, f'itth_{sphere.name}_{str(idx).zfill(4)}.xye')
        write_xye(fname, tth, intensity, np.sqrt(abs(intensity)))

        # Write I(q) to csv
        fname = os.path.join(path, f'iq_{sphere.name}_{str(idx).zfill(4)}.csv')
        write_csv(fname, q, intensity)

        # Write I(tth) to csv
        fname = os.path.join(path, f'itth_{sphere.name}_{str(idx).zfill(4)}.csv')
        write_csv(fname, tth, intensity)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys_int(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys_float(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def natural_sort_ints(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_int)


def natural_sort_float(list_to_sort):
    return sorted(list_to_sort, key=natural_keys_float)
