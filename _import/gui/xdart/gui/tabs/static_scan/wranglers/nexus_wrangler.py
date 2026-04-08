# -*- coding: utf-8 -*-
"""
nexusWrangler — GUI widget for processing NeXus/HDF5 image stacks.

Provides a simple interface for selecting a NeXus file containing
image frames (e.g., from Bluesky suitcase-nexus exports or Eiger
master files) and a PONI calibration file.

@author: thampy
"""

# Standard library imports
import os
from pathlib import Path

# Qt imports
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import ParameterTree, Parameter

# Project imports
from ssrl_xrd_tools.core.containers import PONI
from .wrangler_widget import wranglerWidget
from .nexus_wrangler_thread import nexusThread
from ....gui_utils import NamedActionParameter
from xdart.utils import get_fname_dir
from xdart.utils.session import load_session, save_session

QFileDialog = QtWidgets.QFileDialog

params = [
    {'name': 'Calibration', 'type': 'group', 'children': [
        {'name': 'poni_file', 'title': 'PONI File    ', 'type': 'str', 'value': ''},
        NamedActionParameter(name='poni_file_browse', title='Browse...'),
    ], 'expanded': True},
    {'name': 'NeXus File', 'type': 'group', 'children': [
        {'name': 'nexus_file', 'title': 'File         ', 'type': 'str', 'value': ''},
        NamedActionParameter(name='nexus_file_browse', title='Browse...'),
        {'name': 'entry', 'title': 'Entry', 'type': 'str', 'value': 'entry'},
    ], 'expanded': True},
    {'name': 'Signal', 'type': 'group', 'children': [
        {'name': 'mask_file', 'title': 'Mask File    ', 'type': 'str', 'value': ''},
        NamedActionParameter(name='mask_file_browse', title='Browse...'),
    ], 'expanded': False},
    {'name': 'GI', 'type': 'group', 'children': [
        {'name': 'Grazing', 'title': 'Grazing Incidence', 'type': 'bool', 'value': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'str', 'value': 'th'},
        {'name': 'th_val', 'title': 'Theta Value', 'type': 'float', 'value': 0.0},
        {'name': 'sample_orientation', 'title': 'Sample Orientation', 'type': 'int',
         'value': 4, 'limits': (1, 4)},
        {'name': 'tilt_angle', 'title': 'Tilt Angle', 'type': 'float', 'value': 0.0},
        # gi_mode_1d / gi_mode_2d are controlled by the integrator panel;
        # kept here for session persistence only.
        {'name': 'gi_mode_1d', 'title': '1D Mode', 'type': 'list',
         'values': ['q_total', 'q_ip', 'q_oop', 'exit_angle'],
         'value': 'q_total', 'visible': False},
        {'name': 'gi_mode_2d', 'title': '2D Mode', 'type': 'list',
         'values': ['qip_qoop', 'q_chi', 'exit_angles'],
         'value': 'qip_qoop', 'visible': False},
    ], 'expanded': False},
    {'name': 'Output', 'type': 'group', 'children': [
        {'name': 'h5_dir', 'title': 'Output Dir   ', 'type': 'str', 'value': ''},
        NamedActionParameter(name='h5_dir_browse', title='Browse...'),
    ], 'expanded': False},
]


class nexusWrangler(wranglerWidget):
    """Widget for processing NeXus/HDF5 image stacks.

    A simpler alternative to specWrangler for data already stored
    in NeXus format (e.g. from Bluesky).

    signals:
        showLabel: str, status text
    """
    showLabel = QtCore.Signal(str)

    def __init__(self, fname, file_lock, sphere, data_1d, data_2d, parent=None):
        super().__init__(fname, file_lock, parent)

        self.poni = None
        self.command = None
        self.sphere = sphere
        self.data_1d = data_1d
        self.data_2d = data_2d

        # Attributes
        self.nexus_file = ''
        self.entry = 'entry'
        self.poni_file = ''
        self.mask_file = ''
        self.h5_dir = get_fname_dir()
        self.gi = False
        self.th_mtr = 'th'
        self.sample_orientation = 4
        self.tilt_angle = 0.0
        self.gi_mode_1d = 'q_total'
        self.gi_mode_2d = 'qip_qoop'

        # ── Build UI programmatically ────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Status label
        self.statusLabel = QtWidgets.QLabel('Ready')
        layout.addWidget(self.statusLabel)
        self.showLabel.connect(self.statusLabel.setText)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.startButton = QtWidgets.QPushButton('Start')
        self.stopButton = QtWidgets.QPushButton('Stop')
        self.skip2dCheckBox = QtWidgets.QCheckBox('Skip 2D')

        btn_layout.addWidget(self.startButton)
        btn_layout.addWidget(self.stopButton)
        btn_layout.addWidget(self.skip2dCheckBox)
        layout.addLayout(btn_layout)

        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)
        self.skip2dCheckBox.stateChanged.connect(
            lambda _: setattr(self.sphere, 'skip_2d', self.skip2dCheckBox.isChecked())
        )

        self.stopButton.setEnabled(False)

        # Parameter tree
        self.tree = ParameterTree()
        self.parameters = Parameter.create(
            name='nexus_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        layout.addWidget(self.tree)

        # Connect parameter browse buttons
        self.parameters.child('Calibration').child('poni_file_browse').sigActivated.connect(
            self.browse_poni
        )
        self.parameters.child('NeXus File').child('nexus_file_browse').sigActivated.connect(
            self.browse_nexus
        )
        self.parameters.child('Signal').child('mask_file_browse').sigActivated.connect(
            self.browse_mask
        )
        self.parameters.child('Output').child('h5_dir_browse').sigActivated.connect(
            self.browse_h5_dir
        )

        # Setup thread
        self.thread = nexusThread(
            self.command_queue,
            self.sphere_args,
            self.file_lock,
            self.fname,
            self.nexus_file,
            self.poni,
            self.mask_file,
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
            entry=self.entry,
            parent=self,
        )
        self.thread.showLabel.connect(self.statusLabel.setText)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

        self._restore_from_session()

    # ── Session persistence ──────────────────────────────────────────

    _SESSION_PARAMS = [
        ('poni_file',           ('Calibration', 'poni_file'),    True,  'poni_file'),
        ('nexus_file',          ('NeXus File', 'nexus_file'),    True,  'nexus_file'),
        ('entry',               ('NeXus File', 'entry'),         False, 'entry'),
        ('mask_file',           ('Signal', 'mask_file'),         True,  'mask_file'),
        ('h5_dir',              ('Output', 'h5_dir'),            True,  'h5_dir'),
        ('gi',                  ('GI', 'Grazing'),               False, 'gi'),
        ('sample_orientation',  ('GI', 'sample_orientation'),    False, 'sample_orientation'),
        ('tilt_angle',          ('GI', 'tilt_angle'),            False, 'tilt_angle'),
        ('gi_mode_1d',          ('GI', 'gi_mode_1d'),            False, 'gi_mode_1d'),
        ('gi_mode_2d',          ('GI', 'gi_mode_2d'),            False, 'gi_mode_2d'),
    ]

    def _restore_from_session(self):
        """Restore parameters from session.json."""
        data = load_session()
        if not data:
            return
        for skey, param_path, is_path, attr in self._SESSION_PARAMS:
            val = data.get(skey)
            if val is None:
                continue
            if is_path and not os.path.exists(val):
                continue
            try:
                p = self.parameters
                for name in param_path:
                    p = p.child(name)
                p.setValue(val)
                if attr:
                    setattr(self, attr, val)
            except Exception:
                pass

        # Restore PONI
        poni_file = self.parameters.child('Calibration').child('poni_file').value()
        if poni_file and os.path.exists(poni_file):
            self.poni = PONI.from_poni_file(poni_file)

    def _save_to_session(self):
        """Save parameters to session.json."""
        data = load_session() or {}
        for skey, param_path, _is_path, _attr in self._SESSION_PARAMS:
            try:
                p = self.parameters
                for name in param_path:
                    p = p.child(name)
                data[skey] = p.value()
            except Exception:
                pass
        save_session(data)

    # ── Browse dialogs ───────────────────────────────────────────────

    def browse_poni(self):
        poni_file, _ = QFileDialog.getOpenFileName(
            self, 'Select PONI file', '', 'PONI files (*.poni);;All files (*)')
        if poni_file:
            self.parameters.child('Calibration').child('poni_file').setValue(poni_file)
            self.poni_file = poni_file
            self.poni = PONI.from_poni_file(poni_file)
            self._save_to_session()

    def browse_nexus(self):
        nexus_file, _ = QFileDialog.getOpenFileName(
            self, 'Select NeXus/HDF5 file',
            os.path.dirname(self.nexus_file) if self.nexus_file else '',
            'HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)')
        if nexus_file:
            self.parameters.child('NeXus File').child('nexus_file').setValue(nexus_file)
            self.nexus_file = nexus_file
            self._save_to_session()

    def browse_mask(self):
        mask_file, _ = QFileDialog.getOpenFileName(
            self, 'Select mask file', '', 'Image files (*.tif *.edf *.npy);;All files (*)')
        if mask_file:
            self.parameters.child('Signal').child('mask_file').setValue(mask_file)
            self.mask_file = mask_file
            self._save_to_session()

    def browse_h5_dir(self):
        h5_dir = QFileDialog.getExistingDirectory(
            self, 'Select output directory',
            self.h5_dir, QFileDialog.ShowDirsOnly)
        if h5_dir:
            self.parameters.child('Output').child('h5_dir').setValue(h5_dir)
            self.h5_dir = h5_dir
            self._save_to_session()

    # ── Thread control ───────────────────────────────────────────────

    def setup(self):
        """Sync parameters to thread before starting."""
        self.nexus_file = self.parameters.child('NeXus File').child('nexus_file').value()
        self.entry = self.parameters.child('NeXus File').child('entry').value()
        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        self.mask_file = self.parameters.child('Signal').child('mask_file').value()

        # Load PONI if needed
        if self.poni_file and os.path.exists(self.poni_file):
            self.poni = PONI.from_poni_file(self.poni_file)

        # Output directory
        h5_dir = self.parameters.child('Output').child('h5_dir').value()
        if h5_dir:
            self.h5_dir = h5_dir

        # HDF5 output file
        scan_name = Path(self.nexus_file).stem if self.nexus_file else 'nexus_scan'
        self.fname = os.path.join(self.h5_dir, f'{scan_name}.nxs')
        self.scan_name = scan_name

        # GI parameters
        self.gi = self.parameters.child('GI').child('Grazing').value()
        self.th_mtr = self.parameters.child('GI').child('th_motor').value()
        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        # GI modes are driven by the integrator panel (axis1D / axis2D)
        self.gi_mode_1d = self.sphere.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')

        # Recreate thread with current params
        self.thread = nexusThread(
            self.command_queue,
            self.sphere_args,
            self.file_lock,
            self.fname,
            self.nexus_file,
            self.poni,
            self.mask_file,
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
            entry=self.entry,
            parent=self,
        )
        self.thread.showLabel.connect(self.statusLabel.setText)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)
        self.sigUpdateGI.emit(self.gi)

        self.thread.file_lock = self.file_lock
        self.thread.sphere_args = self.sphere_args
        self.thread.sphere = self.sphere
        self.thread.data_1d = self.data_1d
        self.thread.data_2d = self.data_2d
        self.thread.command = self.command

    def start(self):
        self.command = 'start'
        self.thread.command = 'start'
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self._save_to_session()
        self.sigStart.emit()

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def enabled(self, enable):
        self.startButton.setEnabled(enable)
        self.tree.setEnabled(enable)
