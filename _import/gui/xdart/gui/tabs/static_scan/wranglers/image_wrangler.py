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
# ``_extract_scan_info`` is a private helper but the wrangler uses it
# to predict whether the SPEC sidecar exists for a given image file —
# same parser the SSRL reader uses, so the UI's existence check stays
# in sync with what ``read_image_metadata`` will actually look for.
from ssrl_xrd_tools.io.metadata import _extract_scan_info
from .wrangler_widget import wranglerWidget
from .image_wrangler_thread import imageThread, _get_scan_info  # noqa: F401
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
    # N1: the portable Project Folder.  Setting it stamps entry/@source_base and
    # makes each frame's raw source path RELATIVE to it, so the processed .nxs
    # resolves its raw images after the data moves machines.  Blank -> absolute
    # paths (back-compat).  (The full progressive-disclosure / folder-change
    # reset UX is a follow-up; the portable storage is active once a folder is
    # set.)
    {'name': 'Project', 'title': 'Project Folder', 'type': 'group', 'children': [
        {'name': 'project_folder', 'title': 'Folder', 'type': 'str', 'value': ''},
        NamedActionParameter(name='project_folder_browse', title='Browse...'),
    ], 'expanded': True},
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
        # Optional override for the SPEC file's directory.  Shown only
        # when ``meta_ext == 'SPEC'`` (see ``set_meta_ext``).  Blank
        # → use the default search (image dir + immediate parent);
        # set → that directory is searched first.
        {'name': 'meta_dir', 'title': 'Meta Directory', 'type': 'str',
         'value': '', 'visible': False},
        NamedActionParameter(name='meta_dir_browse', title='Browse...', visible=False),
        {'name': 'Filter', 'type': 'str', 'value': '', 'visible': False},
        {'name': 'write_mode', 'title': 'Write Mode  ', 'type': 'list',
         'values': ['Append', 'Overwrite'], 'value': 'Append'},
        {'name': 'mask_file', 'title': 'Mask File', 'type': 'str', 'value': ''},
        NamedActionParameter(name='mask_file_browse', title='Browse...'),
    ], 'expanded': True, 'visible': False},
    {'name': 'GI', 'title': 'Grazing Incidence', 'type': 'group', 'children': [
        {'name': 'Grazing', 'type': 'bool', 'value': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'list',
         'values': ['th', 'Manual'], 'value': 'th'},
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



class imageWrangler(wranglerWidget):
    """Widget for integrating detector image files (TIFF/EDF/CBF/Eiger
    master; as an image series, image directory, or single image).
    Per-frame metadata is read from an optional sidecar — a SPEC file,
    a ``.txt``/``.pdi`` file, or nothing (the ``Meta Format`` option).
    Can be used "live": it polls the data folder for new images (and
    their metadata, if any) until the scan is complete.

    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
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
        sigUpdateData: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
        showLabel: str, connected to thread showLabel signal, sets text
            in specLabel
    """
    showLabel = QtCore.Signal(str)
    sigSavePathChanged = QtCore.Signal(str)

    def __init__(self, fname, file_lock, scan, data_1d, data_2d, parent=None):
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
        self.scan = scan
        self.data_1d = data_1d
        self.data_2d = data_2d

        # Setup gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Deferred for this release: the NeXus Viewer mode is not yet usable on
        # stacked datasets (integrated_1d/2d store the whole (N, ...) stack
        # under one key, so it needs a per-frame slider).  Hide the option to
        # avoid the half-baked viewer; the controller/mode code stays dormant
        # and the option returns once the slider lands.
        _nx_idx = self.ui.processingModeCombo.findText('NeXus Viewer')
        if _nx_idx >= 0:
            self.ui.processingModeCombo.removeItem(_nx_idx)
        self.ui.startButton.clicked.connect(self._on_start_clicked)
        # self.ui.startButton.clicked.connect(self.sigStart.emit)
        self.ui.stopButton.clicked.connect(self.stop)
        self.ui.processingModeCombo.currentTextChanged.connect(self._on_mode_changed)
        # Live/Batch are checkable QPushButtons now — use ``toggled`` (bool)
        # rather than the QCheckBox-only ``stateChanged``.
        self.ui.liveCheckBox.toggled.connect(self._on_mode_changed)
        self.ui.batchCheckBox.toggled.connect(self._on_mode_changed)
        # Live doubles as a start/stop toggle.  Connected AFTER
        # _on_mode_changed so live_mode is already set when we start.
        self.ui.liveCheckBox.toggled.connect(self._on_live_toggled)
        self.ui.processingModeCombo.currentTextChanged.connect(lambda _: self._save_to_session())
        self.ui.liveCheckBox.toggled.connect(lambda _: self._save_to_session())
        self.ui.batchCheckBox.toggled.connect(lambda _: self._save_to_session())
        self._on_mode_changed()
        self._set_wrangler_tooltips()

        self.showLabel.connect(self.ui.specLabel.setText)

        # Setup parameter tree
        self.tree = ParameterTree()
        # This (and the name-column width below) is the dominant floor on
        # how narrow the right panel drags — the param tree is the widest
        # fixed content.  Keep it small so the panel can shrink.
        self.tree.setMinimumWidth(80)
        self.stylize_ParameterTree()
        self.parameters = Parameter.create(
            name='image_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        # Squeeze parameter tree columns to reduce panel width
        header = self.tree.header()
        header.setStretchLastSection(True)
        header.resizeSection(0, 100)  # name column
        header.setMinimumSectionSize(40)
        self.layout = QtWidgets.QVBoxLayout(self.ui.paramFrame)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.tree)

        # Set attributes from Parameter Tree and a couple more
        # N1: Project Folder (portable @source_base).  Blank -> None (absolute).
        # _restoring gates the folder-change reset off during session restore.
        self._restoring = False
        self.project_folder = self.parameters.child('Project').child('project_folder').value()
        self.source_base = self._compute_source_base()
        # Calibration
        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        # Applies the N1 progressive disclosure for the fresh-start state (no
        # folder -> only Project visible).
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
        # Optional explicit dir for SPEC files; '' falls back to the
        # ssrl_xrd_tools default (image dir + parent search).  Wired
        # to the thread so workers pass it to read_image_metadata.
        self.meta_dir = self.parameters.child('Signal').child('meta_dir').value()

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
        self.incidence_motor = self.parameters.child('GI').child('th_motor').value()
        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        # gi_mode_1d / gi_mode_2d are driven by the integrator panel;
        # default here, actual values set from scan.bai_*_args at thread start.
        self.gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')

        # HDF5 Save Path
        self.h5_dir = self.parameters.child('h5_dir').value()

        # NOTE: Integration Advanced button (self.ui.advancedButton) is wired
        # in static_scan_widget.set_wrangler() to show the integratorTree's
        # existing advancedWidget1D / advancedWidget2D dialogs directly.

        # Wire signals from parameter tree based buttons
        self.parameters.sigTreeStateChanged.connect(self.setup)
        self.parameters.sigTreeStateChanged.connect(self._save_to_session)

        self.parameters.child('Project').child('project_folder_browse').sigActivated.connect(
            self.set_project_folder
        )
        # N1 Decision 2: a folder change (browse OR direct edit) resets the
        # dependent paths.  Guarded by _restoring so a session restore doesn't
        # trip it (see _restore_from_session).
        self.parameters.child('Project').child('project_folder').sigValueChanged.connect(
            self._on_project_folder_changed
        )
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
        self.parameters.child('Signal').child('meta_dir_browse').sigActivated.connect(
            self.set_meta_dir
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
        self.thread = imageThread(
            self.command_queue,
            self.scan_args,
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
            self.incidence_motor,
            self.sample_orientation,
            self.tilt_angle,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.command,
            self.scan,
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
        # self.thread.sigUpdateFrame.connect(self.sigUpdateFrame.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

        # Enable/disable buttons initially
        self.ui.stopButton.setEnabled(False)

        self.setup()
        self._restore_from_session()
        # Open the GI / Threshold / Background groups when their toggle is on
        # (e.g. from a restored session) so the relevant controls are visible
        # instead of folded; collapsed when off.
        self._expand_active_groups()

    def _expand_active_groups(self):
        """Expand each wrangler group whose enabling param is set.

        The GI / Intensity-Threshold / Background groups default folded.  If
        Grazing is checked, Threshold is enabled, or a Background source is
        selected (incl. after a session restore), expand that group via
        ``setOpts(expanded=True)``; leave it collapsed when off."""
        groups = (
            ('GI', ('GI', 'Grazing'), lambda v: bool(v)),
            ('Mask', ('Mask', 'Threshold'), lambda v: bool(v)),
            ('BG', ('BG', 'bg_type'), lambda v: v not in (None, '', 'None')),
        )
        for group_name, child_path, is_on in groups:
            try:
                group = self.parameters.child(group_name)
                value = self.parameters.child(*child_path).value()
            except Exception:
                logger.debug("expand-active-group skipped for %s", group_name,
                             exc_info=True)
                continue
            if is_on(value):
                group.setOpts(expanded=True)

    # --- Session persistence ---

    # Flat map: (session_key, param_path_tuple, is_path, self_attr)
    # is_path=True  → only restore if the path exists on disk
    # self_attr     → attribute to sync on restore (None = handled by sigValueChanged)
    _SESSION_PARAMS = [
        ('project_folder', ('Project', 'project_folder'),            True,  'project_folder'),
        ('poni_file',      ('Calibration', 'poni_file'),             True,  'poni_file'),
        ('inp_type',       ('Signal', 'inp_type'),                   False, 'inp_type'),
        ('img_file',       ('Signal', 'File'),                       True,  'img_file'),
        ('img_dir',        ('Signal', 'img_dir'),                    True,  'img_dir'),
        ('include_subdir', ('Signal', 'include_subdir'),             False, 'include_subdir'),
        ('img_ext',        ('Signal', 'img_ext'),                    False, 'img_ext'),
        ('series_average', ('Signal', 'series_average'),             False, 'series_average'),
        ('meta_ext',       ('Signal', 'meta_ext'),                   False, None),
        ('meta_dir',       ('Signal', 'meta_dir'),                   True,  'meta_dir'),
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
        # J1: json session key stays 'th_mtr' for back-compat with
        # old session.json files; attr name is now 'incidence_motor'.
        ('th_mtr',         ('GI', 'th_motor'),                       False, 'incidence_motor'),
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
        # Live is a momentary start/stop control now (not a persisted mode);
        # its checked state means "a live run is active", which must never be
        # restored — doing so would auto-start a run on launch.
        data['batch_mode'] = self.ui.batchCheckBox.isChecked()
        save_session(data)

    def _restore_from_session(self):
        session = load_session()
        # N1: restoring project_folder fires its sigValueChanged ->
        # _on_project_folder_changed, which would DESTRUCTIVELY clear the very
        # poni_file/img_file this loop is about to restore.  Gate it inert for
        # the duration of the restore.  (A project_folder whose root no longer
        # exists is is_path-skipped below -> stays blank -> Decision 3 fresh-start
        # fallback, prompting the user to re-enter the folder.)
        self._restoring = True
        try:
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
            # Deliberately do NOT restore Live's checked state — it's a
            # start/stop control, and setChecked(True) would fire its toggled
            # handler and auto-start a live run on launch.
            if 'batch_mode' in session:
                self.ui.batchCheckBox.setChecked(session['batch_mode'])
        finally:
            self._restoring = False
        # meta_ext needs None conversion (sigValueChanged fires set_meta_ext automatically)
        # poni_file needs poni_dict loaded; this re-applies the disclosure for the
        # restored Project-Folder + PONI state (or the fresh-start fallback when
        # the root was missing/skipped).
        self.source_base = self._compute_source_base()
        self.get_poni_dict()

    def _sync_h5_dir_from_parameters(self):
        """Sync the Save Path parameter and notify the scans browser on change."""
        path = self.parameters.child('h5_dir').value()
        old_path = getattr(self, 'h5_dir', None)
        self.h5_dir = path
        if path and path != old_path:
            self.sigSavePathChanged.emit(path)

    # Signal to notify static_scan_widget that viewer mode changed.
    # Emits the viewer_mode string ('image', 'xye') or '' for normal.
    sigViewerModeChanged = QtCore.Signal(str)

    def _on_mode_changed(self, *args):
        """Update all flags from the processing mode dropdown and checkboxes."""
        mode_text = self.ui.processingModeCombo.currentText()
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer', 'NeXus Viewer')
        is_file_viewer = mode_text in ('Image Viewer', 'XYE Viewer')
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
            # Stash the user's Batch choice once on entering XYE so it can be
            # restored on leaving -- XYE force-checks Batch, and the normal
            # branch used to leave it stuck checked (UI-2).
            if getattr(self, '_pre_xye_batch', None) is None:
                self._pre_xye_batch = self.ui.batchCheckBox.isChecked()
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setChecked(True)
            self.ui.batchCheckBox.setEnabled(False)
            self.ui.coresLabel.setEnabled(True)
            self.ui.maxCoresSpinBox.setEnabled(True)
        else:
            # Restore the pre-XYE Batch choice when leaving XYE (UI-2).
            if getattr(self, '_pre_xye_batch', None) is not None:
                self.ui.batchCheckBox.setChecked(self._pre_xye_batch)
                self._pre_xye_batch = None
            # Both toggles stay clickable in a normal processing mode; they
            # are kept mutually exclusive by auto-unchecking the other one
            # rather than greying it out.  Greying Live out whenever Batch
            # was checked left it dead after a mode switch until a run
            # finished and enabled(True) reset it (bug #1).
            self.ui.liveCheckBox.setEnabled(True)
            self.ui.batchCheckBox.setEnabled(True)

            is_live = self.ui.liveCheckBox.isChecked()
            is_batch = self.ui.batchCheckBox.isChecked()
            if is_live and is_batch:
                # Uncheck whichever one was NOT just toggled.  When the
                # trigger is the mode combo (or unknown), prefer Live.
                if self.sender() is self.ui.batchCheckBox:
                    self.ui.liveCheckBox.setChecked(False)
                    is_live = False
                else:
                    self.ui.batchCheckBox.setChecked(False)
                    is_batch = False

            # Sync cores enabled state with batch checkbox
            self.ui.coresLabel.setEnabled(is_batch)
            self.ui.maxCoresSpinBox.setEnabled(is_batch)

        self.ui.liveCheckBox.blockSignals(False)
        self.ui.batchCheckBox.blockSignals(False)

        # Cores only matters for parallel batch processing — hide it
        # entirely unless batch is active (XYE forces batch on).
        if is_viewer:
            cores_visible = False
        elif is_xye:
            cores_visible = True
        else:
            cores_visible = self.ui.batchCheckBox.isChecked()
        self.ui.coresLabel.setVisible(cores_visible)
        self.ui.maxCoresSpinBox.setVisible(cores_visible)

        self.batch_mode = self.ui.batchCheckBox.isChecked()
        self.live_mode = self.ui.liveCheckBox.isChecked()
        self.xye_only = is_xye
        
        if mode_text == 'Image Viewer':
            self.viewer_mode = 'image'
            self.scan.skip_2d = False
        elif mode_text == 'XYE Viewer':
            self.viewer_mode = 'xye'
            self.scan.skip_2d = False
        elif mode_text == 'NeXus Viewer':
            self.viewer_mode = 'nexus'
            self.scan.skip_2d = False
        else:
            self.viewer_mode = None
            self.scan.skip_2d = '1D' in mode_text

        # Sync to thread
        self.thread.batch_mode = self.batch_mode
        self.thread.xye_only = self.xye_only
        self.thread.live_mode = self.live_mode

        # Gray out integration controls in viewer mode
        self._set_integration_controls_enabled(not is_viewer)
        # Image/XYE viewers are file-inspection modes.  Disable the processing
        # parameter tree itself so masks/background/calibration state cannot be
        # edited or accidentally interpreted as viewer state.  The mode combo
        # lives outside this tree, so the user can still switch back.
        try:
            self.tree.setEnabled(not is_file_viewer)
        except AttributeError:
            pass
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

    def _set_integration_controls_enabled(self, enabled, *, include_gi=True):
        """Enable or disable parameter tree groups related to integration."""
        group_names = ['Calibration', 'BG', 'Mask']
        if include_gi:
            group_names.append('GI')
        for group_name in group_names:
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

        # N1: Project Folder -> @source_base (raw source paths stored RELATIVE to
        # it -> portable .nxs).  Blank -> None -> absolute paths (back-compat).
        self.project_folder = (
            self.parameters.child('Project').child('project_folder').value() or '')
        self.source_base = self._compute_source_base()
        self.thread.source_base = self.source_base

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
        self.thread.meta_dir = self.meta_dir

        self._sync_h5_dir_from_parameters()
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

        # self.incidence_motor = self.parameters.child('GI').child('th_motor').value()
        self.thread.incidence_motor = self.incidence_motor

        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.thread.sample_orientation = self.sample_orientation

        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        self.thread.tilt_angle = self.tilt_angle

        # GI modes are driven by the integrator panel (axis1D / axis2D),
        # so read them from scan.bai_*_args which the integrator updates.
        self.gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
        self.thread.gi_mode_1d = self.gi_mode_1d
        self.thread.gi_mode_2d = self.gi_mode_2d

        # N3: record the GI output mode + geometry as a first-class
        # ``scan.gi_config`` (persisted by the writer to
        # /entry/reduction/config/gi_config), so read_scan can recover the GI
        # mode + axis meaning without sniffing the q-unit string or digging the
        # mode key out of bai_*_args.
        if self.gi:
            self.scan.gi_config = {
                'gi_mode_1d': str(self.gi_mode_1d),
                'gi_mode_2d': str(self.gi_mode_2d),
                'incidence_motor': str(getattr(self, 'incidence_motor', '') or ''),
                'tilt_angle': float(getattr(self, 'tilt_angle', 0.0) or 0.0),
                'sample_orientation': int(getattr(self, 'sample_orientation', 1) or 1),
            }
        else:
            self.scan.gi_config = {}

        # Notify integrator panel so labels/widgets update immediately.
        self.sigUpdateGI.emit(self.gi)

        # Processing mode flags and parallel cores
        self.thread.live_mode = self.live_mode
        self.thread.batch_mode = self.batch_mode
        self.thread.xye_only = self.xye_only
        self.thread.max_cores = self.ui.maxCoresSpinBox.value()
        self.scan.max_cores = self.thread.max_cores  # used by scan_threads

        self.thread.command = self.command

        self.thread.file_lock = self.file_lock
        self.thread.scan_args = self.scan_args

        self.thread.scan = self.scan
        self.thread.data_1d = self.data_1d
        self.thread.data_2d = self.data_2d

    def _set_wrangler_tooltips(self):
        """Hover tooltips for the wrangler command/run controls."""
        tips = {
            'processingModeCombo': 'What to produce: integrate (1D/2D/XYE), '
                                   'stitch, or just view images/patterns.',
            'liveCheckBox': 'Start/stop live acquisition — process frames as '
                            'they arrive.',
            'batchCheckBox': 'Process all frames as a batch (parallel across '
                             'Cores) instead of one-at-a-time.',
            'maxCoresSpinBox': 'CPU cores for parallel batch processing.',
            'coresLabel': 'CPU cores for parallel batch processing.',
            'advancedButton': 'Advanced integration / detector options.',
            'startButton': 'Start processing with the current settings.',
            'stopButton': 'Stop the running process.',
        }
        for name, tip in tips.items():
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setToolTip(tip)

    def _on_start_clicked(self):
        """Start button = an explicit NON-live processing run.

        The Start button and the Live toggle both funnel into :meth:`start`,
        and ``live_mode`` only ever tracks the Live toggle's checked state.
        If Live happened to be left checked, a Start click would silently run
        in live-watching mode ("Watching for new files...") with the Live
        button lit.  Force a non-live run here: uncheck Live (without firing
        its start/stop handler) and clear ``live_mode`` so the wrangler runs
        once over the existing files and ``enabled(False)`` greys Live out for
        the duration."""
        if self.ui.liveCheckBox.isChecked():
            self.ui.liveCheckBox.blockSignals(True)
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.blockSignals(False)
        self.live_mode = False
        self.thread.live_mode = False
        self.start()

    def _inputs_valid(self):
        """Whether the wrangler can start a run.

        A loaded PONI calibration is required; without this gate a Start/Live
        click with no (or an invalid) PONI ran the *previous* scan with the
        stale calibration (BUG-1).  The image source / save path remain guarded
        inside the run thread."""
        if self.poni is None:
            self.ui.specLabel.setText('Load a PONI calibration file to begin.')
            return False
        return True

    def start(self):
        # Gate both the Start button and the Live toggle (both funnel here):
        # refuse to run without a valid PONI rather than re-running the stale
        # previous scan.
        if not self._inputs_valid():
            return
        self.command = 'start'
        self.thread.command = 'start'
        self.ui.stopButton.setEnabled(True)
        self.sigStart.emit()

    def _on_live_toggled(self, checked):
        """Live button is a start/stop toggle for live acquisition:
        checking it starts a live run (live_mode is already set by
        :meth:`_on_mode_changed`, which runs first); unchecking it stops
        the run.  Fires only on genuine user clicks — the programmatic
        resets in :meth:`stop` / :meth:`enabled` block the signal."""
        if checked:
            self.start()
        else:
            self.stop()

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        self.ui.stopButton.setEnabled(False)
        self.ui.specLabel.setText('')
        # Keep the Live toggle in sync when stopped via the Stop button or
        # programmatically — uncheck it without re-entering stop().  Because
        # the uncheck is done with signals blocked, ``_on_mode_changed`` does
        # NOT run, so reset ``live_mode`` explicitly here; otherwise it stays
        # stale-True and the next Start click silently runs in live mode.
        if self.ui.liveCheckBox.isChecked():
            self.ui.liveCheckBox.blockSignals(True)
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.blockSignals(False)
        self.live_mode = False
        self.thread.live_mode = False

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

    # N1: param groups gated behind the Project Folder + PONI (Project itself is
    # always visible).  Calibration appears once a Project Folder is set; the
    # rest appears once a valid PONI also loads.
    _DISCLOSURE_REST = ('Signal', 'GI', 'Mask', 'BG')

    def _apply_disclosure(self):
        """N1 progressive disclosure (design §2): the tree reveals in stages —
        Project Folder (always) -> Calibration (once a folder is set) -> the rest
        (once a folder is set AND a valid PONI loads).  Pure show()/hide() on the
        groups; orthogonal to the run-lock ``enabled()`` (which only greys)."""
        have_root = self._compute_source_base() is not None
        have_poni = self.poni is not None
        self.parameters.child('Project').show()            # always visible
        cal = self.parameters.child('Calibration')
        if not have_root:
            cal.hide()
            for name in self._DISCLOSURE_REST:
                self.parameters.child(name).hide()
            self.ui.specLabel.setText('Choose a Project Folder to begin.')
        elif not have_poni:
            cal.show()
            for name in self._DISCLOSURE_REST:
                self.parameters.child(name).hide()
            self.ui.specLabel.setText('Load a PONI calibration file to begin.')
        else:
            for child in self.parameters.children():
                child.show()                               # reveal everything
            self.ui.specLabel.setText('')

    def get_poni_dict(self):
        """Load the PONI calibration file and store as a PONI object, then apply
        the N1 progressive disclosure (reveal the rest only with a Project Folder
        AND a valid PONI)."""
        if not os.path.exists(self.poni_file):
            # No calibration: clear any stale PONI so the run guard +
            # _inputs_valid trip (hiding it left the previous scan's PONI live, so
            # a Start click ran the old scan with the stale calibration, BUG-1).
            self.poni = None
            self.thread.poni = None
            self._apply_disclosure()
            return

        try:
            self.poni = PONI.from_poni_file(self.poni_file)
        except (IOError, OSError, ValueError, KeyError) as e:
            logger.debug("Failed to load PONI file %s: %s", self.poni_file, e)
            self.poni = None
        if self.poni is None:
            logger.warning('Invalid Poni File: %s', self.poni_file)
            self.thread.poni = None
            self.thread.signal_q.put(('message', 'Invalid Poni File'))
            self._apply_disclosure()
            return

        self._apply_disclosure()

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

        if ((self.img_file != old_fname)
                or (self.img_file and (len(self.scan_parameters) < 1))):
            if (self.meta_ext and self.img_file
                    and self.exists_meta_file(self.img_file)):
                self.set_pars_from_meta()
            elif self.img_file:
                # New signal file with no sidecar metadata (e.g. Eiger):
                # clear the previous file's stale motor/parameter options and
                # default the GI Theta Motor to Manual (no 'th' to read), so
                # the incidence angle can be entered directly.
                self.scan_parameters = []
                self.motors = []
                self.set_gi_motor_options()

    def set_series_average(self):
        self.series_average = self.parameters.child('Signal').child('series_average').value()

    def set_meta_ext(self):
        self.meta_ext = self.parameters.child('Signal').child('meta_ext').value()
        if self.meta_ext == 'None':
            self.meta_ext = None
        # Show "Meta Directory" + Browse only for SPEC mode.  Other
        # formats (txt/pdi) look next to the image — no separate dir
        # makes sense.  This mirrors the bg_dir pattern.
        is_spec = (self.meta_ext == 'SPEC')
        self.parameters.child('Signal').child('meta_dir').show(is_spec)
        self.parameters.child('Signal').child('meta_dir_browse').show(is_spec)
        self._save_to_session()
        self.get_img_fname()

    def set_meta_dir(self):
        """Opens a directory chooser for the SPEC file's location.

        Sets ``meta_dir`` to the picked path; leaves it alone if the
        user cancels.  Empty string means "use the default search
        (image dir + immediate parent)".
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Meta (SPEC) Directory',
            dir=self.meta_dir or '',
            options=QFileDialog.ShowDirsOnly,
        )
        if path:
            self.parameters.child('Signal').child('meta_dir').setValue(path)
            self.meta_dir = path

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
                # Honour the optional explicit Meta Directory before
                # falling back to the image dir + parent search, so
                # the existence check matches what ``read_image_metadata``
                # will actually look at when SPEC is selected.
                search_dirs = []
                if getattr(self, 'meta_dir', None):
                    search_dirs.append(Path(self.meta_dir))
                search_dirs.extend([img_fpath.parent, img_fpath.parents[1]])
                for parent in search_dirs:
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
            dir='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            Path(path).mkdir(parents=True, exist_ok=True)
            self.parameters.child('h5_dir').setValue(path)
            self._sync_h5_dir_from_parameters()

    def _compute_source_base(self):
        """N1: the absolute project root, or None when the Project Folder is
        blank (-> the writer stores absolute raw paths, back-compat)."""
        pf = (self.parameters.child('Project').child('project_folder').value() or '').strip()
        return os.path.abspath(os.path.expanduser(pf)) if pf else None

    def _default_h5_under_project(self):
        """Default the Save Path to ``<project>/xdart_processed_data`` when the
        user hasn't chosen one (blank or still the app default)."""
        base = self._compute_source_base()
        if not base:
            return
        cur_h5 = (self.parameters.child('h5_dir').value() or '').strip()
        if not cur_h5 or cur_h5 == get_fname_dir():
            self.parameters.child('h5_dir').setValue(
                os.path.join(base, 'xdart_processed_data'))
            self._sync_h5_dir_from_parameters()

    def set_project_folder(self):
        """Browse for the N1 Project Folder.  Setting it stores raw source paths
        RELATIVE to this root (portable .nxs); the value-change handler
        (:meth:`_on_project_folder_changed`) then resets the dependent
        (folder-relative) paths + defaults the Save Path."""
        path = QFileDialog().getExistingDirectory(
            caption='Choose Project Folder',
            dir='',
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            self.parameters.child('Project').child('project_folder').setValue(path)

    def _on_project_folder_changed(self, *args):
        """N1 Decision 2: a Project Folder change INVALIDATES everything stored
        relative to the OLD root.  Recompute ``source_base``, clear the PONI +
        the dependent source paths (which cascades back to the enter-PONI
        disclosure state via :meth:`get_poni_dict`), and default the Save Path
        under the new folder.  Inert during a session restore (the ``_restoring``
        guard) so it doesn't wipe the values the restore is setting."""
        if getattr(self, '_restoring', False):
            return
        self.source_base = self._compute_source_base()
        if getattr(self, 'thread', None) is not None:
            self.thread.source_base = self.source_base
        # Clear the PONI (cascades through get_poni_dict -> _apply_disclosure) and
        # the source paths that were relative to the now-stale old root.
        self.parameters.child('Calibration').child('poni_file').setValue('')
        for seg in (('Signal', 'File'), ('Signal', 'img_dir'),
                    ('Signal', 'mask_file')):
            try:
                self.parameters.child(*seg).setValue('')
            except (AttributeError, KeyError, TypeError):
                pass
        self._default_h5_under_project()
        self._apply_disclosure()

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
        """Reads image metadata to populate possible GI theta motor.

        Always offers a 'Manual' option (enter the incidence angle directly
        via the Theta field).  When no motors are found — e.g. Eiger / no
        metadata — Manual is the default, since there's no ``th`` to read.
        """
        pars = [p for p in self.motors if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        if 'th' in pars:
            pars.insert(0, pars.pop(pars.index('th')))
            value = 'th'
        elif 'theta' in pars:
            pars.insert(0, pars.pop(pars.index('theta')))
            value = 'theta'
        elif pars:
            value = pars[0]
        else:
            # No motors (no metadata) → default to Manual incidence entry.
            value = 'Manual'

        pars = ['Manual'] + pars

        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('GI').child('th_motor').setOpts(**opts)
        # Sync the Theta-value field visibility + incidence_motor to the
        # (possibly newly-defaulted) selection — setOpts may not re-fire
        # sigValueChanged when the value is set programmatically.
        self.set_gi_th_motor()

    def set_gi_th_motor(self):
        """Update Grazing theta motor.

        Reveals the Manual 'Theta' value field only when Theta Motor is
        'Manual', and hides it otherwise.  Eiger / metadata-less files have
        no ``th`` to read, so without this input field the Manual path can't
        supply an incidence angle and the GI cake stays blank.  Use
        ``setOpts(visible=...)`` (not hide()/show()) so the param-tree row
        reliably re-renders.
        """
        th_motor = self.parameters.child('GI').child('th_motor').value()
        th_val = self.parameters.child('GI').child('th_val')
        if th_motor == 'Manual':
            th_val.setOpts(visible=True)
            self.incidence_motor = th_val.value()
        else:
            th_val.setOpts(visible=False)
            self.incidence_motor = th_motor

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
        """Enable/disable the WHOLE wrangler panel for the run lifecycle (#72).

        During a run everything is locked except Stop: the parameter tree is
        hard-disabled (greyed + fully non-interactive, matching the integration
        panel above it), and the non-param widgets (processing-mode combo, Cores
        spinbox + label, Advanced button) are disabled too.  A disabled pyqtgraph
        bool checkbox (Grazing, Average Scan, …) may repaint unchecked during the
        run (#56), but the value is preserved and restored on re-enable — the
        full visible disable was chosen over that cosmetic ("minimize
        complexity").  The running thread uses the setup-time arg snapshot
        regardless.

        args:
            enable: bool, True for enabled False for disabled.
        """
        self.tree.setEnabled(enable)
        self.ui.startButton.setEnabled(enable)
        # Non-param widgets (live outside the ParameterTree): mode combo, Cores
        # spinbox + label, Advanced button.  Stop is left alone (stays enabled).
        for name in ('processingModeCombo', 'maxCoresSpinBox', 'coresLabel',
                     'advancedButton'):
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setEnabled(enable)
        # Live toggle state vs. the run lifecycle:
        if enable:
            # Run finished — reset Live to off (no re-trigger) and re-enable
            # both toggles so the next run can be either live or batch.
            self.ui.liveCheckBox.blockSignals(True)
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.blockSignals(False)
            self.ui.liveCheckBox.setEnabled(True)
            self.ui.batchCheckBox.setEnabled(True)
            # Uncheck is signal-blocked → sync the flag (see stop()).
            self.live_mode = False
            # Re-assert per-mode widget state (cores/labels/toggles, viewer
            # dimming) now that the run lock is lifted.
            self._on_mode_changed()
        else:
            # Run active — keep Live clickable only for a *live* run (so it
            # can be toggled off to stop); mode toggles stay locked until the
            # run finishes.
            self.ui.liveCheckBox.setEnabled(self.live_mode)
            self.ui.batchCheckBox.setEnabled(False)

    def stylize_ParameterTree(self):
        self.tree.setStyleSheet("""
        QTreeView::item:has-children {
            background-color: #44475a;
            color: #f8f8f2;
        }
        QTreeView::item:has-children:disabled {
            background-color: #3a3d4d;
            color: #6272a4;
        }
            """)
