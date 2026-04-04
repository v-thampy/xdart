# -*- coding: utf-8 -*-
"""
@author: thampy
"""

# Standard library imports
import logging
import os
import sys
import subprocess

logger = logging.getLogger(__name__)

import fabio
from xdart.utils.pyFAI_binaries import pyFAI_drawmask_main, get_MaskImageWidgetXdart

# Qt imports
import pyqtgraph as pg
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtGui: Any = None
    QtWidgets: Any = None
    QtCore: Any = None
    class Qt:
        QtCore: Any
        QtGui: Any
        QtWidgets: Any
else:
    from pyqtgraph import Qt
    from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
from pyqtgraph.parametertree import Parameter

# This module imports
from .ui.integratorUI import Ui_Form

from .sphere_threads import integratorThread
from xdart.utils.session import save_session, load_session

_translate = QtCore.QCoreApplication.translate
QFileDialog = QtWidgets.QFileDialog

AA_inv = u'\u212B\u207B\u00B9'
Th = u'\u03B8'
Deg = u'\u00B0'
Units = [f"Q ({AA_inv})", f"2{Th} ({Deg})"]
Units_dict = {Units[0]: 'q_A^-1', Units[1]: '2th_deg'}
# Units_dict_inv = {'q_A^-1': Units[0], '2th_deg': Units[1]}
Units_dict_inv = {'q_A^-1': 0, '2th_deg': 1}

Chi = u'\u03C7'

GI_MODES_1D = ['q_total', 'q_ip', 'q_oop', 'exit_angle']
GI_LABELS_1D = ["Q", "Q\u1D62\u209A", "Q\u2092\u2092\u209A", "Exit"]
GI_MODES_2D = ['qip_qoop', 'q_chi', 'exit_angles']
GI_LABELS_2D = [u"Q\u1D62\u209A\u2013Q\u2092\u2092\u209A", f"Q-{Chi}", "Exit"]

params = [
    {'name': 'Default', 'type': 'group', 'children': [
            {'name': 'Integrate 1D', 'type': 'group', 'children': [
                {'name': 'correctSolidAngle', 'type': 'bool', 'value': True},
                {'name': 'dummy', 'type': 'float', 'value': -1.0},
                {'name': 'delta_dummy', 'type': 'float', 'value': 0.0},
                {'name': 'chi_offset', 'type': 'float', 'value': 90.0},
                {'name': 'Apply polarization factor', 'type': 'bool', 'value': False},
                {'name': 'polarization_factor', 'type': 'float', 'value': 0,
                    'limits': (-1, 1)},
                {'name': 'method', 'type': 'list', 'values': [
                        "numpy", "cython", "BBox", "splitpixel", "lut", "csr",
                        "nosplit_csr", "full_csr", "lut_ocl", "csr_ocl"
                    ], 'value':'csr'},
                {'name': 'safe', 'type': 'bool', 'value': True},
                # {'name': 'block_size', 'type': 'int', 'value': 32},
                # {'name': 'profile', 'type': 'bool', 'value': False},
                ]
             },
            {'name': 'Integrate 2D', 'type': 'group', 'children': [
                {'name': 'correctSolidAngle', 'type': 'bool', 'value': True},
                {'name': 'dummy', 'type': 'float', 'value': -1.0},
                {'name': 'delta_dummy', 'type': 'float', 'value': 0.0},
                {'name': 'chi_offset', 'type': 'float', 'value': 90.0},
                {'name': 'Apply polarization factor', 'type': 'bool', 'value': False},
                {'name': 'polarization_factor', 'type': 'float', 'value': 0,
                    'limits': (-1, 1)},
                {'name': 'method', 'type': 'list', 'values': [
                        "numpy", "cython", "BBox", "splitpixel", "lut",
                        "csr", "lut_ocl", "csr_ocl"
                    ], 'value':'csr'},
                {'name': 'safe', 'type': 'bool', 'value': True}
                ]
             }
        ]

     },
]


class integratorTree(QtWidgets.QWidget):
    """Widget for controlling integration of loaded data. Presents basic
    parameters to the user in easy to control widgets, and also
    launches menus for more advanced options.

    attributes:
        advancedWidget1D, advancedWidget2D: advancedParameters, pop up
            windows with advanced parameters
        azimuthalRange2D, radialRange1D, radialRange2D: rangeWidget,
            widgets which control the integration ranges for 1D and 2D
            integration
        bai_1d_pars, bai_2d_pars: pyqtgraph parameters, children of
            main parameters attribute that hold parameters related to
            1D and 2D integration
        mg_1d_pars, mg_2d_pars: unused, hold paramters for multigeometry
            integration
        mg_pars: unused, holds parameters for setting up multigeometry
        ui: Ui_Form from qtdesigner

    methods:
        get_args: Gets requested parameters and converts them to args
            in EwaldSphere object.
        setEnabled: Enables integration and parameter modification.
        update: Grabs args from EwaldSphere object and sets all params
            to match.
    """
    def __init__(self, sphere, arch, file_lock,
                 arches, arch_ids, data_1d, data_2d, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.sphere = sphere
        self.arch = arch
        self.file_lock = file_lock
        self.arches = arches
        self.arch_ids = arch_ids
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.parameters = Parameter.create(
            name='integrator', type='group', children=params
        )
        self.bai_1d_pars = self.parameters.child('Default', 'Integrate 1D')
        self.bai_2d_pars = self.parameters.child('Default', 'Integrate 2D')
        self.mask_window = None

        # UI adjustments
        _translate = QtCore.QCoreApplication.translate
        self.ui.unit_1D.setItemText(0, _translate("Form", Units[0]))
        self.ui.unit_1D.setItemText(1, _translate("Form", Units[1]))
        self.ui.label_azim_1D.setText(f"{Chi} ({Deg})")

        self.ui.unit_2D.setItemText(0, _translate("Form", Units[0]))
        self.ui.unit_2D.setItemText(1, _translate("Form", Units[1]))
        self.ui.label_azim_2D.setText(f"{Chi} ({Deg})")
        self.ui.label_npts_1D.setText("Pts")
        self.ui.label_npts_2D.setText("Pts")

        # Set constraints on range inputs
        self._validate_ranges()

        # Set AutoRange Flags
        self.radial_autoRange_1D = self.ui.radial_autoRange_1D.isChecked()
        self.azim_autoRange_1D = self.ui.azim_autoRange_1D.isChecked()
        self.radial_autoRange_2D = self.ui.radial_autoRange_2D.isChecked()
        self.azim_autoRange_2D = self.ui.azim_autoRange_2D.isChecked()

        # Setup advanced parameters tree widget
        self.advancedWidget1D = advancedParameters(self.bai_1d_pars, 'bai_1d')
        self.advancedWidget2D = advancedParameters(self.bai_2d_pars, 'bai_2d')

        # Connect input parameter value signals
        self._connect_inp_signals()

        # Connect integrate and advanced button signals
        self.ui.advanced1D.clicked.connect(self.advancedWidget1D.show)
        self.ui.advanced2D.clicked.connect(self.advancedWidget2D.show)

        self.ui.integrate1D.clicked.connect(self.bai_1d)
        self.ui.integrate2D.clicked.connect(self.bai_2d)

        self.integrator_thread = integratorThread(
            self.sphere, self.arch, self.file_lock,
            self.arches, self.arch_ids, self.data_1d, self.data_2d
        )

        # Connect Calibrate and Mask Buttons
        self.ui.pyfai_calib.clicked.connect(self.run_pyfai_calib)
        self.ui.get_mask.clicked.connect(self.run_pyfai_drawmask)

        # Inject npts_oop_1D for GI mode
        self.ui.npts_oop_1D = QtWidgets.QLineEdit(self.ui.frame1D_header)
        self.ui.npts_oop_1D.setObjectName("npts_oop_1D")
        self.ui.npts_oop_1D.setMaximumSize(QtCore.QSize(55, 16777215))
        self.ui.npts_oop_1D.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.ui.horizontalLayout_3.insertWidget(6, self.ui.npts_oop_1D)

        self.ui.npts_oop_1D.textChanged.connect(self._get_npts_1D)

        self.ui.npts_oop_1D.hide()

        # GI radial-range label (replaces unit_1D combo when hidden)
        self.ui.gi_radial_label_1D = QtWidgets.QLabel(self.ui.frame1D_range)
        self.ui.gi_radial_label_1D.setMaximumSize(QtCore.QSize(90, 16777215))
        self.ui.gridLayout_1D.addWidget(self.ui.gi_radial_label_1D, 0, 0, 1, 1)
        self.ui.gi_radial_label_1D.hide()

        # GI radial-range label for 2D (replaces unit_2D combo when hidden)
        self.ui.gi_radial_label_2D = QtWidgets.QLabel(self.ui.frame2D_range)
        self.ui.gi_radial_label_2D.setMaximumSize(QtCore.QSize(90, 16777215))
        self.ui.gridLayout_2D.addWidget(self.ui.gi_radial_label_2D, 0, 0, 1, 1)
        self.ui.gi_radial_label_2D.hide()
        
        self.setEnabled()
        # self.set_image_units()

        # Connect GI mode selectors
        self.ui.axis1D.currentIndexChanged.connect(self._update_gi_mode_1d)
        self.ui.axis2D.currentIndexChanged.connect(self._update_gi_mode_2d)

        # Restore saved integrator settings from last session
        self._restore_from_session()

        # Auto-save integrator settings on any UI change
        self.ui.npts_1D.textChanged.connect(self._save_to_session)
        self.ui.npts_radial_2D.textChanged.connect(self._save_to_session)
        self.ui.npts_azim_2D.textChanged.connect(self._save_to_session)
        self.ui.unit_1D.currentIndexChanged.connect(self._save_to_session)
        self.ui.unit_2D.currentIndexChanged.connect(self._save_to_session)
        self.ui.axis1D.currentIndexChanged.connect(self._save_to_session)
        self.ui.axis2D.currentIndexChanged.connect(self._save_to_session)
        self.ui.radial_autoRange_1D.stateChanged.connect(self._save_to_session)
        self.ui.azim_autoRange_1D.stateChanged.connect(self._save_to_session)
        self.ui.radial_autoRange_2D.stateChanged.connect(self._save_to_session)
        self.ui.azim_autoRange_2D.stateChanged.connect(self._save_to_session)
        self.ui.radial_low_1D.textChanged.connect(self._save_to_session)
        self.ui.radial_high_1D.textChanged.connect(self._save_to_session)
        self.ui.azim_low_1D.textChanged.connect(self._save_to_session)
        self.ui.azim_high_1D.textChanged.connect(self._save_to_session)
        self.ui.radial_low_2D.textChanged.connect(self._save_to_session)
        self.ui.radial_high_2D.textChanged.connect(self._save_to_session)
        self.ui.azim_low_2D.textChanged.connect(self._save_to_session)
        self.ui.azim_high_2D.textChanged.connect(self._save_to_session)
        self.parameters.sigTreeStateChanged.connect(self._save_to_session)

    # --- Session persistence ---

    def _save_to_session(self, *args):
        """Save current integrator UI state to ~/.xdart/session.json."""
        data = {
            'integ_npts_1D': self.ui.npts_1D.text(),
            'integ_npts_radial_2D': self.ui.npts_radial_2D.text(),
            'integ_npts_azim_2D': self.ui.npts_azim_2D.text(),
            'integ_unit_1D': self.ui.unit_1D.currentIndex(),
            'integ_unit_2D': self.ui.unit_2D.currentIndex(),
            'integ_axis1D': self.ui.axis1D.currentIndex(),
            'integ_axis2D': self.ui.axis2D.currentIndex(),
            'integ_radial_autoRange_1D': self.ui.radial_autoRange_1D.isChecked(),
            'integ_azim_autoRange_1D': self.ui.azim_autoRange_1D.isChecked(),
            'integ_radial_autoRange_2D': self.ui.radial_autoRange_2D.isChecked(),
            'integ_azim_autoRange_2D': self.ui.azim_autoRange_2D.isChecked(),
            'integ_radial_low_1D': self.ui.radial_low_1D.text(),
            'integ_radial_high_1D': self.ui.radial_high_1D.text(),
            'integ_azim_low_1D': self.ui.azim_low_1D.text(),
            'integ_azim_high_1D': self.ui.azim_high_1D.text(),
            'integ_radial_low_2D': self.ui.radial_low_2D.text(),
            'integ_radial_high_2D': self.ui.radial_high_2D.text(),
            'integ_azim_low_2D': self.ui.azim_low_2D.text(),
            'integ_azim_high_2D': self.ui.azim_high_2D.text(),
        }
        # Save advanced parameter tree values
        for dim_label, tree in [('1d', self.bai_1d_pars), ('2d', self.bai_2d_pars)]:
            for child in tree.children():
                data[f'integ_{dim_label}_{child.name()}'] = child.value()
        save_session(data)

    def _restore_from_session(self):
        """Restore integrator UI state from ~/.xdart/session.json."""
        session = load_session()
        if not any(k.startswith('integ_') for k in session):
            return

        # Block signals during restoration to avoid feedback loops
        self._disconnect_inp_signals()
        try:
            self.ui.axis1D.currentIndexChanged.disconnect(self._update_gi_mode_1d)
        except TypeError:
            pass
        try:
            self.ui.axis2D.currentIndexChanged.disconnect(self._update_gi_mode_2d)
        except TypeError:
            pass

        # Restore npts
        for key, widget in [
            ('integ_npts_1D', self.ui.npts_1D),
            ('integ_npts_radial_2D', self.ui.npts_radial_2D),
            ('integ_npts_azim_2D', self.ui.npts_azim_2D),
        ]:
            val = session.get(key)
            if val is not None:
                widget.setText(str(val))

        # Restore unit combos
        for key, widget in [
            ('integ_unit_1D', self.ui.unit_1D),
            ('integ_unit_2D', self.ui.unit_2D),
        ]:
            val = session.get(key)
            if val is not None and 0 <= val < widget.count():
                widget.setCurrentIndex(val)

        # Restore axis combos (only if combo has enough items — depends on GI mode)
        for key, widget in [
            ('integ_axis1D', self.ui.axis1D),
            ('integ_axis2D', self.ui.axis2D),
        ]:
            val = session.get(key)
            if val is not None and 0 <= val < widget.count():
                widget.setCurrentIndex(val)

        # Restore auto-range checkboxes
        for key, widget in [
            ('integ_radial_autoRange_1D', self.ui.radial_autoRange_1D),
            ('integ_azim_autoRange_1D', self.ui.azim_autoRange_1D),
            ('integ_radial_autoRange_2D', self.ui.radial_autoRange_2D),
            ('integ_azim_autoRange_2D', self.ui.azim_autoRange_2D),
        ]:
            val = session.get(key)
            if val is not None:
                widget.setChecked(val)

        # Restore range text fields
        for key, widget in [
            ('integ_radial_low_1D', self.ui.radial_low_1D),
            ('integ_radial_high_1D', self.ui.radial_high_1D),
            ('integ_azim_low_1D', self.ui.azim_low_1D),
            ('integ_azim_high_1D', self.ui.azim_high_1D),
            ('integ_radial_low_2D', self.ui.radial_low_2D),
            ('integ_radial_high_2D', self.ui.radial_high_2D),
            ('integ_azim_low_2D', self.ui.azim_low_2D),
            ('integ_azim_high_2D', self.ui.azim_high_2D),
        ]:
            val = session.get(key)
            if val is not None:
                widget.setText(str(val))

        # Restore advanced parameter tree values
        for dim_label, tree in [('1d', self.bai_1d_pars), ('2d', self.bai_2d_pars)]:
            with tree.treeChangeBlocker():
                for child in tree.children():
                    val = session.get(f'integ_{dim_label}_{child.name()}')
                    if val is not None:
                        try:
                            child.setValue(val)
                        except (TypeError, ValueError, AttributeError) as e:
                            logger.debug("Failed to restore integrator parameter value %s: %s", key, e)

        # Update enabled/disabled state based on restored auto-range
        self.radial_autoRange_1D = self.ui.radial_autoRange_1D.isChecked()
        self.azim_autoRange_1D = self.ui.azim_autoRange_1D.isChecked()
        self.radial_autoRange_2D = self.ui.radial_autoRange_2D.isChecked()
        self.azim_autoRange_2D = self.ui.azim_autoRange_2D.isChecked()
        self.setEnabled()

        # Reconnect signals
        self.ui.axis1D.currentIndexChanged.connect(self._update_gi_mode_1d)
        self.ui.axis2D.currentIndexChanged.connect(self._update_gi_mode_2d)
        self._connect_inp_signals()

    def update(self):
        """Grabs args from sphere and uses _sync_ranges and
        _update_params private methods to update.

        args:
            sphere: EwaldSphere, object to get args from.
        """
        self._update_params()

    def setEnabled(self, enable=True):
        """Overrides parent class method. Ensures appropriate child
        widgets are enabled.

        args:
            enable: bool, If True widgets are enabled. If False
                they are disabled.
        """
        self.ui.frame1D.setEnabled(enable)
        self.ui.frame2D.setEnabled(enable)
        self.advancedWidget1D.setEnabled(enable)
        self.advancedWidget2D.setEnabled(enable)

        if self.radial_autoRange_1D:
            self.ui.radial_low_1D.setEnabled(False)
            self.ui.radial_high_1D.setEnabled(False)

        if self.azim_autoRange_1D:
            self.ui.azim_low_1D.setEnabled(False)
            self.ui.azim_high_1D.setEnabled(False)

        if self.radial_autoRange_2D:
            self.ui.radial_low_2D.setEnabled(False)
            self.ui.radial_high_2D.setEnabled(False)

        if self.azim_autoRange_2D:
            self.ui.azim_low_2D.setEnabled(False)
            self.ui.azim_high_2D.setEnabled(False)

        # self.ui.integrate1D.setEnabled(False)
        # self.ui.integrate2D.setEnabled(False)

    def _validate_ranges(self):
        self.ui.npts_1D.setValidator(QtGui.QIntValidator(0, 50000))
        self.ui.npts_radial_2D.setValidator(QtGui.QIntValidator(0, 50000))
        self.ui.npts_azim_2D.setValidator(QtGui.QIntValidator(0, 50000))

        # Allow negative radial values for GI modes (e.g. Q_ip can be negative)
        minmax = (-50, 50)
        if self.ui.unit_1D.currentIndex() == 1:
            minmax = (-180, 180)
        self.ui.radial_low_1D.setValidator(QtGui.QDoubleValidator(minmax[0], minmax[1], 2))
        self.ui.radial_high_1D.setValidator(QtGui.QDoubleValidator(minmax[0], minmax[1], 2))

        minmax = (-50, 50)
        if self.ui.unit_2D.currentIndex() == 1:
            minmax = (-180, 180)
        self.ui.radial_low_2D.setValidator(QtGui.QDoubleValidator(minmax[0], minmax[1], 2))
        self.ui.radial_high_2D.setValidator(QtGui.QDoubleValidator(minmax[0], minmax[1], 2))

        self.ui.azim_low_1D.setValidator(QtGui.QDoubleValidator(-180, 180, 2))
        self.ui.azim_high_1D.setValidator(QtGui.QDoubleValidator(-180, 180, 2))
        self.ui.azim_low_2D.setValidator(QtGui.QDoubleValidator(-180, 180, 2))
        self.ui.azim_high_2D.setValidator(QtGui.QDoubleValidator(-180, 180, 2))

    def _update_params(self):
        """Grabs args from sphere and syncs parameters with them.

        args:
            sphere: EwaldSphere, object to get args from.
        """
        self._disconnect_inp_signals()
        with self.sphere.sphere_lock:
            self._args_to_params(self.sphere.bai_1d_args, self.bai_1d_pars, dim='1D')
            self._args_to_params(self.sphere.bai_2d_args, self.bai_2d_pars, dim='2D')
        self._connect_inp_signals()

    def get_args(self, key):
        """Updates sphere with all parameters held in integrator.

        args:
            sphere: EwaldSphere, object to update
            key: str, which args to update.
        """
        with self.sphere.sphere_lock:
            if key == 'bai_1d':
                self._get_npts_1D()
                self._get_unit_1D()
                self._get_radial_range_1D()
                self._get_azim_range_1D()
                self._params_to_args(self.sphere.bai_1d_args, self.bai_1d_pars)

            elif key == 'bai_2d':
                self._get_npts_radial_2D()
                self._get_npts_azim_2D()
                self._get_unit_2D()
                self._get_radial_range_2D()
                self._get_azim_range_2D()
                self._params_to_args(self.sphere.bai_2d_args, self.bai_2d_pars)

    def _args_to_params(self, args, tree, dim='1D'):
        """Takes in args dictionary and sets all parameters in tree
        to match the args.

        args:
            args: dict, values to use for updating tree
            tree: pyqtgraph Parameter, parameters to update
        """
        if len(args) == 0:
            return

        with tree.treeChangeBlocker():
            for key, val in args.items():
                if key == 'radial_range':
                    if dim == '1D':
                        self._set_radial_range_1D()
                    else:
                        self._set_radial_range_2D()
                elif key == 'azimuth_range':
                    if dim == '1D':
                        self._set_azim_range_1D()
                    else:
                        self._set_azim_range_2D()
                elif key == 'unit':
                    if dim == '1D':
                        self._set_unit_1D()
                    else:
                        self._set_unit_2D()
                elif key == 'numpoints':
                    self._set_npts_1D()
                elif key == 'npt_rad':
                    self._set_npts_radial_2D()
                elif key == 'npt_azim':
                    self._set_npts_azim_2D()
                elif key == 'polarization_factor':
                    if val is None:
                        tree.child('Apply polarization factor').setValue(True)
                    else:
                        tree.child('Apply polarization factor').setValue(True)
                        tree.child(key).setValue(val)
                else:
                    try:
                        child = tree.child(key)
                    except (KeyError, AttributeError):
                        # No specific error thrown for missing child
                        child = None
                    if child is not None:
                        if val is None:
                            child.setValue('None')
                        else:
                            child.setValue(val)

    def _params_to_args(self, args, tree):
        """Sync advanced parameter tree values into the bai_args dict.

        Range keys (radial_range, azimuth_range, x_range, y_range) are
        **skipped** here because they are already set by the dedicated
        _get_*_range_*() helpers that read directly from the UI line-edits.
        The tree's Low/High children may be stale when the user edits the
        main integrator panel without opening the advanced dialog.

        Similarly, unit/numpoints/npt_rad/npt_azim are skipped because
        they are handled by _get_unit_*() and _get_npts_*() above.
        """
        _skip = {'radial_range', 'azimuth_range', 'x_range', 'y_range',
                 'unit', 'numpoints', 'npt_rad', 'npt_azim'}
        for child in tree.children():
            if child.name() in _skip:
                continue
            if 'range' in child.name():
                # Catch any other range-like keys from advanced tree
                continue
            elif child.name() == 'polarization_factor':
                pass
            elif child.name() == 'Apply polarization factor':
                if child.value():
                    args['polarization_factor'] = \
                        tree.child('polarization_factor').value()
                else:
                    args['polarization_factor'] = None
            else:
                val = child.value()
                if val == 'None':
                    args[child.name()] = None
                else:
                    args[child.name()] = val

    def _get_radial_range_1D(self):
        """Sets Sphere 1D radial range in bai_1d_args from UI values"""
        auto = self.ui.radial_autoRange_1D.isChecked()
        self.radial_autoRange_1D = auto

        _range = None
        if not auto:
            _range = self._get_valid_range(self.ui.radial_low_1D,
                                           self.ui.radial_high_1D)
        self.sphere.bai_1d_args['radial_range'] = _range

        self.ui.radial_low_1D.setEnabled(not auto)
        self.ui.radial_high_1D.setEnabled(not auto)

    def _set_radial_range_1D(self):
        """Sets UI values from Sphere 1D radial range in bai_1d_args"""
        self._disconnect_radial_range_1D_signals()

        _range = self.sphere.bai_1d_args['radial_range']
        if _range is None:
            self.ui.radial_autoRange_1D.setChecked(True)
            auto = True
        else:
            self.ui.radial_low_1D.setText(str(_range[0]))
            self.ui.radial_high_1D.setText(str(_range[1]))
            auto = False

        self.radial_autoRange_1D = auto
        self.ui.radial_low_1D.setEnabled(not auto)
        self.ui.radial_high_1D.setEnabled(not auto)

        self._connect_radial_range_1D_signals()

    def _get_azim_range_1D(self):
        """Sets Sphere 1D azimuth range in bai_1d_args from UI values"""
        auto = self.ui.azim_autoRange_1D.isChecked()
        self.azim_autoRange_1D = auto

        _range = None
        if not auto:
            _range = self._get_valid_range(self.ui.azim_low_1D,
                                           self.ui.azim_high_1D)
        self.sphere.bai_1d_args['azimuth_range'] = _range

        self.ui.azim_low_1D.setEnabled(not auto)
        self.ui.azim_high_1D.setEnabled(not auto)

    def _set_azim_range_1D(self):
        """Sets UI values from Sphere 1D azimuth range in bai_1d_args."""
        self._disconnect_azim_range_1D_signals()

        _range = self.sphere.bai_1d_args['azimuth_range']
        if _range is None:
            self.ui.azim_autoRange_1D.setChecked(True)
            auto = True
        else:
            self.ui.azim_low_1D.setText(str(_range[0]))
            self.ui.azim_high_1D.setText(str(_range[1]))
            auto = False

        self.azim_autoRange_1D = auto
        self.ui.azim_low_1D.setEnabled(not auto)
        self.ui.azim_high_1D.setEnabled(not auto)

        self._connect_azim_range_1D_signals()

    def _get_radial_range_2D(self):
        """Sets Sphere 2D radial range in bai_2d_args from UI values"""
        auto = self.ui.radial_autoRange_2D.isChecked()
        self.radial_autoRange_2D = auto

        _range = None
        if not auto:
            _range = self._get_valid_range(self.ui.radial_low_2D,
                                           self.ui.radial_high_2D)

        if self.sphere.gi and self.ui.axis2D.currentIndex() == 0:  # Qip vs Qoop
            self.sphere.bai_2d_args['x_range'] = _range
        else:
            self.sphere.bai_2d_args['radial_range'] = _range

        self.ui.radial_low_2D.setEnabled(not auto)
        self.ui.radial_high_2D.setEnabled(not auto)

    def _set_radial_range_2D(self):
        """Sets UI values from Sphere 2D radial range in bai_2d_args"""
        self._disconnect_radial_range_2D_signals()

        _range = self.sphere.bai_2d_args['radial_range']
        if _range is None:
            self.ui.radial_autoRange_2D.setChecked(True)
            auto = True
        else:
            self.ui.radial_low_2D.setText(str(_range[0]))
            self.ui.radial_high_2D.setText(str(_range[1]))
            auto = False

        self.radial_autoRange_2D = auto
        self.ui.radial_low_2D.setEnabled(not auto)
        self.ui.radial_high_2D.setEnabled(not auto)

        self._connect_radial_range_2D_signals()

    def _get_azim_range_2D(self):
        """Sets Sphere 2D azimuth range in bai_2d_args from UI values"""
        auto = self.ui.azim_autoRange_2D.isChecked()
        self.azim_autoRange_2D = auto

        _range = None
        if not auto:
            _range = self._get_valid_range(self.ui.azim_low_2D,
                                           self.ui.azim_high_2D)

        if self.sphere.gi and self.ui.axis2D.currentIndex() == 0:  # Qip vs Qoop
            self.sphere.bai_2d_args['y_range'] = _range
        else:
            self.sphere.bai_2d_args['azimuth_range'] = _range

        self.ui.azim_low_2D.setEnabled(not auto)
        self.ui.azim_high_2D.setEnabled(not auto)

    def _set_azim_range_2D(self):
        """Sets UI values from Sphere 2D azimuth range in bai_2d_args."""
        self._disconnect_azim_range_2D_signals()

        _range = self.sphere.bai_2d_args['azimuth_range']
        if _range is None:
            self.ui.azim_autoRange_2D.setChecked(True)
            auto = True
        else:
            self.ui.azim_low_2D.setText(str(_range[0]))
            self.ui.azim_high_2D.setText(str(_range[1]))
            auto = False

        self.azim_autoRange_2D = auto
        self.ui.azim_low_2D.setEnabled(not auto)
        self.ui.azim_high_2D.setEnabled(not auto)

        self._connect_azim_range_2D_signals()

    @staticmethod
    def _get_valid_range(low_widget, high_widget):
        """Read two QLineEdit widgets and return a (lo, hi) tuple of floats.

        Returns None if both fields are empty/whitespace-only, indicating
        that no user-specified range is available.  Individual unparseable
        values fall back to 0.0.
        """
        lo_text = low_widget.text().strip()
        hi_text = high_widget.text().strip()
        if not lo_text and not hi_text:
            return None
        try:
            lo = float(lo_text) if lo_text else 0.0
        except ValueError:
            lo = 0.0
        try:
            hi = float(hi_text) if hi_text else 0.0
        except ValueError:
            hi = 0.0
        return (lo, hi)

    def _get_unit_1D(self):
        val = self.ui.unit_1D.currentText()
        self.sphere.bai_1d_args['unit'] = Units_dict[val]
        self._validate_ranges()

    def _set_unit_1D(self):
        self.ui.unit_1D.currentTextChanged.disconnect(self._get_unit_1D)
        val = self.sphere.bai_1d_args['unit']
        self.ui.unit_1D.setCurrentIndex(Units_dict_inv[val])
        self.ui.unit_1D.currentTextChanged.connect(self._get_unit_1D)

    def _get_unit_2D(self):
        val = self.ui.unit_2D.currentText()
        self.sphere.bai_2d_args['unit'] = Units_dict[val]
        self._validate_ranges()

    def _set_unit_2D(self):
        self.ui.unit_2D.currentTextChanged.disconnect(self._get_unit_2D)
        val = self.sphere.bai_2d_args['unit']
        self.ui.unit_2D.setCurrentIndex(Units_dict_inv[val])
        # self.ui.unit_2D.setCurrentText(Units_dict_inv[val])
        self.ui.unit_2D.currentTextChanged.connect(self._get_unit_2D)

    def _get_npts_1D(self):
        val = self.ui.npts_1D.text()
        val = 500 if (not val) else int(val)
        self.sphere.bai_1d_args['numpoints'] = val

        if self.sphere.gi:
            val_oop = self.ui.npts_oop_1D.text()
            val_oop = 500 if (not val_oop) else int(val_oop)
            self.sphere.bai_1d_args['npt_oop'] = val_oop

    def _set_npts_1D(self):
        self.ui.npts_1D.textChanged.disconnect(self._get_npts_1D)
        if self.sphere.gi:
            self.ui.npts_oop_1D.textChanged.disconnect(self._get_npts_1D)
            
        val = str(self.sphere.bai_1d_args.get('numpoints', 500))
        self.ui.npts_1D.setText(val)
        if self.sphere.gi:
            val_oop = str(self.sphere.bai_1d_args.get('npt_oop', 500))
            self.ui.npts_oop_1D.setText(val_oop)
            self.ui.npts_oop_1D.textChanged.connect(self._get_npts_1D)
            
        self.ui.npts_1D.textChanged.connect(self._get_npts_1D)

    def _get_npts_radial_2D(self):
        val = self.ui.npts_radial_2D.text()
        val = 500 if (not val) else int(val)
        self.sphere.bai_2d_args['npt_rad'] = val

    def _set_npts_radial_2D(self):
        self.ui.npts_radial_2D.textChanged.disconnect(self._get_npts_radial_2D)
        val = str(self.sphere.bai_2d_args['npt_rad'])
        self.ui.npts_radial_2D.setText(val)
        self.ui.npts_radial_2D.textChanged.connect(self._get_npts_radial_2D)

    def _get_npts_azim_2D(self):
        val = self.ui.npts_azim_2D.text()
        val = 500 if (not val) else int(val)
        self.sphere.bai_2d_args['npt_azim'] = val

    def _set_npts_azim_2D(self):
        self.ui.npts_azim_2D.textChanged.disconnect(self._get_npts_azim_2D)
        val = str(self.sphere.bai_2d_args['npt_azim'])
        self.ui.npts_azim_2D.setText(val)
        self.ui.npts_azim_2D.textChanged.connect(self._get_npts_azim_2D)

    def _connect_inp_signals(self):
        """Connect signals for all input sphere bai parameters"""
        # Connect points and units signals
        self.ui.npts_1D.textChanged.connect(self._get_npts_1D)
        self.ui.unit_1D.currentTextChanged.connect(self._get_unit_1D)

        self.ui.npts_radial_2D.textChanged.connect(self._get_npts_radial_2D)
        self.ui.npts_azim_2D.textChanged.connect(self._get_npts_azim_2D)
        self.ui.unit_2D.currentTextChanged.connect(self._get_unit_2D)

        # Connect range signals
        self._connect_radial_range_1D_signals()
        self._connect_azim_range_1D_signals()
        self._connect_radial_range_2D_signals()
        self._connect_azim_range_2D_signals()

        # Connect advanced parameters signals
        self.advancedWidget1D.sigUpdateArgs.connect(self.get_args)
        self.advancedWidget2D.sigUpdateArgs.connect(self.get_args)

    def _disconnect_inp_signals(self):
        """Disconnect signals for all input sphere bai parameters"""
        # Disconnect points and units signals
        self.ui.npts_1D.textChanged.disconnect(self._get_npts_1D)
        if self.sphere.gi:
            try:
                self.ui.npts_oop_1D.textChanged.disconnect(self._get_npts_1D)
            except TypeError:
                pass
        self.ui.unit_1D.currentTextChanged.disconnect(self._get_unit_1D)
        self.ui.npts_radial_2D.textChanged.disconnect(self._get_npts_radial_2D)
        self.ui.npts_azim_2D.textChanged.disconnect(self._get_npts_azim_2D)
        self.ui.unit_2D.currentTextChanged.disconnect(self._get_unit_2D)

        # Disconnect range signals
        self._disconnect_radial_range_1D_signals()
        self._disconnect_azim_range_1D_signals()
        self._disconnect_radial_range_2D_signals()
        self._disconnect_azim_range_2D_signals()

        # Disconnect advanced parameters signals
        self.advancedWidget1D.sigUpdateArgs.disconnect(self.get_args)
        self.advancedWidget2D.sigUpdateArgs.disconnect(self.get_args)

    def _connect_radial_range_1D_signals(self):
        """Connect signals for radial 1D range"""
        self.ui.radial_low_1D.textChanged.connect(self._get_radial_range_1D)
        self.ui.radial_high_1D.textChanged.connect(self._get_radial_range_1D)
        self.ui.radial_autoRange_1D.stateChanged.connect(self._get_radial_range_1D)

    def _disconnect_radial_range_1D_signals(self):
        """Disconnect signals for radial 1D range"""
        self.ui.radial_low_1D.textChanged.disconnect(self._get_radial_range_1D)
        self.ui.radial_high_1D.textChanged.disconnect(self._get_radial_range_1D)
        self.ui.radial_autoRange_1D.stateChanged.disconnect(self._get_radial_range_1D)

    def _connect_azim_range_1D_signals(self):
        """Connect signals for azimuth 1D range"""
        self.ui.azim_low_1D.textChanged.connect(self._get_azim_range_1D)
        self.ui.azim_high_1D.textChanged.connect(self._get_azim_range_1D)
        self.ui.azim_autoRange_1D.stateChanged.connect(self._get_azim_range_1D)

    def _disconnect_azim_range_1D_signals(self):
        """Disconnect signals for azimuth 1D range"""
        self.ui.azim_low_1D.textChanged.disconnect(self._get_azim_range_1D)
        self.ui.azim_high_1D.textChanged.disconnect(self._get_azim_range_1D)
        self.ui.azim_autoRange_1D.stateChanged.disconnect(self._get_azim_range_1D)

    def _connect_radial_range_2D_signals(self):
        """Connect signals for radial 2D range"""
        self.ui.radial_low_2D.textChanged.connect(self._get_radial_range_2D)
        self.ui.radial_high_2D.textChanged.connect(self._get_radial_range_2D)
        self.ui.radial_autoRange_2D.stateChanged.connect(self._get_radial_range_2D)

    def _disconnect_radial_range_2D_signals(self):
        """Disconnect signals for radial 2D range"""
        self.ui.radial_low_2D.textChanged.disconnect(self._get_radial_range_2D)
        self.ui.radial_high_2D.textChanged.disconnect(self._get_radial_range_2D)
        self.ui.radial_autoRange_2D.stateChanged.disconnect(self._get_radial_range_2D)

    def _connect_azim_range_2D_signals(self):
        """Connect signals for azimuth 2D range"""
        self.ui.azim_low_2D.textChanged.connect(self._get_azim_range_2D)
        self.ui.azim_high_2D.textChanged.connect(self._get_azim_range_2D)
        self.ui.azim_autoRange_2D.stateChanged.connect(self._get_azim_range_2D)

    def _disconnect_azim_range_2D_signals(self):
        """Disconnect signals for azimuth 2D range"""
        self.ui.azim_low_2D.textChanged.disconnect(self._get_azim_range_2D)
        self.ui.azim_high_2D.textChanged.disconnect(self._get_azim_range_2D)
        self.ui.azim_autoRange_2D.stateChanged.disconnect(self._get_azim_range_2D)

    def bai_1d(self, q):
        """Uses the integrator_thread attribute to call bai_1d
        """
        with self.integrator_thread.lock:
            if len(self.sphere.arches.index) > 0:
                self.integrator_thread.method = 'bai_1d_all'
        self.data_1d.clear()
        self.setEnabled(False)
        if not self.integrator_thread.isRunning():
            self.integrator_thread.start()

    def bai_2d(self, q):
        """Uses the integrator_thread attribute to call bai_2d
        """
        with self.integrator_thread.lock:
            if len(self.sphere.arches.index) > 0:
                self.integrator_thread.method = 'bai_2d_all'
        self.data_2d.clear()
        self.setEnabled(False)
        if not self.integrator_thread.isRunning():
            self.integrator_thread.start()

    @staticmethod
    def run_pyfai_calib():
        # pyFAI_calib2_main()
        # launch(f'{current_directory}/pyFAI-calib2-xdart')
        # if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        #     current_directory = sys._MEIPASS
        #     launch(f'{current_directory}/pyFAI-calib2')
        # else:
        #     pyFAI_calib2_main()
        process = subprocess.run(['pyFAI-calib2'], check=True, shell=True,
                                 stdout=subprocess.PIPE, universal_newlines=True)
        _ = process.stdout

    # @staticmethod
    def run_pyfai_drawmask(self):
        filters = f'Images (*.tif *tiff)'
        processFile, _ = QFileDialog().getOpenFileName(
            filter=filters,
            caption='Choose Image File',
            options=QFileDialog.DontUseNativeDialog
        )
        if not os.path.exists(processFile):
            logger.info('No image chosen for mask creation')
            return

        MaskWidgetClass = get_MaskImageWidgetXdart()
        self.mask_window = MaskWidgetClass()
        self.mask_window.setWindowModality(QtCore.Qt.WindowModal)
        self.mask_window.show()

        image = fabio.open(processFile).data
        pyFAI_drawmask_main(self.mask_window, image, processFile)
        # pyFAI_drawmask_main(self.mask_window, processFile)

        # mask = self.mask_window.getSelectionMask()
        # postProcessId21([processFile], mask)

    def set_image_units(self):
        """Populate GI mode selectors and update range labels / npts widgets.

        In GI mode:
        - axis1D / axis2D combos show GI integration mode options
        - npts_oop_1D is shown (fibre module needs npt_ip + npt_oop)
        - Range labels change based on selected GI mode
        In standard mode:
        - Single-entry combos ("Radial" / "Q-Chi"), npts_oop_1D hidden
        - Standard Q / 2th / Chi labels
        """
        try:
            self.ui.axis1D.currentIndexChanged.disconnect(self._update_gi_mode_1d)
        except TypeError:
            pass
        try:
            self.ui.axis2D.currentIndexChanged.disconnect(self._update_gi_mode_2d)
        except TypeError:
            pass
        if not self.sphere.gi:
            # Populate axis combos with Q / 2θ options for standard mode
            self.ui.axis1D.clear()
            self.ui.axis1D.addItem(_translate("Form", Units[0]))   # Q (Å⁻¹)
            self.ui.axis1D.addItem(_translate("Form", Units[1]))   # 2θ (°)
            self.ui.axis2D.clear()
            self.ui.axis2D.addItem(_translate("Form", f"Q-{Chi}"))
            self.ui.axis2D.addItem(_translate("Form", f"2{Th}-{Chi}"))
            # Sync axis combos to current unit selection
            cur_unit_1d = self.sphere.bai_1d_args.get('unit', 'q_A^-1')
            self.ui.axis1D.setCurrentIndex(Units_dict_inv.get(cur_unit_1d, 0))
            cur_unit_2d = self.sphere.bai_2d_args.get('unit', 'q_A^-1')
            self.ui.axis2D.setCurrentIndex(Units_dict_inv.get(cur_unit_2d, 0))
            # Hide the separate unit combos — axis combos now drive the unit
            self.ui.unit_1D.hide()
            self.ui.unit_2D.hide()
            # Hide GI-specific npts and restore standard labels
            self.ui.npts_oop_1D.hide()
            self.ui.label_npts_1D.setText("Pts")
            self.ui.label_azim_1D.setText(f"{Chi} ({Deg})")
            self.ui.label_to2.setText("to")
            self.ui.label_to1.setText("to")
            # 2D standard labels
            self.ui.label_azim_2D.setText(f"{Chi} ({Deg})")
            self.ui.label_to2_2.setText("to")
            self.ui.label_to1_2.setText("to")
            # Update radial labels + range defaults for current unit
            self._update_standard_1d_label(self.ui.axis1D.currentIndex())
            self._update_standard_2d_label(self.ui.axis2D.currentIndex())
        else:
            self.ui.axis1D.clear()
            for label in GI_LABELS_1D:
                self.ui.axis1D.addItem(_translate("Form", label))
            self.ui.axis2D.clear()
            for label in GI_LABELS_2D:
                self.ui.axis2D.addItem(_translate("Form", label))
            # Show npts_oop_1D for fibre module (npt_ip x npt_oop)
            self.ui.npts_oop_1D.show()
            if not self.ui.npts_oop_1D.text():
                self.ui.npts_oop_1D.setText("1000")
            if not self.ui.npts_1D.text() or self.ui.npts_1D.text() == "3000":
                self.ui.npts_1D.setText("1000")
            self.ui.label_npts_1D.setText("Pts")
            # Sync axis combos to current sphere.bai_args GI mode
            gi_mode_1d = self.sphere.bai_1d_args.get('gi_mode_1d', 'q_total')
            gi_mode_2d = self.sphere.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
            idx_1d = GI_MODES_1D.index(gi_mode_1d) if gi_mode_1d in GI_MODES_1D else 0
            idx_2d = GI_MODES_2D.index(gi_mode_2d) if gi_mode_2d in GI_MODES_2D else 0
            self.ui.axis1D.setCurrentIndex(idx_1d)
            self.ui.axis2D.setCurrentIndex(idx_2d)

        self.ui.axis1D.currentIndexChanged.connect(self._update_gi_mode_1d)
        self.ui.axis2D.currentIndexChanged.connect(self._update_gi_mode_2d)
        self._update_gi_mode_1d(self.ui.axis1D.currentIndex())
        self._update_gi_mode_2d(self.ui.axis2D.currentIndex())

    def _set_range_defaults_1d(self, rad_lo, rad_hi, azim_lo=None, azim_hi=None):
        """Set default range text for 1D integration inputs and push to bai_args.

        If Auto is checked, defaults are written as placeholder text and
        radial_range is set to None.  If Auto is NOT checked, the user's
        existing text values are preserved and synced to bai_args.
        If azim_lo/azim_hi are None, azimuth auto-range is enabled.
        """
        self._disconnect_radial_range_1D_signals()
        self._disconnect_azim_range_1D_signals()
        if self.ui.radial_autoRange_1D.isChecked():
            # Auto mode: write defaults as placeholders, range = None
            self.ui.radial_low_1D.setText(str(rad_lo))
            self.ui.radial_high_1D.setText(str(rad_hi))
            self.sphere.bai_1d_args['radial_range'] = None
        else:
            # Manual mode: keep user's values, sync to sphere
            _range = self._get_valid_range(self.ui.radial_low_1D,
                                           self.ui.radial_high_1D)
            if _range is None:
                # Fields empty/invalid — populate with defaults
                self.ui.radial_low_1D.setText(str(rad_lo))
                self.ui.radial_high_1D.setText(str(rad_hi))
                _range = (rad_lo, rad_hi)
            self.sphere.bai_1d_args['radial_range'] = _range
        if azim_lo is not None and azim_hi is not None:
            if self.ui.azim_autoRange_1D.isChecked():
                self.ui.azim_low_1D.setText(str(azim_lo))
                self.ui.azim_high_1D.setText(str(azim_hi))
                self.sphere.bai_1d_args['azimuth_range'] = None
            else:
                _arange = self._get_valid_range(self.ui.azim_low_1D,
                                                self.ui.azim_high_1D)
                if _arange is None:
                    self.ui.azim_low_1D.setText(str(azim_lo))
                    self.ui.azim_high_1D.setText(str(azim_hi))
                    _arange = (azim_lo, azim_hi)
                self.sphere.bai_1d_args['azimuth_range'] = _arange
        else:
            self.ui.azim_autoRange_1D.setChecked(True)
            self.ui.azim_low_1D.setEnabled(False)
            self.ui.azim_high_1D.setEnabled(False)
            self.sphere.bai_1d_args['azimuth_range'] = None
        self._connect_radial_range_1D_signals()
        self._connect_azim_range_1D_signals()

    def _set_range_defaults_2d(self, rad_lo, rad_hi, azim_lo=None, azim_hi=None):
        """Set default range text for 2D integration inputs and push to bai_args.

        If Auto is checked, defaults are written as placeholder text and
        radial_range is set to None.  If Auto is NOT checked, the user's
        existing text values are preserved and synced to bai_args.
        If azim_lo/azim_hi are None, azimuth auto-range is enabled.
        """
        self._disconnect_radial_range_2D_signals()
        self._disconnect_azim_range_2D_signals()
        if self.ui.radial_autoRange_2D.isChecked():
            self.ui.radial_low_2D.setText(str(rad_lo))
            self.ui.radial_high_2D.setText(str(rad_hi))
            self.sphere.bai_2d_args['radial_range'] = None
        else:
            _range = self._get_valid_range(self.ui.radial_low_2D,
                                           self.ui.radial_high_2D)
            if _range is None:
                self.ui.radial_low_2D.setText(str(rad_lo))
                self.ui.radial_high_2D.setText(str(rad_hi))
                _range = (rad_lo, rad_hi)
            self.sphere.bai_2d_args['radial_range'] = _range
        if azim_lo is not None and azim_hi is not None:
            if self.ui.azim_autoRange_2D.isChecked():
                self.ui.azim_low_2D.setText(str(azim_lo))
                self.ui.azim_high_2D.setText(str(azim_hi))
                self.sphere.bai_2d_args['azimuth_range'] = None
            else:
                _arange = self._get_valid_range(self.ui.azim_low_2D,
                                                self.ui.azim_high_2D)
                if _arange is None:
                    self.ui.azim_low_2D.setText(str(azim_lo))
                    self.ui.azim_high_2D.setText(str(azim_hi))
                    _arange = (azim_lo, azim_hi)
                self.sphere.bai_2d_args['azimuth_range'] = _arange
        else:
            self.ui.azim_autoRange_2D.setChecked(True)
            self.ui.azim_low_2D.setEnabled(False)
            self.ui.azim_high_2D.setEnabled(False)
            self.sphere.bai_2d_args['azimuth_range'] = None
        self._connect_radial_range_2D_signals()
        self._connect_azim_range_2D_signals()

    def _update_gi_mode_1d(self, n):
        """Update 1D integration mode from axis1D combo selection.

        In GI mode, updates sphere.bai_1d_args['gi_mode_1d'] and adjusts
        range / unit labels.  In standard mode, switches between Q and 2θ.
        """
        if not self.sphere.gi:
            self._update_standard_1d_label(n)
            return
        mode = GI_MODES_1D[n] if n < len(GI_MODES_1D) else 'q_total'
        self.sphere.bai_1d_args['gi_mode_1d'] = mode
        if mode in ('q_ip', 'q_oop'):
            self.ui.unit_1D.hide()
            self.ui.gi_radial_label_1D.setText(f"IP ({AA_inv})")
            self.ui.gi_radial_label_1D.show()
            self.ui.label_azim_1D.setText(f"OOP ({AA_inv})")
            self._set_range_defaults_1d(-10, 10, 0, 5)
        elif mode == 'exit_angle':
            self.ui.unit_1D.hide()
            self.ui.gi_radial_label_1D.setText(f"IP ({AA_inv})")
            self.ui.gi_radial_label_1D.show()
            self.ui.label_azim_1D.setText(f"Exit ({Deg})")
            self._set_range_defaults_1d(-5, 5, 0, 90)
        else:  # q_total (polar)
            self.ui.unit_1D.show()
            self.ui.unit_1D.setEnabled(True)
            self.ui.gi_radial_label_1D.hide()
            self.ui.label_azim_1D.setText(f"{Chi} ({Deg})")
            if self.ui.unit_1D.currentIndex() == 1:  # 2th
                self._set_range_defaults_1d(0, 90, -180, 180)
            else:  # Q
                self._set_range_defaults_1d(0, 5, -180, 180)

    def _update_gi_mode_2d(self, n):
        """Update 2D integration mode from axis2D combo selection.

        In GI mode, updates sphere.bai_2d_args['gi_mode_2d'] and adjusts
        range / unit labels.  In standard mode, switches between Q-χ and 2θ-χ.
        """
        if not self.sphere.gi:
            self._update_standard_2d_label(n)
            return
        mode = GI_MODES_2D[n] if n < len(GI_MODES_2D) else 'qip_qoop'
        self.sphere.bai_2d_args['gi_mode_2d'] = mode
        if mode == 'qip_qoop':
            self.ui.unit_2D.hide()
            self.ui.gi_radial_label_2D.setText(f"IP ({AA_inv})")
            self.ui.gi_radial_label_2D.show()
            self.ui.label_azim_2D.setText(f"OOP ({AA_inv})")
            self._set_range_defaults_2d(-10, 10, 0, 5)
        elif mode == 'q_chi':
            self.ui.unit_2D.show()
            self.ui.unit_2D.setEnabled(True)
            self.ui.gi_radial_label_2D.hide()
            self.ui.label_azim_2D.setText(f"{Chi} ({Deg})")
            if self.ui.unit_2D.currentIndex() == 1:  # 2th
                self._set_range_defaults_2d(0, 90, -180, 180)
            else:  # Q
                self._set_range_defaults_2d(0, 5, -180, 180)
        else:  # exit_angles
            self.ui.unit_2D.hide()
            self.ui.gi_radial_label_2D.setText(f"IP ({AA_inv})")
            self.ui.gi_radial_label_2D.show()
            self._set_range_defaults_2d(-5, 5, 0, 90)
            self.ui.label_azim_2D.setText(f"Exit ({Deg})")

    def _update_standard_1d_label(self, n):
        """Update 1D radial label and unit when axis1D changes in standard mode."""
        if n == 1:  # 2θ
            unit = '2th_deg'
            label = f"2{Th} ({Deg})"
            self._set_range_defaults_1d(0, 90, -180, 180)
        else:  # Q
            unit = 'q_A^-1'
            label = f"Q ({AA_inv})"
            self._set_range_defaults_1d(0, 5, -180, 180)
        self.sphere.bai_1d_args['unit'] = unit
        # Sync the hidden unit_1D combo so _get_unit_1D / _set_unit_1D stay consistent
        try:
            self.ui.unit_1D.currentTextChanged.disconnect(self._get_unit_1D)
        except TypeError:
            pass
        self.ui.unit_1D.setCurrentIndex(n)
        self.ui.unit_1D.currentTextChanged.connect(self._get_unit_1D)
        # Update the radial range label on the 1D panel
        self.ui.gi_radial_label_1D.setText(label)
        self.ui.gi_radial_label_1D.show()

    def _update_standard_2d_label(self, n):
        """Update 2D radial label and unit when axis2D changes in standard mode."""
        if n == 1:  # 2θ-χ
            unit = '2th_deg'
            label = f"2{Th} ({Deg})"
            self._set_range_defaults_2d(0, 90, -180, 180)
        else:  # Q-χ
            unit = 'q_A^-1'
            label = f"Q ({AA_inv})"
            self._set_range_defaults_2d(0, 5, -180, 180)
        self.sphere.bai_2d_args['unit'] = unit
        # Sync the hidden unit_2D combo
        try:
            self.ui.unit_2D.currentTextChanged.disconnect(self._get_unit_2D)
        except TypeError:
            pass
        self.ui.unit_2D.setCurrentIndex(n)
        self.ui.unit_2D.currentTextChanged.connect(self._get_unit_2D)
        # Update the radial range label on the 2D panel
        self.ui.gi_radial_label_2D.setText(label)
        self.ui.gi_radial_label_2D.show()


class advancedParameters(QtWidgets.QWidget):
    """Pop up window for setting more advanced integration parameters.

    attributes:
        name: str, name of the window
        parameter: pyqtgraph Parameter, parameters displayed
        tree: pyqtgraph ParameterTree, tree to hold parameter
        layout: QVBoxLayout, holds tree

    methods:
        process_change: Handles sigTreeStateChanged signal from tree

    signals:
        sigUpdateArgs: str, sends own name for updating args.
    """
    sigUpdateArgs = QtCore.Signal(str)

    def __init__(self, parameter, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.parameter = parameter
        self.tree = pg.parametertree.ParameterTree()
        self.tree.setMinimumWidth(150)
        self.tree.addParameters(parameter)
        self.parameter.sigTreeStateChanged.connect(self.process_change)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.tree)

    def process_change(self, tree, changes):
        self.sigUpdateArgs.emit(self.name)
