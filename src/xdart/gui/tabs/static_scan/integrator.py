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

from .scan_threads import integratorThread
from xdart.utils.session import save_session, load_session

_translate = QtCore.QCoreApplication.translate
QFileDialog = QtWidgets.QFileDialog

AA_inv = u'\u212B\u207B\u00B9'
Th = u'\u03B8'
Deg = u'\u00B0'
Chi = u'\u03C7'
# Int-1D radial-unit choices.  The third, chi (azimuthal profile / I vs chi), is a
# 1D-ONLY mode: the OUTPUT axis is chi while the range field is the q band integrated
# over (see frame.integrate_1d's chi_deg branch).  The 2D combo (unit_2D) keeps only
# Q / 2theta.
Units = [f"Q ({AA_inv})", f"2{Th} ({Deg})", f"{Chi} ({Deg})"]
Units_dict = {Units[0]: 'q_A^-1', Units[1]: '2th_deg', Units[2]: 'chi_deg'}
Units_dict_inv = {'q_A^-1': 0, '2th_deg': 1, 'chi_deg': 2}

GI_MODES_1D = ['q_total', 'q_ip', 'q_oop', 'exit_angle', 'chi_gi']
GI_LABELS_1D = ["Q", "Qip", "Qoop", "Exit", f"{Chi}GI"]
GI_MODES_2D = ['qip_qoop', 'q_chi', 'exit_angles']
GI_LABELS_2D = ["Qip-Qoop", f"Q-{Chi}", "Exit"]

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
            in LiveScan object.
        setEnabled: Enables integration and parameter modification.
        update: Grabs args from LiveScan object and sets all params
            to match.
    """
    # Emitted when the integrator's GI on/off toggle changes — staticWidget
    # connects it to update_scattering_geometry (sets scan.gi + refreshes the
    # panel), the same handler the wrangler's GI checkbox used to drive.
    sigUpdateGI = QtCore.Signal(bool)
    # Emitted with the saved mask-file path after Make Mask writes a mask —
    # staticWidget uses it to auto-populate the Mask File field.
    sigMaskCreated = QtCore.Signal(str)

    def __init__(self, scan, frame, file_lock,
                 frames, frame_ids, data_1d, data_2d, parent=None,
                 data_lock=None, publication_store=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.scan = scan
        self.frame = frame
        self.file_lock = file_lock
        self.frames = frames
        self.frame_ids = frame_ids
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.publication_store = publication_store
        # Shared lock guarding data_1d / data_2d; falls back to a private lock
        # when used stand-alone.
        import threading as _threading
        self.data_lock = data_lock if data_lock is not None else _threading.RLock()
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
        # The .ui declares two unit items; add the third (chi, azimuthal profile)
        # for the 1D combo only.  Idempotent across re-inits.
        if self.ui.unit_1D.count() < 3:
            self.ui.unit_1D.addItem("")
        self.ui.unit_1D.setItemText(2, _translate("Form", Units[2]))
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

        # Resurfaced Reintegrate row (the visible buttons; integrate1D/2D above
        # are the retained hidden stubs).  Two SEPARATE buttons so re-doing one
        # output never clobbers the other.  Advanced is re-homed onto advanced_int
        # in staticWidget (it owns the combined 1D+2D dialog).
        self.ui.reintegrate1D.clicked.connect(self.bai_1d)
        self.ui.reintegrate2D.clicked.connect(self.bai_2d)

        # Pixel-rejection (Intensity Threshold + Mask Saturated) lives in THIS
        # panel now (the integrator owns it).  ``get_threshold_config`` (a
        # method below) reads the row widgets fresh; it's the single source for
        # both Reintegrate (via _apply_threshold_config_to_thread) and live runs
        # (staticWidget injects it into the wrangler at run-setup).  Mask
        # Saturated default-on mirrors the historical wrangler default.
        self.ui.threshold_enable.toggled.connect(self._save_to_session)
        self.ui.threshold_min.textChanged.connect(self._save_to_session)
        self.ui.threshold_max.textChanged.connect(self._save_to_session)
        # Entering a non-default Min/Max means "I want to clip" — auto-enable the
        # Threshold toggle so the value actually applies, instead of silently
        # doing nothing until the separate toggle is also clicked (the recurring
        # "I set Max=1000 but it didn't clip" confusion; "Auto" is Mask Saturated,
        # NOT the threshold enable).
        self.ui.threshold_min.editingFinished.connect(
            self._maybe_autoenable_threshold)
        self.ui.threshold_max.editingFinished.connect(
            self._maybe_autoenable_threshold)
        self.ui.mask_saturated.toggled.connect(self._save_to_session)

        self.integrator_thread = integratorThread(
            self.scan, self.frame, self.file_lock,
            self.frames, self.frame_ids, self.data_1d, self.data_2d,
            data_lock=self.data_lock,
            publication_store=self.publication_store,
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

        # The .ui file caps label_azim_{1D,2D} at 40 px (fine for "χ (°)").
        # Widen so they can also display "OOP (Å⁻¹)" or "Exit (°)" without
        # clipping in GI mode.
        self.ui.label_azim_1D.setMinimumSize(QtCore.QSize(70, 0))
        self.ui.label_azim_1D.setMaximumSize(QtCore.QSize(80, 50))
        self.ui.label_azim_2D.setMinimumSize(QtCore.QSize(70, 0))
        self.ui.label_azim_2D.setMaximumSize(QtCore.QSize(80, 50))

        # ── GI geometry section (integrator-owned) ───────────────────────────
        # The integrator owns the GI geometry now (the wrangler's GI group is a
        # hidden carrier).  Mode lives in the 1D/2D headers (axis1D/axis2D); this
        # top row adds GI on/off + sample orientation + tilt + the incidence
        # motor.  get_gi_config() reads it; staticWidget injects it into the
        # wrangler at run-setup (live) and writes scan.gi_config (reintegrate),
        # exactly like the threshold row.
        self.ui.gi_frame = QtWidgets.QFrame(self)
        self.ui.gi_frame.setObjectName('gi_frame')
        _gi_lay = QtWidgets.QHBoxLayout(self.ui.gi_frame)
        _gi_lay.setContentsMargins(2, 0, 2, 0)
        _gi_lay.setSpacing(4)

        # The integrator panel's title sits at the LEFT of this header row; the
        # GI (Fiber) toggle + its "More" options button cluster at the RIGHT.
        self.ui.integration_heading = QtWidgets.QLabel('INTEGRATION')
        self.ui.integration_heading.setObjectName('integration_heading')
        _gi_lay.addWidget(self.ui.integration_heading)
        _gi_lay.addStretch(1)

        self.ui.gi_enable = QtWidgets.QPushButton('GI (Fiber)')
        self.ui.gi_enable.setObjectName('gi_enable')
        self.ui.gi_enable.setCheckable(True)
        self.ui.gi_enable.setMinimumWidth(90)
        _gi_lay.addWidget(self.ui.gi_enable)

        # Motor (incidence-angle source) + its manual-theta value are CONSTRUCTED
        # here but parented into a hidden holder (added below), NOT inline on the
        # row — the header stays just the title + GI toggle, and Controls Panel V2
        # renders the editable rows.  Every consumer (get_gi_config / session /
        # hydrate) reads these widget objects, so their location is display-only.
        self.ui.gi_motor_label = QtWidgets.QLabel('Motor')
        self.ui.gi_motor = QtWidgets.QComboBox()
        self.ui.gi_motor.setObjectName('gi_motor')
        self.ui.gi_motor.addItems(['th', 'Manual'])
        self.ui.gi_motor.setMaximumWidth(110)

        self.ui.gi_motor_value_label = QtWidgets.QLabel('Value')
        self.ui.gi_motor_value = QtWidgets.QLineEdit('0.1')
        self.ui.gi_motor_value.setObjectName('gi_motor_value')
        self.ui.gi_motor_value.setMaximumWidth(55)

        # Orient + Tilt no longer sit inline on the row — they are parented into
        # the hidden holder (below) and surfaced as editable rows by Controls
        # Panel V2.  Created here; every consumer (get_gi_config / session /
        # hydrate) reads these same widget objects, so moving them is
        # display-only.
        self.ui.gi_orient_label = QtWidgets.QLabel('Orient')
        self.ui.gi_sample_orientation = QtWidgets.QSpinBox()
        self.ui.gi_sample_orientation.setObjectName('gi_sample_orientation')
        self.ui.gi_sample_orientation.setMinimum(1)
        self.ui.gi_sample_orientation.setMaximum(8)
        self.ui.gi_sample_orientation.setValue(4)
        self.ui.gi_sample_orientation.setMaximumWidth(70)

        self.ui.gi_tilt_label = QtWidgets.QLabel('Tilt')
        self.ui.gi_tilt = QtWidgets.QLineEdit('0.0')
        self.ui.gi_tilt.setObjectName('gi_tilt')
        self.ui.gi_tilt.setMaximumWidth(70)

        # No separate "More" button and no popup: the GI detail fields render
        # inline in the Controls Panel V2 "Sample & measurement" subsection
        # (shown only when Grazing is selected).  The header row stays just the
        # INTEGRATION title (left) + the GI (Fiber) toggle (right).

        # GI section sits at the very top of the integrator panel.
        self.ui.verticalLayout.insertWidget(0, self.ui.gi_frame)

        # The GI detail widgets (motor / value / orient / tilt) are backing
        # state: get_gi_config / session restore / hydrate read these exact
        # objects, and the Controls Panel V2 rows write THROUGH to them.  They no
        # longer live in a floating popup (removed in favour of inline V2 rows) —
        # keep them ALIVE in a hidden holder so their object identity is
        # preserved without showing a separate window.
        self.ui.gi_hidden_holder = QtWidgets.QWidget(self)
        self.ui.gi_hidden_holder.setObjectName('gi_hidden_holder')
        self.ui.gi_hidden_holder.hide()
        _gi_hidden_form = QtWidgets.QFormLayout(self.ui.gi_hidden_holder)
        _gi_hidden_form.addRow(self.ui.gi_motor_label, self.ui.gi_motor)
        _gi_hidden_form.addRow(self.ui.gi_motor_value_label, self.ui.gi_motor_value)
        _gi_hidden_form.addRow(self.ui.gi_orient_label, self.ui.gi_sample_orientation)
        _gi_hidden_form.addRow(self.ui.gi_tilt_label, self.ui.gi_tilt)

        self.ui.gi_enable.toggled.connect(self._on_gi_toggled)
        self.ui.gi_enable.toggled.connect(self._save_to_session)
        self.ui.gi_sample_orientation.valueChanged.connect(self._save_to_session)
        self.ui.gi_tilt.textChanged.connect(self._save_to_session)
        self.ui.gi_motor.currentIndexChanged.connect(self._on_gi_motor_changed)
        self.ui.gi_motor.currentIndexChanged.connect(self._save_to_session)
        # `activated` fires only on an explicit user pick (not programmatic
        # setCurrentIndex), so it records the user's *deliberate* motor choice —
        # used to keep a chosen 'Manual' sticky across motor-list repopulation
        # while the initial default 'Manual' still yields to th/eta (F3).
        self.ui.gi_motor.activated.connect(self._on_gi_motor_user_pick)
        self._gi_motor_user_choice = None
        self.ui.gi_motor_value.textChanged.connect(self._save_to_session)

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
        # Auto-range toggles are checkable QPushButtons now → ``toggled``.
        self.ui.radial_autoRange_1D.toggled.connect(self._save_to_session)
        self.ui.azim_autoRange_1D.toggled.connect(self._save_to_session)
        self.ui.radial_autoRange_2D.toggled.connect(self._save_to_session)
        self.ui.azim_autoRange_2D.toggled.connect(self._save_to_session)
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
        """Save current integrator UI state to ~/.xdart/session.json.

        This slot is wired to widget / param-tree change signals.  Reading
        Qt widgets (``.text()``, ``.value()``, …) is only safe on the GUI
        thread, but a programmatic param-tree write that happens on a worker
        thread during a batch run (e.g. the GI range sync) would deliver
        these signals on that worker thread and run us there too — which is
        what produced the off-thread ``save_session`` crash.  Session
        persistence is a convenience, so if we're not on the GUI thread we
        skip rather than touch Qt off-thread.
        """
        if QtCore.QThread.currentThread() is not self.thread():
            logger.debug(
                "_save_to_session called off the GUI thread; skipping "
                "(unsafe to read Qt widgets here)"
            )
            return
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
            # Pixel-rejection policy (moved here from the wrangler).
            'integ_threshold_enable': self.ui.threshold_enable.isChecked(),
            'integ_threshold_min': self.ui.threshold_min.text(),
            'integ_threshold_max': self.ui.threshold_max.text(),
            'integ_mask_saturated': self.ui.mask_saturated.isChecked(),
            # GI geometry (moved here from the wrangler).
            'integ_gi_enable': self.ui.gi_enable.isChecked(),
            'integ_gi_sample_orientation': self.ui.gi_sample_orientation.value(),
            'integ_gi_tilt': self.ui.gi_tilt.text(),
            'integ_gi_motor': self.ui.gi_motor.currentText(),
            'integ_gi_motor_value': self.ui.gi_motor_value.text(),
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

        # Restore npts.  npts_1D deliberately NOT here: the per-axis
        # memory ('npts_1d' in the integrator session blob) owns it --
        # restoring raw text put a stale value (e.g. 1234) into whatever
        # axis happened to be active.
        for key, widget in [
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

        # Restore pixel-rejection policy (moved here from the wrangler).  Prefer
        # the integrator's own integ_ keys; fall back to the OLD flat wrangler
        # keys so an existing session's Threshold / Mask-Saturated settings carry
        # over.  (Mask Saturated stays default-ON when nothing is stored.)
        _thr_on = session.get('integ_threshold_enable')
        if _thr_on is None:
            _thr_on = session.get('apply_threshold')
        if _thr_on is not None:
            self.ui.threshold_enable.setChecked(bool(_thr_on))
        for ikey, okey, widget in [
            ('integ_threshold_min', 'threshold_min', self.ui.threshold_min),
            ('integ_threshold_max', 'threshold_max', self.ui.threshold_max),
        ]:
            val = session.get(ikey)
            if val is None:
                val = session.get(okey)
            if val is not None:
                widget.setText(str(val))
        _sat = session.get('integ_mask_saturated')
        if _sat is None:
            _sat = session.get('mask_sentinel')
        if _sat is not None:
            self.ui.mask_saturated.setChecked(bool(_sat))

        # Restore GI geometry (integrator-owned).  Block the toggle's signals so
        # restore can't fire _on_gi_toggled (sigUpdateGI) mid-init; sync the
        # reveal state + scan.gi manually after.
        self.ui.gi_enable.blockSignals(True)
        _gi_on = session.get('integ_gi_enable')
        if _gi_on is not None:
            self.ui.gi_enable.setChecked(bool(_gi_on))
        _so = session.get('integ_gi_sample_orientation')
        if _so is not None:
            try:
                self.ui.gi_sample_orientation.setValue(int(_so))
            except (TypeError, ValueError):
                pass
        _tilt = session.get('integ_gi_tilt')
        if _tilt is not None:
            self.ui.gi_tilt.setText(str(_tilt))
        _mot = session.get('integ_gi_motor')
        if _mot is not None:
            _mi = self.ui.gi_motor.findText(str(_mot))
            if _mi >= 0:
                self.ui.gi_motor.setCurrentIndex(_mi)
        _mv = session.get('integ_gi_motor_value')
        if _mv is not None:
            self.ui.gi_motor_value.setText(str(_mv))
        self.ui.gi_enable.blockSignals(False)
        self._update_gi_section_visibility()
        if _gi_on is not None and getattr(self, 'scan', None) is not None:
            self.scan.gi = bool(_gi_on)

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
        """Grabs args from scan and uses _sync_ranges and
        _update_params private methods to update.

        args:
            scan: LiveScan, object to get args from.
        """
        self._update_params()

    def _hydrate_gi_motor(self, gic):
        """Set the GI motor dropdown + manual theta from saved ``gi_config``.

        A numeric ``incidence_motor`` (old-build manual gi_config) or the literal
        'Manual' -> select Manual + the numeric theta (``th_val``, else the
        numeric ``incidence_motor``).  A real motor name -> select that motor."""
        mot = str(gic.get('incidence_motor', '') or '')
        thv = gic.get('th_val')
        try:
            _num = float(mot)
            manual = True
        except (TypeError, ValueError):
            _num = None
            manual = mot in ('', 'Manual')
        if manual:
            i = self.ui.gi_motor.findText('Manual')
            if i >= 0:
                self.ui.gi_motor.setCurrentIndex(i)
            val = thv if thv is not None else _num
            if val is not None:
                self.ui.gi_motor_value.setText(str(val))
            # A saved gi_config that says Manual is a deliberate choice — keep it
            # sticky so a later metadata/source repopulation doesn't auto-switch
            # it onto a file motor (F3).
            self._gi_motor_user_choice = 'Manual'
        else:
            i = self.ui.gi_motor.findText(mot)
            if i < 0:
                self.ui.gi_motor.addItem(mot)
                i = self.ui.gi_motor.findText(mot)
            if i >= 0:
                self.ui.gi_motor.setCurrentIndex(i)
            self._gi_motor_user_choice = mot

    @staticmethod
    def _hydrate_range(rng, low_w, high_w, auto_w):
        """A saved (lo, hi) range -> Auto OFF + the values; None/missing -> Auto ON."""
        if (rng is not None and len(rng) == 2
                and rng[0] is not None and rng[1] is not None):
            auto_w.setChecked(False)
            low_w.setText(str(rng[0]))
            high_w.setText(str(rng[1]))
        else:
            auto_w.setChecked(True)

    def hydrate_from_scan(self):
        """Stage C (2-way sync): populate the integration panel from the LOADED
        scan's saved reduction settings (``scan.bai_*_args`` + ``scan.gi_config``)
        so the panel SHOWS what produced the data and Reintegrate reproduces it
        by default.  No-op for a not-yet-loaded scan; all widget writes are
        signal-blocked so it can't fire a reintegrate / session-churn / arg
        re-derivation.  ``set_image_units`` rewrites the mode-default ranges into
        ``scan.bai_*_args``, so the saved ranges/npts are snapshot first and
        restored after."""
        scan = getattr(self, 'scan', None)
        if scan is None:
            return
        a1 = dict(getattr(scan, 'bai_1d_args', {}) or {})
        a2 = dict(getattr(scan, 'bai_2d_args', {}) or {})
        gic = dict(getattr(scan, 'gi_config', {}) or {})
        if not a1 and not a2 and not gic:
            return  # nothing saved to hydrate from (fresh/empty scan)

        # 1. GI section + scan.gi (the reload-Manual fix) — signal-blocked.
        gi_widgets = (self.ui.gi_enable, self.ui.gi_motor,
                      self.ui.gi_sample_orientation, self.ui.gi_tilt,
                      self.ui.gi_motor_value)
        for w in gi_widgets:
            w.blockSignals(True)
        try:
            gi_on = bool(getattr(scan, 'gi', False)) or bool(gic)
            self.ui.gi_enable.setChecked(gi_on)
            if gic:
                so = gic.get('sample_orientation')
                if so is not None:
                    try:
                        self.ui.gi_sample_orientation.setValue(int(so))
                    except (TypeError, ValueError):
                        pass
                tl = gic.get('tilt_angle')
                if tl is not None:
                    self.ui.gi_tilt.setText(str(tl))
                self._hydrate_gi_motor(gic)
            scan.gi = gi_on
        finally:
            for w in gi_widgets:
                w.blockSignals(False)

        # 2. Rebuild axis/unit combos + sync gi_mode from scan.bai_args (existing
        # sync).  NB: this rewrites mode-default ranges/npts into scan.bai_args.
        self.set_image_units()

        # 3. Restore the SAVED npts + ranges over those mode-defaults, in BOTH
        # the widgets and scan.bai_args, so the panel shows the saved reduction
        # and reintegrate reproduces it.
        range_widgets = (
            self.ui.npts_1D, self.ui.npts_oop_1D, self.ui.npts_radial_2D,
            self.ui.npts_azim_2D, self.ui.radial_low_1D, self.ui.radial_high_1D,
            self.ui.azim_low_1D, self.ui.azim_high_1D, self.ui.radial_autoRange_1D,
            self.ui.azim_autoRange_1D, self.ui.radial_low_2D, self.ui.radial_high_2D,
            self.ui.azim_low_2D, self.ui.azim_high_2D, self.ui.radial_autoRange_2D,
            self.ui.azim_autoRange_2D,
        )
        for w in range_widgets:
            w.blockSignals(True)
        try:
            with self.scan.scan_lock:
                for k in ('numpoints', 'npt_oop'):
                    if k in a1:
                        self.scan.bai_1d_args[k] = a1[k]
                for k in ('npt_rad', 'npt_azim'):
                    if k in a2:
                        self.scan.bai_2d_args[k] = a2[k]
                self.scan.bai_1d_args['radial_range'] = a1.get('radial_range')
                self.scan.bai_1d_args['azimuth_range'] = a1.get('azimuth_range')
                self.scan.bai_2d_args['radial_range'] = a2.get('radial_range')
                self.scan.bai_2d_args['azimuth_range'] = a2.get('azimuth_range')

            def _txt(widget, value):
                if value is not None:
                    try:
                        widget.setText(str(int(value)))
                    except (TypeError, ValueError):
                        widget.setText(str(value))

            _txt(self.ui.npts_1D, a1.get('numpoints', a1.get('npt')))
            _txt(self.ui.npts_oop_1D, a1.get('npt_oop'))
            _txt(self.ui.npts_radial_2D, a2.get('npt_rad', a2.get('npt')))
            _txt(self.ui.npts_azim_2D, a2.get('npt_azim'))
            self._hydrate_range(a1.get('radial_range'), self.ui.radial_low_1D,
                                self.ui.radial_high_1D, self.ui.radial_autoRange_1D)
            self._hydrate_range(a1.get('azimuth_range'), self.ui.azim_low_1D,
                                self.ui.azim_high_1D, self.ui.azim_autoRange_1D)
            self._hydrate_range(a2.get('radial_range'), self.ui.radial_low_2D,
                                self.ui.radial_high_2D, self.ui.radial_autoRange_2D)
            self._hydrate_range(a2.get('azimuth_range'), self.ui.azim_low_2D,
                                self.ui.azim_high_2D, self.ui.azim_autoRange_2D)
        finally:
            for w in range_widgets:
                w.blockSignals(False)

        self._update_gi_section_visibility()
        self.setEnabled(self.ui.frame1D.isEnabled())

    def setEnabled(self, enable=True):
        """Overrides parent class method. Ensures appropriate child
        widgets are enabled, with autorange fields disabled when auto is on.

        args:
            enable: bool, If True widgets are enabled. If False
                they are disabled.
        """
        self.ui.frame1D.setEnabled(enable)
        self.ui.frame2D.setEnabled(enable)
        self.advancedWidget1D.setEnabled(enable)
        self.advancedWidget2D.setEnabled(enable)

        # Disable range fields when their autorange checkbox is checked
        for cfg in self._RANGE_SIGNAL_CONFIGS:
            name, low, high, auto_name, _ = cfg
            is_auto = getattr(self.ui, auto_name).isChecked()
            if is_auto:
                getattr(self.ui, low).setEnabled(False)
                getattr(self.ui, high).setEnabled(False)

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
        """Grabs args from scan and syncs parameters with them.

        args:
            scan: LiveScan, object to get args from.
        """
        self._disconnect_inp_signals()
        with self.scan.scan_lock:
            self._args_to_params(self.scan.bai_1d_args, self.bai_1d_pars, dim='1D')
            self._args_to_params(self.scan.bai_2d_args, self.bai_2d_pars, dim='2D')
        self._connect_inp_signals()

    # ── Session persistence (panel fields + Advanced params) ─────────────
    _SESSION_UI_FIELDS = (
        'unit_1D',
        'radial_low_1D', 'radial_high_1D', 'azim_low_1D', 'azim_high_1D',
        'radial_autoRange_1D', 'azim_autoRange_1D',
        'unit_2D', 'npts_radial_2D', 'npts_azim_2D',
        'radial_low_2D', 'radial_high_2D', 'azim_low_2D', 'azim_high_2D',
        'radial_autoRange_2D', 'azim_autoRange_2D',
        'axis1D', 'axis2D',
    )

    def session_state(self) -> dict:
        """JSON-serializable snapshot of the integration panel: units, pts,
        ranges + Auto flags, GI mode combos, and the Advanced parameter
        tree.  Saved to session.json on app close; restored at startup."""
        ui = {}
        for name in self._SESSION_UI_FIELDS:
            w = getattr(self.ui, name, None)
            if w is None:
                continue
            if hasattr(w, 'isChecked'):
                ui[name] = bool(w.isChecked())
            elif hasattr(w, 'currentIndex'):
                ui[name] = int(w.currentIndex())
            elif hasattr(w, 'text'):
                ui[name] = str(w.text())
        try:
            advanced = self.parameters.saveState(filter='user')
        except Exception:
            advanced = None
        # Per-axis 1-D Pts memory (stash the live boxes first so the
        # active axis's values are included).
        self._stash_npts_1d()
        npts_mem = {k: list(v) for k, v in
                    getattr(self, '_npts_memory_1d', {}).items()}
        return {'ui': ui, 'advanced': advanced, 'npts_1d': npts_mem}

    def restore_session_state(self, data) -> None:
        if not isinstance(data, dict):
            return
        for name, val in (data.get('ui') or {}).items():
            w = getattr(self.ui, name, None)
            if w is None:
                continue
            try:
                if hasattr(w, 'setChecked'):
                    w.setChecked(bool(val))
                elif hasattr(w, 'setCurrentIndex'):
                    # Clamp: a saved index can exceed the combo's CURRENT
                    # item count (e.g. saved in GI mode, restored in
                    # standard mode) -- setCurrentIndex(-1/out-of-range)
                    # leaves currentText empty, which poisoned
                    # _get_unit_1D (KeyError: '').
                    _idx = int(val)
                    if 0 <= _idx < w.count():
                        w.setCurrentIndex(_idx)
                elif hasattr(w, 'setText'):
                    w.setText(str(val))
            except Exception:
                logger.debug("integrator restore skipped %s", name,
                             exc_info=True)
        npts_mem = data.get('npts_1d')
        if isinstance(npts_mem, dict):
            self._npts_memory_1d = {
                str(k): (str(v[0]), str(v[1]))
                for k, v in npts_mem.items()
                if isinstance(v, (list, tuple)) and len(v) == 2}
            # Re-apply for the CURRENT axis (forced: clear the key first).
            self._npts_key_1d = None
            self._apply_npts_1d_for_mode()
        advanced = data.get('advanced')
        if advanced:
            try:
                self.parameters.restoreState(
                    advanced, addChildren=False, removeChildren=False)
            except Exception:
                logger.debug("integrator advanced restore failed",
                             exc_info=True)
        # Re-derive bai args from the restored fields so the next run uses them.
        try:
            self.get_args('bai_1d')
            self.get_args('bai_2d')
        except Exception:
            logger.debug("integrator get_args after restore failed",
                         exc_info=True)

    def get_args(self, key):
        """Updates scan with all parameters held in integrator.

        args:
            scan: LiveScan, object to update
            key: str, which args to update.
        """
        with self.scan.scan_lock:
            if key == 'bai_1d':
                self._get_npts_1D()
                self._get_unit_1D()
                self._get_radial_range_1D()
                self._get_azim_range_1D()
                self._params_to_args(self.scan.bai_1d_args, self.bai_1d_pars)

            elif key == 'bai_2d':
                self._get_npts_radial_2D()
                self._get_npts_azim_2D()
                self._get_unit_2D()
                self._get_radial_range_2D()
                self._get_azim_range_2D()
                self._params_to_args(self.scan.bai_2d_args, self.bai_2d_pars)

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
                continue  # Handled via 'Apply polarization factor' toggle below
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

    # G3: the eight range-handler methods used to be hand-written
    # copies of the same pattern (read auto checkbox → toggle line
    # edits, push/pull the (low, high) tuple to/from bai_*_args
    # under a per-handler key).  Lifted into two table-driven
    # primitives below.  The Qip/Qoop key-swap in the 2D get is
    # preserved verbatim; the 2D set reads ``radial_range`` even
    # when GI is on (verified intentional — the UI line edits only
    # bind to ``radial_range`` regardless of axis mode).

    def _get_range_into_args(self, *, auto_cb, low_edit, high_edit,
                              args_dict, key, auto_attr,
                              gi_alt_key=None, after=None):
        """Read auto-checkbox + (low, high) line edits into args_dict.

        ``gi_alt_key`` (2D only): when set, and the scan is in GI
        mode with axis2D == Qip vs Qoop, store under that key instead
        of ``key``.  Pre-G3 this was a hand-written `if` in every
        2D ``_get_*`` method.

        ``after``: optional callable run after the dict + UI updates
        (e.g. ``_update_npts_oop_visibility_1d``).
        """
        auto = auto_cb.isChecked()
        setattr(self, auto_attr, auto)
        _range = None
        if not auto:
            _range = self._get_valid_range(low_edit, high_edit)
        effective_key = key
        if (gi_alt_key
                and self.scan.gi
                and self.ui.axis2D.currentIndex() == 0):
            effective_key = gi_alt_key
        args_dict[effective_key] = _range
        if gi_alt_key:
            stale_key = key if effective_key == gi_alt_key else gi_alt_key
            args_dict.pop(stale_key, None)
        low_edit.setEnabled(not auto)
        high_edit.setEnabled(not auto)
        if after is not None:
            after()

    def _set_range_from_args(self, *, auto_cb, low_edit, high_edit,
                              args_dict, key, auto_attr,
                              disconnect, connect, after=None):
        """Mirror of :meth:`_get_range_into_args`: push args_dict[key]
        back onto the UI widgets.

        ``disconnect`` / ``connect`` bracket the widget mutations so
        the line-edit change signals don't fire back into
        ``_get_range_into_args`` during the update.
        """
        disconnect()
        _range = args_dict[key]
        if _range is None:
            auto_cb.setChecked(True)
            auto = True
        else:
            low_edit.setText(str(_range[0]))
            high_edit.setText(str(_range[1]))
            auto = False
        setattr(self, auto_attr, auto)
        low_edit.setEnabled(not auto)
        high_edit.setEnabled(not auto)
        if after is not None:
            after()
        connect()

    def _get_radial_range_1D(self):
        """Sets Scan 1D radial range in bai_1d_args from UI values"""
        self._get_range_into_args(
            auto_cb=self.ui.radial_autoRange_1D,
            low_edit=self.ui.radial_low_1D,
            high_edit=self.ui.radial_high_1D,
            args_dict=self.scan.bai_1d_args,
            key='radial_range',
            auto_attr='radial_autoRange_1D',
        )

    def _set_radial_range_1D(self):
        """Sets UI values from Scan 1D radial range in bai_1d_args"""
        self._set_range_from_args(
            auto_cb=self.ui.radial_autoRange_1D,
            low_edit=self.ui.radial_low_1D,
            high_edit=self.ui.radial_high_1D,
            args_dict=self.scan.bai_1d_args,
            key='radial_range',
            auto_attr='radial_autoRange_1D',
            disconnect=self._disconnect_radial_range_1D_signals,
            connect=self._connect_radial_range_1D_signals,
        )

    def _get_azim_range_1D(self):
        """Sets Scan 1D azimuth range in bai_1d_args from UI values"""
        # Toggling auto can flip the q_total path between fast (1 Pts) and
        # slow (2 Pts); refresh the npts_oop_1D visibility after the
        # dict update lands.
        self._get_range_into_args(
            auto_cb=self.ui.azim_autoRange_1D,
            low_edit=self.ui.azim_low_1D,
            high_edit=self.ui.azim_high_1D,
            args_dict=self.scan.bai_1d_args,
            key='azimuth_range',
            auto_attr='azim_autoRange_1D',
            after=self._update_npts_oop_visibility_1d,
        )

    def _set_azim_range_1D(self):
        """Sets UI values from Scan 1D azimuth range in bai_1d_args."""
        self._set_range_from_args(
            auto_cb=self.ui.azim_autoRange_1D,
            low_edit=self.ui.azim_low_1D,
            high_edit=self.ui.azim_high_1D,
            args_dict=self.scan.bai_1d_args,
            key='azimuth_range',
            auto_attr='azim_autoRange_1D',
            disconnect=self._disconnect_azim_range_1D_signals,
            connect=self._connect_azim_range_1D_signals,
            after=self._update_npts_oop_visibility_1d,
        )

    def _get_radial_range_2D(self):
        """Sets Scan 2D radial range in bai_2d_args from UI values.

        GI Qip-vs-Qoop swaps the dict key from ``radial_range`` to
        ``x_range`` (handled by ``gi_alt_key``).
        """
        self._get_range_into_args(
            auto_cb=self.ui.radial_autoRange_2D,
            low_edit=self.ui.radial_low_2D,
            high_edit=self.ui.radial_high_2D,
            args_dict=self.scan.bai_2d_args,
            key='radial_range',
            auto_attr='radial_autoRange_2D',
            gi_alt_key='x_range',
        )

    def _set_radial_range_2D(self):
        """Sets UI values from Scan 2D radial range in bai_2d_args.

        Reads ``radial_range`` regardless of GI mode (the UI line
        edits only bind to that key; see the _get counterpart).
        """
        self._set_range_from_args(
            auto_cb=self.ui.radial_autoRange_2D,
            low_edit=self.ui.radial_low_2D,
            high_edit=self.ui.radial_high_2D,
            args_dict=self.scan.bai_2d_args,
            key='radial_range',
            auto_attr='radial_autoRange_2D',
            disconnect=self._disconnect_radial_range_2D_signals,
            connect=self._connect_radial_range_2D_signals,
        )

    def _get_azim_range_2D(self):
        """Sets Scan 2D azimuth range in bai_2d_args from UI values.

        GI Qip-vs-Qoop swaps to ``y_range``.
        """
        self._get_range_into_args(
            auto_cb=self.ui.azim_autoRange_2D,
            low_edit=self.ui.azim_low_2D,
            high_edit=self.ui.azim_high_2D,
            args_dict=self.scan.bai_2d_args,
            key='azimuth_range',
            auto_attr='azim_autoRange_2D',
            gi_alt_key='y_range',
        )

    def _set_azim_range_2D(self):
        """Sets UI values from Scan 2D azimuth range in bai_2d_args."""
        self._set_range_from_args(
            auto_cb=self.ui.azim_autoRange_2D,
            low_edit=self.ui.azim_low_2D,
            high_edit=self.ui.azim_high_2D,
            args_dict=self.scan.bai_2d_args,
            key='azimuth_range',
            auto_attr='azim_autoRange_2D',
            disconnect=self._disconnect_azim_range_2D_signals,
            connect=self._connect_azim_range_2D_signals,
        )

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
        if val in Units_dict:
            self.scan.bai_1d_args['unit'] = Units_dict[val]
        # else: empty/relabeled combo text (e.g. Share Axis toggled with no
        # data yet, or GI mode swaps the labels) -- keep the current arg.
        self._validate_ranges()

    def _set_unit_1D(self):
        self.ui.unit_1D.currentTextChanged.disconnect(self._get_unit_1D)
        val = self.scan.bai_1d_args['unit']
        self.ui.unit_1D.setCurrentIndex(Units_dict_inv[val])
        self.ui.unit_1D.currentTextChanged.connect(self._get_unit_1D)

    def _get_unit_2D(self):
        val = self.ui.unit_2D.currentText()
        if val in Units_dict:
            self.scan.bai_2d_args['unit'] = Units_dict[val]
        self._validate_ranges()

    def _set_unit_2D(self):
        self.ui.unit_2D.currentTextChanged.disconnect(self._get_unit_2D)
        val = self.scan.bai_2d_args['unit']
        self.ui.unit_2D.setCurrentIndex(Units_dict_inv[val])
        # self.ui.unit_2D.setCurrentText(Units_dict_inv[val])
        self.ui.unit_2D.currentTextChanged.connect(self._get_unit_2D)

    def _get_npts_1D(self):
        val = self.ui.npts_1D.text()
        val = 500 if (not val) else int(val)
        self.scan.bai_1d_args['numpoints'] = val
        # npts_oop_1D is only authoritative when visible: for GI modes that
        # use the fiber methods (q_ip / q_oop / exit_angle), or for q_total
        # with a restricted χ range.  When hidden, frame.py falls back to
        # npt_oop = numpoints.
        if self.scan.gi and self.ui.npts_oop_1D.isVisible():
            oop_val = self.ui.npts_oop_1D.text()
            oop_val = val if (not oop_val) else int(oop_val)
            self.scan.bai_1d_args['npt_oop'] = oop_val

    def _set_npts_1D(self):
        self.ui.npts_1D.textChanged.disconnect(self._get_npts_1D)
        if self.scan.gi:
            try:
                self.ui.npts_oop_1D.textChanged.disconnect(self._get_npts_1D)
            except TypeError:
                pass
        val = str(self.scan.bai_1d_args.get('numpoints', 500))
        self.ui.npts_1D.setText(val)
        if self.scan.gi:
            oop_val = str(self.scan.bai_1d_args.get('npt_oop',
                            self.scan.bai_1d_args.get('numpoints', 500)))
            self.ui.npts_oop_1D.setText(oop_val)
            self.ui.npts_oop_1D.textChanged.connect(self._get_npts_1D)
        self.ui.npts_1D.textChanged.connect(self._get_npts_1D)

    def _get_npts_radial_2D(self):
        val = self.ui.npts_radial_2D.text()
        val = 500 if (not val) else int(val)
        self.scan.bai_2d_args['npt_rad'] = val

    def _set_npts_radial_2D(self):
        self.ui.npts_radial_2D.textChanged.disconnect(self._get_npts_radial_2D)
        val = str(self.scan.bai_2d_args['npt_rad'])
        self.ui.npts_radial_2D.setText(val)
        self.ui.npts_radial_2D.textChanged.connect(self._get_npts_radial_2D)

    def _get_npts_azim_2D(self):
        val = self.ui.npts_azim_2D.text()
        val = 500 if (not val) else int(val)
        self.scan.bai_2d_args['npt_azim'] = val

    def _set_npts_azim_2D(self):
        self.ui.npts_azim_2D.textChanged.disconnect(self._get_npts_azim_2D)
        val = str(self.scan.bai_2d_args['npt_azim'])
        self.ui.npts_azim_2D.setText(val)
        self.ui.npts_azim_2D.textChanged.connect(self._get_npts_azim_2D)

    def _connect_inp_signals(self):
        """Connect signals for all input scan bai parameters"""
        # Connect points and units signals
        self.ui.npts_1D.textChanged.connect(self._get_npts_1D)
        self.ui.unit_1D.currentTextChanged.connect(self._get_unit_1D)

        self.ui.npts_radial_2D.textChanged.connect(self._get_npts_radial_2D)
        self.ui.npts_azim_2D.textChanged.connect(self._get_npts_azim_2D)
        self.ui.unit_2D.currentTextChanged.connect(self._get_unit_2D)

        # Connect range signals
        self._connect_all_range_signals()

        # Connect advanced parameters signals
        self.advancedWidget1D.sigUpdateArgs.connect(self.get_args)
        self.advancedWidget2D.sigUpdateArgs.connect(self.get_args)

    def _disconnect_inp_signals(self):
        """Disconnect signals for all input scan bai parameters"""
        # Disconnect points and units signals
        self.ui.npts_1D.textChanged.disconnect(self._get_npts_1D)
        if self.scan.gi:
            try:
                self.ui.npts_oop_1D.textChanged.disconnect(self._get_npts_1D)
            except TypeError:
                pass
        self.ui.unit_1D.currentTextChanged.disconnect(self._get_unit_1D)
        self.ui.npts_radial_2D.textChanged.disconnect(self._get_npts_radial_2D)
        self.ui.npts_azim_2D.textChanged.disconnect(self._get_npts_azim_2D)
        self.ui.unit_2D.currentTextChanged.disconnect(self._get_unit_2D)

        # Disconnect range signals
        self._disconnect_all_range_signals()

        # Disconnect advanced parameters signals
        self.advancedWidget1D.sigUpdateArgs.disconnect(self.get_args)
        self.advancedWidget2D.sigUpdateArgs.disconnect(self.get_args)

    # ── Data-driven range signal connect/disconnect ──────────────────
    # Each entry maps a range name to its (low_widget, high_widget,
    # auto_widget, getter_method) names.  _connect_range_signals and
    # _disconnect_range_signals iterate over these instead of 8
    # near-identical methods.

    _RANGE_SIGNAL_CONFIGS = [
        ('radial_1D', 'radial_low_1D', 'radial_high_1D',
         'radial_autoRange_1D', '_get_radial_range_1D'),
        ('azim_1D', 'azim_low_1D', 'azim_high_1D',
         'azim_autoRange_1D', '_get_azim_range_1D'),
        ('radial_2D', 'radial_low_2D', 'radial_high_2D',
         'radial_autoRange_2D', '_get_radial_range_2D'),
        ('azim_2D', 'azim_low_2D', 'azim_high_2D',
         'azim_autoRange_2D', '_get_azim_range_2D'),
    ]

    def _connect_range_signals(self, name):
        """Connect signals for a named range config (e.g. 'radial_1D')."""
        for cfg in self._RANGE_SIGNAL_CONFIGS:
            if cfg[0] == name:
                _, low, high, auto, getter = cfg
                handler = getattr(self, getter)
                getattr(self.ui, low).textChanged.connect(handler)
                getattr(self.ui, high).textChanged.connect(handler)
                getattr(self.ui, auto).toggled.connect(handler)
                return

    def _disconnect_range_signals(self, name):
        """Disconnect signals for a named range config (e.g. 'radial_1D')."""
        for cfg in self._RANGE_SIGNAL_CONFIGS:
            if cfg[0] == name:
                _, low, high, auto, getter = cfg
                handler = getattr(self, getter)
                getattr(self.ui, low).textChanged.disconnect(handler)
                getattr(self.ui, high).textChanged.disconnect(handler)
                getattr(self.ui, auto).toggled.disconnect(handler)
                return

    def _connect_all_range_signals(self):
        """Connect signals for all 4 range configs."""
        for cfg in self._RANGE_SIGNAL_CONFIGS:
            self._connect_range_signals(cfg[0])

    def _disconnect_all_range_signals(self):
        """Disconnect signals for all 4 range configs."""
        for cfg in self._RANGE_SIGNAL_CONFIGS:
            self._disconnect_range_signals(cfg[0])

    # Backwards-compatible aliases for existing callers
    def _connect_radial_range_1D_signals(self):
        self._connect_range_signals('radial_1D')

    def _disconnect_radial_range_1D_signals(self):
        self._disconnect_range_signals('radial_1D')

    def _connect_azim_range_1D_signals(self):
        self._connect_range_signals('azim_1D')

    def _disconnect_azim_range_1D_signals(self):
        self._disconnect_range_signals('azim_1D')

    def _connect_radial_range_2D_signals(self):
        self._connect_range_signals('radial_2D')

    def _disconnect_radial_range_2D_signals(self):
        self._disconnect_range_signals('radial_2D')

    def _connect_azim_range_2D_signals(self):
        self._connect_range_signals('azim_2D')

    def _disconnect_azim_range_2D_signals(self):
        self._disconnect_range_signals('azim_2D')

    def _reintegrate_is_live(self):
        """Reintegrate runs LIVE (per-frame, abortable) by default; the shared
        **Batch** toggle (StaticControls) switches it to the fast multicore path.
        The host installs ``_reintegrate_batch_provider`` (reads
        controls.batchButton); absent it (tests / standalone) we default to live."""
        prov = getattr(self, '_reintegrate_batch_provider', None)
        batch = bool(prov()) if callable(prov) else False
        return not batch

    def bai_1d(self, q):
        """Uses the integrator_thread attribute to call bai_1d
        """
        if self._block_if_no_frames('1D'):
            return
        if self._block_if_reload_only_frames('1D'):
            return
        if not self._ensure_reintegration_calibration('1D'):
            return
        self._apply_gi_config_to_scan()
        self._apply_threshold_config_to_thread()
        with self.integrator_thread.lock:
            if len(self.scan.frames.index) > 0:
                self.integrator_thread.method = 'bai_1d_all'
                self.integrator_thread.reintegrate_live = self._reintegrate_is_live()
        # N3: clear under data_lock to avoid racing with the
        # integrator thread's _publish or with the GUI's
        # _absorb_chunk arrivals (which also write through
        # data_lock).
        data_lock = getattr(self.integrator_thread, 'data_lock', None)
        if data_lock is not None:
            with data_lock:
                self.data_1d.clear()
        else:
            self.data_1d.clear()
        self.setEnabled(False)
        if not self.integrator_thread.isRunning():
            self.integrator_thread.start()

    def bai_2d(self, q):
        """Uses the integrator_thread attribute to call bai_2d
        """
        if self._block_if_no_frames('2D'):
            return
        if self._block_if_reload_only_frames('2D'):
            return
        if not self._ensure_reintegration_calibration('2D'):
            return
        self._apply_gi_config_to_scan()
        self._apply_threshold_config_to_thread()
        with self.integrator_thread.lock:
            if len(self.scan.frames.index) > 0:
                self.integrator_thread.method = 'bai_2d_all'
                self.integrator_thread.reintegrate_live = self._reintegrate_is_live()
        # N3: same data_lock discipline as bai_1d above.
        from .hydrated_raw import clear_hydrated_raw
        data_lock = getattr(self.integrator_thread, 'data_lock', None)
        if data_lock is not None:
            with data_lock:
                self.data_2d.clear()
                clear_hydrated_raw(self.data_2d)
        else:
            self.data_2d.clear()
            clear_hydrated_raw(self.data_2d)
        self.setEnabled(False)
        if not self.integrator_thread.isRunning():
            self.integrator_thread.start()

    # Default-select order for the GI incidence motor when the current selection
    # isn't in the loaded scan's motor list (case-insensitive).  Keep in sync
    # with image_wrangler._GI_MOTOR_PREFERENCE (the wrangler th_motor default).
    _GI_MOTOR_PREFERENCE = ('th', 'eta', 'theta', 'gonth', 'halpha')

    def set_gi_motor_options(self, motors):
        """Populate the GI motor dropdown from the active wrangler's available
        SPEC motor columns (Stage B).  Always offers 'Manual'; keeps the current
        selection if still present, else default-selects by ``_GI_MOTOR_PREFERENCE``
        (case-insensitive), then the first motor, then Manual."""
        motors = [str(m) for m in (motors or []) if str(m)]
        items = ['Manual'] + motors
        current = self.ui.gi_motor.currentText()
        self.ui.gi_motor.blockSignals(True)
        self.ui.gi_motor.clear()
        self.ui.gi_motor.addItems(items)
        # Selection precedence on repopulation:
        #  1. A DELIBERATE 'Manual' (user-picked or hydrated from a saved
        #     gi_config) is sticky — keep it so a source/format switch doesn't
        #     silently swap the user's manual incidence angle for a file motor
        #     (F3).  The initial *default* 'Manual' is not deliberate, so it
        #     falls through to the preference order below (auto-select th).
        #  2. Otherwise keep a still-present real motor the user already has.
        #  3. Else default by _GI_MOTOR_PREFERENCE, then first motor, then Manual.
        lower = {m.lower(): m for m in motors}
        if getattr(self, '_gi_motor_user_choice', None) == 'Manual':
            target = 'Manual'
        elif current in motors and current != 'Manual':
            target = current
        else:
            target = next((lower[p] for p in self._GI_MOTOR_PREFERENCE
                           if p in lower),
                          motors[0] if motors else 'Manual')
        idx = self.ui.gi_motor.findText(target)
        if idx >= 0:
            self.ui.gi_motor.setCurrentIndex(idx)
        self.ui.gi_motor.blockSignals(False)
        self._update_gi_section_visibility()
        self._save_to_session()

    def _on_gi_motor_user_pick(self, *args):
        """Record the user's explicit GI motor selection (fires only on a real
        user activation).  A deliberately-picked 'Manual' becomes sticky across
        later motor-list repopulations; picking a real motor clears that (F3)."""
        self._gi_motor_user_choice = str(self.ui.gi_motor.currentText())

    def _on_gi_motor_changed(self, *args):
        self._update_gi_section_visibility()

    def _update_gi_section_visibility(self):
        """Show the manual-theta Value field only when the motor is 'Manual'.

        The GI detail widgets live in a hidden holder and are surfaced as inline
        rows by Controls Panel V2, so there is no popup to show/hide here."""
        manual = (self.ui.gi_motor.currentText() == 'Manual')
        self.ui.gi_motor_value_label.setVisible(manual)
        self.ui.gi_motor_value.setVisible(manual)

    def _on_gi_toggled(self, checked):
        """GI on/off.  Drives ``scan.gi`` through the same
        ``update_scattering_geometry`` seam the wrangler checkbox used (sets
        scan.gi + refreshes the panel's axis units/labels).  The GI detail fields
        render inline in Controls Panel V2; there is no popup to open."""
        self._update_gi_section_visibility()
        scan = getattr(self, 'scan', None)
        if scan is not None:
            scan.gi = bool(checked)
        self.sigUpdateGI.emit(bool(checked))

    def get_gi_config(self):
        """The integrator's CURRENT GI geometry, read FRESH from the GI row.

        Single source of truth: a live run reads it via ``staticWidget`` at
        run-setup (injected into the wrangler's hidden GI carrier params) and
        Reintegrate reads it via ``_apply_gi_config_to_scan`` (writes scan.gi +
        scan.gi_config) — so both apply identical GI geometry.  Mode comes from
        the 1D/2D axis combos (already mirrored into ``scan.bai_*_args``)."""
        def _f(widget, default=0.0):
            try:
                return float(widget.text())
            except (TypeError, ValueError):
                return default
        return {
            'gi': bool(self.ui.gi_enable.isChecked()),
            'sample_orientation': int(self.ui.gi_sample_orientation.value()),
            'tilt_angle': _f(self.ui.gi_tilt, 0.0),
            'incidence_motor': str(self.ui.gi_motor.currentText()),
            'th_val': _f(self.ui.gi_motor_value, 0.0),
            'gi_mode_1d': self.scan.bai_1d_args.get('gi_mode_1d', 'q_total'),
            'gi_mode_2d': self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop'),
        }

    def _apply_gi_config_to_scan(self) -> None:
        """Before a Reintegrate, write the integrator's CURRENT GI geometry onto
        the scan so ``plan_from_live_scan`` reproduces it: ``scan.gi`` +
        ``scan.gi_config`` (the gi_mode keys already live in ``scan.bai_*_args``).
        Mirrors ``_apply_threshold_config_to_thread``."""
        try:
            cfg = self.get_gi_config()
        except Exception:
            logger.debug("could not read integrator GI config", exc_info=True)
            return
        scan = getattr(self, 'scan', None)
        if scan is None:
            return
        scan.gi = bool(cfg['gi'])
        if cfg['gi']:
            scan.gi_config = {
                'gi_mode_1d': str(cfg['gi_mode_1d']),
                'gi_mode_2d': str(cfg['gi_mode_2d']),
                'incidence_motor': str(cfg['incidence_motor'] or ''),
                # th_val (the Manual incidence angle) is the ONLY round-trip
                # source for a Manual scan (no motor metadata, no baked angle),
                # so persist it for reload + panel hydration (Stage C).
                'th_val': float(cfg['th_val'] or 0.0),
                'tilt_angle': float(cfg['tilt_angle'] or 0.0),
                'sample_orientation': int(cfg['sample_orientation'] or 1),
            }
            # Also write the GI geometry to direct scan attributes, mirroring the
            # live path's ``sync_live_scan_gi_settings`` — so a RELOADED scan
            # reintegrates with the panel's geometry.  The incidence source is
            # written so ``plan_from_live_scan`` resolves the SAME per-frame angle
            # live did:
            #   * a real MOTOR ('th', ...) -> write the NAME; plan leaves
            #     incident_angle=None and the per-frame angle baked from the
            #     motor metadata is used (the working th-motor path).
            #   * 'Manual' -> there is NO motor metadata AND no baked per-frame
            #     angle for this source, so write the NUMERIC theta value; plan
            #     float()s it into GIMode.incident_angle.  (Writing the literal
            #     'Manual' instead made the reduction raise "cannot resolve GI
            #     incident angle from metadata motor 'Manual'".)  The value comes
            #     from the GI-row theta field — set it to the real incidence.
            _motor = cfg['incidence_motor']
            _incidence = (str(cfg['th_val']) if _motor == 'Manual'
                          else str(_motor or ''))
            scan.incidence_motor = _incidence
            scan.th_mtr = _incidence
            scan.sample_orientation = int(cfg['sample_orientation'] or 1)
            scan.tilt_angle = float(cfg['tilt_angle'] or 0.0)
        else:
            scan.gi_config = {}

    def _maybe_autoenable_threshold(self):
        """Auto-enable the Intensity-Threshold toggle once the user sets a
        non-default Min/Max, so the value takes effect without a second click.
        Only ever ENABLES (never disables) — an explicit toggle-off is respected
        until the user edits a value again.  Default 0/0 (or blank) never enables
        (and Max=0 would mask everything, so we must not auto-enable it)."""
        def _f(widget):
            try:
                return float(widget.text())
            except (TypeError, ValueError):
                return 0.0
        if (_f(self.ui.threshold_min) != 0.0 or _f(self.ui.threshold_max) != 0.0):
            if not self.ui.threshold_enable.isChecked():
                self.ui.threshold_enable.setChecked(True)

    def get_threshold_config(self):
        """The integrator's CURRENT pixel-rejection policy (Intensity Threshold +
        Mask Saturated), read FRESH from the row widgets, as a
        ``ThresholdSaturationConfig``.

        Single source of truth: Reintegrate reads it here
        (``_apply_threshold_config_to_thread``) and a live run reads it via
        ``staticWidget`` at run-setup, so both apply identical pixel rejection.
        """
        from xdart.modules.reduction import ThresholdSaturationConfig

        def _num(widget, default=0.0):
            try:
                return float(widget.text())
            except (TypeError, ValueError):
                return default

        return ThresholdSaturationConfig(
            apply_threshold=bool(self.ui.threshold_enable.isChecked()),
            threshold_min=_num(self.ui.threshold_min, 0.0),
            threshold_max=_num(self.ui.threshold_max, 0.0),
            mask_saturation=bool(self.ui.mask_saturated.isChecked()),
        )

    def _apply_threshold_config_to_thread(self) -> None:
        """Snapshot the GUI's CURRENT Intensity-Threshold / Mask-Saturated policy
        onto the integrator thread before a reintegrate.

        Runs on the GUI thread (button click) and reads the wrangler params
        fresh via the host-supplied provider, so the reintegrate applies what the
        user has set RIGHT NOW (they tune these and re-run).  The thread reads
        ``threshold_config`` in ``_plan_for_reintegration`` after ``.start()`` —
        a clean happens-before, no cross-thread param read in the worker.
        """
        cfg = None
        provider = getattr(self, 'get_threshold_config', None)
        if callable(provider):
            try:
                cfg = provider()
            except Exception as e:        # host-supplied; never crash the click
                logger.debug("threshold-config provider errored: %s", e)
                cfg = None
        self.integrator_thread.threshold_config = cfg

    def _ensure_reintegration_calibration(self, dim_label: str) -> bool:
        """Make sure the scan has a detector-bearing integrator before a
        re-integration runs; surface a clear message if it can't.

        The geometry a re-integration uses comes entirely from the scan's *own*
        calibration — never the GUI's configured PONI File (that is for new
        scans).  A live run caches it on ``_cached_integrator`` /
        ``_cached_poni``; a scan reloaded from a ``.nxs`` restores it from the
        file's instrument/detector group (see
        ``LiveScan._load_from_nexus_v2``).  Only the integrator-panel params
        (npts / ranges / units) come from the GUI.

        If neither source produced a usable integrator — e.g. a ``.nxs`` written
        before calibration round-trip, which stored no detector identity — we
        surface a clear message instead of letting pyFAI crash deep inside
        calc_cartesian_positions (``self._pixel1 is None``).  Returns True iff a
        usable integrator is in place.
        """
        have = getattr(self.scan, '_cached_integrator', None) is not None
        logger.info("[REINTEGRATE-CAL] %s click: cached_integrator=%s "
                    "data_file=%s", dim_label, have,
                    getattr(self.scan, 'data_file', None))
        if not have:
            # Self-heal: re-read calibration from the scan's OWN .nxs on demand,
            # independent of how the GUI loaded it (a live data_only refresh does
            # not restore calibration).
            ensure = getattr(self.scan, 'ensure_calibration_loaded', None)
            if callable(ensure):
                have = bool(ensure())
                logger.info("[REINTEGRATE-CAL] %s click: after "
                            "ensure_calibration_loaded -> %s", dim_label, have)
        if have:
            return True
        # No stored calibration: a clear message beats a cryptic pyFAI crash.
        try:
            from pyqtgraph.Qt import QtWidgets
            QtWidgets.QMessageBox.information(
                self,
                f'Cannot re-integrate {dim_label}',
                'This scan has no stored calibration to re-integrate with.\n\n'
                'It was likely saved before the calibration was stored in the '
                '.nxs file — re-process the scan once, then Reintegrate will '
                'use the calibration read back from the file.',
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(
                'Re-integrate %s blocked — no stored calibration (no '
                'QMessageBox: %s)', dim_label, e)
        return False

    def _block_if_no_frames(self, dim_label: str) -> bool:
        """Abort re-integration with a status message when the scan has no
        processed frames to re-integrate.

        The Reintegrate buttons mirror the integration panels' enabled state
        (any non-viewer Int mode), so they're clickable even before a scan has
        been run/loaded — and in Image-Directory mode where the selected entry
        may have no corresponding processed ``.nxs`` yet.  Rather than gate the
        buttons on per-selection file probes (fragile, and the probe opens the
        .nxs), we let the click surface a clear message when there's nothing to
        do.  Returns True iff the action was blocked.
        """
        try:
            has_frames = len(self.scan.frames.index) > 0
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.debug("frame-count check errored, not blocking: %s", e)
            has_frames = True       # don't block on an unexpected scan shape
        if has_frames:
            return False
        try:
            from pyqtgraph.Qt import QtWidgets
            QtWidgets.QMessageBox.information(
                self,
                f'Nothing to re-integrate ({dim_label})',
                'No processed data was found for this scan to re-integrate.\n\n'
                'Run a scan, or load a processed .nxs file, then try again.',
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(
                'Re-integrate %s blocked — no processed frames (no QMessageBox: %s)',
                dim_label, e,
            )
        return True

    def _block_if_reload_only_frames(self, dim_label: str) -> bool:
        """R3 guardrail.  Abort re-integration with a status message when
        any of the scan's frames lacks a raw image AND no resolvable
        source file.

        After L1 wired lazy raw load, ``LiveFrame.integrate_*``
        auto-loads ``map_raw`` from ``frame.source_file`` /
        ``frame.source_frame_idx`` when it's missing — so this guard
        only fires when the source file has been moved/deleted
        relative to the .nxs.  See ``LiveScan.has_reload_only_frames``
        for the predicate.

        Returns True iff the action was blocked.
        """
        try:
            blocked = self.scan.has_reload_only_frames()
        except (AttributeError, RuntimeError) as e:
            # Scan torn down mid-call, or has_reload_only_frames
            # itself errored.  Don't block — the integrate path's
            # own guard will catch a genuinely empty map_raw.
            logger.debug("reload-only check errored, not blocking: %s", e)
            blocked = False
        if not blocked:
            return False
        try:
            from pyqtgraph.Qt import QtWidgets
            QtWidgets.QMessageBox.information(
                self,
                f'Cannot re-integrate {dim_label}',
                'This scan was reloaded from a .nxs file and its '
                'raw source images can no longer be found '
                '(frame.source_file did not resolve to an existing '
                'file).  Re-integration needs the raw frames.\n\n'
                'Make sure the original detector files are still in '
                'the location stored in the .nxs, or re-run the '
                'wrangler from a machine that can see them.',
            )
        except (ImportError, RuntimeError) as e:
            # In a headless test or no-Qt context the message box
            # can't pop — log the block instead.
            logger.warning(
                'Re-integrate %s blocked (no QMessageBox available: %s)',
                dim_label, e,
            )
        return True

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
        # Re-emit upward so staticWidget can auto-populate the Mask File field
        # once the user saves + closes the editor.
        self.mask_window.maskSaved.connect(self.sigMaskCreated)
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
        # GI ignores the transmission chi offset (the integration uses
        # FiberIntegrator's own polar convention) -- make the Advanced
        # panel's chi_offset reflect that instead of showing a live-looking
        # 90 that does nothing.
        _gi_now = bool(getattr(self.scan, 'gi', False))
        for _pars in (self.bai_1d_pars, self.bai_2d_pars):
            try:
                _p = _pars.child('chi_offset')
                _p.setReadonly(_gi_now)
                _p.setOpts(tip=('Ignored in grazing incidence '
                                '(transmission-only chi rotation)')
                           if _gi_now else '')
            except Exception:
                pass

        try:
            self.ui.axis1D.currentIndexChanged.disconnect(self._update_gi_mode_1d)
        except TypeError:
            pass
        try:
            self.ui.axis2D.currentIndexChanged.disconnect(self._update_gi_mode_2d)
        except TypeError:
            pass
        if not self.scan.gi:
            # Populate axis combos with Q / 2θ options for standard mode
            self.ui.axis1D.clear()
            self.ui.axis1D.addItem(_translate("Form", Units[0]))   # Q (Å⁻¹)
            self.ui.axis1D.addItem(_translate("Form", Units[1]))   # 2θ (°)
            self.ui.axis1D.addItem(_translate("Form", Units[2]))   # χ (°) azimuthal profile
            self.ui.axis2D.clear()
            self.ui.axis2D.addItem(_translate("Form", f"Q-{Chi}"))
            self.ui.axis2D.addItem(_translate("Form", f"2{Th}-{Chi}"))
            # Sync axis combos to current unit selection
            cur_unit_1d = self.scan.bai_1d_args.get('unit', 'q_A^-1')
            self.ui.axis1D.setCurrentIndex(Units_dict_inv.get(cur_unit_1d, 0))
            cur_unit_2d = self.scan.bai_2d_args.get('unit', 'q_A^-1')
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
            # Restore the 2th unit choice a GI visit removed.
            for _combo in (self.ui.unit_1D, self.ui.unit_2D):
                if _combo.count() < 2:
                    _combo.addItem(_translate("Form", Units[1]))
            # The 1D combo also offers chi (azimuthal profile, 1D-only); a GI visit
            # trimmed it to <=2 items, so restore the third here too.
            if self.ui.unit_1D.count() < 3:
                self.ui.unit_1D.addItem(_translate("Form", Units[2]))
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
            # 1-D Pts defaults are owned by _apply_npts_1d_for_mode (per-axis
            # memory) -- the legacy force-to-2000 snippet that lived here
            # clobbered the fiber-axis 1000 default at every new_scan and
            # poisoned the per-axis memory via the trailing stash.
            self.ui.label_npts_1D.setText("Pts")
            # GI has no 2th option: a unit retained from standard mode
            # (combo + bai args) integrated GI with unit='2th_deg' under a
            # Q-labelled axis -- silently wrong results.  Force Q on entry;
            # the trailing _update_gi_mode_* calls then re-derive labels and
            # range defaults from the Q unit.
            for _combo, _args in ((self.ui.unit_1D, self.scan.bai_1d_args),
                                  (self.ui.unit_2D, self.scan.bai_2d_args)):
                if _combo.currentIndex() != 0:
                    _combo.setCurrentIndex(0)   # Q (fires _get_unit_*)
                _args['unit'] = 'q_A^-1'        # belt-and-braces arg sync
                # GI offers no 2th at all -- remove the item (restored by
                # the standard branch on the way back).
                while _combo.count() > 1:
                    _combo.removeItem(_combo.count() - 1)
            # Sync axis combos to current scan.bai_args GI mode
            gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
            gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
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
            self.scan.bai_1d_args['radial_range'] = None
        else:
            # Manual mode: keep user's values, sync to scan
            _range = self._get_valid_range(self.ui.radial_low_1D,
                                           self.ui.radial_high_1D)
            if _range is None:
                # Fields empty/invalid — populate with defaults
                self.ui.radial_low_1D.setText(str(rad_lo))
                self.ui.radial_high_1D.setText(str(rad_hi))
                _range = (rad_lo, rad_hi)
            self.scan.bai_1d_args['radial_range'] = _range
        if azim_lo is not None and azim_hi is not None:
            if self.ui.azim_autoRange_1D.isChecked():
                self.ui.azim_low_1D.setText(str(azim_lo))
                self.ui.azim_high_1D.setText(str(azim_hi))
                self.scan.bai_1d_args['azimuth_range'] = None
            else:
                _arange = self._get_valid_range(self.ui.azim_low_1D,
                                                self.ui.azim_high_1D)
                if _arange is None:
                    self.ui.azim_low_1D.setText(str(azim_lo))
                    self.ui.azim_high_1D.setText(str(azim_hi))
                    _arange = (azim_lo, azim_hi)
                self.scan.bai_1d_args['azimuth_range'] = _arange
        else:
            self.ui.azim_autoRange_1D.setChecked(True)
            self.ui.azim_low_1D.setEnabled(False)
            self.ui.azim_high_1D.setEnabled(False)
            self.scan.bai_1d_args['azimuth_range'] = None
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
            self.scan.bai_2d_args['radial_range'] = None
        else:
            _range = self._get_valid_range(self.ui.radial_low_2D,
                                           self.ui.radial_high_2D)
            if _range is None:
                self.ui.radial_low_2D.setText(str(rad_lo))
                self.ui.radial_high_2D.setText(str(rad_hi))
                _range = (rad_lo, rad_hi)
            self.scan.bai_2d_args['radial_range'] = _range
        if azim_lo is not None and azim_hi is not None:
            if self.ui.azim_autoRange_2D.isChecked():
                self.ui.azim_low_2D.setText(str(azim_lo))
                self.ui.azim_high_2D.setText(str(azim_hi))
                self.scan.bai_2d_args['azimuth_range'] = None
            else:
                _arange = self._get_valid_range(self.ui.azim_low_2D,
                                                self.ui.azim_high_2D)
                if _arange is None:
                    self.ui.azim_low_2D.setText(str(azim_lo))
                    self.ui.azim_high_2D.setText(str(azim_hi))
                    _arange = (azim_lo, azim_hi)
                self.scan.bai_2d_args['azimuth_range'] = _arange
        else:
            self.ui.azim_autoRange_2D.setChecked(True)
            self.ui.azim_low_2D.setEnabled(False)
            self.ui.azim_high_2D.setEnabled(False)
            self.scan.bai_2d_args['azimuth_range'] = None
        self._connect_radial_range_2D_signals()
        self._connect_azim_range_2D_signals()

    # 1-D Pts defaults per integration axis: the fiber methods (q_ip /
    # q_oop / exit_angle) want a balanced (npt_ip, npt_oop) grid; q_total
    # ('Q'/'Chi') and the standard transmission axes use plain pyFAI
    # integrate1d, where 2000 is the house default.
    _NPTS_1D_DEFAULTS = {
        'q_ip': ('1000', '1000'),
        'q_oop': ('1000', '1000'),
        'exit_angle': ('1000', '1000'),
        'chi_gi': ('500', '1000'),       # (χ_GI output bins, q_total sampling)
        'q_total': ('2000', ''),
        'std': ('2000', ''),
    }

    def _npts_1d_mode_key(self):
        if not getattr(self.scan, 'gi', False):
            return 'std'
        return self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')

    def _stash_npts_1d(self):
        """Remember the current Pts boxes under the CURRENT axis key, so a
        user-chosen value survives axis switches."""
        key = getattr(self, '_npts_key_1d', None)
        if key is not None:
            if not hasattr(self, '_npts_memory_1d'):
                self._npts_memory_1d = {}
            self._npts_memory_1d[key] = (self.ui.npts_1D.text(),
                                         self.ui.npts_oop_1D.text())

    def _apply_npts_1d_for_mode(self):
        """Load the remembered (or default) Pts for the new axis key."""
        key = self._npts_1d_mode_key()
        if key == getattr(self, '_npts_key_1d', None):
            return
        self._npts_key_1d = key
        npt, oop = getattr(self, '_npts_memory_1d', {}).get(
            key, self._NPTS_1D_DEFAULTS.get(key, ('2000', '')))
        self.ui.npts_1D.setText(npt)
        if oop:
            self.ui.npts_oop_1D.setText(oop)
        elif self.ui.npts_oop_1D.isVisible():
            # visible-but-unset (q_total with a chi wedge): mirror npt
            self.ui.npts_oop_1D.setText(npt)

    def _update_gi_mode_1d(self, n):
        """Update 1D integration mode from axis1D combo selection.

        In GI mode, updates scan.bai_1d_args['gi_mode_1d'] and adjusts
        range / unit labels.  In standard mode, switches between Q and 2θ.
        """
        self._stash_npts_1d()
        if not self.scan.gi:
            self._update_standard_1d_label(n)
            self._apply_npts_1d_for_mode()
            return
        mode = GI_MODES_1D[n] if n < len(GI_MODES_1D) else 'q_total'
        self.scan.bai_1d_args['gi_mode_1d'] = mode
        if mode in ('q_ip', 'q_oop'):
            self.ui.unit_1D.hide()
            self.ui.gi_radial_label_1D.setText(f"Qip ({AA_inv})")
            self.ui.gi_radial_label_1D.show()
            self.ui.label_azim_1D.setText(f"Qoop ({AA_inv})")
            self._set_range_defaults_1d(-10, 10, 0, 5)
        elif mode == 'exit_angle':
            self.ui.unit_1D.hide()
            self.ui.gi_radial_label_1D.setText(f"Qip ({AA_inv})")
            self.ui.gi_radial_label_1D.show()
            self.ui.label_azim_1D.setText(f"Exit ({Deg})")
            self._set_range_defaults_1d(-5, 5, 0, 90)
        elif mode == 'chi_gi':
            # GI azimuthal profile: OUTPUT axis is χ_GI; the radial-range field is
            # the q_total BAND integrated over, the azim field clips χ_GI.
            self.ui.unit_1D.hide()
            self.ui.gi_radial_label_1D.setText(f"Q ({AA_inv})")
            self.ui.gi_radial_label_1D.show()
            self.ui.label_azim_1D.setText(f"{Chi}GI ({Deg})")
            self._set_range_defaults_1d(0, 5, -180, 180)
        else:  # q_total (polar)
            self.ui.unit_1D.show()
            self.ui.unit_1D.setEnabled(True)
            self.ui.gi_radial_label_1D.hide()
            self.ui.label_azim_1D.setText(f"{Chi} ({Deg})")
            if self.ui.unit_1D.currentIndex() == 1:  # 2th
                self._set_range_defaults_1d(0, 90, -180, 180)
            else:  # Q
                self._set_range_defaults_1d(0, 5, -180, 180)
        self._update_npts_oop_visibility_1d()
        self._apply_npts_1d_for_mode()

    def _update_npts_oop_visibility_1d(self):
        """Show/hide the second 1-D Pts box (npts_oop_1D).

        The q_total fast path uses pyFAI ``integrate1d`` (single npt) only
        when the χ range is unrestricted.  For every other GI 1-D case
        (modes q_ip / q_oop / exit_angle, or q_total with an explicit χ
        wedge), the underlying fiber methods need both npt_ip and npt_oop.
        Expose the second Pts field iff the slow path will be used.
        """
        if not self.scan.gi:
            self.ui.npts_oop_1D.hide()
            return
        mode = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        azim_auto = self.ui.azim_autoRange_1D.isChecked()
        show_oop = (mode != 'q_total') or (not azim_auto)
        if show_oop:
            if not self.ui.npts_oop_1D.text():
                self.ui.npts_oop_1D.setText(
                    self.ui.npts_1D.text() or "2000"
                )
            self.ui.npts_oop_1D.show()
        else:
            self.ui.npts_oop_1D.hide()

    def _update_gi_mode_2d(self, n):
        """Update 2D integration mode from axis2D combo selection.

        In GI mode, updates scan.bai_2d_args['gi_mode_2d'] and adjusts
        range / unit labels.  In standard mode, switches between Q-χ and 2θ-χ.
        """
        if not self.scan.gi:
            self._update_standard_2d_label(n)
            return
        mode = GI_MODES_2D[n] if n < len(GI_MODES_2D) else 'qip_qoop'
        self.scan.bai_2d_args['gi_mode_2d'] = mode
        if mode == 'qip_qoop':
            self.ui.unit_2D.hide()
            self.ui.gi_radial_label_2D.setText(f"Qip ({AA_inv})")
            self.ui.gi_radial_label_2D.show()
            self.ui.label_azim_2D.setText(f"Qoop ({AA_inv})")
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
            self.ui.gi_radial_label_2D.setText(f"Qip ({AA_inv})")
            self.ui.gi_radial_label_2D.show()
            self._set_range_defaults_2d(-5, 5, 0, 90)
            self.ui.label_azim_2D.setText(f"Exit ({Deg})")

    def _update_standard_1d_label(self, n):
        """Update 1D radial label and unit when axis1D changes in standard mode."""
        if n == 1:  # 2θ
            unit = '2th_deg'
            label = f"2{Th} ({Deg})"
            self._set_range_defaults_1d(0, 90, -180, 180)
        elif n == 2:  # χ (azimuthal profile): OUTPUT axis is χ, but the editable
            # radial-range field is the q BAND integrated over (always Q), so the
            # range label stays Q and the band defaults to a Q range.
            unit = 'chi_deg'
            label = f"Q ({AA_inv})"
            self._set_range_defaults_1d(0, 5, -180, 180)
        else:  # Q
            unit = 'q_A^-1'
            label = f"Q ({AA_inv})"
            self._set_range_defaults_1d(0, 5, -180, 180)
        self.scan.bai_1d_args['unit'] = unit
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
        self.scan.bai_2d_args['unit'] = unit
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
