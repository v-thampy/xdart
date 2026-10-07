# -*- coding: utf-8 -*-
"""The diffractometer geometry and the scan motor driving each circle, as one editor.

Every page that needs to know which motor is which mounts this (Stitch and its
calibration window now, RSM next). The value is
:class:`xrd_tools.session.instrument.InstrumentGeometry`. Each circle's row gives the axis it
turns about (psic's, or reversed for a circle turning the other way), its motor and its known
motor-zero reading (physical angle = motor readback − zero); a blank box gives none.
"""
from __future__ import annotations

import math
from dataclasses import replace

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.widgets.controls_panel import ControlsComboBox
from xrd_tools.session.instrument import (
    GEOMETRIES, GEOMETRY_LABELS, LAB_FRAME, PLANE_LABELS, PSIC_AXES, InstrumentGeometry, signed_axis,
)

#: Every circle any geometry has, in display order (a row each; a geometry shows its own).
_ALL_CIRCLES = ('mu', 'eta', 'th', 'chi', 'phi', 'nu', 'del', 'tth')
#: The detector arm's circles: their motor-zero readings seed the fits.
_ZERO_CIRCLES = ('nu', 'del', 'tth')
_ZERO_TIP = ("{circle}'s motor-zero reading (degrees): the motor's readback where the circle is at zero, "
             "so its physical angle = readback − this. The fits start here and stay within ± the zero "
             "freedom under 4 · Calibrate and fit ▸ Advanced. Blank: 0, or the seed calibration's own "
             "zero when refining from one.")
#: The sample's circles: their motor-zero readings correct the angles Stitch's grazing incidence reads
#: (maintainer, 2026-09-30: "a place to enter a zero offset for each angle ... set to 0 by default").
_SAMPLE_ZERO_TIP = ("{circle}'s motor-zero reading (degrees): the motor's readback where the circle is at "
                    "zero, so its physical angle = readback − this (grazing incidence reads it). Blank: 0.")
#: The axis box (maintainer, 2026-09-30: "the sense of rotation for each motor should be selectable";
#: "let's just stick with swapping the signs for now"; "default to standard").
_AXIS_TIP = ("The axis {circle} turns about, right-handed, for a positive angle: {standard} is psic's, "
             "{reverse} a circle that turns the other way. {readers}")
#: Who reads a circle's axis, by the page the editor serves (``used_by``; G6 choice 4): none claimed for a page
#: it does not know.
_AXIS_READERS = {"Stitch": {"detector": "Stitch's detector-arm fits and its xrayutilities file take it.",
                            "sample": "Stitch's grazing incidence takes it."},
                 "RSM": {"detector": "RSM's gridding (its xrayutilities geometry) takes it.",
                         "sample": "RSM's gridding (its xrayutilities geometry) takes it."}}


class GeometryEditor(QtWidgets.QWidget):
    """Choose the geometry; each circle's motor is an editable list of the scan's motors."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, *, used=(), used_by=''):
        super().__init__(parent)
        self._used, self._used_by = frozenset(used), str(used_by)
        self._form = QtWidgets.QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.geometry = ControlsComboBox()
        for key, label in GEOMETRY_LABELS.items():
            self.geometry.addItem(label, key)
        picker = QtWidgets.QWidget(self)
        picker_layout = QtWidgets.QHBoxLayout(picker)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.addWidget(self.geometry)
        self.default_senses = QtWidgets.QPushButton('↺', picker)
        self.default_senses.setToolTip("Restore the selected diffractometer's standard motor senses. "
                                      "Keep motor names, zero offsets and scattering plane.")
        self.default_senses.setFixedWidth(32)
        self.default_senses.clicked.connect(self._default_senses)
        picker_layout.addWidget(self.default_senses)
        picker_layout.addStretch(1)
        self._form.addRow('Diffractometer', picker)
        self.plane = ControlsComboBox()
        for key, label in PLANE_LABELS.items():
            self.plane.addItem(label, key)
        self.plane.setToolTip('The 2θ arm\'s scattering plane. Vertical: the detector turns about a '
                              'horizontal axis, as psic\'s del. Horizontal: about the vertical axis, as '
                              'psic\'s nu. The other psic detector circle is held at zero.')
        self._form.addRow('Scattering plane', self.plane)
        self._motors, self._zeros, self._rows, self._axes = {}, {}, {}, {}
        for circle in _ALL_CIRCLES:
            combo = ControlsComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            combo.setEditText(circle)
            combo.editTextChanged.connect(self._edited)
            self._motors[circle] = combo
            row = QtWidgets.QWidget()  # the axis, the motor, and beside it its zero
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            axis = ControlsComboBox()  # psic's axis or its reverse, filled by _show_rows
            axis.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            axis.currentIndexChanged.connect(self._edited)
            self._axes[circle] = axis
            layout.addWidget(axis)
            layout.addWidget(combo, 1)
            # every circle's zero: the arm's seed the fits, the sample's correct grazing incidence
            zero = QtWidgets.QLineEdit()
            zero.setPlaceholderText('0')
            zero.setValidator(QtGui.QDoubleValidator(-360., 360., 6, zero))
            zero.setMaximumWidth(72)
            zero.setToolTip((_ZERO_TIP if circle in _ZERO_CIRCLES else _SAMPLE_ZERO_TIP).format(circle=circle))
            zero.textChanged.connect(self._edited)
            label = QtWidgets.QLabel('zero (°)')
            label.setToolTip(zero.toolTip())
            layout.addWidget(label)
            layout.addWidget(zero)
            self._zeros[circle] = zero
            self._rows[circle] = row
            self._form.addRow(QtWidgets.QLabel(), row)
        # the frame the axes below are given in, and the geometry's standard axes (a small schematic can
        # replace this line later); _show_rows writes it
        self.frame_note = QtWidgets.QLabel()
        self.frame_note.setObjectName('geometryFrameNote')
        self.frame_note.setWordWrap(True)
        self._form.addRow(self.frame_note)
        self.geometry.currentIndexChanged.connect(self._geometry_changed)
        self.plane.currentIndexChanged.connect(self._geometry_changed)
        self._geometry_changed()

    def motor_edit(self, circle) -> QtWidgets.QLineEdit:
        """The text box of one circle's motor (a host may read or set it directly)."""
        return self._motors[circle].lineEdit()

    def _default_senses(self):
        self.set_value(replace(self.value(), reversed_circles=()))

    def zero_edit(self, circle) -> QtWidgets.QLineEdit:
        """The box of a detector circle's motor-zero reading (blank: none given)."""
        return self._zeros[circle]

    def _zero_text(self, circle):
        """A shown circle's zero: None when blank, a finite number, or the text itself when malformed."""
        edit = self._zeros.get(circle)
        text = edit.text().strip() if edit is not None else ''
        if not text:
            return None
        try:
            zero = float(text)
        except ValueError:
            return text
        return zero if math.isfinite(zero) else text

    def invalid_zeros(self) -> tuple[str, ...]:
        """The shown circles whose zero box holds text that is not a number (blank gives none): an
        operation refuses them rather than read them as blank (Codex review F4)."""
        return tuple(circle for circle, _role in GEOMETRIES[self.geometry.currentData()]
                     if isinstance(self._zero_text(circle), str))

    def value(self) -> InstrumentGeometry:
        geometry = self.geometry.currentData()
        circles = [circle for circle, _role in GEOMETRIES[geometry]]
        zeros = []
        for circle in circles:  # a blank or half-typed box gives none here; invalid_zeros() says which
            zero = self._zero_text(circle)
            if isinstance(zero, float):
                zeros.append((circle, zero))
        return InstrumentGeometry(geometry, tuple((circle, self._motors[circle].currentText()) for circle in circles),
                                  'vertical' if geometry == 'psic' else self.plane.currentData(), tuple(zeros),
                                  tuple(circle for circle in circles if self._axes[circle].currentData() == -1))

    def set_value(self, value: InstrumentGeometry):
        blockers = [QtCore.QSignalBlocker(widget)
                    for widget in (self.geometry, self.plane, *self._motors.values(), *self._zeros.values())]
        self.geometry.setCurrentIndex(max(0, self.geometry.findData(value.geometry)))
        self.plane.setCurrentIndex(max(0, self.plane.findData(value.plane)))
        for circle, motor in value.motors:
            self._motors[circle].setEditText(motor)
        for circle, edit in self._zeros.items():
            zero = value.zero(circle) if circle in value.circles else None
            edit.setText('' if zero is None else f'{zero:.12g}')
        del blockers
        self._show_rows()
        blockers = [QtCore.QSignalBlocker(combo) for combo in self._axes.values()]
        for circle, combo in self._axes.items():  # every circle a value does not reverse turns as psic's
            combo.setCurrentIndex(combo.findData(value.sense(circle) if circle in value.circles else 1))
        del blockers
        self.changed.emit()

    def set_available_motors(self, names):
        """List a scan's motors in every circle; a circle whose motor the scan lacks takes
        the standard name, an alias, or (incidence) the GI preference from the list."""
        names = tuple(str(name) for name in names or ())
        picked = InstrumentGeometry.pick(self.geometry.currentData(), names, keep=self.value())
        blockers = [QtCore.QSignalBlocker(combo) for combo in self._motors.values()]
        for circle, combo in self._motors.items():
            text = combo.currentText()
            combo.clear()
            combo.addItems(names)
            combo.setEditText(text)
        for circle, motor in picked.motors:
            self._motors[circle].setEditText(motor)
        del blockers
        self.changed.emit()

    def _geometry_changed(self, *_args):
        self._show_rows()
        self.changed.emit()

    def _show_rows(self):
        geometry = self.geometry.currentData()
        circles = dict(GEOMETRIES[geometry])
        self._form.setRowVisible(self.plane, geometry != 'psic')
        shown = InstrumentGeometry(geometry, plane='vertical' if geometry == 'psic' else self.plane.currentData())
        plays = shown.as_psic()
        # the selected geometry's standard axes (maintainer, 2026-09-30: "add the default directions for the
        # different motors as per standard psic, or fourc or twoc when they are selected")
        name = geometry if geometry == 'psic' else f'{geometry} ({shown.plane} plane)'
        standard = ', '.join(f'{circle} {axis}' for circle, axis in shown.circle_axes())
        self.frame_note.setText(f'Lab frame: {LAB_FRAME}. Each circle turns right-handed about the axis beside '
                                f'it for a positive angle; standard {name}: {standard}.')
        for circle, combo in self._motors.items():
            self._form.setRowVisible(self._rows[circle], circle in circles)
            label = self._form.labelForField(self._rows[circle])
            if circle in circles:
                used = f'  ·  used by {self._used_by}' if circle in self._used and self._used_by else ''
                label.setText(f'{circle} ({circles[circle]}){used}')
                combo.setToolTip(f'The scan motor that drives the {circle} circle'
                                 + (f" (it plays psic's {plays[circle]})" if circle in plays else ''))
                self._fill_axis(circle, PSIC_AXES[plays.get(circle, circle)], circles[circle])

    def _fill_axis(self, circle, standard, role):
        """The circle's axis box: psic's axis for the circle it plays, and its reverse; the sense
        chosen stays when the geometry or plane changes the axis."""
        combo = self._axes[circle]
        sense = combo.currentData() if combo.count() else 1
        blocker = QtCore.QSignalBlocker(combo)
        combo.clear()
        for value in (1, -1):
            combo.addItem(signed_axis(standard, value), value)
        combo.setCurrentIndex(combo.findData(sense))
        del blocker
        combo.setToolTip(_AXIS_TIP.format(circle=circle, standard=standard, reverse=signed_axis(standard, -1),
                                          readers=_AXIS_READERS.get(self._used_by, {}).get(role, "")).rstrip())

    def axis_box(self, circle) -> QtWidgets.QComboBox:
        """A circle's axis box (its data: 1 for psic's sense, −1 reversed)."""
        return self._axes[circle]

    def _edited(self, *_args):
        self.changed.emit()


__all__ = ['GeometryEditor']
