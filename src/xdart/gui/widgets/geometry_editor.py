# -*- coding: utf-8 -*-
"""The diffractometer geometry and the scan motor driving each circle, as one editor.

Every page that needs to know which motor is which mounts this (Stitch and its
calibration window now, RSM next). The value is
:class:`xrd_tools.session.instrument.InstrumentGeometry`.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.session.instrument import GEOMETRIES, GEOMETRY_LABELS, InstrumentGeometry

#: Every circle any geometry has, in display order (a row each; a geometry shows its own).
_ALL_CIRCLES = ('mu', 'eta', 'th', 'chi', 'phi', 'nu', 'del', 'tth')


class GeometryEditor(QtWidgets.QWidget):
    """Choose the geometry; each circle's motor is an editable list of the scan's motors."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, *, used=(), used_by=''):
        super().__init__(parent)
        self._used, self._used_by = frozenset(used), str(used_by)
        self._form = QtWidgets.QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self.geometry = QtWidgets.QComboBox()
        for key, label in GEOMETRY_LABELS.items():
            self.geometry.addItem(label, key)
        self._form.addRow('Diffractometer', self.geometry)
        self._motors = {}
        for circle in _ALL_CIRCLES:
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            combo.setEditText(circle)
            combo.editTextChanged.connect(self._edited)
            self._motors[circle] = combo
            self._form.addRow(QtWidgets.QLabel(), combo)
        self.geometry.currentIndexChanged.connect(self._geometry_changed)
        self._geometry_changed()

    def motor_edit(self, circle) -> QtWidgets.QLineEdit:
        """The text box of one circle's motor (a host may read or set it directly)."""
        return self._motors[circle].lineEdit()

    def value(self) -> InstrumentGeometry:
        geometry = self.geometry.currentData()
        return InstrumentGeometry(geometry, tuple((circle, self._motors[circle].currentText())
                                                  for circle, _role in GEOMETRIES[geometry]))

    def set_value(self, value: InstrumentGeometry):
        blockers = [QtCore.QSignalBlocker(widget) for widget in (self.geometry, *self._motors.values())]
        self.geometry.setCurrentIndex(max(0, self.geometry.findData(value.geometry)))
        for circle, motor in value.motors:
            self._motors[circle].setEditText(motor)
        del blockers
        self._show_rows()
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
        circles = dict(GEOMETRIES[self.geometry.currentData()])
        for circle, combo in self._motors.items():
            self._form.setRowVisible(combo, circle in circles)
            label = self._form.labelForField(combo)
            if circle in circles:
                used = f'  ·  used by {self._used_by}' if circle in self._used and self._used_by else ''
                label.setText(f'{circle} ({circles[circle]}){used}')
                combo.setToolTip(f'The scan motor that drives the {circle} circle')

    def _edited(self, *_args):
        self.changed.emit()


__all__ = ['GeometryEditor']
