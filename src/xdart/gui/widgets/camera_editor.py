# -*- coding: utf-8 -*-
"""The detector camera as two linked boxes: the lab directions a stored image's rows and columns run along.

The maintainer's design (2026-09-30): "two drop-down boxes linked together. The first has options x+, x-,
z+, z-. Once the first ... is selected, the other box shows the remaining two transverse options." The
value is a :class:`xrd_tools.core.geometry.camera.DetectorCamera`, xrayutilities' detectorDir1 and
detectorDir2: with every circle at zero the detector faces the beam, x is up and z to the left looking
downstream. Its eight values are every 90° turn of the image and their mirrors.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.widgets.controls_panel import ControlsComboBox
from xrd_tools.core.geometry.camera import CAMERA_AXES, STITCH_CAMERA, DetectorCamera, stitch_orientation


_AXIS_LABELS = {'x+': 'x+ (vert ↑)', 'x-': 'x- (vert ↓)',
                'z+': 'z+ (horz ←)', 'z-': 'z- (horz →)'}


class CameraEditor(QtWidgets.QWidget):
    """Rows along one of x+, x-, z+, z-; columns along one of the two directions transverse to them."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rows, self.columns = ControlsComboBox(), ControlsComboBox()
        for axis in CAMERA_AXES:
            self.rows.addItem(_AXIS_LABELS[axis], axis)
        for caption, combo in (('rows', self.rows), ('columns', self.columns)):
            layout.addWidget(QtWidgets.QLabel(caption))
            layout.addWidget(combo, 1)
        self.rows.currentIndexChanged.connect(self._rows_changed)
        self.columns.currentIndexChanged.connect(self._edited)
        self.set_value(STITCH_CAMERA)
        self.default_button = QtWidgets.QPushButton('↺')
        self.default_button.setAutoDefault(False)
        self.default_button.setAccessibleName('Reset camera orientation')
        self.default_button.setFixedWidth(self.default_button.sizeHint().height())
        self.default_button.setToolTip('Reset to xdart\'s reference camera: rows x− (down), columns z+ '
                                      '(left looking downstream). Image rotation 0°, no flip.')
        self.default_button.clicked.connect(lambda: self.set_value(STITCH_CAMERA))
        layout.addWidget(self.default_button)
        self.image_size = QtWidgets.QLabel()
        layout.addWidget(self.image_size)

    def value(self) -> DetectorCamera:
        return DetectorCamera(self.rows.currentData(), self.columns.currentData())

    def transform_description(self) -> str:
        """The stored image's correction into xdart's reference, with the turn before the mirror."""
        transform = stitch_orientation(self.value())
        if transform.rotation == 0:
            return 'horizontal flip' if transform.flip_horizontal else 'standard'
        if transform.rotation == 180 and transform.flip_horizontal:
            return 'vertical flip'
        turn = {90: '90° CCW', 180: '180° rotation', 270: '90° CW'}[transform.rotation]
        return turn + (', then horizontal flip' if transform.flip_horizontal else '')

    def set_value(self, camera: DetectorCamera):
        blocker = QtCore.QSignalBlocker(self.rows)
        self.rows.setCurrentIndex(self.rows.findData(camera.rows))
        del blocker
        self._fill_columns(camera.columns)
        self.changed.emit()

    def _fill_columns(self, keep):
        """The two directions transverse to the rows', ``keep``'s sign chosen."""
        axis = 'z' if self.rows.currentData()[0] == 'x' else 'x'
        blocker = QtCore.QSignalBlocker(self.columns)
        self.columns.clear()
        for sign in '+-':
            self.columns.addItem(_AXIS_LABELS[axis + sign], axis + sign)
        self.columns.setCurrentIndex(self.columns.findData(axis + (keep or '++')[1]))
        del blocker

    def _rows_changed(self, *_args):
        self._fill_columns(self.columns.currentData())  # the columns keep their sign on the new axis
        self.changed.emit()

    def _edited(self, *_args):
        self.changed.emit()


__all__ = ['CameraEditor']
