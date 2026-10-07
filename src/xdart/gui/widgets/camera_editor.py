# -*- coding: utf-8 -*-
"""Mounting controls for the eight detector cameras, retaining signed lab axes internally.

The upright Pilatus 300K-W and Eiger2 500K datasets store rows x−, columns z+.
This GUI reference is a mounting default, independent of the native pyFAI calibration frame.
Rotations describe the detector looking downstream; L/R and U/D flips reverse stored columns and rows.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.widgets.controls_panel import ControlsComboBox
from xrd_tools.core.geometry.camera import DetectorCamera


UPRIGHT_CAMERA = DetectorCamera('x-', 'z+')
_ROTATIONS = {0: ('x-', 'z+'), 90: ('z+', 'x+'), 180: ('x+', 'z-'), 270: ('z-', 'x-')}
_ROTATION_LABELS = {0: '0° (upright)', 90: '90°', 180: '180°', 270: '270°'}


def mounting_camera(rotation: int, flip: bool, flip_vertical: bool = False) -> DetectorCamera:
    """Clockwise mounting rotation, then reversals of stored columns and rows."""
    rows, columns = _ROTATIONS[rotation]
    if flip:
        columns = columns[0] + ('-' if columns[1] == '+' else '+')
    if flip_vertical:
        rows = rows[0] + ('-' if rows[1] == '+' else '+')
    return DetectorCamera(rows, columns)


def mounting_controls(camera: DetectorCamera) -> tuple[int, bool]:
    for rotation, (rows, columns) in _ROTATIONS.items():
        if camera.rows == rows:
            return rotation, camera.columns != columns
    raise ValueError('not a detector camera')


def mounting_description(camera: DetectorCamera) -> str:
    rotation, flip = mounting_controls(camera)
    return ('upright' if rotation == 0 else f'{rotation}° CW') + (
        ', horizontal flip' if flip else '')


CAMERA_LABELS = {f'{camera.rows}, {camera.columns}': mounting_description(camera)
                 for rotation in _ROTATIONS for flip in (False, True)
                 for camera in (mounting_camera(rotation, flip),)}


class CameraEditor(QtWidgets.QWidget):
    """Four mounting rotations and two flips, resolving to eight signed-axis cameras."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.rotation = ControlsComboBox()
        for rotation, caption in _ROTATION_LABELS.items():
            self.rotation.addItem(caption, rotation)
        layout.addWidget(QtWidgets.QLabel('Rotation'))
        layout.addWidget(self.rotation, 1)
        self.fliplr = QtWidgets.QPushButton('Flip ⇔')
        self.flipud = QtWidgets.QPushButton('Flip ⇕')
        self.fliplr.setAccessibleName('Flip left/right')
        self.flipud.setAccessibleName('Flip up/down')
        self.fliplr.setToolTip('Flip left/right: reverse stored image columns.')
        self.flipud.setToolTip('Flip up/down: reverse stored image rows.')
        for button in (self.fliplr, self.flipud):
            button.setObjectName('controlsToggleButton')
            button.setAutoDefault(False)
            button.setCheckable(True)
            button.ensurePolished()
            font = button.font()
            scale = 1.4 if button is self.flipud else 1.25
            if font.pointSizeF() > 0:
                font.setPointSizeF(font.pointSizeF() * scale)
            else:
                font.setPixelSize(round(font.pixelSize() * scale))
            button.setFont(font)
            button.setMinimumWidth(round(button.sizeHint().width() * 1.25))
            button.toggled.connect(self._edited)
            layout.addWidget(button)
        self.rotation.currentIndexChanged.connect(self._edited)
        self.set_value(UPRIGHT_CAMERA)
        self.default_button = QtWidgets.QPushButton('↺')
        self.default_button.setAutoDefault(False)
        self.default_button.setAccessibleName('Reset camera orientation')
        self.default_button.setFixedWidth(self.default_button.sizeHint().height())
        self.default_button.setToolTip('Reset to upright mounting: 0°, no flip (stored rows x−, columns z+). '
                                      'Calibration still uses pyFAI\'s native frame internally.')
        self.default_button.clicked.connect(lambda: self.set_value(UPRIGHT_CAMERA))
        layout.addWidget(self.default_button)
        self.image_size = QtWidgets.QLabel()
        layout.addWidget(self.image_size)

    def value(self) -> DetectorCamera:
        return mounting_camera(self.rotation.currentData(), self.fliplr.isChecked(), self.flipud.isChecked())

    def transform_description(self) -> str:
        """The mounting shown by the controls, relative to the upright detector."""
        rotation = self.rotation.currentData()
        description = 'upright' if rotation == 0 else f'{rotation}° CW'
        flips = [name for button, name in ((self.fliplr, 'horizontal'), (self.flipud, 'vertical'))
                 if button.isChecked()]
        return description + (', ' + ' and '.join(flips) + ' flip' if flips else '')

    def set_value(self, camera: DetectorCamera):
        rotation, flip = mounting_controls(camera)
        blockers = [QtCore.QSignalBlocker(widget) for widget in (self.rotation, self.fliplr, self.flipud)]
        self.rotation.setCurrentIndex(self.rotation.findData(rotation))
        self.fliplr.setChecked(flip)
        self.flipud.setChecked(False)
        del blockers
        self.changed.emit()

    def _edited(self, *_args):
        self.changed.emit()


__all__ = ['CameraEditor']
