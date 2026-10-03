"""Mounting controls keep all eight signed cameras and pyFAI image transforms intact."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.widgets.camera_editor import CameraEditor, UPRIGHT_CAMERA
from xrd_tools.core.geometry.camera import STITCH_CAMERA, DetectorCamera, stitch_orientation


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_rotations_and_independent_flips_cover_eight_cameras(qapp):
    editor = CameraEditor()
    try:
        assert [editor.rotation.itemData(i) for i in range(editor.rotation.count())] == [0, 90, 180, 270]
        assert editor.value() == UPRIGHT_CAMERA
        cameras = set()
        for rotation, axes in ((0, ('x-', 'z+')), (90, ('z+', 'x+')),
                               (180, ('x+', 'z-')), (270, ('z-', 'x-'))):
            editor.rotation.setCurrentIndex(editor.rotation.findData(rotation))
            for lr in (False, True):
                for ud in (False, True):
                    editor.fliplr.setChecked(lr)
                    editor.flipud.setChecked(ud)
                    expected = tuple(axis[0] + ('-' if axis[1] == '+' else '+') if flip else axis
                                     for axis, flip in zip(axes, (ud, lr)))
                    assert editor.value() == DetectorCamera(*expected)
                    cameras.add(editor.value())
        assert len(cameras) == 8
        editor.rotation.setCurrentIndex(0)
        editor.fliplr.setChecked(True)
        editor.flipud.setChecked(True)
        assert editor.value() == STITCH_CAMERA  # both flips equal a half turn
    finally:
        editor.deleteLater()


def test_default_click_restores_upright_without_changing_the_native_frame(qapp):
    editor = CameraEditor()
    try:
        editor.set_value(DetectorCamera('z-', 'x+'))
        changed = []
        editor.changed.connect(lambda: changed.append(editor.value()))
        editor.default_button.click()
        assert changed == [UPRIGHT_CAMERA]
        assert editor.value() == DetectorCamera('x-', 'z+')
        assert (editor.rotation.currentData(), editor.fliplr.isChecked(), editor.flipud.isChecked()) == (0, False, False)
        transform = stitch_orientation(editor.value())
        assert transform.rotation == 180 and not transform.flip_horizontal and not transform.flip_vertical
        assert STITCH_CAMERA == DetectorCamera('x+', 'z-')
        assert editor.default_button.text() == '↺'
        assert editor.default_button.accessibleName() == 'Reset camera orientation'
        assert editor.layout().itemAt(editor.layout().count() - 2).widget() is editor.default_button
        assert editor.layout().itemAt(editor.layout().count() - 1).widget() is editor.image_size
    finally:
        editor.deleteLater()


def test_fresh_stitch_page_window_and_camera_editor_share_upright_default(qapp, tmp_path):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xdart.gui.tabs.stitch.page import StitchPage, camera_orientation
    from xdart.gui.pages.values import PageCleanup
    from tests.xdart.pages.test_stitch_page import _Status

    page = StitchPage(empty_host_services(_Status()))
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    editor = CameraEditor()
    try:
        for orientation in (camera_orientation(page._value('camera')),
                stitch_orientation(dialog.camera.value()), stitch_orientation(editor.value())):
            assert (orientation.rotation, orientation.flip_horizontal, orientation.flip_vertical) == (180, False, False)
        assert page._rows['camera'].editor.currentText() == 'upright'
        assert page._rows['camera'].current_value() == 'x-, z+'
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        dialog.close()
        editor.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('axes,description,turn,mirror', [
    (('x+', 'z-'), '180° CW', 0, ''),
    (('x+', 'z+'), '180° CW, horizontal flip', 0, 'horizontal'),
    (('x-', 'z-'), 'upright, horizontal flip', 0, 'vertical'),
    (('x-', 'z+'), 'upright', 2, ''),
    (('z-', 'x-'), '270° CW', 1, ''),
    (('z+', 'x-'), '90° CW, horizontal flip', 1, 'horizontal'),
    (('z+', 'x+'), '90° CW', -1, ''),
    (('z-', 'x+'), '270° CW, horizontal flip', -1, 'horizontal'),
])
def test_camera_image_correction_matches_the_labelled_transform(qapp, axes, description, turn, mirror):
    editor = CameraEditor()
    try:
        editor.set_value(DetectorCamera(*axes))
        assert editor.transform_description() == description
        marked = np.arange(15).reshape(3, 5)
        expected = np.rot90(marked, turn)
        if mirror == 'horizontal':
            expected = np.fliplr(expected)
        elif mirror == 'vertical':
            expected = np.flipud(expected)
        np.testing.assert_array_equal(stitch_orientation(editor.value()).apply(marked), expected)
    finally:
        editor.deleteLater()
