"""The detector camera's two linked boxes (maintainer, 2026-09-30: "two drop-down boxes linked together. The first
has options x+, x-, z+, z-. Once the first ... is selected, the other box shows the remaining two transverse
options")."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.widgets.camera_editor import CameraEditor
from xrd_tools.core.geometry.camera import STITCH_CAMERA, DetectorCamera, stitch_orientation


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_columns_box_offers_the_two_directions_transverse_to_the_rows(qapp):
    editor = CameraEditor()
    items = lambda combo: [combo.itemData(i) for i in range(combo.count())]  # noqa: E731
    assert items(editor.rows) == ['x+', 'x-', 'z+', 'z-'] and items(editor.columns) == ['z+', 'z-']
    assert editor.value() == STITCH_CAMERA  # Stitch's frame, the default
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    editor.rows.setCurrentIndex(editor.rows.findData('z-'))  # rows along z: columns along x, their sign kept
    assert changed and items(editor.columns) == ['x+', 'x-'] and editor.value() == DetectorCamera('z-', 'x+')
    editor.columns.setCurrentIndex(editor.columns.findData('x-'))
    assert editor.value() == DetectorCamera('z-', 'x-')
    editor.set_value(DetectorCamera('x-', 'z-'))  # 17-2's Pilatus 300K-W, mirrored left-right
    assert (editor.rows.currentData(), editor.columns.currentData()) == ('x-', 'z-')
    editor.deleteLater()


def test_camera_descriptions_keep_all_eight_axis_values(qapp):
    editor = CameraEditor()
    try:
        captions = {'x+': 'vert ↑', 'x-': 'vert ↓', 'z+': 'horz ←', 'z-': 'horz →'}
        for rows in captions:
            editor.rows.setCurrentIndex(editor.rows.findData(rows))
            assert captions[rows] in editor.rows.currentText()
            for index in range(editor.columns.count()):
                editor.columns.setCurrentIndex(index)
                columns = editor.columns.currentData()
                assert captions[columns] in editor.columns.currentText()
                assert editor.value() == DetectorCamera(rows, columns)
    finally:
        editor.deleteLater()


def test_default_click_restores_unturned_unflipped_stitch_camera(qapp):
    editor = CameraEditor()
    try:
        editor.set_value(DetectorCamera('z-', 'x+'))
        changed = []
        editor.changed.connect(lambda: changed.append(editor.value()))
        editor.default_button.click()
        assert changed == [STITCH_CAMERA]
        assert editor.value() == DetectorCamera('x-', 'z+')
        transform = stitch_orientation(editor.value())
        assert transform.rotation == 0 and not transform.flip_horizontal and not transform.flip_vertical
        assert editor.default_button.text() == '↺'
        assert editor.default_button.accessibleName() == 'Reset camera orientation'
        assert editor.layout().itemAt(editor.layout().count() - 2).widget() is editor.default_button
        assert editor.layout().itemAt(editor.layout().count() - 1).widget() is editor.image_size
    finally:
        editor.deleteLater()


@pytest.mark.parametrize('axes,description,turn,mirror', [
    (('x-', 'z+'), 'standard', 0, ''),
    (('x-', 'z-'), 'horizontal flip', 0, 'horizontal'),
    (('x+', 'z+'), 'vertical flip', 0, 'vertical'),
    (('x+', 'z-'), '180° rotation', 2, ''),
    (('z+', 'x+'), '90° CCW', 1, ''),
    (('z+', 'x-'), '90° CW, then horizontal flip', -1, 'horizontal'),
    (('z-', 'x-'), '90° CW', -1, ''),
    (('z-', 'x+'), '90° CCW, then horizontal flip', 1, 'horizontal'),
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
