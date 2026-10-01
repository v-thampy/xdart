"""The detector camera's two linked boxes (maintainer, 2026-09-30: "two drop-down boxes linked together. The first
has options x+, x-, z+, z-. Once the first ... is selected, the other box shows the remaining two transverse
options")."""
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.widgets.camera_editor import CameraEditor
from xrd_tools.core.geometry.camera import STITCH_CAMERA, DetectorCamera


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_columns_box_offers_the_two_directions_transverse_to_the_rows(qapp):
    editor = CameraEditor()
    items = lambda combo: [combo.itemText(i) for i in range(combo.count())]  # noqa: E731
    assert items(editor.rows) == ['x+', 'x-', 'z+', 'z-'] and items(editor.columns) == ['z+', 'z-']
    assert editor.value() == STITCH_CAMERA  # Stitch's frame, the default
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    editor.rows.setCurrentIndex(editor.rows.findData('z-'))  # rows along z: columns along x, their sign kept
    assert changed and items(editor.columns) == ['x+', 'x-'] and editor.value() == DetectorCamera('z-', 'x+')
    editor.columns.setCurrentIndex(editor.columns.findData('x-'))
    assert editor.value() == DetectorCamera('z-', 'x-')
    editor.set_value(DetectorCamera('x-', 'z-'))  # 17-2's Pilatus 300K-W, mirrored left-right
    assert (editor.rows.currentText(), editor.columns.currentText()) == ('x-', 'z-')
    editor.deleteLater()
