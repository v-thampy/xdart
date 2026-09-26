"""The shared diffractometer editor: a geometry's circles and the scan's motor for each."""
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.widgets.geometry_editor import GeometryEditor
from xrd_tools.session.instrument import InstrumentGeometry


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_editor_shows_each_geometrys_circles_and_takes_the_scans_names(qapp):
    editor = GeometryEditor(used=('nu', 'del'), used_by='Stitch')
    shown = lambda: [c for c in editor._motors if editor._form.isRowVisible(editor._motors[c])]  # noqa: E731
    assert shown() == ['mu', 'eta', 'chi', 'phi', 'nu', 'del']
    assert 'used by Stitch' in editor._form.labelForField(editor._motors['del']).text()
    editor.geometry.setCurrentIndex(editor.geometry.findData('fourc'))
    assert shown() == ['th', 'chi', 'phi', 'tth']
    editor.geometry.setCurrentIndex(editor.geometry.findData('psic'))
    editor.set_available_motors(['Delta', 'nu', 'halpha', 'mu', 'chi', 'phi'])
    assert (editor.value().motor('del'), editor.value().motor('eta')) == ('Delta', 'halpha')
    assert editor._motors['del'].findText('Delta') >= 0  # the dropdown lists the scan's motors
    editor.motor_edit('del').setText('tth2')  # or typed by hand
    assert editor.value().motor('del') == 'tth2'
    editor.set_value(InstrumentGeometry('twoc'))
    assert editor.value() == InstrumentGeometry('twoc') and shown() == ['th', 'tth']
    editor.deleteLater()
