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
    shown = lambda: [c for c in editor._motors if editor._form.isRowVisible(editor._rows[c])]  # noqa: E731
    assert shown() == ['mu', 'eta', 'chi', 'phi', 'nu', 'del']
    assert 'used by Stitch' in editor._form.labelForField(editor._rows['del']).text()
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


def test_fourc_and_twoc_take_a_scattering_plane(qapp):
    editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    assert not editor._form.isRowVisible(editor.plane)  # psic moves both detector circles
    editor.geometry.setCurrentIndex(editor.geometry.findData('twoc'))
    assert editor._form.isRowVisible(editor.plane) and editor.value().plane == 'vertical'
    assert "psic's del" in editor._motors['tth'].toolTip()
    changed.clear()
    editor.plane.setCurrentIndex(editor.plane.findData('horizontal'))
    assert changed and editor.value() == InstrumentGeometry('twoc', plane='horizontal')
    assert editor.value().stitch_arm() == (None, 'tth') and "psic's nu" in editor._motors['tth'].toolTip()
    editor.set_available_motors(['nu', 'mu'])  # a psic run as a horizontal twoc: its nu is the 2θ
    assert (editor.value().plane, editor.value().motor('tth'), editor.value().motor('th')) == ('horizontal', 'nu', 'mu')
    editor.set_value(InstrumentGeometry('fourc', plane='vertical'))
    assert editor.plane.currentData() == 'vertical' and editor._form.isRowVisible(editor.plane)
    editor.set_value(InstrumentGeometry())
    assert not editor._form.isRowVisible(editor.plane) and editor.value().plane == 'vertical'
    editor.deleteLater()


def test_the_editor_states_the_lab_frame_and_each_circles_axis(qapp):
    editor = GeometryEditor()
    label = lambda circle: editor._form.labelForField(editor._rows[circle]).text()  # noqa: E731
    assert 'x up, y along the beam' in editor.frame_note.text()
    assert label('mu').startswith('mu (sample) · x+') and label('del').startswith('del (detector) · z-')
    editor.set_value(InstrumentGeometry('twoc', plane='horizontal'))
    assert label('th').startswith('th (sample) · x+') and label('tth').startswith('tth (detector) · x+')
    editor.set_value(InstrumentGeometry('twoc'))
    assert label('tth').startswith('tth (detector) · z-')


def test_each_detector_circle_takes_its_motor_zero_beside_its_motor(qapp):
    """ARM-ZERO-BOUNDS-20260929: the detector circles' motor-zero readings (physical angle = motor
    readback − zero) sit beside their motors, the single arm's tth too; a blank box gives none."""
    editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    boxes = lambda: [c for c in editor._zeros if editor._zeros[c].isVisibleTo(editor)  # noqa: E731
                     and editor._form.isRowVisible(editor._rows[c])]
    editor.show()
    assert boxes() == ['nu', 'del'] and editor.value().zeros == ()
    assert set(editor._zeros) == {'nu', 'del', 'tth'}  # the sample circles have none yet
    editor.zero_edit('del').setText('2.3')
    assert changed and editor.value().zeros == (('del', 2.3),) and editor.value().arm_zeros() == {'del': 2.3}
    editor.zero_edit('nu').setText('-')  # half typed: none yet
    assert editor.value().zeros == (('del', 2.3),)
    editor.zero_edit('nu').setText('-0.8')
    assert editor.value().arm_zeros() == {'del': 2.3, 'nu': -.8}
    editor.set_value(InstrumentGeometry('twoc', zeros=(('tth', 1.5),)))
    assert boxes() == ['tth'] and editor.zero_edit('tth').text() == '1.5'
    assert editor.value() == InstrumentGeometry('twoc', zeros=(('tth', 1.5),))
    assert editor.value().arm_zeros() == {'del': 1.5}
    editor.set_value(InstrumentGeometry('psic', zeros=(('del', 2.636063),)))
    assert editor.zero_edit('del').text() == '2.636063' and editor.zero_edit('nu').text() == ''
    assert editor.value().zeros == (('del', 2.636063),)
    editor.set_available_motors(['del', 'nu'])  # a scan's motors keep the zeros
    assert editor.value().zeros == (('del', 2.636063),)
    editor.deleteLater()
