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
    """Each circle's axis box: psic's axis for the circle it plays, or its reverse (maintainer, 2026-09-30:
    "the sense of rotation for each motor should be selectable"; "let's just stick with swapping the signs
    for now"; "yes default to standard now")."""
    editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
    label = lambda circle: editor._form.labelForField(editor._rows[circle]).text()  # noqa: E731
    axis = lambda circle: editor.axis_box(circle).currentText()  # noqa: E731
    choices = lambda circle: [editor.axis_box(circle).itemText(i) for i in range(2)]  # noqa: E731
    note = editor.frame_note.text
    assert 'x up, y along the beam' in note()
    # the selected geometry's standard axes (maintainer, 2026-09-30: "add the default directions for the
    # different motors as per standard psic, or fourc or twoc when they are selected")
    assert note().endswith('standard psic: mu x+, eta z-, chi y+, phi z-, nu x+, del z-.')
    assert [axis(c) for c in ('mu', 'eta', 'chi', 'phi', 'nu', 'del')] == ['x+', 'z-', 'y+', 'z-', 'x+', 'z-']
    assert choices('nu') == ['x+', 'x-'] and choices('del') == ['z-', 'z+'] and choices('chi') == ['y+', 'y-']
    assert label('mu') == 'mu (sample)' and label('del') == 'del (detector)  ·  used by Stitch'
    assert 'xrayutilities' in editor.axis_box('nu').toolTip() and 'grazing' in editor.axis_box('chi').toolTip()
    editor.set_value(InstrumentGeometry('twoc', plane='horizontal'))
    assert (axis('th'), axis('tth')) == ('x+', 'x+')
    assert note().endswith('standard twoc (horizontal plane): th x+, tth x+.')
    editor.set_value(InstrumentGeometry('twoc'))
    assert axis('tth') == 'z-' and note().endswith('standard twoc (vertical plane): th z-, tth z-.')
    editor.geometry.setCurrentIndex(editor.geometry.findData('fourc'))  # selected in the box, as he would
    assert note().endswith('standard fourc (vertical plane): th z-, chi y+, phi z-, tth z-.')
    # 17-2's psic until November 2026: [x-, z-, y-, z-], [x-, z-]
    editor.set_value(InstrumentGeometry())
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    for circle in ('mu', 'chi', 'nu'):
        editor.axis_box(circle).setCurrentIndex(editor.axis_box(circle).findData(-1))
    beamline = editor.value()
    assert changed and beamline.reversed_circles == ('mu', 'chi', 'nu')
    assert note().endswith('standard psic: mu x+, eta z-, chi y+, phi z-, nu x+, del z-.')  # psic's, whatever chosen
    assert [value for _circle, value in beamline.circle_axes()] == ['x-', 'z-', 'y-', 'z-', 'x-', 'z-']
    editor.set_value(InstrumentGeometry())  # psic's again
    assert editor.value().reversed_circles == () and axis('nu') == 'x+'
    editor.set_value(beamline)
    assert editor.value() == beamline and axis('chi') == 'y-'
    editor.set_available_motors(['mu', 'chi', 'nu', 'del'])  # a scan's motors keep the senses
    assert editor.value().reversed_circles == ('mu', 'chi', 'nu')
    # a 2θ arm keeps its sense when its plane changes the axis it turns about
    editor.set_value(InstrumentGeometry('twoc', reversed_circles=('tth',)))
    assert axis('tth') == 'z+'
    editor.plane.setCurrentIndex(editor.plane.findData('horizontal'))
    assert axis('tth') == 'x-' and editor.value().arm_senses() == {'nu': -1}
    editor.deleteLater()


def test_each_detector_circle_takes_its_motor_zero_beside_its_motor(qapp):
    """ARM-ZERO-BOUNDS-20260929: the detector circles' motor-zero readings (physical angle = motor
    readback − zero) sit beside their motors, the single arm's tth too; a blank box gives none."""
    editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
    changed = []
    editor.changed.connect(lambda: changed.append(True))
    boxes = lambda: [c for c in editor._zeros if editor._zeros[c].isVisibleTo(editor)  # noqa: E731
                     and editor._form.isRowVisible(editor._rows[c])]
    editor.show()
    # deferred item 7: every circle has its zero beside its motor, the sample's too (blank: 0)
    assert boxes() == ['mu', 'eta', 'chi', 'phi', 'nu', 'del'] and editor.value().zeros == ()
    assert set(editor._zeros) == {'mu', 'eta', 'th', 'chi', 'phi', 'nu', 'del', 'tth'}
    editor.zero_edit('del').setText('2.3')
    assert changed and editor.value().zeros == (('del', 2.3),) and editor.value().arm_zeros() == {'del': 2.3}
    editor.zero_edit('nu').setText('-')  # half typed: none yet
    assert editor.value().zeros == (('del', 2.3),)
    editor.zero_edit('nu').setText('-0.8')
    assert editor.value().arm_zeros() == {'del': 2.3, 'nu': -.8}
    editor.set_value(InstrumentGeometry('twoc', zeros=(('tth', 1.5),)))
    assert boxes() == ['th', 'tth'] and editor.zero_edit('tth').text() == '1.5'
    assert editor.value() == InstrumentGeometry('twoc', zeros=(('tth', 1.5),))
    assert editor.value().arm_zeros() == {'del': 1.5}
    editor.set_value(InstrumentGeometry('psic', zeros=(('del', 2.636063),)))
    assert editor.zero_edit('del').text() == '2.636063' and editor.zero_edit('nu').text() == ''
    assert editor.value().zeros == (('del', 2.636063),)
    editor.set_available_motors(['del', 'nu'])  # a scan's motors keep the zeros
    assert editor.value().zeros == (('del', 2.636063),)
    editor.deleteLater()
