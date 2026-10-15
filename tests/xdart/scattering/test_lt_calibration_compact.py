"""LT-SMALL-FIXES-20261014 stage 3: frozen oracle rows C1–C4 (Claude, October 14).

LT-007–LT-009 under the maintainer's decisions ("Inline expansion"; the title is standard or custom from the current
motor names and senses, never from fitted zeros; a pure mirror stays custom). 1 · Instrument shows the Diffractometer
Type and the Detector Model and Rotation, each with an inline Config (the shared SubsectionCard showing the same
editors), a title summarising the geometry, one exposed Mask file and Quick below both. The Orientation check has
one mode-aware readiness fact that the checklist, the footer entry, the panel's Run, its finding pane and the
checklist navigation agree on. The instrument schematic separates its labels, names its pose, explains once, hides
dimensional ticks only for schematic units and gives its message pane 4–6 lines in an adjustable split.

Real windows from their pages, real clicks, typing and row activation; a job is held only at the window's real
dispatcher. Rows marked "preservation" were green on the parent; every other row failed there.
"""
from dataclasses import replace
import threading
import time

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_calibration_quick_chain import _close, _commit, _type, _window
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _wait
from xdart.gui.widgets.controls_panel import FormRow, RangeRow, SectionCard, SubsectionCard
from xrd_tools.session.instrument import GEOMETRY_LABELS, InstrumentGeometry


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _instrument(dialog):
    """1 · Instrument, shown."""
    tabs = dialog.tabs
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == '1 · Instrument')
    tabs.setCurrentIndex(index)
    QtWidgets.QApplication.processEvents()
    return tabs.widget(index)


def _box(page, name):
    boxes = [box for box in page.findChildren(QtWidgets.QGroupBox) if box.title().split(' · ')[0] == name]
    assert len(boxes) == 1, [box.title() for box in page.findChildren(QtWidgets.QGroupBox)]
    return boxes[0]


def _config(box):
    """The box's one Config trigger and the one card it shows and hides."""
    buttons = [b for b in box.findChildren(QtWidgets.QToolButton) if b.text().startswith('Config')]
    assert len(buttons) == 1, [b.text() for b in box.findChildren(QtWidgets.QToolButton)]
    cards = box.findChildren(SubsectionCard)
    assert len(cards) == 1, len(cards)
    return buttons[0], cards[0]


def _press(widget):
    QtTest.QTest.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()


def _reverse(editor, circle):
    box = editor.axis_box(circle)
    box.setCurrentIndex(box.findData(-box.currentData()))


def _title(value):
    """The maintainer's rule: custom when a motor name or a sense differs from the geometry's standard."""
    standard = not value.reversed_circles and all(motor == circle for circle, motor in value.motors)
    return f"Diffractometer · {GEOMETRY_LABELS[value.geometry]} · {'standard' if standard else 'custom'}"


def _checklist(dialog):
    """The checklist table, its card opened as a person opens it."""
    from tests.xdart.scattering.test_calibration_checklist import _table
    card = next(c for c in dialog.findChildren(SectionCard) if c.title.text() == 'CALIBRATION CHECKLIST')
    if not card.body.isVisible():
        _press(card.toggle)
    return _table(dialog)


def _orientation_row(table):
    return table.item(7, 1).text(), table.item(7, 2).text()


def _activate(table, row):
    """Activate a checklist row with a real double click."""
    item = table.item(row, 0)
    table.scrollToItem(item)
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.mouseDClick(table.viewport(), QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                             table.visualItemRect(item).center())
    QtWidgets.QApplication.processEvents()


def _tab(dialog):
    return dialog.tabs.tabText(dialog.tabs.currentIndex())


def _show(dialog, title):
    """The tab ``title``, shown as a person selects it before typing there."""
    dialog.tabs.setCurrentIndex(next(i for i in range(dialog.tabs.count()) if dialog.tabs.tabText(i) == title))
    QtWidgets.QApplication.processEvents()


def _tabs(dialog):
    return [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]


def _until(condition, seconds=20):
    until = time.monotonic() + seconds
    while not condition() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    return condition()


# C1 · Diffractometer Type, inline Config and its title ------------------------------------------------------------

@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_c1_diffractometer_config_keeps_its_editors_and_the_title_follows_names_and_senses(qapp, tmp_path, page_name):
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        instrument = _instrument(dialog)
        box = _box(instrument, 'Diffractometer')
        editor = dialog.geometry_editor
        button, card = _config(box)
        # exposed: Type and Config; the configuration collapsed at first
        assert editor.geometry.isVisible() and button.isVisible() and button.toolTip()
        assert button.text() == 'Config ▸' and not card.isAncestorOf(editor.geometry) and not card.header.isVisible()
        inside = (editor.axis_box('del'), editor.motor_edit('del'), editor.zero_edit('del'), editor.default_senses,
                  editor.frame_note)
        assert all(card.isAncestorOf(widget) for widget in inside) and card.isAncestorOf(editor.plane)
        assert not any(widget.isVisible() for widget in inside)
        assert box.title() == _title(editor.value()) and box.toolTip()
        _press(button)
        assert button.text() == 'Config ▾' and all(widget.isVisible() for widget in inside)
        assert len(dialog.findChildren(type(editor))) == 1  # the one editor, shown and hidden, never rebuilt
        # standard: every motor named as its circle, every sense psic's; a typed or fitted zero changes nothing
        dialog.set_instrument(InstrumentGeometry('psic'))
        assert box.title() == 'Diffractometer · psic (6-circle) · standard'
        _commit(editor.zero_edit('del'), '1.25')
        assert box.title() == 'Diffractometer · psic (6-circle) · standard'
        # a draft kept through the disclosure: hiding and showing commits nothing and invalidates nothing
        _reverse(editor, 'nu')
        state = (editor.value(), dialog.fit, dialog._orientation_key())
        _press(button)
        assert button.text() == 'Config ▸' and not editor.zero_edit('del').isVisible()
        _press(button)
        assert (editor.value(), dialog.fit, dialog._orientation_key()) == state
        assert editor.zero_edit('del').text() == '1.25' and editor.value().sense('nu') == -1
        assert box.title() == 'Diffractometer · psic (6-circle) · custom' and 'nu' in box.toolTip()
        # a renamed motor is custom; ↺ restores the senses only, so the renamed motor keeps it custom
        _commit(editor.motor_edit('eta'), 'halpha')
        assert box.title().endswith(' · custom') and 'halpha' in box.toolTip()
        before = editor.value()
        _press(editor.default_senses)
        assert editor.value() == replace(before, reversed_circles=())
        assert box.title() == 'Diffractometer · psic (6-circle) · custom' and 'halpha' in box.toolTip()
        _commit(editor.motor_edit('eta'), 'eta')
        assert box.title() == 'Diffractometer · psic (6-circle) · standard'
        # the pure 17-2 mirror stays custom, named precisely; its residual reversal is named too
        for circle in ('mu', 'chi', 'nu'):
            _reverse(editor, circle)
        tip = box.toolTip()
        assert box.title().endswith(' · custom') and 'mirror' in tip and all(c in tip for c in ('mu', 'chi', 'nu'))
        assert 'phi' not in tip
        _reverse(editor, 'phi')
        assert box.title().endswith(' · custom') and 'mirror' in box.toolTip() and 'phi' in box.toolTip()
        # a restored session: the title follows the restored values
        saved = dialog.settings()
        dialog.set_instrument(InstrumentGeometry('psic'))
        assert box.title().endswith(' · standard')
        dialog.apply_settings(saved)
        _wait(dialog)
        assert editor.value().reversed_circles == ('mu', 'chi', 'phi', 'nu') and box.title().endswith(' · custom')
        # another geometry chosen in Type: its own label, and its scattering plane as a tooltip fact
        editor.geometry.setCurrentIndex(editor.geometry.findData('fourc'))
        assert box.title() == _title(editor.value()) and box.title().startswith('Diffractometer · fourc (4-circle) · ')
        assert 'plane' in box.toolTip().lower()
    finally:
        _close(qapp, page, dialog)


def test_c1_a_hidden_invalid_zero_and_the_motor_sign_reveal_config(qapp, tmp_path):
    from tests.xdart.scattering.test_orientation_diagnosis import dialog_at_fit
    dialog = dialog_at_fit(tmp_path)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()
    try:
        editor = dialog.geometry_editor
        button, _card = _config(_box(_instrument(dialog), 'Diffractometer'))
        # Show motor sign focuses the scanned circle's sense box inside the collapsed configuration
        _press(dialog.orientation_button)
        panel = dialog.orientation_panel
        assert panel is not None and button.text() == 'Config ▸'
        _press(panel.show_sign_button)
        assert button.text() == 'Config ▾' and editor.axis_box('del').hasFocus()
        # a malformed zero typed there, then hidden: the checklist names it, and its row reveals and focuses it
        _instrument(dialog)
        _type(editor.zero_edit('nu'), '-')
        _press(button)
        assert button.text() == 'Config ▸' and not editor.zero_edit('nu').isVisible()
        table = _checklist(dialog)
        assert table.item(0, 1).text() == 'Needs input' and 'nu' in table.item(0, 2).text()
        _activate(table, 0)
        assert _tab(dialog) == '1 · Instrument' and button.text() == 'Config ▾'
        assert editor.zero_edit('nu').isVisible() and editor.zero_edit('nu').hasFocus()
    finally:
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()


# C2 · Detector Model, Rotation and Config; one Mask file; Quick below ---------------------------------------------

@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_c2_detector_row_config_one_exposed_mask_and_quick_below(qapp, tmp_path, page_name):
    from xrd_tools.integrate.calibration_seed import detector_choices, detector_shape
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        instrument = _instrument(dialog)
        box = _box(instrument, 'Detector')
        model, camera = dialog._fields['beam_detector'], dialog.camera
        button, card = _config(box)
        # one primary row: Model, Rotation and Config, outside the configuration
        for widget in (model, camera.rotation, button):
            assert widget.isVisible() and box.isAncestorOf(widget) and not card.isAncestorOf(widget)

        def middle(widget):
            return widget.mapTo(box, widget.rect().center()).y()
        assert abs(middle(camera.rotation) - middle(model)) < model.height() / 2
        assert abs(middle(button) - middle(model)) < model.height() / 2
        assert button.text() == 'Config ▸' and button.toolTip() and model.toolTip()
        # the caption: the model and its native rows × columns; no RAW, and no pixel type the window cannot know
        labels = {name: label for label, name in detector_choices()}
        rows, columns = detector_shape(dialog._text('beam_detector'))
        caption = f"{labels[dialog._text('beam_detector')]} · {rows} × {columns}"
        assert model.currentText() == caption
        assert not any('RAW' in model.itemText(i) or '[' in model.itemText(i) for i in range(model.count()))
        # inside Config, collapsed at first: RAW layout and header, RAW images, flips, reset, first pixel, saturation
        raw_images = (dialog._fields['shape'], dialog._fields['dtype'])
        inside = (dialog.raw_layout, dialog._fields['header'], camera.fliplr, camera.flipud, camera.default_button,
                  dialog._fields['beam_orientation'], dialog._fields['saturation'], *raw_images)
        assert all(card.isAncestorOf(widget) for widget in inside) and not any(w.isVisible() for w in inside)
        masks = [row for row in instrument.findChildren(FormRow) if row.label.text() == 'Mask file']
        assert len(masks) == 1 and masks[0].isVisible() and box.isAncestorOf(masks[0])
        assert not card.isAncestorOf(masks[0]) and masks[0].editor is dialog._fields['mask']
        _press(button)
        assert button.text() == 'Config ▾'
        assert all(widget.isVisible() for widget in inside if widget not in raw_images)
        assert len(dialog.findChildren(type(camera))) == 1
        # a configured RAW type is a known type: the caption shows it; the detector default shows none
        dialog.raw_layout.setCurrentIndex(dialog.raw_layout.findData('manual'))
        assert all(widget.isVisible() for widget in raw_images)
        _commit(dialog._fields['dtype'], 'uint16')
        assert model.currentText() == f'{caption} · uint16'
        dialog.raw_layout.setCurrentIndex(dialog.raw_layout.findData('factory'))
        assert model.currentText() == caption
        # four rotations by both flips: the eight cameras through the split editor, named in the title
        cameras = set()
        for index in range(camera.rotation.count()):
            camera.rotation.setCurrentIndex(index)
            for _ in range(2):
                cameras.add(camera.value())
                assert 'orientation' in box.title()
                _press(camera.fliplr)
        assert len(cameras) == 8
        # Quick calibration below both sections, with no Mask file of its own; its one Threshold row unchanged
        quick = [c for c in instrument.findChildren(SectionCard) if c.title.text() == 'QUICK CALIBRATION']
        assert len(quick) == 1 and quick[0].isVisible()
        quick = quick[0]
        bottom = max(b.mapTo(instrument, QtCore.QPoint(0, b.height())).y()
                     for b in (_box(instrument, 'Diffractometer'), box))
        assert quick.mapTo(instrument, QtCore.QPoint(0, 0)).y() >= bottom
        assert not [row for row in quick.findChildren(FormRow) if row.label.text() == 'Mask file']
        assert len(quick.findChildren(RangeRow)) == 1
    finally:
        _close(qapp, page, dialog)


def test_c2_detector_configuration_restores_and_the_one_mask_reaches_the_jobs(qapp, tmp_path, monkeypatch):
    """Preservation, apart from the restored caption: every detector setting round-trips through a new window, and
    the Detector panel's one Mask file reaches the direct-beam jobs."""
    from pyFAI.detectors import detector_factory
    from xdart.gui.tabs.calibration import dialog as dialog_module
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from tests.xdart.scattering.test_direct_beam_readiness import _fit_container, detect
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    other = None
    try:
        _fit_container(dialog, tmp_path, 270)
        masked = np.zeros(detector_factory('Pilatus100k').shape, bool)
        masked[5:9, 30:40] = True
        path = tmp_path / 'mask.npy'
        np.save(path, masked)
        requests = []
        real = dialog_module._dispatch_job

        def recorded(request, cancelled, publish):
            requests.append(request)
            return real(request, cancelled, publish)
        monkeypatch.setattr(dialog_module, '_dispatch_job', recorded)
        _commit(dialog._fields['mask'], str(path))
        _wait(dialog)
        detect(dialog)
        dialog.beam_button.click()
        _wait(dialog)
        fits = [arguments for kind, arguments in requests if kind == 'distance']
        assert len(fits) == 1, [kind for kind, _arguments in requests]
        _spec, _motor, options, _held = fits[0]
        assert options['mask'].any()
        np.testing.assert_array_equal(options['mask'], dialog._oriented_mask())
        # every detector setting, restored into a new window
        dialog._set_field_text('beam_orientation', '1')
        dialog.raw_layout.setCurrentIndex(dialog.raw_layout.findData('manual'))
        for key, text in (('shape', '195, 487'), ('dtype', 'uint16'), ('header', '64'), ('saturation', '5000')):
            dialog._set_field_text(key, text)
        dialog.camera.rotation.setCurrentIndex(dialog.camera.rotation.findData(90))
        dialog.camera.fliplr.click()
        saved = dialog.settings()
        other = CalibrationDialog(CalibrationContext('Stitch', tmp_path / 'other'))
        other.show()
        other.apply_settings(saved)
        _wait(other)
        for key in ('beam_detector', 'beam_orientation', 'shape', 'dtype', 'header', 'saturation', 'mask'):
            assert other._text(key) == dialog._text(key), key
        assert other.raw_layout.currentData() == 'manual' and other.camera.value() == dialog.camera.value()
        assert other._fields['beam_detector'].currentText().endswith(' · uint16')
        assert other._fields['beam_detector'].currentText() == dialog._fields['beam_detector'].currentText()
    finally:
        for window in (dialog, other):
            if window is not None:
                window.close()
                _wait(window)
                window.deleteLater()
        qapp.processEvents()


# C3 · one Orientation readiness fact -------------------------------------------------------------------------------

def test_c3_checklist_owner_projects_orientation_readiness():
    from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
    assert calibration_checklist(ChecklistFacts())[7].state == 'Optional'  # preservation: no fact given
    row = calibration_checklist(replace(ChecklistFacts(), orientation_ready=False, orientation_message='why',
                                        orientation_target='rough'))[7]
    assert (row.state, row.message, row.target) == ('Needs input', 'why', 'rough')
    row = calibration_checklist(replace(ChecklistFacts(), orientation_waiting=True, orientation_message='wait'))[7]
    assert (row.state, row.message) == ('Waiting', 'wait')


def test_c3_direct_beam_route_agrees_everywhere_and_a_late_finish_reprojects(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.calibration import dialog as dialog_module
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from tests.xdart.scattering.test_direct_beam_readiness import _fit_container
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    qapp.processEvents()
    gate = threading.Event()
    try:
        table = _checklist(dialog)
        footer = dialog.orientation_button
        # no staged frames and no direct-beam fit: nothing to check, and each surface says so
        state, reason = _orientation_row(table)
        assert state == 'Needs input' and 'direct-beam' in reason.lower(), reason
        assert not footer.isEnabled() and reason in footer.toolTip()
        _activate(table, 7)
        assert 'Orientation check' not in _tabs(dialog) and _tab(dialog) == '2 · Rough geometry'
        assert dialog.status.text() == reason
        # a fresh direct-beam fit is a valid route without frames
        _fit_container(dialog, tmp_path, 270)
        assert _orientation_row(table)[0] == 'Optional' and footer.isEnabled()
        _press(footer)
        panel = dialog.orientation_panel
        assert dialog.tabs.currentWidget() is panel and panel.run_button.isEnabled()
        # a stale fit: one precise reason on the checklist, the footer, Run and the finding pane, not an older status
        dialog.status.setText('An unrelated earlier message.')
        strip = dialog._text('beam_strip')
        dialog._set_field_text('beam_strip', '3')
        assert _orientation_row(table) == ('Needs input', dialog._STALE_DISTANCE)
        assert not footer.isEnabled() and not panel.run_button.isEnabled()
        assert dialog._STALE_DISTANCE in footer.toolTip() and dialog._STALE_DISTANCE in panel.run_button.toolTip()
        assert panel.finding_pane.toPlainText() == dialog._STALE_DISTANCE
        _activate(table, 7)
        assert _tab(dialog) == '2 · Rough geometry' and dialog.status.text() == dialog._STALE_DISTANCE
        dialog._set_field_text('beam_strip', strip)
        assert _orientation_row(table)[0] == 'Optional' and footer.isEnabled() and panel.run_button.isEnabled()
        # preservation: the valid route runs
        dialog.tabs.setCurrentWidget(panel)
        _press(panel.run_button)
        _wait(dialog)
        assert panel.finding is not None
        # held at the window's real dispatcher: every surface waits; an edit during the check is not adopted, and
        # the late finish projects the current readiness instead of enabling Run
        real = dialog_module._dispatch_job

        def held(request, cancelled, publish):
            if request[0] == 'orientation':
                gate.wait(20)
            return real(request, cancelled, publish)
        monkeypatch.setattr(dialog_module, '_dispatch_job', held)
        _press(panel.run_button)
        assert _until(lambda: dialog._owner.busy)
        qapp.processEvents()
        assert _orientation_row(table)[0] == 'Waiting'
        assert not footer.isEnabled() and not panel.run_button.isEnabled() and panel.cancel_button.isEnabled()
        dialog._set_field_text('beam_strip', '3')
        gate.set()
        _wait(dialog)
        assert panel.finding is None and _orientation_row(table) == ('Needs input', dialog._STALE_DISTANCE)
        assert not panel.run_button.isEnabled() and not footer.isEnabled()
        assert panel.finding_pane.toPlainText() == dialog._STALE_DISTANCE
    finally:
        gate.set()
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()


def _staged(qapp, tmp_path):
    """Two staged LaB6 frames and a saved fixed arm, as test_orientation_diagnosis stages them; the arm not loaded."""
    import json
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xrd_tools import geometry_figures as gf
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.core.geometry import ImageOrientation
    from xrd_tools.core.geometry.camera import stitch_camera
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from tests.core.test_geometry_figures import reference
    value = gf.GeometryInputs()
    camera = stitch_camera(ImageOrientation())
    images = tmp_path / 'images'
    images.mkdir()
    for index, angle in enumerate((0., 15.)):
        _, image = reference(replace(value, readings=(('del', angle), ('nu', 0.))), camera)
        (images / f'rings_scan1_{index:04d}.raw').write_bytes(image.astype('<f4').tobytes())
    spec = tmp_path / 'rings'
    spec.write_text('#F rings\n#O0 del nu energy\n#S 1 ascan del 0 15 1 1\n'
                    f'#P0 0 0 {wavelength_m_to_energy_eV(value.wavelength_m)}\n#N 2\n#L del  nu\n0 0\n15 0\n')
    record = rough_arm_record(detector=value.detector, distance_m=value.distance_m, wavelength_m=value.wavelength_m,
                              beam_px=((value.shape[1] - 1) / 2 + value.beam_offset_px[0],
                                       (value.shape[0] - 1) / 2 + value.beam_offset_px[1]),
                              camera=ImageOrientation(), orientation=value.first_pixel)
    saved = tmp_path / 'fixed-arm.json'
    saved.write_text(json.dumps(record))
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images), 'image_stem': 'rings',
                        'read_image_kwargs': {'detector_shape': value.shape, 'raw_dtype': '<f4'}})
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    dialog.activateWindow()
    dialog.set_source_context(source, image_rotation=0)
    _wait(dialog)
    dialog._enumerate()
    _wait(dialog)
    dialog._prepare()
    _wait(dialog)
    assert len(dialog.frames) == 2, dialog.status.text()
    return dialog, saved, value


def test_c3_staged_route_names_the_arm_camera_motors_source_and_cached_mask(qapp, tmp_path, monkeypatch):
    dialog, saved, value = _staged(qapp, tmp_path)
    try:
        table = _checklist(dialog)
        footer = dialog.orientation_button
        editor = dialog.geometry_editor
        # staged frames but no fixed arm to check them against: refused before a run, with where to fix it
        state, reason = _orientation_row(table)
        assert state == 'Needs input' and 'rough geometry' in reason.lower(), reason
        assert not footer.isEnabled() and reason in footer.toolTip()
        _activate(table, 7)
        assert 'Orientation check' not in _tabs(dialog) and _tab(dialog) == '2 · Rough geometry'
        dialog.load_calibration(saved)
        dialog._source_details()
        _wait(dialog)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        assert _orientation_row(table)[0] == 'Optional' and footer.isEnabled()
        # another camera than the frames were staged with
        original = dialog.camera.value()
        dialog.camera.rotation.setCurrentIndex(dialog.camera.rotation.findData(90))
        assert _orientation_row(table) == ('Needs input', dialog._staged_camera_message())
        assert not footer.isEnabled()
        _activate(table, 7)
        assert _tab(dialog) == '1 · Instrument' and dialog.camera.rotation.hasFocus()
        dialog.camera.set_value(original)
        assert _orientation_row(table)[0] == 'Optional'
        # another motor mapping
        button, _card = _config(_box(_instrument(dialog), 'Diffractometer'))
        if button.text() == 'Config ▸':
            _press(button)
        _commit(editor.motor_edit('del'), 'del2')
        state, reason = _orientation_row(table)
        assert state == 'Needs input' and 'motor' in reason.lower() and not footer.isEnabled()
        _commit(editor.motor_edit('del'), 'del')
        assert _orientation_row(table)[0] == 'Optional'
        # another calibrant scan than the staged one
        _show(dialog, '3 · Calibrant scan')
        _commit(dialog._fields['scan'], '2')
        _wait(dialog)
        state, reason = _orientation_row(table)
        assert state == 'Needs input' and 'source' in reason.lower() and not footer.isEnabled()
        _activate(table, 7)
        assert _tab(dialog) == '3 · Calibrant scan'
        _show(dialog, '3 · Calibrant scan')
        _commit(dialog._fields['scan'], '1')
        _wait(dialog)
        assert _orientation_row(table)[0] == 'Optional'
        # an unreadable mask, known from the window's one cached read; edits project without reading it again
        loads = []
        read = dialog._oriented_mask

        def counted():
            loads.append(True)
            return read()
        monkeypatch.setattr(dialog, '_oriented_mask', counted)
        bad = tmp_path / 'bad-mask.npy'
        bad.write_text('not a mask')
        _instrument(dialog)
        _commit(dialog._fields['mask'], str(bad))
        _wait(dialog)
        state, reason = _orientation_row(table)
        assert state == 'Needs input' and 'mask' in reason.lower() and not footer.isEnabled(), reason
        _activate(table, 7)
        assert _tab(dialog) == '1 · Instrument' and dialog._fields['mask'].hasFocus()
        count = len(loads)
        for text in ('0.02', '0.03', ''):
            dialog._fields['rms'].setText(text)
        assert len(loads) == count and _orientation_row(table)[0] == 'Needs input'
        good = tmp_path / 'mask.npy'
        np.save(good, np.zeros(value.shape, bool))
        _instrument(dialog)
        _commit(dialog._fields['mask'], str(good))
        _wait(dialog)
        assert _orientation_row(table)[0] == 'Optional' and footer.isEnabled()
        # preservation: the valid staged route runs, and its finding figure keeps its axes' ticks
        _press(footer)
        panel = dialog.orientation_panel
        _press(panel.run_button)
        _wait(dialog)
        assert panel.finding is not None, panel.finding_pane.toPlainText()
        figure = panel.figure_view.keyed_panel(panel.figure_view.keyed_projection.panels[0].key)
        assert all(figure.plot.getAxis(axis).style['showValues'] for axis in ('bottom', 'left'))
    finally:
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()


# C4 · the instrument schematic ------------------------------------------------------------------------------------

def _schematic(qapp, window):
    window.grab()  # rendered: every label item holds its scene transform
    qapp.processEvents()
    view = window.figure_view
    projection = view.keyed_projection
    return projection, [view.keyed_panel(panel.key) for panel in projection.panels]


def _assert_schematic(qapp, window, value, pose):
    projection, panels = _schematic(qapp, window)
    assert len(panels) == 3
    titles = [panel.figure.title for panel in projection.panels]
    if pose is None:
        assert not any('zero-readback' in title for title in titles) and all('readback' in t for t in titles)
    else:
        assert all(pose in title for title in titles), titles
    captions = [panel.caption.text() for panel in panels]
    assert sum(bool(caption) for caption in captions) == 1, captions  # one shared explanation
    shared = next(caption for caption in captions if caption).lower()
    assert all(word in shared for word in ('beam', 'sample', 'detector', 'not to scale')), shared
    for figure, panel in zip(projection.panels, panels):
        assert all(panel.plot.getAxis(axis).style['showValues'] is False for axis in ('bottom', 'left'))
        labels = [item for item in panel.overlay_items if isinstance(item, pg.TextItem)]
        assert len(labels) >= len(value.circles)
        frame = panel.viewbox.sceneBoundingRect()
        rects = [item.sceneBoundingRect() for item in labels]
        for index, rect in enumerate(rects):
            text = labels[index].textItem.toPlainText()
            assert frame.contains(rect), (figure.figure.title, text, rect, frame)
            for other, label in zip(rects[index + 1:], labels[index + 1:]):
                assert not rect.intersects(other), (figure.figure.title, text, label.textItem.toPlainText())


@pytest.mark.parametrize('geometry', ['psic', 'fourc', 'twoc'])
def test_c4_schematic_separates_labels_names_its_pose_and_sizes_its_message_pane(qapp, tmp_path, geometry):
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    dialog.show()
    try:
        value = InstrumentGeometry(geometry, zeros=((InstrumentGeometry(geometry).circles[-1], 1.5),))
        dialog.set_instrument(value)
        _press(dialog.schematic_button)
        window = dialog.instrument_schematic
        assert window is not None and window.isVisible()
        _assert_schematic(qapp, window, value, 'zero-readback example')
        readings = {motor: 4. * (index + 1) for index, (_circle, motor) in enumerate(value.motors)}
        dialog.set_schematic_readbacks(readings)
        _assert_schematic(qapp, window, value, None)
        # each circle's readback, zero and resulting angle in the message pane
        lines = window.pane.toPlainText().splitlines()
        for circle in value.circles:
            reading, zero = readings[value.motor(circle)], value.zero(circle) or 0.
            line = next((line for line in lines if line.startswith(circle + ' ')), '')
            assert all(f'{number:g}' in line for number in (reading, zero, reading - zero)), (circle, lines)
        # 4–6 readable lines at first, in an adjustable split under the views
        pane = window.pane
        splitter = window.findChild(QtWidgets.QSplitter)
        assert splitter is not None and splitter.orientation() == QtCore.Qt.Orientation.Vertical
        assert splitter.indexOf(pane) == splitter.count() - 1 and splitter.indexOf(window.figure_view) >= 0
        assert 4 <= pane.viewport().height() / pane.fontMetrics().lineSpacing() < 7
        handle = splitter.handle(splitter.indexOf(pane))
        height = pane.height()
        centre = handle.rect().center()
        QtTest.QTest.mousePress(handle, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, centre)
        QtTest.QTest.mouseMove(handle, centre - QtCore.QPoint(0, 90))
        QtTest.QTest.mouseRelease(handle, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                                  centre - QtCore.QPoint(0, 90))
        qapp.processEvents()
        assert pane.height() > height + 45
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_c4_instrument_circles_carry_their_motor_readback_zero_and_role():
    from xrd_tools.geometry_figures import instrument_figure
    for geometry in ('psic', 'fourc', 'twoc'):
        base = InstrumentGeometry(geometry)
        value = replace(base, motors=tuple((circle, f'm_{circle}') for circle in base.circles),
                        zeros=tuple((circle, .25 * (index + 1)) for index, circle in enumerate(base.circles)
                                    if index % 2 == 0),
                        reversed_circles=(base.circles[-1],))
        readings = {f'm_{circle}': 3. * (index + 1) for index, circle in enumerate(base.circles)}
        figure = instrument_figure(value, readings)
        assert {circle.name for circle in figure.circles} == set(base.circles)
        for circle in figure.circles:
            assert (circle.motor, circle.role) == (value.motor(circle.name), value.role(circle.name))
            assert circle.readback == readings[circle.motor] and circle.zero == (value.zero(circle.name) or 0.)
            assert circle.angle_deg == circle.readback - circle.zero  # the zero subtracted once


def test_c4_ticks_hide_only_for_schematic_units_on_reused_keyed_panels(qapp):
    from xrd_tools import geometry_figures as gf
    from xdart.gui.tabs.calibration.orientation_views import show_figures
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    view = ScientificView()
    view.show()
    try:
        psic = InstrumentGeometry('psic')
        schematic = gf.instrument_figure(psic, {circle: 0. for circle in psic.circles}).panels
        show_figures(view, schematic)
        panels = [view.keyed_panel(panel.key) for panel in view.keyed_projection.panels]
        assert all(panel.plot.getAxis(axis).style['showValues'] is False for panel in panels
                   for axis in ('bottom', 'left'))
        # the same keyed panels reused by the help's pixel and residual figures keep their ticks
        others = (gf.first_pixel_figure(3, (5, 7)), tuple(gf.residual_example().values())[0], schematic[2])
        show_figures(view, others)
        reused = [view.keyed_panel(panel.key) for panel in view.keyed_projection.panels]
        assert [id(panel) for panel in reused] == [id(panel) for panel in panels]
        for panel, shown in zip(reused, (True, True, False)):
            assert all(panel.plot.getAxis(axis).style['showValues'] is shown for axis in ('bottom', 'left'))
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_c4_preservation_a_plain_read_only_window_keeps_one_pane_and_no_split(qapp):
    from xdart.gui.widgets.read_only_text import ReadOnlyTextWindow
    window = ReadOnlyTextWindow('Preview', (420, 300))
    try:
        label = QtWidgets.QLabel('mounted above')
        window.insert_content(label)
        assert window.findChild(QtWidgets.QSplitter) is None
        assert window.layout().indexOf(label) == 0 and window.layout().indexOf(window.pane) == 1
        window.set_text('one\ntwo')
        assert window.pane.toPlainText() == 'one\ntwo'
    finally:
        window.deleteLater()
        qapp.processEvents()
