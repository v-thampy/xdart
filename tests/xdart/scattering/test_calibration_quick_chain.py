"""Quick calibration (CALIBRATION-QUICK-CHAIN-20261008, M6): one action in 1 · Instrument, below the Diffractometer and
Detector sections (LT-009), that fits the direct-beam scan, reads the calibrant scan's metadata and refines automatically from the rough geometry, through the
window's own jobs, in the calibration window Stitch and RSM share. It stops at the first problem and names the step,
keeps what it completed, and never saves.

Every row opens the real window from its page and first finds the Quick calibration card where the window shows it,
so on a window without the card each row fails there, on what the window shows. The scans come from one truth (the
existing beam and ring fixtures are different detectors and distances, so they are not a pair): a direct-beam del scan
and a LaB6 ring scan rendered in the window's frame for the page's detector and camera, written raw beside their SPEC
files under the default naming, so a file and a scan number identify each."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtTest import QTest
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xrd_tools.core.energy import wavelength_m_to_energy_eV
from xrd_tools.core.geometry.camera import STITCH_CAMERA, UPRIGHT_CAMERA, DetectorCamera
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _page_with_dialog, _wait
from tests.xdart.scattering.test_browse_reference_rings_run import _pump

WAVELENGTH = .7293e-10
#: LaB6 frames: four nu rows of four del angles, as the RSM camera probe and the guided rows space them
RING_ANGLES = [(delta, nu) for nu in (-2., 0., 2., 4.) for delta in (8., 13., 18., 23.)]
ZEROS = {'del': 1.5, 'nu': -.8}
STOPPED = 'Quick calibration stopped at '
STEPS = {1: ('1 · Direct-beam fit', '2 · Rough geometry'), 2: ('2 · Calibrant metadata', '3 · Calibrant scan'),
         3: ('3 · Refine', '4 · Calibrate and fit')}


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _matched_pair(root, *, detector, camera, beam_px, calibrant_motors=('del', 'nu'), write_ring_frames=True,
                  ring_rows=None):
    """A direct-beam scan (del ±1° about its zero, nu held at 0) and a LaB6 ring scan from one truth: the product's
    rough arm for this detector seen through ``camera`` (the window's frame), distance 0.5 m, zeros ``ZEROS``."""
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import detector_factory
    from xrd_tools.core.geometry.camera import stitch_orientation
    from xrd_tools.integrate.calibration_seed import predict_frame_seed, rough_arm_record
    from tests.core.test_calibration_seed import _frame
    root.mkdir(parents=True)
    orientation = stitch_orientation(camera)
    shape = orientation.apply(np.zeros(tuple(detector_factory(detector).shape))).shape
    truth = rough_arm_record(distance_m=.5, detector=detector, beam_px=beam_px, zeros_deg=ZEROS,
                             wavelength_m=WAVELENGTH, camera=orientation)
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    rings = 2 * np.pi / np.asarray(calibrant.dspacing)[:25]

    def scan(name, angles, render, motors=('del', 'nu'), frames=True):
        for index, (delta, nu) in enumerate(angles if frames else ()):
            seed = predict_frame_seed(truth, _frame({'del': delta, 'nu': nu}, shape=shape), include_rings=False)
            q = seed.integrator.array_from_unit(shape, 'center', 'q_A^-1')
            orientation.unapply(render(q)).astype('<i4').tofile(root / f'{name}_scan1_{index:04d}.raw')
        rows = '\n'.join(f'{delta:.6g} {nu:.6g}' for delta, nu in angles)
        (root / name).write_text(f'#F {name}\n#E 1\n#D today\n#O0 {motors[0]} {motors[1]}\n'
                                 f'#S 1 ascan {motors[0]} {angles[0][0]:.6g} {angles[-1][0]:.6g} {len(angles) - 1} 1\n'
                                 f'#D today\n#P0 0 0\n#N 2\n#L {motors[0]}  {motors[1]}\n{rows}\n'
                                 if angles else
                                 f'#F {name}\n#E 1\n#D today\n#O0 {motors[0]} {motors[1]}\n'
                                 f'#S 1 ascan {motors[0]} 8 23 15 1\n#D today\n#P0 0 0\n#N 2\n#L {motors[0]}  {motors[1]}\n')
        return root / name

    beam = scan('beam', [(ZEROS['del'] + step, 0.) for step in np.linspace(-1., 1., 11)],
                lambda q: 10 + 4000 * np.exp(-(q / .006) ** 2 / 2))
    ring = scan('ring', RING_ANGLES if ring_rows is None else ring_rows,
                lambda q: 10 + 5000 * np.exp(-((q[..., None] - rings) ** 2) / (2 * .008 ** 2)).sum(-1),
                motors=calibrant_motors, frames=write_ring_frames)
    return SimpleNamespace(beam=beam, ring=ring, truth=truth)


def _stitch_pair(root, **kwargs):
    return _matched_pair(root, detector='Pilatus300kw', camera=UPRIGHT_CAMERA, beam_px=(737.5, 97.5), **kwargs)


def _rsm_pair(root, **kwargs):
    return _matched_pair(root, detector='Pilatus100k', camera=DetectorCamera('z-', 'x-'), beam_px=(97.5, 300.),
                         **kwargs)


def _rsm_window(qapp, project):
    """RSM's calibration window as its Calibrate opens it: Pilatus 100K, camera z−, x− (no saved settings)."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.rsm.page import RSMPage
    from xrd_tools.session.readiness import ControlAction
    from tests.xdart.pages.test_rsm_page import _Status
    page = RSMPage(empty_host_services(_Status()))
    page.set_field('project_root', str(project))
    page._control_action(ControlAction.CALIBRATE)
    dialog = page._calibration_dialog
    _wait(dialog)
    return page, dialog


def _window(qapp, project, page_name):
    if page_name == 'RSM':
        return _rsm_window(qapp, project)
    return _page_with_dialog(project)


def _close(qapp, page, dialog):
    from xdart.gui.pages.values import PageCleanup
    if dialog.isVisible():
        dialog.close()
    close = getattr(page, 'close_page', None) or page.shutdown  # RSM's page closes through close_page
    until = time.monotonic() + 10
    receipt = close()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = close()
    assert receipt.status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def _quick_card(dialog):
    """The one Quick calibration card, shown in 1 · Instrument below the Diffractometer and Detector sections
    (LT-SMALL-FIXES-20261014, LT-009: "Place Quick calibration below Diffractometer and Detector")."""
    tabs = dialog.tabs
    index = next(i for i in range(tabs.count()) if tabs.tabText(i) == '1 · Instrument')
    tabs.setCurrentIndex(index)
    QtWidgets.QApplication.processEvents()
    page = tabs.widget(index)
    cards = [card for card in page.findChildren(SectionCard) if card.title.text() == 'QUICK CALIBRATION']
    assert len(cards) == 1, 'no Quick calibration card in 1 · Instrument'
    card = cards[0]
    layout = card.parentWidget().layout()
    boxes = [box for box in page.findChildren(QtWidgets.QGroupBox)
             if box.title().split(' · ')[0] in ('Diffractometer', 'Detector')]
    assert card.isVisible() and len(boxes) == 2
    top = card.mapTo(page, QtCore.QPoint(0, 0)).y()
    for box in boxes:
        assert layout.indexOf(box) < layout.indexOf(card)
        assert top >= box.mapTo(page, QtCore.QPoint(0, box.height())).y()
    return card


def _card_rows(card, scan):
    """The card's file row for ``scan`` ('Direct-beam' or 'Calibrant') and the scan-number row beside it."""
    rows = [row for row in card.findChildren(FormRow)
            if row.label.text().startswith(scan) and row.browse_button is not None]
    assert len(rows) == 1, [row.label.text() for row in card.findChildren(FormRow)]
    number = [row for row in rows[0].findChildren(FormRow) if row.label.text() == 'Scan']
    assert len(number) == 1
    return rows[0], number[0]


def _quick_button(card):
    buttons = [b for b in card.findChildren(QtWidgets.QPushButton) if b.text() == 'Run quick calibration']
    assert len(buttons) == 1
    return buttons[0]


def _type(editor, text):
    """Type into a row as a person would, once the box takes typing (a running job disables the window's tabs): its
    whole text replaced, no commit yet."""
    until = time.monotonic() + 60
    while not editor.isEnabled() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    assert editor.isEnabled(), 'the box stayed disabled'
    editor.setFocus()
    QtWidgets.QApplication.processEvents()
    editor.selectAll()
    QTest.keyClicks(editor, text)


def _click(button):
    """Press a button as a person can: once it is enabled (a running job disables the window's tabs)."""
    until = time.monotonic() + 60
    while not button.isEnabled() and time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    assert button.isEnabled(), 'the button stayed disabled'
    button.click()


def _commit(editor, text):
    _type(editor, text)
    QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()


def _settle(seconds=.2):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.01)


def _configure(dialog, card, pair, *, camera=None, energy=True):
    """Everything as a person sets it: the camera, the two scans on the card, the motor zeros beside the motors and,
    as the scans record none, the energy on 3 · Calibrant scan."""
    if camera is not None:
        dialog.camera.set_value(camera)
    beam_file, beam_scan = _card_rows(card, 'Direct-beam')
    ring_file, ring_scan = _card_rows(card, 'Calibrant')
    _commit(beam_file.editor, str(pair.beam))
    _commit(beam_scan.editor, '1')
    _commit(ring_file.editor, str(pair.ring))
    _commit(ring_scan.editor, '1')
    for circle, zero in ZEROS.items():
        dialog.geometry_editor.zero_edit(circle).setText(f'{zero:g}')
    if energy:
        dialog._fields['rough_energy'].setText(f'{wavelength_m_to_energy_eV(WAVELENGTH) / 1000:.6f}')
    _wait(dialog)


def _starts(dialog, monkeypatch, action=None):
    """Record each job the window starts (a pass-through: the window's own ``_start`` runs unchanged), and run
    ``action(kind)`` right after a job starts, while it is still running (to cancel or close at that stage)."""
    started, original = [], dialog._start

    def recording(kind, arguments):
        busy = dialog._owner.busy
        original(kind, arguments)
        if not busy and dialog._owner.busy:
            started.append(kind)
            if action is not None:
                action(kind)
    monkeypatch.setattr(dialog, '_start', recording)
    return started


def _run(dialog, card, timeout=300):
    """Press Quick and return the status line as the chain left it (stopped or done): the window's own background
    checks may update it afterwards, as after any action; the card keeps the step and reason."""
    _click(_quick_button(card))
    until = time.monotonic() + timeout
    while time.monotonic() < until and not card.status.text().startswith(('Stopped', 'Done')):
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)
    status = dialog.status.text()
    _wait(dialog, timeout=timeout)
    return status


def _current_tab(dialog):
    return dialog.tabs.tabText(dialog.tabs.currentIndex())


def _assert_stopped(dialog, card, status, step, reason):
    """Stopped at ``step``: its tab shown, the step named with the reason on the status line as the chain left
    it (``status``) and on the card."""
    name, tab = STEPS[step]
    assert _current_tab(dialog) == tab
    assert status.startswith(STOPPED + name + ': '), status
    assert reason in status, status
    assert name in card.status.text() and reason in card.status.text(), card.status.text()
    assert _quick_button(card).isEnabled()


def _calibration_files(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob('*')
                  if path.is_file() and path.suffix in ('.json', '.poni') and 'session' not in path.name)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_quick_use_for_both_copies_only_the_effective_folder(qapp, tmp_path, page_name):
    """Separate SPEC locations keep Auto stems; either Quick folder can be copied once, including a draft."""
    from xdart.gui.widgets.controls_panel import ActionButton
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        _quick_card(dialog)
        rows = dialog._quick_rows
        buttons = {}
        for prefix in ('beam_', ''):
            found = [button for button in rows[prefix + 'images'].findChildren(ActionButton)
                     if button.text() == 'Use for both']
            assert len(found) == 1, 'each Quick Image Path needs its own Use for both action'
            buttons[prefix] = found[0]
            assert not buttons[prefix].isEnabled() and buttons[prefix].toolTip()
            for key in ('images', 'stem'):
                assert rows[prefix + key].current_value() == ''
                assert 'auto' in rows[prefix + key].editor.placeholderText().lower()
            spec = tmp_path / (prefix or 'calibrant') / 'scan'
            spec.parent.mkdir()
            spec.write_text('#F scan\n')
            _commit(rows[prefix + 'source'].editor, str(spec))
        _wait(dialog)
        # Copying Auto means the source SPEC's actual folder, not another blank Auto value.
        _click(buttons['beam_'])
        _wait(dialog)
        assert dialog._spec().options['image_dir'] == str(tmp_path / 'beam_')
        assert dialog._text('beam_images') == ''
        assert dialog._text('images') == str(tmp_path / 'beam_')
        assert (dialog._text('beam_stem'), dialog._text('stem')) == ('', '')
        # Reverse direction: the typed full path is used even before Return, and explicit stems survive.
        _commit(rows['beam_stem'].editor, 'direct_')
        _commit(rows['stem'].editor, 'powder_')
        common = tmp_path / 'all images'
        common.mkdir()
        _type(rows['images'].editor, str(common))
        _click(buttons[''])
        _wait(dialog)
        assert (dialog._text('images'), dialog._text('beam_images')) == (str(common), str(common))
        assert (dialog._text('beam_stem'), dialog._text('stem')) == ('direct_', 'powder_')
        # Editing either folder afterwards does not link them; clearing restores that role's Auto.
        _type(rows['images'].editor, '')
        QTest.keyClick(rows['images'].editor, QtCore.Qt.Key.Key_Backspace)
        QTest.keyClick(rows['images'].editor, QtCore.Qt.Key.Key_Return)
        _wait(dialog)
        assert dialog._text('beam_images') == str(common) and dialog._text('images') == ''
        assert dialog._spec().options['image_dir'] == str(tmp_path / 'calibrant')
        assert (dialog._text('beam_stem'), dialog._text('stem')) == ('direct_', 'powder_')
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_quick_card_placement_and_shared_values(qapp, tmp_path, monkeypatch, page_name):
    """Q1 (A1, A2): the card's two scans are the same values as 2 · Rough geometry's and 3 · Calibrant scan's,
    edited in either place, full paths kept behind the file name; a scan number edited on the card keeps tab 3's
    images, stem and typed energy, while a new file resets them as tab 3 does; the rows follow the tabs' rows and
    their 📁 is the tab's; a scan typed on the card and not yet committed is the one Quick uses; and the window saves
    no key of its own for the card."""
    from xdart.gui.widgets import file_dialogs
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        card = _quick_card(dialog)
        beam_file, beam_scan = _card_rows(card, 'Direct-beam')
        ring_file, ring_scan = _card_rows(card, 'Calibrant')
        assert [field for field in dialog._fields.values() if card.isAncestorOf(field)] == []
        assert card.toolTip() and _quick_button(card).toolTip()
        assert all(row.label.toolTip() for row in (beam_file, beam_scan, ring_file, ring_scan))
        source = tmp_path / 'data' / 'direct_beam_dist_p33m'
        source.parent.mkdir()
        source.write_text('#F direct\n')
        # tab 2 → card: the card shows the file's name, the full path on hover, and is the same value
        _commit(dialog._fields['beam_source'], str(source))
        _settle()
        assert beam_file.current_value() == str(source) == dialog._text('beam_source')
        assert beam_file.editor.text() == source.name and beam_file.editor.toolTip() == str(source)
        # card → tab 2
        _commit(beam_scan.editor, '7')
        assert dialog._text('beam_scan') == '7' == dialog._fields['beam_scan'].text()
        # card → tab 3, a scan number only: tab 3's images, stem and typed energy stay
        calibrant = tmp_path / 'data' / 'LaB6_11p27keV_dist_p33m'
        calibrant.write_text('#F lab6\n')
        _commit(ring_file.editor, str(calibrant))
        assert dialog._text('source') == str(calibrant) and dialog._text('scan') == '1' == ring_scan.current_value()
        for key, value in (('images', str(tmp_path / 'data')), ('stem', 'LaB6_'), ('rough_energy', '11.27')):
            _commit(dialog._fields[key], value)
        _commit(ring_scan.editor, '7')
        assert (dialog._text('scan'), dialog._text('images'), dialog._text('stem'), dialog._text('rough_energy')) == (
            '7', str(tmp_path / 'data'), 'LaB6_', '11.27')
        # card → tab 3, a new file: the scan, images, stem and typed energy reset, as a new file on tab 3 does
        other = tmp_path / 'data' / 'CeO2'
        other.write_text('#F ceo2\n')
        _commit(ring_file.editor, str(other))
        tab3 = (dialog._text('source'), dialog._text('scan'), dialog._text('images'), dialog._text('stem'),
                dialog._text('rough_energy'))
        assert tab3 == (str(other), '1', '', '', '')
        assert ring_scan.current_value() == '1'
        _commit(ring_file.editor, str(calibrant))
        _commit(ring_scan.editor, '7')
        _commit(dialog._fields['source'], str(other))  # the same edit on tab 3 itself
        assert (dialog._text('source'), dialog._text('scan'), dialog._text('images'), dialog._text('stem'),
                dialog._text('rough_energy')) == tab3
        assert ring_file.current_value() == str(other) and ring_scan.current_value() == '1'
        # NeXus files have no scan number; Quick always exposes its direct-beam route.
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('nexus_stack'))
        assert not ring_scan.isVisibleTo(card) and ring_file.label.text() == 'Calibrant NeXus file'
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('spec'))
        assert ring_scan.isVisibleTo(card) and ring_file.label.text() == 'Calibrant SPEC file'
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('nexus'))
        assert not beam_scan.isVisibleTo(card) and beam_file.label.text() == 'Direct-beam NeXus file'
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_series'))
        assert beam_scan.isVisibleTo(card) and beam_file.label.text() == 'Direct-beam SPEC file'
        for mode in ('manual', 'skip'):
            dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData(mode))
            assert beam_file.isVisibleTo(card) and ring_file.isVisibleTo(card)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('scan'))
        assert beam_file.isVisibleTo(card)
        # the card's 📁 is the tab's: SPEC files, or NeXus files for a NeXus source, and the pick is the tab's value
        picked = tmp_path / 'data' / 'picked'
        picked.write_text('#F picked\n')
        calls = []
        monkeypatch.setattr(file_dialogs, 'browse_spec', lambda *a, **k: calls.append('spec') or [str(picked)])
        monkeypatch.setattr(file_dialogs, 'browse', lambda *a, **k: calls.append(k.get('file_filter')) or [])
        _click(beam_file.browse_button)
        assert calls == ['spec'] and dialog._text('beam_source') == str(picked) == beam_file.current_value()
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('nexus_stack'))
        _click(ring_file.browse_button)
        from xdart.gui.tabs.calibration.dialog import NEXUS_FILTER
        assert calls == ['spec', NEXUS_FILTER]
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('spec'))
        _wait(dialog)
        # a scan typed on the card and not committed is the one Quick uses
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        _type(ring_scan.editor, '12')
        _click(_quick_button(card))
        _wait(dialog)
        assert dialog._text('scan') == '12'
        # the window saves the scans once, under the tabs' keys
        saved = dialog.settings()
        assert set(saved['fields']) == set(dialog._fields)
        assert (saved['fields']['beam_source'], saved['fields']['scan']) == (str(picked), '12')
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_quick_card_follows_restore_and_session_resume(qapp, tmp_path, monkeypatch, page_name):
    """Q1: a window opened on saved settings shows their scans on the card; opening a saved session (which restores
    frames and fit choices, not sources) leaves the card showing the scans the tabs show."""
    from xrd_tools.integrate.stitch_calibration import save_calibration_session
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _guided_frames
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    restored = None
    try:
        card = _quick_card(dialog)
        beam, ring = tmp_path / 'beam', tmp_path / 'ring'
        for path in (beam, ring):
            path.write_text('#F scan\n')
        for key, value in (('beam_source', str(beam)), ('beam_scan', '3'), ('source', str(ring)), ('scan', '9')):
            _commit(dialog._fields[key], value)
        restored = CalibrationDialog(CalibrationContext(page_name, tmp_path / 'project'))
        restored.apply_settings(dialog.settings())
        restored.show()
        restored_card = _quick_card(restored)
        for scan, (file_value, number) in (('Direct-beam', (str(beam), '3')), ('Calibrant', (str(ring), '9'))):
            file_row, scan_row = _card_rows(restored_card, scan)
            assert (file_row.current_value(), scan_row.current_value()) == (file_value, number)
        (tmp_path / 'frames').mkdir()
        frames = _guided_frames(tmp_path)
        (tmp_path / 'session-a').mkdir()
        path = save_calibration_session(tmp_path / 'session-a', frames, arm=('del', 'nu'),
                                        fit_policy=dialog._fit_policy())
        monkeypatch.setattr('xdart.gui.widgets.file_dialogs.browse', lambda *args, **kwargs: [str(path)])
        _wait(dialog)  # the reads the commits queued end first: Open saved session waits for them (CAL-HELD-BATCH)
        dialog._resume()
        _wait(dialog)
        assert len(dialog.frames) == len(frames)
        for scan, (source_key, scan_key) in (('Direct-beam', ('beam_source', 'beam_scan')),
                                             ('Calibrant', ('source', 'scan'))):
            file_row, scan_row = _card_rows(card, scan)
            assert (file_row.current_value(), scan_row.current_value()) == (
                dialog._text(source_key), dialog._text(scan_key))
    finally:
        if restored is not None:
            restored.close()
        _close(qapp, page, dialog)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_quick_card_shares_the_windows_threshold_and_mask(qapp, tmp_path, monkeypatch, page_name):
    """Maintainer, 2026-10-08: the card carries the window's Threshold (Int's shared row and rules, on by default:
    saturated pixels left out), the same value as 4 · Calibrate and fit's Advanced, edited in either place, with its
    tooltips. Its Mask file is the Detector panel's one row, exposed above the card (LT-SMALL-FIXES-20261014, LT-009:
    "remove its duplicate Mask file row"): the card holds none, and the window saves no key of its own for it."""
    from xdart.gui.widgets import file_dialogs
    from xdart.gui.widgets.controls_panel import RangeRow
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        card = _quick_card(dialog)
        rows = card.findChildren(RangeRow)
        assert len(rows) == 1 and rows[0] is not dialog.threshold_range
        threshold, window = rows[0], dialog.threshold_range
        assert threshold.label.text() == 'Threshold' and not card.isAncestorOf(window)
        assert threshold._toggle[1].isChecked() and dialog._conditioning() == StitchConditioning('auto')
        assert threshold.editors[1].text() == '' and 'Auto' in threshold.editors[1].placeholderText()
        assert threshold.label.toolTip() and all(box.toolTip() for box in threshold.editors)
        assert threshold._toggle[1].toolTip() == window._toggle[1].toolTip()
        assert '4294967295' in threshold._toggle[1].toolTip()
        _commit(threshold.editors[1], '50000')  # card → window
        band = dialog._conditioning()
        assert (band.mode, band.threshold_max) == ('manual', 50000.) and window.editors[1].text() == '50000'
        window._toggle[1].setChecked(False)  # window → card
        assert dialog._conditioning().mode == 'off' and not threshold._toggle[1].isChecked()
        threshold._toggle[1].setChecked(True)
        assert dialog._conditioning().mode == 'manual' and window._toggle[1].isChecked()
        assert not [row for row in card.findChildren(FormRow) if row.label.text() == 'Mask file']
        masks = [row for row in card.parentWidget().findChildren(FormRow) if row.label.text() == 'Mask file']
        assert len(masks) == 1 and masks[0].label.toolTip() and masks[0].editor is dialog._fields['mask']
        mask = masks[0]
        assert mask.isVisible() and not card.isAncestorOf(mask)
        drawn, picked = tmp_path / 'drawn.npy', tmp_path / 'picked.npy'
        for path in (drawn, picked):
            np.save(path, np.zeros((4, 4), bool))
        _commit(mask.editor, str(drawn))
        assert dialog._text('mask') == str(drawn) == mask.current_value()
        dialog._fields['mask'].clearFocus()  # a file row takes no set while it holds the focus
        calls = []
        monkeypatch.setattr(file_dialogs, 'browse', lambda *a, **k: calls.append(k.get('file_filter')) or [str(picked)])
        _click(mask.browse_button)
        assert calls == ['Masks (*.edf *.tif *.tiff *.npy);;All files (*)']
        assert dialog._text('mask') == str(picked) == mask.current_value()
        saved = dialog.settings()
        assert set(saved['fields']) == set(dialog._fields) and saved['fields']['mask'] == str(picked)
    finally:
        _close(qapp, page, dialog)


def test_quick_calibrates_the_real_17_2_pair_with_the_windows_defaults(qapp, tmp_path):
    """Q9 (maintainer, 2026-10-08): the 17-2 Fatigue pair, an Eiger2 500k on the upright mount and LaB6 at 11.27 keV.
    Quick with the window's own defaults (the threshold on) ends at Refine's summary, near the Oct 3 save (scan7_2).
    The threshold used to start off, and the detector's 28 invalid pixels a frame (4294967295) then reached the
    ring search: Refine refused (fit RMS 0.03906). One nu row leaves the distance and del zero degenerate with the
    beam position, so only runaways (349.94 mm, del zero on its -5° bound) are bounded against the save."""
    configured = os.environ.get('XDART_TEST_DATA')
    data = Path(configured or '') / 'stitching' / 'GIXRD_Fatigue_Radha'
    if not configured or not data.is_dir():
        pytest.skip('needs XDART_TEST_DATA with stitching/GIXRD_Fatigue_Radha')
    saved = json.loads((data / 'LaB6_11p27keV_dist_p33m_scan7_2.json.provenance.json').read_text())['fit_bounds']
    before = {path.name: path.stat().st_mtime_ns for path in data.iterdir()}
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        dialog.camera.set_value(UPRIGHT_CAMERA)
        dialog._set_field_text('beam_detector', 'Eiger2_500k')
        dialog.beam_source_kind.setCurrentIndex(dialog.beam_source_kind.findData('spec_container'))
        dialog.source_kind.setCurrentIndex(dialog.source_kind.findData('spec_container'))
        beam_file, beam_scan = _card_rows(card, 'Direct-beam')
        ring_file, ring_scan = _card_rows(card, 'Calibrant')
        _commit(beam_file.editor, str(data / 'direct_beam_dist_p33m'))
        _commit(beam_scan.editor, '1')
        _commit(ring_file.editor, str(data / 'LaB6_11p27keV_dist_p33m'))
        _commit(ring_scan.editor, '7')
        for key in ('beam_images', 'images'):
            _commit(dialog._fields[key], str(data))
        _wait(dialog)
        status = _run(dialog, card, timeout=600)
        assert card.status.text().startswith('Done'), (status, card.status.text())
        fitted = dialog.fit.asset['provenance']['fit_bounds']['parameters']
        print(f"Q9: rms {dialog.fit.rms_q:.6f} points {dialog.fit.n_points} distance "
              f"{fitted['distance_mm']['fitted']:.4f} mm del zero {fitted['del_zero_deg']['fitted']:.4f}°")
        assert dialog.fit.rms_q < .003 and dialog.fit.n_points >= 3000
        for name in ('distance_mm', 'del_zero_deg', 'nu_zero_deg'):
            assert abs(fitted[name]['fitted'] - saved['parameters'][name]['fitted']) < 1., (name, fitted[name])
            assert not fitted[name]['at_bound'], (name, fitted[name])
        assert _calibration_files(tmp_path / 'project') == []
    finally:
        _close(qapp, page, dialog)
    assert {path.name: path.stat().st_mtime_ns for path in data.iterdir()} == before


@pytest.mark.parametrize('clear_imported_beam', (False, True))
def test_quick_rerun_keeps_loaded_energy_offset_and_finds_the_scan_strip(qapp, tmp_path, clear_imported_beam):
    """A fitted wavelength is not the recorded energy; the normal-incidence pixel is not a scan strip."""
    pair = _stitch_pair(tmp_path / 'data')
    recorded = float(wavelength_m_to_energy_eV(WAVELENGTH)) - 40.
    for source in (pair.beam, pair.ring):
        source.write_text(source.read_text().replace('#O0 del nu', '#O0 del  nu  energy').replace(
            '#P0 0 0', f'#P0 0 0 {recorded!r}'))
    geometry = tmp_path / 'loaded.json'
    geometry.write_text(json.dumps(dict(pair.truth, source_energy_eV=recorded)))
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA, energy=False)
        assert dialog._source_energy_eV == pytest.approx(recorded)
        geometry.with_suffix('.json.provenance.json').write_text(json.dumps({
            'image_rotation': dialog._orientation().rotation,
            'image_flip': dialog._orientation().flip_horizontal,
            'instrument_geometry': dialog.geometry_editor.value().to_record(),
            'source_motors': {'del_angle': 'del', 'nu_angle': 'nu'},
        }))
        dialog.load_calibration(geometry)
        dialog.refine_energy.setChecked(False)
        if clear_imported_beam:  # Independently expose the energy defect after the original step-1 refusal.
            dialog._fields['beam_pixel'].selectAll()
            QTest.keyClick(dialog._fields['beam_pixel'], QtCore.Qt.Key.Key_Backspace)
        _wait(dialog)
        status = _run(dialog, card)
        assert card.status.text().startswith('Done'), (status, card.status.text())
        assert dialog._text('rough_energy') == ''
        assert dialog._source_energy_eV == pytest.approx(recorded)
        assert dialog._distance_result.used.all()
        calibration = dialog.fit.asset['calibration']
        assert calibration['wavelength'] == pytest.approx(WAVELENGTH, rel=1e-12, abs=0)
        assert calibration['source_energy_eV'] == pytest.approx(recorded)
        assert 'energy_refinement' not in dialog.fit.asset['provenance']
        assert dialog.fit.rms_q < .005
        assert dialog._calibration_fit_window.pattern_view.status.text() == 'Calibrant preview complete.'
        # A deliberate acquisition override is retained, and genuinely different recorded inputs still refuse.
        _commit(dialog._fields['rough_energy'], f'{(recorded + 500.) / 1000:.12g}')
        status = _run(dialog, card)
        assert card.status.text().startswith('Stopped at 3'), status
        assert 'source energy and calibration wavelength differ' in card.status.text()
    finally:
        _close(qapp, page, dialog)


def test_quick_chain_resumes_then_saves_and_uses_geometry(qapp, tmp_path, monkeypatch):
    """Q2: on a matched pair Quick fits the direct-beam scan (no motion warning), reads the calibrant's metadata and
    refines from the rough geometry. Its common session restores the completed fit without rerunning it; the
    explicit finish action saves and adopts geometry only after its session can be written. Cancel and a write
    failure keep both windows open. Numbers: the rough-start Refine row's approved tolerances."""
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    saved = []
    try:
        card = _quick_card(dialog)
        dialog.calibrationSaved.connect(lambda *args: saved.append(args))
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('fit'))
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        before = dialog.settings()
        started = _starts(dialog, monkeypatch)
        _quick_button(card).click()
        until = time.monotonic() + 300
        while dialog.fit is None and time.monotonic() < until and not dialog.status.text().startswith(STOPPED):
            QtWidgets.QApplication.processEvents()
            time.sleep(.005)
        assert dialog.fit is not None, dialog.status.text()
        assert card.status.text().startswith('Done'), card.status.text()  # done at Refine's end, not the preview's
        _wait(dialog, timeout=300)
        assert [kind for kind in started if kind in ('distance', 'enumerate', 'refine')] == [
            'distance', 'enumerate', 'refine']
        assert dialog._distance_warning is None and dialog._distance_fresh()
        assert dialog._calibrant_source_ready() and len(dialog._choices) == len(RING_ANGLES)
        assert _current_tab(dialog) == '4 · Calibrate and fit' and dialog.fit_summary.toPlainText()
        from xdart.gui.tabs.calibration.dialog import calibration_fit_text
        assert dialog.status.text() == calibration_fit_text(dialog.fit, 'refine')
        window = dialog._calibration_fit_window
        assert window is not None and window.isVisible()
        assert window.pattern_view.status.text() == 'Calibrant preview complete.'
        assert dialog.refine_seed.currentData() == 'rough'
        after = dialog.settings()
        changed = {key for key in set(before['fields']) | set(after['fields'])
                   if before['fields'].get(key) != after['fields'].get(key)}
        assert changed <= {'refine_seed_path'} and after['camera'] == before['camera']
        assert after['instrument_geometry'] == before['instrument_geometry']
        assert {name: choice for name, choice in after['choices'].items() if before['choices'][name] != choice} == {
            'refine_seed': {'data': 'rough', 'text': 'Rough geometry (tab 2)'}}
        assert saved == [] and _calibration_files(tmp_path / 'project') == []
        refined = dict(zip(dialog.fit.asset['calibration']['param_names'], dialog.fit.asset['calibration']['param']))
        truth = dict(zip(pair.truth['param_names'], pair.truth['param']))
        assert refined['dist'] == pytest.approx(truth['dist'], abs=1e-3)
        assert refined['del_offset'] == pytest.approx(truth['del_offset'], abs=2e-3)
        assert refined['nu_offset'] == pytest.approx(truth['nu_offset'], abs=2e-3)
        assert dialog.fit.rms_q < .005, dialog.fit.rms_q
        # A normal calibration session retains Quick's shared settings and completed fit, not only staged frames.
        session = dialog.session_dir / 'calibration-session.json'
        assert session.is_file(), 'Quick must save a resumable session even without staged frames'
        original = dialog.fit
        original_beam = dialog._distance_result
        dialog.refine_calibrant.setCurrentText('CeO2')
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        from xdart.gui.widgets import file_dialogs
        with monkeypatch.context() as picker:
            picker.setattr(file_dialogs, 'browse', lambda *args, **kwargs: [str(session)])
            dialog.open_session_button.click()
            _pump(qapp, lambda: dialog.fit is not None and not dialog.active())
        assert dialog.fit is not original
        np.testing.assert_array_equal(dialog.fit.asset['calibration']['param'], original.asset['calibration']['param'])
        np.testing.assert_array_equal(dialog.fit.point_residuals['dq'], original.point_residuals['dq'])
        assert dialog.refine_calibrant.currentText() == dialog.quick_calibrant.editor.currentText() == 'LaB6'
        assert dialog.rough_mode.currentData() == 'scan' and dialog._distance_fresh()
        assert dialog._distance_result is not original_beam
        assert started.count('refine') == started.count('distance') == 1, 'resume never reruns either fit'
        window = dialog._calibration_fit_window
        assert window.isVisible() and window.panels.currentWidget() is window.view
        # Finish refuses an unencodable session before asking for a path or starting the pair writer.
        from dataclasses import replace
        retained_fit, retained_dir = dialog.fit, dialog.session_dir
        retained_identity = dialog.settings()['loaded_calibration']
        prior_session = session.read_bytes()
        dialog.session_dir = retained_dir / 'not-created-by-validation'
        dialog.fit = replace(retained_fit, message='x' * (2 << 20))
        picks = []
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                            staticmethod(lambda *args: (picks.append(args) or '', '')))
        window.save_button.click()
        assert picks == [] and started.count('save') == 0
        assert 'calibration session exceeds the size limit' in dialog.status.text()
        assert dialog.isVisible() and window.isVisible() and not dialog.active()
        assert not dialog.session_dir.exists() and session.read_bytes() == prior_session
        assert not saved and _calibration_files(tmp_path / 'project') == []
        # A long old output identity must not block saving the same data to a shorter new path.
        dialog.fit = replace(retained_fit, message='')
        short_target = dialog.project_root / 'x.json'
        short_size = len(dialog._save_session(validate_only=True, selected_path=short_target))
        dialog.fit = replace(retained_fit, message='x' * ((2 << 20) - short_size - 8))
        old_identity = (str(dialog.project_root / ('old-calibration-' * 12 + '.json')), 'a' * 64, 'b' * 64)
        dialog._loaded_calibration_identity = old_identity
        assert len(dialog._save_session(validate_only=True)) <= 2 << 20
        assert len(dialog._save_session(validate_only=True, selected_path=short_target)) <= 2 << 20
        assert dialog.settings()['loaded_calibration'] == old_identity
        dialog._loaded_calibration_identity = retained_identity
        # A selected output path adds its identity to the final session. Check that growth before writing too.
        dialog.fit = replace(retained_fit, message='')
        current = dialog._save_session(validate_only=True)
        dialog.fit = replace(retained_fit, message='x' * ((2 << 20) - len(current) - 16))
        target = dialog.project_root / ('long-calibration-name-' * 4 + '.json')
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', staticmethod(lambda *args: (str(target), '')))
        assert len(dialog._save_session(validate_only=True)) <= 2 << 20
        window.save_button.click()
        assert 'calibration session exceeds the size limit' in dialog.status.text()
        assert started.count('save') == 0 and not target.exists()
        assert not dialog.session_dir.exists() and session.read_bytes() == prior_session
        assert dialog.settings()['loaded_calibration'] == retained_identity
        assert not saved and dialog.isVisible() and window.isVisible() and not dialog.active()
        dialog.fit, dialog.session_dir = retained_fit, retained_dir
        # Cancel and a real session-write failure keep both windows and the result available.
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', staticmethod(lambda *args: ('', '')))
        window.save_button.click()
        assert dialog.isVisible() and window.isVisible() and not dialog.active()
        target = dialog.project_root / 'finished.json'
        monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', staticmethod(lambda *args: (str(target), '')))
        blocked = session.with_suffix('.tmp')
        blocked.mkdir()
        window.save_button.click()
        _wait(dialog)
        assert dialog.isVisible() and window.isVisible() and dialog.fit is not None
        assert not saved and 'calibration-session' in dialog.status.text()
        blocked.rename(blocked.with_suffix('.blocked'))
        window.save_button.click()
        _wait(dialog)
        assert len(saved) == 1 and Path(page._value('geometry_path')) == target
        assert not dialog.isVisible() and not window.isVisible()
        from xrd_tools.integrate.stitch_calibration import load_calibration_session
        _frames, _arm, _policy, restored = load_calibration_session(session, with_resume=True)
        assert restored.fit is not None and restored.matches_source(dialog._spec())
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('problem', ['refused', 'stale', 'stale_after_metadata', 'camera', 'sense'])
def test_quick_stops_at_direct_beam_problem(qapp, tmp_path, monkeypatch, problem):
    """Q3: a direct-beam fit that cannot run, one whose inputs changed while it ran, one made stale by reading the
    calibrant's metadata (its motor names re-pick the arm, A4), and the motion advice for a turned camera or a
    reversed sense each stop the chain at step 1 on 2 · Rough geometry with the reason; nothing later runs."""
    pair = _stitch_pair(tmp_path / 'data', calibrant_motors=('delta', 'nu') if problem == 'stale_after_metadata'
                        else ('del', 'nu'), write_ring_frames=problem == 'stale_after_metadata')
    if problem == 'refused':
        for frame in pair.beam.parent.glob('beam_scan1_*.raw'):
            frame.unlink()
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        camera = STITCH_CAMERA if problem == 'camera' else UPRIGHT_CAMERA
        _configure(dialog, card, pair, camera=camera)
        if problem == 'sense':
            box = dialog.geometry_editor.axis_box('del')
            box.setCurrentIndex(box.findData(-1))

        def change_inputs(kind):
            if kind == 'distance':  # while the fit runs, the del zero changes (as a page's instrument push would)
                dialog.geometry_editor.zero_edit('del').setText('0.7')
        started = _starts(dialog, monkeypatch, change_inputs if problem == 'stale' else None)
        status = _run(dialog, card)
        reason = {'refused': '', 'stale': 'fit the direct-beam distance again',
                  'stale_after_metadata': 'fit the direct-beam distance again',
                  'camera': 'mounted the other way round', 'sense': 'mounted the other way round'}[problem]
        _assert_stopped(dialog, card, status, 1, reason)
        assert 'refine' not in started and dialog.fit is None
        if problem == 'stale_after_metadata':
            assert started.count('enumerate') == 1 and 'metadata' in status
        else:
            assert 'enumerate' not in started and dialog._choices == []
        if problem in ('camera', 'sense'):  # the fit itself is kept, with its advice
            assert dialog._distance_result is not None and dialog._distance_warning
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('problem', ['missing_energy', 'unreadable', 'zero_frames'])
def test_quick_stops_at_calibrant_problem(qapp, tmp_path, monkeypatch, problem):
    """Q4: no energy anywhere (no typed energy, neither scan records one), an unreadable calibrant and a calibrant
    scan with no frames (which the metadata read itself refuses) each stop at step 2 on 3 · Calibrant scan; the
    direct-beam fit is kept; Refine never runs."""
    pair = _stitch_pair(tmp_path / 'data', write_ring_frames=problem == 'missing_energy',
                        ring_rows=[] if problem == 'zero_frames' else None)
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA, energy=problem != 'missing_energy')
        if problem == 'unreadable':
            ring_file, _ = _card_rows(card, 'Calibrant')
            _commit(ring_file.editor, str(tmp_path / 'data' / 'missing_spec'))
            _wait(dialog)
        started = _starts(dialog, monkeypatch)
        status = _run(dialog, card)
        reason = {'missing_energy': 'type Energy (keV) in 3 · Calibrant scan', 'unreadable': '',
                  'zero_frames': ''}[problem]  # the metadata read's own refusal, whatever it says
        _assert_stopped(dialog, card, status, 2, reason)
        assert 'refine' not in started and dialog.fit is None
        assert started.count('distance') == 1 and dialog._distance_result is not None and dialog._distance_fresh()
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('problem', ['rms_cap', 'energy_bound'])
def test_quick_keeps_refine_refusal(qapp, tmp_path, monkeypatch, problem):
    """Q5: Refine's own refusals stop at step 3 on 4 · Calibrate with Refine's reason, the direct-beam fit and the
    metadata kept, nothing written. The RMS cap is the window's own RMS field (a typed input; chosen over a mismatched
    energy because its refusal does not depend on how far the fit lands); a seed-arm mismatch cannot arise from Rough
    (A6)."""
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        if problem == 'rms_cap':
            dialog._fields['rms'].setText('1e-6')
        else:
            dialog.refine_energy.setChecked(True)
            dialog._fields['energy_bound'].setText('-5')
        started = _starts(dialog, monkeypatch)
        status = _run(dialog, card)
        reason = {'rms_cap': 'exceeds', 'energy_bound': 'type the energy bound as a positive number of eV'}[problem]
        _assert_stopped(dialog, card, status, 3, reason)
        assert dialog.fit is None and dialog._distance_result is not None and dialog._calibrant_source_ready()
        assert started.count('distance') == started.count('enumerate') == 1
        assert started.count('refine') == (1 if problem == 'rms_cap' else 0)
        assert _calibration_files(tmp_path / 'project') == []
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('stage', ['distance', 'enumerate', 'refine'])
def test_cancel_stops_quick_chain(qapp, tmp_path, monkeypatch, stage):
    """Q6: Cancel operation during a stage stops the chain there, keeps what finished before it and starts nothing
    after it; the step's own button then works by hand."""
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        started = _starts(dialog, monkeypatch, lambda kind: kind == stage and dialog.cancel_button.click())
        status = _run(dialog, card)
        step = {'distance': 1, 'enumerate': 2, 'refine': 3}[stage]
        _assert_stopped(dialog, card, status, step, 'Cancelled')
        order = ['distance', 'enumerate', 'refine']
        assert [kind for kind in started if kind in order] == order[:step]
        assert dialog.fit is None
        button = {'distance': dialog.beam_button, 'enumerate': dialog.list_button, 'refine': dialog.refine_button}[stage]
        if stage == 'refine':
            dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData('rough'))
        if stage == 'distance':  # its readiness first, as a person waits for it
            dialog._detect_beam()
            _wait(dialog)
        _click(button)
        _wait(dialog, timeout=300)
        assert started.count(stage) == 2 and not dialog.status.text().startswith(STOPPED)
    finally:
        _close(qapp, page, dialog)


def test_quick_waits_through_the_opening_refresh_and_probe(qapp, tmp_path, monkeypatch):
    """Q6 busy (A3): pressed as the page opens the window on saved settings, before the source refresh and the
    direct-beam check that restoring queues have run, Quick waits through them and runs each step once."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _Status
    pair = _stitch_pair(tmp_path / 'data')
    first, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        _configure(dialog, _quick_card(dialog), pair, camera=UPRIGHT_CAMERA)
        settings = dialog.settings()
    finally:
        _close(qapp, first, dialog)
    page = StitchPage(empty_host_services(_Status()))
    page.set_field('project_root', str(tmp_path / 'project'))
    page._calibration_settings = settings
    started = []
    from xdart.gui.tabs.calibration import dialog as dialog_module
    original = dialog_module.CalibrationDialog._start

    def recording(self, kind, arguments):  # pass-through: records each job the window starts
        busy = self._owner.busy
        original(self, kind, arguments)
        if not busy and self._owner.busy:
            started.append(kind)
    monkeypatch.setattr(dialog_module.CalibrationDialog, '_start', recording)
    page._calibrate()
    dialog = page._calibration_dialog
    try:
        assert dialog._pending_source_details is not None and not dialog._owner.busy  # restoring queued its refresh
        card, = [card for card in dialog.findChildren(SectionCard) if card.title.text() == 'QUICK CALIBRATION']
        button = _quick_button(card)
        assert button.isEnabled()
        button.click()  # no event processed between opening the window and the press
        _wait(dialog, timeout=300)
        assert dialog.fit is not None, dialog.status.text()
        assert [kind for kind in started if kind in ('distance', 'enumerate', 'refine')] == [
            'distance', 'enumerate', 'refine']
        assert 'source_details' in started and 'beam_detect' in started
    finally:
        _close(qapp, page, dialog)


def test_closing_the_window_mid_chain_starts_nothing_more(qapp, tmp_path, monkeypatch):
    """Q6 close: closing the window while Quick reads the metadata cancels that step, starts no later one and closes
    the window; nothing is written."""
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        started = _starts(dialog, monkeypatch, lambda kind: kind == 'enumerate' and dialog.close())
        _quick_button(card).click()
        until = time.monotonic() + 120
        while (dialog.isVisible() or dialog.active()) and time.monotonic() < until:
            QtWidgets.QApplication.processEvents()
            time.sleep(.005)
        _settle()
        assert not dialog.isVisible() and not dialog.active()
        assert 'refine' not in started and dialog.fit is None
        assert _calibration_files(tmp_path / 'project') == []
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('outcome', ['success', 'stop'])
def test_quick_preserves_rsm_defaults(qapp, tmp_path, monkeypatch, outcome):
    """Q7: in RSM's window, on its own Pilatus 100K and camera z−, x−, Quick runs on a pair made in that frame (and
    stops at step 2 without an energy), leaving RSM's detector, camera, project and Calibration row its own."""
    pair = _rsm_pair(tmp_path / 'data')
    page, dialog = _rsm_window(qapp, tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        assert (dialog._text('beam_detector'), tuple(dialog.camera.value().to_xu())) == ('Pilatus100k', ('z-', 'x-'))
        geometry_path = page._value('geometry_path')
        _configure(dialog, card, pair, energy=outcome == 'success')
        status = _run(dialog, card)
        if outcome == 'success':
            assert dialog.fit is not None, dialog.status.text()
            assert card.status.text().startswith('Done') and _current_tab(dialog) == '4 · Calibrate and fit'
            truth = dict(zip(pair.truth['param_names'], pair.truth['param']))
            refined = dict(zip(dialog.fit.asset['calibration']['param_names'], dialog.fit.asset['calibration']['param']))
            assert refined['dist'] == pytest.approx(truth['dist'], abs=1e-3)
            assert dialog.fit.rms_q < .005, dialog.fit.rms_q
        else:
            _assert_stopped(dialog, card, status, 2, 'type Energy (keV) in 3 · Calibrant scan')
        assert (dialog._text('beam_detector'), tuple(dialog.camera.value().to_xu())) == ('Pilatus100k', ('z-', 'x-'))
        assert dialog.project_root == (tmp_path / 'project').resolve()
        assert page._value('geometry_path') == geometry_path
        assert _calibration_files(tmp_path / 'project') == []
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('mode', ['manual', 'skip'])
def test_quick_explicitly_selects_the_direct_beam_route(qapp, tmp_path, monkeypatch, mode):
    """LF-008 supersedes D1: Quick selects its beam route; manual inputs remain available in their tab."""
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData(mode))
        if mode == 'manual':
            _commit(dialog._fields['rough_distance'], '500')
        _wait(dialog)
        started = _starts(dialog, monkeypatch)
        status = _run(dialog, card)
        assert dialog.rough_mode.currentData() == 'scan'
        assert status.startswith(STOPPED + STEPS[1][0]), 'missing beam input still stops at the beam step'
        assert 'distance' not in started and 'refine' not in started
        assert dialog._text('rough_distance') == ('500' if mode == 'manual' else '')
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_quick_groups_project_the_shared_calibrant(qapp, tmp_path, page_name):
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        fields = set(dialog.settings()['fields'])
        choices = set(dialog.settings()['choices'])
        card = _quick_card(dialog)
        headers = [label for label in card.findChildren(QtWidgets.QLabel)
                   if label.objectName() == 'controlsGroupHeader']
        assert [label.text() for label in headers] == [
            '1 · Direct-beam scan', '2 · Calibrant scan', '3 · Conditioning']
        assert all(label.toolTip() for label in headers)
        assert not any(row.label.text() == 'Rough geometry' for row in card.findChildren(FormRow))
        row = dialog.quick_calibrant
        assert isinstance(row, FormRow) and row.toolTip() and row.label.toolTip()
        assert row.editor.currentText() == dialog.refine_calibrant.currentText() == 'LaB6'
        _commit(row.editor.lineEdit(), 'CeO2')
        assert dialog.refine_calibrant.currentText() == 'CeO2'
        dialog.refine_calibrant.setCurrentText('custom powder')
        assert row.editor.currentText() == 'custom powder', 'custom names use the same editable owner'
        dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('skip'))
        assert dialog._quick_rows['beam_source'].isVisible()
        assert set(dialog.settings()['fields']) == fields
        assert set(dialog.settings()['choices']) == choices, 'the projection adds no saved mode of its own'
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize('page_name', ['Stitch', 'RSM'])
def test_tabbing_from_the_card_file_to_its_scan_shows_the_reset_scan(qapp, tmp_path, page_name):
    """Q1, added after Claude's self-review of 6d00dc8b: a new calibrant file typed on the card and left with Tab
    resets the scan to 1 (as tab 3 does), and the card's scan box, which now has the focus, shows that 1 rather than
    the scan the old file had (otherwise Quick would run on scan 1 under a card showing another)."""
    page, dialog = _window(qapp, tmp_path / 'project', page_name)
    try:
        card = _quick_card(dialog)
        ring_file, ring_scan = _card_rows(card, 'Calibrant')
        (tmp_path / 'data').mkdir()
        first, second = tmp_path / 'data' / 'LaB6', tmp_path / 'data' / 'CeO2'
        for path in (first, second):
            path.write_text('#F scan\n')
        _commit(ring_file.editor, str(first))
        _commit(ring_scan.editor, '7')
        _wait(dialog)
        assert dialog._text('scan') == '7' == ring_scan.current_value()
        dialog.activateWindow()
        ring_file.editor.setFocus()
        qapp.processEvents()
        assert ring_file.editor.hasFocus()
        ring_file.editor.selectAll()
        QTest.keyClicks(ring_file.editor, str(second))
        ring_scan.editor.setFocus()  # Tab: the file commits as its box loses the focus (the source refresh it
        qapp.processEvents()         # starts then disables the tabs, so the scan box does not keep the focus)
        assert dialog._text('source') == str(second)
        assert dialog._text('scan') == '1' == ring_scan.current_value() == ring_scan.editor.text()
        _wait(dialog)
    finally:
        _close(qapp, page, dialog)


def test_quick_stops_with_its_reason_when_rough_geometry_leaves_the_scan_mid_chain(qapp, tmp_path, monkeypatch):
    """Q8, added after Claude's self-review of 6d00dc8b: Rough geometry switched to Enter the distance while Quick's
    direct-beam check runs (the tabs stay enabled during that check) stops it at step 1 with D1's reason, not the
    check's interrupted status."""
    pair = _stitch_pair(tmp_path / 'data')
    page, dialog = _page_with_dialog(tmp_path / 'project')
    try:
        card = _quick_card(dialog)
        _configure(dialog, card, pair, camera=UPRIGHT_CAMERA)
        dialog.camera.set_value(STITCH_CAMERA)  # an edit and back: the direct-beam check must run again
        dialog.camera.set_value(UPRIGHT_CAMERA)

        def leave_scan(kind):
            if kind == 'beam_detect':
                dialog.rough_mode.setCurrentIndex(dialog.rough_mode.findData('manual'))
        started = _starts(dialog, monkeypatch, leave_scan)
        status = _run(dialog, card)
        _assert_stopped(dialog, card, status, 1, 'choose "Direct-beam scan" in 2 · Rough geometry')
        assert started.count('beam_detect') == 1 and 'distance' not in started
    finally:
        _close(qapp, page, dialog)
