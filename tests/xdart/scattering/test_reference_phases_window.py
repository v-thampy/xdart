"""REFERENCE-PHASES-20261011 frozen GUI oracle RW1-RW8 (Claude, October 11).

The Tools-panel Reference Phases window on Int's and Stitch's real pages, per-phase isolation on every pane,
the legend toggle, calibrants on the phase path with P3a's semantics, persistence, the no-pymatgen case and the
Phase Fitter's per-phase cutoffs. Real Runs, real key and mouse events, the real worker and the real fit; no
painter, listing or fit is replaced. New APIs are reached inside test bodies.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import run_page, select
from tests.xdart.scattering.test_phase_reference_overlays import _cif, _navigation, _settle, _stitch
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

CAVEAT = "radial agreement alone does not identify detector mounting"
LAB6 = "calibrant:LaB6"


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phase-library'))


def _window(qapp, view):
    view.show_reference_phases()
    qapp.processEvents()
    window = view.reference_phases
    assert window.isVisible() and window.windowTitle() == 'Reference Phases'
    return window


def _commands(view):
    seen = []
    view.commandRequested.connect(seen.append)
    return seen


def _row(source, identifier):
    scans = source.browser.scans
    index = next(index for index, entry in enumerate(source.listed_entries()) if entry[0].id == identifier)
    return scans.item(index)


def _check(qapp, source, identifier, checked):
    _row(source, identifier).setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
    qapp.processEvents()


def _type(qapp, editor, text):
    editor.window().activateWindow()
    editor.setFocus()
    qapp.processEvents()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()


def _options(page):
    return page._preferences.plot_options if hasattr(page, '_preferences') else page._view.plot_options


def _marker(view, identifier):
    items = [item for item in view.curve.getPlotItem().items if item.data(0) == 'phase:' + identifier]
    assert len(items) <= 1
    return items[0] if items else None


def _legend(view, name):
    entries = [(sample, label) for sample, label in view.legend.items if label.text == name]
    assert len(entries) == 1, [label.text for _sample, label in view.legend.items]
    return entries[0][0]


def _pane_items(pane, identifier):
    return tuple(pane.overlay_families.get('phase:' + identifier, ()))


def _same(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _two_cifs(root):
    return _cif(root, 'phaseA', 4.1), _cif(root, 'phaseB', 4.25)


def _ids(source):
    return [entry.id for entry, _visible in source.listed_entries()]


def test_rw1_tools_buttons_open_the_view_owned_window_and_plot_options_lose_phase_rows(qapp, tmp_path):
    with run_page(qapp, tmp_path / 'int') as (page, view):
        shell = page.findChild(ScatteringWorkspaceShell)
        button = shell.tools.page_tool('reference_phases')
        assert 'Reference Phases' in button.text() and button.toolTip()
        button.click()
        qapp.processEvents()
        window = view.reference_phases
        assert window.isVisible() and window.windowTitle() == 'Reference Phases' and window.toolTip()
        assert window.parent() is view
        assert window.close_button.toolTip()
        dialog = view.plot_options_dialog
        assert not [label for label in dialog.findChildren(QtWidgets.QLabel) if label.text() == 'Calibrant']
        assert not [group for group in dialog.findChildren(QtWidgets.QGroupBox) if group.title() == 'Reference phases']
        assert not hasattr(dialog, 'phase_source') and not hasattr(dialog, 'phase_floor')
        window.close_button.click()
        qapp.processEvents()
        assert not window.isVisible()
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        button = page.tools.page_tool('reference_phases')
        assert 'Reference Phases' in button.text() and button.toolTip()
        button.click()
        qapp.processEvents()
        assert view.reference_phases.isVisible()
    from tests.xdart.pages.test_rsm_page import _close, _page
    rsm = _page(qapp)
    try:
        with pytest.raises(KeyError):
            rsm.tools.page_tool('reference_phases')
    finally:
        _close(rsm, qapp)


def test_rw2_rows_switch_and_coherent_selection_floor_updates(qapp, tmp_path):
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        source = _window(qapp, view).phase_source
        commands = _commands(view)
        a_path, b_path = _two_cifs(tmp_path)
        source.add_cif_paths((str(a_path), str(b_path)))
        source.add_calibrant('LaB6')
        lattice = source.add_lattice('cell', a=4.1, space_group=221)
        _settle(qapp, view)
        options = _options(page)
        a_id, b_id = _ids(source)[:2]
        assert [identifier for identifier, _visible in options.phase_references] == [a_id, b_id, LAB6, lattice.id]
        assert options.phase_global_floor is True and options.phase_min_intensity == 5. and not options.phase_floors
        assert source.global_switch.editor.isChecked() and source.global_row.editor.isEnabled()
        for identifier in (a_id, b_id):
            editor = source.cutoff_editor(identifier)
            assert not editor.isEnabled() and 'same minimum' in editor.toolTip().lower()
        for identifier in (LAB6, lattice.id):
            editor = source.cutoff_editor(identifier)
            assert not editor.isEnabled() and 'intensit' in editor.toolTip().lower()
        source.global_switch.editor.click()
        _settle(qapp, view)
        options = _options(page)
        assert options.phase_global_floor is False
        assert dict(options.phase_floors) == {a_id: 5., b_id: 5.}, 'switching off materializes the global value'
        assert source.cutoff_editor(a_id).isEnabled() and not source.cutoff_editor(LAB6).isEnabled()
        _type(qapp, source.cutoff_editor(a_id), '1')
        _settle(qapp, view)
        assert dict(_options(page).phase_floors) == {a_id: 1., b_id: 5.}
        source.global_switch.editor.click()
        _settle(qapp, view)
        assert _options(page).phase_global_floor is True and dict(_options(page).phase_floors) == {a_id: 1., b_id: 5.}
        source.global_switch.editor.click()
        _settle(qapp, view)
        assert source.cutoff_editor(a_id).value() == 1.
        _check(qapp, source, a_id, False)
        _settle(qapp, view)
        options = _options(page)
        assert (a_id, False) in options.phase_references and dict(options.phase_floors)[a_id] == 1., 'hide keeps it'
        commands.clear()
        source.browser.scans.setCurrentItem(_row(source, a_id))
        source.remove_btn.click()
        _settle(qapp, view)
        paths = [command.path for command in commands if command.kind is ShellCommandKind.SET_PLOT_OPTION]
        assert paths.index(('other', 'phase_floors')) < paths.index(('other', 'phase_references')), paths
        options = _options(page)
        assert a_id not in dict(options.phase_references) and a_id not in dict(options.phase_floors)
        source.add_cif_paths((str(a_path),))
        _settle(qapp, view)
        assert dict(_options(page).phase_floors)[a_id] == 5., 're-adding starts from the global value'


@pytest.mark.parametrize('mode', ('Int 2D', 'Int 1D'))
def test_rw3_one_phase_edit_leaves_every_other_phase_item_identical(qapp, tmp_path, mode):
    with run_page(qapp, tmp_path / 'int', energies=(16, 16), mode=mode) as (page, view):
        if mode == 'Int 1D':
            view.raw_popup_button.click()
            qapp.processEvents()
            select(qapp, page, page._context_controller.navigation.current)
        source = _window(qapp, view).phase_source
        a_path, b_path = _two_cifs(tmp_path)
        source.add_cif_paths((str(a_path), str(b_path)))
        a_id, b_id = _ids(source)
        _settle(qapp, view, lambda: len(view.phase_families) == 2)
        panes = (view.raw, view.cake) if mode == 'Int 2D' else (view.raw_popup_image,)

        def items(identifier):
            return (_marker(view, identifier),) + tuple(_pane_items(pane, identifier) for pane in panes)

        def same(left, right):
            return left[0] is right[0] and all(_same(a, b) for a, b in zip(left[1:], right[1:]))
        before = items(b_id)
        colour = before[0].lines.opts['pen'].color().getRgb()
        assert before[0] is not None and all(before[1:]), 'B draws on every pane'
        _check(qapp, source, a_id, False)
        _settle(qapp, view)
        assert _marker(view, a_id) is None and not any(_pane_items(pane, a_id) for pane in panes)
        assert not _legend(view, 'phaseA').item.isVisible(), 'a hidden phase keeps a greyed legend entry'
        assert same(items(b_id), before) and _marker(view, b_id).lines.opts['pen'].color().getRgb() == colour
        _check(qapp, source, a_id, True)
        _settle(qapp, view, lambda: _marker(view, a_id) is not None)
        assert same(items(b_id), before)
        source.global_switch.editor.click()
        _settle(qapp, view)
        _type(qapp, source.cutoff_editor(a_id), '1')
        _settle(qapp, view)
        assert same(items(b_id), before)
        a_items = items(a_id)
        QtWidgets.QApplication.sendEvent(page, QtCore.QEvent(QtCore.QEvent.Type.WindowActivate))
        qapp.processEvents()
        page._refresh_shell()
        qapp.processEvents()
        assert same(items(b_id), before) and same(items(a_id), a_items), 'an unchanged activation reuses items'
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, '2theta'))
        _settle(qapp, view)
        assert _marker(view, b_id) is not None and _marker(view, b_id) is not before[0], 'a real context change redraws'


def test_rw4_a_legend_click_excludes_and_restores_the_phase(qapp, tmp_path):
    with run_page(qapp, tmp_path / 'int', energies=(16, 16), mode='Int 1D') as (page, view):
        source = _window(qapp, view).phase_source
        a_path, b_path = _two_cifs(tmp_path)
        source.add_cif_paths((str(a_path), str(b_path)))
        a_id, b_id = _ids(source)
        _settle(qapp, view, lambda: _marker(view, a_id) is not None)
        b_item = _marker(view, b_id)

        def click(sample):
            point = view.curve.mapFromScene(sample.sceneBoundingRect().center())
            QtTest.QTest.mouseClick(view.curve.viewport(), QtCore.Qt.MouseButton.LeftButton,
                                    QtCore.Qt.KeyboardModifier.NoModifier, point)
            qapp.processEvents()
        click(_legend(view, 'phaseA'))
        _settle(qapp, view)
        assert _marker(view, a_id) is None and not _legend(view, 'phaseA').item.isVisible()
        assert (a_id, False) in _options(page).phase_references
        assert _row(source, a_id).checkState() == QtCore.Qt.CheckState.Unchecked
        assert _marker(view, b_id) is b_item
        click(_legend(view, 'phaseA'))
        _settle(qapp, view, lambda: _marker(view, a_id) is not None)
        assert (a_id, True) in _options(page).phase_references and _legend(view, 'phaseA').item.isVisible()
        assert _row(source, a_id).checkState() == QtCore.Qt.CheckState.Checked and _marker(view, b_id) is b_item


def test_rw5_calibrants_draw_on_the_phase_path_with_p3a_semantics(qapp, tmp_path):
    import xrd_tools.geometry_figures as geometry
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        window = _window(qapp, view)
        source = window.phase_source
        source.add_calibrant('LaB6')
        cif_path, _other = _two_cifs(tmp_path)
        source.add_cif_paths((str(cif_path),))
        cif_id = _ids(source)[1]
        _settle(qapp, view, lambda: LAB6 in view.phase_families and view.phase_families[LAB6].rings)
        heavy = page._last_scientific_projection.heavy
        expected = geometry.predict_detector_rings(dict(heavy.ring_geometry), 'LaB6', shape=heavy.detector_shape)
        rings = view.phase_families[LAB6].rings
        assert len(rings) == len(expected.curves)
        for actual, reference in zip(rings, expected.curves):
            np.testing.assert_allclose(actual.xy, reference.xy, rtol=0., atol=1e-9)
        raw = _pane_items(view.raw, LAB6)
        assert raw and all(CAVEAT in item.toolTip() for item in raw)
        assert _marker(view, LAB6) is not None and _pane_items(view.cake, LAB6)
        assert not window.rings_note.text(), 'rings are available on a Run frame'
        # Retained display with the same frame keeps every family and still applies edits (R10, CR2).
        state, navigation = page._last_scientific_projection, _navigation(page)
        retained = replace(state, heavy=None, retain_display=True)
        kept = _pane_items(view.raw, LAB6)
        view.reconcile(retained, navigation, completed=0, total=0, detail='')
        _settle(qapp, view)
        assert _same(_pane_items(view.raw, LAB6), kept) and _pane_items(view.raw, cif_id)
        hidden = replace(retained.plot_options, phase_references=((LAB6, True), (cif_id, False)))
        view.reconcile(replace(retained, plot_options=hidden), navigation, completed=0, total=0, detail='')
        _settle(qapp, view)
        assert not _pane_items(view.raw, cif_id) and _same(_pane_items(view.raw, LAB6), kept)
        recut = replace(retained.plot_options, phase_global_floor=False, phase_floors=((cif_id, 1.),))
        view.reconcile(replace(retained, plot_options=recut), navigation, completed=0, total=0, detail='')
        _settle(qapp, view, lambda: bool(_pane_items(view.raw, cif_id)))
        assert _pane_items(view.raw, cif_id) and _same(_pane_items(view.raw, LAB6), kept)
        other = next(key for key in page._context_controller.frame_keys if key is not navigation.current)
        moved = replace(navigation, current=other)
        view.reconcile(retained, moved, completed=0, total=0, detail='')
        _settle(qapp, view)
        assert not _pane_items(view.raw, LAB6), 'a different frame retires the retained rings'
    from tests.xdart.scattering.test_poni_v3_start import _V3_PONI
    with run_page(qapp, tmp_path / 'parallax', energies=(16, 16),
                  poni_text=_V3_PONI.replace('Parallax: False', 'Parallax: True')) as (page, view):
        window = _window(qapp, view)
        window.phase_source.add_calibrant('LaB6')
        _settle(qapp, view, lambda: LAB6 in view.phase_families)
        assert 'parallax' in window.rings_note.text().lower() and window.rings_note.toolTip()
        assert not _pane_items(view.raw, LAB6) and _marker(view, LAB6) is not None, '1D lines stay valid'
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    for mode in ('Int 1D', 'Int 2D', 'Average', 'Reintegrate', '2D Viewer'):
        shell = ScatteringWorkspaceShell()
        state = make_shell_projection()
        options = replace(state.scientific.plot_options, phase_references=((LAB6, True),))
        state = replace(state, scientific=replace(state.scientific, processing_mode=mode, plot_options=options))
        try:
            shell.show()
            shell.apply_state(state)
            qapp.processEvents()
            view = shell.scientific
            window = _window(qapp, view)
            _settle(qapp, view)
            assert not _pane_items(view.raw, LAB6)
            if mode == '2D Viewer':
                assert not view.phase_families and not window.phase_source.isEnabled()
                assert window.phase_source.toolTip()
            else:
                # BROWSE-REFERENCE-RINGS-20261013 (superseded text): saved Int outputs now draw rings too, so the
                # generic reason states the rule rather than "only on a Run's frames".
                from xrd_tools.geometry_figures import RUN_RING_REASON
                assert window.rings_note.text() == RUN_RING_REASON, window.rings_note.text()
                assert 'opened from a file' not in RUN_RING_REASON and "only on a Run's frames" not in RUN_RING_REASON
        finally:
            shell.close()
            shell.deleteLater()
            qapp.processEvents()


def test_rw6_int_and_stitch_persist_selection_switch_and_floors(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.scattering_workspace import build_scattering_workspace
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.pages.values import PageCleanup
    from tests.xdart.pages.test_stitch_page import _Status
    from xdart.utils.session import load_session, save_session
    from xrd_tools.analysis.phase_references import PhaseLibrary
    monkeypatch.setenv('XDART_SESSION_FILE', str(tmp_path / 'restart-session.json'))
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    services = empty_host_services(_Status())
    library = PhaseLibrary()
    a, b = (library.add_cif(path) for path in _two_cifs(tmp_path))
    saved = {'references': [[a.id, True], [b.id, False], [LAB6, True]], 'min_intensity': 7.,
             'global_floor': False, 'floors': [[a.id, 1.5], [b.id, 3.], ['f' * 64, 9.]]}
    save_session({'scattering_phase_references': saved})
    handle = build_scattering_workspace(services, None)
    try:
        options = handle.widget._preferences.plot_options
        assert options.phase_references == ((a.id, True), (b.id, False), (LAB6, True))
        assert options.phase_min_intensity == 7. and options.phase_global_floor is False
        assert options.phase_floors == ((a.id, 1.5), (b.id, 3.)), 'floors naming an unlisted phase are pruned'
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        handle.widget.deleteLater()
        qapp.processEvents()
    written = load_session()['scattering_phase_references']
    assert written['global_floor'] is False and written['floors'] == [[a.id, 1.5], [b.id, 3.]]
    assert written['references'] == [[a.id, True], [b.id, False], [LAB6, True]] and written['min_intensity'] == 7.
    save_session({'scattering_phase_references': {'references': [[a.id, True]], 'min_intensity': 6.}})
    handle = build_scattering_workspace(services, None)
    try:
        options = handle.widget._preferences.plot_options
        assert options.phase_references == ((a.id, True),) and options.phase_min_intensity == 6.
        assert options.phase_global_floor is True and options.phase_floors == ()
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        handle.widget.deleteLater()
        qapp.processEvents()
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        source = _window(qapp, view).phase_source
        source.set_selection(((a.id, True), (LAB6, True)))
        _settle(qapp, view)
        source.global_switch.editor.click()
        _settle(qapp, view)
        _type(qapp, source.cutoff_editor(a.id), '2')
        _settle(qapp, view)
        document = page.settings()
        section = document['phase_references']
        assert section['references'] == [[a.id, True], [LAB6, True]]
        assert section['global_floor'] is False and section['floors'] == [[a.id, 2.]]
        old = json.loads(json.dumps(document))
        old['phase_references'] = {'references': [[a.id, True]], 'min_intensity': 4.}
        page.apply_settings(old)
        _settle(qapp, view)
        options = _options(page)
        assert options.phase_references == ((a.id, True),) and options.phase_min_intensity == 4.
        assert options.phase_global_floor is True and options.phase_floors == ()
        page.apply_settings(document)
        _settle(qapp, view)
        assert _options(page).phase_floors == ((a.id, 2.),) and _options(page).phase_global_floor is False


_HEADLESS_WINDOW = textwrap.dedent("""
    import json, sys, time
    sys.modules["pymatgen"] = None
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from dataclasses import replace
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    shell = ScatteringWorkspaceShell(); state = make_shell_projection()
    shell.show(); shell.apply_state(state); app.processEvents()
    view = shell.scientific
    view.show_reference_phases(); app.processEvents()
    source = view.reference_phases.phase_source
    report = {"cif": [source.add_cif_btn.isEnabled(), source.add_cif_btn.toolTip()],
              "lattice": [source.add_lattice_btn.isEnabled(), source.add_lattice_btn.toolTip()],
              "calibrant": [source.calibrant_row.editor.isEnabled(), source.calibrant_row.editor.toolTip()]}
    options = replace(state.scientific.plot_options, phase_references=(("calibrant:LaB6", True),))
    shell.apply_state(replace(state, scientific=replace(state.scientific, plot_options=options)))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        app.processEvents()
        lines = [item for item in view.curve.getPlotItem().items if item.data(0) == "phase:calibrant:LaB6"]
        if lines and not view.phase_worker.busy:
            break
        time.sleep(.01)
    report["lines"] = len(lines[0].reference_positions) if lines else 0
    report["phase_imported"] = "xrd_tools.analysis.phase" in sys.modules and bool(
        getattr(sys.modules["xrd_tools.analysis.phase"], "HAS_PYMATGEN", False))
    print(json.dumps(report), flush=True)
    shell.close()
    import os as _os; _os._exit(0)
""")


def test_rw7_without_pymatgen_calibrants_still_draw_and_cif_actions_say_why(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, XDART_PHASE_LIBRARY=str(tmp_path / 'phases'),
               XDART_SESSION_FILE=str(tmp_path / 'session.json'), XDART_SETTINGS_FILE=str(tmp_path / 'settings.ini'),
               QT_QPA_PLATFORM='offscreen')
    done = subprocess.run([sys.executable, '-c', _HEADLESS_WINDOW], capture_output=True, text=True, env=env,
                          cwd=str(root), timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report['cif'][0] is False and 'pymatgen' in report['cif'][1].lower(), report
    assert report['lattice'][0] is False and 'pymatgen' in report['lattice'][1].lower(), report
    assert report['calibrant'][0] is True and report['calibrant'][1], report
    assert report['lines'] > 0, report
    assert report['phase_imported'] is False, report


def test_rw8_phase_fitter_rows_switch_real_fit_and_stitch_batch(qapp, tmp_path):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan, run_displayed_phase_fit
    from tests.core.test_reference_phase_sources import (CSCL_WEAK_Q, NACL_WEAK_Q, WAVELENGTH, _bump,
                                                         _fit_inputs)
    from tests.xdart.pages.test_stitch_fit_host import _finish
    paths, trace = _fit_inputs(tmp_path)
    dialog = PhaseFitDialog()
    try:
        source = dialog.phase_source
        advanced = [row.label.text() for row in dialog.advanced.findChildren(QtWidgets.QWidget)
                    if row.objectName() == 'controlsFormRow']
        assert 'Min peak intensity %' not in advanced, 'the cutoff moved from Advanced to the Phases card'
        source.add_cif_paths(tuple(str(path) for path in paths))
        nacl, cscl = _ids(source)
        baseline = dialog.vnext_fit_values()
        assert 'phase_min_intensities' not in baseline and baseline['min_intensity'] == 5.
        assert source.global_switch.editor.isChecked() and not source.cutoff_editor(nacl).isEnabled()
        dialog.show()
        source.global_switch.editor.click()
        qapp.processEvents()
        _type(qapp, source.cutoff_editor(nacl), '1')
        values = dialog.vnext_fit_values()
        assert values['phase_min_intensities'] == (1., 5.)
        result = run_displayed_phase_fit(DisplayedPhaseFitPlan(trace, wavelength_angstrom=WAVELENGTH, **values))
        assert result.code == 'OK', (result.code, result.diagnostics)
        axis = np.asarray(trace.axis)
        first, second = result.phase_components
        assert _bump(axis, first, NACL_WEAK_Q) > .01 * float(first.max())
        assert all(_bump(axis, second, weak) < .005 * float(second.max()) for weak in CSCL_WEAK_Q)
        source.global_switch.editor.click()
        qapp.processEvents()
        assert dialog.vnext_fit_values() == baseline, 'switch on: the settings are exactly as before'
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        page.tools.page_tool('phase_fit').click()
        qapp.processEvents()
        host = page._fit_hosts['phase']
        _finish(host, qapp)  # the host's opening reload, as test_stitch_fit_host waits (Batch while busy cancels)
        source = host.dialog.phase_source
        source.add_cif_paths(tuple(str(path) for path in paths))
        nacl, _cscl = _ids(source)
        host.dialog.show()
        source.global_switch.editor.click()
        qapp.processEvents()
        _type(qapp, source.cutoff_editor(nacl), '1')
        assert host.dialog.vnext_fit_values()['phase_min_intensities'] == (1., 5.)
        targets = page._fit_batch('phase')
        host.dialog.batch_btn.click()
        _finish(host, qapp)
        assert len(host.outcomes) == len(targets) >= 1
        for outcome in host.outcomes:
            assert 'unexpected keyword' not in outcome.message and 'INVALID_PHASE_PLAN' not in outcome.message, (
                outcome.message)
