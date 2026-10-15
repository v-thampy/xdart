"""Frozen P3b phase GUI oracle; visible parent failures precede every new API.

These tests mount the real Int/Phase Fitter/Stitch owners. Instrument/publication,
phase listing and worker seams are never replaced with successful fakes.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import run_page, select, choose
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind

#: REFERENCE-PHASES-20261011 moved the phase controls from Plot Options to the Tools-panel window and made
#: calibrants phase entries; these successor rows keep every scientific assertion (declared in its kickoff).
LAB6 = 'calibrant:LaB6'


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phase-library'))


def _phase_section(view):
    """The Reference Phases window (successor of Plot Options' Reference phases section)."""
    view.show_reference_phases()
    window = view.reference_phases
    assert window.isVisible() and window.toolTip()
    return window


def _cif(root, name='CsCl', a=4.1):
    """A real structural CIF; unequal atoms give nontrivial relative intensities."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    path = root / f'{name}.cif'
    structure = Structure(Lattice.cubic(a), ['Cs', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    path.write_text(str(CifWriter(structure)), encoding='utf-8')
    return path


def _entries(root, count=1):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    return tuple(library.add_cif(_cif(root, f'phase{index}', 4.1 + .15*index))
                 for index in range(count))


def _settle(qapp, view, predicate=lambda: True):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not view.phase_worker.busy and predicate():
            qapp.processEvents()
            return
        time.sleep(.005)
    raise AssertionError('the shared phase worker did not settle')


def _edit(qapp, view, entries, *, floor=5.):
    """Select ``entries`` and the global minimum in the window; each edit applies at once (no Okay)."""
    source = _phase_section(view).phase_source
    qapp.processEvents()
    source.set_selection(tuple((entry.id, True) for entry in entries))
    qapp.processEvents()
    if source.global_row.current_value() != float(floor):
        source.global_row.editor.setValue(float(floor))
    qapp.processEvents()
    _settle(qapp, view)


def _phase_items(pane, entry):
    return pane.overlay_families.get('phase:' + entry.id, ())


def _markers(view, entry):
    """Positions drawn by the phase's one joined 1D item (maintainer, October 11: one object per phase)."""
    family = 'phase:' + entry.id
    items = [item for item in view.curve.getPlotItem().items if item.data(0) == family]
    assert len(items) <= 1, 'a phase draws one joined 1D reference item, not one item per reflection'
    return tuple(items[0].reference_positions) if items else ()


def _navigation(page):
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    controller = page._context_controller
    return FrameNavigationProjection(controller.frame_keys, controller.navigation.current,
                                     controller.navigation.selected)


def _command(page, kind, value, path=()):
    page._handle_shell_command(ShellCommand(kind, value, path))


@contextmanager
def _stitch(qapp, root, *, two_dimensional=False):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.pages.test_stitch_page import _Status, _published, _wait
    from tests.xdart.pages.test_stitch_fit_host import _dispose
    from tests.xdart.tools.test_stitch_native_values import _native_form
    root.mkdir(parents=True, exist_ok=True)
    form = _native_form(root, 'multigeometry', scans='1')
    if two_dimensional:
        form = replace(form, mode='2d', npt_rad_2d=48, npt_azim_2d=32)
    path, payload = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form); page.open_result(path); _wait(page, qapp)
        yield page, page.scientific, payload
    finally:
        _dispose(page, qapp)


def test_phase_fitter_adds_cif_and_lattice_phases_through_the_shared_source(qapp, tmp_path):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    from xdart.gui.tabs.scattering.browser_view import BrowserView
    dialog = PhaseFitDialog()
    comparison = None
    try:
        # Parent failure is the visible missing action, not a missing owner import.
        buttons = [button for button in dialog.findChildren(QtWidgets.QPushButton)
                   if button.text().replace('…', '').strip() == 'Add lattice']
        assert len(buttons) == 1, 'the Phase Fitter must offer Add lattice'
        from xdart.gui.analysis.phase_source import PhaseSourceWidget
        from xrd_tools.analysis.phase import PhaseModel
        assert isinstance(dialog.phase_source, PhaseSourceWidget)
        assert isinstance(dialog.phase_source.browser, BrowserView)
        assert buttons[0].toolTip()
        path = _cif(tmp_path)
        dialog.phase_source.add_cif_paths((str(path),))
        assert len(dialog._phases) == 1
        entry_id = dialog.phase_source.selection[0][0]
        entry = dialog.phase_source.library.resolve(entry_id).entry
        assert entry.cif_bytes == path.read_bytes()
        # Same real fit path, once from the mounted library and once from the
        # exact CIF as the tool accepted before promotion; no optimizer fake.
        model = PhaseModel.from_cif(path)
        q = np.linspace(1., 4., 301)
        y = np.full(q.shape, 10.)
        for peak in model.peaks:
            if 1. < peak.q < 4.:
                y += peak.intensity * np.exp(-.5*((q-peak.q)/.025)**2)
        dialog._show_pattern(q, y, 'q', clear_fit=True)
        comparison = PhaseFitDialog()
        comparison._phases = [(str(path), model)]
        comparison._show_pattern(q, y, 'q', clear_fit=True)
        request = dialog.build_fit_request()
        reference = comparison.build_fit_request()
        assert request is not None and reference is not None
        actual = request[1].analyze(request[0])
        expected = reference[1].analyze(reference[0])
        assert actual.ok and expected.ok
        np.testing.assert_allclose(actual.overlay.traces['fit'], expected.overlay.traces['fit'], rtol=1e-10, atol=1e-10)
        dialog.phase_source.add_lattice('Cubic positions', a=4.1, space_group=221)
        assert len(dialog.phase_source.selection) == 2
        assert len(dialog._phases) == 2
        assert dialog.phase_source.library.resolve(dialog.phase_source.selection[-1][0]).entry.source_kind == 'lattice'
        for child in dialog.phase_source.findChildren(QtWidgets.QWidget):
            if isinstance(child, (QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QComboBox)):
                assert child.toolTip(), child.objectName()
    finally:
        if comparison is not None:
            comparison.close(); comparison.deleteLater()
        dialog.close(); dialog.deleteLater(); qapp.processEvents()


def test_plot_options_phase_selection_is_shared_by_int_and_stitch(qapp, tmp_path, monkeypatch):
    with run_page(qapp, tmp_path/'int') as (page, view):
        _phase_section(view)
        from xdart.gui.widgets.controls_panel import FormRow
        from xdart.gui.analysis.phase_source import PhaseSourceWidget
        from xdart.gui.tabs.scattering.scientific_plot_options import plot_option_update
        from xrd_tools.analysis.phase import XRDCalculator
        entries = _entries(tmp_path)
        dialog = view.reference_phases
        assert isinstance(dialog.phase_source, PhaseSourceWidget)
        assert isinstance(dialog.phase_source.global_row, FormRow)
        assert dialog.phase_source.global_row.current_value() == 5.
        assert not dialog.phase_source.global_row.editor.isEnabled()
        assert dialog.phase_source.global_row.editor.toolTip()
        calls = []; actual = XRDCalculator.get_pattern
        def counted(self, *args, **kwargs):
            calls.append(threading.get_ident()); return actual(self, *args, **kwargs)
        monkeypatch.setattr(XRDCalculator, 'get_pattern', counted)
        _edit(qapp, view, entries)
        assert calls and all(thread != threading.get_ident() for thread in calls)
        selection = tuple((entry.id, True) for entry in entries)
        assert page._preferences.plot_options.phase_references == selection
        assert dialog.phase_source.global_row.editor.isEnabled()
        prior = len(calls)
        _edit(qapp, view, entries, floor=20.)
        assert len(calls) == prior, 'editing the floor must reuse listed reflections'
        assert page._preferences.plot_options.phase_min_intensity == 20.
        dialog.close_button.click(); qapp.processEvents()  # live apply: closing keeps what was applied
        assert dialog.phase_source.selection == selection
        assert dialog.phase_source.global_row.current_value() == 20.
        assert page._preferences.plot_options.phase_min_intensity == 20.
        options = page._preferences.plot_options
        assert plot_option_update(options, ('other', 'phase_min_intensity'), -1.) is None
        assert plot_option_update(options, ('other', 'phase_min_intensity'), 101.) is None
        from xrd_tools.analysis.phase_references import PhaseLibrary
        lattice = PhaseLibrary().add_lattice('positions', a=4.1, space_group=221)
        _edit(qapp, view, (lattice,))
        assert not dialog.phase_source.global_row.editor.isEnabled()
        assert 'lattice' in dialog.phase_source.global_row.editor.toolTip().lower()
    with _stitch(qapp, tmp_path/'stitch') as (page, view, _payload):
        _phase_section(view)
        _edit(qapp, view, entries, floor=20.)
        assert view.reference_phases.projection.phase_references == selection
        assert view.reference_phases.projection.phase_min_intensity == 20.
        assert view.phase_families
    import xrd_tools.analysis.phase as phase_owner
    with run_page(qapp, tmp_path/'dependency', energies=(16, 16)) as (page, view):
        _phase_section(view)
        monkeypatch.setattr(phase_owner, 'HAS_PYMATGEN', False)
        page._refresh_shell(); qapp.processEvents()
        source = view.reference_phases.phase_source
        # Calibrants need no pymatgen (as P3a's rings never did); CIF and lattice sources say why they are off.
        assert not source.add_cif_btn.isEnabled() and 'pymatgen' in source.add_cif_btn.toolTip().lower()
        assert not source.add_lattice_btn.isEnabled() and 'pymatgen' in source.add_lattice_btn.toolTip().lower()
        assert source.calibrant_row.editor.isEnabled()


def test_phase_lines_on_q_and_2theta_axes_in_int_and_stitch(qapp, tmp_path):
    for mode in ('Int 1D', 'Int 2D'):
        with run_page(qapp, tmp_path/mode, energies=(16, 16), mode=mode) as (page, view):
            _phase_section(view)
            entry, = _entries(tmp_path)
            _edit(qapp, view, (entry,))
            q_positions = tuple(sorted(_markers(view, entry)))
            assert q_positions
            original = page._last_scientific_projection
            # Projection-owned consumers: browse and operation results retain
            # line references without inventing detector geometry.
            for shown in ('Int 1D', 'Int 2D', 'Average', 'Reintegrate'):
                state = replace(original, processing_mode=shown)
                view.reconcile(state, _navigation(page), completed=1, total=1, detail='')
                _settle(qapp, view)
                assert _markers(view, entry)
            _command(page, ShellCommandKind.SET_PLOT_AXIS, '2theta')
            _settle(qapp, view)
            payload = page._context_controller.project(page._context_controller.navigation.current)
            wavelength = payload.view.geometry.poni['wavelength'] * 1e10
            expected = 2. * np.rad2deg(np.arcsin(np.asarray(q_positions)*wavelength/(4.*np.pi)))
            np.testing.assert_allclose(sorted(_markers(view, entry)), expected, atol=1e-10)
            if mode == 'Int 2D':
                assert _phase_items(view.cake, entry)
            state = replace(page._last_scientific_projection, measurement_mode='Grazing')
            view.reconcile(state, _navigation(page), completed=1, total=1, detail='')
            _settle(qapp, view)
            assert not _markers(view, entry) and not _phase_items(view.cake, entry)
    for mapped in (False, True):
        with _stitch(qapp, tmp_path/f'stitch-{mapped}', two_dimensional=mapped) as (page, view, payload):
            _phase_section(view); _edit(qapp, view, (entry,))
            assert _markers(view, entry)
            if mapped:
                assert _phase_items(view.cake, entry)
            page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, '2theta'))
            _settle(qapp, view)
            assert _markers(view, entry)
    with run_page(qapp, tmp_path/'scan', energies=(16, 17)) as (page, view):
        _phase_section(view); _edit(qapp, view, (entry,))
        _command(page, ShellCommandKind.SET_PLOT_MODE, 'Overlay')
        controller = page._context_controller
        keys = tuple(controller.frame_keys)
        for key in keys:
            select(qapp, page, key)
        assert controller.select_navigation(keys[-1], keys)
        page._refresh_shell(); qapp.processEvents()
        _command(page, ShellCommandKind.SET_PLOT_AXIS, '2theta'); _settle(qapp, view)
        assert not _markers(view, entry)
        assert 'ener' in view.phase_families[entry.id].reason.lower()


def test_phase_rings_on_ints_detector_image_follow_p3a(qapp, tmp_path, monkeypatch):
    import xrd_tools.geometry_figures as figures
    for mode in ('Int 1D', 'Int 2D'):
        with run_page(qapp, tmp_path/mode, energies=(16, 16), mode=mode) as (page, view):
            _phase_section(view)
            from xdart.gui.pages.operation_owner import SingleWorkerOwner
            entry, = _entries(tmp_path)
            _edit(qapp, view, (entry,))
            if mode == 'Int 1D':
                view.raw_popup_button.click(); qapp.processEvents()
                # Opening Int's popup requests full raw asynchronously. Compare
                # family identity only after that real image transition settles.
                _settle(qapp, view, lambda: view.raw_popup_image.image.image is not None
                        and view.raw_popup_image.image.image.shape == (487, 195))
            pane = view.raw_popup_image if mode == 'Int 1D' else view.raw
            assert isinstance(view.phase_worker, SingleWorkerOwner)
            items = _phase_items(pane, entry)
            assert len(items) == 1, 'one NaN-separated item per phase family'
            curves = view.phase_families[entry.id].rings
            # Independent reflection positions, then P3a's existing cone/arcs
            # owner. The newly added phase-ring provider is not its own oracle.
            from pymatgen.core import Structure
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            from xrd_tools.core.containers import PONI
            from xrd_tools.core.geometry.diffractometer import DetectorCalibration
            from xrd_tools.integrate.calibration import detector_calibration_to_integrator
            snapshot = page._context_controller.project(
                page._context_controller.navigation.current).view.geometry.poni
            ai = detector_calibration_to_integrator(DetectorCalibration(
                PONI.from_dict(snapshot), json.loads(snapshot['detector_config_json']),
                parallax=snapshot.get('parallax')))
            structure = Structure.from_str(entry.cif_bytes.decode('utf-8'), fmt='cif')
            # Independent upstream angle map bounds the reachable listing.
            # Normalize only within this range, as the maintainer requires.
            angles = np.rad2deg(ai.center_array(ai.detector.shape, unit='2th_rad', scale=False))
            bounds = (float(angles.min()), float(angles.max()))
            pattern = XRDCalculator(wavelength=ai.wavelength*1e10).get_pattern(
                structure, two_theta_range=bounds)
            expected_arcs = []
            for theta, intensity in zip(pattern.x, pattern.y):
                if intensity >= 5.:
                    expected_arcs.extend(figures.display_arcs(figures.cone_plane(ai),
                                                            np.deg2rad(theta), ai.detector.shape, 5.))
            assert len(curves) == len(expected_arcs)
            for curve, expected in zip(curves, expected_arcs):
                np.testing.assert_allclose(curve.xy, expected, rtol=0., atol=1e-9)
            actual_xy = np.column_stack(items[0].getData())
            finite = actual_xy[np.isfinite(actual_xy).all(axis=1)]
            assert len(finite) == sum(len(curve.xy) for curve in curves)
            # The calibrant is its own independent phase family (P3a's rings, on the phase path).
            choose(qapp, view, 'LaB6'); _settle(qapp, view)
            calibrant = pane.overlay_families.get('phase:' + LAB6, ())
            assert calibrant and all(np.isfinite(np.column_stack(item.getData())).any() for item in calibrant)
            assert _phase_items(pane, entry) == items
            # Observe the real cone owner, never substitute a successful result.
            calls = []; actual = figures.display_arcs
            def recorded(*args, **kwargs):
                calls.append(threading.get_ident()); return actual(*args, **kwargs)
            monkeypatch.setattr(figures, 'display_arcs', recorded)
            _edit(qapp, view, ())
            _edit(qapp, view, (entry,))
            assert not calls or all(thread != threading.get_ident() for thread in calls)
            controller = page._context_controller
            controller.select_browse(); page._refresh_shell(); _settle(qapp, view)
            # BROWSE-REFERENCE-RINGS-20261013 (superseded clause): terminal Browse keeps the phase's rings,
            # computed from the saved output's own record.
            select(qapp, page, controller.frame_keys[0])
            _settle(qapp, view, lambda: _phase_items(pane, entry))
            assert _phase_items(pane, entry)
    # Latest-only request test: block a real listing, supersede it, then allow
    # it to finish. It must never paint the stale entry into the new selection.
    with run_page(qapp, tmp_path/'stale', energies=(16, 16)) as (page, view):
        _phase_section(view)
        from xrd_tools.analysis.phase_references import build_phase_references
        import xrd_tools.analysis.phase_references as references
        entries = _entries(tmp_path, 2)
        entered, release = threading.Event(), threading.Event()
        calls = []
        def blocked(*args, **kwargs):
            calls.append(threading.get_ident())
            if not entered.is_set():
                entered.set(); assert release.wait(15)
            return build_phase_references(*args, **kwargs)
        monkeypatch.setattr(references, 'build_phase_references', blocked)
        try:
            source = view.reference_phases.phase_source
            source.set_selection(((entries[0].id, True),)); qapp.processEvents()
            assert entered.wait(5)
            source.set_selection(((entries[1].id, True),)); qapp.processEvents()
            assert entries[0].id not in view.phase_families
            release.set(); _settle(qapp, view, lambda: entries[1].id in view.phase_families)
            assert tuple(view.phase_families) == (entries[1].id,)
            assert all(thread != threading.get_ident() for thread in calls)
        finally:
            release.set()


def test_several_phases_have_colours_legend_and_independent_families(qapp, tmp_path):
    with run_page(qapp, tmp_path/'int', energies=(16, 16)) as (page, view):
        _phase_section(view)
        from xdart.gui.widgets.scientific_plot import reference_colors
        entries = _entries(tmp_path, 3)
        _edit(qapp, view, entries)
        choose(qapp, view, 'LaB6'); _settle(qapp, view)
        calibrant = tuple(view.raw.overlay_families.get('phase:' + LAB6, ()))
        assert calibrant
        view.options.click(); qapp.processEvents()
        grid_label = next(label for label in view.plot_options_dialog.findChildren(QtWidgets.QLabel)
                          if label.text() == 'Detector coordinate grid')
        grid_label.parentWidget().findChild(QtWidgets.QComboBox).setCurrentText('q')
        view.plot_options_dialog.accept_button.click(); qapp.processEvents(); _settle(qapp, view)
        grid = tuple(view.raw.overlay_families['grid'])
        assert grid
        # HELP-RESIDUALS-20261011 successor edit (decision D4): phases take the reference palette, not trace colours.
        colors = reference_colors(len(entries))
        for entry, color in zip(entries, colors):
            items = _phase_items(view.raw, entry)
            assert len(items) == 1
            assert items[0].opts['pen'].color().getRgb()[:3] == color[:3]
            assert entry.name in items[0].toolTip() and 'omitted' in items[0].toolTip().lower()
            assert _markers(view, entry)
        legend_names = tuple(label.text for _item, label in view.curve.getPlotItem().legend.items)
        assert all(entry.name in legend_names for entry in entries)
        source = view.reference_phases.phase_source
        source.set_selection(tuple((entry.id, True) for entry in entries[1:]) + ((LAB6, True),))
        _settle(qapp, view)
        assert not _phase_items(view.raw, entries[0])
        assert all(_phase_items(view.raw, entry) for entry in entries[1:])
        assert tuple(view.raw.overlay_families.get('phase:' + LAB6, ())) == calibrant
        assert tuple(view.raw.overlay_families['grid']) == grid
        choose(qapp, view, 'Off'); _settle(qapp, view)
        assert not view.raw.overlay_families.get('phase:' + LAB6)
        assert all(_phase_items(view.raw, entry) for entry in entries[1:])


def test_phase_selection_and_library_persist_across_restart(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.scattering_workspace import build_scattering_workspace
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.pages.values import PageCleanup
    from tests.xdart.pages.test_stitch_page import _Status
    from xdart.utils.session import load_session
    session_path = tmp_path/'restart-session.json'
    monkeypatch.setenv('XDART_SESSION_FILE', str(session_path))
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    services = empty_host_services(_Status())
    handle = build_scattering_workspace(services, None)
    assert handle.close().status is PageCleanup.CLEAN
    handle.widget.deleteLater(); qapp.processEvents()
    # Parent really saves its profile, but omits this additive preference key.
    assert 'scattering_phase_references' in load_session(), 'workspace close must save reference phases'
    entries = _entries(tmp_path)
    handle = build_scattering_workspace(services, None)
    try:
        view = handle.widget._shell.scientific
        _phase_section(view); _edit(qapp, view, entries, floor=13.)
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        handle.widget.deleteLater(); qapp.processEvents()
    saved = load_session()['scattering_phase_references']
    assert saved['references'] == [[entries[0].id, True]] and saved['min_intensity'] == 13.
    restored = build_scattering_workspace(services, None)
    try:
        options = restored.widget._preferences.plot_options
        assert options.phase_references == ((entries[0].id, True),)
        assert options.phase_min_intensity == 13.
        from xrd_tools.analysis.phase_references import PhaseLibrary
        assert PhaseLibrary().resolve(entries[0].id).entry is not None
    finally:
        assert restored.close().status is PageCleanup.CLEAN
        restored.widget.deleteLater(); qapp.processEvents()
    with _stitch(qapp, tmp_path/'stitch') as (page, view, _payload):
        _phase_section(view); _edit(qapp, view, entries, floor=13.)
        settings = page.settings()
        assert settings['phase_references']['references'] == [[entries[0].id, True]]
        assert settings['phase_references']['min_intensity'] == 13.
        _edit(qapp, view, ()); page.apply_settings(settings); _settle(qapp, view)
        assert view.reference_phases.projection.phase_references == ((entries[0].id, True),)
        assert view.reference_phases.projection.phase_min_intensity == 13.
        missing = replace(view.reference_phases.projection, phase_references=(('f'*64, True),))
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, missing.phase_references,
                                       ('other', 'phase_references')))
        _settle(qapp, view)
        assert not view.phase_families
        assert 'missing' in view.reference_phases.phase_source.toolTip().lower()
    assert str(PhaseLibrary().path).startswith(str(tmp_path))
