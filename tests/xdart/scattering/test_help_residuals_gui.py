"""HELP-RESIDUALS-20261011 frozen GUI oracle, phase sources, overlays and rings (Claude, October 11): GC1, GC3-GC8,
GE1-GE3 and GE6.

Real pages, windows, Fitters, workers, listings and fits, driven by real clicks and real mouse moves. The CIF parser is
pymatgen's real one, held by an event only so an add can be observed while it runs. Each [fail] row's first assertion
is behaviour a user sees; new names are reached after it, inside test bodies.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import choice, choose, rings, run_page, select, settle
from tests.xdart.scattering.test_phase_reference_overlays import _cif, _settle, _stitch
from tests.xdart.scattering.test_reference_phases_window import LAB6, _commands, _marker, _options, _pane_items, _window
from tests.xdart.scattering.test_predicted_ring_overlays import close_phase_window
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell

#: the CsCl-type line pymatgen merges: (300) and (221) share h^2 + k^2 + l^2 = 9
MERGED = ('(300)', '(221)')


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phase-library'))


def _until(qapp, predicate, seconds=15., what='the condition'):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError(f'{what} was not reached in {seconds:g} s')


def _spin(qapp, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _held_parser(monkeypatch, seconds=3.):
    """pymatgen's real CIF parser, held at entry until released (or ``seconds``); records whether each parse ran on
    the GUI thread, and each return."""
    import xrd_tools.analysis.phase_references as provider
    real, gate, on_gui, returned = provider._parse_cif, threading.Event(), [], []

    def parse(raw, structure_class=None):
        on_gui.append(threading.current_thread() is threading.main_thread())
        gate.wait(seconds)
        try:
            return real(raw, structure_class)
        finally:
            returned.append(True)
    monkeypatch.setattr(provider, '_parse_cif', parse)
    try:
        yield gate, on_gui, returned
    finally:
        gate.set()


def _chooser(monkeypatch, *paths):
    """The shared file chooser, as PhaseSourceWidget calls it, returning ``paths``."""
    import xdart.gui.analysis.phase_source as owner
    monkeypatch.setattr(owner, 'browse', lambda *_args, **_kwargs: [str(path) for path in paths])


def _busy(source):
    """Add CIF is unavailable, with a busy reason, while the chosen CIFs are read (C1)."""
    return not source.add_cif_btn.isEnabled() and 'reading' in source.add_cif_btn.toolTip().lower()


def _close_view(qapp, view):
    deadline = time.monotonic() + 15
    while view.shutdown_phase_overlays().status is PageCleanup.PENDING and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)


@contextmanager
def _shell(qapp, mode='Int 1D'):
    """Int's real shell and view without a Run: the Reference Phases window, its source and the view's owners."""
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection()
    shell.show()
    shell.apply_state(replace(state, scientific=replace(state.scientific, processing_mode=mode)))
    qapp.processEvents()
    try:
        yield shell.scientific
    finally:
        _close_view(qapp, shell.scientific)
        shell.close()
        shell.deleteLater()
        qapp.processEvents()


def _close_handle(qapp, handle):
    deadline = time.monotonic() + 15
    receipt = handle.close()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
        receipt = handle.close()
    handle.widget.deleteLater()
    qapp.processEvents()
    return receipt


def _phase_commands(seen):
    return [command for command in seen if command.path == ('other', 'phase_references')]


# ---- GC1 (R1, CR1): Add CIF on a worker, with its lifecycle on three paths ---------------------------------------

def test_gc1a_add_cif_reads_on_a_worker_lists_both_and_resolves_only_the_selection(qapp, tmp_path, monkeypatch):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    first, second = _cif(tmp_path, 'phaseA', 4.1), _cif(tmp_path, 'phaseB', 4.25)
    kept = PhaseLibrary().add_cif(_cif(tmp_path, 'kept', 4.4))  # a library entry this window never selects
    _chooser(monkeypatch, first, second)
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    # declared oracle repair (PHASE-ANALYSIS-128): the list is a host's window's, here Int's page without a Run
    with empty_page(qapp) as page:
        source = _open(qapp, page).phase_source
        idle = source.add_cif_btn.toolTip()
        with _held_parser(monkeypatch) as (gate, on_gui, _returned):
            source.add_cif_btn.click()
            assert _busy(source), ('Add CIF answers at once and stays busy while the CIFs are read',
                                   source.add_cif_btn.isEnabled(), source.add_cif_btn.toolTip())
            assert source.selection == ()
            _until(qapp, lambda: on_gui, what='the first parse')
            gate.set()
            _until(qapp, source.add_cif_btn.isEnabled, what='the add to finish')
        assert on_gui == [False, False], 'each CIF is parsed on the worker, never on the GUI thread'
        assert [entry.id for entry, _shown in source.listed_entries()] == [_sha(first), _sha(second)]
        assert source.add_cif_btn.toolTip() == idle and not source.reason
        resolved = []
        real = PhaseLibrary.resolve
        monkeypatch.setattr(PhaseLibrary, 'resolve',
                            lambda self, identity: resolved.append(identity) or real(self, identity))
        source.refresh_library()
        # declared oracle repair (PHASE-ANALYSIS-128): an explicit refresh also re-reads the view's lookups of the
        # same listed ids, by design; the catalogue entry is still listed without being resolved
        assert set(resolved) == {_sha(first), _sha(second)} and kept.id not in resolved, (
            'the chooser lists without resolving')
        editor = source.library_row.editor
        assert [editor.itemData(index) for index in range(1, editor.count())] == [kept.id]
        assert [editor.itemText(index) for index in range(1, editor.count())] == ['kept']


def test_gc1b_a_partial_failure_keeps_the_added_prefix_and_shows_the_first_error(qapp, tmp_path, monkeypatch):
    good = _cif(tmp_path, 'phaseA', 4.1)
    broken = tmp_path / 'broken.cif'
    broken.write_text('data_broken\n_cell_length_a not-a-number\n', encoding='utf-8')
    _chooser(monkeypatch, good, broken, _cif(tmp_path, 'phaseC', 4.4))
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    # declared oracle repair (PHASE-ANALYSIS-128): the list is a host's window's, here Int's page without a Run
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        source = dialog.phase_source
        with _held_parser(monkeypatch) as (gate, on_gui, _returned):
            source.add_cif_btn.click()
            assert _busy(source) and source.selection == (), source.add_cif_btn.toolTip()
            gate.set()
            _until(qapp, source.add_cif_btn.isEnabled, what='the add to finish')
        assert [entry.id for entry, _shown in source.listed_entries()] == [_sha(good)]
        # the window shows the list's message once, on its own status line (gc3c)
        assert dialog.status.isVisible() and 'CIF could not be parsed' in dialog.status.text(), dialog.status.text()
        assert len(on_gui) == 2, 'the add stops at its first error'


def test_gc1c_completion_applies_to_the_current_selection_and_cap(qapp, tmp_path, monkeypatch):
    from xrd_tools.integrate.calibration import CALIBRANT_NAMES
    first, second = _cif(tmp_path, 'phaseA', 4.1), _cif(tmp_path, 'phaseB', 4.25)
    _chooser(monkeypatch, first, second)
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    # declared oracle repair (PHASE-ANALYSIS-128): the list is a host's window's, here Int's page without a Run
    with empty_page(qapp) as page:
        source = _open(qapp, page).phase_source
        with _held_parser(monkeypatch) as (gate, _on_gui, _returned):
            source.add_cif_btn.click()
            assert _busy(source) and source.selection == (), source.add_cif_btn.toolTip()
            # While the CIFs are read, the list grows to seven phases: six calibrants and a lattice.
            for name in CALIBRANT_NAMES:
                source.add_calibrant(name)
            assert source.add_lattice('cell', a=4.0, space_group=221) is not None
            during = source.selection
            assert len(during) == 7
            gate.set()
            _until(qapp, source.add_cif_btn.isEnabled, what='the add to finish')
        assert source.selection == (*during, (_sha(first), True)), 'the add joins the current list, not a captured one'
        assert 'at most eight' in source.status.text().lower(), source.status.text()


def test_gc1d_closing_the_window_or_its_view_drops_a_held_add(qapp, tmp_path, monkeypatch):
    """PHASE-ANALYSIS-128 (declared oracle repair): the window is Int's, deleted on close with its add retired to the
    view, whose shutdown waits for that worker; gc1e and gc1f cover the pages' own closes."""
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    first = _cif(tmp_path, 'phaseA', 4.1)
    _chooser(monkeypatch, first)
    with empty_page(qapp) as page:
        view = page._shell.scientific
        source = _open(qapp, page).phase_source
        seen = _commands(view)
        with _held_parser(monkeypatch) as (gate, _on_gui, returned):
            source.add_cif_btn.click()
            assert _busy(source) and source.selection == (), source.add_cif_btn.toolTip()
            page._phase_dialog.close()
            qapp.processEvents()
            reopened = _open(qapp, page)  # reopened while the CIF is still read
            gate.set()
            _until(qapp, lambda: returned, what='the held parse to return')
            _spin(qapp, .3)
        source = reopened.phase_source
        assert source.selection == () and not _phase_commands(seen), 'nothing is selected late'
        with _held_parser(monkeypatch) as (gate, _on_gui, _returned):
            _until(qapp, source.add_cif_btn.isEnabled, what='Add CIF to be available')
            source.add_cif_btn.click()
            assert _busy(source)
            reopened.close()  # its add is retired to the view
            qapp.processEvents()
            assert view.shutdown_phase_overlays().status is PageCleanup.PENDING, 'closing waits for the worker'
            gate.set()
            _until(qapp, lambda: view.shutdown_phase_overlays().status is PageCleanup.CLEAN, what='a clean close')
        _spin(qapp, .3)
        assert not _phase_commands(seen)


def test_gc1e_ints_fitter_close_and_page_close_drop_a_held_add(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.scattering_workspace import build_scattering_workspace
    from xdart.gui.pages.services import empty_host_services
    from tests.xdart.pages.test_stitch_page import _Status
    monkeypatch.setenv('XDART_SESSION_FILE', str(tmp_path / 'fitter-session.json'))
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    _chooser(monkeypatch, _cif(tmp_path, 'phaseA', 4.1))
    handle = build_scattering_workspace(empty_host_services(_Status()), None)
    page = handle.widget
    try:
        page._open_analysis_mount('phase')
        qapp.processEvents()
        dialog = page._phase_dialog
        late = []
        dialog.phase_source.changed.connect(late.append)
        with _held_parser(monkeypatch) as (gate, _on_gui, returned):
            dialog.phase_source.add_cif_btn.click()
            assert _busy(dialog.phase_source) and dialog.phase_source.selection == (), (
                dialog.phase_source.add_cif_btn.toolTip())
            dialog.close()
            qapp.processEvents()
            page._open_analysis_mount('phase')
            qapp.processEvents()
            reopened = page._phase_dialog
            assert reopened is not None and reopened is not dialog
            gate.set()
            _until(qapp, lambda: returned, what='the held parse to return')
            _spin(qapp, .3)
        assert not late and reopened.phase_source.selection == (), 'neither Fitter receives the old result'
        reopened_late = []
        reopened.phase_source.changed.connect(reopened_late.append)
        with _held_parser(monkeypatch) as (gate, _on_gui, _returned):
            _until(qapp, reopened.phase_source.add_cif_btn.isEnabled, what='Add CIF to be available')
            reopened.phase_source.add_cif_btn.click()
            assert _busy(reopened.phase_source)
            assert handle.close().status is PageCleanup.PENDING, "the page closes only after the Fitter's worker"
            gate.set()
            _until(qapp, lambda: handle.close().status is PageCleanup.CLEAN, what='a clean page close')
        _spin(qapp, .3)
        assert not reopened_late
    finally:
        _close_handle(qapp, handle)


def test_gc1f_stitchs_fit_host_close_and_page_shutdown_drop_a_held_add(qapp, tmp_path, monkeypatch):
    from tests.xdart.pages.test_stitch_fit_host import _finish
    _chooser(monkeypatch, _cif(tmp_path, 'phaseA', 4.1))
    with _stitch(qapp, tmp_path / 'stitch') as (page, _view, _payload):
        page.tools.page_tool('phase_analysis').click()
        qapp.processEvents()
        host = page._fit_hosts['phase']
        _finish(host, qapp)
        source = host.dialog.phase_source
        late = []
        source.changed.connect(late.append)
        with _held_parser(monkeypatch) as (gate, _on_gui, returned):
            source.add_cif_btn.click()
            assert _busy(source) and source.selection == (), source.add_cif_btn.toolTip()
            host.dialog.close()
            qapp.processEvents()
            page.tools.page_tool('phase_analysis').click()  # reopened while the CIF is still read
            qapp.processEvents()
            gate.set()
            _until(qapp, lambda: returned, what='the held parse to return')
            _spin(qapp, .3)
        assert not late and source.selection == (), 'the reopened Fitter receives no old result'
        _finish(host, qapp)
        with _held_parser(monkeypatch) as (gate, _on_gui, _returned):
            _until(qapp, source.add_cif_btn.isEnabled, what='Add CIF to be available')
            source.add_cif_btn.click()
            assert _busy(source)
            assert page.shutdown().status is PageCleanup.PENDING, "Stitch closes only after the Fitter's worker"
            gate.set()
            _until(qapp, lambda: page.shutdown().status is PageCleanup.CLEAN, what='a clean Stitch shutdown')
        _spin(qapp, .3)
        assert not late


# ---- GC3 (R4): grid wording, dash-cell reasons, one status line, Fitter tooltips ---------------------------------

def test_gc3a_the_grid_rows_disabled_tooltip_names_the_coordinate_grid(qapp):
    from xdart.gui.tabs.scattering.scientific_plot_options import GRID_TOOLTIP
    with _shell(qapp, 'Int 2D') as view:
        dialog = view.plot_options_dialog
        grid = dialog.coordinate_grid
        for apply in (lambda: None, lambda: dialog.set_grid_available(False)):
            apply()
            tip = grid.toolTip()
            assert not grid.isEnabled() and tip.startswith(GRID_TOOLTIP), tip
            reason = tip[len(GRID_TOOLTIP):].strip().lower()
            assert 'coordinate grid' in reason and 'ring' not in reason, tip


@contextmanager
def _fitted_dialog(qapp, tmp_path):
    """PHASE-ANALYSIS-128: Int's mounted Phase Analysis window with a real lattice-plus-CIF displayed fit delivered
    (the retired pattern-provider path's successor)."""
    from tests.core.test_reference_phase_sources import WAVELENGTH, _fit_inputs
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    from xrd_tools.analysis.display_fit_operations import DisplayedPhaseFitPlan, run_displayed_phase_fit
    paths, trace = _fit_inputs(tmp_path)
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        dialog.phase_source.add_cif_paths((str(paths[1]),))
        assert dialog.phase_source.add_lattice('NaCl cell', a=5.64, space_group=225) is not None
        dialog.set_vnext_trace(trace)
        result = run_displayed_phase_fit(DisplayedPhaseFitPlan(trace, wavelength_angstrom=WAVELENGTH,
                                                               **dialog.vnext_fit_values()))
        assert result.code == 'OK', (result.code, result.diagnostics)
        dialog.set_vnext_result(result)
        qapp.processEvents()
        yield dialog


def test_gc3b_a_lattice_plus_cif_fits_dash_cells_carry_their_own_columns_reason(qapp, tmp_path):
    with _fitted_dialog(qapp, tmp_path) as dialog:
        table = dialog.table
        headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
        dashes = [(row, column) for row in range(table.rowCount()) for column in range(table.columnCount())
                  if table.item(row, column) is not None and table.item(row, column).text() == '—']
        assert table.rowCount() and dashes, (dialog.status.text(), headers)
        # declared oracle repair (PHASE-ANALYSIS-128): the one displayed path's table (Name, Value, Lattice, Kind,
        # StdErr), as Int and Stitch showed it; the a/b/c columns were the retired standalone path's
        assert {headers[column] for _row, column in dashes} == {'Value'}, headers
        for row, column in dashes:
            tip = table.item(row, column).toolTip()
            assert tip and 'fraction' in tip.lower(), (row, headers[column], tip)


def test_gc3c_a_source_message_shows_in_exactly_one_visible_fitter_label(qapp, tmp_path):
    """PHASE-ANALYSIS-128: the one window shows a source message once, on its own status line (the retired
    Reference Phases window's own-status half went with it)."""
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        dialog.phase_source.add_library_entry('0' * 64)
        qapp.processEvents()
        message = dialog.phase_source.reason
        shown = [label for label in dialog.findChildren(QtWidgets.QLabel)
                 if label.isVisible() and message and message in label.text()]
        assert message and len(shown) == 1, (message, [label.objectName() for label in shown])
        assert shown[0] is dialog.status and dialog.status.toolTip()


def test_gc3d_every_visible_fitter_widget_resolves_a_tooltip_and_each_control_has_its_own(qapp, tmp_path):
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    from xdart.gui.tabs.scattering.results_table import ResultsTable
    shown = (QtWidgets.QAbstractButton, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QLineEdit,
             QtWidgets.QLabel, QtWidgets.QAbstractItemView, QtWidgets.QHeaderView)
    controls = (QtWidgets.QAbstractButton, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QLineEdit,
                ResultsTable)
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        dialog.phase_source.add_cif_paths(tuple(str(_cif(tmp_path, f'p{index}', 4.1 + .15 * index))
                                                for index in range(2)))
        assert dialog.phase_source.add_lattice('cell', a=4.0, space_group=221) is not None
        dialog.show()
        qapp.processEvents()
        widgets = [widget for widget in dialog.findChildren(QtWidgets.QWidget)
                   if widget.isVisibleTo(dialog) and isinstance(widget, shown)
                   and not isinstance(widget.parent(), (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox))]

        def name(widget):
            text = widget.text() if hasattr(widget, 'text') and callable(widget.text) else ''
            return type(widget).__name__, widget.objectName(), str(text)[:30]

        def resolves(widget):
            while widget is not None and widget is not dialog:
                if widget.toolTip():
                    return True
                widget = widget.parentWidget()
            return False
        own = [name(widget) for widget in widgets
               if (isinstance(widget, controls) or widget.objectName() == 'controlsSectionTitle')
               and not widget.toolTip()]
        assert not own, own
        unresolved = [name(widget) for widget in widgets if not resolves(widget)]
        assert not unresolved, unresolved


# ---- GC4 (R6d): the owner accessor ---------------------------------------------------------------------------------

def test_gc4_no_module_reads_a_latest_request_owners_private_worker(qapp):
    import xdart
    root = Path(xdart.__file__).parent
    reads = [f'{path.relative_to(root)}:{number}' for path in sorted(root.rglob('*.py'))
             for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
             if re.search(r'(?<!self)\._owner\b', line)]
    assert not reads, reads
    with _shell(qapp) as view:
        assert view.phase_worker is view._phase_requests.worker


# ---- GC5 (R7, D4, CR8): the reference palette and dashed lines ------------------------------------------------------

def test_gc5_phase_colours_come_from_the_reference_palette_and_draw_dashed(qapp, tmp_path):
    from xdart.gui.widgets.scientific_plot import trace_colors
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    entries = [library.add_cif(_cif(tmp_path, f'phase{index}', 4.1 + .15 * index)) for index in range(3)]
    with _stitch(qapp, tmp_path / 'stitch') as (_page, view, _payload):
        source = _window(qapp, view).phase_source
        source.set_selection(tuple((entry.id, True) for entry in entries))
        _settle(qapp, view, lambda: all(_marker(view, entry.id) for entry in entries))

        def colour(identifier):
            return tuple(_marker(view, identifier).lines.opts['pen'].color().getRgb()[:3])
        colours = [colour(entry.id) for entry in entries]
        assert not set(colours) & set(trace_colors('Default', 10)), ('no phase shares a trace colour', colours)
        dash = QtCore.Qt.PenStyle.DashLine
        for entry in entries:
            assert _marker(view, entry.id).lines.opts['pen'].style() == dash
            assert view._phase_legends[entry.id].opts['pen'].style() == dash
        from xdart.gui.widgets.scientific_plot import REFERENCE_COLORS, reference_colors
        assert REFERENCE_COLORS == ((230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
                                    (213, 94, 0), (204, 121, 167), (96, 96, 96))
        assert reference_colors(3) == REFERENCE_COLORS[:3] and colours == list(REFERENCE_COLORS[:3])
        # RW3's slot stability: hiding or removing a phase recolours no other.
        source.set_selection(((entries[0].id, True), (entries[1].id, False), (entries[2].id, True)))
        _settle(qapp, view, lambda: not _marker(view, entries[1].id))
        assert [colour(entries[0].id), colour(entries[2].id)] == [colours[0], colours[2]]
        source.set_selection(((entries[1].id, True), (entries[2].id, True)))
        _settle(qapp, view, lambda: not _marker(view, entries[0].id) and _marker(view, entries[1].id))
        assert [colour(entries[1].id), colour(entries[2].id)] == colours[1:]


# ---- GC6 (C6, CR6): the top bar's Phases toggle ---------------------------------------------------------------------

def _phases_button(view):
    """The top bar's widget right after Set BG: the Phases toggle (C6)."""
    bar = view._top_bar_row
    item = bar.itemAt(bar.indexOf(view.background) + 1)
    return None if item is None else item.widget()


def _phase_state(view, identifiers, panes=()):
    return {identifier: (_marker(view, identifier) is not None,
                         tuple(bool(_pane_items(pane, identifier)) for pane in panes)) for identifier in identifiers}


def _legend_names(view):
    return [label.text for _sample, label in view.legend.items]


def test_gc6a_phases_switches_every_int_phase_off_and_on_with_real_clicks(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        button = _phases_button(view)
        assert button is not None and button.text() == 'Phases' and button.isCheckable(), (
            'a checkable Phases toggle follows Set BG', None if button is None else button.text())
        window = _window(qapp, view)
        window.phase_source.set_selection(((entry.id, True), (LAB6, True)))
        ids, panes = (entry.id, LAB6), (view.raw, view.cake)
        _settle(qapp, view, lambda: all(_marker(view, i) and _pane_items(view.raw, i) for i in ids))
        assert button.isEnabled() and button.isChecked() and button.toolTip()
        assert button.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
        before = _options(page)
        drawn = _phase_state(view, ids, panes)
        colours, slots = dict(view._phase_colors), dict(view._phase_color_slots)
        names = _legend_names(view)
        focus = QtWidgets.QApplication.focusWidget()
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: not any(_marker(view, i) for i in ids))
        assert not button.isChecked() and QtWidgets.QApplication.focusWidget() is focus
        assert _phase_state(view, ids, panes) == {i: (False, (False, False)) for i in ids}
        assert not {entry.name, 'LaB6'} & set(_legend_names(view)), _legend_names(view)
        after = _options(page)
        assert (after.phase_references, after.phase_floors, after.phase_min_intensity, after.phase_global_floor) == (
            before.phase_references, before.phase_floors, before.phase_min_intensity, before.phase_global_floor)
        assert after.phase_references_shown is False
        assert window.phase_source.selection == before.phase_references
        assert 'switched off' in window.rings_note.text().lower(), window.rings_note.text()
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: all(_marker(view, i) and _pane_items(view.raw, i) for i in ids))
        assert button.isChecked() and QtWidgets.QApplication.focusWidget() is focus
        assert _options(page).phase_references_shown is True
        assert _phase_state(view, ids, panes) == drawn and _legend_names(view) == names
        assert dict(view._phase_colors) == colours and dict(view._phase_color_slots) == slots
        (x0, x1), (y0, y1) = view.curve.getPlotItem().vb.viewRange()
        point = view.curve.getPlotItem().vb.mapViewToScene(QtCore.QPointF((x0 + x1) / 2, (y0 + y1) / 2))
        view._mouse_moved((point,))
        assert view.cursor_position.text.startswith('x='), 'the cursor readout is undisturbed'


def test_gc6b_phases_switches_stitchs_phases_off_and_on(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        page.show()
        qapp.processEvents()
        button = _phases_button(view)
        assert button is not None and button.text() == 'Phases' and button.isVisible(), (
            None if button is None else button.text())
        window = _window(qapp, view)
        window.phase_source.set_selection(((entry.id, True), (LAB6, True)))
        _settle(qapp, view, lambda: _marker(view, entry.id) and _marker(view, LAB6))
        names = _legend_names(view)
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: not _marker(view, entry.id) and not _marker(view, LAB6))
        assert not {entry.name, 'LaB6'} & set(_legend_names(view))
        assert _options(page).phase_references == ((entry.id, True), (LAB6, True))
        QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: _marker(view, entry.id) and _marker(view, LAB6))
        assert _legend_names(view) == names


def test_gc6c_phases_is_disabled_without_phases_and_hidden_where_phases_are_unavailable(qapp):
    with _shell(qapp) as view:
        button = _phases_button(view)
        assert button is not None and button.text() == 'Phases', None if button is None else button.text()
        assert not button.isEnabled() and 'no reference phase' in button.toolTip().lower(), button.toolTip()
    with _shell(qapp, '2D Viewer') as view:
        assert _phases_button(view).text() == 'Phases' and not _phases_button(view).isVisibleTo(view)
    from tests.xdart.pages.test_rsm_page import _close, _page
    rsm = _page(qapp)
    try:
        assert _phases_button(rsm.view).text() == 'Phases' and not _phases_button(rsm.view).isVisibleTo(rsm)
    finally:
        _close(rsm, qapp)
    from xdart.gui.tabs.calibration.fit_windows import CalibrationFitWindow
    from xdart.gui.tabs.rsm.tool_windows import RSMCustomSliceWindow
    for build in (CalibrationFitWindow, RSMCustomSliceWindow):
        window = build()
        view = window.pattern_view if hasattr(window, 'pattern_view') else window.view
        try:
            window.show()
            qapp.processEvents()
            assert _phases_button(view).text() == 'Phases' and not _phases_button(view).isVisibleTo(window)
        finally:
            window.close()
            window.deleteLater()
            qapp.processEvents()


def test_gc6d_int_session_and_stitch_settings_keep_phases_shown(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.scattering_workspace import build_scattering_workspace
    from xdart.gui.pages.services import empty_host_services
    from xdart.utils.session import load_session, save_session
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from tests.xdart.pages.test_stitch_page import _Status
    monkeypatch.setenv('XDART_SESSION_FILE', str(tmp_path / 'restart-session.json'))
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    services = empty_host_services(_Status())
    save_session({'scattering_phase_references': {'references': [[entry.id, True]], 'min_intensity': 5.,
                                                  'shown': False}})
    handle = build_scattering_workspace(services, None)
    try:
        view = handle.widget.findChild(ScatteringWorkspaceShell).scientific
        button = _phases_button(view)
        assert button is not None and button.text() == 'Phases' and not button.isChecked(), (
            None if button is None else button.text())
        assert handle.widget._preferences.plot_options.phase_references_shown is False
    finally:
        receipt = _close_handle(qapp, handle)
    assert receipt.status is PageCleanup.CLEAN
    assert load_session()['scattering_phase_references']['shown'] is False
    save_session({'scattering_phase_references': {'references': [[entry.id, True]], 'min_intensity': 5.}})
    handle = build_scattering_workspace(services, None)
    try:
        assert handle.widget._preferences.plot_options.phase_references_shown is True, 'a missing key means on'
    finally:
        receipt = _close_handle(qapp, handle)
    assert receipt.status is PageCleanup.CLEAN
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        page.show()
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        _settle(qapp, view, lambda: _marker(view, entry.id))
        QtTest.QTest.mouseClick(_phases_button(view), QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: not _marker(view, entry.id))
        document = page.settings()
        assert document['phase_references']['shown'] is False
        old = json.loads(json.dumps(document))
        del old['phase_references']['shown']
        page.apply_settings(old)
        _settle(qapp, view, lambda: _marker(view, entry.id))
        assert _options(page).phase_references_shown is True and _phases_button(view).isChecked()
        page.apply_settings(document)
        _settle(qapp, view, lambda: not _marker(view, entry.id))
        assert _options(page).phase_references_shown is False


# ---- GC7 (C7, CR6): hkl on hover, with real mouse moves -------------------------------------------------------------

def _tip_text():
    return QtWidgets.QToolTip.text() if QtWidgets.QToolTip.isVisible() else ''


def _hide_tip(qapp):
    QtWidgets.QToolTip.hideText()
    _until(qapp, lambda: not QtWidgets.QToolTip.isVisible(), seconds=3, what='the previous tooltip to hide')


def _move(widget, local):
    """A real mouse move to ``local`` of ``widget``, delivered to its own top-level window. Oracle repairs: offscreen,
    a widget-level move lands on whichever overlapping window is on top, and an earlier test's page can stay visible
    until its deferred delete (probe p06); pyqtgraph's scene ignores a move within 10 ms of the one before
    (``mouseRateLimit``), so each move waits 20 ms (the GC7d diagnosis, scratch-state/flake8)."""
    _spin(QtWidgets.QApplication.instance(), .02)
    window = widget.window()
    QtTest.QTest.mouseMove(window.windowHandle(), widget.mapTo(window, local))


def _hover(qapp, canvas, box, point, *, want=None, seconds=1.5):
    """A real mouse move (QTest) to view coordinates ``point`` of ``box`` on the graphics view ``canvas``; the
    tooltip shown by then ('' when none). ``want`` ends the wait as soon as the text satisfies it.

    Oracle repair (probe p05): the move starts from the viewport's corner, because Qt drops a move to the position
    it last recorded, and a QTest click does not record one."""
    _move(canvas.viewport(), QtCore.QPoint(1, 1))
    _spin(qapp, .05)
    local = canvas.mapFromScene(box.mapViewToScene(QtCore.QPointF(*map(float, point))))
    _move(canvas.viewport(), local)
    deadline = time.monotonic() + seconds
    text = ''
    while time.monotonic() < deadline:
        qapp.processEvents()
        text = _tip_text()
        if want is not None and want(text):
            break
        time.sleep(.01)
    return text


def _plot(view):
    return view.curve, view.curve.getPlotItem().vb


def _pane(pane):
    return pane.canvas.image_win, pane.canvas.imageViewBox


def _hex(view, identifier):
    return '#%02x%02x%02x' % tuple(view._phase_colors[identifier][:3])


def _label(hkl):
    return '(' + ''.join(str(index) for index in hkl) + ')'


def _q_range(view):
    x = np.concatenate([np.asarray(item.getData()[0], dtype=float) for item in view.curve.getPlotItem().listDataItems()
                        if item.getData()[0] is not None])
    x = x[np.isfinite(x)]
    return float(x.min()), float(x.max())


def _segments(item):
    x, y = (np.asarray(values, dtype=float) for values in item.getData())
    breaks = np.flatnonzero(~np.isfinite(x) | ~np.isfinite(y))
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, len(x)]
    return [np.column_stack((x[a:b], y[a:b])) for a, b in zip(starts, stops) if b - a >= 2]


def test_gc7a_a_merged_cif_stick_lists_every_family_its_d_spacing_and_intensity(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        page.resize(1600, 1000)
        page.show()
        qapp.processEvents()
        low_q, high_q = _q_range(view)
        a = 6. * math.pi / (low_q + .55 * (high_q - low_q))  # the merged (300)/(221) line mid-range
        entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', a))
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        close_phase_window(view)
        _settle(qapp, view, lambda: _marker(view, entry.id))
        family = view.phase_families[entry.id]
        index = int(np.argmin(np.abs(np.asarray(family.q_lines) - 6. * math.pi / a)))
        x = _marker(view, entry.id).reference_positions[index]
        canvas, box = _plot(view)
        low, high = box.viewRange()[1]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, (x, low + .02 * (high - low)), want=lambda t: all(m in t for m in MERGED))
        assert all(merged in text for merged in MERGED), ('the merged stick lists both families', text)
        value = family.line_values[index]
        labels = [_label(hkl) for hkl in value.hkls]
        assert sorted(labels) == sorted(MERGED), labels
        assert [text.index(label) for label in labels] == sorted(text.index(label) for label in labels)
        assert f'd {value.d_spacing:.3f} Å' in text and f'{value.intensity:.0f} %' in text, text
        assert _hex(view, entry.id) in text.lower() and '■' in text, text
        short = [i for i, line in enumerate(family.line_values) if line.intensity is not None and line.intensity < 40]
        assert short, [line.intensity for line in family.line_values]
        _hide_tip(qapp)
        above = _hover(qapp, canvas, box, (_marker(view, entry.id).reference_positions[short[0]],
                                           low + .8 * (high - low)))
        assert '■' not in above and ' Å' not in above, ('above a short stick there is no phase tooltip', above)
        _hide_tip(qapp)
        away = _hover(qapp, canvas, box, ((x + _marker(view, entry.id).reference_positions[index - 1]) / 2.,
                                          low + .02 * (high - low)))
        assert '■' not in away, away


def test_gc7b_a_ring_on_the_raw_pane_and_cake_and_the_raw_popup_lists_its_reflection(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        page.resize(1600, 1000)
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        close_phase_window(view)
        _settle(qapp, view, lambda: _pane_items(view.raw, entry.id) and _pane_items(view.cake, entry.id))
        canvas, box = _pane(view.raw)
        segment = _segments(_pane_items(view.raw, entry.id)[0])[0]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, segment[len(segment) // 2], want=lambda t: re.search(r'\(\d{3}\)', t))
        assert re.search(r'\(\d{3}\)', text), ('a ring names its reflection', text)
        family = view.phase_families[entry.id]
        expected = family.rings[0].value
        assert all(_label(hkl) in text for hkl in expected.hkls) and f'd {expected.d_spacing:.3f} Å' in text, text
        cake_canvas, cake_box = _pane(view.cake)
        cut = _segments(_pane_items(view.cake, entry.id)[0])[1]
        _hide_tip(qapp)
        text = _hover(qapp, cake_canvas, cake_box, cut.mean(axis=0), want=lambda t: re.search(r'\(\d{3}\)', t))
        line = family.line_values[1]
        assert all(_label(hkl) in text for hkl in line.hkls) and f'd {line.d_spacing:.3f} Å' in text, text
    with run_page(qapp, tmp_path / 'popup', energies=(16, 16), mode='Int 1D') as (page, view):
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        close_phase_window(view)
        view.raw_popup_button.click()
        qapp.processEvents()
        select(qapp, page, page._context_controller.navigation.current)
        popup = view.raw_popup_image
        _settle(qapp, view, lambda: popup.image.image is not None and popup.image.image.shape == (487, 195)
                and _pane_items(popup, entry.id))
        # Oracle repair (probe p05): offscreen, a move over the page's overlapping window never reaches the popup.
        popup.window().move(page.window().geometry().right() + 40, page.window().geometry().top())
        qapp.processEvents()
        canvas, box = _pane(popup)
        segment = _segments(_pane_items(popup, entry.id)[0])[0]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, segment[len(segment) // 2], want=lambda t: re.search(r'\(\d{3}\)', t))
        expected = view.phase_families[entry.id].rings[0].value
        assert all(_label(hkl) in text for hkl in expected.hkls), text


def test_gc7c_coincident_lines_list_each_phase_and_a_calibrant_line_shows_its_d_spacing(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        page.resize(1600, 1000)
        page.show()
        qapp.processEvents()
        low_q, high_q = _q_range(view)
        a = 2. * math.pi / (low_q + .3 * (high_q - low_q))  # (100) a third of the way along
        cif = library.add_cif(_cif(tmp_path, 'CsCl', a))
        lattice = library.add_lattice('cell', a=a, space_group=221)
        _window(qapp, view).phase_source.set_selection(((cif.id, True), (lattice.id, True), (LAB6, True)))
        close_phase_window(view)
        _settle(qapp, view, lambda: all(_marker(view, i) for i in (cif.id, lattice.id, LAB6)))
        canvas, box = _plot(view)
        low, high = box.viewRange()[1]
        families = view.phase_families
        index = int(np.argmin(np.abs(np.asarray(families[cif.id].q_lines) - 2. * math.pi / a)))
        x = _marker(view, cif.id).reference_positions[index]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, (x, low + .02 * (high - low)), want=lambda t: t.count('(100)') >= 2)
        assert text.count('(100)') == 2, ('both coincident phases list the line', text)
        assert _hex(view, cif.id) in text.lower() and _hex(view, lattice.id) in text.lower(), text
        assert text.count('%') == 1, 'the lattice entry omits the intensity'
        # A LaB6 line at least 8 px from every CIF and lattice line shows only its d-spacing.
        def pixels(position):
            return box.mapViewToScene(QtCore.QPointF(float(position), low)).x()
        lab6 = _marker(view, LAB6).reference_positions
        cif_x = np.asarray(_marker(view, cif.id).reference_positions)
        others = np.array([pixels(position) for position in (*cif_x, *_marker(view, lattice.id).reference_positions)])
        alone = [i for i, position in enumerate(lab6) if np.min(np.abs(others - pixels(position))) > 8]
        assert alone, 'a separate calibrant line exists'
        q = families[LAB6].line_values[alone[0]].q
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, (lab6[alone[0]], low + .5 * (high - low)), want=lambda t: 'd =' in t)
        assert f'd = {2. * math.pi / q:.3f} Å' in text and '(' not in text and '%' not in text, text
        # After reachable-range filtering every entry still names its own reflection: the highest drawn CIF line.
        last = len(cif_x) - 1
        q = families[cif.id].q_lines[last]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, (cif_x[last], low + .02 * (high - low)),
                      want=lambda t: f'd {2. * math.pi / q:.3f} Å' in t)
        assert f'd {2. * math.pi / q:.3f} Å' in text, (q, text)
        assert all(_label(hkl) in text for hkl in families[cif.id].line_values[last].hkls), text


def test_gc7d_a_phase_tooltip_clears_when_the_cursor_leaves_phases_switch_off_or_the_frame_changes(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    with run_page(qapp, tmp_path / 'int', energies=(16, 17)) as (page, view):
        page.resize(1600, 1000)
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        close_phase_window(view)
        _settle(qapp, view, lambda: _pane_items(view.raw, entry.id))
        canvas, box = _pane(view.raw)

        def show():
            segment = _segments(_pane_items(view.raw, entry.id)[0])[0]
            _hide_tip(qapp)
            text = _hover(qapp, canvas, box, segment[len(segment) // 2], want=lambda t: re.search(r'\(\d{3}\)', t))
            assert re.search(r'\(\d{3}\)', text), ('a ring names its reflection', text)
        show()
        _move(canvas.viewport(), QtCore.QPoint(2, 2))
        _until(qapp, lambda: '■' not in _tip_text(), seconds=3, what='the tooltip to clear on leaving')
        show()
        QtTest.QTest.mouseClick(_phases_button(view), QtCore.Qt.MouseButton.LeftButton)
        _until(qapp, lambda: '■' not in _tip_text(), seconds=3, what='the tooltip to clear when phases switch off')
        QtTest.QTest.mouseClick(_phases_button(view), QtCore.Qt.MouseButton.LeftButton)
        _settle(qapp, view, lambda: _pane_items(view.raw, entry.id))
        show()
        select(qapp, page, page._context_controller.frame_keys[0])
        _until(qapp, lambda: '■' not in _tip_text(), seconds=3, what='the tooltip to clear when the frame changes')
    _until(qapp, lambda: '■' not in _tip_text(), seconds=3, what='no tooltip outlives the retired view')


# ---- GC8 (C8): intensity-scaled sticks ----------------------------------------------------------------------------

def test_gc8_cif_sticks_scale_with_relative_intensity_and_other_lines_span_the_view(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    cif = library.add_cif(_cif(tmp_path, 'CsCl', 4.1))
    lattice = library.add_lattice('cell', a=5.64, space_group=225)
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        _window(qapp, view).phase_source.set_selection(((cif.id, True), (lattice.id, True), (LAB6, True)))
        _settle(qapp, view, lambda: all(_marker(view, i) for i in (cif.id, lattice.id, LAB6)))
        box = view.curve.getPlotItem().vb

        def sticks(identifier):
            _x, y = _marker(view, identifier).lines.getData()
            y = np.asarray(y, dtype=float)
            return y[0::3], y[1::3]
        low, high = box.viewRange()[1]
        _bottom, top = sticks(cif.id)
        assert top.min() < high - 1e-6 * (high - low), 'a weak CIF reflection draws a short stick'

        def check():
            low, high = box.viewRange()[1]
            intensities = np.array([line.intensity for line in view.phase_families[cif.id].line_values])
            bottom, top = sticks(cif.id)
            np.testing.assert_allclose(bottom, low)
            np.testing.assert_allclose(top, low + intensities / 100. * (high - low), rtol=1e-9, atol=1e-9 * (high - low))
            for identifier in (lattice.id, LAB6):
                bottom, top = sticks(identifier)
                np.testing.assert_allclose(bottom, low)
                np.testing.assert_allclose(top, high)
        check()
        tip = _marker(view, cif.id).toolTip()
        assert 'relative reference intensities' in tip and 'not measured counts' in tip, tip
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, 'Log', ('other', 'intensity_scale')))
        _settle(qapp, view)
        qapp.processEvents()
        assert _options(page).intensity_scale == 'Log'
        check()
        low, high = box.viewRange()[1]
        box.setYRange(low + .2 * (high - low), high - .1 * (high - low), padding=0)
        qapp.processEvents()
        check()


# ---- Stage E: rings and geometry ------------------------------------------------------------------------------------

def test_ge1_point_frames_at_distinct_energies_draw_their_own_rings(qapp, tmp_path):
    from tests.xdart.scattering.test_p1b_output_graph import _write_tiff
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.sources.point_images import PointImagePolicy
    from xrd_tools.sources.selection import image_series_spec
    points = tmp_path / 'points'
    points.mkdir()
    (points / 'calibrant').write_text('#F calibrant\n#E 1\n#O0 del  nu\n\n#S 1 ascan del 10 12 2 1\n#P0 10 0\n'
                                      '#N 4\n#L del  nu  energy  i0\n10 0 16 100\n11 0 17 200\n')
    paths = [points / f'calibrant_scan001_{index:04}.tif' for index in range(4)]
    for path, value in zip(paths, (10, 6, 12, 8)):
        _write_tiff(path, value)
    spec = image_series_spec(paths[0], metadata_format='spec')
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'average').to_dict()})
    with run_page(qapp, tmp_path / 'run', source_spec=spec) as (page, view):
        controller = page._context_controller
        keys = controller.frame_keys
        assert len(keys) == 2, keys
        choice(view)
        choose(qapp, view)
        settle(qapp, view, lambda: rings(view.raw))
        last = controller.project(keys[-1]).view.geometry.poni
        drawn = tuple(np.column_stack(item.getData()) for item in rings(view.raw))
        first = select(qapp, page, keys[0]).view.geometry.poni
        energies = sorted(round(wavelength_m_to_energy_eV(poni['wavelength'])) for poni in (first, last))
        assert energies == [16000, 17000], energies

        def moved():
            now = tuple(np.column_stack(item.getData()) for item in rings(view.raw))
            return bool(now) and (len(now) != len(drawn) or any(
                a.shape != b.shape or not np.array_equal(a, b, equal_nan=True) for a, b in zip(now, drawn)))
        settle(qapp, view, moved)
        assert moved(), 'each point draws the rings of its own energy'


@contextmanager
def _average_page(qapp, root):
    """Int's real page running Average Scan over a two-frame series; yields once its result shows in Browse."""
    from tests.xdart.scattering.test_energy_scan_run import _series
    from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.sources import image_series_spec
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, (16, 16))
    poni = root / 'cal.poni'
    write_poni(poni)
    poni.write_text(poni.read_text().replace('Distance: 0.1234', 'Distance: 0.03'))
    intent = _intent(raw, root / 'average.nexus', poni, processing_mode='Int 1D')
    intent.source_spec = image_series_spec(raw, metadata_format='txt')
    intent.run_options['series_average'] = True
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2.))
    shell = page.findChild(ScatteringWorkspaceShell)
    try:
        page.show()
        shell.run_controls.startButton.click()
        controller = page._context_controller
        _until(qapp, lambda: bool(controller.frame_keys) and not controller.browse_pending
               and page._processed_browser.pending_average_reload is None, seconds=60,
               what=f'the Average result in Browse ({page._notice_text})')
        page._refresh_shell()
        qapp.processEvents()
        yield page, shell.scientific
    finally:
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_ge2_a_real_average_result_shows_the_run_reason_no_raw_rings_and_reference_lines(qapp, tmp_path):
    from xrd_tools.geometry_figures import RUN_RING_REASON
    with _average_page(qapp, tmp_path / 'average') as (page, view):
        window = choice(view)
        choose(qapp, view)
        settle(qapp, view, lambda: _marker(view, LAB6))
        assert window.rings_note.text() == RUN_RING_REASON, window.rings_note.text()
        assert not rings(view.raw) and (view.raw_popup_image is None or not rings(view.raw_popup_image))
        assert _marker(view, LAB6).reference_positions, 'the 1D reference lines remain'


@contextmanager
def _live_run(qapp, root, prepare):
    """A live Int 2D Run on a watched directory holding one frame, with ``prepare(page, view)`` called after the page
    shows and before Run. Yields the page, its shell, ``arrive(index)`` (one more frame at the calibration's energy,
    its image renamed into place whole, as an acquisition writes) and the drained Run events. A Run still going at
    exit is stopped before the page closes. The frames carry no metadata sidecar: a sidecar read that overlaps a new
    arrival fails the Run (LIVE-METADATA-DISCOVERY-RACE, pre-existing at 305df401, ledgered)."""
    import os
    import fabio
    from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.sources.selection import DirectorySourceSpec
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    raw = root / 'raw'
    raw.mkdir(parents=True)
    rows, columns = np.mgrid[:195, :487]
    image = (100 + (rows + 2 * columns) % 97).astype(np.uint16)  # test_energy_scan_run._series' frame

    def arrive(index):
        target = raw / f'escan_{index:04d}.tif'
        partial = raw / f'escan_{index:04d}.part'
        fabio.tifimage.TifImage(data=image).write(str(partial))
        os.replace(partial, target)
    arrive(1)
    poni = root / 'cal.poni'
    write_poni(poni)
    poni.write_text(poni.read_text().replace('Distance: 0.1234', 'Distance: 0.03'))
    intent = _intent(raw / 'escan_0001.tif', root / 'processed', poni, processing_mode='Int 2D', live_mode=True)
    intent.source_spec = DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format=None)
    executor = StandardRunExecutor(join_timeout=2.)
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    observed, drain = [], executor.drain_events

    def observe():
        events = drain()
        observed.extend(events)
        return events
    executor.drain_events = observe
    terminal = (StandardEventKind.STOPPED, StandardEventKind.FINISHED, StandardEventKind.FAILED)
    try:
        page.show()
        prepare(page, shell.scientific)
        shell.run_controls.startButton.click()
        yield page, shell, arrive, observed
    finally:
        if shell.run_controls.stopButton.isEnabled():
            shell.run_controls.stopButton.click()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not any(event.kind in terminal for event in observed):
            qapp.processEvents()
            time.sleep(.005)
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_ge3_frames_arriving_at_one_energy_list_the_calibrant_rings_once(qapp, tmp_path, monkeypatch):
    """Each frame of a live Run arrives, and is drawn with its own image and LaB6's rings, while the Run is still
    going; the rings are listed once. Oracle repair (October 12, the maintainer's word): this row first ran a
    four-frame series, which reached the page in one 125 ms drain with its terminal, so no frame was drawn while it
    ran and a ring context keyed on the frame survived (probes p08, p09)."""
    import xrd_tools.analysis.phase_references as provider
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    calls, actual = [], provider.build_phase_references

    def count(*args, **kwargs):
        if 'two_theta_range' in kwargs:  # a detector-ring listing (1D lines list by q range)
            calls.append(tuple(entry.id for entry in args))
        return actual(*args, **kwargs)

    def prepare(_page, view):
        choice(view)
        choose(qapp, view)
        monkeypatch.setattr(provider, 'build_phase_references', count)
    with _live_run(qapp, tmp_path, prepare) as (page, shell, arrive, observed):
        view, controller = shell.scientific, page._context_controller
        for index in range(1, 5):
            if index > 1:
                arrive(index)

            def drawn():
                state, current = page._last_scientific_projection, controller.navigation.current
                return (current is not None and current.local_frame_label == index and state is not None
                        and state.heavy is not None and state.heavy.frame is current
                        and not view.phase_worker.busy and bool(rings(view.raw)))
            _until(qapp, drawn, seconds=30, what=f'frame {index} drawn with its image and rings')
            assert shell.run_controls.stopButton.isEnabled(), 'the frame is drawn while the Run is still going'
            assert view.phase_rings_reason == '', view.phase_rings_reason
        assert len(controller.frame_keys) == 4
        assert calls == [(LAB6,)], calls
        shell.run_controls.stopButton.click()
        _until(qapp, lambda: any(event.kind is StandardEventKind.STOPPED for event in observed), seconds=60,
               what='the stopped Run')
        settle(qapp, view)
        assert calls == [(LAB6,)], calls


def test_ge6_an_evicted_run_frame_says_its_rings_wait_for_its_image(qapp, tmp_path):
    from xrd_tools.geometry_figures import RUN_RING_REASON
    with run_page(qapp, tmp_path, energies=(16,) * 17, heavy_window=16) as (page, view):
        window = choice(view)
        choose(qapp, view)
        settle(qapp, view, lambda: rings(view.raw))
        controller = page._context_controller
        keys = controller.frame_keys
        store = controller.acquisition_context.publication_store.artifacts[keys[0].artifact].publications
        assert store.get(keys[0].local_frame_label).view.intensity_2d is None
        assert controller.select_navigation(keys[0], (keys[0],))
        page._refresh_shell()
        loading = window.rings_note.text()
        assert loading and loading != RUN_RING_REASON, ('the rings wait for the frame image', loading)
        from xrd_tools.geometry_figures import LOADING_RING_REASON
        assert loading == LOADING_RING_REASON
        select(qapp, page, keys[0])
        settle(qapp, view, lambda: rings(view.raw))
        assert rings(view.raw) and window.rings_note.text() == ''
