"""LT-SMALL-FIXES-20261014 stage 1: frozen oracle rows P1–P3 (Claude, October 14).

LT-001 (include checks ignore clicks and Space), PHASE-ROW-CLICK-SELECT-REMOVE (Remove deletes a phase the user
did not choose) and LT-002 (rows do not fit the window), in the Tools-panel Reference Phases window on Int's real
page and in the real Phase Fitter. Every check state, current row and selection on the seam under test comes from a
real QTest mouse or key event: on the style's check indicator, a row's label, a row's own number editor or the
Remove button. A click on a disabled number box is sent to the list's viewport at that point, which is where Qt
delivers a press a disabled child does not take. Setup (adding phases, the shared floor switch) uses the existing
owners' own paths.
"""
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

from tests.xdart.scattering.test_help_residuals_gui import _phases_button
from tests.xdart.scattering.test_phase_reference_overlays import _cif, _settle
from tests.xdart.scattering.test_predicted_ring_overlays import run_page
from tests.xdart.scattering.test_reference_phases_window import (LAB6, _commands, _ids, _marker, _options, _row,
                                                                 _window)
from xdart.gui.tabs.scattering.shell_values import ShellCommandKind

LEFT = QtCore.Qt.MouseButton.LeftButton
NONE = QtCore.Qt.KeyboardModifier.NoModifier
CHECKED = QtCore.Qt.CheckState.Checked
TIN = 'TiN_mp-492_conventional_standard'


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phase-library'))


class _Probe(QtWidgets.QStyledItemDelegate):
    """Only for ``initStyleOption``: where the style draws a row's check indicator."""


def _indicator(scans, item):
    option = QtWidgets.QStyleOptionViewItem()
    option.rect = scans.visualItemRect(item)
    option.state = QtWidgets.QStyle.StateFlag.State_Enabled
    _Probe().initStyleOption(option, scans.indexFromItem(item))
    return scans.style().subElementRect(QtWidgets.QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, scans)


def _label_point(scans, item):
    """A point on the row's name: right of its check box, left of its number box."""
    return QtCore.QPoint(_indicator(scans, item).right() + 12, scans.visualItemRect(item).center().y())


def _click(qapp, widget, point=None):
    QtTest.QTest.mouseClick(widget, LEFT, NONE, point if point is not None else widget.rect().center())
    qapp.processEvents()


def _focus_name(widget):
    return None if widget is None else f'{type(widget).__name__}:{widget.objectName() or "-"}'


def _focus_state(qapp, window, editor, line):
    """Where the keyboard focus is, for a failure message (CI-REDS diagnosis, 2026-10-29)."""
    return dict(platform=QtGui.QGuiApplication.platformName(), focus=_focus_name(qapp.focusWidget()),
                active=_focus_name(qapp.activeWindow()), window_active=window.isActiveWindow(),
                editor_focus=editor.hasFocus(), line_focus=line.hasFocus(), text=line.text(), value=editor.value())


def _selected(source):
    return [item.text() for item in source.browser.scans.selectedItems()]


def _remove_follows_selection(source):
    assert source.remove_btn.isEnabled() == bool(source.browser.scans.selectedItems()), (
        source.remove_btn.isEnabled(), _selected(source))
    assert source.remove_btn.toolTip(), 'Remove explains itself, enabled or not'


def _of(commands, path):
    return [command for command in commands
            if command.kind is ShellCommandKind.SET_PLOT_OPTION and command.path == ('other', path)]


def _flags(page):
    return dict(_options(page).phase_references)


def _plain_double_click():
    """Qt's own checkable list, no persistent editors: a double-click on the indicator, as the reference."""
    plain = QtWidgets.QListWidget()
    item = QtWidgets.QListWidgetItem('plain', plain)
    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(CHECKED)
    changes = []
    plain.itemChanged.connect(lambda changed: changes.append(changed.checkState()))
    plain.show()
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.mouseDClick(plain.viewport(), LEFT, NONE, _indicator(plain, item).center())
    QtWidgets.QApplication.processEvents()
    result = len(changes), item.checkState() == CHECKED
    plain.close()
    plain.deleteLater()
    QtWidgets.QApplication.processEvents()
    return result


def _rows_fit(qapp, source):
    """Every row's number box lies wholly inside the visible list; nothing scrolls sideways."""
    qapp.processEvents()
    scans = source.browser.scans
    viewport = scans.viewport().width()
    assert scans.horizontalScrollBar().maximum() == 0, ('no horizontal scroll range',
                                                        scans.horizontalScrollBar().maximum())
    width = source.browser.profile.result_values.width
    for entry, _shown in source.listed_entries():
        editor = source.cutoff_editor(entry.id)
        if editor is None:
            continue
        box = editor.geometry()
        assert editor.isVisible() and box.width() == width and box.left() >= 0 and box.right() < viewport, (
            entry.name, box, viewport)


def test_p1_include_checks_follow_real_indicator_clicks_and_space(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    a, b = (library.add_cif(_cif(tmp_path, name, a)) for name, a in (('phaseA', 4.1), ('phaseB', 4.25)))
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        window = _window(qapp, view)
        source, scans = window.phase_source, window.phase_source.browser.scans
        source.set_selection(((a.id, True), (b.id, True), (LAB6, True)))
        _settle(qapp, view, lambda: _marker(view, a.id) is not None and _marker(view, b.id) is not None)
        window.activateWindow()
        qapp.processEvents()
        commands = _commands(view)
        colour = _marker(view, b.id).lines.opts['pen'].color().getRgb()
        floors = _options(page).phase_floors
        assert source.global_switch.editor.isChecked() and not source.cutoff_editor(a.id).isEnabled()

        # A disabled box (shared minimum on): a real click on A's check box unchecks A, exactly once.
        _click(qapp, scans.viewport(), _indicator(scans, _row(source, a.id)).center())
        _settle(qapp, view)
        assert [command.value for command in _of(commands, 'phase_references')] == [
            ((a.id, False), (b.id, True), (LAB6, True))], 'one click sends exactly one include edit'
        assert _row(source, a.id).checkState() != CHECKED and _marker(view, a.id) is None
        assert _marker(view, b.id).lines.opts['pen'].color().getRgb() == colour
        assert _options(page).phase_floors == floors

        # A label click selects B; a press on A's indicator keeps B current and the release toggles A once.
        _click(qapp, scans.viewport(), _label_point(scans, _row(source, b.id)))
        assert scans.currentItem() is _row(source, b.id) and _selected(source) == [_row(source, b.id).text()]
        commands.clear()
        point = _indicator(scans, _row(source, a.id)).center()
        QtTest.QTest.mousePress(scans.viewport(), LEFT, NONE, point)
        qapp.processEvents()
        assert scans.currentItem() is _row(source, b.id) and not _of(commands, 'phase_references')
        QtTest.QTest.mouseRelease(scans.viewport(), LEFT, NONE, point)
        _settle(qapp, view, lambda: _marker(view, a.id) is not None)
        assert len(_of(commands, 'phase_references')) == 1 and _flags(page)[a.id] is True
        assert scans.currentItem() is _row(source, b.id)

        # Space and Select on the focused list toggle the current row (B), one edit each.
        scans.setFocus()
        qapp.processEvents()
        for key, expected in ((QtCore.Qt.Key.Key_Space, False), (QtCore.Qt.Key.Key_Select, True)):
            commands.clear()
            QtTest.QTest.keyClick(scans, key)
            _settle(qapp, view)
            assert len(_of(commands, 'phase_references')) == 1 and _flags(page)[b.id] is expected, key

        # The shared minimum off: an enabled box's row toggles the same way; floors are untouched by checks.
        source.global_switch.editor.click()
        _settle(qapp, view)
        assert source.cutoff_editor(b.id).isEnabled()
        floors = _options(page).phase_floors
        commands.clear()
        _click(qapp, scans.viewport(), _indicator(scans, _row(source, b.id)).center())
        _settle(qapp, view)
        assert len(_of(commands, 'phase_references')) == 1 and _flags(page)[b.id] is False
        assert _options(page).phase_floors == floors

        # Empty space below the rows is no instruction (the shared navigation filter): nothing changes.
        selected, commands[:] = _selected(source), []
        _click(qapp, scans.viewport(), QtCore.QPoint(8, scans.viewport().height() - 3))
        assert _selected(source) == selected and not commands

        # With the top bar's Phases off, a check still edits the include flag; back on, only B is missing.
        flags = _flags(page)
        _click(qapp, _phases_button(view))
        _settle(qapp, view)
        assert _options(page).phase_references_shown is False
        _click(qapp, scans.viewport(), _indicator(scans, _row(source, b.id)).center())
        _settle(qapp, view)
        assert _flags(page) == {**flags, b.id: True}
        _click(qapp, _phases_button(view))
        _settle(qapp, view, lambda: _marker(view, b.id) is not None)
        assert _marker(view, a.id) is not None and _marker(view, b.id) is not None

        # Close and reopen: the window reconciles the accepted flags.
        _click(qapp, scans.viewport(), _indicator(scans, _row(source, a.id)).center())
        _settle(qapp, view)
        window.close()
        qapp.processEvents()
        reopened = _window(qapp, view)
        assert _row(reopened.phase_source, a.id).checkState() != CHECKED
        assert _row(reopened.phase_source, b.id).checkState() == CHECKED and _flags(page)[a.id] is False
        # declared oracle repair (PHASE-ANALYSIS-128): Int's window deletes on close, so its list is the reopened one
        source, scans = reopened.phase_source, reopened.phase_source.browser.scans

        # A double-click on B's indicator behaves exactly as Qt's own checkable list does (B starts checked).
        changes, checked_after = _plain_double_click()
        commands.clear()
        QtTest.QTest.mouseDClick(scans.viewport(), LEFT, NONE, _indicator(scans, _row(source, b.id)).center())
        _settle(qapp, view)
        assert (len(_of(commands, 'phase_references')), _flags(page)[b.id] is True) == (changes, checked_after)


def test_p2_remove_acts_only_on_the_selected_phase_in_the_window(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    library = PhaseLibrary()
    a, b = (library.add_cif(_cif(tmp_path, name, a)) for name, a in (('phaseA', 4.1), ('phaseB', 4.25)))
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        _window(qapp, view).phase_source.set_selection(((a.id, True), (b.id, True), (LAB6, True)))
        _settle(qapp, view)
        window = _window(qapp, view)  # its first show, with three phases listed
        window.activateWindow()
        qapp.processEvents()
        source, scans = window.phase_source, window.phase_source.browser.scans
        commands = _commands(view)

        # Freshly opened: nothing is selected, so Remove is disabled and says why; pressing it does nothing.
        assert _selected(source) == [] and not source.remove_btn.isEnabled()
        _remove_follows_selection(source)
        _click(qapp, source.remove_btn)
        assert _ids(source) == [a.id, b.id, LAB6] and not commands

        # A label click on the last row (its box disabled) selects it; Remove removes exactly that phase.
        _click(qapp, scans.viewport(), _label_point(scans, _row(source, LAB6)))
        assert _selected(source) == [_row(source, LAB6).text()]
        _remove_follows_selection(source)
        _click(qapp, source.remove_btn)
        _settle(qapp, view)
        assert _ids(source) == [a.id, b.id] and LAB6 not in _flags(page)
        _remove_follows_selection(source)

        # The shared minimum off: a click inside A's enabled box edits it (one commit).
        source.global_switch.editor.click()
        _settle(qapp, view)
        editor = source.cutoff_editor(a.id)
        assert editor.isEnabled() and dict(_options(page).phase_floors) == {a.id: 5., b.id: 5.}
        _click(qapp, editor.lineEdit())
        assert QtWidgets.QApplication.focusWidget() in (editor, editor.lineEdit())
        commands.clear()
        editor.lineEdit().selectAll()
        QtTest.QTest.keyClicks(editor.lineEdit(), '2')
        QtTest.QTest.keyClick(editor.lineEdit(), QtCore.Qt.Key.Key_Return)
        _settle(qapp, view)
        assert len(_of(commands, 'phase_floors')) == 1 and dict(_options(page).phase_floors) == {a.id: 2., b.id: 5.}

        # A label click on B selects B without moving focus into B's enabled box.
        b_editor = source.cutoff_editor(b.id)
        _click(qapp, scans.viewport(), _label_point(scans, _row(source, b.id)))
        assert _selected(source) == [_row(source, b.id).text()]
        assert QtWidgets.QApplication.focusWidget() not in (b_editor, b_editor.lineEdit())
        _remove_follows_selection(source)

        # The shared minimum on again: a click on A's disabled box selects A; Remove prunes A's minimum first.
        source.global_switch.editor.click()
        _settle(qapp, view)
        editor = source.cutoff_editor(a.id)
        assert not editor.isEnabled()
        _click(qapp, scans.viewport(), editor.geometry().center())
        assert _selected(source) == [_row(source, a.id).text()]
        commands.clear()
        _click(qapp, source.remove_btn)
        _settle(qapp, view)
        paths = [command.path for command in commands if command.kind is ShellCommandKind.SET_PLOT_OPTION]
        assert paths.index(('other', 'phase_floors')) < paths.index(('other', 'phase_references')), paths
        assert _ids(source) == [b.id] and a.id not in _flags(page)
        assert dict(_options(page).phase_floors) == {b.id: 5.}
        _remove_follows_selection(source)


def test_p2_phase_fitter_remove_acts_only_on_the_selected_phase(qapp, tmp_path):
    # PHASE-ANALYSIS-128: the window Int's page mounts (the standalone constructor is retired)
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    paths = tuple(str(_cif(tmp_path, name, a)) for name, a in (('CubicA', 4.0), ('CubicB', 4.5), ('CubicC', 5.0)))
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        source = dialog.phase_source
        source.add_cif_paths(paths)
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        scans = source.browser.scans
        ids = _ids(source)
        assert source.global_switch.editor.isChecked() and not source.cutoff_editor(ids[2]).isEnabled()

        def fitted():
            """The phases the fit would use: library entries ride as phase_entries (no deferred CIFs here)."""
            return [entry.id for entry in dialog.vnext_fit_values()['phase_entries']]

        assert fitted() == ids
        # A label click on the last phase, then Remove: that phase leaves the fit and the others stay.
        _click(qapp, scans.viewport(), _label_point(scans, scans.item(2)))
        assert _selected(source) == [scans.item(2).text()]
        _click(qapp, source.remove_btn)
        assert _ids(source) == ids[:2] and fitted() == ids[:2]

        # A click on the first phase's disabled box selects it; Remove takes exactly that one.
        _click(qapp, scans.viewport(), source.cutoff_editor(ids[0]).geometry().center())
        assert _selected(source) == [scans.item(0).text()]
        _click(qapp, source.remove_btn)
        assert _ids(source) == [ids[1]] and fitted() == [ids[1]]
        _remove_follows_selection(source)

    with empty_page(qapp) as other:  # a fresh page: its own, empty reference list
        fresh = _open(qapp, other)
        fresh.phase_source.add_cif_paths(paths[:2])
        fresh.show()
        qapp.processEvents()
        assert _selected(fresh.phase_source) == [] and not fresh.phase_source.remove_btn.isEnabled()
        _remove_follows_selection(fresh.phase_source)
        _click(qapp, fresh.phase_source.remove_btn)
        assert len(_ids(fresh.phase_source)) == 2


def test_p3_value_rows_fit_the_window_at_open_after_adds_and_resizes(qapp, tmp_path):
    from xrd_tools.analysis.phase_references import PhaseLibrary
    tin = PhaseLibrary().add_lattice(TIN, a=4.24, space_group=225)
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        _window(qapp, view).phase_source.set_selection(((LAB6, True), (tin.id, True)))
        _settle(qapp, view)
        window = _window(qapp, view)  # its first show, with the reported pair listed
        source, scans = window.phase_source, window.phase_source.browser.scans
        cap = 72 * scans.fontMetrics().averageCharWidth()
        screen = window.screen().availableGeometry().width()

        # At open: every number box shows whole, beside the full names; the opening width stays bounded.
        _rows_fit(qapp, source)
        for identifier in (LAB6, tin.id):
            item = _row(source, identifier)
            room = source.cutoff_editor(identifier).geometry().left() - _indicator(scans, item).right()
            assert room >= scans.fontMetrics().horizontalAdvance(item.text()), (item.text(), room)
        assert TIN in _row(source, tin.id).toolTip()
        assert scans.sizeHint().width() <= min(cap, screen)

        # A much longer phase joins while the window is open: boxes stay whole, the name elides, nothing scrolls.
        long_name = 'Ti2AlN_mp-1234_conventional_standard_' + 'superlattice' * 8
        lattice = source.add_lattice(long_name, a=3.0, space_group=221)
        _settle(qapp, view)
        _rows_fit(qapp, source)
        assert long_name in _row(source, lattice.id).toolTip()
        assert scans.sizeHint().width() <= min(cap, screen)
        assert not scans.verticalScrollBar().isVisible(), 'three rows fit under the list maximum height'

        # Narrowed to its minimum and widened again: the boxes stay inside the visible list.
        height = window.height()
        window.resize(window.minimumSizeHint().width(), height)
        _rows_fit(qapp, source)
        window.resize(window.minimumSizeHint().width() + 260, height)
        _rows_fit(qapp, source)


def test_p3_phase_fitter_rows_fit_with_a_long_name(qapp, tmp_path):
    # PHASE-ANALYSIS-128: the window Int's page mounts (the standalone constructor is retired)
    from tests.xdart.scattering.test_phase_analysis_window import _open, empty_page
    with empty_page(qapp) as page:
        dialog = _open(qapp, page)
        source = dialog.phase_source
        source.add_lattice(TIN, a=4.24, space_group=225)
        source.add_lattice('Ti2AlN_mp-1234_conventional_standard_' + 'superlattice' * 8, a=3.0, space_group=221)
        dialog.show()
        qapp.processEvents()
        _rows_fit(qapp, source)


def test_p3_row_editor_commits_once_and_keeps_its_draft(qapp, tmp_path):
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        window = _window(qapp, view)
        source = window.phase_source
        source.add_cif_paths((str(_cif(tmp_path, 'phaseA', 4.1)),))
        _settle(qapp, view)
        (a_id,) = _ids(source)
        source.global_switch.editor.click()
        _settle(qapp, view)
        window.activateWindow()
        qapp.processEvents()
        editor = source.cutoff_editor(a_id)
        line = editor.lineEdit()
        commands = _commands(view)

        def edit(text):
            _click(qapp, line)
            line.selectAll()
            QtTest.QTest.keyClicks(line, text)

        def floor():
            return dict(_options(page).phase_floors)[a_id]

        edit('2')
        QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Return)
        _settle(qapp, view)
        assert len(_of(commands, 'phase_floors')) == 1 and floor() == 2., 'Return commits once'
        commands.clear()
        edit('3')
        # CI-REDS diagnosis (2026-10-29): on Linux CI this Tab sent no commit (run 40996196112); record where the
        # focus went, so a recurrence names its cause in the failure message.
        moves = []

        def record(old, new):
            moves.append((_focus_name(old), _focus_name(new)))

        def changed(value):  # the box's own commit, before BrowserView's same-value guard
            moves.append(('valueChanged', value))
        qapp.focusChanged.connect(record)
        editor.valueChanged.connect(changed)
        try:
            before = _focus_state(qapp, window, editor, line)
            QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Tab)
            _settle(qapp, view)
            after = _focus_state(qapp, window, editor, line)
        finally:
            qapp.focusChanged.disconnect(record)
            editor.valueChanged.disconnect(changed)
        assert len(_of(commands, 'phase_floors')) == 1 and floor() == 3., (
            f'leaving the box commits once; before={before} after={after} moves={moves} commands={len(commands)}')
        commands.clear()
        _click(qapp, line)
        QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Up)
        _settle(qapp, view)
        assert len(_of(commands, 'phase_floors')) == 1 and floor() == 3. + editor.singleStep(), 'a step commits once'
        commands.clear()
        edit('6')
        window.resize(window.width() + 120, window.height())
        qapp.processEvents()
        assert '6' in line.text() and not _of(commands, 'phase_floors'), 'a relayout keeps the typed draft'
        QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Return)
        _settle(qapp, view)
        assert len(_of(commands, 'phase_floors')) == 1 and floor() == 6.
