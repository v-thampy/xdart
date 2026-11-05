"""NEXT-128-WINDOW-POLISH item 14, the maintainer's live check of e83a67e2 (2026-11-04, DECISIONS item 21).

His words: "right now in batch mode the tracked parameter updates but the 1d plot doesn't as the fits progress",
"make the width of the track parameter box wider so that it is not cut off", "when the batch fit is cancelled it
clears the parameter tracking window - it shouldn't. also when cancelled it shows "running solve is draining" in the
status. why?", "there is a "No template: use the displayed trace..." message under the Background in the panel. don't
think that is required. this can go in the bottom status", "the spacing between the labels and the input boxes is
inconsistent", "the Components label is clipped on both ends", and "add a Pause/Resume button in addition to Cancel. And
replace Cancel with Stop. Also replace Fit/Batch with Fit Current/Fit All". His clickable answers: "Lock during a run
(Recommended)" and "Whole panel (Recommended)".

So: during Fit all the plot and table follow the latest fitted frame, touching no option; Stop keeps the frames fitted;
Pause holds the run between frames and Resume continues; a run's options are locked until it ends; the buttons read Fit
current, Fit all, Stop, Pause and Resume; the Track boxes show whole names; the Components toggle fits its bold checked
text; the template line shows only while a template is held; every box of an options column starts at one x."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _until(predicate, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


def _settle(seconds=0.3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)


def _frame(index):
    """Frame ``index``: one Gaussian whose centre moves with the index, named ``frame<index>``."""
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    x = np.linspace(1., 4., 301)
    y = 20 + 500 * np.exp(-.5 * ((x - (2.5 + .02 * index)) / .03) ** 2)
    return FitInputSnapshot(FitInputKey(f"frame{index}", f"/data/frame{index}.nexus", "admitted", 1),
                            (FitAxis("q", "q_A^-1", x),), y, wavelength_angstrom=1.)


def _host(count, gates=None, loaded=None, *, kind="peak", fail=()):
    """A host whose current input is frame 9 and whose batch is frames 0..count-1, each read waiting for its gate."""
    from xdart.gui.analysis.snapshot_fit_host import FitTarget, SnapshotFitHost
    gates = gates if gates is not None else [threading.Event() for _ in range(count)]
    loaded = loaded if loaded is not None else []

    def target(index):
        def load():
            loaded.append(index)
            assert gates[index].wait(20)
            if index in fail:
                raise ValueError(f"frame {index} cannot be read")
            return _frame(index)
        return FitTarget(f"frame {index}", load, x=float(index))

    state = [_frame(9)]
    host = SnapshotFitHost(kind, current=lambda: state[0], scope=lambda: state[0].key,
                           batch=lambda: tuple(target(index) for index in range(count)))
    host.test_gates = gates
    return host


def _close(host):
    for gate in host.test_gates:  # a row that failed early leaves no read waiting
        gate.set()
    if host.dialog is not None:
        host.dialog.close()
    deadline = time.monotonic() + 20
    while host.shutdown().status.value == "pending" and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
    assert host.shutdown().status.value == "clean"


def _loaded_host(qapp, count, gates=None, loaded=None, **options):
    host = _host(count, gates, loaded, **options)
    host.dialog.show()
    host.reload()
    assert _until(lambda: not host.computing)
    return host


def test_fit_all_shows_each_fitted_frame_as_it_goes_and_changes_no_option(qapp):
    from xdart.gui.analysis.fit_export import plain_settings
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)
    dialog = host.dialog
    try:
        assert dialog.fit_view.title.text() == "frame9"
        dialog.full_btn.setChecked(False)  # a typed range: reading a trace rewrites it to four digits, a frame shown
        dialog.range_lo.setText("1.000001")  # during the run must not
        dialog.range_lo.editingFinished.emit()
        before = plain_settings(dialog.vnext_fit_values())
        dialog.batch_btn.click()
        gates[0].set()
        assert _until(lambda: dialog.fit_view.title.text() == "frame0"), dialog.fit_view.title.text()
        data = dialog.fit_view.projection.upper.series[0]
        assert data.role == "data" and np.array_equal(data.y, _frame(0).trace_input().intensity)
        assert any(series.role == "fit" for series in dialog.fit_view.projection.upper.series)
        assert dialog.fit_view.projection.lower.series, "its residual is drawn"
        assert dialog.table.rowCount() > 0, "its parameters are in the table"
        gates[1].set()
        assert _until(lambda: dialog.fit_view.title.text() == "frame1"), "the next fitted frame follows unprompted"
        assert np.array_equal(dialog.fit_view.projection.upper.series[0].y, _frame(1).trace_input().intensity)
        assert dialog.range_lo.text() == "1.000001" and plain_settings(dialog.vnext_fit_values()) == before
        assert host.computing and dialog.status.text() == "Fitting all frames… 2/3", dialog.status.text()
        gates[2].set()
        assert _until(lambda: not host.computing)
        assert dialog.status.text() == "Done: 3 of 3 frames fitted", dialog.status.text()
        assert len(host.outcomes) == 3 and dialog.fit_view.title.text() == "frame2"
    finally:
        _close(host)


def test_stop_keeps_the_frames_fitted_and_says_so(qapp, tmp_path):
    gates, loaded = [threading.Event() for _ in range(4)], []
    host = _loaded_host(qapp, 4, gates, loaded)
    dialog = host.dialog
    try:
        dialog.batch_btn.click()
        assert dialog.batch_btn.text() == "Stop" and dialog.batch_btn.isEnabled()
        gates[0].set()
        assert _until(lambda: len(dialog.trend.table()[0]) == 1)
        assert _until(lambda: loaded == [0, 1])  # the worker is reading frame 1
        dialog.batch_btn.click()  # Stop
        assert dialog.status.text() == "Stopping after the frame being fitted…", dialog.status.text()
        assert not dialog.batch_btn.isEnabled() and not dialog.pause_btn.isEnabled()
        gates[1].set()  # the frame under way is finished; no further frame is read
        assert _until(lambda: not host.computing)
        assert loaded == [0, 1]
        assert len(host.outcomes) == 2 and all(outcome.ok for outcome in host.outcomes)
        assert dialog.trend.table()[1] == (0.0, 1.0), "the trend keeps the frames fitted"
        assert dialog.status.text() == "Stopped: 2 of 4 frames fitted", dialog.status.text()
        assert host.holds_batch, "a stopped run's rows are held as a run's (a single Fit then leaves them)"
        assert dialog.batch_btn.text() == "Fit all" and dialog.batch_btn.isEnabled() and dialog.pause_btn.isHidden()
        path = tmp_path / "fit.csv"
        host.export_csv(path)
        text = path.read_text(encoding="utf-8")
        assert "frame1" in text and "frame2" not in text
    finally:
        _close(host)


def test_a_stop_before_any_frame_is_fitted_keeps_the_rows_held_before_it(qapp):
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded, fail=(0,))
    dialog = host.dialog
    try:
        dialog.fit_btn.click()  # one Fit: the rows held before the run
        assert _until(lambda: not host.computing) and len(dialog.trend.table()[0]) == 1
        dialog.batch_btn.click()
        assert _until(lambda: loaded == [0])
        dialog.batch_btn.click()  # Stop while frame 0 is read; it then fails
        gates[0].set()
        assert _until(lambda: not host.computing)
        assert loaded == [0]
        assert dialog.status.text() == "Stopped before any frame was fitted", dialog.status.text()
        assert len(dialog.trend.table()[0]) == 1 and len(host.outcomes) == 1 and not host.holds_batch
    finally:
        _close(host)


def test_pause_holds_the_run_between_frames_and_resume_continues(qapp):
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)
    dialog = host.dialog
    try:
        assert dialog.pause_btn.isHidden(), "Pause shows only while a run runs"
        dialog.batch_btn.click()
        assert dialog.pause_btn.isVisible() and dialog.pause_btn.text() == "Pause" and dialog.pause_btn.toolTip()
        gates[0].set()
        assert _until(lambda: loaded == [0, 1])
        dialog.pause_btn.click()
        assert dialog.pause_btn.text() == "Resume" and dialog.pause_btn.toolTip()
        assert dialog.status.text() == "Pausing after the frame being fitted…", dialog.status.text()
        gates[1].set()
        assert _until(lambda: dialog.status.text().startswith("Paused at 2/3")), dialog.status.text()
        _settle()
        assert loaded == [0, 1] and host.computing, "a paused run reads no further frame"
        # the paused frame's row and plot follow within one flush interval (0.5 s), as a running one's do (CI
        # 41938821859 checked them 0.3 s after the pause, before the throttled flush; disclosed timing repair)
        assert _until(lambda: len(dialog.trend.table()[0]) == 2 and dialog.fit_view.title.text() == "frame1")
        dialog.pause_btn.click()  # Resume
        assert dialog.pause_btn.text() == "Pause"
        assert _until(lambda: loaded == [0, 1, 2])
        gates[2].set()
        assert _until(lambda: not host.computing)
        assert dialog.status.text() == "Done: 3 of 3 frames fitted", dialog.status.text()
        assert dialog.pause_btn.isHidden()
    finally:
        _close(host)


def test_stop_or_closing_the_window_ends_a_paused_run(qapp):
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)
    dialog = host.dialog
    try:
        dialog.batch_btn.click()
        gates[0].set()
        assert _until(lambda: loaded == [0, 1])
        dialog.pause_btn.click()
        gates[1].set()
        assert _until(lambda: dialog.status.text().startswith("Paused at 2/3")), dialog.status.text()
        dialog.batch_btn.click()  # Stop while paused
        assert _until(lambda: not host.computing, 5)
        assert dialog.status.text() == "Stopped: 2 of 3 frames fitted", dialog.status.text()
        assert loaded == [0, 1] and len(host.outcomes) == 2
    finally:
        _close(host)
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)
    dialog = host.dialog
    dialog.batch_btn.click()
    gates[0].set()
    assert _until(lambda: loaded == [0, 1])
    dialog.pause_btn.click()
    gates[1].set()
    assert _until(lambda: dialog.status.text().startswith("Paused at 2/3")), dialog.status.text()
    started = time.monotonic()
    _close(host)  # the paused worker wakes for the close: clean, never a hang
    assert time.monotonic() - started < 5 and loaded == [0, 1]


def test_a_runs_options_are_locked_until_it_ends(qapp):
    gates = [threading.Event() for _ in range(2)]
    host = _loaded_host(qapp, 2, gates)
    dialog = host.dialog
    try:
        options = (dialog.model_combo, dialog.bkg_combo, dialog.full_btn, dialog.auto_check)
        assert all(widget.isEnabled() for widget in options) and dialog.refresh_btn.isEnabled()
        assert dialog.fit_view._region.movable
        dialog.batch_btn.click()
        assert not any(widget.isEnabled() for widget in options), "the run's options are locked"
        assert not dialog.refresh_btn.isEnabled() and not dialog.fit_view._region.movable
        assert dialog.advanced_card.help_button.isEnabled(), "each card's ? still opens help"
        assert dialog.fit_view.components_toggle.isEnabled() and dialog.batch_btn.isEnabled()
        for gate in gates:
            gate.set()
        assert _until(lambda: not host.computing)
        assert all(widget.isEnabled() for widget in options) and dialog.refresh_btn.isEnabled()
        assert dialog.fit_view._region.movable
    finally:
        _close(host)


def test_phase_analysis_locks_its_options_and_advanced_during_a_run(qapp):
    host = _host(2, kind="phase")
    dialog = host.dialog
    try:
        dialog.show()
        options = (dialog.profile_combo, dialog.fit_background_row.editor, dialog.lattice_pct, dialog.cif_list)
        assert all(widget.isEnabled() for widget in options) and dialog.advanced_btn.isEnabled()
        dialog.set_batch_running(True)
        assert not any(widget.isEnabled() for widget in options) and not dialog.advanced_btn.isEnabled()
        assert not dialog.refresh_btn.isEnabled() and not dialog.fit_view._region.movable
        assert dialog.shape_card.help_button.isEnabled()
        assert dialog.batch_btn.text() == "Stop" and dialog.pause_btn.isVisible()
        dialog.set_batch_running(False)
        assert all(widget.isEnabled() for widget in options) and dialog.advanced_btn.isEnabled()
        assert dialog.fit_view._region.movable and dialog.pause_btn.isHidden()
    finally:
        _close(host)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_the_run_buttons_say_what_they_fit(qapp, kind):
    host = _host(2, kind=kind)
    dialog = host.dialog
    try:
        dialog.show()
        assert dialog.fit_btn.text() == "Fit current" and dialog.batch_btn.text() == "Fit all"
        assert dialog.pause_btn.text() == "Pause" and dialog.pause_btn.isHidden() and dialog.pause_btn.toolTip()
        assert dialog.fit_btn.toolTip() and dialog.batch_btn.toolTip()
        dialog.set_batch_running(True)
        assert dialog.batch_btn.text() == "Stop" and "Stop" in dialog.batch_btn.toolTip()
        assert "draining" not in dialog.batch_btn.toolTip()
        assert dialog.pause_btn.isVisible() and dialog.pause_btn.isEnabled()
        dialog.set_batch_running(False)
        assert dialog.batch_btn.text() == "Fit all" and dialog.pause_btn.isHidden()
    finally:
        _close(host)


def test_the_track_boxes_show_their_whole_names(qapp):
    from xdart.gui.tabs.scattering.trend_view import TrendView
    trend = TrendView(None, name="trendWidth")
    trend.resize(1000, 300)
    trend.show()  # shown empty first, as a fit window's trend is
    _settle(.05)
    trend.set_rows([("frame 0", 0., {"center_0": 2.5, "fwhm_0": .07, "voigt_gamma_0": .01, "bg_c0": 1.})])
    _settle(.05)
    try:
        for combo in (trend.family, trend.right_family):
            metrics = combo.fontMetrics()
            longest = max((combo.itemText(index) for index in range(combo.count())), key=metrics.horizontalAdvance)
            assert combo.width() > metrics.horizontalAdvance(longest), (combo.width(), longest)
    finally:
        trend.close()


def test_the_components_toggle_fits_its_bold_checked_text(qapp):
    host = _host(2)
    dialog = host.dialog
    try:
        dialog.show()
        _settle(.05)
        button = dialog.fit_view.components_toggle
        mnemonic = QtCore.Qt.TextFlag.TextShowMnemonic  # the text's size as QPushButton's size hint measures it
        normal = QtGui.QFontMetrics(button.font()).size(mnemonic, button.text()).width()
        bold_font = QtGui.QFont(button.font())
        bold_font.setBold(True)
        bold = QtGui.QFontMetrics(bold_font).size(mnemonic, button.text()).width()
        assert bold > normal
        chrome = button.sizeHint().width() - normal
        # the theme draws a checked button's text bold (dark.py's QPushButton:checked), centred: it must fit
        assert button.width() >= bold + chrome, (button.width(), bold, chrome)
    finally:
        _close(host)


@pytest.mark.parametrize("kind, idle", (("peak", "No template: use the displayed trace or choose a file"),
                                        ("phase", "No substrate: use the displayed trace or choose a file")))
def test_the_template_line_shows_only_while_a_template_is_held(qapp, kind, idle):
    host = _host(2, kind=kind)
    dialog = host.dialog
    try:
        dialog.show()
        row = dialog.background_template_row
        assert row.status.isHidden() and row.status.text() == idle, "the idle hint is not in the panel"
        dialog.capture_background_template(_frame(5).trace_input(), None, "a test trace")
        assert not row.status.isHidden() and "frame5" in row.status.text(), "a held template is named"
        row.clear()  # Clear (its button waits for a Template background)
        assert row.status.isHidden()
        assert dialog.status.text() == idle, "the window's status line says it"
    finally:
        _close(host)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_every_box_of_an_int_fit_windows_options_starts_at_one_x(qapp, kind):
    from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
    from tests.xdart.test_fit_source_gui import _window
    from xdart.gui.widgets.controls_panel import FormRow, RangeRow
    page = _page()
    try:
        dialog, _host_ = _window(page, kind)
        dialog.show()
        _settle(.1)
        column = dialog.split.controls.widget()
        starts = {}
        for row in column.findChildren(QtWidgets.QWidget):
            if not isinstance(row, (FormRow, RangeRow)) or row.label.isHidden() or not row.isVisibleTo(column):
                continue
            parent = row.parentWidget()
            while parent is not None and parent is not column and not isinstance(parent, (FormRow, RangeRow)):
                parent = parent.parentWidget()
            if parent is not column:
                continue  # a row inside another row (the Degree beside its background) keeps its own label
            editor = row.editor if isinstance(row, FormRow) else row.editors[0]
            starts[row.label.text()] = editor.mapTo(column, QtCore.QPoint(0, 0)).x()
        assert len(starts) >= 8, starts
        assert len(set(starts.values())) == 1, starts
    finally:
        _close_page(page, qapp)


def test_int_s_window_pauses_and_resumes_on_its_own_route(qapp, tmp_path, monkeypatch):
    """The window Int's page mounts routes Fit all through the page (``_fit_batch``); its Pause reaches the host."""
    from tests.core.test_int_fit_source import _result
    from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
    from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window
    from xrd_tools.analysis import display_fit_operations as operations
    path = _result(tmp_path / "sample_int1d.nexus")
    entered, release = threading.Event(), threading.Event()
    original = operations.run_displayed_peak_fit

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)

    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 0)
        _drain(qapp, host)
        monkeypatch.setattr(operations, "run_displayed_peak_fit", gated)
        dialog.batch_btn.click()
        assert entered.wait(20)
        dialog.pause_btn.click()
        release.set()
        assert _until(lambda: dialog.status.text().startswith("Paused at 1/5")), dialog.status.text()
        _settle()
        assert host.computing and len(dialog.trend.table()[0]) == 1
        dialog.pause_btn.click()  # Resume
        _drain(qapp, host)
        assert dialog.status.text() == "Done: 5 of 5 frames fitted", dialog.status.text()
        assert len(host.outcomes) == 5
    finally:
        _close_page(page, qapp)
