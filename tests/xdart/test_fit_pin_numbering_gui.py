"""NEXT-128-WINDOW-POLISH item 14, round 3 (DECISIONS item 23): the maintainer's "Yes fix it now. Also the other two old
behaviors. Fit Should also start from 1 like Int."

- A running Fit all on Int's display finishes on the file it started on and keeps its rows when Int moves to another
  result (before: the run was thrown away at its end, or dropped at once when an earlier run's rows were held); the
  rows stay after Browsing, and the next Fit current follows Int and replaces them, drawing the trace it fits.
- Save CSV during a run saves the frames fitted so far in every window (a result page's own window saved its previous
  result).
- The fit windows number frames as Int's footer does, by place from 1, in the title, the trend, the Frame row, the
  Source card and the frame popups; the stored labels stay the data."""
from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import _result
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
from tests.xdart.test_batch_rows_review8_gui import _browse
from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window

#: a result whose frames skip labels 3 and 4: Int numbers its frames 1-5 by place
GAPPED = (0, 1, 2, 5, 6)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _gate(monkeypatch):
    """The real peak solve, held at its first call until released."""
    from xrd_tools.analysis import display_fit_operations as operations
    entered, release = threading.Event(), threading.Event()
    original = operations.run_displayed_peak_fit

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)

    monkeypatch.setattr(operations, "run_displayed_peak_fit", gated)
    return entered, release


def _fitted(page, kind="peak"):
    def done():
        page._drain_executor()
        return True if page._analysis_identity is None and getattr(page, f"_{kind}_result") is not None else None
    _wait(done, timeout=30)


def _int_shows(page, label):
    def shown():
        page._drain_executor()
        page._refresh_shell()
        trace, _anchor = page._capture_analysis_trace("peak")
        return trace if trace is not None and trace.receipt.frame_label == str(label) else None
    return _wait(shown, timeout=30)


def test_a_run_finishes_on_its_file_and_keeps_its_rows_when_int_moves_on(qapp, tmp_path, monkeypatch):
    first = _result(tmp_path / "first_int1d.nexus")
    second = _result(tmp_path / "second_int1d.nexus", labels=(0, 1))
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, first)
        dialog, host = _window(page, "peak")
        entered, release = _gate(monkeypatch)
        dialog.batch_btn.click()
        assert entered.wait(30)
        _browse(page, second)  # Int moves on while the run fits its first frame
        assert host.computing, "the run goes on"
        release.set()
        _drain(qapp, host)
        assert host.holds_batch and len(host.outcomes) == 5
        assert all(key.path == str(first) for key in host._keys.values())
        assert len(dialog.trend.table()[0]) == 5
        status = dialog.status.text()
        assert status.startswith("Done: 5 of 5 frames fitted") and "Int now shows another result" in status, status
    finally:
        _close_page(page, qapp)


def test_a_run_over_held_rows_is_not_dropped_when_int_moves_on(qapp, tmp_path, monkeypatch):
    first = _result(tmp_path / "first_int1d.nexus")
    second = _result(tmp_path / "second_int1d.nexus", labels=(0, 1))
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, first)
        dialog, host = _window(page, "peak")
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert host.holds_batch
        entered, release = _gate(monkeypatch)
        dialog.batch_btn.click()  # a second run over the held rows
        assert entered.wait(30)
        _browse(page, second)
        assert host.computing, "Browsing another result does not drop the running Fit all"
        release.set()
        _drain(qapp, host)
        assert host.holds_batch and len(host.outcomes) == 5
        assert all(key.path == str(first) for key in host._keys.values())
    finally:
        _close_page(page, qapp)


def test_held_rows_survive_browsing_and_the_next_fit_current_follows_int(qapp, tmp_path, monkeypatch):
    first = _result(tmp_path / "first_int1d.nexus")
    second = _result(tmp_path / "second_int1d.nexus", labels=(0, 1))
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, first)
        dialog, host = _window(page, "peak")
        dialog.batch_btn.click()
        _drain(qapp, host)
        _browse(page, second)
        assert host.holds_batch and len(dialog.trend.table()[0]) == 5, "Browsing keeps the rows"
        assert "Int now shows another result" in dialog.status.text(), dialog.status.text()
        dialog.fit_btn.click()
        _fitted(page)
        assert not host.holds_batch and dialog.trend.table()[0] == (), "Fit current follows Int and replaces them"
        trace, _anchor = page._capture_analysis_trace("peak")
        assert Path(trace.receipt.artifact) == second
        assert np.array_equal(dialog._x, trace.axis) and np.array_equal(dialog._y, trace.intensity)
        assert dialog.fit_view.title.text().endswith("· frame 1"), dialog.fit_view.title.text()
    finally:
        _close_page(page, qapp)


def test_fit_current_on_ints_display_draws_the_frame_it_fits_numbered_as_int(qapp, tmp_path, monkeypatch):
    path = _result(tmp_path / "first_int1d.nexus")
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, path)
        dialog, host = _window(page, "peak")
        assert dialog.fit_view.title.text().endswith("· frame 1"), dialog.fit_view.title.text()
        dialog.batch_btn.click()
        _drain(qapp, host)  # the window ends on the run's last frame
        assert dialog.fit_view.title.text().endswith("· frame 5"), dialog.fit_view.title.text()
        page._shell.scientific.frame_selector.setCurrentIndex(1)  # Int's footer steps to its frame 2
        _int_shows(page, 1)
        dialog.fit_btn.click()
        _fitted(page)
        trace, _anchor = page._capture_analysis_trace("peak")
        assert np.array_equal(dialog._y, trace.intensity), "the window shows the trace it fitted"
        assert dialog.fit_view.title.text().endswith("· frame 2"), dialog.fit_view.title.text()
        assert host.holds_batch and len(dialog.trend.table()[0]) == 5, "the run's rows stay (F5)"
    finally:
        _close_page(page, qapp)


def test_save_csv_during_a_run_saves_the_frames_fitted_so_far(qapp, tmp_path, monkeypatch):
    from tests.xdart.test_fit_run_controls_gui import _close, _loaded_host, _until
    from xdart.gui.widgets import file_dialogs
    gates, loaded = [threading.Event() for _ in range(3)], []
    host = _loaded_host(qapp, 3, gates, loaded)  # a result page's own window
    dialog = host.dialog
    path = tmp_path / "during.csv"
    monkeypatch.setattr(file_dialogs, "browse", lambda *args, **kwargs: [str(path)])
    try:
        dialog.fit_btn.click()  # one Fit of frame 9: the rows held before the run
        assert _until(lambda: not host.computing) and host.outcomes
        dialog.batch_btn.click()
        gates[0].set()
        assert _until(lambda: len(dialog.trend.table()[0]) == 1)
        dialog.trend.save.click()
        text = path.read_text(encoding="utf-8")
        assert "frame9" not in text, "not the result held before the run"
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1 and abs(float(rows[0]["center_0"]) - 2.5) < 1e-3, rows
    finally:
        _close(host)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_a_nexus_sources_frames_are_numbered_as_int_numbers_them(qapp, tmp_path, kind):
    path = _result(tmp_path / "gapped_int1d.nexus", labels=GAPPED)
    page = _page()
    try:
        dialog, host = _window(page, kind)
        _choose_nexus(dialog, path, 5)
        _drain(qapp, host)
        selector = dialog.source_row.frames_row.editor.selector
        assert [selector.itemText(index) for index in range(selector.count())] == ["1", "2", "3", "4", "5"]
        assert selector.currentText() == "4" and selector.currentData() == 5, "label 5 is Int's frame 4"
        assert dialog.fit_view.title.text() == "gapped_int1d · frame 4", dialog.fit_view.title.text()
        assert "frame 4" in dialog.source_row.status.text(), dialog.source_row.status.text()
        if kind == "peak":
            dialog.batch_btn.click()
            _drain(qapp, host)
            assert dialog.trend.table()[1] == (1.0, 2.0, 3.0, 4.0, 5.0)
            assert dialog.fit_view.title.text() == "gapped_int1d · frame 5", dialog.fit_view.title.text()
    finally:
        _close_page(page, qapp)


def test_the_frame_popup_asks_ints_number_and_returns_the_stored_label(qapp, tmp_path):
    from xdart.gui.analysis.background_template_row import FitSourceRow
    path = _result(tmp_path / "gapped_int1d.nexus", labels=GAPPED)
    row = FitSourceRow()
    real = row._frame_dialog

    def answered(target, frame, mode):
        dialog, frame_row, mode_row = real(target, frame, mode)
        assert frame_row.editor.text() == "4", "label 5 is shown as Int's frame 4"
        frame_row.editor.setText("5")  # Int's frame 5 is label 6
        dialog.exec = lambda: QtWidgets.QDialog.DialogCode.Accepted
        return dialog, frame_row, mode_row

    row._frame_dialog = answered
    try:
        assert row._ask_frame(str(path), 5, "default") == (6, "default")
    finally:
        row.close()


def test_a_nexus_templates_frame_field_shows_ints_number(qapp, tmp_path):
    from xdart.gui.analysis.background_template_row import BackgroundTemplateRow
    path = _result(tmp_path / "gapped_int1d.nexus", labels=GAPPED)
    row = BackgroundTemplateRow()
    try:
        row.set_available(True, "")
        row._file_chooser = lambda: str(path)
        row._ask_frame = lambda _target, _frame, _mode: (5, "default")
        row.choose_btn.click()
        assert row.source is not None and row.source.frame == 5
        assert row.frame_row.editor.text() == "4", "label 5 is Int's frame 4"
        assert "frame 4" in row.status.text(), row.status.text()
        row.frame_row.editor.setText("5")
        row.frame_row.editor.editingFinished.emit()
        assert row.source.frame == 6, "Int's frame 5 is label 6"
    finally:
        row.close()
