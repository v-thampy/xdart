"""R8 contract boundaries on the real Browse, fit-worker and Int analysis routes."""
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import _result
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _browse(page, path):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    request = page._context_controller.begin_browse(str(path), source_root=str(path.parent))
    outcome = _wait(page._context_controller.poll_browse, timeout=30)
    assert outcome.request is request and outcome.status is BrowseLoadStatus.READY
    page._refresh_shell()

    def painted():
        page._drain_executor()
        page._refresh_shell()
        trace, _anchor = page._capture_analysis_trace("peak")
        return trace if trace is not None and Path(trace.receipt.artifact) == path else None
    _wait(painted, timeout=30)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_int_reload_clears_the_held_batch_and_leaves_fit_usable(qapp, tmp_path, monkeypatch, kind):
    path = _result(tmp_path / "first_int1d.nexus")
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, path)
        dialog, host = _window(page, kind)
        if kind == "phase":
            from xrd_tools.analysis.phase_references import PhaseLibrary
            entry = PhaseLibrary().add_lattice("film", a=2.856, space_group="Pm-3m")
            dialog.phase_source.add_library_entry(entry.id)
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert host.holds_batch and len(host.outcomes) == 5
        dialog.refresh_btn.click()
        assert not host.holds_batch
        assert host.outcomes == () and host._keys == {} and host._trend_rows == []
        assert dialog.trend.table()[0] == () and dialog._y is not None
        assert dialog.fit_btn.isEnabled()
        dialog.fit_btn.click()

        def fitted():
            page._drain_executor()
            return True if page._analysis_identity is None and getattr(page, f"_{kind}_result") is not None else None
        _wait(fitted, timeout=30)
        assert "The trend keeps the rows from Fit all" not in dialog.status.text()  # BATCH_KEPT's text (item 14)
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_another_int_result_keeps_the_old_rows_until_a_new_batch(qapp, tmp_path, monkeypatch, kind):
    first = _result(tmp_path / "first_int1d.nexus")
    second = _result(tmp_path / "second_int1d.nexus", labels=(0, 1))
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        page.show()
        _browse(page, first)
        dialog, host = _window(page, kind)
        if kind == "phase":
            from xrd_tools.analysis.phase_references import PhaseLibrary
            entry = PhaseLibrary().add_lattice("film", a=2.856, space_group="Pm-3m")
            dialog.phase_source.add_library_entry(entry.id)
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert host.holds_batch and len(host.outcomes) == 5
        _browse(page, second)
        # the maintainer's "Yes fix it now" (DECISIONS 23): Browsing keeps the first file's rows, tied to their file,
        # until Fit current, Reload or Fit all follows Int (this row asserted R8's clearing here; disclosed)
        assert host.holds_batch and len(host.outcomes) == 5 and len(dialog.trend.table()[0]) == 5
        assert all(key.path == str(first) for key in host._keys.values())
        assert dialog.fit_btn.isEnabled() and dialog.batch_btn.isEnabled()
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert host.holds_batch and len(host.outcomes) == 2
        assert all(key.path == str(second) for key in host._keys.values())
    finally:
        _close_page(page, qapp)


def test_file_single_fit_with_changed_settings_keeps_the_whole_batch_and_csv(qapp, tmp_path):
    path = _result(tmp_path / "first_int1d.nexus")
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 4)
        _drain(qapp, host)
        dialog.batch_btn.click()
        _drain(qapp, host)
        outcomes, keys, settings, table = host.outcomes, dict(host._keys), host._last_settings, dialog.trend.table()
        csv = tmp_path / "batch.csv"
        host.export_csv(csv)
        original = csv.read_bytes()
        dialog.model_combo.setCurrentIndex(dialog.model_combo.findData("gaussian"))
        dialog.source_row.frames_row.editor.previous.click()
        _drain(qapp, host)
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert host.holds_batch and host.outcomes is outcomes and host._keys == keys
        assert host._last_settings == settings and dialog.trend.table() == table
        assert host._shown_outcome.result.model == "gaussian"
        host.export_csv(csv)
        assert csv.read_bytes() == original
        assert dialog.status.text().endswith("The trend keeps the rows from Fit all; run Fit all again to refresh them.")
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert host.holds_batch and host.outcomes is not outcomes
        assert dict(host._last_settings)["model"] == "gaussian"
        dialog.refresh_btn.click()
        _drain(qapp, host)
        assert not host.holds_batch and host.outcomes == ()
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert not host.holds_batch and len(host.outcomes) == 1
    finally:
        _close_page(page, qapp)
