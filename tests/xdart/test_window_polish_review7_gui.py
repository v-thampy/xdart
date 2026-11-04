"""R7: real reader, fit host and Int Browse route; only OS file answers are supplied."""
import numpy as np
import pytest
import threading
from pyqtgraph.Qt import QtWidgets

from tests.core.test_int_fit_source import Q, _result
from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page
from tests.xdart.test_fit_source_gui import _choose_nexus, _drain, _window


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_binding_a_template_keeps_the_batch_on_a_same_settings_refit(qapp, tmp_path):
    path = _result(tmp_path / "sample_int1d.nexus")
    template = tmp_path / "bare_q.xye"
    np.savetxt(template, np.column_stack((Q, np.full(Q.size, 10.), np.ones(Q.size))))
    page = _page()
    try:
        dialog, host = _window(page, "peak")
        _choose_nexus(dialog, path, 4)
        _drain(qapp, host)
        dialog.bkg_combo.setCurrentIndex(dialog.bkg_combo.findData("template"))
        row = dialog.background_template_row
        row._file_chooser = lambda: str(template)
        row._ask_unit = lambda _path, _hint: "q_A^-1"
        row.choose_btn.click()
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert len(host.outcomes) == 5 and all(each.ok for each in host.outcomes)
        assert row.identity
        dialog.source_row.frames_row.editor.previous.click()
        _drain(qapp, host)
        assert dialog.fit_source().frame == 3
        dialog.fit_btn.click()
        _drain(qapp, host)
        assert len(host.outcomes) == 5 and len(dialog.trend.table()[0]) == 5
        assert host.outcomes[3].ok and host.outcomes[3].result is not None
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("file_source", (False, True))
def test_a_single_fit_holds_the_frame_navigator_until_its_terminal(qapp, tmp_path, monkeypatch, file_source):
    from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
    from tests.xdart.test_fit_frame_navigator_gui import _until
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xrd_tools.analysis import display_fit_operations as operations
    path = _result(tmp_path / "sample_int1d.nexus")
    page, _store = _real_page(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = operations.run_displayed_peak_fit

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        return original(*args, **kwargs)

    try:
        request = page._context_controller.begin_browse(str(path), source_root=str(path.parent))
        outcome = _wait(page._context_controller.poll_browse, timeout=30)
        assert outcome.request is request and outcome.status is BrowseLoadStatus.READY
        page.show()
        page._refresh_shell()

        def painted():
            page._drain_executor()
            value = page._last_scientific_projection
            return value if value is not None and value.traces else None
        _wait(painted, timeout=30)
        dialog, host = _window(page, "peak")
        if file_source:
            _choose_nexus(dialog, path, 2)
            _drain(qapp, host)
        monkeypatch.setattr(operations, "run_displayed_peak_fit", gated)
        dialog.fit_btn.click()
        assert _until(qapp, entered.is_set)
        assert not dialog.source_row.frames_row.editor.isEnabled()
        assert dialog.batch_btn.text() == "Batch"  # a single Fit does not become a cancellable Batch
    finally:
        release.set()

        def finished():
            page._drain_executor()
            return page._analysis_identity is None and not page._fit_hosts.get("peak", host).computing
        assert _until(qapp, finished)
        assert dialog.source_row.frames_row.editor.isEnabled()
        _close_page(page, qapp)


def test_int_frame_step_retires_the_previous_overlay_without_losing_batch(qapp, tmp_path, monkeypatch):
    from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page as _real_page
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    path = _result(tmp_path / "sample_int1d.nexus")
    page, _store = _real_page(tmp_path, monkeypatch)
    try:
        request = page._context_controller.begin_browse(str(path), source_root=str(path.parent))
        outcome = _wait(page._context_controller.poll_browse, timeout=30)
        assert outcome.request is request and outcome.status is BrowseLoadStatus.READY
        page.resize(1200, 800)
        page.show()
        page._refresh_shell()

        def painted():
            page._drain_executor()
            projection = page._last_scientific_projection
            return projection if projection is not None and projection.traces else None
        _wait(painted, timeout=30)
        dialog, host = _window(page, "peak")
        dialog.batch_btn.click()
        _drain(qapp, host)
        assert len(host.outcomes) == 5 and host._shown_outcome is not None
        navigator = dialog.source_row.frames_row.editor
        wanted = navigator.selector.itemData(2)
        navigator.selector.setCurrentIndex(2)

        def followed():
            page._drain_executor()
            page._refresh_shell()
            trace, _anchor = page._capture_analysis_trace("peak")
            return (trace is not None and page._context_controller.navigation.current is wanted
                    and np.array_equal(dialog._y, trace.intensity))
        _wait(lambda: True if followed() else None, timeout=30)
        assert len(host.outcomes) == 5
        dialog.fit_view.components_toggle.click()
        drawn = {item.name() for item in dialog.fit_view.plot.getPlotItem().listDataItems()}
        assert "fit" not in drawn
        assert host._shown_outcome is None
        assert dialog.fit_btn.isEnabled()
    finally:
        _close_page(page, qapp)
