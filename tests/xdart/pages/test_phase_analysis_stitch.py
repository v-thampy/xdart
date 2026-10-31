"""PHASE-ANALYSIS-128 frozen Stitch oracle S1-S3 (Claude, 2026-10-29).

Stitch's one Phase Analysis page tool opens its fit host's window even when fitting is impossible (that reason moves
to Fit and Batch), captures a substrate in Å⁻¹ through the host's own converted input, and seeds a Sequential Batch.
Real StitchPage, published result, host worker and fits; new APIs are reached inside test bodies.
"""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def test_s1_stitchs_one_phase_tool_opens_the_window_when_fitting_is_impossible(qapp):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.pages.test_stitch_page import _Status
    from tests.xdart.pages.test_stitch_fit_host import _dispose
    page = StitchPage(empty_host_services(_Status()))
    try:
        names = tuple(name for name, _button in page.tools._tool_buttons.items())
        assert names == ("peak_fit", "phase_analysis"), names
        button = page.tools.page_tool("phase_analysis")
        assert button.text() == "≈ Phase Analysis" and button.toolTip()
        reason = page._fit_reason("phase")
        assert reason, "no result is open, so fitting is impossible"
        page._sync_fit_tools()
        assert button.isEnabled(), "the tool opens the phase list even when a fit is impossible"
        button.click()
        qapp.processEvents()
        host = page._fit_hosts["phase"]
        dialog = host.dialog
        assert dialog.isVisible() and dialog.windowTitle() == "Phase Analysis"
        assert page.scientific.phase_list is dialog.phase_binding
        assert dialog.phase_source.isEnabled() and dialog.phase_source.add_cif_btn.isEnabled()
        assert not dialog.fit_btn.isEnabled() and reason in dialog.fit_btn.toolTip()
        assert not dialog.batch_btn.isEnabled() and reason in dialog.batch_btn.toolTip()
        assert page.tools.page_tool("peak_fit").isEnabled() is False  # the Peak Fitter's own rule is unchanged
    finally:
        _dispose(page, qapp)


def test_s2_a_two_theta_results_substrate_is_captured_in_inverse_angstrom(qapp, tmp_path):
    from xrd_tools.core.containers import convert_radial_axis
    from tests.xdart.pages.test_stitch_fit_host import _finish
    from tests.xdart.scattering.test_phase_reference_overlays import _stitch
    with _stitch(qapp, tmp_path / "stitch") as (page, view, payload):
        page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_AXIS, "2theta"))
        qapp.processEvents()
        page.tools.page_tool("phase_analysis").click()
        qapp.processEvents()
        host = page._fit_hosts["phase"]
        _finish(host, qapp)
        dialog = host.dialog
        assert dialog.fit_btn.isEnabled()
        dialog.substrate_capture_btn.click()
        _finish(host, qapp)
        template = dialog.substrate_template
        assert template is not None and template.axis_unit == "q_A^-1"
        shown = host.snapshot.axes[0]
        assert shown.unit in {"2th_deg", "2th_rad"}
        expected = convert_radial_axis(np.asarray(shown.values), shown.unit, "q_A^-1", host.snapshot.wavelength_angstrom)
        finite = np.isfinite(expected)
        np.testing.assert_allclose(template.axis, expected[finite], rtol=1e-12, atol=1e-12)
        from xdart.gui.analysis.fit_export import plain_settings
        dialog.fit_background_row.editor.setCurrentText("Substrate")
        qapp.processEvents()
        first = plain_settings(dict(host._request_settings("fit")))
        assert ("background_template", ("template", template.trace_fingerprint, template.label,
                                        template.axis_unit)) in first
        dialog.substrate_capture_btn.click()  # an equal re-capture: a new object, the same plain identity
        _finish(host, qapp)
        assert dialog.substrate_template is not template
        assert plain_settings(dict(host._request_settings("fit"))) == first
        host._request("fit", (page._fit_current("phase"),))
        _finish(host, qapp)
        assert "FIT_INPUT_CHANGED" not in dialog.status.text(), dialog.status.text()


def test_s3_a_sequential_batch_seeds_each_fit_and_the_trend_column_sits_in_the_batch_card(qapp, tmp_path):
    from tests.core.test_reference_phase_sources import _fit_inputs
    from tests.xdart.pages.test_stitch_fit_host import _finish
    from tests.xdart.scattering.test_phase_reference_overlays import _stitch
    paths, _trace = _fit_inputs(tmp_path)
    with _stitch(qapp, tmp_path / "stitch") as (page, view, payload):
        page.tools.page_tool("phase_analysis").click()
        qapp.processEvents()
        host = page._fit_hosts["phase"]
        _finish(host, qapp)
        dialog = host.dialog
        parent, cards = host.column_row.parentWidget(), set()
        while parent is not None:
            cards.add(parent)
            parent = parent.parentWidget()
        assert dialog.batch_card in cards, "the Trend column row sits in the Batch card"
        dialog.phase_source.add_cif_paths((str(paths[0]),))
        qapp.processEvents()
        dialog._maxfev_row.set_number(8)
        target = page._fit_current("phase")
        host._request("batch", (target, target))
        _finish(host, qapp)
        plain = tuple(outcome.params for outcome in host.outcomes)
        assert len(plain) == 2 and plain[0] == plain[1]
        dialog.sequential_row.editor.setChecked(True)
        qapp.processEvents()
        assert dialog.sequential_batch() is True and "sequential" not in dialog.vnext_fit_values()
        host._request("batch", (target, target))
        _finish(host, qapp)
        seeded = tuple(outcome.params for outcome in host.outcomes)
        assert seeded[0] == plain[0] and seeded[1] != seeded[0], "the second fit starts from the first's values"
