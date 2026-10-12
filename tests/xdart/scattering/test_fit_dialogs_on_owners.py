"""Int's Peak Fit and Phase Fit on the shared owners (FIT-TOOLS-2026-10 phase 0, step 4; hard rule 14): their rows are
the shared Controls rows, their Advanced values the shared Advanced dialog, their plots FitView (the fit range its
own, kept over a result), their trend TrendView, their table ResultsTable and their phases the shared browser's keyed
list. What they fit and show is pinned, unedited, in ``test_fit_tools_preservation.py`` and ``test_p37_analysis_mount.py``."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.fit_view import FitView
from xdart.gui.tabs.scattering.results_table import ResultsTable
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.widgets.controls_panel import FormRow, IntegerLineEdit, RangeRow


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _trace(x, y):
    from tests.xdart.scattering.test_fit_tools_preservation import _trace as trace
    return trace(x, y)[0]


def _rows(dialog):
    return [widget for widget in dialog.findChildren(QtWidgets.QWidget) if isinstance(widget, (FormRow, RangeRow))]


def test_the_peak_fit_mounts_the_shared_owners_and_its_advanced_values_are_the_advanced_dialogs_rows(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    dialog = PeakFitDialog(vnext_submit=lambda *_args: None)
    try:
        assert isinstance(dialog.fit_view, FitView) and (dialog.plot, dialog.resid_plot) == (
            dialog.fit_view.plot, dialog.fit_view.residuals)
        assert isinstance(dialog.table, ResultsTable) and isinstance(dialog.trend, TrendView)
        assert isinstance(dialog.advanced, AdvancedDialog) and dialog.advanced.parent() is dialog
        assert isinstance(dialog.npeaks_spin, IntegerLineEdit) and dialog.model_combo.objectName() == "controlsComboBox"
        assert (dialog.range_lo, dialog.range_hi) == dialog.range_row.editors
        advanced = {dialog.adv_centers, dialog.adv_sigma_init, dialog.adv_sigma_min, dialog.adv_sigma_max,
                    dialog.adv_center_delta, dialog.adv_fraction, dialog.adv_maxfev, dialog.use_uncertainties}
        assert all(dialog.advanced.isAncestorOf(editor) for editor in advanced)
        assert len(_rows(dialog.advanced)) == 7
        assert not dialog.use_uncertainties.isChecked()  # the original unweighted default is held
        dialog.show()
        dialog.advanced_btn.click()
        assert dialog.advanced.isVisible()
    finally:
        dialog.close()


def test_the_fit_range_is_the_fit_views_dragged_into_the_rows_and_kept_over_a_result(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
    dialog = PeakFitDialog(vnext_submit=lambda *_args: None)
    try:
        x = np.linspace(1.0, 4.0, 301)
        trace = _trace(x, 10.0 + 400.0 * np.exp(-0.5 * ((x - 2.5) / 0.04) ** 2))
        dialog.set_vnext_trace(trace)
        assert dialog.fit_view.fit_range == (1.0, 4.0) and dialog.vnext_fit_values()["fit_bounds"] == (1.0, 4.0)
        dialog.fit_view._region.lines[0].setValue(2.0)  # the user drags the low edge
        assert dialog.range_lo.text() == "2" and dialog.vnext_fit_values()["fit_bounds"] == (2.0, 4.0)
        result = run_displayed_peak_fit(DisplayedPeakFitPlan(trace, **dialog.vnext_fit_values()))
        dialog.set_vnext_result(result)
        assert dialog.fit_view.fit_range == (2.0, 4.0) and dialog.fit_view._region.isVisible()
        assert dialog.fit_view._region in dialog.plot.getPlotItem().items
    finally:
        dialog.close()


def test_a_refused_peak_count_is_named_and_nothing_is_asked(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    asked = []
    dialog = PeakFitDialog(vnext_submit=lambda *args: asked.append(args))
    try:
        x = np.linspace(1.0, 4.0, 301)
        dialog.set_vnext_trace(_trace(x, np.ones_like(x)))
        dialog.auto_check.setChecked(False)
        dialog.npeaks_spin.setValue(13)
        dialog.fit_btn.click()
        assert dialog.status.text() == "Peaks: must be at most 12" and not asked
        dialog.npeaks_spin.setValue(2)
        dialog.fit_btn.click()
        assert asked and asked[-1][0] == "fit" and asked[-1][1]["n_peaks"] == 2
    finally:
        dialog.close()


def test_an_analyzer_fit_of_a_frame_is_a_row_of_the_shared_trend(qapp):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    dialog = PeakFitDialog(analysis_context=None)
    try:
        x = np.linspace(0.0, 5.0, 20)
        dialog.set_live_pattern(x, np.ones_like(x), "q (Å⁻¹)")
        payload = SimpleNamespace(peak_centers=[2.0], peak_centers_err=[0.01], peak_sigmas=[0.1],
                                  peak_amplitudes=[50.0], params=None, success=True, model_name="Gaussian",
                                  background_name="linear")
        overlay = SimpleNamespace(x=x, traces={"fit": np.ones_like(x), "residual": np.zeros_like(x)})
        for frame in (3, 1):
            dialog._draw_outcome(SimpleNamespace(overlay=overlay, result=SimpleNamespace(payload=payload),
                                                 label=str(frame), params={"center_0": 2.0 + frame}))
        assert dialog.trend.table() == (("1", "3"), (1.0, 3.0), {"center_0": [3.0, 5.0]})
        assert dialog.table.rowCount() == 1 and dialog.table.item(0, 1).text() == "2"
        dialog.reset_param_trend()
        assert dialog.trend.table() == ((), (), {})
    finally:
        dialog.close()


def test_the_phase_fit_mounts_the_shared_owners_and_its_phases_are_a_keyed_list(qapp, monkeypatch):
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    dialog = PhaseFitDialog(vnext_submit=lambda *_args: None)
    try:
        assert isinstance(dialog.cif_browser, BrowserView) and dialog.cif_list is dialog.cif_browser.scans
        assert dialog.cif_browser.profile.results and not dialog.cif_browser.profile.heading
        assert isinstance(dialog.table, ResultsTable) and isinstance(dialog.trend, TrendView)
        assert isinstance(dialog.advanced, AdvancedDialog) and dialog.advanced.isAncestorOf(dialog.max_nfev)
        # REFERENCE-PHASES-20261011 successor: the cutoff left Advanced for the Phases card (global switch, value
        # and per-row cutoffs on the shared phase source).
        assert dialog.phase_source.isAncestorOf(dialog.min_intensity) and not dialog.advanced.isAncestorOf(
            dialog.min_intensity) and isinstance(dialog.max_nfev, IntegerLineEdit)
        picks = iter(("/tmp/alpha.cif", "/tmp/beta.cif"))
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (next(picks), "CIF"))
        dialog._add_cif(); dialog._add_cif()
        assert [dialog.cif_list.item(row).text() for row in range(dialog.cif_list.count())] == ["alpha", "beta"]
        dialog.cif_browser.set_current_result("0", keep_selection=False)
        dialog.remove_cif_btn.click()
        assert dialog._phases == [("/tmp/beta.cif", "beta")]
        assert [dialog.cif_list.item(row).text() for row in range(dialog.cif_list.count())] == ["beta"]
        assert dialog.vnext_fit_values()["cif_paths"] == ("/tmp/beta.cif",)
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_fit_pressed_before_any_trace_was_shown_takes_the_displayed_trace_first(qapp, monkeypatch, kind):
    """Pre-existing (d352977f): a fit window opened before a 1-D trace was displayed, then Fit pressed once one is,
    fitted the page's trace and raised drawing the result on the window's empty pattern. Fit now takes the displayed
    trace first, as Reload does, and the result is drawn on it."""
    import xdart.gui.tabs.scattering.analysis_mount as mount
    from tests.xdart.scattering.test_fit_tools_preservation import _close_page, _page, _peaks, _settle, _trace
    trace, anchor = _trace(*_peaks())
    captured = {"trace": (None, None)}
    monkeypatch.setattr(mount, "display_anchor_matches", lambda *args: True)
    monkeypatch.setattr(mount, "phase_wavelength_angstrom", lambda controller, frame: 1.0)
    page = _page()
    try:
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: captured["trace"])
        page._open_analysis_mount(kind)  # nothing displayed yet
        dialog = getattr(page, f"_{kind}_dialog")
        assert dialog._x is None
        captured["trace"] = (trace, anchor)  # the user displays a frame, then presses Fit
        if kind == "phase":
            monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                                lambda *args, **kwargs: ("/nonexistent/alpha.cif", "CIF"))
            dialog._add_cif()
        dialog.fit_btn.click()
        assert dialog._x is not None and page._analysis_identity is not None
        assert page._consume_analysis_update(_settle(page))
        if kind == "peak":
            assert page._peak_result is not None and dialog.table.rowCount() == len(page._peak_result.parameter_names)
        else:  # the CIF is missing: refused by name, nothing raised
            assert page._phase_result is None and page._notice_text == "CIF_UNAVAILABLE"
    finally:
        _close_page(page, qapp)
