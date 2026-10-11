"""Int's Peak Fit and Phase Fit, pinned before their widgets move onto the shared owners (FIT-TOOLS-2026-10, phase 0,
step 1: parent-passing preservation; the design's "Frozen oracle", Codex's design review F3).

The ACTIVE path, as the native Int page drives it: the dialog's values become a ``DisplayedPeakFitPlan`` or
``DisplayedPhaseFitPlan`` on the exact displayed trace, the page's one analysis ``OperationSlot`` runs
``run_displayed_peak_fit`` / ``run_displayed_phase_fit`` on its worker (no fake on that seam), and the page adopts the
terminal only while it is still current. The page context (which frame is displayed) is supplied as the existing p37
rows supply it; it is not what moves. These rows read the dialogs only through what survives the move: their values
(``vnext_fit_values``), the controls the p37 rows already drive, the plots' data items, the table and the status.
Numbers are pinned with tolerances, not bit for bit (the fitting kernels have another owner in this objective).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

import xdart.gui.tabs.scattering.analysis_mount as mount
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.operation_values import OperationUpdate
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import AxisProjection, ScientificProjection, TraceProjection
from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitResult, DisplayedPhaseFitResult
from xrd_tools.analysis.scan_operations import AnalysisDisposition
from xrd_tools.help_topics import diagnostic_text
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

#: three Gaussian peaks on a sloped background (the peak rows' displayed pattern)
_CENTRES, _SIGMA, _HEIGHTS = (1.6, 2.4, 3.3), 0.025, (900.0, 600.0, 400.0)
_TIN = Path(os.environ.get("XDART_TEST_DATA", "/nonexistent")) / "HZO_Fitting" / "CIF Files" / "TiN.cif"


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _frame(label: int = 3) -> DisplayFrameKey:
    return DisplayFrameKey(RunIdentity(2, "run-fingerprint"), "scan-7", "/tmp/a.nxs", label, 9)


def _trace(x, y, *, unit="A^-1", label=3):
    """The exact displayed trace and its anchor, as the page captures them (``displayed_trace_input``)."""
    frame = _frame(label)
    projection = ScientificProjection(traces=(
        TraceProjection(frame, AxisProjection(np.asarray(x, dtype="<f8"), "q", unit), np.asarray(y, dtype="<f8"),
                        "current", 1700000000.25),))
    return mount.displayed_trace_input(projection, frame), SimpleNamespace(frame=frame)


def _peaks(x=None):
    x = np.linspace(1.0, 4.0, 601) if x is None else x
    y = 50.0 + 10.0 * x
    for centre, height in zip(_CENTRES, _HEIGHTS):
        y = y + height * np.exp(-0.5 * ((x - centre) / _SIGMA) ** 2)
    return x, y


def _page():
    return ScatteringWorkspace(intents=RunIntentStore(RunIntent()), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter())


def _close_page(page, qapp):
    page.close_workspace(); page.deleteLater(); qapp.processEvents()


def _show(page, monkeypatch, kind, trace, anchor):
    """The displayed frame is ``trace`` (the page context as the p37 rows supply it), and the dialog shows it."""
    monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (trace, anchor))
    monkeypatch.setattr(mount, "display_anchor_matches", lambda *args: True)
    page._open_analysis_mount(kind)
    return getattr(page, f"_{kind}_dialog")


def _settle(page) -> OperationUpdate:
    """The real worker's terminal for the fit the dialog started (the page's analysis slot, joined)."""
    identity = page._analysis_identity
    assert identity is not None, page._notice_text
    worker = page._analysis_slot._worker
    worker.join(60)
    assert not worker.is_alive()
    update = page._analysis_slot.poll(identity)
    assert type(update) is OperationUpdate and update.terminal is not None
    return update


def _upper(dialog):
    items = dialog.plot.listDataItems()
    lines = [item for item in dialog.plot.getPlotItem().items if type(item) is pg.InfiniteLine]
    return items, sorted(line.value() for line in lines)


def _table(dialog):
    table = dialog.table
    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    rows = [[table.item(row, column).text() for column in range(table.columnCount())]
            for row in range(table.rowCount())]
    return headers, rows


def _parameter(result, name):
    return result.parameter_values[result.parameter_names.index(name)]


def test_peak_auto_fits_every_displayed_peak_and_shows_its_rows_errors_and_residual(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        assert dialog.vnext_fit_values()["selection_mode"] == "auto"
        dialog.fit_btn.click()
        update = _settle(page)
        assert page._consume_analysis_update(update)
        result = page._peak_result
        assert type(result) is DisplayedPeakFitResult and result.code == "OK" and result.fit_success
        # the headless selection policy found the three peaks, fitted where they are
        assert len(result.marker_positions) == 3
        np.testing.assert_allclose(result.marker_positions, _CENTRES, atol=0.002)
        for index in range(3):
            assert _parameter(result, f"p{index}_center") == pytest.approx(_CENTRES[index], abs=0.002)
            assert _parameter(result, f"p{index}_fwhm") == pytest.approx(2.354820 * _SIGMA, rel=0.03)
        # the residual is data minus model on the fitted x (the whole pattern here)
        np.testing.assert_allclose(result.residual, y - result.fit, atol=1e-9 * float(np.max(y)))
        # the upper panel: the pattern's points, the fit, the background and a marker at each fitted centre
        items, markers = _upper(dialog)
        assert [item.name() for item in items] == ["data", "fit", "background"]
        np.testing.assert_array_equal(items[0].getData()[0], x)
        np.testing.assert_array_equal(items[1].getData()[1], result.fit)
        np.testing.assert_allclose(markers, sorted(result.marker_positions))
        residual = dialog.resid_plot.listDataItems()
        assert len(residual) == 1
        np.testing.assert_array_equal(residual[0].getData()[1], result.residual)
        # the table: one row per fitted parameter, its value and its error as they came ("None" when unknown)
        headers, rows = _table(dialog)
        assert headers == ["Parameter", "Value", "StdErr"]
        assert [row[0] for row in rows] == list(result.parameter_names)
        assert [row[1] for row in rows] == [str(value) for value in result.parameter_values]
        assert [row[2] for row in rows] == [str(error) for error in result.parameter_stderr]
        assert dialog.status.text() == result.message
        assert dialog.status.toolTip() == "\n".join(
            (result.plan_fingerprint, result.policy_fingerprint, result.result_fingerprint))
    finally:
        _close_page(page, qapp)


def test_peak_manual_count_and_range_fit_only_the_chosen_window(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        dialog.auto_check.setChecked(False)
        dialog.npeaks_spin.setValue(2)
        dialog.range_lo.setText("1.2"); dialog.range_hi.setText("2.8")
        dialog._fields_to_region()
        values = dialog.vnext_fit_values()
        assert (values["selection_mode"], values["n_peaks"], values["fit_bounds"]) == ("count", 2, (1.2, 2.8))
        dialog.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        result = page._peak_result
        assert result.code == "OK" and len(result.marker_positions) == 2
        np.testing.assert_allclose(sorted(result.marker_positions), _CENTRES[:2], atol=0.003)
        fitted = (x >= 1.2) & (x <= 2.8)
        assert result.fit.size == int(np.count_nonzero(fitted))
        items, _markers = _upper(dialog)
        fit_x = items[1].getData()[0]
        assert fit_x.min() >= 1.2 and fit_x.max() <= 2.8
        np.testing.assert_allclose(result.residual, y[fitted] - result.fit, atol=1e-9 * float(np.max(y)))

        # Advanced centres take precedence: their count is the peak count, those outside the range are dropped
        dialog.auto_check.setChecked(True)
        dialog.range_lo.setText("1.0"); dialog.range_hi.setText("4.0")
        dialog._fields_to_region()
        dialog.adv_centers.setText("1.6, 3.3, 9.0")
        values = dialog.vnext_fit_values()
        assert values["manual_centers"] == (1.6, 3.3) and values["n_peaks"] == 2
        dialog.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        np.testing.assert_allclose(sorted(page._peak_result.marker_positions), (1.6, 3.3), atol=0.003)
    finally:
        _close_page(page, qapp)


def test_peak_model_background_and_advanced_defaults_and_choices(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        assert dialog.vnext_fit_values() == {
            "selection_mode": "auto", "n_peaks": 1, "model": "pseudovoigt", "background": "linear",
            "fit_bounds": (1.0, 4.0), "manual_centers": (), "sigma_init": None, "sigma_bounds": None,
            "center_bounds_delta": None, "fraction_init": 0.5, "max_nfev": None,
        }
        models = [dialog.model_combo.itemText(index) for index in range(dialog.model_combo.count())]
        backgrounds = [dialog.bkg_combo.itemText(index) for index in range(dialog.bkg_combo.count())]
        assert models == ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Voigt"]
        assert backgrounds == ["Linear", "Constant", "None", "Chebyshev (3)"]
        chosen = []
        for index in range(len(models)):
            dialog.model_combo.setCurrentIndex(index)
            chosen.append(dialog.vnext_fit_values()["model"])
        assert chosen == ["pseudovoigt", "gaussian", "lorentzian", "voigt"]
        chosen = []
        for index in range(len(backgrounds)):
            dialog.bkg_combo.setCurrentIndex(index)
            chosen.append(dialog.vnext_fit_values()["background"])
        assert chosen == ["linear", "constant", "none", "chebyshev3"]
        dialog.adv_sigma_init.setText("0.05")
        dialog.adv_sigma_min.setText("0.01"); dialog.adv_sigma_max.setText("0.2")
        dialog.adv_center_delta.setText("0.1")
        dialog.adv_fraction.setText("0.3")
        dialog.adv_maxfev.setValue(500)
        values = dialog.vnext_fit_values()
        assert (values["sigma_init"], values["sigma_bounds"], values["center_bounds_delta"],
                values["fraction_init"], values["max_nfev"]) == (0.05, (0.01, 0.2), 0.1, 0.3, 500)
    finally:
        _close_page(page, qapp)


def test_peak_reload_keeps_the_range_full_resets_it_and_a_narrower_pattern_clamps_it(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        dialog.range_lo.setText("1.5"); dialog.range_hi.setText("3.0")
        dialog._fields_to_region()
        dialog.refresh_btn.click()  # Reload: the page captures the displayed trace again
        assert dialog.vnext_fit_values()["fit_bounds"] == (1.5, 3.0)
        narrow, _anchor = _trace(*_peaks(np.linspace(2.0, 4.0, 401)), label=4)
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (narrow, anchor))
        dialog.refresh_btn.click()
        assert dialog.vnext_fit_values()["fit_bounds"] == (2.0, 3.0)
        dialog.full_btn.click()
        assert dialog.vnext_fit_values()["fit_bounds"] == (2.0, 4.0)
    finally:
        _close_page(page, qapp)


def test_peak_result_for_another_displayed_frame_is_rejected_as_stale(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        dialog.fit_btn.click()
        update = _settle(page)
        other, _anchor = _trace(x, y * 2.0, label=4)  # the user moved to another frame while it fitted
        monkeypatch.setattr(page, "_capture_analysis_trace", lambda _kind: (other, anchor))
        assert page._consume_analysis_update(update)
        assert page._peak_result is None and page._notice_text == "P3_7_ANALYSIS_STALE"
        assert dialog.table.rowCount() == 0 and not dialog.resid_plot.listDataItems()
    finally:
        _close_page(page, qapp)


def test_closing_the_peak_dialog_mid_fit_adopts_nothing_and_a_reopened_one_fits(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "peak", trace, anchor)
        dialog.fit_btn.click()
        identity = page._analysis_identity
        assert identity is not None
        dialog.close()
        assert page._peak_dialog is None
        update = _settle(page)
        assert page._consume_analysis_update(update)
        assert page._peak_result is None and page._notice_text == "P3_7_ANALYSIS_STALE"
        page._open_analysis_mount("peak")
        reopened = page._peak_dialog
        assert reopened is not None and reopened is not dialog
        reopened.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        assert page._peak_result is not None and reopened.table.rowCount() == len(page._peak_result.parameter_names)
    finally:
        _close_page(page, qapp)


def test_held_controls_stay_held_through_a_fit(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        peak = _show(page, monkeypatch, "peak", trace, anchor)
        phase = _show(page, monkeypatch, "phase", trace, anchor)
        peak.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        for dialog, controls in ((peak, (("live_check", "peak_live"), ("batch_btn", "peak_batch"))),
                                 (phase, (("texture_combo", "phase_texture"), ("batch_btn", "phase_batch")))):
            for name, reason in controls:
                control = getattr(dialog, name)
                # a held control explains its code through the shared guidance text (c47d4a9f)
                assert not control.isEnabled() and control.toolTip() == diagnostic_text(mount.HELD_REASONS[reason])
        assert not peak.live_check.isChecked()
    finally:
        _close_page(page, qapp)


def test_phase_defaults_choices_and_values(monkeypatch, qapp):
    x, y = _peaks()
    trace, anchor = _trace(x, y)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "phase", trace, anchor)
        assert dialog.vnext_fit_values() == {
            "cif_paths": (), "phase_names": (), "expected_hashes": (), "prefit_background": "none",
            "phase_profile": "pseudovoigt", "lattice_pct": 0.05, "min_intensity": 5.0, "max_nfev": None,
        }
        assert [dialog.profile_combo.itemText(i) for i in range(dialog.profile_combo.count())] == [
            "Pseudo-Voigt", "Gaussian", "Lorentzian", "Voigt"]
        assert [dialog.bkg_combo.itemText(i) for i in range(dialog.bkg_combo.count())] == ["None", "SNIP", "Chebyshev"]
        assert [dialog.texture_combo.itemText(i) for i in range(dialog.texture_combo.count())] == [
            "None", "March-Dollase", "Free (per-peak)"]
        dialog.profile_combo.setCurrentIndex(1); dialog.bkg_combo.setCurrentIndex(1)
        dialog.lattice_pct.setText("2.5"); dialog.min_intensity.setText("1"); dialog.max_nfev.setValue(300)
        values = dialog.vnext_fit_values()
        assert (values["phase_profile"], values["prefit_background"], values["lattice_pct"],
                values["min_intensity"], values["max_nfev"]) == ("gaussian", "snip", 0.025, 1.0, 300)
    finally:
        _close_page(page, qapp)


@pytest.mark.parametrize("case", ("two_theta", "no_wavelength", "missing_cif"))
def test_phase_refusals_are_named_and_adopt_nothing(case, monkeypatch, qapp, tmp_path):
    x, y = _peaks()
    trace, anchor = _trace(x, y, unit="deg" if case == "two_theta" else "A^-1")
    monkeypatch.setattr(mount, "phase_wavelength_angstrom",
                        lambda controller, frame: None if case == "no_wavelength" else 1.0)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "phase", trace, anchor)
        cif = tmp_path / "missing.cif"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(cif), "CIF"))
        dialog._add_cif()
        assert dialog._phases == [(str(cif), "missing")]
        dialog.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        expected = {"two_theta": "PHASE_Q_UNIT_REQUIRED", "no_wavelength": "INVALID_PHASE_PLAN",
                    "missing_cif": "CIF_UNAVAILABLE"}[case]
        assert page._phase_result is None and page._notice_text == expected
        assert dialog.table.rowCount() == 0
    finally:
        _close_page(page, qapp)


@pytest.mark.skipif(not _TIN.is_file(), reason="needs XDART_TEST_DATA (HZO_Fitting/CIF Files/TiN.cif)")
def test_phase_fits_a_real_cif_and_shows_its_receipt_lattice_component_markers_and_residual(monkeypatch, qapp):
    from xrd_tools.analysis.phase import PhaseModel

    reference = PhaseModel.from_cif(_TIN)
    reference.calculate_peaks(wavelength=1.0)
    x = np.linspace(2.0, 6.0, 1601)
    y = np.full_like(x, 5.0)
    for peak in reference.peaks:
        y = y + float(peak.intensity) * np.exp(-0.5 * ((x - float(peak.q)) / 0.015) ** 2)
    trace, anchor = _trace(x, y)
    monkeypatch.setattr(mount, "phase_wavelength_angstrom", lambda controller, frame: 1.0)
    page = _page()
    try:
        dialog = _show(page, monkeypatch, "phase", trace, anchor)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(_TIN), "CIF"))
        dialog._add_cif()
        assert dialog._phases == [(str(_TIN), "TiN")]
        dialog.fit_btn.click()
        assert page._consume_analysis_update(_settle(page))
        result = page._phase_result
        assert type(result) is DisplayedPhaseFitResult and result.code == "OK"
        assert result.wavelength_angstrom == 1.0
        receipt, = result.cif_receipts
        assert receipt.sha256 == hashlib.sha256(_TIN.read_bytes()).hexdigest()
        assert receipt.sha256 in dialog.cif_list.item(0).toolTip() and dialog.cif_list.item(0).text() == "TiN"
        assert result.phase_fractions == (("TiN", pytest.approx(1.0)),)
        lattice = dict(dict(result.lattice_parameters)["TiN"])
        assert lattice["a"] == pytest.approx(reference.structure.lattice.a, rel=0.005)
        in_range = sorted(float(peak.q) for peak in reference.peaks
                          if x[0] <= float(peak.q) <= x[-1] and float(peak.intensity) >= 5.0)
        np.testing.assert_allclose(sorted(result.marker_positions), in_range)
        np.testing.assert_allclose(result.residual, y - result.fit, atol=1e-9 * float(np.max(y)))
        items, markers = _upper(dialog)
        assert [(item.name()) for item in items] == ["data", "fit", "background", "TiN"]
        np.testing.assert_array_equal(items[3].getData()[1], result.phase_components[0])
        np.testing.assert_allclose(markers, sorted(result.marker_positions))
        residual = dialog.resid_plot.listDataItems()
        assert len(residual) == 1
        np.testing.assert_array_equal(residual[0].getData()[1], result.residual)
        headers, rows = _table(dialog)
        assert headers == ["Name", "Value", "Lattice", "Kind", "StdErr"]
        assert rows[0][0] == "TiN" and rows[0][3] == "phase" and rows[0][4] == ""
        assert rows[0][2] == ", ".join(f"{key}={value}" for key, value in dict(result.lattice_parameters)["TiN"])
        assert [row[0] for row in rows[1:]] == list(result.parameter_names)
        assert [row[2] for row in rows[1:]] == [""] * len(result.parameter_names)
        assert [row[3] for row in rows[1:]] == ["parameter"] * len(result.parameter_names)
        assert [row[4] for row in rows[1:]] == [str(error) for error in result.parameter_stderr]
        assert dialog.status.text() == (result.message or result.code) + "; wavelength=1.0"
    finally:
        _close_page(page, qapp)
