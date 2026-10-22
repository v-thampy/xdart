"""The displayed-fit operations as analyzers (FIT-TOOLS-2026-10, Codex's design review F3: "make the accepted
displayed-fit operations the common single-input policy, and adapt them to the analyzer/batch interface"): a host's
Batch fits each input with Int's selection policy, limits and refusals. The older ``runner`` analyzers keep their own
callers (``test_analysis_runner.py``)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.runner import (
    AnalysisInput, DisplayedPeakFitAnalyzer, DisplayedPhaseFitAnalyzer, run_batch,
)
from xrd_tools.analysis.scan_operations import AnalysisDisposition

_TIN = Path(os.environ.get("XDART_TEST_DATA", "/nonexistent")) / "HZO_Fitting" / "CIF Files" / "TiN.cif"


def _snapshot(identity, centre, *, wavelength=1.0, flat=False, noise=0.0):
    q = np.linspace(1.0, 4.0, 301)
    y = 20.0 + (0.0 if flat else 500.0) * np.exp(-0.5 * ((q - centre) / 0.03) ** 2)
    y = y + noise * np.random.default_rng(len(identity)).standard_normal(q.size)
    return FitInputSnapshot(FitInputKey(identity, f"/data/{identity}.nexus", f"fp-{identity}", 1),
                            (FitAxis("q", "A^-1", q),), y, wavelength_angstrom=wavelength)


def test_a_batch_fits_each_snapshot_with_the_displayed_policy_and_projects_the_trends_families():
    analyzer = DisplayedPeakFitAnalyzer({"model": "gaussian", "background": "constant"})
    snapshots = [_snapshot(f"scan_{index}", centre, noise=2.0) for index, centre in enumerate((2.40, 2.45, 2.50))]
    outcomes = run_batch(analyzer, snapshots)
    assert [outcome.label for outcome in outcomes] == ["scan_0", "scan_1", "scan_2"]
    assert all(outcome.ok for outcome in outcomes)
    np.testing.assert_allclose([outcome.params["center_0"] for outcome in outcomes], (2.40, 2.45, 2.50), atol=1e-3)
    for outcome in outcomes:
        assert set(outcome.params) == {"center_0", "center_err_0", "fwhm_0", "amplitude_0"}
        assert outcome.params["fwhm_0"] == pytest.approx(2.354820 * 0.03, rel=0.02)
        overlay = outcome.overlay
        assert overlay.x.shape == overlay.traces["fit"].shape == overlay.traces["residual"].shape
        assert overlay.markers == pytest.approx([outcome.params["center_0"]])
    # the same numbers as the single-input policy on the same snapshot (a notebook's or the GUI's single fit)
    single = run_displayed_peak_fit(DisplayedPeakFitPlan(snapshots[1].trace_input(), model="gaussian",
                                                         background="constant"))
    assert outcomes[1].result.parameter_values == single.parameter_values
    assert outcomes[1].result.result_fingerprint == single.result_fingerprint


def test_a_refused_input_is_an_outcome_that_is_not_ok_and_the_batch_goes_on():
    flat = _snapshot("flat", 2.5, flat=True)
    peaked = AnalysisInput("notebook", np.linspace(1.0, 4.0, 301),
                           20.0 + 500.0 * np.exp(-0.5 * ((np.linspace(1.0, 4.0, 301) - 3.0) / 0.03) ** 2),
                           x_unit="A^-1")
    outcomes = run_batch(DisplayedPeakFitAnalyzer(), [flat, peaked])
    assert [outcome.label for outcome in outcomes] == ["flat", "notebook"]
    assert not outcomes[0].ok and outcomes[0].message.startswith("PEAKS_NOT_FOUND")
    assert outcomes[0].result.disposition is AnalysisDisposition.REFUSED and not outcomes[0].params
    assert outcomes[1].ok and outcomes[1].params["center_0"] == pytest.approx(3.0, abs=1e-3)
    # an error the fit did not estimate is unavailable, never a zero
    centre = outcomes[1].result.parameter_names.index("p0_center")
    assert ("center_err_0" in outcomes[1].params) is (outcomes[1].result.parameter_stderr[centre] is not None)


def test_a_phase_fits_wavelength_is_the_inputs_own_never_a_default():
    analyzer = DisplayedPhaseFitAnalyzer({"cif_paths": ("/missing/alpha.cif",), "expected_hashes": (None,),
                                          "phase_names": ("alpha",)}, wavelength_angstrom=1.0)
    unrecorded, = run_batch(analyzer, [_snapshot("scan_1", 2.5, wavelength=None)])
    assert not unrecorded.ok and unrecorded.message == "INVALID_PHASE_PLAN; invalid wavelength"
    two_theta, = run_batch(analyzer, [AnalysisInput("pattern", np.linspace(10, 40, 301), np.ones(301),
                                                    x_unit="deg")])
    assert not two_theta.ok and two_theta.message.startswith("PHASE_Q_UNIT_REQUIRED")


@pytest.mark.skipif(not _TIN.is_file(), reason="needs XDART_TEST_DATA (HZO_Fitting/CIF Files/TiN.cif)")
def test_a_snapshot_is_phase_fitted_at_its_recorded_wavelength():
    from xrd_tools.analysis.phase import PhaseModel
    reference = PhaseModel.from_cif(_TIN)
    reference.calculate_peaks(wavelength=1.0)
    q = np.linspace(2.0, 6.0, 1601)
    y = np.full_like(q, 5.0)
    for peak in reference.peaks:
        y = y + float(peak.intensity) * np.exp(-0.5 * ((q - float(peak.q)) / 0.015) ** 2)
    snapshot = FitInputSnapshot(FitInputKey("tin", "/data/tin.nexus", "fp", 1), (FitAxis("q", "A^-1", q),), y,
                                wavelength_angstrom=1.0)
    analyzer = DisplayedPhaseFitAnalyzer({"cif_paths": (_TIN,), "expected_hashes": (None,), "phase_names": ("TiN",)})
    outcome, = run_batch(analyzer, [snapshot])
    assert outcome.result.code == "OK" and outcome.result.wavelength_angstrom == 1.0
    assert outcome.params["frac_TiN"] == pytest.approx(1.0)
    assert outcome.params["TiN_a"] == pytest.approx(reference.structure.lattice.a, rel=0.005)
    assert outcome.overlay.traces["fit"].shape == q.shape
