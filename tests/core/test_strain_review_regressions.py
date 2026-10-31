"""Concrete stage-1 review defects, through the real strain/record/phase owners."""
import importlib.util
import numpy as np
import pytest

from tests.core.test_strain_analysis_128 import _polar_map, _receipt
from xrd_tools.analysis.plans import StrainPlan, run_strain

pytestmark = pytest.mark.skipif(importlib.util.find_spec("lmfit") is None, reason="needs lmfit")


def _plan(**kwargs):
    return StrainPlan(q_range=(3.16, 3.28), chi_centers=(20., 40., 60., 80.),
                      chi_width=4., model="gaussian", background="constant", **kwargs)


def test_peak_window_coverage_ignores_unrelated_empty_map_rows():
    polar = _polar_map()
    outside = (polar.radial < 3.16) | (polar.radial > 3.28)
    polar.intensity[outside] = np.nan
    polar.sigma[outside] = np.nan
    result = run_strain(_plan(), polar)
    assert result.payload.fit.n_used == 4
    assert all(s.finite_fraction == 1. and s.excluded is None for s in result.payload.sectors)


def test_peak_window_clipping_controls_admission_and_reporting_angle():
    polar = _polar_map()
    rows = (polar.radial >= 3.16) & (polar.radial <= 3.28)
    cols = (polar.azimuthal >= 18) & (polar.azimuthal < 22)
    polar.intensity[np.ix_(rows, cols)] = np.nan
    polar.sigma[np.ix_(rows, cols)] = np.nan
    result = run_strain(_plan(), polar)
    clipped = result.payload.sectors[0]
    assert clipped.excluded and clipped.finite_fraction < 0.5
    assert clipped.psi == 22.
    assert result.payload.fit.n_used == 3


def test_library_d0_never_substitutes_an_inequivalent_plane(tmp_path, monkeypatch):
    pytest.importorskip("pymatgen")
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.strain import d0_from_library
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phases"))
    entry = PhaseLibrary().add_lattice("unequal axes", a=1., b=2., c=4., space_group=1)
    assert d0_from_library(entry, (0, 0, 1), wavelength_angstrom=5.).value == pytest.approx(4.)
    with pytest.raises(ValueError, match="no 100 line"):
        d0_from_library(entry, (1, 0, 0), wavelength_angstrom=5.)
    cubic = PhaseLibrary().add_lattice("cubic", a=4., space_group=221)
    assert d0_from_library(cubic, (0, 2, 0), wavelength_angstrom=1.).value == pytest.approx(2.)


@pytest.mark.parametrize("missing", ["incident_angle_deg", "energy_eV"])
def test_missing_refraction_context_is_apparent_with_a_reason(tmp_path, missing):
    from xrd_tools.core.frame_view import FrameView
    from xrd_tools.analysis.runner import StrainAnalyzer, strain_input_from_view
    from xrd_tools.analysis.strain import D0Reference, ElasticConstants
    view = FrameView.from_results(label=1, result_2d=_polar_map(), incident_angle=0.4)
    inp = strain_input_from_view(view, receipt=_receipt(tmp_path / "scan.nexus", 1), wavelength_m=0.97625e-10)
    # Public headless input can omit context; its result must never imply a passed diagnostic.
    inp.metadata.pop(missing)
    outcome = StrainAnalyzer(_plan(d0=D0Reference.typed(1.962),
        elastic=ElasticConstants.from_E_nu((2, 0, 0), E=170e9, nu=.39))).analyze(inp)
    assert outcome.ok and np.isfinite(outcome.params["stress"])
    assert outcome.params.get("apparent") == 1.
    assert "refraction unknown" in outcome.message
    assert ("incidence" if missing == "incident_angle_deg" else "energy") in outcome.message


def test_explicit_signed_centres_produce_both_descriptive_branch_fits():
    plan = StrainPlan(q_range=(3.16, 3.28), chi_centers=(-80., -60., -40., -20., 20., 40., 60., 80.),
                      chi_width=4., model="gaussian", background="constant")
    result = run_strain(plan, _polar_map())
    assert set(result.payload.branch_fits) == {-1, 1}
    assert all(fit.stress is None for fit in result.payload.branch_fits.values())


def test_compact_batch_keeps_d0_and_its_uncertainty(tmp_path):
    from xrd_tools.core.frame_view import FrameView
    from xrd_tools.analysis.runner import StrainAnalyzer, run_batch, strain_input_from_view
    from xrd_tools.analysis.strain import D0Reference
    view = FrameView.from_results(label=1, result_2d=_polar_map(), incident_angle=.4)
    inp = strain_input_from_view(view, receipt=_receipt(tmp_path / "scan.nexus", 1), wavelength_m=.97625e-10)
    outcomes = run_batch(StrainAnalyzer(_plan(d0=D0Reference.typed(1.962, stderr=.002))), [inp, inp], compact=True)
    assert all(o.ok for o in outcomes)
    assert outcomes[0].result is None
    assert outcomes[0].params["d0"] == 1.962
    assert outcomes[0].params["d0_err"] == .002


def test_excess_scatter_is_exposed_without_silently_changing_formal_covariance():
    from xrd_tools.analysis.strain import StrainPoint, strain_regression
    psi = np.array([20., 35., 50., 65., 80.])
    x = np.sin(np.deg2rad(psi))**2
    d = 1.94 + .04*x + np.array([.003, -.002, .003, -.002, .003])
    err = np.full(5, 1e-4)
    fit = strain_regression([StrainPoint(p, v, e) for p, v, e in zip(psi, d, err)])
    coefficients, covariance = np.polyfit(x, d, 1, w=1/err, cov="unscaled")
    reduced = sum(((d - np.polyval(coefficients, x))/err)**2) / 3
    assert fit.reduced_chi_squared == pytest.approx(reduced)
    np.testing.assert_allclose(fit.cov, covariance, rtol=1e-8)
    assert "excess scatter" in fit.notice and "conditional" in fit.notice
