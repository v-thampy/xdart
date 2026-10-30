"""STRAIN-ANALYSIS-128 frozen headless oracle S1-S10 and P1-P2 (Claude, 2026-10-30).

The maintainer's Strain Analysis ("the sin2psi analysis will be implemented like the phase analysis in a tool for
strain"), headless first, held to the deferred ledger's STRAIN/A3 gates: an explicit d0 (the intercept is not a
stress-free d0), a weighted regression with covariance, reflection-specific elastic constants, the true sample-frame
psi, and a refraction check. S rows use synthetic data with known answers; P rows run his reference notebook's
pipeline on test_data/sin2psi inside the test and compare with today's sin2psi_analysis on the same map. New APIs are
reached inside test bodies, so a missing one fails its own row.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult2D

_HAS_LMFIT = importlib.util.find_spec("lmfit") is not None
_HAS_PYMATGEN = importlib.util.find_spec("pymatgen") is not None
needs_lmfit = pytest.mark.skipif(not _HAS_LMFIT, reason="needs lmfit")


def _polar_map(*, d0=1.94, slope=0.04, shear=0.0, chi=(-85.0, 85.0), nchi=341, wedge=12.0, nq=600,
               q_halfwidth=0.25, sigma_q=0.012, height=1000.0, background=20.0, seed=7, weak=()):
    """A GI fiber map: radial q_total, azimuthal chi_GI, intensity (nq, nchi), with d(psi, branch) = d0 +
    slope*sin^2(psi) + branch*shear*|sin 2psi| for psi = |chi|. NaN inside the wedge |chi| < ``wedge``, sigma =
    sqrt(I), Poisson noise; ``weak`` chi ranges are scaled down 100x."""
    rng = np.random.default_rng(seed)
    q_c = 2 * np.pi / d0
    q = np.linspace(q_c - q_halfwidth, q_c + q_halfwidth, nq)
    chis = np.linspace(chi[0], chi[1], nchi)
    intensity = np.empty((nq, nchi))
    for j, c in enumerate(chis):
        psi = np.deg2rad(abs(c))
        d = d0 + slope * np.sin(psi) ** 2 + np.sign(c) * shear * abs(np.sin(2 * psi))
        peak = height * np.exp(-0.5 * ((q - 2 * np.pi / d) / sigma_q) ** 2)
        scale = 0.01 if any(lo <= c <= hi for lo, hi in weak) else 1.0
        intensity[:, j] = scale * peak + background
    intensity = rng.poisson(intensity).astype(float)
    sigma = np.sqrt(np.maximum(intensity, 1.0))
    inside = np.abs(chis) < wedge
    intensity[:, inside] = np.nan
    sigma[:, inside] = np.nan
    return IntegrationResult2D(radial=q, azimuthal=chis, intensity=intensity, sigma=sigma,
                               unit="qtot_A^-1", azimuthal_unit="chigi_deg")


def _line(psi, d0=1.94, slope=0.04):
    return d0 + slope * np.sin(np.deg2rad(psi)) ** 2


# S1: psi is the mean |chi| of the bins a sector uses; no sector straddles chi = 0; the wedge comes from coverage ----

def test_s1_sector_psi_is_the_mean_of_the_bins_it_uses_and_no_sector_straddles_zero():
    from xrd_tools.analysis.strain import strain_sectors
    result = _polar_map(wedge=12.0)
    # 10-16 deg, half inside the wedge (|chi| < 12): its psi is the mean |chi| of the bins 12-16 deg, not 13
    (sector,) = strain_sectors(result, chi_centers=(13.0,), chi_width=6.0, min_finite_fraction=0.0)
    chis = result.azimuthal
    used = chis[(chis >= 10.0) & (chis <= 16.0) & (np.abs(chis) >= 12.0)]
    assert sector.psi == pytest.approx(np.mean(np.abs(used)), abs=1e-9)
    assert abs(sector.psi - 13.0) > 0.5
    assert sector.branch == 1 and 0.0 < sector.finite_fraction < 1.0 and sector.excluded is None
    # 6-12 deg is almost all wedge: kept in the list, left out with its reason
    (clipped,) = strain_sectors(result, chi_centers=(9.0,), chi_width=6.0)
    assert clipped.excluded
    with pytest.raises(ValueError):
        strain_sectors(result, chi_centers=(0.0,), chi_width=6.0)
    both = strain_sectors(result, chi_range=(12.0, 84.0), chi_width=6.0, branches=(1, -1))
    assert sorted({s.branch for s in both}) == [-1, 1] and len(both) == 24
    assert all(12.0 <= s.psi <= 84.0 for s in both)


# S2: the weighted fit is numpy's, with its covariance; the intercept is never a d0 ------------------------------

def test_s2_weighted_regression_matches_numpy_with_its_covariance():
    from xrd_tools.analysis.strain import StrainPoint, strain_regression
    rng = np.random.default_rng(3)
    psi = np.linspace(15, 80, 14)
    x = np.sin(np.deg2rad(psi)) ** 2
    err = np.linspace(2e-4, 1.2e-3, psi.size)
    d = 1.94 + 0.04 * x + rng.normal(0, err)
    fit = strain_regression([StrainPoint(p, v, e) for p, v, e in zip(psi, d, err)])
    coeffs, cov = np.polyfit(x, d, 1, w=1 / err, cov="unscaled")
    assert fit.weighted is True and fit.n_used == psi.size
    assert fit.slope == pytest.approx(coeffs[0], rel=1e-9)
    assert fit.intercept == pytest.approx(coeffs[1], rel=1e-9)
    np.testing.assert_allclose(np.asarray(fit.cov), cov, rtol=1e-8)  # (slope, intercept) order
    assert fit.d0 is None and fit.strain is None and fit.stress is None


# S3: a point without an error is left out by name; too few weighted points fall back to unweighted --------------

def test_s3_a_point_without_an_error_is_left_out_and_too_few_fall_back_to_unweighted():
    from xrd_tools.analysis.strain import StrainPoint, strain_regression
    points = [StrainPoint(p, _line(p), 5e-4) for p in (15.0, 30.0, 45.0, 60.0)]
    points.append(StrainPoint(75.0, 1.98, 0.0))
    fit = strain_regression(points)
    assert fit.weighted is True and fit.n_used == 4 and "75" in fit.notice
    few = [StrainPoint(15.0, 1.95, 5e-4), StrainPoint(40.0, 1.96, None), StrainPoint(70.0, 1.975, 0.0)]
    fit = strain_regression(few)
    assert fit.weighted is False and fit.n_used == 3 and "unweighted" in fit.notice.lower()


# S4: every d0 is labelled: a library line, typed, a stress-free reference, psi* (equibiaxial or uniaxial) --------

@pytest.mark.skipif(not _HAS_PYMATGEN, reason="needs pymatgen")
def test_s4_each_d0_is_labelled_and_the_intercept_is_never_called_d0(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from xrd_tools.analysis.strain import (D0Reference, ElasticConstants, StrainPoint, d0_from_library,
                                           d0_from_reference, strain_regression)
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phases"))
    cif = tmp_path / "Pt.cif"
    cif.write_text(str(CifWriter(Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.924), ["Pt"], [[0, 0, 0]]))),
                   encoding="utf-8")
    entry = PhaseLibrary().add_cif(cif)
    library = d0_from_library(entry, (2, 0, 0), wavelength_angstrom=0.97625)
    assert library.kind == "library" and library.value == pytest.approx(3.924 / 2, rel=1e-6)
    assert "200" in library.label
    elastic = ElasticConstants.from_E_nu((2, 0, 0), E=170e9, nu=0.39)
    points = [StrainPoint(p, _line(p), 4e-4) for p in np.arange(15.0, 85.0, 5.0)]
    typed = strain_regression(points, d0=D0Reference.typed(1.96), elastic=elastic)
    assert typed.d0.kind == "typed" and typed.d0.value == 1.96
    np.testing.assert_allclose(typed.strain, [(point.d - 1.96) / 1.96 for point in points], rtol=1e-12)
    reference = d0_from_reference([StrainPoint(p, 1.950, 3e-4) for p in np.arange(15.0, 85.0, 10.0)])
    assert reference.kind == "reference" and reference.value == pytest.approx(1.950, rel=1e-12)
    equi = strain_regression(points, d0="psi_star", elastic=elastic, stress_state="equibiaxial")
    uni = strain_regression(points, d0="psi_star", elastic=elastic, stress_state="uniaxial")
    assert equi.sin2psi_star == pytest.approx(2 * 0.39 / 1.39, rel=1e-9)
    assert uni.sin2psi_star == pytest.approx(0.39 / 1.39, rel=1e-9)
    assert equi.d0.kind == "psi_star"
    assert equi.d0.value == pytest.approx(equi.intercept + equi.slope * equi.sin2psi_star, rel=1e-12)


# S5: E/nu and s1/half-s2 give one stress; a known stress comes back with its covariance error ------------------

def test_s5_elastic_constants_two_ways_recover_a_known_stress_with_its_covariance_error():
    from xrd_tools.analysis.strain import D0Reference, ElasticConstants, StrainPoint, strain_regression
    E, nu, d0, stress = 170e9, 0.39, 1.962, -500e6
    a = ElasticConstants.from_E_nu((2, 0, 0), E=E, nu=nu)
    b = ElasticConstants((2, 0, 0), s1=-nu / E, half_s2=(1 + nu) / E)
    assert (a.s1, a.half_s2) == pytest.approx((b.s1, b.half_s2), rel=1e-12)
    rng = np.random.default_rng(11)
    psi = np.arange(15.0, 85.0, 5.0)
    err = np.full(psi.size, 2e-4)
    d = d0 * (1 + 2 * b.s1 * stress) + b.half_s2 * stress * d0 * np.sin(np.deg2rad(psi)) ** 2 + rng.normal(0, err)
    points = [StrainPoint(p, v, e) for p, v, e in zip(psi, d, err)]
    fit_a = strain_regression(points, d0=D0Reference.typed(d0), elastic=a)
    fit_b = strain_regression(points, d0=D0Reference.typed(d0), elastic=b)
    assert fit_a.stress == pytest.approx(fit_b.stress, rel=1e-12)
    assert abs(fit_a.stress - stress) < 4 * fit_a.stress_err
    # a typed d0 carries no error: the stress error is the slope's, through the covariance
    assert fit_a.stress_err == pytest.approx(np.sqrt(np.asarray(fit_a.cov)[0, 0]) / (d0 * b.half_s2), rel=1e-9)
    assert fit_a.reading == "compressive"


# S6: +chi and -chi kept apart recover a shear term; averaged, the shear cancels ---------------------------------

def test_s6_plus_and_minus_chi_branches_recover_a_shear_term():
    from xrd_tools.analysis.strain import StrainPoint, strain_regression
    psi = np.arange(15.0, 85.0, 5.0)
    points = [StrainPoint(p, _line(p) + branch * 0.003 * abs(np.sin(2 * np.deg2rad(p))), 1e-4, branch=branch)
              for branch in (1, -1) for p in psi]
    fit = strain_regression(points, shear=True)
    assert fit.shear == pytest.approx(0.003, rel=1e-6) and fit.slope == pytest.approx(0.04, rel=1e-6)
    plain = strain_regression(points)
    assert plain.slope == pytest.approx(0.04, rel=1e-6) and plain.shear is None


# S7: the shared fitter gains Pearson VII and split pseudo-Voigt -------------------------------------------------

@needs_lmfit
@pytest.mark.parametrize("model", ("pearson7", "split_pseudovoigt"))
def test_s7_the_shared_fitter_has_pearson_vii_and_split_pseudo_voigt(model):
    from xrd_tools.analysis.fitting import fit_peaks
    x = np.linspace(2.9, 3.6, 700)
    center = 3.24
    width = np.where(x < center, 0.010, 0.016)
    y = 500.0 / (1 + ((x - center) / width) ** 2) ** 1.6 + 5.0
    result = fit_peaks(x, y, model=model, n_peaks=1, background="constant")
    assert result.success
    assert result.peak_centers[0] == pytest.approx(center, abs=2e-3)


# S8: weak sectors are left out; each fit starts at its own peak, so tight centre bounds still hold it --------------

@needs_lmfit
def test_s8_weak_sectors_are_left_out_and_each_fit_starts_at_its_peak():
    from xrd_tools.analysis.strain import fit_strain_sectors, strain_sectors
    result = _polar_map(weak=((38.0, 44.0),))
    sectors = strain_sectors(result, chi_range=(14.0, 80.0), chi_width=6.0)
    q_c = 2 * np.pi / 1.94
    # a wide, off-centre window: its midpoint (q_c + 0.08) is no peak, and +-0.05 around it excludes every peak
    points = fit_strain_sectors(sectors, (q_c - 0.10, q_c + 0.26), model="gaussian", background="constant",
                                center_bounds_delta=0.05)
    weak = [point for point in points if 38.0 <= point.psi <= 44.0]
    assert weak and all(point.excluded for point in weak)
    kept = [point for point in points if not point.excluded]
    assert len(kept) >= 8
    for point in kept:
        assert point.d == pytest.approx(_line(point.psi), abs=2e-3)


# S9: refraction makes Pt at 0.4 deg (12.7 keV) "apparent" and leaves 2 deg alone --------------------------------

def test_s9_refraction_marks_pt_at_0_4_degrees_apparent_and_not_at_2_degrees():
    pytest.importorskip("xrayutilities")
    from xrd_tools.analysis.strain import StrainPoint, refraction_estimate, strain_regression
    points = [StrainPoint(p, _line(p, slope=0.043), 3e-4) for p in np.arange(15.0, 85.0, 5.0)]
    fit = strain_regression(points)
    low = refraction_estimate(points, fit, incident_angle_deg=0.4, energy_eV=12700.0, material="Pt")
    high = refraction_estimate(points, fit, incident_angle_deg=2.0, energy_eV=12700.0, material="Pt")
    assert low.apparent is True and high.apparent is False
    assert 0.2 < low.alpha_c_deg < 0.4
    assert abs(low.slope) > abs(high.slope)


# S10: two reflections, each with its own constants, combine into one stress -------------------------------------

def test_s10_two_reflections_with_their_own_constants_give_one_stress():
    from xrd_tools.analysis.strain import (D0Reference, ElasticConstants, StrainPoint, combined_stress,
                                           strain_regression)
    stress = 300e6
    fits = []
    for hkl, d0, E, nu, err in (((1, 1, 1), 2.266, 190e9, 0.36, 2e-4), ((2, 0, 0), 1.962, 150e9, 0.41, 3e-4)):
        constants = ElasticConstants.from_E_nu(hkl, E=E, nu=nu)
        psi = np.arange(15.0, 85.0, 5.0)
        d = d0 + constants.half_s2 * stress * d0 * np.sin(np.deg2rad(psi)) ** 2
        fits.append(strain_regression([StrainPoint(p, v, err) for p, v in zip(psi, d)],
                                      d0=D0Reference.typed(d0), elastic=constants))
    value, error = combined_stress(fits)
    weights = np.array([1 / fit.stress_err ** 2 for fit in fits])
    assert value == pytest.approx(np.sum(weights * [fit.stress for fit in fits]) / weights.sum(), rel=1e-12)
    assert value == pytest.approx(stress, rel=1e-9) and error == pytest.approx(weights.sum() ** -0.5, rel=1e-12)


# P rows: his reference notebook's pipeline on test_data/sin2psi (Pt50nm sample 1) ----------------------------------

_DATA = Path(os.environ.get("XDART_TEST_DATA", "/nonexistent"))
_PT = _DATA / "sin2psi"
# The notebook's test_data/mask.edf does not exist; this LaB6 mask matches the folder's PONI.
_MASK = _DATA / "Tiff" / "LaB6_detz190_dety72_th5_03261554_0001-mask.edf"
_RUNS = ("asDep", "FLA_8p7_2ms_1Flash", "FLA_8p7_2ms_5Flash", "FLA_8p7_2ms_11Flash")
needs_pt = pytest.mark.skipif(
    not ((_PT / "Pt50nm_sample1_asDep_0001.tif").is_file() and _MASK.is_file()),
    reason="needs test_data/sin2psi and its LaB6 mask (set XDART_TEST_DATA)")


@pytest.fixture(scope="module")
def pt_maps():
    """The four Pt maps as the notebook makes them, each at its own file's th (read-only on test_data). The mask is
    the notebook's: the file mask plus every pixel over 10000 in that image (without the threshold, saturated pixels
    leave today's path with no sin^2(psi) trend at all, R^2 0.001)."""
    from xrd_tools.integrate import create_fiber_integrator, integrate_gi_polar
    from xrd_tools.integrate.calibration import load_poni
    from xrd_tools.io import load_mask, read_image, read_txt_metadata
    poni = load_poni(_PT / "LaB6_detz190_dety72_th5_03261554_0001.poni")
    file_mask = load_mask(_MASK)
    maps = []
    for run in _RUNS:
        stem = _PT / f"Pt50nm_sample1_{run}_0001"
        meta = read_txt_metadata(stem.with_suffix(".txt"))
        data = read_image(stem.with_suffix(".tif"), mask=file_mask, threshold=10000)
        mask = load_mask(file_mask, threshold=10000, data=data)
        fiber = create_fiber_integrator(poni, incident_angle=float(meta["th"]), sample_orientation=4)
        polar = integrate_gi_polar(data, fiber, npt_rad=800, npt_azim=360, unit="q_A^-1", mask=mask,
                                   radial_range=(1, 5), azimuth_range=(-90, 90))
        maps.append((float(meta["th"]), polar))
    return poni, maps


@needs_pt
@needs_lmfit
def test_p1_pt_200_as_deposited_agrees_with_todays_path_and_reads_apparent(pt_maps):
    from xrd_tools.analysis.strain import (fit_strain_sectors, refraction_estimate, sin2psi_analysis,
                                           strain_regression, strain_sectors)
    poni, maps = pt_maps
    th, polar = maps[0]
    # the notebook's batch settings (chi 10-85 deg), which gave its Pt 200 numbers
    reference = sin2psi_analysis(polar, q_range=(3.0, 3.48), chi_width=5.0, n_sectors=15, chi_range=(10.0, 85.0),
                                 model="pseudovoigt", background="linear")
    sectors = strain_sectors(polar, chi_range=(10.0, 85.0), chi_width=5.0, n_sectors=15)
    points = fit_strain_sectors(sectors, (3.0, 3.48), model="pseudovoigt", background="linear")
    fit = strain_regression([point for point in points if not point.excluded])
    print(f"P1 Pt 200 as deposited: intercept {fit.intercept:.5f} A, slope {fit.slope:.5f} A; today's path "
          f"{reference.d0:.5f} / {reference.slope:.5f}; notebook 1.93176 / 0.04302 (cross-check only)")
    assert fit.intercept == pytest.approx(reference.d0, rel=2e-3)
    assert np.sign(fit.slope) == np.sign(reference.slope)
    assert fit.slope == pytest.approx(reference.slope, rel=0.2)
    energy_eV = 12398.419843320026 / (float(poni.wavelength) * 1e10)
    estimate = refraction_estimate(points, fit, incident_angle_deg=th, energy_eV=energy_eV, material="Pt")
    assert estimate.apparent is True


@needs_pt
@needs_lmfit
def test_p2_pt_200_batch_over_the_four_anneals_runs_frame_by_frame(pt_maps):
    from xrd_tools.analysis.plans import StrainPlan
    from xrd_tools.analysis.runner import AnalysisInput, StrainAnalyzer, run_batch
    from xrd_tools.analysis.strain import D0Reference, ElasticConstants
    poni, maps = pt_maps
    energy_eV = 12398.419843320026 / (float(poni.wavelength) * 1e10)
    plan = StrainPlan(q_range=(3.0, 3.48), chi_range=(10.0, 85.0), chi_width=5.0, n_sectors=15,
                      model="pseudovoigt", background="linear", d0=D0Reference.typed(3.924 / 2),
                      elastic=ElasticConstants.from_E_nu((2, 0, 0), E=168e9, nu=0.38), material="Pt")
    analyzer = StrainAnalyzer(plan)
    assert analyzer.unit == "frame"
    inputs = [AnalysisInput(label=run, x=np.empty(0), y=np.empty(0),
                            metadata={"result_2d": polar, "incident_angle_deg": th, "energy_eV": energy_eV})
              for run, (th, polar) in zip(_RUNS, maps)]
    outcomes = run_batch(analyzer, inputs)
    assert len(outcomes) == 4 and all(outcome.ok for outcome in outcomes), [o.message for o in outcomes]
    for outcome in outcomes:
        values = [outcome.params[key] for key in ("intercept", "slope", "stress", "stress_err")]
        assert np.all(np.isfinite(values))
        assert outcome.params["apparent"] == 1.0
    print("P2 Pt 200 intercept/slope/stress:",
          [(round(o.params["intercept"], 5), round(o.params["slope"], 5), round(o.params["stress"] / 1e6, 1))
           for o in outcomes])
