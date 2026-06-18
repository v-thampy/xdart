"""Tests for xrd_tools.integrate.gid."""

from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.integrate.gid import (
    create_fiber_integrator,
    freeze_common_axes_2d,
    freeze_common_axis,
    gi_1d_output_axis_key,
    integrate_gi_1d,
    integrate_gi_2d,
    integrate_gi_azimuthal_1d,
    integrate_gi_exitangles,
    integrate_gi_exitangles_1d,
    integrate_gi_polar,
    integrate_gi_polar_1d,
)

try:
    from pyFAI.integrator.fiber import FiberIntegrator

    _HAS_FIBER = True
except ImportError:
    _HAS_FIBER = False


pytestmark = pytest.mark.skipif(
    not _HAS_FIBER,
    reason="FiberIntegrator requires pyFAI >= 2025.01",
)


def test_create_fiber_integrator(poni_fixture):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.3)

    assert "FiberIntegrator" in type(fi).__name__
    assert hasattr(fi, "_gi_incident_angle")
    np.testing.assert_allclose(fi._gi_incident_angle, np.deg2rad(0.3), rtol=1e-8, atol=1e-12)


def test_create_fiber_integrator_radians(poni_fixture):
    fi = create_fiber_integrator(
        poni_fixture,
        incident_angle=0.005,
        angle_unit="rad",
    )

    assert hasattr(fi, "_gi_incident_angle")
    np.testing.assert_allclose(fi._gi_incident_angle, 0.005, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
def test_integrate_gi_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)


@pytest.mark.slow
def test_integrate_gi_2d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_2d(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_polar(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_polar(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_exitangles(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_exitangles(synthetic_image, fi, npt_rad=200, npt_azim=100)

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (200,)
    assert result.azimuthal.shape == (100,)
    assert result.intensity.shape == (200, 100)


@pytest.mark.slow
def test_integrate_gi_1d_angle_override(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)

    _ = integrate_gi_1d(synthetic_image, fi, npt=500, incident_angle=0.5)

    np.testing.assert_allclose(fi._gi_incident_angle, np.deg2rad(0.5), rtol=1e-8, atol=1e-12)


@pytest.mark.slow
def test_integrate_gi_polar_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_polar_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)


@pytest.mark.slow
def test_integrate_gi_exitangles_1d(poni_fixture, synthetic_image):
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_exitangles_1d(synthetic_image, fi, npt=500)

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (500,)
    assert result.intensity.shape == (500,)


@pytest.mark.slow
def test_integrate_gi_azimuthal_1d(poni_fixture, synthetic_image):
    """GI azimuthal profile: I vs χ_GI (chigi_deg) over a q_total band.

    The χ_GI output bins come from ``npt`` (not ``npt_q``, which is the q
    sampling), the axis is degrees (±180), and the unit is ``chigi_deg``.
    """
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_azimuthal_1d(
        synthetic_image, fi, npt=360, npt_q=500, radial_range=(0.5, 5.0))

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (360,)        # χ_GI output bins == npt
    assert result.intensity.shape == (360,)
    assert result.unit == "chigi_deg"
    chi = result.radial[np.isfinite(result.radial)]
    assert chi.min() >= -180.0 and chi.max() <= 180.0


# ---------------------------------------------------------------------------
# Common-grid freeze primitive (pure — no pyFAI, always runs in CI)
# ---------------------------------------------------------------------------

def _r1d(radial):
    radial = np.asarray(radial, float)
    return IntegrationResult1D(
        radial=radial, intensity=np.ones(radial.shape[0]), unit="q_A^-1")


def _r2d(radial, azimuthal):
    radial = np.asarray(radial, float)
    azimuthal = np.asarray(azimuthal, float)
    return IntegrationResult2D(
        radial=radial, azimuthal=azimuthal,
        intensity=np.ones((radial.shape[0], azimuthal.shape[0])), unit="q_A^-1")


def test_gi_1d_output_axis_key_by_mode():
    assert gi_1d_output_axis_key("q_total") == "radial_range"
    assert gi_1d_output_axis_key("q_ip") == "radial_range"
    assert gi_1d_output_axis_key(None) == "radial_range"
    assert gi_1d_output_axis_key("q_oop") == "azimuth_range"
    assert gi_1d_output_axis_key("exit_angle") == "azimuth_range"
    # chi_gi's output axis is χ_GI (oop/azimuth grid), so it freezes on azimuth.
    assert gi_1d_output_axis_key("chi_gi") == "azimuth_range"


def test_freeze_common_axis_single_scout_pads():
    key, rng = freeze_common_axis(_r1d(np.linspace(0.0, 10.0, 50)),
                                  gi_mode_1d="q_total", pad_fraction=0.02)
    assert key == "radial_range"
    lo, hi = rng
    # q_total is a magnitude: the low pad (0 - 2% of span) is clamped at 0 so
    # the frozen range never requests empty negative-q bins.  The high pad is
    # unaffected.
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(10.2, abs=1e-9)


def test_nan_empty_1d_marks_zero_count_bins():
    """Empty bins (pyFAI count==0 — the GI-freeze coverage pad, or masked gaps)
    become NaN so they're not plotted/aggregated as a spurious 0/flat line;
    real bins are untouched; missing count is a safe pass-through."""
    from types import SimpleNamespace
    from xrd_tools.integrate.gid import _nan_empty_1d
    res = SimpleNamespace(intensity=np.array([0.0, 5.0, 0.0, 3.0]),
                          count=np.array([0, 10, 0, 7]))
    out = _nan_empty_1d(res)
    assert np.isnan(out[0]) and np.isnan(out[2])    # empty -> NaN
    assert out[1] == 5.0 and out[3] == 3.0          # real data kept
    # defensive: no per-bin count exposed -> unchanged
    res2 = SimpleNamespace(intensity=np.array([1.0, 2.0]), count=None)
    np.testing.assert_array_equal(_nan_empty_1d(res2), [1.0, 2.0])


def test_nan_empty_2d_marks_zero_count_bins():
    """2D analog of the 1D guarantee: empty (count==0) bins -> NaN, but a GENUINE
    zero-photon bin (count>0, value 0) is PRESERVED -- keyed on count, not value,
    so the missing-wedge dummy (-1) is masked while real zeros survive."""
    from types import SimpleNamespace
    from xrd_tools.integrate.gid import _nan_empty_2d
    intensity = np.array([[0.0, 5.0, -1.0],
                          [2.0, 0.0, -1.0]])    # the -1 column is the empty wedge
    count = np.array([[3, 2, 0],
                      [1, 4, 0]])               # only the last column has count==0
    res = SimpleNamespace(intensity=intensity, count=count)
    out = _nan_empty_2d(res.intensity, res)
    assert np.isnan(out[:, 2]).all()            # empty wedge -> NaN (no dummy)
    assert out[0, 0] == 0.0 and out[1, 1] == 0.0   # genuine zeros (count>0) kept
    assert out[0, 1] == 5.0 and out[1, 0] == 2.0
    # defensive: no shape-matching count -> unchanged
    res2 = SimpleNamespace(intensity=np.array([[1.0, 2.0]]), count=None)
    np.testing.assert_array_equal(_nan_empty_2d(res2.intensity, res2), [[1.0, 2.0]])


def test_to_result_2d_nan_fills_empty_bins_in_intensity_and_sigma():
    """_to_result_2d masks empty bins in BOTH intensity and sigma, in the
    transposed (npt_ip, npt_oop) orientation, while preserving genuine zeros."""
    from types import SimpleNamespace
    from xrd_tools.integrate.gid import _to_result_2d
    # pyFAI orientation (npt_oop, npt_ip) = (2, 3); _to_result_2d transposes -> (3, 2)
    intensity = np.array([[0.0, 5.0, -1.0],
                          [2.0, 0.0, 7.0]])
    sigma = np.array([[0.1, 0.2, -1.0],
                      [0.3, 0.4, 0.5]])
    count = np.array([[4, 2, 0],
                      [1, 3, 6]])
    res = SimpleNamespace(intensity=intensity, sigma=sigma, count=count,
                          radial=np.arange(3.0), azimuthal=np.arange(2.0),
                          inplane=None, outofplane=None, ip_unit=None, oop_unit=None)
    out = _to_result_2d(res, unit_fallback="qip_A^-1")
    assert out.intensity.shape == (3, 2)            # transposed
    assert np.isnan(out.intensity[2, 0])            # (oop=0, ip=2) empty -> NaN
    assert out.intensity[0, 0] == 0.0               # genuine zero (count 4) preserved
    assert out.intensity[1, 1] == 0.0               # genuine zero (count 3) preserved
    assert np.isnan(out.sigma[2, 0])                # sigma empty bin -> NaN too
    assert out.sigma[0, 0] == 0.1


@pytest.mark.slow
def test_gi_2d_cake_empty_bins_are_nan_not_dummy(poni_fixture, synthetic_image):
    """Regression (the reported bug): the live GI 2D cake marks empty wedge/gap
    bins NaN, not the pyFAI dummy (-1 under method='no').  So projecting the cake
    to 1D via nanmean is never dragged negative by the missing wedge -- the cause
    of the negative/depressed GI Q-projected profile."""
    fi = create_fiber_integrator(poni_fixture, incident_angle=0.2)
    result = integrate_gi_polar(synthetic_image, fi, npt_rad=200, npt_azim=100)
    cake = np.asarray(result.intensity, dtype=float)
    assert np.isnan(cake).any()                     # the empty wedge became NaN
    # the synthetic image is non-negative -> no real bin is < 0; if the dummy -1
    # had survived (the bug), the finite minimum would be ~-1.
    finite = cake[np.isfinite(cake)]
    assert finite.min() >= -1e-9
    # the cake->1D projection (nanmean over the oop axis) stays non-negative
    # (an all-NaN wedge column yields NaN -> warns; production uses the
    # warning-suppressing nanmean_slice, so silence it here):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proj = np.nanmean(cake, axis=1)
    proj = proj[np.isfinite(proj)]
    assert proj.min() >= -1e-9


def test_freeze_common_axis_qtotal_floor_at_zero():
    """Regression: q_total's symmetric pad must not push the frozen lower bound
    below 0 (negative-q bins integrate to a spurious flat dummy line — the
    'data points at negative Q' artifact).  q_ip (signed) is NOT floored."""
    # span 0.002..5.88 like the real GI data: 2% pad would give lo ~= -0.116.
    scout = _r1d(np.linspace(0.002, 5.88, 2000))
    _, (lo, hi) = freeze_common_axis(scout, gi_mode_1d="q_total",
                                     pad_fraction=0.02)
    assert lo == 0.0                     # floored, not negative
    assert hi > 5.88                     # high pad preserved
    # default (None) behaves as q_total
    _, (lo_none, _hi) = freeze_common_axis(scout, pad_fraction=0.02)
    assert lo_none == 0.0
    # q_ip is a signed projection — its negative pad is preserved
    _, (lo_ip, _h) = freeze_common_axis(_r1d(np.linspace(0.0, 5.0, 100)),
                                        gi_mode_1d="q_ip", pad_fraction=0.02)
    assert lo_ip < 0.0


def test_freeze_common_axis_union_covers_both_drifted_scouts():
    """The crux: two scouts at different incidences have DRIFTED output extents;
    the frozen range must be the padded UNION covering BOTH — and a single scout
    would clip the other (that's why the union is load-bearing)."""
    lo_scout = _r1d(np.linspace(-0.5, 5.0, 60))    # widest low end
    hi_scout = _r1d(np.linspace(0.3, 5.6, 60))     # widest high end
    key, (lo, hi) = freeze_common_axis([lo_scout, hi_scout],
                                       gi_mode_1d="q_oop", pad_fraction=0.0)
    assert key == "azimuth_range"
    # Union brackets both scouts' extents exactly (pad=0).
    assert lo == pytest.approx(-0.5)
    assert hi == pytest.approx(5.6)

    # Load-bearing: a SINGLE scout clips the other.
    _, (s1_lo, s1_hi) = freeze_common_axis(hi_scout, gi_mode_1d="q_oop",
                                           pad_fraction=0.0)
    assert s1_lo > lo_scout.radial.min()           # hi-scout range clips lo end
    _, (s0_lo, s0_hi) = freeze_common_axis(lo_scout, gi_mode_1d="q_oop",
                                           pad_fraction=0.0)
    assert s0_hi < hi_scout.radial.max()           # lo-scout range clips hi end


def test_freeze_common_axis_degenerate_returns_none():
    key, rng = freeze_common_axis(_r1d(np.full(20, 3.0)), gi_mode_1d="q_total")
    assert key == "radial_range"
    assert rng is None                              # collapsed span → unfrozen
    # NaN-only axis → also None.
    key, rng = freeze_common_axis(_r1d(np.full(20, np.nan)))
    assert rng is None


def test_freeze_common_axes_2d_union_and_keys():
    # qip_qoop → x_range/y_range; union over two drifted scouts.
    s0 = _r2d(np.linspace(-3.6, 3.5, 40), np.linspace(-0.1, 5.0, 30))
    s1 = _r2d(np.linspace(-3.4, 3.7, 40), np.linspace(0.0, 5.2, 30))
    out = freeze_common_axes_2d([s0, s1], gi_mode_2d="qip_qoop", pad_fraction=0.0)
    assert set(out) == {"x_range", "y_range"}
    assert out["x_range"] == pytest.approx((-3.6, 3.7))     # union of radials
    assert out["y_range"] == pytest.approx((-0.1, 5.2))     # union of azimuthals

    # non-qip_qoop → radial_range/azimuth_range keys.
    out2 = freeze_common_axes_2d(s0, gi_mode_2d="q_chi", pad_fraction=0.0)
    assert set(out2) == {"radial_range", "azimuth_range"}


def test_freeze_common_axes_2d_omits_degenerate_axis():
    # azimuthal collapsed → y key omitted, x key still frozen.
    s = _r2d(np.linspace(0.0, 5.0, 40), np.full(30, 2.0))
    out = freeze_common_axes_2d(s, gi_mode_2d="qip_qoop")
    assert "x_range" in out
    assert "y_range" not in out
