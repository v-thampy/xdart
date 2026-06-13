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
