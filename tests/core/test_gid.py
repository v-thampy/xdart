"""Tests for xrd_tools.integrate.gid."""

from __future__ import annotations

import copy
import hashlib
import inspect
import pickle

import numpy as np
import pytest

from xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
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


def test_public_gi_integrations_default_to_compiled_histogram():
    from xrd_tools.reduction import GIMode

    assert GIMode().method == "cython"
    integrations = (
        integrate_gi_1d,
        integrate_gi_2d,
        integrate_gi_polar,
        integrate_gi_exitangles,
        integrate_gi_polar_1d,
        integrate_gi_azimuthal_1d,
        integrate_gi_exitangles_1d,
    )

    for integrate in integrations:
        assert inspect.signature(integrate).parameters["method"].default == (
            "cython"
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


def test_project_fiber_resets_engines_without_eager_global_gc(
    poni_fixture,
    monkeypatch,
):
    """Geometry changes stay exact but never serialize workers through GC."""

    from pyFAI.integrator import common as pyfai_common

    collections = []
    monkeypatch.setattr(
        pyfai_common.gc,
        "collect",
        lambda: collections.append("collected"),
    )

    fi = create_fiber_integrator(poni_fixture, incident_angle=0.15)
    assert collections == []

    class SentinelEngine:
        reset_count = 0

        def reset(self):
            self.reset_count += 1

    engine = SentinelEngine()
    fi.engines["sentinel"] = engine
    fi.reset_integrator(np.deg2rad(0.25), 0.0, 1)

    assert engine.reset_count == 1
    assert fi.engines == {}
    assert fi.incident_angle == pytest.approx(np.deg2rad(0.25))
    assert collections == []

    # The production worker provider deep-copies the owner integrator.  The
    # project reset policy and its bounded empty engine cache must survive.
    worker = copy.deepcopy(fi)
    worker_engine = SentinelEngine()
    worker.engines["worker-sentinel"] = worker_engine
    worker.reset_integrator(np.deg2rad(0.30), 0.0, 1)
    assert worker_engine.reset_count == 1
    assert worker.engines == {}
    assert collections == []

    # create_fiber_integrator is public and its historical pyFAI result was
    # pickleable.  The lazy project policy must retain that compatibility.
    restored = pickle.loads(pickle.dumps(fi))
    restored_engine = SentinelEngine()
    restored.engines["restored-sentinel"] = restored_engine
    restored.reset_integrator(np.deg2rad(0.35), 0.0, 1)
    assert restored_engine.reset_count == 1
    assert restored.engines == {}
    assert collections == []


@pytest.mark.slow
def test_variable_incidence_a_b_a_matches_fresh_fiber_integrators():
    """Skipping eager GC must not retain geometry arrays across GI angles."""

    shape = (195, 487)
    poni = PONI(
        dist=0.2,
        poni1=shape[0] * 172e-6 / 2.0,
        poni2=shape[1] * 172e-6 / 2.0,
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=1.0e-10,
        detector="Pilatus100k",
    )
    image = (
        np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
        % 97.0
    ) + 1.0

    def _integrate(fi, angle):
        return integrate_gi_2d(
            image,
            fi,
            npt_rad=32,
            npt_azim=24,
            incident_angle=angle,
        )

    reused = create_fiber_integrator(poni, incident_angle=0.15)
    first_a = _integrate(reused, 0.15)
    result_b = _integrate(reused, 0.25)
    second_a = _integrate(reused, 0.15)
    fresh_a = _integrate(
        create_fiber_integrator(poni, incident_angle=0.15),
        0.15,
    )
    fresh_b = _integrate(
        create_fiber_integrator(poni, incident_angle=0.25),
        0.25,
    )

    for actual, expected in (
        (first_a, fresh_a),
        (second_a, fresh_a),
        (result_b, fresh_b),
    ):
        np.testing.assert_allclose(actual.radial, expected.radial)
        np.testing.assert_allclose(actual.azimuthal, expected.azimuthal)
        np.testing.assert_allclose(
            actual.intensity,
            expected.intensity,
            equal_nan=True,
        )


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


def test_xdart_exit_equations_match_independent_signed_orientation_matrices():
    """Pin signs for orientation 1/default 4 plus rotated 3/6 and tilt.

    The expected q vector is derived directly from the outgoing unit ray and
    explicit sample-frame rotation matrices; it does not use a pyFAI exit map.
    """
    from xrd_tools.integrate.gid import (
        _xdart_exit_angle_horz_equation,
        _xdart_exit_angle_vert_equation,
    )

    x = np.array([-0.03, 0.01, 0.04])
    y = np.array([0.02, -0.04, 0.01])
    z = np.full(x.shape, 0.2)
    ai = np.deg2rad(0.3)
    wavelength_m = 1.0e-10
    mappings = {
        1: (x, y, 0.0),
        4: (x, -y, 0.0),
        3: (-x, -y, 0.0),
        6: (-y, x, np.deg2rad(2.0)),
    }

    for orientation, (xo, yo, tilt) in mappings.items():
        radius = np.sqrt(xo * xo + yo * yo + z * z)
        q_beam_over_k0 = z / radius - 1.0
        q_horz_over_k0 = xo / radius
        q_vert_over_k0 = yo / radius
        q_vert_after_incidence = (
            -np.sin(ai) * q_beam_over_k0
            + np.cos(ai) * q_vert_over_k0
        )
        qoop_over_k0 = (
            -np.sin(tilt) * q_horz_over_k0
            + np.cos(tilt) * q_vert_after_incidence
        )
        expected_vertical = np.arcsin(qoop_over_k0 - np.sin(ai))

        pitched_y = np.cos(ai) * yo - np.sin(ai) * z
        pitched_z = np.sin(ai) * yo + np.cos(ai) * z
        expected_horizontal = np.arctan2(
            np.cos(tilt) * xo - np.sin(tilt) * pitched_y,
            pitched_z,
        )

        actual_vertical = _xdart_exit_angle_vert_equation(
            x, y, z,
            wavelength=wavelength_m,
            incident_angle=ai,
            tilt_angle=tilt,
            sample_orientation=orientation,
        )
        actual_horizontal = _xdart_exit_angle_horz_equation(
            x, y, z,
            wavelength=wavelength_m,
            incident_angle=ai,
            tilt_angle=tilt,
            sample_orientation=orientation,
        )
        np.testing.assert_allclose(actual_vertical, expected_vertical, atol=2e-15)
        np.testing.assert_allclose(actual_horizontal, expected_horizontal, atol=2e-15)

    # These are signed coordinates, not magnitudes.
    positive_y = _xdart_exit_angle_vert_equation(
        np.array([0.0]), np.array([0.02]), np.array([0.2]),
        wavelength=wavelength_m, incident_angle=ai, tilt_angle=0.0,
        sample_orientation=1,
    )
    flipped_y = _xdart_exit_angle_vert_equation(
        np.array([0.0]), np.array([0.02]), np.array([0.2]),
        wavelength=wavelength_m, incident_angle=ai, tilt_angle=0.0,
        sample_orientation=4,
    )
    assert positive_y[0] > 0.0 > flipped_y[0]


def test_public_exit_axes_keep_xdart_physical_names_and_signs():
    shape = (195, 487)
    poni = PONI(
        dist=0.2,
        poni1=shape[0] * 172e-6 / 2.0,
        poni2=shape[1] * 172e-6 / 2.0,
        wavelength=1.0e-10,
        detector="Pilatus100k",
    )
    fi = create_fiber_integrator(poni, incident_angle=0.3, sample_orientation=4)
    image = np.ones(shape, dtype=np.float64)
    result = integrate_gi_exitangles(
        image, fi, npt_rad=48, npt_azim=40,
        incident_angle=0.3, sample_orientation=4,
    )

    assert result.unit == "exit_angle_horz_deg"
    assert result.azimuthal_unit == "exit_angle_vert_deg"
    assert result.radial[0] < 0.0 < result.radial[-1]
    assert result.azimuthal[0] < -0.3 < result.azimuthal[-1]

    with pytest.raises(ValueError, match="legacy GI exit-angle axes"):
        integrate_gi_exitangles(
            image, fi, npt_rad=16, npt_azim=12,
            gi_exit_angle_convention="legacy_xdart_2025_reflection",
        )


def test_pyfai_2025_q_space_intensity_hashes_and_nan_occupancy_are_preserved():
    """Synthetic frozen cross-version oracle captured on pyFAI 2025.3.0.

    Axis endpoints are compared numerically because the upgrade probe found
    last-bit axis drift; intensity bytes and NaN occupancy remain exact.
    """
    shape = (195, 487)
    poni = PONI(
        dist=0.2,
        poni1=shape[0] * 172e-6 / 2.0,
        poni2=shape[1] * 172e-6 / 2.0,
        wavelength=1.0e-10,
        detector="Pilatus100k",
    )
    image = (
        np.arange(np.prod(shape), dtype=np.float64).reshape(shape) % 97.0
    ) + 1.0
    fi = create_fiber_integrator(poni, incident_angle=0.3, sample_orientation=4)
    products = {
        "qoop_1d": integrate_gi_1d(
            image, fi, npt=96, npt_oop=64, method="cython",
            sample_orientation=4,
        ),
        "qip_qoop_2d": integrate_gi_2d(
            image, fi, npt_rad=48, npt_azim=40, method="cython",
            sample_orientation=4,
        ),
        "polar": integrate_gi_polar(
            image, fi, npt_rad=48, npt_azim=40, method="cython",
            sample_orientation=4,
        ),
    }
    expected = {
        "qoop_1d": (
            "b412ee1baf9f06741d39dcd74e9482988caa28399cd6cc1019cfe6db41905835",
            0,
            (-0.5140467375365425, 0.5142745493078682),
            None,
        ),
        "qip_qoop_2d": (
            "f9b9e44836f941340330fbc59ec7ed606c067d1b8ec2a8faf8c3223365a603e5",
            0,
            (-1.2652318611077744, 1.2652320151442593),
            (-0.5091499695039502, 0.5093777812752758),
        ),
        "polar": (
            "f9c0a66760d7a620f6e843c7878c769169a7051374d5661ae8a571431252c87b",
            642,
            (0.014456354043530887, 1.3733536341354342),
            (-175.52373004171199, 173.62532670475272),
        ),
    }

    for name, result in products.items():
        digest, nan_count, radial_ends, azimuthal_ends = expected[name]
        intensity = np.ascontiguousarray(result.intensity, dtype=np.float64)
        assert hashlib.sha256(intensity.tobytes()).hexdigest() == digest
        assert int(np.isnan(intensity).sum()) == nan_count
        np.testing.assert_allclose(
            [result.radial[0], result.radial[-1]], radial_ends,
            rtol=0.0, atol=5.0e-15,
        )
        if azimuthal_ends is not None:
            np.testing.assert_allclose(
                [result.azimuthal[0], result.azimuthal[-1]], azimuthal_ends,
                rtol=0.0, atol=5.0e-14,
            )


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


def test_freeze_common_axis_clamps_chigi_to_rendered_domain():
    scout = _r1d(np.linspace(-180.0, 180.0, 2000))

    key, (lo, hi) = freeze_common_axis(
        scout, gi_mode_1d="chi_gi", pad_fraction=0.02)

    assert key == "azimuth_range"
    assert lo == -180.0
    assert hi == 180.0


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
