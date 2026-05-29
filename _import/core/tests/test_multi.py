"""Tests for ssrl_xrd_tools.integrate.multi."""

from __future__ import annotations

import numpy as np
import pytest
from pyFAI.detectors import Detector

from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.integrate.multi import (
    create_multigeometry_integrators,
    stitch_1d,
    stitch_2d,
)


def _small_base_poni(poni_fixture: PONI) -> PONI:
    """Return a copy of poni_fixture with detector unset for small test images."""
    return PONI(
        dist=poni_fixture.dist,
        poni1=poni_fixture.poni1,
        poni2=poni_fixture.poni2,
        rot1=poni_fixture.rot1,
        rot2=poni_fixture.rot2,
        rot3=poni_fixture.rot3,
        wavelength=poni_fixture.wavelength,
        detector="",
    )


def _make_small_integrators(poni_fixture: PONI, rot1_angles, rot2_angles=None):
    """
    Build per-angle integrators and attach a generic 100x100 detector.

    MultiGeometry requires the detector shape to match the image shape used in
    each test.  We therefore override detectors explicitly on the returned AIs.
    """
    base = _small_base_poni(poni_fixture)
    integrators = create_multigeometry_integrators(base, rot1_angles, rot2_angles)

    det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(100, 100))
    for ai in integrators:
        ai.detector = det

    return integrators


def _synthetic_images(n=3, shape=(100, 100), seed=123):
    """Small synthetic detector images with a broad Gaussian ring + noise."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((y - ny / 2.0) ** 2 + (x - nx / 2.0) ** 2)
    base = 500.0 * np.exp(-((r - 30.0) / 8.0) ** 2)
    return [base + rng.poisson(5, size=shape) for _ in range(n)]


# ---------------------------------------------------------------------------
# create_multigeometry_integrators
# ---------------------------------------------------------------------------

def test_create_integrators_count(poni_fixture):
    rot1_angles = [0.0, 5.0, 10.0]
    integrators = create_multigeometry_integrators(poni_fixture, rot1_angles)
    assert len(integrators) == 3


def test_create_integrators_rot1_offsets(poni_fixture):
    rot1_angles = [0.0, 10.0]
    integrators = create_multigeometry_integrators(poni_fixture, rot1_angles)

    delta_rot1 = integrators[1].rot1 - integrators[0].rot1
    np.testing.assert_allclose(delta_rot1, np.deg2rad(10.0), rtol=1e-8, atol=1e-12)


def test_create_integrators_rot2(poni_fixture):
    rot1_angles = [0.0, 5.0]
    rot2_angles = [0.0, 3.0]
    integrators = create_multigeometry_integrators(poni_fixture, rot1_angles, rot2_angles)

    delta_rot2 = integrators[1].rot2 - integrators[0].rot2
    np.testing.assert_allclose(delta_rot2, np.deg2rad(3.0), rtol=1e-8, atol=1e-12)


def test_create_integrators_mismatched_lengths(poni_fixture):
    rot1_angles = [0.0, 1.0, 2.0]
    rot2_angles = [0.0, 1.0]

    with pytest.raises(ValueError, match="rot1_angles length"):
        create_multigeometry_integrators(poni_fixture, rot1_angles, rot2_angles)


# ---------------------------------------------------------------------------
# stitch_1d / stitch_2d
# ---------------------------------------------------------------------------

def test_stitch_1d_runs(poni_fixture):
    integrators = _make_small_integrators(poni_fixture, rot1_angles=[0.0, 5.0, 10.0])
    images = _synthetic_images(n=3, shape=(100, 100), seed=1)

    result = stitch_1d(
        images,
        integrators,
        npt=200,
        unit="q_A^-1",
        method="BBox",
        correctSolidAngle=False,
    )

    assert isinstance(result, IntegrationResult1D)
    assert result.radial.shape == (200,)
    assert result.intensity.shape == (200,)


def test_stitch_1d_normalization(poni_fixture):
    integrators = _make_small_integrators(poni_fixture, rot1_angles=[0.0, 5.0, 10.0])
    images = _synthetic_images(n=3, shape=(100, 100), seed=2)

    unnorm = stitch_1d(
        images,
        integrators,
        npt=200,
        unit="q_A^-1",
        method="BBox",
        correctSolidAngle=False,
    )
    normed = stitch_1d(
        images,
        integrators,
        npt=200,
        unit="q_A^-1",
        method="BBox",
        normalization=[1.0, 2.0, 0.5],
        correctSolidAngle=False,
    )

    assert isinstance(normed, IntegrationResult1D)
    assert normed.radial.shape == (200,)
    assert not np.allclose(normed.intensity, unnorm.intensity, rtol=1e-6, atol=1e-12)


def test_stitch_2d_runs(poni_fixture):
    integrators = _make_small_integrators(poni_fixture, rot1_angles=[0.0, 5.0, 10.0])
    images = _synthetic_images(n=3, shape=(100, 100), seed=3)

    result = stitch_2d(
        images,
        integrators,
        npt_rad=100,
        npt_azim=50,
        unit="q_A^-1",
        method="BBox",
        correctSolidAngle=False,
    )

    assert isinstance(result, IntegrationResult2D)
    assert result.radial.shape == (100,)
    assert result.azimuthal.shape == (50,)
    assert result.intensity.shape == (100, 50)


def test_stitch_2d_normalization(poni_fixture):
    """``normalization=`` on stitch_2d must match pre-dividing the images.

    Regression test for the stitch_1d / stitch_2d asymmetry: prior to the
    fix, stitch_2d ignored normalization entirely.  Both paths go through
    the same ``_prepare_images`` helper now.
    """
    integrators = _make_small_integrators(poni_fixture, rot1_angles=[0.0, 5.0, 10.0])
    images = _synthetic_images(n=3, shape=(100, 100), seed=42)
    norm = np.array([1.0, 2.0, 0.5])

    via_param = stitch_2d(
        images,
        integrators,
        npt_rad=100,
        npt_azim=50,
        unit="q_A^-1",
        method="BBox",
        normalization=norm,
        correctSolidAngle=False,
    )

    pre_divided = [img / n for img, n in zip(images, norm)]
    via_premul = stitch_2d(
        pre_divided,
        integrators,
        npt_rad=100,
        npt_azim=50,
        unit="q_A^-1",
        method="BBox",
        correctSolidAngle=False,
    )

    assert isinstance(via_param, IntegrationResult2D)
    np.testing.assert_allclose(
        via_param.intensity, via_premul.intensity,
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(via_param.radial, via_premul.radial)
    np.testing.assert_allclose(via_param.azimuthal, via_premul.azimuthal)


def test_stitch_2d_normalization_mismatched_length(poni_fixture):
    """Same length-mismatch error as stitch_1d (shared via _prepare_images)."""
    integrators = _make_small_integrators(poni_fixture, rot1_angles=[0.0, 5.0, 10.0])
    images = _synthetic_images(n=3, shape=(100, 100), seed=4)

    with pytest.raises(ValueError, match="normalization length"):
        stitch_2d(
            images,
            integrators,
            npt_rad=100,
            npt_azim=50,
            unit="q_A^-1",
            method="BBox",
            normalization=[1.0, 2.0],  # only 2 for 3 images
            correctSolidAngle=False,
        )


def test_stitch_images_routes_and_handles_rot2(monkeypatch):
    """stitch_images dispatches mode->stitch_1d/2d, builds integrators from
    rot1/rot2, and treats all-zero rot2 as a pure rot1 scan."""
    import ssrl_xrd_tools.integrate.multi as multi

    monkeypatch.setattr(
        multi, "create_multigeometry_integrators",
        lambda poni, rot1_angles, rot2_angles=None: ("INTEG", rot1_angles, rot2_angles),
    )
    monkeypatch.setattr(multi, "stitch_1d", lambda images, integ, **kw: ("1d", integ, kw))
    monkeypatch.setattr(multi, "stitch_2d", lambda images, integ, **kw: ("2d", integ, kw))

    imgs = np.zeros((2, 3, 3))

    r1 = multi.stitch_images(imgs, "PONI", [10.0, 20.0], mode="1d", npt_1d=500)
    assert r1[0] == "1d"
    assert r1[1][2] is None              # rot2=None when absent
    assert r1[2]["npt"] == 500

    r1z = multi.stitch_images(imgs, "PONI", [10.0, 20.0], [0.0, 0.0], mode="1d")
    assert r1z[1][2] is None             # all-zero rot2 -> None

    r2 = multi.stitch_images(imgs, "PONI", [10.0, 20.0], [1.0, 2.0],
                             mode="2d", npt_rad_2d=300)
    assert r2[0] == "2d"
    assert list(r2[1][2]) == [1.0, 2.0]  # nonzero rot2 passed through
    assert r2[2]["npt_rad"] == 300

    with pytest.raises(ValueError):
        multi.stitch_images(imgs, "PONI", [10.0], mode="bogus")


def test_stitch_images_rejects_count_mismatch():
    """stitch_images must reject images != angles before MultiGeometry."""
    imgs = [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
    poni = PONI(dist=0.1, poni1=0.01, poni2=0.02, wavelength=1e-10, detector="Detector")
    from ssrl_xrd_tools.integrate.multi import stitch_images
    with pytest.raises(ValueError, match="images != "):
        stitch_images(imgs, poni, [10.0, 20.0])  # 3 images, 2 angles


def test_prepare_images_rejects_bad_normalization():
    """Zero / non-finite normalization values fail early (no divide-by-zero)."""
    from ssrl_xrd_tools.integrate.multi import _prepare_images
    imgs = [np.ones((2, 2)), np.ones((2, 2))]
    with pytest.raises(ValueError, match="zero"):
        _prepare_images(imgs, [1.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        _prepare_images(imgs, [1.0, np.inf])
    # valid normalization still works
    out = _prepare_images(imgs, [2.0, 4.0])
    assert np.allclose(out[0], 0.5) and np.allclose(out[1], 0.25)
