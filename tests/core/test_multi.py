"""Tests for xrd_tools.integrate.multi."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from pyFAI.detectors import Detector

from xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from xrd_tools.integrate.multi import (
    _streaming_multigeometry,
    create_multigeometry_integrators,
    stitch_1d,
    stitch_1d_streaming,
    stitch_2d,
    stitch_2d_streaming,
)
from xrd_tools.integrate.detector_mask import mask_with_detector


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
    """Monitor normalization follows pyFAI MultiGeometry/notebook semantics."""
    from pyFAI.multi_geometry import MultiGeometry

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

    reference = MultiGeometry(integrators, unit="q_A^-1").integrate2d(
        images,
        100,
        50,
        method="BBox",
        correctSolidAngle=False,
        normalization_factor=norm,
    )

    assert isinstance(via_param, IntegrationResult2D)
    np.testing.assert_allclose(
        via_param.intensity, np.asarray(reference.intensity).T,
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(via_param.radial, reference.radial)
    np.testing.assert_allclose(via_param.azimuthal, reference.azimuthal)


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


@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_one_ai_streaming_matches_multigeometry_raw_accumulators(
    poni_fixture,
    mode,
):
    from pyFAI.multi_geometry import MultiGeometry

    angles = [0.0, 4.0, 9.0]
    images = _synthetic_images(n=3, shape=(100, 100), seed=91)
    monitor = np.asarray([1.0, 2.5, 0.75])
    mask = np.zeros((100, 100), dtype=bool)
    mask[8:11, 17:23] = True
    radial_range = (0.1, 6.0)
    azimuth_range = (-180.0, 180.0) if mode == "2d" else None
    reference_integrators = _make_small_integrators(poni_fixture, angles)
    reference = MultiGeometry(
        reference_integrators,
        unit="q_A^-1",
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        threadpoolsize=0,
    )
    masks = [mask_with_detector(ai, mask) for ai in reference_integrators]
    if mode == "1d":
        eager = reference.integrate1d(
            images,
            120,
            method="BBox",
            error_model="poisson",
            normalization_factor=monitor,
            lst_mask=masks,
        )
        streamed, diagnostics, _ = stitch_1d_streaming(
            images,
            len(images),
            lambda: iter(_make_small_integrators(poni_fixture, angles)),
            npt=120,
            unit="q_A^-1",
            method="BBox",
            radial_range=radial_range,
            mask=mask,
            normalization=monitor,
            error_model="poisson",
        )
        np.testing.assert_array_equal(streamed.radial, eager.radial)
        np.testing.assert_array_equal(streamed.intensity, eager.intensity)
        np.testing.assert_array_equal(streamed.sigma, eager.sigma)
        np.testing.assert_array_equal(diagnostics.coverage, eager.count)
        np.testing.assert_array_equal(
            diagnostics.normalization,
            eager.sum_normalization,
        )
    else:
        eager = reference.integrate2d(
            images,
            80,
            36,
            method="BBox",
            error_model="poisson",
            normalization_factor=monitor,
            lst_mask=masks,
        )
        streamed, diagnostics = stitch_2d_streaming(
            images,
            len(images),
            lambda: iter(_make_small_integrators(poni_fixture, angles)),
            npt_rad=80,
            npt_azim=36,
            unit="q_A^-1",
            method="BBox",
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            mask=mask,
            normalization=monitor,
            error_model="poisson",
        )
        np.testing.assert_array_equal(streamed.radial, eager.radial)
        np.testing.assert_array_equal(streamed.azimuthal, eager.azimuthal)
        np.testing.assert_array_equal(streamed.intensity, eager.intensity.T)
        np.testing.assert_array_equal(streamed.sigma, eager.sigma.T)
        np.testing.assert_array_equal(diagnostics.coverage, eager.count.T)
        np.testing.assert_array_equal(
            diagnostics.normalization,
            eager.sum_normalization.T,
        )


@pytest.mark.parametrize("error_model", (None, "poisson", "azimuthal"))
@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_streaming_multigeometry_matches_native_union_result(
    poni_fixture,
    mode,
    error_model,
):
    """The bounded fold is the exact native pyFAI 2026 union operation."""
    from pyFAI.multi_geometry import MultiGeometry

    angles = [0.0, 4.0, 9.0]
    images = _synthetic_images(n=3, shape=(100, 100), seed=191)
    monitor = np.asarray([1.0, 2.5, 0.75])
    mask = np.zeros((100, 100), dtype=bool)
    mask[8:11, 17:23] = True
    radial_range = (0.1, 6.0)
    azimuth_range = (-180.0, 180.0) if mode == "2d" else None
    reference_integrators = _make_small_integrators(poni_fixture, angles)
    reference = MultiGeometry(
        reference_integrators,
        unit="q_A^-1",
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        threadpoolsize=0,
    )
    masks = [mask_with_detector(ai, mask) for ai in reference_integrators]
    common = dict(
        method="BBox",
        error_model=error_model,
        normalization_factor=monitor,
        lst_mask=masks,
        correctSolidAngle=False,
    )
    if mode == "1d":
        eager = reference.integrate1d(images, 120, **common)
        npt_rad = 120
        npt_azim = 1
    else:
        eager = reference.integrate2d(images, 80, 36, **common)
        npt_rad = 80
        npt_azim = 36

    streamed, _gi, _profile = _streaming_multigeometry(
        images,
        len(images),
        lambda: iter(_make_small_integrators(poni_fixture, angles)),
        mode=mode,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit="q_A^-1",
        method="BBox",
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=mask,
        normalization=monitor,
        correct_solid_angle=False,
        error_model=error_model,
        polarization_factor=None,
    )

    np.testing.assert_array_equal(streamed.radial, eager.radial)
    if mode == "2d":
        np.testing.assert_array_equal(streamed.azimuthal, eager.azimuthal)
    for field in (
        "intensity",
        "sum_signal",
        "sum_normalization",
        "count",
        "sigma",
        "sum_variance",
        "sum_normalization2",
        "sem",
        "std",
    ):
        actual = getattr(streamed, field)
        expected = getattr(eager, field)
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)

    assert streamed.intensity.dtype == eager.intensity.dtype == np.dtype(np.float32)
    assert streamed.error_model == eager.error_model
    if mode == "1d":
        assert np.isnan(eager.dummy)
    else:
        assert eager.dummy == np.float32(0.0)
    if np.isnan(eager.dummy):
        assert np.isnan(streamed.dummy)
    else:
        assert streamed.dummy == eager.dummy
    empty = eager.sum_normalization == 0
    assert np.any(empty)
    assert np.any(~empty)
    if np.isnan(eager.dummy):
        np.testing.assert_array_equal(np.isnan(streamed.intensity), empty)
    else:
        np.testing.assert_array_equal(
            streamed.intensity[empty],
            np.full(np.count_nonzero(empty), eager.dummy, dtype=eager.intensity.dtype),
        )
    if error_model is None:
        assert streamed.sigma is None
        assert streamed.sum_variance is None
        assert streamed.sum_normalization2 is None
    else:
        assert streamed.sigma is not None
        assert streamed.sum_variance is not None
        assert streamed.sum_normalization2 is not None


def test_streaming_multigeometry_preserves_native_mixed_variance_union(
    poni_fixture,
    monkeypatch,
):
    """A missing variance contributor follows pyFAI's native union contract."""
    from pyFAI.multi_geometry import MultiGeometry

    angles = [0.0, 4.0]
    images = _synthetic_images(n=2, shape=(100, 100), seed=291)
    radial_range = (0.1, 6.0)
    azimuth_range = (-180.0, 180.0)
    native_results = []
    for integrator, image in zip(
        _make_small_integrators(poni_fixture, angles),
        images,
        strict=True,
    ):
        native_results.append(
            MultiGeometry(
                [integrator],
                unit="q_A^-1",
                radial_range=radial_range,
                azimuth_range=azimuth_range,
                threadpoolsize=0,
            ).integrate1d(
                [image],
                120,
                method="BBox",
                error_model="poisson",
                correctSolidAngle=False,
            )
        )

    # The public streaming API applies one error model to every frame, so make
    # the otherwise-unreachable mixed upstream state from genuine pyFAI
    # results.  union() deliberately drops the aggregate raw variance when any
    # contributor lacks it while preserving pyFAI's own recalculated payload.
    native_results[1]._set_sum_variance(None)
    native_results[1]._set_sum_normalization2(None)
    expected = native_results[0].union(native_results[1])
    results = iter(native_results)
    monkeypatch.setattr(
        MultiGeometry,
        "integrate1d",
        lambda _owner, *_args, **_kwargs: next(results),
    )

    streamed, _gi, _profile = _streaming_multigeometry(
        images,
        len(images),
        lambda: iter(_make_small_integrators(poni_fixture, angles)),
        mode="1d",
        npt_rad=120,
        npt_azim=1,
        unit="q_A^-1",
        method="BBox",
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=None,
        normalization=None,
        correct_solid_angle=False,
        error_model="poisson",
        polarization_factor=None,
    )

    assert streamed.sum_variance is None
    assert streamed.sum_normalization2 is None
    for field in (
        "intensity",
        "sum_signal",
        "sum_normalization",
        "count",
        "sigma",
        "sem",
        "std",
    ):
        actual = getattr(streamed, field)
        reference = getattr(expected, field)
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference)


def test_one_ai_streaming_uses_bounded_range_scout_and_cache_residency(
    poni_fixture,
    monkeypatch,
):
    from pyFAI.multi_geometry import MultiGeometry

    real_integrate = MultiGeometry.integrate1d
    references = []
    image_references = []
    released_image_counts = []
    active_cache_counts = []
    factory_calls = []

    def image_stream():
        for seed in (101, 102, 103):
            image = _synthetic_images(n=1, shape=(100, 100), seed=seed)[0]
            image_references.append(weakref.ref(image))
            yield image
            del image
            gc.collect()
            released = sum(reference() is None for reference in image_references)
            released_image_counts.append(released)
            assert released == len(image_references)

    def factory():
        factory_calls.append("factory")
        for integrator in _make_small_integrators(
            poni_fixture,
            [0.0, 4.0, 9.0],
        ):
            references.append(weakref.ref(integrator))
            yield integrator

    def observe_caches(owner, *args, **kwargs):
        result = real_integrate(owner, *args, **kwargs)
        active_cache_counts.append(
            sum(
                bool(getattr(value, "_cached_array", {}))
                for reference in references
                if (value := reference()) is not None
            )
        )
        return result

    def forbidden_guess(_owner):
        raise AssertionError("one-AI integration must receive globally fixed ranges")

    monkeypatch.setattr(MultiGeometry, "integrate1d", observe_caches)
    monkeypatch.setattr(MultiGeometry, "_guess_radial_range", forbidden_guess)
    monkeypatch.setattr(MultiGeometry, "_guess_azimuth_range", forbidden_guess)
    result, diagnostics, _ = stitch_1d_streaming(
        image_stream(),
        3,
        factory,
        npt=120,
        radial_range=(0.1, 6.0),
        method="BBox",
    )
    assert result.intensity.shape == (120,)
    assert diagnostics.coverage.shape == (120,)
    assert factory_calls == ["factory", "factory"]
    assert active_cache_counts and max(active_cache_counts) == 1
    assert released_image_counts == [1, 2, 3]
    assert all(reference() is None for reference in image_references)


def test_one_ai_streaming_scout_matches_multigeometry_chi_convention(
    poni_fixture,
):
    """A caller's 0..360 AIs must not poison the default -180..180 scout."""
    from pyFAI.multi_geometry import MultiGeometry

    angles = [0.0, 7.0, 13.0]
    images = _synthetic_images(n=3, shape=(100, 100), seed=303)

    def factory():
        integrators = _make_small_integrators(poni_fixture, angles)
        for integrator in integrators:
            integrator.setChiDiscAtZero()
            yield integrator

    eager_integrators = list(factory())
    eager = MultiGeometry(
        eager_integrators,
        unit="q_A^-1",
        threadpoolsize=0,
    ).integrate2d(
        images,
        80,
        72,
        method="BBox",
        correctSolidAngle=False,
    )
    streamed, diagnostics = stitch_2d_streaming(
        images,
        len(images),
        factory,
        npt_rad=80,
        npt_azim=72,
        unit="q_A^-1",
        method="BBox",
        correct_solid_angle=False,
    )

    np.testing.assert_array_equal(streamed.radial, eager.radial)
    np.testing.assert_array_equal(streamed.azimuthal, eager.azimuthal)
    np.testing.assert_array_equal(streamed.intensity, eager.intensity.T)
    np.testing.assert_array_equal(diagnostics.coverage, eager.count.T)
    np.testing.assert_array_equal(
        diagnostics.normalization,
        eager.sum_normalization.T,
    )


def test_stitch_images_routes_and_handles_rot2(monkeypatch):
    """stitch_images dispatches mode->stitch_1d/2d, builds integrators from
    rot1/rot2, and treats all-zero rot2 as a pure rot1 scan."""
    import xrd_tools.integrate.multi as multi

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
    from xrd_tools.integrate.multi import stitch_images
    with pytest.raises(ValueError, match="images != "):
        stitch_images(imgs, poni, [10.0, 20.0])  # 3 images, 2 angles


def test_stitch_images_accepts_single_2d_ndarray(monkeypatch):
    """A bare (H, W) ndarray is one image, not H images — the count guard
    must not reject it (P2 regression: len(2D ndarray) == H was wrong)."""
    import xrd_tools.integrate.multi as multi

    monkeypatch.setattr(
        multi, "create_multigeometry_integrators",
        lambda poni, rot1_angles, rot2_angles=None: "INTEG",
    )
    monkeypatch.setattr(multi, "stitch_1d", lambda images, integ, **kw: "OK")

    # (4, 4) image + a single angle: count must resolve to 1, not 4.
    assert multi.stitch_images(np.zeros((4, 4)), "PONI", [10.0], mode="1d") == "OK"

    # And a genuine mismatch (1 image, 2 angles) still raises.
    with pytest.raises(ValueError, match="images != "):
        multi.stitch_images(np.zeros((4, 4)), "PONI", [10.0, 20.0], mode="1d")


def test_prepare_images_rejects_bad_normalization():
    """Zero / non-finite normalization values fail early (no divide-by-zero)."""
    from xrd_tools.integrate.multi import _prepare_images
    imgs = [np.ones((2, 2)), np.ones((2, 2))]
    with pytest.raises(ValueError, match="zero"):
        _prepare_images(imgs, [1.0, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        _prepare_images(imgs, [1.0, np.inf])
    # valid normalization still works
    out = _prepare_images(imgs, [2.0, 4.0])
    assert np.allclose(out[0], 0.5) and np.allclose(out[1], 0.25)


def test_stitch_images_backend_pyfai_hist_routes_to_histogram(monkeypatch):
    """backend='pyfai_hist' builds the same integrators but merges through the
    q-histogram (pyfai_q_frames -> stitch_q_grid) instead of MultiGeometry,
    forwarding the radial bin count + azim + unit."""
    import xrd_tools.integrate.multi as multi
    import xrd_tools.integrate.stitch_hist as sh

    monkeypatch.setattr(
        multi, "create_multigeometry_integrators",
        lambda poni, rot1_angles, rot2_angles=None: "INTEG")
    seen = {}

    def fake_pyfai_q_frames(images, integrators, *, mask=None, normalization=None):
        seen["qframes"] = (integrators, mask, normalization)
        return iter([])

    def fake_stitch_q_grid(frames, *, mode, npt, npt_azim, unit,
                           radial_range, azimuth_range):
        frames()                       # the factory must be invokable (replayable)
        seen["grid"] = dict(mode=mode, npt=npt, npt_azim=npt_azim, unit=unit)
        return f"HIST_{mode}"

    monkeypatch.setattr(sh, "pyfai_q_frames", fake_pyfai_q_frames)
    monkeypatch.setattr(sh, "stitch_q_grid", fake_stitch_q_grid)

    imgs = np.zeros((2, 3, 3))
    r1 = multi.stitch_images(imgs, "PONI", [10.0, 20.0], mode="1d",
                             npt_1d=512, unit="q_A^-1", backend="pyfai_hist")
    assert r1 == "HIST_1d"
    assert seen["grid"]["mode"] == "1d" and seen["grid"]["npt"] == 512
    assert seen["qframes"][0] == "INTEG"

    r2 = multi.stitch_images(imgs, "PONI", [10.0, 20.0], mode="2d",
                             npt_rad_2d=256, npt_azim_2d=180, backend="pyfai_hist")
    assert r2 == "HIST_2d"
    assert seen["grid"]["npt"] == 256 and seen["grid"]["npt_azim"] == 180


def test_stitch_images_pyfai_hist_rejects_non_q_unit(monkeypatch):
    """The pyfai_hist merge emits q only — a non-q unit must fail loud."""
    import xrd_tools.integrate.multi as multi
    monkeypatch.setattr(
        multi, "create_multigeometry_integrators",
        lambda poni, rot1_angles, rot2_angles=None: "INTEG")
    with pytest.raises(ValueError, match="q_A"):
        multi.stitch_images(np.zeros((1, 3, 3)), "PONI", [10.0],
                            mode="1d", unit="2th_deg", backend="pyfai_hist")


def test_stitch_images_rejects_unknown_backend(monkeypatch):
    import xrd_tools.integrate.multi as multi
    monkeypatch.setattr(
        multi, "create_multigeometry_integrators",
        lambda poni, rot1_angles, rot2_angles=None: "INTEG")
    with pytest.raises(ValueError, match="backend"):
        multi.stitch_images(np.zeros((1, 3, 3)), "PONI", [10.0],
                            mode="1d", backend="bogus")


@pytest.mark.parametrize("angles", [(.4, 0.), (.4, 2.)])
def test_the_streaming_gi_cuts_equal_pyfais_multigeometryfiber_and_leave_iq_alone(angles):
    """A GI Stitch 1D with pyFAI (maintainer, 2026-09-29): each cut, streamed one frame at a time with
    the frame's incidence and tilt, equals pyFAI's MultiGeometryFiber over all frames at once
    (npt_ip = npt_oop, the size pyFAI's accumulators take), level or tilted; Qip + Qoop gives exactly
    the two single-cut results; I(q) beside them is the plain streaming result."""
    import numpy as np
    from pyFAI.multi_geometry import MultiGeometryFiber
    from pyFAI.units import get_unit_fiber
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.gid import _fiber_from_integrator
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
    from tests.core.test_stacked_arm_extension import _calibration
    arm = StackedDetectorArm(nu_motor='nu', del_motor='del', nu_offset=.01, del_offset=-.02)
    diff = Diffractometer.from_pyfai_goniometer(arm.to_pyfai_goniometer(_calibration()),
                                                source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    motors = {'del': np.array([10., 24., 38.]), 'nu': np.array([2., 9., 15.])}
    series = create_multigeometry_integrator_series_from_geometry(diff, motors)
    rng = np.random.default_rng(5)
    images = [rng.uniform(5., 50., (195, 1475)) for _ in range(3)]
    ranges = dict(ip_range=(-2.5, 2.5), oop_range=(0., 6.))
    common = dict(npt=64, radial_range=(.5, 6.), correct_solid_angle=True, polarization_factor=.99)

    def stitch(*cuts):
        gi = StitchGI(cuts, incident_angle_deg=.4, **ranges) if cuts else None
        return stitch_1d_streaming(images, 3, series.__iter__, gi=gi, gi_angles=[angles] * 3 if cuts else None,
                                   **common)

    payload, _diagnostics, both = stitch('q_oop', 'q_ip')
    plain, _, none = stitch()
    assert none == () and len(both) == 2
    np.testing.assert_array_equal(payload.intensity, plain.intensity)
    incidence, tilt = angles
    fibers = [_fiber_from_integrator(ai, incident_angle=incidence, tilt_angle=tilt, sample_orientation=2)
              for ai in series]
    # pyFAI's histogram admits a fractional index below the first bin. Apply the same
    # closed physical window as Stitch rather than crediting those outside pixels.
    window_masks = []
    options = dict(incident_angle=incidence, tilt_angle=tilt, sample_orientation=2, angle_unit='deg')
    for fiber in fibers:
        outside = np.zeros(images[0].shape, dtype=bool)
        for name, limits in (('qip_A^-1', ranges['ip_range']), ('qoop_A^-1', ranges['oop_range'])):
            values = fiber.array_from_unit(images[0].shape, 'center', get_unit_fiber(name, **options), scale=True)
            outside |= (values < limits[0]) | (values > limits[1])
        if fiber.detector.mask is not None:
            outside |= np.asarray(fiber.detector.mask, dtype=bool)
        window_masks.append(outside)
    for name, (cut, cut_diagnostics) in zip(('q_ip', 'q_oop'), both):
        (single, single_diagnostics), = stitch(name)[2]
        for mine, theirs in ((cut.radial, single.radial), (cut.intensity, single.intensity),
                             (cut_diagnostics.coverage, single_diagnostics.coverage),
                             (cut_diagnostics.normalization, single_diagnostics.normalization)):
            np.testing.assert_array_equal(mine, theirs)
        reference = MultiGeometryFiber(fibers, unit=('qip_A^-1', 'qoop_A^-1'), incident_angle=np.radians(incidence),
                                       tilt_angle=np.radians(tilt), sample_orientation=2, **ranges
                                       ).integrate1d_grazing_incidence(images, npt_ip=64, npt_oop=64,
                                                                       vertical_integration=name == 'q_oop',
                                                                       polarization_factor=.99,
                                                                       lst_mask=window_masks)
        assert cut.unit == GI_UNITS[name]
        np.testing.assert_allclose(cut.radial, reference.radial, rtol=0, atol=1e-12)
        filled = np.asarray(reference.count) > 0
        assert filled.sum() > 20
        np.testing.assert_array_equal(cut_diagnostics.coverage, reference.count)
        np.testing.assert_allclose(cut.intensity[filled], np.asarray(reference.intensity)[filled], rtol=1e-12)


def test_an_auto_gi_range_windows_every_pixel_as_a_covering_one_does():
    """C12: a cut's window on Auto keeps every pixel. The one 2-D pass needs both ranges, so Auto is
    the scouted span; the cut equals the same cut windowed by a range that covers every pixel (no
    pixel lost at the span's edges)."""
    import numpy as np
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    from xrd_tools.integrate.stitch_gi import StitchGI
    from tests.core.test_stacked_arm_extension import _calibration
    arm = StackedDetectorArm(nu_motor='nu', del_motor='del', nu_offset=.01, del_offset=-.02)
    diff = Diffractometer.from_pyfai_goniometer(arm.to_pyfai_goniometer(_calibration()),
                                                source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    series = create_multigeometry_integrator_series_from_geometry(
        diff, {'del': np.array([10., 24.]), 'nu': np.array([2., 9.])})
    rng = np.random.default_rng(7)
    images = [rng.uniform(5., 50., (195, 1475)) for _ in range(2)]

    def cut(name, **ranges):
        gi = StitchGI((name,), incident_angle_deg=.4, **ranges)
        (result, diagnostics), = stitch_1d_streaming(images, 2, series.__iter__, npt=64, gi=gi,
                                                     gi_angles=[(.4, 0.)] * 2)[2]
        return result, diagnostics

    for name, own, other in (('q_ip', 'ip_range', 'oop_range'), ('q_oop', 'oop_range', 'ip_range')):
        bins = {own: (-2., 2.) if name == 'q_ip' else (0., 6.)}
        auto, auto_diagnostics = cut(name, **bins)
        wide, wide_diagnostics = cut(name, **bins, **{other: (-50., 50.)})
        assert auto_diagnostics.coverage.sum() > 0
        np.testing.assert_array_equal(auto_diagnostics.coverage, wide_diagnostics.coverage)
        np.testing.assert_array_equal(auto.intensity, wide.intensity)


def test_a_gi_cuts_scratch_grows_with_its_points_not_their_square():
    """Codex audit G-F1: each cut is a thin pass (its axis in npt bins, the other coordinate in one),
    so a frame's working memory stays O(npt); the square npt × npt histogram grew 12× from 160 to
    640 points (1.7 → 21.6 MB traced on a 48 × 60 detector)."""
    import gc
    import tracemalloc
    import numpy as np
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from xrd_tools.integrate.multi import _gi_frame_sums
    from xrd_tools.integrate.stitch_gi import StitchGI
    image = np.full((48, 60), 100.)
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.2, ip_range=(-1., 1.), oop_range=(-1., 1.))
    peaks = {}
    for npt in (160, 640):
        integrator = AzimuthalIntegrator(dist=.2, poni1=21.1 * 75e-6, poni2=30.3 * 75e-6, rot1=.02, rot2=.05,
                                         rot3=.01, wavelength=1e-10,
                                         detector=Detector(pixel1=75e-6, pixel2=75e-6, max_shape=image.shape))
        gc.collect()
        tracemalloc.start()
        sums = _gi_frame_sums(integrator, image, (.2, 0.), gi, {"q_ip": (-1., 1.), "q_oop": (-1., 1.)}, npt,
                              mask=None, correct_solid_angle=True, polarization_factor=None, dark=None, flat=None,
                              monitor=1.)
        peaks[npt] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert [len(values[0]) for values in sums] == [npt, npt]
    assert peaks[640] < 2 * peaks[160], peaks  # O(npt): the square grew 12x


def _window_probe_integrator(detector_mask=None):
    """Codex final review F1's geometry: a generic 29 × 43 detector (100 × 120 µm pixels) at 81 mm,
    off-centre PONI, three rotations, 0.76 Å."""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    detector = Detector(pixel1=100e-6, pixel2=120e-6, max_shape=(29, 43))
    if detector_mask is not None:
        detector.mask = detector_mask
    return AzimuthalIntegrator(dist=.081, poni1=11.3 * 100e-6, poni2=26.7 * 120e-6, rot1=.047, rot2=-.032,
                               rot3=.081, wavelength=7.6e-11, detector=detector)


def _window_probe_coordinates(angles):
    """Every pixel's (Qip, Qoop) in Å⁻¹ at pyFAI's (incidence, tilt) in degrees, orientation 2, from pyFAI's
    own FiberIntegrator and fiber units rather than the kernel's."""
    from pyFAI.integrator.fiber import FiberIntegrator
    from pyFAI.units import get_unit_fiber
    source = _window_probe_integrator()
    fiber = FiberIntegrator(dist=source.dist, poni1=source.poni1, poni2=source.poni2, rot1=source.rot1,
                            rot2=source.rot2, rot3=source.rot3, wavelength=source.wavelength,
                            detector=source.detector)
    angle = dict(incident_angle=angles[0], tilt_angle=angles[1], sample_orientation=2, angle_unit="deg")
    return tuple(np.asarray(fiber.array_from_unit((29, 43), "center", get_unit_fiber(name, **angle), scale=True))
                 for name in ("qip_A^-1", "qoop_A^-1"))


def _window_sums(gi, bins, mask, image=None, integrator=None, **corrections):
    from xrd_tools.integrate.multi import _gi_frame_sums
    options = dict(correct_solid_angle=False, polarization_factor=None, dark=None, flat=None, monitor=1.)
    return _gi_frame_sums(integrator or _window_probe_integrator(), np.full((29, 43), 100.) if image is None
                          else image, (.37, 1.8), gi, bins, 37, mask=mask, **{**options, **corrections})


def test_codex_f1_a_pixel_below_the_other_window_counts_in_neither_gi_cut():
    """Codex final review F1: a thin pass's one-bin coordinate admitted pixels up to the whole window
    width below it (pyFAI's histogram truncates a bin index in (-1, 0) to bin 0). Its row: only pixel
    (28, 0) unmasked, its Qip 0.0907 inside the (-0.4, 0.4) window and its Qoop -0.4357 below it; the Qip
    cut counted it once (signal 100), the 37 × 37 square not at all. It counts in neither cut."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    q_ip, q_oop = _window_probe_coordinates((.37, 1.8))
    assert -.4 < q_ip[28, 0] < .4 and q_oop[28, 0] < -.4
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.37, tilt_angle_deg=1.8, ip_range=(-.4, .4),
                  oop_range=(-.4, .4))
    mask = np.ones((29, 43), dtype=np.int8)
    mask[28, 0] = 0
    sums = _window_sums(gi, {"q_ip": (-.4, .4), "q_oop": (-.4, .4)}, mask)
    assert [(values[0].sum(), values[2].sum()) for values in sums] == [(0., 0.), (0., 0.)]


@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize("coordinate", ["q_ip", "q_oop"])
def test_a_gi_pixel_counts_in_both_cuts_only_inside_the_closed_window(coordinate, side):
    """The GI window is the closed rectangle of the explicit Qip and Qoop ranges, for a cut's own
    coordinate and the other alike. One unmasked pixel a hair (1e-9 relative) outside one bound counts
    in neither cut; a hair inside, in both. Below a minimum pyFAI's truncation admitted it (within a bin
    on the cut's own axis, within the whole window on the other's one bin); above a maximum, the top
    edge pyFAI stretches to take the maximum."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    pixel = (14, 21)
    coordinates = {name: float(values[pixel]) for name, values in
                   zip(("q_ip", "q_oop"), _window_probe_coordinates((.37, 1.8)))}
    assert min(abs(value) for value in coordinates.values()) > .05
    mask = np.ones((29, 43), dtype=np.int8)
    mask[pixel] = 0
    for outside in (True, False):
        ranges = {name: (value - .3, value + .3) for name, value in coordinates.items()}
        value = coordinates[coordinate]
        hair = abs(value) * 1e-9 * (1 if outside else -1)
        ranges[coordinate] = (value + hair, value + .3) if side == "low" else (value - .3, value - hair)
        gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.37, tilt_angle_deg=1.8, ip_range=ranges["q_ip"],
                      oop_range=ranges["q_oop"])
        counted = 0. if outside else 1.
        sums = _window_sums(gi, ranges, mask)
        assert [(values[0].sum(), values[2].sum()) for values in sums] == [(100. * counted, counted)] * 2, ranges


@pytest.mark.parametrize("solid_angle", [False, True])
@pytest.mark.parametrize("ranges", [{}, {"ip_range": (.2, .45)}, {"oop_range": (-.35, -.2)},
                                    {"ip_range": (.2, .45), "oop_range": (-.35, -.2)}],
                         ids=["auto", "explicit_ip", "explicit_oop", "explicit_both"])
def test_the_gi_window_holds_for_auto_and_explicit_ranges_under_every_correction(ranges, solid_angle):
    """Each Auto/explicit combination, with an asymmetric mask, dark, flat, polarization, a monitor and
    solid angle on or off: each thin cut counts exactly the valid pixels inside the window and equals
    pyFAI's full Qip × Qoop square over the same window (applied as a mask, so the square's own
    one-bin edge leak is out too) summed over the other axis."""
    from xrd_tools.integrate.gid import _fiber_from_integrator
    from xrd_tools.integrate.multi import _gi_bins
    from xrd_tools.integrate.stitch_gi import StitchGI
    rng = np.random.default_rng(194)
    shape = (29, 43)
    image, dark, flat = rng.uniform(3., 180., shape), rng.uniform(.1, 1., shape), rng.uniform(.5, 1.6, shape)
    mask = np.zeros(shape, dtype=np.int8)
    mask[1::4, 2::7] = 1
    angles = (.37, 1.8)
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.37, tilt_angle_deg=1.8, **ranges)
    bins = _gi_bins(lambda: iter((_window_probe_integrator(),)), (angles,), gi)
    inside = mask == 0
    for values, window in zip(_window_probe_coordinates(angles), (gi.ip_range, gi.oop_range)):
        if window is not None:
            inside &= (values >= window[0]) & (values <= window[1])
    assert bool(0 < inside.sum() < (mask == 0).sum()) == bool(ranges)  # the window bites unless Auto
    corrections = dict(polarization_factor=.67, dark=dark, flat=flat)
    thin = _window_sums(gi, bins, mask, image, correct_solid_angle=solid_angle, monitor=3.7, **corrections)
    fiber = _fiber_from_integrator(_window_probe_integrator(), incident_angle=angles[0], tilt_angle=angles[1],
                                   sample_orientation=2)
    square = fiber.integrate2d_fiber(
        image, npt_ip=37, unit_ip="qip_A^-1", ip_range=bins["q_ip"], npt_oop=37, unit_oop="qoop_A^-1",
        oop_range=bins["q_oop"], correctSolidAngle=solid_angle, mask=~inside, normalization_factor=3.7,
        method=("no", "histogram", "cython"), incident_angle=angles[0], tilt_angle=angles[1],
        sample_orientation=2, angle_unit="deg", **corrections)
    scale = fiber.pixel1 * fiber.pixel2 / fiber.dist ** 2 if solid_angle else 1.
    for (signal, normalization, count, axis), cut in zip(thin, gi.cuts):
        other = -2 if cut == "q_ip" else -1
        assert count.sum() == inside.sum()
        np.testing.assert_array_equal(count, np.asarray(square.count, dtype=np.float64).sum(axis=other))
        np.testing.assert_array_equal(axis, square.inplane if cut == "q_ip" else square.outofplane)
        np.testing.assert_allclose(signal, np.asarray(square.sum_signal, dtype=np.float64).sum(axis=other),
                                   rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            normalization, np.asarray(square._sum_normalization, dtype=np.float64).sum(axis=other) * scale,
            rtol=1e-12, atol=0)


def test_the_gi_window_keeps_the_detectors_own_mask():
    """pyFAI falls back to the detector's mask only when handed none: the window mask joins the
    detector's, so a detector-masked pixel inside the window stays out of both cuts."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    detector_mask = np.zeros((29, 43), dtype=np.int8)
    detector_mask[::3, ::5] = 1
    q_ip, q_oop = _window_probe_coordinates((.37, 1.8))
    window = (q_ip >= .2) & (q_ip <= .45) & (q_oop >= -.35) & (q_oop <= -.2)
    assert (window & (detector_mask == 1)).any()
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.37, tilt_angle_deg=1.8, ip_range=(.2, .45),
                  oop_range=(-.35, -.2))
    sums = _window_sums(gi, {"q_ip": (.2, .45), "q_oop": (-.35, -.2)}, None,
                        integrator=_window_probe_integrator(detector_mask))
    assert [values[2].sum() for values in sums] == [float((window & (detector_mask == 0)).sum())] * 2
