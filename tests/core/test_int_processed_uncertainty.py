"""Int sigma for processed images: a background-subtracted frame and an averaged frame.

pyFAI's Poisson model assumes the integrated image is raw counts; for raw − background
and for a mean of N frames the reducer passes the propagated variance instead.
"""
from pathlib import Path

import numpy as np
import tifffile

from xrd_tools.core.containers import PONI
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.reduction import (
    FrameBackgroundPlan, Frame, Integration1DPlan, ReductionPlan, Scan, resolve_frame_background,
    run_reduction,
)

SHAPE = (195, 487)
_PONI = PONI(dist=0.15, poni1=0.002, poni2=0.04, wavelength=1.0e-10, detector="Pilatus100k")


def _fact(path: Path, shape=(2, 2)):
    return (0, str(path.resolve()), None, 0, shape, ())


def test_the_resolved_background_carries_its_poisson_variance_factor(tmp_path):
    target = tmp_path / "target.tif"
    tifffile.imwrite(target, np.zeros((2, 2), np.float32))
    single = tmp_path / "background.tif"
    tifffile.imwrite(single, np.array([[4., 9.], [16., 25.]], np.float32))
    result = resolve_frame_background(
        FrameBackgroundPlan(mode="Single BG File", locator=str(single), scale=2.0), _fact(target))
    assert result.disposition == "RESOLVED" and result.variance_factor == 2.0

    series = tmp_path / "series"
    series.mkdir()
    for index, value in enumerate((10., 30.), 1):
        tifffile.imwrite(series / f"bg_{index}.tif", np.full((2, 2), value, np.float32))
    uniform = resolve_frame_background(
        FrameBackgroundPlan(mode="Series Average", locator=str(series / "bg_1.tif")), _fact(target))
    assert uniform.variance_factor == .5  # var(mean of 2) = mean / 2
    uneven_member = np.full((2, 2), 30., np.float32)
    uneven_member[0, 0] = np.nan
    tifffile.imwrite(series / "bg_2.tif", uneven_member)
    uneven = resolve_frame_background(
        FrameBackgroundPlan(mode="Series Average", locator=str(series / "bg_1.tif")), _fact(target))
    np.testing.assert_array_equal(uneven.variance_factor, [[1., .5], [.5, .5]])


def _reduce(frame, error_model="poisson"):
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=60, error_model=error_model), integration_2d=None)
    scan = Scan("bg", [frame], poni=_PONI, integrator=poni_to_integrator(_PONI))
    result = run_reduction(plan, scan, executor=1)
    assert not result.failed, result.error
    return result.frames[0].result_1d


def test_a_subtracted_background_propagates_its_variance():
    from xrd_tools.integrate.single import integrate_1d
    rng = np.random.default_rng(11)
    raw = rng.poisson(60, SHAPE).astype(float)
    background = 1.5 * rng.poisson(30, SHAPE).astype(float)  # scale 1.5 on raw counts
    reduced = _reduce(Frame(0, image=raw.copy(), metadata={}, background=background,
                            background_variance_factor=1.5))
    variance = np.maximum(raw, 1.0) + 1.5 * background
    direct = integrate_1d(raw - background, poni_to_integrator(_PONI), npt=60, method="csr",
                          variance=variance)
    np.testing.assert_allclose(reduced.intensity, direct.intensity, rtol=1e-12)
    np.testing.assert_allclose(reduced.sigma, direct.sigma, rtol=1e-12)
    # Poisson on the difference alone understates it
    unpropagated = _reduce(Frame(0, image=raw.copy(), metadata={}, background=background))
    assert np.nanmedian(reduced.sigma / unpropagated.sigma) > 1.5
    plain = _reduce(Frame(0, image=raw.copy(), metadata={}, background=background,
                          background_variance_factor=1.5), error_model=None)
    assert plain.sigma is None


def test_an_averaged_frame_has_the_uncertainty_of_the_mean(tmp_path):
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    from xrd_tools.integrate.single import integrate_1d
    from xrd_tools.io import get_1d
    from xrd_tools.reduction import AverageScanRecipe
    from xrd_tools.reduction import average as module
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues
    from xrd_tools.sources.selection import image_series_spec
    rng = np.random.default_rng(5)
    frames = [rng.poisson(20, (16, 16)).astype(np.uint32) for _ in range(4)]
    for index, frame in enumerate(frames, 1):
        tifffile.imwrite(tmp_path / f"scan_{index:04d}.tif", frame)
    config = {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [16, 16], "orientation": 3}
    values = PoniValues(0.2, 0.0008, 0.0008, 0.0, 0.0, 0.0, 1e-10)
    from xrd_tools.reduction import Integration2DPlan
    runner = module.AverageScanRunner(AverageScanRecipe(
        image_series_spec(tmp_path / "scan_0001.tif", metadata_format=None), tmp_path / "scan.nexus",
        ReductionPlan(integration_1d=Integration1DPlan(npt=8, method="numpy", error_model="poisson"),
                      integration_2d=Integration2DPlan(npt_rad=8, npt_azim=6, method="numpy",
                                                       error_model="poisson")),
        calibration=CalibrationState(values, "Detector", config, status=FactStatus.PRESENT)))
    result = runner.start()
    runner.close()
    assert result.disposition == "COMMITTED", result
    saved = get_1d(result.target, frame=1)
    mean = np.mean(np.stack(frames).astype(float), axis=0)
    integrator = detector_calibration_to_integrator(DetectorCalibration(
        PONI(0.2, 0.0008, 0.0008, 0.0, 0.0, 0.0, 1e-10, "Detector"), config))
    expected = integrate_1d(mean, integrator, npt=8, method="numpy", variance=np.maximum(mean, 1) / 4)
    poisson_on_mean = integrate_1d(mean, integrator, npt=8, method="numpy", error_model="poisson")
    finite = np.isfinite(expected.sigma) & (expected.sigma > 0)
    assert finite.any()
    np.testing.assert_allclose(saved.sigma[finite], expected.sigma[finite], rtol=1e-6)
    np.testing.assert_allclose(poisson_on_mean.sigma[finite] / saved.sigma[finite], 2.0, rtol=1e-6)


def test_a_subtracted_background_propagates_its_variance_in_2d():
    from xrd_tools.integrate.single import integrate_2d
    from xrd_tools.reduction import Integration2DPlan
    rng = np.random.default_rng(11)
    raw = rng.poisson(60, SHAPE).astype(float)
    background = 1.5 * rng.poisson(30, SHAPE).astype(float)
    plan = ReductionPlan(integration_1d=None,
                         integration_2d=Integration2DPlan(npt_rad=30, npt_azim=24, error_model="poisson"))
    scan = Scan("bg2d", [Frame(0, image=raw.copy(), metadata={}, background=background,
                               background_variance_factor=1.5)],
                poni=_PONI, integrator=poni_to_integrator(_PONI))
    result = run_reduction(plan, scan, executor=1)
    assert not result.failed, result.error
    reduced = result.frames[0].result_2d
    direct = integrate_2d(raw - background, poni_to_integrator(_PONI), npt_rad=30, npt_azim=24,
                          method=plan.integration_2d.method,
                          variance=np.maximum(raw, 1.0) + 1.5 * background)
    finite = np.isfinite(direct.sigma)
    assert finite.any()
    np.testing.assert_allclose(reduced.sigma[finite], direct.sigma[finite], rtol=1e-12)
    np.testing.assert_allclose(reduced.intensity[finite], direct.intensity[finite], rtol=1e-12)
