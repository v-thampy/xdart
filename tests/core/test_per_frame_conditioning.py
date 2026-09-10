"""Actual reduction and persistence with moving value-based invalid pixels."""
from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.io.frame_view import read_frame_record
from xrd_tools.io.read import get_1d, get_2d
from xrd_tools.reduction import (
    Frame, GIMode, Integration1DPlan, Integration2DPlan, NexusSink,
    ReductionPlan, Scan, run_reduction,
)


@pytest.mark.parametrize("thresholds", (False, True))
@pytest.mark.parametrize("saturation", (False, True))
def test_preview_priority_keeps_static_mask_and_uses_only_selected_values(thresholds,
                                                                        saturation):
    from xrd_tools.io.frame_preview import DetectorPreviewProjection, _masked_detector
    from xrd_tools.session.display_logic import sentinel_mask

    raw = np.array([[-1, 20], [100, 4294967295]], dtype=np.int64)
    static = np.array([[False, True], [False, False]])
    projection = DetectorPreviewProjection.from_mask(
        static, mask_saturation=saturation, saturation_ceiling=4294967295,
        apply_threshold=thresholds, threshold_max=4294967295 if thresholds else None,
    )
    result = _masked_detector(raw, projection)
    np.testing.assert_array_equal(np.isnan(result),
        static | (np.array([[True, False], [False, True]])
                  if saturation and not thresholds else False))
    off = sentinel_mask(raw, mask_saturation=False)
    np.testing.assert_array_equal(off, raw)


@pytest.mark.parametrize("static_mask", (False, True))
@pytest.mark.parametrize("mask_saturated", (False, True))
@pytest.mark.parametrize("threshold", ("off", "low", "ceiling"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_value_toggle_precedence_and_sparse_saturation(dtype, threshold, mask_saturated,
                                                      static_mask):
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    shape = (128, 128)  # One ceiling pixel is below the retired fraction cutoff.
    ceiling = np.iinfo(dtype).max
    raw = [np.full(shape, 20, dtype=dtype) for _ in range(2)]
    raw[0][100, 100] = ceiling
    raw[1][101, 100] = ceiling
    for image in raw:
        image[20, 20] = 200
    before = [image.copy() for image in raw]
    mask = np.zeros(shape, dtype=bool) if static_mask else None
    if mask is not None:
        mask[20, 20] = True
    maximum = None if threshold == "off" else 100 if threshold == "low" else float(ceiling)

    def scan(images):
        ai = AzimuthalIntegrator(
            dist=0.1, poni1=0.0, poni2=0.0, wavelength=1e-10,
            detector=Detector(pixel1=100e-6, pixel2=100e-6, max_shape=shape),
        )
        return Scan("toggle-values", [Frame(i, image=x) for i, x in enumerate(images)],
                    integrator=ai)

    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=16, method="csr", error_model="poisson"),
        integration_2d=Integration2DPlan(npt_rad=16, npt_azim=8, method="csr",
                                       error_model="poisson"),
        mask=mask, threshold_max=maximum, mask_saturation=mask_saturated,
    )
    actual = run_reduction(plan, scan(raw), execution="chunked", executor=False)
    expected_images = []
    for image in raw:
        working = image.astype(float)
        if maximum is not None:
            working[image > maximum] = np.nan
        elif mask_saturated:
            working[image == ceiling] = np.nan
        expected_images.append(working)
    expected = run_reduction(
        replace(plan, threshold_max=None, mask_saturation=False), scan(expected_images),
        execution="chunked", executor=False,
    )
    for index in actual.frames:
        for dimension in ("result_1d", "result_2d"):
            left, right = (getattr(result.frames[index], dimension)
                           for result in (actual, expected))
            np.testing.assert_array_equal(left.radial, right.radial)
            for field in ("intensity", "sigma"):
                np.testing.assert_allclose(getattr(left, field), getattr(right, field),
                                           rtol=2e-6, atol=1e-7, equal_nan=True)
        np.testing.assert_array_equal(raw[index], before[index])


@pytest.mark.parametrize("static_mask", (False, True), ids=("no-mask-file", "static-mask"))
@pytest.mark.parametrize("method", ("no", "csr"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_saturation_matches_threshold_with_stable_axes(tmp_path, static_mask, method, dtype):
    """Value exclusions change counts and errors, never the geometric grid."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    shape = (32, 32)
    ceiling = np.iinfo(dtype).max
    ordinary = (np.arange(np.prod(shape)).reshape(shape) % 100 + 1).astype(dtype)
    raw = [ordinary.copy() for _ in range(3)]
    raw[0][-4:, :] = ceiling
    raw[1][:4, :] = ceiling
    before = [image.copy() for image in raw]
    mask = np.zeros(shape, dtype=bool) if static_mask else None
    if mask is not None:
        mask[10, 10] = True

    def scan():
        ai = AzimuthalIntegrator(
            dist=0.1, poni1=0.0, poni2=0.0, wavelength=1e-10,
            detector=Detector(pixel1=100e-6, pixel2=100e-6, max_shape=shape),
        )
        return Scan("value-threshold", [Frame(i, image=x) for i, x in enumerate(raw)],
                    integrator=ai)

    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=16, method=method, error_model="poisson"),
        integration_2d=Integration2DPlan(npt_rad=16, npt_azim=8, method=method,
                                       error_model="poisson"),
        mask=mask, mask_saturation=True,
    )
    target = tmp_path / "saturation.nexus"
    actual = run_reduction(plan, scan(), NexusSink(target, overwrite=True),
                           execution="chunked", executor=False)
    reference = run_reduction(
        replace(plan, mask_saturation=False, threshold_max=float(ceiling) - 1),
        scan(), execution="chunked", executor=False,
    )
    assert actual.n_processed == reference.n_processed == 3
    for index, expected in reference.frames.items():
        one, two = get_1d(target, index), get_2d(target, index)
        for observed, wanted in (
            (one.q, expected.result_1d.radial),
            (one.intensity, expected.result_1d.intensity),
            (one.sigma, expected.result_1d.sigma),
            (two.q, expected.result_2d.radial),
            (two.chi, expected.result_2d.azimuthal),
            (two.intensity, expected.result_2d.intensity.T),
            (read_frame_record(target, index).active_view().sigma_2d,
             expected.result_2d.sigma.T),
        ):
            assert observed is not None and wanted is not None
            np.testing.assert_allclose(observed, wanted, rtol=2e-6, atol=1e-7,
                                       equal_nan=True)
        np.testing.assert_array_equal(one.q, get_1d(target, 2).q)
        np.testing.assert_array_equal(two.q, get_2d(target, 2).q)
        np.testing.assert_array_equal(raw[index], before[index])


@pytest.mark.parametrize("execution", ("chunked", "streaming"))
@pytest.mark.parametrize("gi", (False, True), ids=("standard", "gi"))
def test_each_frame_matches_local_value_exclusions(tmp_path, execution, gi):
    shape = (100, 100)
    static = np.zeros(shape, dtype=bool)
    static[80, 90] = True
    raw = [np.full(shape, 20 + index, dtype=np.uint32) for index in range(3)]
    for index, image in enumerate(raw):
        image[90 + index, 95] = np.iinfo(np.uint32).max
        image[90 + index, 96] = 1001
        image[90 + index, 97] = 0
        image[80, 90] = 900
    before = [image.copy() for image in raw]
    poni = PONI(0.2, 0.008, 0.008, wavelength=1e-10, detector="Pilatus100k")
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=32, method="no"),
        integration_2d=Integration2DPlan(npt_rad=32, npt_azim=16, method="no"),
        gi=GIMode(incident_angle=0.2, mode_1d="q_total", mode_2d="qip_qoop",
                  method="no") if gi else None,
        mask=static, mask_saturation=True, threshold_min=1, threshold_max=1000,
    )
    target = tmp_path / "conditioned.nexus"
    actual = run_reduction(
        plan, Scan("dynamic", [Frame(i, image=image, background=2.0) for i, image in enumerate(raw)],
                   poni=poni, integrator=poni_to_integrator(poni)),
        NexusSink(target, overwrite=True, thumbnail_max=128),
        execution=execution, executor=2 if execution == "streaming" else False,
        chunk_size=1,
    )
    assert actual.n_processed == 3
    for index, image in enumerate(raw):
        # Independently reject the native sentinel in this frame's working
        # values, keeping the accepted static geometry and thresholds.
        local_mask = static  # Thresholds override the separate raw saturation mask.
        working = image.astype(float)
        working[image >= np.iinfo(np.uint32).max] = np.nan
        reference = run_reduction(
            replace(plan, mask_saturation=False),
            Scan("reference", [Frame(index, image=working, background=2.0)], poni=poni,
                 integrator=poni_to_integrator(poni)),
        ).frames[index]
        one, two = get_1d(target, index), get_2d(target, index)
        for left, right in ((one.q, reference.result_1d.radial),
                            (one.intensity, reference.result_1d.intensity),
                            (two.q, reference.result_2d.radial),
                            (two.chi, reference.result_2d.azimuthal),
                            (two.intensity, reference.result_2d.intensity.T)):
            np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-7,
                                       equal_nan=True)
        view = read_frame_record(target, index).active_view()
        assert view.mask_baked and view.thumbnail is not None
        np.testing.assert_array_equal(np.isnan(view.thumbnail), local_mask)
        np.testing.assert_array_equal(image, before[index])
    np.testing.assert_array_equal(plan.mask, static)
