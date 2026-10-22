"""Native Stitch processing controls through the real MG/XU kernels."""


import numpy as np
import pytest

from xrd_tools.analysis.plans import StitchPlan, run_stitch
from xrd_tools.core.containers import PONI, convert_radial_axis
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.invalid import detector_value_mask
from xrd_tools.core.scan import ScanFrame
from xrd_tools.integrate.stitch_conditioning import (
    StitchConditioning, condition_stitch_frame, prepare_stitch_mask,
)
from xrd_tools.reduction.core import ReductionPlan, _apply_thresholds
from xrd_tools.sources.memory import MemoryFrameSource


@pytest.mark.parametrize("dtype", [np.int32, np.uint16, np.uint32, np.float32])
@pytest.mark.parametrize("mode", ["off", "auto", "manual"])
def test_conditioning_matches_int_native_policy(dtype, mode):
    high = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 4294967295.0
    raw = np.asarray([[0, 5, 10], [20, high, -1 if dtype in (np.int32, np.float32) else 1]], dtype=dtype)
    original = raw.copy()
    settings = StitchConditioning(mode, 5.0 if mode == "manual" else None,
                                 20.0 if mode == "manual" else None)
    plan = ReductionPlan(threshold_min=settings.threshold_min,
                         threshold_max=settings.threshold_max,
                         mask_saturation=mode == "auto")
    expected = raw.astype(float)
    invalid = detector_value_mask(None, raw, enabled=plan.mask_saturation)
    if invalid is not None:
        expected[invalid] = np.nan
    expected = _apply_thresholds(expected, plan)
    np.testing.assert_array_equal(condition_stitch_frame(raw, settings), expected)
    np.testing.assert_array_equal(raw, original)


def test_external_mask_orientation_and_strict_shape():
    raw = np.asarray([[0, 1, 0], [0, 0, np.nan]])
    mask = prepare_stitch_mask(raw, ImageOrientation(rotation=90), (3, 2))
    np.testing.assert_array_equal(mask, np.rot90(raw != 0))
    assert not mask.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        prepare_stitch_mask(raw, ImageOrientation(rotation=90), (2, 3))


@pytest.mark.parametrize("unit", ["q_A^-1", "2th_deg"])
@pytest.mark.parametrize("mode", ["auto", "off", "manual"])
def test_multigeometry_native_axis_auto_all_frames_and_conditioning(unit, mode):
    from xrd_tools.integrate.multi import create_multigeometry_integrators, stitch_1d

    shape = (195, 487)
    base = PONI(dist=.2, poni1=.015, poni2=.04, rot1=0, rot2=0, rot3=0,
                wavelength=1e-10, detector="Pilatus100k")
    images = [np.full(shape, value, dtype=np.uint16) for value in (10, 40)]
    images[0][20, 30] = 65535
    source = MemoryFrameSource([
        ScanFrame(index=i, image=image, metadata={"rot1": angle})
        for i, (image, angle) in enumerate(zip(images, (0., 25.), strict=True))
    ])
    mask = np.zeros(shape, dtype=bool)
    mask[10:20] = True
    conditioning = StitchConditioning(mode, 10.0 if mode == "manual" else None,
                                      40.0 if mode == "manual" else None)
    plan = StitchPlan(base_poni=base, npt_1d=48, unit=unit, method="full",
                      streaming_multigeometry=True, mask=mask,
                      conditioning=conditioning)
    captured = []
    result = run_stitch(plan, source, collect_frame_records=False,
                        frame_record_callback=lambda label, raw: captured.append((label, raw.dtype)))
    integrators = create_multigeometry_integrators(base, (0., 25.))
    expected_images = []
    for raw in images:
        image = raw.astype(float)
        invalid = detector_value_mask(None, raw, enabled=mode == "auto")
        if invalid is not None:
            image[invalid] = np.nan
        expected_images.append(_apply_thresholds(image, ReductionPlan(
            threshold_min=conditioning.threshold_min, threshold_max=conditioning.threshold_max,
        )))
    expected = stitch_1d(expected_images, integrators, npt=48, unit=unit,
                         method="full", mask=mask)
    assert result.payload.unit == unit
    np.testing.assert_allclose(result.payload.radial, expected.radial, rtol=1e-7)
    np.testing.assert_allclose(result.payload.intensity, expected.intensity, rtol=1e-6, equal_nan=True)
    assert captured == [(0, np.dtype("uint16")), (1, np.dtype("uint16"))]
    assert result.payload.radial[-1] > np.max(integrators[0].array_from_unit(unit=unit))
    companion = convert_radial_axis(result.payload.radial, unit,
                                    "2th_deg" if unit == "q_A^-1" else "q_A^-1", 1.)
    assert not np.allclose(np.diff(companion), np.diff(companion)[0], rtol=1e-5)
    restored = StitchPlan.from_provenance(plan.provenance(), base_poni=base, mask=mask)
    assert restored.conditioning == plan.conditioning


@pytest.mark.parametrize("unit", ["q_A^-1", "2th_deg"])
@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("mode", ["auto", "off", "manual"])
def test_xu_native_histogram_grid_all_member_range_and_int_policy(tmp_path, unit, auto, mode):
    from tests.core.test_xu_stitch import _receipt
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import (
        XuPowderQProvider, resolve_xu_stitch_effective_geometry, run_xu_hist_stitch_1d,
    )

    receipt = _receipt(tmp_path)
    shape = (195, 1475)
    raw = np.full(shape, 20, dtype=np.uint16)
    raw[20:30] = 65535
    raw[30:40] = 0
    frames = [ScanFrame(index=i, image=raw.copy(), metadata={"del": d, "nu": n, "mon": i + 1})
              for i, (d, n) in enumerate(((14., -10.), (25., 20.)))]
    source = MemoryFrameSource(frames)
    mask = np.zeros(shape, dtype=bool)
    mask[80:100] = True
    coordinates = []
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        provider = XuPowderQProvider(geometry, receipt.projection.value["acquisition"])
        wavelength_A = energy_eV_to_wavelength_m(geometry.projection.energy_eV) * 1e10
        for frame in frames:
            lease = provider.frame(nu=frame.metadata["nu"], del_=frame.metadata["del"])
            q = lease.q_magnitude().copy()
            lease.release()
            coordinates.append(q if unit == "q_A^-1" else 2 * np.degrees(np.arcsin(q * wavelength_A / (4 * np.pi))))
        static_mask = geometry.mask.copy() | mask
        solid_angle = geometry.solid_angle.copy()
    bounds = (min(float(a.min()) for a in coordinates), max(float(a.max()) for a in coordinates)) if auto else ((1., 7.) if unit == "q_A^-1" else (5., 70.))
    edges = np.linspace(*bounds, 49)
    sig = np.zeros(48)
    norm = np.zeros(48)
    count = np.zeros(48)
    conditioning = StitchConditioning(mode, 5.0 if mode == "manual" else None,
                                      20.0 if mode == "manual" else None)
    reference_image = raw.astype(float)
    invalid = detector_value_mask(None, raw, enabled=mode == "auto")
    if invalid is not None:
        reference_image[invalid] = np.nan
    reference_image = _apply_thresholds(reference_image, ReductionPlan(
        threshold_min=conditioning.threshold_min, threshold_max=conditioning.threshold_max,
    ))
    valid = ~static_mask & np.isfinite(reference_image)
    for i, axis in enumerate(coordinates):
        sig += np.histogram(axis[valid], edges, weights=raw[valid].astype(float) / (i + 1))[0]
        norm += np.histogram(axis[valid], edges, weights=solid_angle[valid])[0]
        count += np.histogram(axis[valid], edges)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = sig / norm
    captured = []
    result = run_xu_hist_stitch_1d(
        receipt, source, frame_indices=(0, 1), unit=unit,
        radial_range=None if auto else bounds, npt=48, monitor_key="mon",
        max_frame_bytes=4 * 1024 * 1024, conditioning=conditioning,
        mask=mask, frame_record_callback=lambda label, image: captured.append((label, image.dtype)),
    )
    assert result.payload.unit == unit
    np.testing.assert_allclose(result.payload.radial, (edges[1:] + edges[:-1]) / 2, rtol=1e-12)
    np.testing.assert_allclose(result.payload.intensity, expected, rtol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(result.diagnostics.coverage, count)
    np.testing.assert_allclose(result.diagnostics.normalization, norm, rtol=1e-12)
    assert captured == [(0, np.dtype("uint16")), (1, np.dtype("uint16"))]
    assert result.release_check_frame_count == 2
    assert result.runtime.restore_passed
