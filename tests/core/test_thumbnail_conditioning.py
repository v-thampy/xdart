"""Published thumbnails use the same native conditioning as real integration."""

from dataclasses import replace

import numpy as np
import pytest
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from xrd_tools.io import FrameViewReader
from xrd_tools.core.containers import PONI
from xrd_tools.reduction import (
    Frame, GIMode, Integration1DPlan, Integration2DPlan, NexusSink,
    MemorySink, ReductionPlan, Scan, run_reduction,
)


@pytest.mark.parametrize("gi", (False, True), ids=("standard", "grazing"))
@pytest.mark.parametrize("policy", ("manual", "off", "automatic"))
def test_published_thumbnail_matches_conditioning_without_changing_science(tmp_path, gi, policy):
    raw = np.resize(np.array([0, 5, 20, 50, 51, 4294967295], dtype=np.uint32), (32, 32)).copy()
    before = raw.tobytes()
    raw.setflags(write=False)
    mask = np.zeros(raw.shape, dtype=bool)
    mask[7, 7] = True
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=16),
        integration_2d=Integration2DPlan(npt_rad=16, npt_azim=8),
        gi=GIMode(incident_angle=.1) if gi else None,
        mask=mask, dark=np.full(raw.shape, 2.), flat=np.full(raw.shape, 2.),
        threshold_min=5 if policy == "manual" else None,
        threshold_max=50 if policy == "manual" else None,
        mask_saturation=policy == "automatic", saturation=50,
    )
    rejected = ((raw < 5) | (raw > 50) if policy == "manual"
                else raw >= 50 if policy == "automatic" else np.zeros(raw.shape, dtype=bool))
    expected_thumbnail = ((raw.astype(float) - 2) / 2).astype(np.float32)
    expected_thumbnail[mask | rejected] = np.nan

    def scan(image):
        return Scan("thumbnail", [Frame(0, image=image)],
                    poni=PONI(dist=.1, poni1=.001, poni2=.001, wavelength=1e-10, detector="Detector"),
                    integrator=AzimuthalIntegrator(
            dist=.1, poni1=.001, poni2=.001, wavelength=1e-10,
            detector=Detector(pixel1=100e-6, pixel2=100e-6, max_shape=raw.shape),
        ))

    conditioned = raw.astype(float)
    conditioned[rejected] = np.nan
    expected = run_reduction(replace(plan, threshold_min=None, threshold_max=None,
                                     mask_saturation=False), scan(conditioned), executor=False).frames[0]
    path = tmp_path / "published.nexus"
    memory = MemorySink()
    run_reduction(plan, scan(raw), (NexusSink(path, overwrite=True, thumbnail_max=64), memory),
                  executor=False)
    actual = memory.frames[0]
    with FrameViewReader(path) as reader:
        saved = reader.read(0)
    np.testing.assert_array_equal(np.isnan(saved.thumbnail), np.isnan(expected_thumbnail))
    # The persisted preview is quantized, unlike the scientific integrated arrays.
    np.testing.assert_allclose(saved.thumbnail, expected_thumbnail,
                               atol=max(1., float(np.nanmax(expected_thumbnail) - np.nanmin(expected_thumbnail))) / 255)
    for dimension in ("result_1d", "result_2d"):
        left, right = getattr(actual, dimension), getattr(expected, dimension)
        for name in ("radial", "intensity", "sigma"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    np.testing.assert_array_equal(actual.result_2d.azimuthal, expected.result_2d.azimuthal)
    assert raw.tobytes() == before


def test_direct_sink_thumbnail_fallback_keeps_manual_inclusive_bounds(tmp_path):
    from xrd_tools.reduction import FrameReduction

    raw = np.array([[0, 5, 50, 51]], dtype=np.uint32)
    reduction = FrameReduction(0)
    sink = NexusSink(tmp_path / "unused.nexus", thumbnail_max=64)
    sink._plan = ReductionPlan(threshold_min=5, threshold_max=50, integration_2d=None)
    sink.worker_process(Frame(0, image=raw), reduction)
    np.testing.assert_array_equal(reduction.thumbnail, [[np.nan, 5., 50., np.nan]])
    np.testing.assert_array_equal(raw, [[0, 5, 50, 51]])
