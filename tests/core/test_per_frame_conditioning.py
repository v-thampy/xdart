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


@pytest.mark.parametrize("execution", ("chunked", "streaming"))
@pytest.mark.parametrize("gi", (False, True), ids=("standard", "gi"))
def test_each_frame_matches_explicit_local_mask(tmp_path, execution, gi):
    shape = (100, 100)
    static = np.zeros(shape, dtype=bool)
    static[80, 90] = True
    raw = [np.full(shape, 20 + index, dtype=np.uint32) for index in range(3)]
    for index, image in enumerate(raw):
        image[90 + index, 95] = np.iinfo(np.uint32).max
        image[90 + index, 96] = 100_000
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
        plan, Scan("dynamic", [Frame(i, image=image) for i, image in enumerate(raw)],
                   poni=poni, integrator=poni_to_integrator(poni)),
        NexusSink(target, overwrite=True, thumbnail_max=128),
        execution=execution, executor=2 if execution == "streaming" else False,
        chunk_size=1,
    )
    assert actual.n_processed == 3
    for index, image in enumerate(raw):
        # The reference explicitly excludes this frame's sentinel, with the
        # value-mask toggle OFF. It cannot inherit a session's seeded mask.
        local_mask = static | (image >= np.iinfo(np.uint32).max)
        reference = run_reduction(
            replace(plan, mask=local_mask, mask_saturation=False),
            Scan("reference", [Frame(index, image=image)], poni=poni,
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
