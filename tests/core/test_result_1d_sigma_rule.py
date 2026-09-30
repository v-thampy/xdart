"""The 1-D uncertainty rule equals what each pyFAI producer actually returns.

The light-1D publication reserves an uncertainty buffer by ``result_1d_has_sigma``;
a mismatch with the producer fails every frame. Each case runs one real frame
through the public ``run_reduction``.
"""
import itertools

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.reduction import Frame, GIMode, Integration1DPlan, ReductionPlan, Scan, run_reduction
from xrd_tools.reduction.core import result_1d_has_sigma

SHAPE = (195, 487)
_PONI = PONI(dist=0.15, poni1=0.002, poni2=0.04, wavelength=1.0e-10, detector="Pilatus100k")
_IMAGE = 50 + np.random.default_rng(7).poisson(40, SHAPE).astype(float)
_OUTPUTS = {
    "q": (None, {}),
    "2th": (None, {"unit": "2th_deg"}),
    "chi": (None, {"unit": "chi_deg", "radial_range": (0.5, 2.0)}),
    "gi_q_total": ("q_total", {}),
    "gi_q_total_wedge": ("q_total", {"azimuth_range": (-60.0, 60.0)}),
    "gi_q_ip": ("q_ip", {}),
    "gi_q_oop": ("q_oop", {}),
    "gi_exit_angle": ("exit_angle", {}),
    "gi_chi_gi": ("chi_gi", {}),
}


def _reduce(integration, gi_mode=None, backend="cython"):
    plan = ReductionPlan(
        integration_1d=integration, integration_2d=None,
        gi=None if gi_mode is None else GIMode(incident_angle=0.2, mode_1d=gi_mode, method=backend))
    scan = Scan("rule", [Frame(0, image=_IMAGE.copy(), metadata={})], poni=_PONI,
                integrator=poni_to_integrator(_PONI))
    result = run_reduction(plan, scan, executor=1)
    assert not result.failed, result.error
    return plan, result.frames[0].result_1d


def _cases():
    for (name, (gi_mode, kwargs)), model, variance, backend in itertools.product(
            _OUTPUTS.items(), (None, "no", "poisson", "azimuthal"), (False, True), ("cython", "no")):
        if gi_mode is None and backend == "no":
            continue
        if backend == "no" and model == "azimuthal" and not variance:
            continue  # pyFAI's numpy histogram cannot assess azimuthal variance; it raises
        yield pytest.param(gi_mode, kwargs, model, variance, backend,
                           id=f"{name}-{backend}-{model}-{'variance' if variance else 'plain'}")


@pytest.mark.parametrize(("gi_mode", "kwargs", "model", "variance", "backend"), list(_cases()))
def test_rule_matches_every_producer(gi_mode, kwargs, model, variance, backend):
    extra = {"variance": np.full(SHAPE, 45.0)} if variance else {}
    plan, result = _reduce(Integration1DPlan(npt=40, error_model=model, extra=extra, **kwargs),
                           gi_mode, backend)
    assert (result.sigma is not None) is result_1d_has_sigma(plan)


def test_explicit_variance_keeps_its_sigma_through_run_reduction():
    from xrd_tools.integrate.single import integrate_1d
    variance = np.full(SHAPE, 45.0)
    plan, result = _reduce(Integration1DPlan(npt=40, extra={"variance": variance}))
    direct = integrate_1d(_IMAGE, poni_to_integrator(_PONI), npt=40,
                          method=plan.integration_1d.method, variance=variance)
    assert result.sigma is not None
    np.testing.assert_allclose(result.sigma, direct.sigma, rtol=1e-12)
