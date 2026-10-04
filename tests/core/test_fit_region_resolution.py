"""Principal model-shape resolution is independent of axis order and units."""
from itertools import permutations

import numpy as np
import pytest

from xrd_tools.analysis.fitting.region import (
    PeakSeed, RegionAxis, RegionFitRefused, fit_2d_region, fit_3d_region,
)


def _truth(ndim, resolved):
    axes = [np.arange(49.) - 24., np.arange(13.) - 6.]
    rho = np.sqrt(.99) if not resolved else .96
    covariance = np.array([[36., 9. * rho], [9. * rho, 2.25]])
    center = [.3, -.2]
    if ndim == 3:
        axes.append(np.arange(11.) - 5.)
        extended = np.zeros((3, 3))
        extended[:2, :2] = covariance
        extended[2, 2] = 4.
        covariance = extended
        center.append(.1)
    grid = np.meshgrid(*axes, indexing="ij")
    delta = np.stack([value - c for value, c in zip(grid, center)], axis=-1)
    data = 1. + 10. * np.exp(-.5 * np.einsum("...i,ij,...j->...", delta,
                                           np.linalg.inv(covariance), delta))
    return axes, center, covariance, data


def _fit(ndim, order, resolved, scales=None):
    axes, center, covariance, data = _truth(ndim, resolved)
    scales = np.ones(ndim) if scales is None else np.asarray(scales)
    axes = [value * factor for value, factor in zip(axes, scales)]
    center = np.asarray(center) * scales
    covariance *= np.outer(scales, scales)
    names = tuple(RegionAxis("xyz"[i], axes[i]) for i in order)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(np.diag(covariance))
    seed = PeakSeed("streak", tuple(center[i] for i in order),
                    fwhm=tuple(fwhm[i] for i in order))
    kwargs = {"seeds": (seed,), "background": "constant"}
    if ndim == 2:
        outcome = fit_2d_region(data.transpose(order), *names, **kwargs)
    else:
        transform = np.diag(1 / scales)[:, order]
        outcome = fit_3d_region(data.transpose(order), names,
                                reciprocal_transform=transform, **kwargs)
    return outcome, covariance[np.ix_(order, order)], data.transpose(order)


@pytest.mark.parametrize("order", list(permutations(range(2))))
def test_historic_undersampled_streak_has_named_refusal_in_both_orders(order):
    with pytest.raises(RegionFitRefused) as refused:
        _fit(2, order, False)
    assert refused.value.code == "PEAK_UNDERSAMPLED"
    assert "streak" in str(refused.value)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_undersampled_volume_has_the_same_refusal_in_all_orders(order):
    with pytest.raises(RegionFitRefused) as refused:
        _fit(3, order, False)
    assert refused.value.code == "PEAK_UNDERSAMPLED"
    assert "streak" in str(refused.value)


@pytest.mark.parametrize("ndim,order", [(n, p) for n in (2, 3) for p in permutations(range(n))])
def test_resolved_correlated_peak_keeps_covariance_and_model_in_all_orders(ndim, order):
    outcome, covariance, truth = _fit(ndim, order, True)
    assert outcome.diagnostics.success
    np.testing.assert_allclose(outcome.components[0].covariance, covariance, rtol=1e-5, atol=1e-9)
    assert np.sqrt(np.mean((outcome.model() - truth) ** 2)) < 1e-4 * np.ptp(truth)


@pytest.mark.parametrize("resolved", [False, True])
def test_resolution_decision_is_invariant_to_axis_unit_rescaling(resolved):
    if resolved:
        outcome, covariance, _truth_data = _fit(2, (1, 0), True, [1e-3, 1e5])
        np.testing.assert_allclose(outcome.components[0].covariance, covariance, rtol=1e-5, atol=1e-9)
    else:
        with pytest.raises(RegionFitRefused) as refused:
            _fit(2, (1, 0), False, [1e-3, 1e5])
        assert refused.value.code == "PEAK_UNDERSAMPLED"


@pytest.mark.parametrize("ndim", [2, 3])
def test_real_region_analyzers_preserve_the_resolution_refusal(ndim):
    from xrd_tools.analysis.fitting.region_analyzers import (
        REGION_2D_KEY, REGION_3D_KEY, Region2DInput, Region3DInput,
        Region2DFitAnalyzer, Region3DFitAnalyzer,
    )
    from xrd_tools.analysis.runner import AnalysisInput
    axes, center, covariance, data = _truth(ndim, False)
    axes = tuple(RegionAxis("xyz"[i], values) for i, values in enumerate(axes))
    seed = PeakSeed("streak", tuple(center),
        fwhm=tuple(2 * np.sqrt(2 * np.log(2)) * np.sqrt(np.diag(covariance))))
    region = Region2DInput(data, *axes) if ndim == 2 else Region3DInput(data, axes)
    key = REGION_2D_KEY if ndim == 2 else REGION_3D_KEY
    analyzer = (Region2DFitAnalyzer if ndim == 2 else Region3DFitAnalyzer)(
        seeds=(seed,), background="constant")
    outcome = analyzer.analyze(AnalysisInput("region", np.empty(0), np.empty(0), metadata={key: region}))
    assert not outcome.ok and outcome.result is None
    assert "PEAK_UNDERSAMPLED" in outcome.message and "streak" in outcome.message
