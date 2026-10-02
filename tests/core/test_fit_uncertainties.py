"""Optional measured uncertainties through the public displayed fitting policy."""
from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.runner import AnalysisInput, DisplayedPeakFitAnalyzer
from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
from xrd_tools.analysis.plans import PeakFitPlan, run_peak_fit


def snapshot(sigma=None, mask=None):
    x = np.linspace(1, 4, 301)
    y = 10 + 400 * np.exp(-.5 * ((x - 2.5) / .06) ** 2)
    y += np.random.default_rng(14).normal(size=x.size) * np.where(x < 2.5, 1, 15)
    return FitInputSnapshot(FitInputKey('scan', '/scan.nexus', 'admitted', 1),
                            (FitAxis('q', 'A^-1', x),), y, mask, sigma=sigma)


def test_owned_sigma_is_optional_and_only_weighting_changes_the_default_fit():
    sigma = np.where(np.linspace(1, 4, 301) < 2.5, 1., 15.)
    s = snapshot(sigma)
    sigma[:] = 999
    assert s.sigma[0] == 1 and not s.sigma.flags.writeable
    assert not s.trace_input().sigma.flags.writeable
    settings = dict(model='gaussian', background='constant', selection_mode='count',
                    manual_centers=(2.5,), n_peaks=1)
    default = DisplayedPeakFitAnalyzer(settings).analyze(s)
    bare = DisplayedPeakFitAnalyzer(settings).analyze(snapshot())
    assert default.result.parameter_values == bare.result.parameter_values
    weighted = DisplayedPeakFitAnalyzer({**settings, 'use_uncertainties': True}).analyze(s)
    direct = run_peak_fit(PeakFitPlan(positions=(2.5,), n_peaks=1, model='gaussian', background='constant',
                                    fit_kwargs={'weights': 1 / s.sigma}), s.axes[0].values, s.values).payload
    assert weighted.ok and weighted.result.parameter_values == tuple(float(v.value) for v in direct.params.values())
    assert weighted.result.parameter_values != default.result.parameter_values
    np.testing.assert_array_equal(weighted.result.residual, s.values - weighted.result.fit)
    inp = AnalysisInput('scan', s.axes[0].values, s.values, s.sigma, x_unit='A^-1')
    assert DisplayedPeakFitAnalyzer({**settings, 'use_uncertainties': True}).analyze(inp).result.parameter_values == weighted.result.parameter_values


def test_weighted_refusals_use_only_the_admitted_fitted_samples():
    settings = dict(use_uncertainties=True, selection_mode='count', manual_centers=(2.5,))
    result = run_displayed_peak_fit(DisplayedPeakFitPlan(snapshot().trace_input(), **settings))
    assert result.code == 'FIT_UNCERTAINTIES_REQUIRED'
    sigma = np.ones(301)
    sigma[0] = np.nan
    bad = snapshot(sigma)
    assert run_displayed_peak_fit(DisplayedPeakFitPlan(bad.trace_input(), **settings)).code == 'INVALID_FIT_UNCERTAINTIES'
    mask = np.ones(301, bool)
    mask[0] = False
    assert DisplayedPeakFitAnalyzer(settings).analyze(snapshot(sigma, mask)).ok
    assert run_displayed_peak_fit(DisplayedPeakFitPlan(bad.trace_input(), fit_bounds=(1.1, 3.9), **settings)).code == 'OK'
    with pytest.raises(ValueError, match='sigma'):
        snapshot(np.ones(300))


def test_absent_sigma_retains_trace_and_plan_fingerprints():
    s = snapshot()
    assert s.sigma is None and s.trace_input().sigma is None
    p = DisplayedPeakFitPlan(s.trace_input(), selection_mode='count', manual_centers=(2.5,))
    assert run_displayed_peak_fit(p).result_fingerprint == run_displayed_peak_fit(replace(p, use_uncertainties=False)).result_fingerprint
