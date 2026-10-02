"""FIT-TOOLS-2026-10 oracle row 4: the 2-D extension (rotation, several seeded peaks, planar background).

Truth is generated here from the profile definitions with explicit rotations, independent of the fitter.
Frozen tolerances (row 3's): centres within 0.05 bin (local spacing), principal FWHM within 1 %, evaluated
surface RMS within 1e-4 of the dynamic range. Declared here: rotation within 0.5 degree when identifiable.
Components are matched to their declared seed IDs.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.analysis.fitting.region import PeakSeed, RegionAxis, RegionFitRefused, fit_2d_region

LN2 = np.log(2.0)
KAPPA = np.sqrt(2.0) - 1.0
X = np.linspace(2.0, -1.0, 41)                                   # descending
Y = 0.04 * np.arange(33) + 0.0008 * np.arange(33) ** 2            # nonuniform, 33 != 41
GX, GY = np.meshgrid(X, Y, indexing="ij")                         # data order (x, y)
PLANE = 1.0 + 0.3 * GX - 0.2 * GY


def _rho2(cx, cy, fwhm_major, fwhm_minor, angle_deg, gx=GX, gy=GY):
    """Squared distance in half widths along the rotated principal axes (1 on the half-height contour)."""
    theta = np.radians(angle_deg)
    u = (gx - cx) * np.cos(theta) + (gy - cy) * np.sin(theta)
    v = -(gx - cx) * np.sin(theta) + (gy - cy) * np.cos(theta)
    return (u / (fwhm_major / 2)) ** 2 + (v / (fwhm_minor / 2)) ** 2


def _gaussian(height, *shape, **grid):
    return height * np.exp(-LN2 * _rho2(*shape, **grid))


def _pseudo_voigt(height, fraction, *shape, **grid):
    rho2 = _rho2(*shape, **grid)
    return height * ((1 - fraction) * np.exp(-LN2 * rho2) + fraction / (1 + KAPPA * rho2) ** 2)


PEAKS = {"left": (10.0, 0.0, 0.8, 0.6, 0.25, 30.0), "right": (6.0, 1.2, 1.4, 0.5, 0.3, -50.0)}


def _local_spacing(axis, value):
    index = int(np.argmin(np.abs(axis - value)))
    neighbours = [abs(axis[i] - axis[index]) for i in (index - 1, index + 1) if 0 <= i < axis.size]
    return min(neighbours)


def _assert_component(component, height, cx, cy, fwhm_major, fwhm_minor, angle):
    assert component.height.value == pytest.approx(height, rel=1e-2)
    assert abs(component.center[0].value - cx) <= 0.05 * _local_spacing(X, cx)
    assert abs(component.center[1].value - cy) <= 0.05 * _local_spacing(Y, cy)
    assert [item.value for item in component.principal_fwhm] == pytest.approx([fwhm_major, fwhm_minor], rel=1e-2)
    assert component.rotation_identifiable is True
    assert abs(component.rotation_deg.value - angle) <= 0.5


def _assert_surface(outcome, truth):
    model = outcome.model()
    assert model.shape == truth.shape == (41, 33)
    assert np.sqrt(np.mean((model - truth) ** 2)) <= 1e-4 * (truth.max() - truth.min())


SEEDS = (PeakSeed("left", (0.05, 0.75)), PeakSeed("right", (1.15, 1.45)))


@pytest.mark.parametrize("masked", [False, True], ids=["full", "masked"])
def test_two_rotated_peaks_on_a_plane_match_their_seeds(masked):
    truth = PLANE + sum(_gaussian(*peak) for peak in PEAKS.values())
    data = truth.copy()
    mask = None
    if masked:
        data[:6, :5] = np.nan                     # an all-NaN corner block
        mask = np.zeros(data.shape, dtype=bool)
        mask[20:23, :] = True                     # an excluded stripe (True = excluded)
    outcome = fit_2d_region(data, RegionAxis("qx", X, "1/A"), RegionAxis("qz", Y, "1/A"), seeds=SEEDS, mask=mask)
    assert outcome.diagnostics.success
    assert [component.id for component in outcome.components] == ["left", "right"]
    for component in outcome.components:
        _assert_component(component, *PEAKS[component.id])
    _assert_surface(outcome, truth)
    # the plane, evaluated apart: offset at the box centre and slopes per native unit
    reference = outcome.background.reference
    assert outcome.background.offset.value == pytest.approx(1.0 + 0.3 * reference[0] - 0.2 * reference[1], abs=1e-3)
    assert [item.value for item in outcome.background.slopes] == pytest.approx([0.3, -0.2], abs=1e-3)
    # outcomes carry their axes, units, shape order and the samples used
    assert [(axis.name, axis.unit) for axis in outcome.axes] == [("qx", "1/A"), ("qz", "1/A")]
    assert outcome.shape == (41, 33)
    expected_valid = np.isfinite(data) & (~mask if masked else True)
    np.testing.assert_array_equal(outcome.valid, expected_valid)
    assert outcome.diagnostics.n_used == int(expected_valid.sum())


def test_rotated_pseudo_voigt_recovers_fraction_and_common_hwhm():
    peak = (1.2, 0.7, 0.55, 0.3, 65.0)
    truth = PLANE + _pseudo_voigt(8.0, 0.4, *peak)
    outcome = fit_2d_region(truth, X, Y, profile="pseudovoigt", seeds=(PeakSeed("p", (1.1, 0.75)),))
    (component,) = outcome.components
    _assert_component(component, 8.0, *peak)
    assert component.fraction.value == pytest.approx(0.4, abs=1e-3)
    _assert_surface(outcome, truth)


def test_circular_peak_reports_an_unidentifiable_rotation():
    truth = PLANE + _gaussian(5.0, 0.6, 1.0, 0.45, 0.45, 0.0)
    outcome = fit_2d_region(truth, X, Y)  # the default seed: one peak at the data maximum
    (component,) = outcome.components
    assert component.id == "p0"
    assert [item.value for item in component.principal_fwhm] == pytest.approx([0.45, 0.45], rel=1e-2)
    assert component.rotation_identifiable is False
    assert component.rotation_deg is None
    _assert_surface(outcome, truth)


def test_data_xy_order_against_displayed_yx():
    truth = PLANE + _gaussian(9.0, -0.4, 1.6, 0.3, 0.2, 10.0)
    outcome = fit_2d_region(truth, RegionAxis("x", X), RegionAxis("y", Y))
    shown = outcome.comparison(truth).displayed()
    assert [axis.name for axis in shown.axes] == ["y", "x"]          # rows are y, columns are x
    assert shown.model.shape == (33, 41)
    row, column = np.unravel_index(np.argmax(shown.model - PLANE.T), shown.model.shape)
    assert (row, column) == (int(np.argmin(np.abs(Y - 1.6))), int(np.argmin(np.abs(X + 0.4))))
    np.testing.assert_array_equal(shown.data, truth.T)
    np.testing.assert_allclose(shown.residual, truth.T - shown.model, rtol=0, atol=1e-12)


def test_the_2d_analyzer_wraps_the_same_fit():
    from xrd_tools.analysis.fitting.region_analyzers import REGION_2D_KEY, Region2DFitAnalyzer, Region2DInput
    from xrd_tools.analysis.runner import AnalysisInput

    truth = PLANE + sum(_gaussian(*peak) for peak in PEAKS.values())
    region = Region2DInput(truth, RegionAxis("qx", X), RegionAxis("qz", Y))
    inp = AnalysisInput("slice-3", np.empty(0), np.empty(0), metadata={REGION_2D_KEY: region})
    analyzed = Region2DFitAnalyzer(seeds=SEEDS).analyze(inp)
    assert analyzed.ok and analyzed.overlay is None
    assert analyzed.params == fit_2d_region(truth, region.x, region.y, seeds=SEEDS).flat_params()
    assert {"left.angle_deg", "right.fwhm_major", "bg.slope_qz"} <= set(analyzed.params)
    weighted = Region2DFitAnalyzer(seeds=SEEDS, use_uncertainties=True).analyze(inp)
    assert not weighted.ok and "WEIGHTS_UNUSABLE" in weighted.message


def test_empty_and_insufficient_regions_are_refused_by_name():
    with pytest.raises(RegionFitRefused) as empty:
        fit_2d_region(np.full((41, 33), np.nan), X, Y)
    assert empty.value.code == "REGION_EMPTY"
    sparse = np.full((41, 33), np.nan)
    sparse[:2, :4] = 1.0
    with pytest.raises(RegionFitRefused) as few:
        fit_2d_region(sparse, X, Y)
    assert few.value.code == "INSUFFICIENT_SAMPLES"
    with pytest.raises(RegionFitRefused) as outside:
        fit_2d_region(PLANE, X, Y, seeds=(PeakSeed("far", (5.0, 1.0)),))
    assert outside.value.code == "SEED_OUTSIDE_REGION"
    with pytest.raises(ValueError, match="shape"):
        fit_2d_region(PLANE.T, X, Y)
