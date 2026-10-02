"""FIT-TOOLS-2026-10 oracle row 10: Codex's 3-D rows 1-6 (REVIEW.md "Additional finite oracle"), headless parts.

Truth comes from an independent generator: Gaussians written with explicit full covariance matrices (nonzero cross
terms) and an affine background. The public path is importer (``RSMVolume.from_axes``) -> extractor and fitter
(``fit_volume_region``) -> outcome -> comparison slice. Frozen tolerances: centres within 0.05 bin per native axis,
covariance relative Frobenius error within 1 %, principal physical FWHM within 1 % with a supplied transform,
evaluated model RMS within 1e-4 of the peak dynamic range.
"""
from __future__ import annotations

import threading
import tracemalloc

import numpy as np
import pytest

from xrd_tools.analysis.fitting.region import (
    PeakSeed, RegionAxis, RegionFitCancelled, RegionFitRefused, fit_3d_region, fit_volume_region,
)
from xrd_tools.analysis.fitting.region_analyzers import REGION_3D_KEY, Region3DFitAnalyzer, Region3DInput
from xrd_tools.analysis.runner import AnalysisInput
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.rsm.volume import RSMVolume

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
#: A non-orthogonal reciprocal transform (native H/K/L -> Q, 2pi included), like a UB with a tilted cell.
J = np.array([[1.61, 0.12, -0.05], [0.03, 1.58, 0.21], [-0.08, 0.04, 1.62]])


def _gaussian(grid, height, centre, covariance):
    """height * exp(-1/2 d^T C^-1 d) on a meshgrid tuple (independent of the fitter)."""
    d = np.stack([axis - c for axis, c in zip(grid, centre)], axis=-1)
    inverse = np.linalg.inv(np.asarray(covariance, dtype=float))
    return height * np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, inverse, d))


def _affine(grid, offset, slopes, reference):
    return offset + sum(s * (axis - r) for s, axis, r in zip(slopes, grid, reference))


def _volume(axes, intensity, frame=RSMCoordinateFrame.HKL):
    return RSMVolume.from_axes(frame, tuple(zip(frame.axis_names, axes)), intensity)


def _frobenius(fitted, truth):
    return np.linalg.norm(np.asarray(fitted) - truth) / np.linalg.norm(truth)


def _spacing(axis, value):
    index = int(np.argmin(np.abs(axis - value)))
    return min(abs(axis[i] - axis[index]) for i in (index - 1, index + 1) if 0 <= i < axis.size)


def _assert_centre(component, axes, centre):
    for estimate, axis, value in zip(component.center, axes, centre):
        assert abs(estimate.value - value) <= 0.05 * _spacing(axis, value)


SIGMA_1 = np.array([[0.0100, 0.0040, -0.0020], [0.0040, 0.0200, 0.0050], [-0.0020, 0.0050, 0.0150]])
CENTRE_1 = (1.03, -0.02, 2.98)
H1 = CENTRE_1[0] + np.linspace(-0.45, 0.55, 17)
K1 = CENTRE_1[1] + np.linspace(-0.60, 0.75, 19)
L1 = CENTRE_1[2] + np.linspace(-0.55, 0.50, 23)


def _row1_truth():
    grid = np.meshgrid(H1, K1, L1, indexing="ij")
    reference = tuple((axis[0] + axis[-1]) / 2 for axis in (H1, K1, L1))
    return grid, _gaussian(grid, 50.0, CENTRE_1, SIGMA_1) + _affine(grid, 2.0, (1.5, -0.8, 0.6), reference)


def test_row1_correlated_anisotropic_gaussian_on_an_affine_background():
    _grid, truth = _row1_truth()
    assert truth.shape == (17, 19, 23)
    outcome = fit_volume_region(_volume((H1, K1, L1), truth), reciprocal_transform=J,
                                seeds=(PeakSeed("peak", (1.0, 0.0, 3.0)),))
    assert outcome.diagnostics.success
    (component,) = outcome.components
    _assert_centre(component, (H1, K1, L1), CENTRE_1)
    assert _frobenius(component.covariance, SIGMA_1) <= 1e-2
    np.testing.assert_allclose([item.value for item in component.marginal_fwhm],
                               FWHM_PER_SIGMA * np.sqrt(np.diag(SIGMA_1)), rtol=1e-2)
    # physical principal widths from the supplied transform, against an independent eigen-decomposition
    truth_q = np.sort(np.linalg.eigvalsh(J @ SIGMA_1 @ J.T))[::-1]
    np.testing.assert_allclose([item.value for item in component.physical.principal_fwhm],
                               FWHM_PER_SIGMA * np.sqrt(truth_q), rtol=1e-2)
    assert all(component.physical.identifiable)
    assert [axis.name for axis in outcome.axes] == ["h", "k", "l"] and outcome.shape == (17, 19, 23)


def test_row2_two_correlated_components_keep_their_seed_ids():
    h = np.linspace(0.0, 2.0, 21)
    k = np.linspace(-1.0, 1.2, 25)
    l = np.linspace(2.0, 3.5, 18)
    grid = np.meshgrid(h, k, l, indexing="ij")
    peaks = {
        "a": (40.0, (0.55, -0.35, 2.55), np.array([[0.012, 0.004, 0.0], [0.004, 0.018, -0.003], [0.0, -0.003, 0.010]])),
        "b": (25.0, (1.45, 0.60, 3.00), np.array([[0.015, -0.005, 0.002], [-0.005, 0.012, 0.0], [0.002, 0.0, 0.020]])),
    }
    reference = (1.0, 0.1, 2.75)
    background = _affine(grid, 3.0, (0.5, 1.0, -0.4), reference)
    truth = background + sum(_gaussian(grid, *peak) for peak in peaks.values())
    seeds = (PeakSeed("b", (1.4, 0.65, 3.05)), PeakSeed("a", (0.6, -0.3, 2.5)))  # declared order is kept
    outcome = fit_volume_region(_volume((h, k, l), truth), seeds=seeds)
    assert [component.id for component in outcome.components] == ["b", "a"]
    for component in outcome.components:
        height, centre, covariance = peaks[component.id]
        _assert_centre(component, (h, k, l), centre)
        assert _frobenius(component.covariance, covariance) <= 1e-2
        assert component.height.value == pytest.approx(height, rel=1e-2)
        # the reported integral against an independent quadrature of the truth component (fine grid, +-6 sigma)
        axes = [np.linspace(c - 6 * s, c + 6 * s, 81) for c, s in zip(centre, np.sqrt(np.diag(covariance)))]
        fine = np.meshgrid(*axes, indexing="ij")
        quadrature = _gaussian(fine, height, centre, covariance).sum() * np.prod([a[1] - a[0] for a in axes])
        assert component.integral.value == pytest.approx(quadrature, rel=1e-2)
    model = outcome.model()
    peak_range = (truth - background).max() - (truth - background).min()
    assert np.sqrt(np.mean((model - truth) ** 2)) <= 1e-4 * peak_range
    assert [item.value for item in outcome.background.slopes] == pytest.approx([0.5, 1.0, -0.4], abs=1e-3)
    assert outcome.background.offset.value == pytest.approx(
        _affine(tuple(np.array([r]) for r in outcome.background.reference), 3.0, (0.5, 1.0, -0.4), reference)[0],
        abs=1e-3)


def test_row3_missing_voxels_nonuniform_spacing_refusals_and_axis_permutation():
    h = H1[0] + (H1[-1] - H1[0]) * np.linspace(0, 1, 17) ** 1.2   # nonuniform
    grid = np.meshgrid(h, K1, L1, indexing="ij")
    reference = tuple((axis[0] + axis[-1]) / 2 for axis in (h, K1, L1))
    truth = _gaussian(grid, 50.0, CENTRE_1, SIGMA_1) + _affine(grid, 2.0, (1.5, -0.8, 0.6), reference)
    data = truth.copy()
    rng = np.random.default_rng(11)
    data[rng.random(data.shape) < 0.15] = np.nan                    # deterministic missing voxels
    data[:, :, -3:] = np.nan                                         # an uncovered slab
    seed = PeakSeed("peak", (1.0, 0.0, 3.0))
    outcome = fit_volume_region(_volume((h, K1, L1), data), reciprocal_transform=J, seeds=(seed,))
    (component,) = outcome.components
    _assert_centre(component, (h, K1, L1), CENTRE_1)
    assert _frobenius(component.covariance, SIGMA_1) <= 1e-2
    np.testing.assert_array_equal(outcome.valid, np.isfinite(data))

    # the same data with its axes permuted to (l, h, k) and J's columns with them: native results permute,
    # the physical fit does not
    axes = tuple(RegionAxis(name, values) for name, values in zip("hkl", (h, K1, L1)))
    order = (2, 0, 1)
    direct = fit_3d_region(data, axes, reciprocal_transform=J, seeds=(seed,))
    permuted = fit_3d_region(np.transpose(data, order), tuple(axes[i] for i in order),
                             reciprocal_transform=J[:, order],
                             seeds=(PeakSeed("peak", tuple(seed.center[i] for i in order)),))
    (mine,), (other,) = direct.components, permuted.components
    assert [axis.name for axis in permuted.axes] == ["l", "h", "k"]
    np.testing.assert_allclose([e.value for e in other.center], [mine.center[i].value for i in order], rtol=1e-6)
    np.testing.assert_allclose(other.covariance, mine.covariance[np.ix_(order, order)], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(other.physical.covariance_q, mine.physical.covariance_q, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose([e.value for e in other.physical.center_q],
                               [e.value for e in mine.physical.center_q], rtol=1e-6)

    with pytest.raises(RegionFitRefused) as empty:
        fit_3d_region(np.full(data.shape, np.nan), axes)
    assert empty.value.code == "REGION_EMPTY"
    sparse = np.full(data.shape, np.nan)
    sparse[:2, :2, :4] = 1.0
    with pytest.raises(RegionFitRefused) as few:
        fit_3d_region(sparse, axes)
    assert few.value.code == "INSUFFICIENT_SAMPLES"
    with pytest.raises(RegionFitRefused) as weights:
        fit_3d_region(truth, axes, sigma=np.zeros(truth.shape))
    assert weights.value.code == "WEIGHTS_UNUSABLE"


def test_row4_non_orthogonal_transform_without_transform_and_degenerate_directions():
    _grid, truth = _row1_truth()
    with_j = fit_volume_region(_volume((H1, K1, L1), truth), reciprocal_transform=J)
    (component,) = with_j.components
    native_centre = np.array([item.value for item in component.center])
    np.testing.assert_allclose([item.value for item in component.physical.center_q], J @ native_centre,
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(component.physical.covariance_q, J @ component.covariance @ J.T,
                               rtol=1e-12, atol=1e-15)
    assert component.physical.integral.value == pytest.approx(component.integral.value * abs(np.linalg.det(J)),
                                                              rel=1e-12)
    without = fit_volume_region(_volume((H1, K1, L1), truth))
    assert without.components[0].physical is None
    assert without.physical_reason == "RECIPROCAL_TRANSFORM_REQUIRED"

    # Q-space truth with two equal principal widths: those two directions are unidentifiable, the third is not
    rotation = np.linalg.qr(np.array([[1.0, 0.3, -0.2], [0.1, 1.0, 0.4], [0.2, -0.5, 1.0]]))[0]
    sigma_q = rotation @ np.diag([0.030, 0.030, 0.012]) @ rotation.T
    inverse = np.linalg.inv(J)
    sigma_hkl = inverse @ sigma_q @ inverse.T
    grid = np.meshgrid(H1, K1, L1, indexing="ij")
    degenerate = fit_volume_region(
        _volume((H1, K1, L1), _gaussian(grid, 30.0, CENTRE_1, sigma_hkl) + 1.0), reciprocal_transform=J)
    physical = degenerate.components[0].physical
    assert physical.identifiable == (False, False, True)
    assert physical.principal_directions[0] is None and physical.principal_directions[1] is None
    direction = physical.principal_directions[2]
    assert abs(abs(direction @ rotation[:, 2]) - 1.0) <= 1e-3


def test_row4_cartesian_q_frame_is_its_own_metric():
    _grid, truth = _row1_truth()
    q_volume = _volume((H1, K1, L1), truth, RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)
    (component,) = fit_volume_region(q_volume).components
    np.testing.assert_allclose(component.physical.covariance_q, component.covariance, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        fit_volume_region(q_volume, reciprocal_transform=J)


def test_row5_comparison_slices_share_plane_index_and_axes():
    _grid, truth = _row1_truth()
    volume = _volume((H1, K1, L1), truth)
    outcome = fit_volume_region(volume, box={"h": (0.7, 1.4), "l": (2.6, 3.3)})
    assert [axis.values.size for axis in outcome.axes] == [int(((H1 >= 0.7) & (H1 <= 1.4)).sum()), 19,
                                                          int(((L1 >= 2.6) & (L1 <= 3.3)).sum())]
    box = volume.crop_by_axes({"h": (0.7, 1.4), "l": (2.6, 3.3)}).intensity
    plane = outcome.comparison_slice(box, "k", 7)
    assert plane.fixed == ("k", 7, K1[7])
    assert [axis.name for axis in plane.axes] == ["h", "l"]
    assert plane.data.shape == plane.model.shape == plane.residual.shape == plane.valid.shape
    assert plane.data.shape[0] != plane.data.shape[1]  # non-square
    np.testing.assert_array_equal(plane.data, box[:, 7, :])
    np.testing.assert_allclose(plane.model, outcome.model()[:, 7, :], rtol=0, atol=1e-12)
    np.testing.assert_allclose(plane.residual, plane.data - plane.model, rtol=0, atol=0)
    shown = plane.displayed()
    assert [axis.name for axis in shown.axes] == ["l", "h"]
    np.testing.assert_array_equal(shown.residual, plane.residual.T)
    # a signed linear residual: an off-model perturbation keeps its sign whatever the display does
    bumped = box.copy()
    bumped[2, 7, 3] -= 5.0
    assert outcome.comparison_slice(bumped, "k", 7).residual[2, 3] < -4.0


def test_row6_caps_refuse_before_allocation_and_the_capped_fit_meets_its_budget():
    big = RSMVolume(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 11),
                    np.ones((100, 100, 11)))
    tracemalloc.start()
    try:
        with pytest.raises(RegionFitRefused) as large:
            fit_volume_region(big)
        refused_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert large.value.code == "REGION_TOO_LARGE" and "narrow" in str(large.value)
    assert refused_peak < 1 << 20, refused_peak  # no mesh, box copy or Jacobian before the refusal
    seeds = tuple(PeakSeed(f"s{i}", (0.2 + 0.2 * i, 0.5, 0.5)) for i in range(4))
    with pytest.raises(RegionFitRefused) as many:
        fit_volume_region(big, box={"l": (0.0, 0.25)}, seeds=seeds)
    assert many.value.code == "TOO_MANY_COMPONENTS"

    # the cap: 40 x 50 x 50 = 100,000 voxels, three components, 34 free parameters
    h, k, l = np.linspace(0.0, 2.0, 40), np.linspace(-1.0, 1.0, 50), np.linspace(2.0, 3.0, 50)
    grid = np.meshgrid(h, k, l, indexing="ij")
    covariance = np.array([[0.02, 0.003, 0.0], [0.003, 0.015, 0.002], [0.0, 0.002, 0.004]])
    centres = ((0.5, -0.4, 2.4), (1.0, 0.3, 2.6), (1.5, -0.1, 2.5))
    truth = 1.0 + sum(_gaussian(grid, 20.0, centre, covariance) for centre in centres)
    volume = _volume((h, k, l), truth)
    seeds = tuple(PeakSeed(f"c{i}", centre) for i, centre in enumerate(centres))
    tracemalloc.start()
    try:
        outcome = fit_volume_region(volume, seeds=seeds)
        workspace_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert outcome.diagnostics.n_parameters == 34 and outcome.diagnostics.n_used == 100_000
    assert workspace_peak < 128 << 20, workspace_peak
    # the large-problem solver path meets the same recovery tolerances
    for component, centre in zip(outcome.components, centres):
        _assert_centre(component, (h, k, l), centre)
        assert _frobenius(component.covariance, covariance) <= 1e-2
    assert outcome.diagnostics.uncertainty == "available"
    # the outcome keeps compact parameters and the boolean validity mask, never a float cube
    retained = [value for value in vars_of(outcome) if isinstance(value, np.ndarray)]
    assert all(array.dtype == bool or array.size < 1000 for array in retained)


def vars_of(outcome):
    return [getattr(outcome, name) for name in outcome.__slots__]


def test_cancellation_user_mask_and_flat_units():
    _grid, truth = _row1_truth()
    volume = _volume((H1, K1, L1), truth)
    token = threading.Event()
    token.set()
    with pytest.raises(RegionFitCancelled):
        fit_volume_region(volume, cancel_token=token)
    mask = np.zeros(truth.shape, dtype=bool)
    mask[:, :, :4] = True
    outcome = fit_volume_region(volume, mask=mask, reciprocal_transform=J, seeds=(PeakSeed("peak", (1.0, 0.0, 3.0)),))
    np.testing.assert_array_equal(outcome.valid, ~mask)
    _assert_centre(outcome.components[0], (H1, K1, L1), CENTRE_1)
    units = outcome.flat_units()
    assert set(units) == set(outcome.flat_params())
    assert (units["peak.center_h"], units["peak.height"], units["peak.q_fwhm_1"], units["peak.q_integral"]) == (
        "", "I", "A^-1", "I·A^-3")
    q_units = fit_volume_region(_volume((H1, K1, L1), truth, RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU)).flat_units()
    assert q_units["p0.cov_qx_qy"] == "q_A^-1·q_A^-1" and q_units["bg.slope_qz"] == "I/q_A^-1"


def test_row6_notebook_and_analyzer_requests_give_the_same_outcome():
    _grid, truth = _row1_truth()
    axes = tuple(RegionAxis(name, values) for name, values in zip("hkl", (H1, K1, L1)))
    seeds = (PeakSeed("peak", (1.0, 0.0, 3.0)),)
    direct = fit_3d_region(truth, axes, seeds=seeds, reciprocal_transform=J)
    region = Region3DInput(truth, axes, reciprocal_transform=J)
    analyzed = Region3DFitAnalyzer(seeds=seeds).analyze(
        AnalysisInput("snapshot-1", np.empty(0), np.empty(0), metadata={REGION_3D_KEY: region}))
    assert analyzed.ok and analyzed.label == "snapshot-1"
    assert analyzed.params == direct.flat_params()
    assert "peak.center_h" in analyzed.params and "peak.q_fwhm_1" in analyzed.params
    assert analyzed.result.payload.flat_params() == direct.flat_params()
    refused = Region3DFitAnalyzer().analyze(AnalysisInput("x", np.empty(0), np.empty(0)))
    assert not refused.ok and "fit_region_3d" in refused.message
