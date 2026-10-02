"""Custom Slice's headless plane cutter (AUD-9 stage 2b, S4; the maintainer: "The ability to only slices which are
not just along h, k or l. So say HH and l")."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.rsm.custom_slice import (
    MAX_SLAB_SAMPLES,
    MAX_SLICE_POINTS,
    CustomSliceRefused,
    custom_slice,
    direction_label,
    plane_offsets,
    sample_volume,
    volume_centre,
)


def _axes():
    return (np.linspace(-1.0, 1.0, 11), np.linspace(0.0, 2.0, 9), np.linspace(1.0, 3.0, 13))


def _linear(axes):
    h, k, l = np.meshgrid(*axes, indexing="ij")
    return 2.0 + 3.0 * h - 4.0 * k + 0.5 * l


def test_a_linear_volume_is_reproduced_exactly() -> None:
    axes = _axes()
    volume = _linear(axes)
    rng = np.random.default_rng(11)
    points = np.stack([rng.uniform(axis[0], axis[-1], 500) for axis in axes], axis=-1)
    expected = 2.0 + 3.0 * points[:, 0] - 4.0 * points[:, 1] + 0.5 * points[:, 2]
    np.testing.assert_allclose(sample_volume(volume, axes, points), expected, rtol=1e-12, atol=1e-12)
    plane = custom_slice(volume, axes)  # HH–L through the centre
    grid = np.asarray(plane.origin) + plane.t[:, None, None] * np.asarray(plane.u) \
        + plane.s[None, :, None] * np.asarray(plane.v)
    analytic = 2.0 + 3.0 * grid[..., 0] - 4.0 * grid[..., 1] + 0.5 * grid[..., 2]
    assert plane.values.dtype == np.float32 and plane.values.shape == (13, 13)
    np.testing.assert_allclose(plane.values, analytic.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_the_trilinear_gather_equals_map_coordinates_on_a_finite_volume() -> None:
    from scipy.ndimage import map_coordinates
    axes = (np.linspace(0.0, 5.0, 6), np.linspace(-3.0, 3.0, 7), np.linspace(10.0, 17.0, 8))
    rng = np.random.default_rng(5)
    volume = rng.normal(size=(6, 7, 8))
    index = np.stack([rng.uniform(0, n - 1, 400) for n in volume.shape])
    points = np.stack([np.interp(index[k], np.arange(volume.shape[k]), axes[k]) for k in range(3)], axis=-1)
    np.testing.assert_allclose(sample_volume(volume, axes, points),
                               map_coordinates(volume, index, order=1), rtol=1e-12, atol=1e-12)


def test_an_axis_aligned_plane_is_the_viewers_hk_slice_bit_for_bit() -> None:
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
    from xrd_tools.session.rsm_viewer_model import RSM_HK, RSMViewerModel, make_rsm_viewer_values
    shape = (5, 6, 7)
    axes = tuple((name, np.linspace(0.1 * index, 0.1 * index + 1.0, size))
                 for index, (name, size) in enumerate(zip(("h", "k", "l"), shape)))
    intensity = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 0.25
    intensity[2, 3, 4] = np.nan  # an empty voxel on the slice
    intensity[1, 1, 3] = np.nan  # and one beside it, off the slice
    values = make_rsm_viewer_values(project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=axes, axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=intensity, sigma=None, coverage=None, normalization=None))
    product = RSMViewerModel().snapshot(values, l_index=4).product(RSM_HK)
    centres = tuple(axis for _name, axis in values.axes)
    h, k, l = centres
    plane = custom_slice(values.intensity, centres, origin=(h[0], k[0], l[4]), u=(1, 0, 0), v=(0, 1, 0),
                         u_range=(0.0, h[-1] - h[0]), v_range=(0.0, k[-1] - k[0]), points=(shape[0], shape[1]))
    # the viewer's image is (y, x) = (K, H); the plane's values are (t, s) = (H, K)
    assert product.x_axis.shape == (5,) and product.values.shape == (6, 5) and plane.values.shape == (5, 6)
    assert np.array_equal(plane.values.T, product.values, equal_nan=True)
    assert np.isnan(plane.values[2, 3])


def test_an_empty_voxel_does_not_spread() -> None:
    axes = (np.arange(4.0), np.arange(4.0), np.arange(4.0))
    volume = np.full((4, 4, 4), 5.0)
    volume[1, 1, 1] = np.nan
    volume[2, 1, 1] = 7.0
    assert np.isnan(sample_volume(volume, axes, [[1.0, 1.0, 1.0]]))[0]  # at the empty voxel: no data
    # between the empty voxel and its finite neighbour: the neighbour's value from halfway, NaN nearer the hole,
    # so the hole keeps its size (neither filled nor spread)
    np.testing.assert_allclose(sample_volume(volume, axes, [[1.5, 1.0, 1.0], [1.75, 1.0, 1.0]]), [7.0, 7.0])
    assert np.isnan(sample_volume(volume, axes, [[1.25, 1.0, 1.0]]))[0]
    assert np.isnan(sample_volume(volume, axes, [[1.0 + 3e-8, 1.0, 1.0]]))[0]  # a float32 hair off its centre
    empty = np.full((2, 2, 2), np.nan)
    assert np.isnan(sample_volume(empty, (np.arange(2.0),) * 3, [[0.5, 0.5, 0.5]]))[0]
    assert np.isnan(sample_volume(volume, axes, [[3.5, 1.0, 1.0]]))[0]  # outside the extent: no extrapolation


def test_hh_l_labels_axes_and_defaults() -> None:
    axes = _axes()
    plane = custom_slice(_linear(axes), axes)
    assert (plane.t_label, plane.s_label) == ("HH (r.l.u.)", "L (r.l.u.)")
    assert plane.interpolated is True and "interpolated from the gridded volume" in plane.note
    assert plane.origin == (0.0, 1.0, 2.0)  # the volume's centre
    # HH from the centre runs until H or K leaves the box: H from −1 to 1, K from 0 to 2, both one away
    assert (plane.t[0], plane.t[-1]) == (-1.0, 1.0)
    assert (plane.s[0], plane.s[-1]) == (-1.0, 1.0)
    assert direction_label((0, 0, 1), ("Qx", "Qy", "Qz"), "Å⁻¹") == "Qz (Å⁻¹)"
    assert direction_label((1, -1, 0)) == "t·[1 -1 0] (r.l.u.)"
    assert direction_label((1, 1, 0), ("Qx", "Qy", "Qz"), "Å⁻¹") == "t·[1 1 0] (Å⁻¹)"


def test_plane_offsets_give_the_coordinates_along_the_directions() -> None:
    """The window's axes (G9): a sample at (t, s) lies at the plane's point nearest the reciprocal-space origin
    + (t − a)·u + (s − b)·v, so on an H–K plane t − a and s − b are H and K, and on HH–L through any point
    (H + K)/2 and L; oblique directions too. The centre is custom_slice's default origin."""
    axes = _axes()
    assert volume_centre(axes) == custom_slice(_linear(axes), axes).origin == (0.0, 1.0, 2.0)
    a, b = plane_offsets((0.25, 0.75, 2.0), (1, 0, 0), (0, 1, 0))
    assert (a, b) == (-0.25, -0.75)
    origin = np.array([1.5, 0.5, 1.25])  # HH–L through a point off the H = K plane
    a, b = plane_offsets(origin)
    assert (a, b) == ((origin[0] + origin[1]) / -2.0, -origin[2])
    t, s = 0.375, -0.5
    point = origin + t * np.array([1.0, 1.0, 0.0]) + s * np.array([0.0, 0.0, 1.0])
    assert ((point[0] + point[1]) / 2.0, point[2]) == (t - a, s - b)
    u, v = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.5])  # oblique: the foot point is normal to both
    a, b = plane_offsets(origin, u, v)
    foot = origin + a * u + b * v
    assert abs(foot @ u) < 1e-12 and abs(foot @ v) < 1e-12
    for kwargs, code in ((dict(u=(0, 0, 0)), "CUSTOM_SLICE_DIRECTION_ZERO"),
                         (dict(u=(1, 1, 0), v=(2, 2, 0)), "CUSTOM_SLICE_DIRECTIONS_PARALLEL")):
        with pytest.raises(CustomSliceRefused) as refused:
            plane_offsets(origin, **kwargs)
        assert refused.value.code == code, kwargs


def test_the_projections_follow_the_viewers_finite_mean() -> None:
    from xrd_tools.session.rsm_viewer_model import _projection_values
    axes = _axes()
    volume = _linear(axes)
    volume[3:6, 2:4, :] = np.nan
    plane = custom_slice(volume, axes)
    assert np.array_equal(plane.projection_t, _projection_values(plane.values[:, :, None], retained_axis=0),
                          equal_nan=True)
    assert np.array_equal(plane.projection_s, _projection_values(plane.values[:, :, None], retained_axis=1),
                          equal_nan=True)


def test_refusals_name_their_reason() -> None:
    axes = _axes()
    volume = _linear(axes)
    cases = (
        (dict(u=(0, 0, 0)), "CUSTOM_SLICE_DIRECTION_ZERO"),
        (dict(u=(1, 1, 0), v=(2, 2, 0)), "CUSTOM_SLICE_DIRECTIONS_PARALLEL"),
        (dict(origin=(0.0, 1.0, 100.0), u=(1, 0, 0), v=(0, 1, 0)), "CUSTOM_SLICE_OUTSIDE"),
        (dict(u_range=(1.0, 0.0)), "CUSTOM_SLICE_RANGE_INVALID"),
        (dict(points=(MAX_SLICE_POINTS, 2)), "CUSTOM_SLICE_TOO_LARGE"),
    )
    for kwargs, code in cases:
        with pytest.raises(CustomSliceRefused) as refused:
            custom_slice(volume, axes, **kwargs)
        assert refused.value.code == code, kwargs
    with pytest.raises(CustomSliceRefused) as refused:
        custom_slice(volume, (axes[0][::-1], axes[1], axes[2]))
    assert refused.value.code == "CUSTOM_SLICE_SHAPE_INVALID"



# AUD-9 stage 2b G11: Thickness, a slab's finite mean (the maintainer, 2026-10-01: "Do you want the Thickness
# option? Yes")


def _spacing_along(axes, normal):
    """The voxel spacing along a unit normal, as the slab's planes must not exceed it: the least step / |n_k|."""
    return min(float(np.min(np.diff(axis))) / abs(component)
               for axis, component in zip(axes, normal) if component != 0.0)


def _stack(volume, axes, plane):
    """Every plane of ``plane``'s slab sampled on its own, (t, s, planes), float64: the slab's samples."""
    offsets = np.linspace(-plane.thickness / 2.0, plane.thickness / 2.0, plane.planes)
    u, v, normal = (np.asarray(item) for item in (plane.u, plane.v, plane.normal))
    return np.stack([sample_volume(volume, axes, np.asarray(plane.origin) + offset * normal
                                   + plane.t[:, None, None] * u + plane.s[None, :, None] * v)
                     for offset in offsets], axis=-1)


def test_thickness_0_is_the_plane_bit_for_bit() -> None:
    """Thickness 0 (or -0) is the plane exactly as cut before the slab: its values, projections and note."""
    axes = _axes()
    volume = _linear(axes)
    volume[3:6, 2:4, 5:7] = np.nan
    for kwargs in ({}, dict(u=(1, 0, 0), v=(0, 1, 1), origin=(0.1, 0.9, 2.2), points=(17, 9))):
        plane = custom_slice(volume, axes, **kwargs)
        for thickness in (0.0, -0.0, 0):
            same = custom_slice(volume, axes, **kwargs, thickness=thickness)
            for name in ("t", "s", "values", "projection_t", "projection_s"):
                assert np.array_equal(getattr(same, name), getattr(plane, name), equal_nan=True), name
                assert getattr(same, name).tobytes() == getattr(plane, name).tobytes(), name
            assert (same.note, same.thickness, same.planes) == (plane.note, 0.0, 1)
            assert str(same.thickness) == "0.0"
    assert "averaged" not in plane.note
    np.testing.assert_allclose(custom_slice(volume, axes).normal, (2 ** -0.5, -(2 ** -0.5), 0.0), rtol=0, atol=1e-15)


def test_a_slab_of_a_volume_constant_along_its_normal_is_the_plane() -> None:
    """Along an axis (H–K, normal L) a volume of H and K alone, and obliquely (HH–L, normal [1 -1 0]) one of
    H + K and L: every plane of the slab is the plane, so the slab is too."""
    axes = _axes()
    h, k, l = np.meshgrid(*axes, indexing="ij")
    cases = (
        (np.sin(3.0 * h) + k ** 2, dict(u=(1, 0, 0), v=(0, 1, 0))),
        (2.0 * (h + k) + l ** 2, dict(u_range=(-0.6, 0.6), v_range=(-0.9, 0.9))),
    )
    for volume, kwargs in cases:
        plane = custom_slice(volume, axes, **kwargs)
        slab = custom_slice(volume, axes, **kwargs, thickness=1.0)
        assert slab.planes > 3
        assert np.array_equal(np.isnan(slab.values), np.isnan(plane.values))
        np.testing.assert_allclose(slab.values, plane.values, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(slab.projection_t, plane.projection_t, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(slab.projection_s, plane.projection_s, rtol=1e-6, atol=1e-6)


def test_a_slab_of_a_volume_linear_along_its_normal_is_its_middle_plane() -> None:
    """A volume linear along the normal: the planes either side differ from the middle one by equal and opposite
    amounts, so the slab's mean is the middle plane's value (every plane inside the volume)."""
    axes = _axes()
    volume = _linear(axes)  # 2 + 3h − 4k + l/2: it changes by 7/√2 a unit along [1 -1 0]
    inside = dict(u_range=(-0.6, 0.6), v_range=(-0.9, 0.9))
    plane = custom_slice(volume, axes, **inside)
    slab = custom_slice(volume, axes, **inside, thickness=1.0)
    stack = _stack(volume, axes, slab)
    assert slab.planes == 5 and np.isfinite(stack).all()
    np.testing.assert_allclose(stack[..., -1] - stack[..., 2], 0.5 * 7.0 / np.sqrt(2.0), rtol=1e-9)
    np.testing.assert_allclose(stack[..., 0] - stack[..., 2], -0.5 * 7.0 / np.sqrt(2.0), rtol=1e-9)
    np.testing.assert_allclose(slab.values, plane.values, rtol=1e-6, atol=1e-6)


def test_a_slab_does_not_spread_an_empty_voxel() -> None:
    """An empty voxel counts in no mean: where the plane crosses it, the slab is its other planes' mean (not NaN,
    not lowered); where no plane has data, it is NaN."""
    axes = _axes()
    volume = np.full((11, 9, 13), 5.0)
    volume[5, 4, 6] = np.nan  # (0, 1, 2): on the plane, the volume's centre
    volume[2, 6, :] = np.nan  # (-0.6, 1.5) empty through the whole slab
    flat = dict(u=(1, 0, 0), v=(0, 1, 0), points=(11, 9))  # H–K through the centre: its samples are voxels
    plane = custom_slice(volume, axes, **flat)
    slab = custom_slice(volume, axes, **flat, thickness=1.0)
    assert np.isnan(plane.values[5, 4]) and slab.planes == 7
    assert slab.values[5, 4] == 5.0
    assert np.isnan(slab.values[2, 6]) and np.isnan(plane.values[2, 6])
    finite = np.isfinite(slab.values)
    assert finite.sum() == finite.size - 1 and np.all(slab.values[finite] == 5.0)
    assert np.all(slab.projection_t == 5.0) and np.all(slab.projection_s == 5.0)


def test_a_slabs_projections_are_the_finite_means_of_all_its_samples() -> None:
    """Each projection is the finite mean of every sample of the slab at its coordinate, the viewer's recipe over
    a volume, so a value of few planes counts as few samples (not as one averaged pixel); each value is the
    finite mean of its planes."""
    from xrd_tools.session.rsm_viewer_model import _projection_values
    axes = _axes()
    volume = _linear(axes)
    rng = np.random.default_rng(7)
    volume[rng.random(volume.shape) < 0.3] = np.nan
    slab = custom_slice(volume, axes, thickness=0.8)
    stack = _stack(volume, axes, slab)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=-1)
    means = np.full(counts.shape, np.nan)
    means[counts > 0] = np.where(finite, stack, 0.0).sum(axis=-1)[counts > 0] / counts[counts > 0]
    assert (counts == 0).any() and ((counts > 0) & (counts < slab.planes)).any()  # empty and partial values
    np.testing.assert_allclose(slab.values, means, rtol=1e-6, atol=1e-6)
    expected_t = _projection_values(stack, retained_axis=0)
    expected_s = _projection_values(stack, retained_axis=1)
    np.testing.assert_allclose(slab.projection_t, expected_t, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(slab.projection_s, expected_s, rtol=1e-6, atol=1e-6)
    image_means = _projection_values(slab.values[:, :, None], retained_axis=0)  # the averaged image's: not these
    assert not np.allclose(image_means, expected_t, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_a_slab_names_its_planes_half_thickness_and_normal() -> None:
    """The note: the plane's, then "averaged over 7 planes, ±0.05 along [1 −1 0]"; the labels are the plane's."""
    axes = (np.linspace(-0.3, 0.3, 41), np.linspace(-0.3, 0.3, 41), np.linspace(1.0, 3.0, 13))  # 0.015 in H, K
    volume = _linear(axes)
    plane = custom_slice(volume, axes)
    slab = custom_slice(volume, axes, thickness=0.1)
    assert slab.note == plane.note + "; averaged over 7 planes, ±0.05 along [1 −1 0]"
    assert (slab.t_label, slab.s_label) == (plane.t_label, plane.s_label) == ("HH (r.l.u.)", "L (r.l.u.)")
    assert (slab.thickness, slab.planes, slab.interpolated) == (0.1, 7, True)
    np.testing.assert_allclose(slab.normal, (2 ** -0.5, -(2 ** -0.5), 0.0), rtol=0, atol=1e-15)
    q = custom_slice(volume, axes, u=(0, 0, 1), v=(1, 0, 0), thickness=0.03, symbols=("Qx", "Qy", "Qz"),
                     unit="Å⁻¹")
    assert q.note.endswith("; averaged over 3 planes, ±0.015 along [0 1 0]") and q.normal == (0.0, 1.0, 0.0)
    assert (q.t_label, q.s_label) == ("Qz (Å⁻¹)", "Qx (Å⁻¹)")


def test_the_plane_count_follows_the_voxel_spacing_along_the_normal() -> None:
    """The least odd count whose spacing is no more than the voxel spacing along the normal (one voxel along the
    axis it crosses fastest): a half-thickness of a whole number of spacings adds no planes for rounding."""
    axes = _axes()  # steps 0.2, 0.25 and 1/6
    volume = _linear(axes)
    for u, v, thickness, expected in (((1, 0, 0), (0, 1, 0), 1.0, 7),  # along L: 0.5 is 3 spacings of 1/6
                                      ((1, 0, 0), (0, 1, 0), 1.01, 9),
                                      ((1, 1, 0), (0, 0, 1), 1.0, 5),  # 0.2·√2 along [1 -1 0]
                                      ((1, 0, 0), (0, 1, 1), 0.5, 5),  # (1/6)·√2 along [0 -1 1]
                                      ((1, 1, 0), (0, 0, 1), 1e-9, 3)):
        slab = custom_slice(volume, axes, u=u, v=v, thickness=thickness)
        spacing = _spacing_along(axes, slab.normal)
        assert slab.planes == expected, (u, v, thickness, slab.planes)
        assert slab.planes % 2 == 1 and thickness / (slab.planes - 1) <= spacing * (1 + 1e-9)
        assert slab.planes == 3 or thickness / (slab.planes - 3) > spacing


def test_a_negative_non_finite_or_too_thick_thickness_is_refused() -> None:
    axes = _axes()
    volume = _linear(axes)
    for thickness in (-0.1, -np.inf, np.inf, np.nan, "thick", None):
        with pytest.raises(CustomSliceRefused) as refused:
            custom_slice(volume, axes, thickness=thickness)
        assert refused.value.code == "CUSTOM_SLICE_THICKNESS_INVALID", thickness
        assert str(refused.value) == "the thickness must be a finite number, 0 or more"
    with pytest.raises(CustomSliceRefused, match="needs more than 16,000,000 planes") as refused:
        custom_slice(volume, axes, thickness=1e12)
    assert refused.value.code == "CUSTOM_SLICE_TOO_LARGE"
    with pytest.raises(CustomSliceRefused, match="5 planes × 4,000,000 samples is above Custom Slice's 16,000,000"):
        custom_slice(volume, axes, points=(2000, 2000), thickness=1.0)
    assert MAX_SLAB_SAMPLES == 16_000_000


# Codex's combined stage-2b review (2026-10-01) F1: the Auto ranges came from the two LINES through the point,
# origin + t·u and origin + s·v, so a Through whose lines miss the box refused a plane that crosses it. They are
# the plane's (the slab's) own extent inside the box now, the same for every Through on the plane.


def test_auto_hk_plane_is_independent_of_through_point_within_the_same_plane():
    """Codex's reproduction (its test_custom_slice_origin.py), adopted as is."""
    h = k = np.linspace(0.8, 1.2, 5)
    l = np.linspace(2.8, 3.2, 5)
    axes = (h, k, l)
    values = (h[:, None, None] + 2 * k[None, :, None] + 3 * l[None, None, :]).astype(np.float32)
    first, second = (1, 0, 0), (0, 1, 0)
    centered = custom_slice(values, axes, origin=(1, 1, 3), u=first, v=second, points=(5, 5))
    # Both equations describe L = 3, which intersects every H/K bin in this volume.
    # The GUI accepts a Through point and defaults its Along ranges to Auto.
    from_origin = custom_slice(values, axes, origin=(0, 0, 3), u=first, v=second, points=(5, 5))
    np.testing.assert_array_equal(from_origin.values, centered.values)


def _dyadic_volume():
    """Axes in steps of 1/4 and a volume that is no linear function (a shifted range shows), with empty voxels:
    every Auto bound, offset and sample coordinate below is exact, so a plane cut through any point on it is the
    same samples bit for bit."""
    axes = (np.linspace(0.0, 2.0, 9), np.linspace(0.0, 2.0, 9), np.linspace(1.0, 3.0, 9))
    volume = np.random.default_rng(3).normal(size=(9, 9, 9))
    volume[2, 3, 4] = volume[6, 6, 1] = np.nan
    return volume, axes


def _same_cut(first, second):
    """The same plane, cut through two of its points: the t and s offsets differ by the points' own offset along
    the directions, and the values and projections are the same bytes."""
    assert first.values.shape == second.values.shape
    for name in ("values", "projection_t", "projection_s"):
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes(), name
    assert np.isfinite(first.values).any()


def test_a_plane_through_any_of_its_points_is_the_same_cut() -> None:
    """Through moved within the plane changes no sample: H–K, HH–L (u = (1, 1, 0), v = (0, 0, 1)) and a coupled,
    non-orthogonal basis (u = (1, 1, 0), v = (0, 1, 1): K moves with both). Each moved point's line along one
    direction misses the box (refused before), or (the coupled basis) crosses it on a shorter span than the plane
    (a different, narrower cut before)."""
    volume, axes = _dyadic_volume()
    centre = np.array([1.0, 1.0, 2.0])
    for u, v, moves, points in (
            ((1, 0, 0), (0, 1, 0), ((-1.0, 2.0), (-3.0, -0.5)), ((0.0, 3.0, 2.0), (-2.0, 0.5, 2.0))),
            ((1, 1, 0), (0, 0, 1), ((-1.0, 2.0), (2.0, -2.0)), ((0.0, 0.0, 4.0), (3.0, 3.0, 0.0))),
            ((1, 1, 0), (0, 1, 1), ((0.0, 2.0), (-0.5, 0.5)), ((1.0, 3.0, 4.0), (0.5, 1.0, 2.5)))):
        reference = custom_slice(volume, axes, origin=centre, u=u, v=v, points=(9, 9))
        for (a, b), point in zip(moves, points, strict=True):
            assert tuple(centre + a * np.array(u) + b * np.array(v)) == point  # a point of the same plane
            cut = custom_slice(volume, axes, origin=point, u=u, v=v, points=(9, 9))
            _same_cut(cut, reference)
            np.testing.assert_array_equal(cut.t, reference.t - a)
            np.testing.assert_array_equal(cut.s, reference.s - b)


def test_a_slab_through_any_point_of_its_middle_plane_is_the_same_cut() -> None:
    """With a thickness, the slab's planes are the same through any point of the middle plane: H–K (normal L,
    every offset exact) bit for bit; HH–L (normal [1 −1 0]/√2, offsets not exact) to rounding."""
    volume, axes = _dyadic_volume()
    reference = custom_slice(volume, axes, origin=(1.0, 1.0, 2.0), u=(1, 0, 0), v=(0, 1, 0), points=(9, 9),
                             thickness=0.5)
    assert reference.planes == 3
    _same_cut(custom_slice(volume, axes, origin=(0.0, 3.0, 2.0), u=(1, 0, 0), v=(0, 1, 0), points=(9, 9),
                           thickness=0.5), reference)
    reference = custom_slice(volume, axes, origin=(1.0, 1.0, 2.0), points=(9, 9), thickness=0.5)  # HH–L
    moved = custom_slice(volume, axes, origin=(0.0, 0.0, 4.0), points=(9, 9), thickness=0.5)
    assert moved.planes == reference.planes == 3
    np.testing.assert_array_equal(moved.t, reference.t + 1.0)
    np.testing.assert_array_equal(moved.s, reference.s - 2.0)
    for name in ("values", "projection_t", "projection_s"):
        np.testing.assert_allclose(getattr(moved, name), getattr(reference, name), rtol=1e-6, atol=1e-6,
                                   equal_nan=True, err_msg=name)


def _line_span(direction, origin, axes):
    """The range Auto gave before F1 (G9–G13): where the line origin + c·direction stays inside the box. The
    reference for the planes on which it is the plane's own extent."""
    centres = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    low, high = np.array([centre[0] for centre in centres]), np.array([centre[-1] for centre in centres])
    direction, origin = np.asarray(direction, dtype=np.float64), np.asarray(origin, dtype=np.float64)
    lowest, highest = -np.inf, np.inf
    for k in range(3):
        if direction[k] != 0.0:
            ends = sorted(((low[k] - origin[k]) / direction[k], (high[k] - origin[k]) / direction[k]))
            lowest, highest = max(lowest, ends[0]), min(highest, ends[1])
        else:
            assert low[k] <= origin[k] <= high[k]
    assert highest > lowest
    return float(lowest), float(highest)


def test_where_the_lines_crossed_on_a_separable_basis_auto_is_the_old_range_bit_for_bit() -> None:
    """Thickness 0, on a basis that moves each axis with one direction at most (H–K, H–L, K–L, HH–L, H and K
    against L, (1, 0, 0) against (0, 1, 1), non-unit and negative directions), through points whose two lines
    cross the box: the plane's extent is the lines' own, so Auto is the old range to the last bit (float32 axes
    too), and so is the cut. The default HH–L slab through the centre keeps the old range too, its edge columns
    drawn."""
    axes = _axes()
    volume = _linear(axes)
    volume[3:6, 2:4, 5:7] = np.nan
    axes32 = tuple(axis.astype(np.float32) for axis in axes)
    inside = (0.1, 0.9, 2.2)
    for case_axes, u, v, origin, points in (
            (axes, (1, 1, 0), (0, 0, 1), None, None),  # the window's default
            (axes32, (1, 1, 0), (0, 0, 1), None, None),
            (axes, (1, 0, 0), (0, 1, 0), inside, (11, 9)),
            (axes, (1, 0, 0), (0, 0, 1), inside, (11, 13)),
            (axes, (0, 1, 0), (0, 0, 1), inside, (9, 13)),
            (axes, (1, 1, 0), (0, 0, 1), inside, (15, 13)),
            (axes, (1, 0, 0), (0, 1, 1), inside, (17, 9)),
            (axes32, (1, 0, 0), (0, 1, 1), inside, (17, 9)),
            (axes, (2, 0, 0), (0, 0, 3), (0.3, 1.1, 1.7), (12, 10)),
            (axes, (1, -1, 0), (0, 0, 1), (0.25, 1.3, 2.0), (12, 10)),
            (axes, (0, 0, 1), (1, 0, 0), None, None)):
        point = volume_centre(case_axes) if origin is None else origin
        auto = custom_slice(volume, case_axes, origin, u, v, points=points)
        old = custom_slice(volume, case_axes, origin, u, v, _line_span(u, point, case_axes),
                           _line_span(v, point, case_axes), points)
        for name in ("t", "s", "values", "projection_t", "projection_s"):
            assert getattr(auto, name).tobytes() == getattr(old, name).tobytes(), (u, v, origin, name)
    slab = custom_slice(volume, axes, thickness=0.8)
    assert (slab.t[0], slab.t[-1], slab.s[0], slab.s[-1]) == (-1.0, 1.0, -1.0, 1.0)  # the old range
    assert np.isfinite(slab.values[[0, -1], :]).all() and np.isfinite(slab.values[:, [0, -1]]).all()


def test_a_slab_whose_middle_plane_misses_the_box_but_crosses_it_is_cut() -> None:
    """The planes averaged decide: a slab around L = 3.2 (the box ends at 3) whose lower planes cross the box is
    cut over the box's H and K, each value the finite mean of its planes inside it; refused before (the middle
    plane's lines miss)."""
    axes = _axes()
    volume = _linear(axes)
    slab = custom_slice(volume, axes, origin=(0.0, 1.0, 3.2), u=(1, 0, 0), v=(0, 1, 0), points=(11, 9),
                        thickness=1.0)
    assert (slab.t[0], slab.t[-1], slab.s[0], slab.s[-1]) == (-1.0, 1.0, -1.0, 1.0)  # H −1…1 and K 0…2
    stack = _stack(volume, axes, slab)
    finite = np.isfinite(stack)
    assert finite.any(axis=-1).all() and not finite.all()  # every value from the planes inside, some outside
    expected = np.where(finite, stack, 0.0).sum(axis=-1) / finite.sum(axis=-1)
    np.testing.assert_allclose(slab.values, expected, rtol=1e-6, atol=1e-6)


def test_a_plane_or_slab_that_misses_the_box_is_still_refused() -> None:
    """Off the box altogether, Auto still refuses CUSTOM_SLICE_OUTSIDE: an H–K plane above it, HH–L planes
    beside it (H − K = 20), a coupled plane past its corner, and a slab too thin to reach it."""
    axes = _axes()  # H −1…1, K 0…2, L 1…3
    volume = _linear(axes)
    for kwargs in (dict(origin=(0.0, 1.0, 100.0), u=(1, 0, 0), v=(0, 1, 0)),
                   dict(origin=(10.0, -10.0, 2.0)),
                   dict(origin=(10.0, -10.0, 2.0), thickness=1.0),
                   dict(origin=(5.0, 2.0, 3.0), u=(1, 1, 0), v=(0, 1, 1)),
                   dict(origin=(0.0, 1.0, 3.6), u=(1, 0, 0), v=(0, 1, 0), thickness=1.0)):
        with pytest.raises(CustomSliceRefused) as refused:
            custom_slice(volume, axes, **kwargs)
        assert refused.value.code == "CUSTOM_SLICE_OUTSIDE", kwargs
        assert str(refused.value) == "the plane does not cross the volume", kwargs


def test_typed_ranges_are_cut_as_typed() -> None:
    """A range typed (not Auto) is the cut's own, sample for sample, whatever the plane's extent; one typed and
    one Auto keeps the typed one."""
    axes = _axes()
    volume = _linear(axes)
    volume[3:6, 2:4, 5:7] = np.nan
    origin, u, v = np.array([0.1, 0.9, 2.2]), np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])
    cut = custom_slice(volume, axes, origin, u, v, (-0.7, 0.4), (-1.5, 0.3), (12, 10))
    t, s = np.linspace(-0.7, 0.4, 12), np.linspace(-1.5, 0.3, 10)
    expected = sample_volume(volume, axes, origin + t[:, None, None] * u + s[None, :, None] * v).astype(np.float32)
    assert cut.t.tobytes() == t.tobytes() and cut.s.tobytes() == s.tobytes()
    assert cut.values.tobytes() == expected.tobytes()
    mixed = custom_slice(volume, axes, origin, u, v, (-0.7, 0.4), None, (12, 10))
    assert mixed.t.tobytes() == t.tobytes()


def test_importing_custom_slice_loads_no_qt_plotting_or_scipy() -> None:
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import sys
        import xrd_tools.rsm.custom_slice
        roots = ("xdart", "PySide6", "PyQt5", "PyQt6", "pyqtgraph", "matplotlib", "scipy")
        bad = sorted(n for n in sys.modules if any(n == r or n.startswith(r + ".") for r in roots))
        assert not bad, bad
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True,
                            env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stdout + result.stderr
