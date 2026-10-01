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
