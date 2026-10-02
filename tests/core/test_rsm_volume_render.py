"""The 3D Viewer's headless transfer function (AUD-9 stage 2b, S4; the maintainer: "A button to bring up a 3d
viewer for the entire 3d RSM volume")."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.rsm.volume_render import (
    MAX_RENDER_VOXELS,
    VolumeRenderRefused,
    check_render_size,
    color_table,
    downsample_volume_2x,
    log_transform,
    volume_levels,
    volume_rgba,
)


def test_colours_at_the_level_ends_are_the_colour_maps_ends_and_default_is_viridis() -> None:
    values = np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)
    table = color_table("Default")
    np.testing.assert_array_equal(table, color_table("viridis"))
    rgba = volume_rgba(values, levels=(0.0, 1.0))
    assert rgba.shape == (1, 1, 3, 4) and rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[0, 0, 0, :3], table[0, :3])
    np.testing.assert_array_equal(rgba[0, 0, 1, :3], table[255, :3])
    np.testing.assert_array_equal(rgba[0, 0, 2, :3], table[128, :3])  # rint(0.5 · 255) = 128
    magma = volume_rgba(values, colormap="magma", levels=(0.0, 1.0))
    np.testing.assert_array_equal(magma[0, 0, 1, :3], color_table("magma")[255, :3])
    with pytest.raises(VolumeRenderRefused) as refused:
        volume_rgba(values, colormap="no-such-map", levels=(0.0, 1.0))
    assert refused.value.code == "VOLUME_COLOR_MAP_UNKNOWN"


def test_non_finite_and_below_floor_voxels_are_transparent_and_opacity_ramps_to_the_upper_level() -> None:
    values = np.array([[[np.nan, -1.0, 0.0, 0.5, 1.0, 2.0, np.inf]]], dtype=np.float32)
    alpha = volume_rgba(values, levels=(0.0, 1.0))[0, 0, :, 3]
    np.testing.assert_array_equal(alpha, [0, 0, 0, 128, 255, 255, 0])
    alpha = volume_rgba(values, levels=(0.0, 1.0), floor=0.5)[0, 0, :, 3]
    np.testing.assert_array_equal(alpha, [0, 0, 0, 0, 255, 255, 0])
    quarter = volume_rgba(np.array([[[0.75]]]), levels=(0.0, 1.0), floor=0.5)[0, 0, 0, 3]
    assert quarter == 128  # rint(0.5 · 255) between the floor and the upper level


def test_log_is_ints_shift_to_one_then_log10() -> None:
    values = np.array([[[-3.0, 0.0, 9.0, np.nan]]], dtype=np.float32)
    shifted = log_transform(values)
    np.testing.assert_array_equal(np.isnan(shifted), [[[False, False, False, True]]])
    np.testing.assert_allclose(shifted[0, 0, :3], np.log10(np.array([1.0, 4.0, 13.0])), rtol=1e-6)
    positive = np.array([[[1.0, 10.0, 100.0]]], dtype=np.float32)
    np.testing.assert_allclose(log_transform(positive)[0, 0], [0.0, 1.0, 2.0], atol=1e-7)  # no shift at ≥ 1
    levels = (0.0, float(np.log10(13.0)))
    np.testing.assert_array_equal(volume_rgba(values, log=True, levels=levels),
                                  volume_rgba(shifted, levels=levels))


def test_default_levels_are_ints_percentiles_in_the_displayed_space() -> None:
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 5.0, size=(9, 8, 7)).astype(np.float32)
    values[0, 0, :3] = np.nan
    finite = values[np.isfinite(values)].astype(np.float64)
    np.testing.assert_allclose(volume_levels(values), np.percentile(finite, (0.5, 99.5)))
    shown = log_transform(values)
    np.testing.assert_allclose(volume_levels(shown, log=True),
                               np.percentile(shown[np.isfinite(shown)].astype(np.float64), (0.1, 99.9)))
    flat = np.full((2, 2, 2), 3.0)
    assert volume_levels(flat) == (2.5, 3.5)  # equal ends spread by ±0.5, as Int's non-detector pane
    with pytest.raises(VolumeRenderRefused) as refused:
        volume_levels(np.full((2, 2, 2), np.nan))
    assert refused.value.code == "VOLUME_EMPTY"


def test_the_2x_view_is_a_nan_aware_block_mean_and_an_odd_axis_ends_in_a_half_block() -> None:
    even = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    axes = (np.arange(4.0), np.arange(4.0) * 2, np.arange(4.0) + 10)
    half = downsample_volume_2x(axes, even)
    assert half.values.shape == (2, 2, 2) and half.values.dtype == np.float32
    np.testing.assert_allclose(half.values[0, 0, 0], np.mean(even[:2, :2, :2]))
    np.testing.assert_allclose(half.values[1, 1, 1], np.mean(even[2:, 2:, 2:]))
    np.testing.assert_allclose(half.axes[0], [0.5, 2.5])
    np.testing.assert_allclose(half.axes[1], [1.0, 5.0])
    np.testing.assert_allclose(half.axes[2], [10.5, 12.5])
    odd = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    odd[0, 0, 0] = np.nan  # one NaN in the first block: the mean of its seven finite voxels
    odd[2, 2, 2] = np.nan  # the last block holds this voxel alone: NaN
    half = downsample_volume_2x((np.arange(3.0),) * 3, odd)
    assert half.values.shape == (2, 2, 2)
    np.testing.assert_allclose(half.values[0, 0, 0], np.nanmean(odd[:2, :2, :2]))
    assert np.isnan(half.values[1, 1, 1])
    np.testing.assert_allclose(half.values[1, 0, 0], np.mean(odd[2:, :2, :2]))
    np.testing.assert_allclose(half.axes[0], [0.5, 2.0])  # the lone last centre stays itself


def test_a_volume_above_the_limit_is_refused_naming_its_size_and_the_2x_view() -> None:
    check_render_size((200, 200, 200))  # Full 200³ renders
    assert MAX_RENDER_VOXELS == 256 ** 3
    with pytest.raises(VolumeRenderRefused) as refused:
        check_render_size((400, 400, 400))
    assert refused.value.code == "VOLUME_TOO_LARGE"
    message = str(refused.value)
    assert "256,000,000" in message and f"{4 * 256 ** 3:,}" in message and "200×200×200" in message
    assert "32,000,000" in message
    with pytest.raises(VolumeRenderRefused) as refused:
        volume_rgba(np.zeros((3, 3, 3)), levels=(0.0, 1.0), max_voxels=26)
    assert refused.value.code == "VOLUME_TOO_LARGE"


# AUD-9 stage 2b item 6 (the maintainer: "yes go ahead with both"): the 3D Viewer colours off the GUI thread through
# an 8-bit index volume and a 256-entry table, which must be volume_rgba's bytes exactly.


def _reference_rgba(values, colormap="Default", log=False, levels=None, floor=None):
    """``volume_rgba`` as it stood at 2a9dfe76, its slab loop verbatim: the bytes every route must reproduce."""
    source = np.asarray(values)
    table = color_table(colormap)
    shown = log_transform(source) if log else source
    low, high = volume_levels(shown, log=log) if levels is None else (float(levels[0]), float(levels[1]))
    cutoff = low if floor is None else float(floor)
    out = np.zeros(source.shape + (4,), dtype=np.uint8)
    plane = max(1, int(np.prod(source.shape[1:], dtype=np.int64)))
    step = max(1, (1 << 18) // plane)
    for start in range(0, source.shape[0], step):
        slab = np.asarray(shown[start:start + step], dtype=np.float64)
        finite = np.isfinite(slab)
        place = np.clip((np.where(finite, slab, low) - low) / (high - low), 0.0, 1.0)
        out[start:start + step, ..., :3] = table[np.rint(place * 255.0).astype(np.intp), :3]
        if high > cutoff:
            ramp = np.clip((np.where(finite, slab, cutoff) - cutoff) / (high - cutoff), 0.0, 1.0)
        else:
            ramp = np.ones(slab.shape)
        visible = finite & (slab > cutoff)
        out[start:start + step, ..., 3] = np.where(visible, np.rint(ramp * 255.0), 0).astype(np.uint8)
    return out


def _colouring_cases():
    """Volumes over several slabs, with empty voxels and values below 1 and 0; Log off and on; automatic and
    typed levels; a floor inside and one above the upper level; two colour maps."""
    rng = np.random.default_rng(6)
    volume = (rng.gamma(0.7, 40.0, size=(300, 40, 30)) - 3.0).astype(np.float32)
    volume[rng.random(volume.shape) < 0.04] = np.nan
    for log in (False, True):
        shown = log_transform(volume) if log else volume
        auto = volume_levels(shown, log=log)
        middle = (auto[0] + auto[1]) / 2.0
        for levels, floor in ((None, None), ((auto[0], middle), None), (auto, middle), (auto, auto[1] + 1.0)):
            for colormap in ("Default", "magma"):
                yield volume, colormap, log, levels, floor


def test_volume_rgba_is_its_own_slab_loop_byte_for_byte() -> None:
    """A preservation row: whatever route volume_rgba takes, its bytes are the reference loop's."""
    for volume, colormap, log, levels, floor in _colouring_cases():
        expected = _reference_rgba(volume, colormap, log, levels, floor)
        assert volume_rgba(volume, colormap, log, levels, floor).tobytes() == expected.tobytes(), (
            colormap, log, levels, floor)


def test_an_index_volume_and_a_colour_table_give_volume_rgbas_bytes() -> None:
    """The index route: ``volume_index`` (each voxel's colour-table row and opacity, from the levels and the
    floor, in the displayed space) and ``colour_volume`` (the table applied) give volume_rgba's bytes, so a colour
    map change only applies another table to the same index."""
    from xrd_tools.rsm.volume_render import colour_volume, volume_index
    for volume, colormap, log, levels, floor in _colouring_cases():
        shown = log_transform(volume) if log else volume
        chosen = volume_levels(shown, log=log) if levels is None else levels
        index, alpha = volume_index(shown, chosen, floor)
        assert index.dtype == alpha.dtype == np.uint8 and index.shape == alpha.shape == volume.shape
        rgba = colour_volume(color_table(colormap), index, alpha)
        assert rgba.tobytes() == _reference_rgba(volume, colormap, log, levels, floor).tobytes(), (
            colormap, log, levels, floor)
    with pytest.raises(VolumeRenderRefused) as refused:
        volume_index(np.zeros((2, 2, 2)), (1.0, 1.0))
    assert refused.value.code == "VOLUME_LEVELS_INVALID"
    with pytest.raises(VolumeRenderRefused) as refused:
        volume_index(np.zeros((2, 2, 2)), (0.0, 1.0), np.nan)
    assert str(refused.value) == "the opacity floor must be finite"


def test_importing_the_transfer_function_loads_no_qt_or_plotting_library() -> None:
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import sys
        import xrd_tools.rsm.volume_render
        roots = ("xdart", "PySide6", "PyQt5", "PyQt6", "pyqtgraph", "matplotlib")
        bad = sorted(n for n in sys.modules if any(n == r or n.startswith(r + ".") for r in roots))
        assert not bad, bad
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True,
                            env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stdout + result.stderr
