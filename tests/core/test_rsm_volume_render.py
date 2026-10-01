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
