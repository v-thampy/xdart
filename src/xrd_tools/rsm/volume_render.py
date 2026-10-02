"""The 3D Viewer's transfer function: an RSM volume's voxels to RGBA (AUD-9).

The maintainer, 2026-10-01: "A button to bring up a 3d viewer for the entire 3d RSM volume". The window is a
shared GUI component; this module is its headless half (hard invariant 7: no Qt, and no ``matplotlib.pyplot``).

- Colour: a colour map is read once into a 256-entry uint8 table (``color_table``) and indexed per voxel, never
  called on the volume (matplotlib's call returns float64 RGBA: 256 MB at 200³). "Default" is viridis, as
  Int's colour-map list draws it (``image_widget.py``: ``'viridis' if cmap == 'Default'``).
- Log: Int's image pane's convention (``image_widget.py`` ``setImage``): when the finite minimum is below 1, every
  value is shifted so that minimum becomes 1, then ``log10``.
- Levels: by default the percentiles Int's non-detector image pane uses: (0.5, 99.5) linear
  (``shell_widgets.py``, the cake pane's ``linear_percentiles``), (0.1, 99.9) on Log (``image_widget.py``),
  taken in the displayed (post-Log) space. Above ``_LEVEL_SAMPLE`` finite voxels they are taken on a
  deterministic stride sample of the finite voxels, so no second volume-sized copy is made.
- Opacity: zero for non-finite voxels and for voxels at or below ``floor``; above it a linear ramp to opaque at the
  upper level. ``floor`` is in the same space as the levels (post-Log when Log is on) and defaults to the lower
  level.

Memory: the RGBA volume (4 B/voxel) and, on Log, one float32 transformed copy are the only volume-sized
allocations; the rest is done in slabs along the first axis. The 3D Viewer colours through ``volume_index`` (each
voxel's colour-table row and opacity, two uint8 volumes, 2 B/voxel) and ``colour_volume`` (a table applied), the
same bytes, so a colour map change only applies another table to the index it keeps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_COLOR_MAP = "viridis"
#: Full 200³ (8,000,000 voxels, 32 MB of RGBA) renders; 400³ (64,000,000, 256 MB) is refused with the 2× view.
MAX_RENDER_VOXELS = 256 ** 3
_LINEAR_PERCENTILES = (0.5, 99.5)
_LOG_PERCENTILES = (0.1, 99.9)
_LEVEL_SAMPLE = 1 << 20
_SLAB_VOXELS = 1 << 18


class VolumeRenderRefused(ValueError):
    """A volume the 3D Viewer does not render, with the reason the window shows."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def color_table(name: str = "Default") -> np.ndarray:
    """A colour map as a read-only (256, 4) uint8 table. "Default" is viridis; an unknown name is refused."""
    from matplotlib import colormaps  # lazy: importing this module loads no plotting library

    key = DEFAULT_COLOR_MAP if name in (None, "", "Default") else str(name)
    try:
        cmap = colormaps[key]
    except KeyError:
        raise VolumeRenderRefused("VOLUME_COLOR_MAP_UNKNOWN", f"no colour map named {key!r}") from None
    table = np.rint(np.asarray(cmap(np.linspace(0.0, 1.0, 256)), dtype=np.float64) * 255.0).astype(np.uint8)
    table.setflags(write=False)
    return table


def log_transform(values: np.ndarray) -> np.ndarray:
    """Int's Log, as a float32 copy: shifted so the finite minimum becomes 1 when it is below 1, then log10.
    Non-finite voxels stay NaN; a volume with no finite voxel is all NaN."""
    source = np.asarray(values)
    out = np.empty(source.shape, dtype=np.float32)
    finite_min = _finite_minimum(source)
    shift = 0.0 if finite_min is None or finite_min >= 1.0 else finite_min - 1.0
    flat_in, flat_out = source.reshape(-1), out.reshape(-1)
    for start in range(0, flat_in.size, _SLAB_VOXELS):
        chunk = np.asarray(flat_in[start:start + _SLAB_VOXELS], dtype=np.float64)
        finite = np.isfinite(chunk)
        result = np.full(chunk.shape, np.nan)
        np.log10(chunk - shift, out=result, where=finite)
        flat_out[start:start + _SLAB_VOXELS] = result
    return out


def volume_levels(values: np.ndarray, *, log: bool = False) -> tuple[float, float]:
    """The default display levels of ``values`` (already in the displayed space): Int's percentiles of the
    finite voxels, on a deterministic stride sample above ``_LEVEL_SAMPLE`` of them. Equal ends are spread by
    ±0.5, as Int's non-detector pane expands degenerate levels."""
    sample = _finite_sample(np.asarray(values))
    if sample.size == 0:
        raise VolumeRenderRefused("VOLUME_EMPTY", "the volume has no finite voxel to show")
    low, high = (float(item) for item in np.percentile(sample, _LOG_PERCENTILES if log else _LINEAR_PERCENTILES))
    if not high > low:
        low, high = low - 0.5, high + 0.5
    return low, high


def check_render_size(shape: tuple[int, ...], max_voxels: int = MAX_RENDER_VOXELS) -> None:
    """Refuse a volume above ``max_voxels``, naming its RGBA size, the limit, and the 2× view's size."""
    if len(shape) != 3 or any(int(n) < 1 for n in shape):
        raise VolumeRenderRefused("VOLUME_SHAPE_INVALID", "the 3D Viewer needs one 3-D volume")
    voxels = int(np.prod([int(n) for n in shape], dtype=np.int64))
    if voxels > int(max_voxels):
        half = tuple((int(n) + 1) // 2 for n in shape)
        raise VolumeRenderRefused(
            "VOLUME_TOO_LARGE",
            f"the volume {'×'.join(map(str, shape))} needs {4 * voxels:,} bytes of RGBA, above the 3D Viewer's "
            f"{4 * int(max_voxels):,}; its 2× view {'×'.join(map(str, half))} needs "
            f"{4 * int(np.prod(half, dtype=np.int64)):,}")


def volume_rgba(values: np.ndarray, colormap: str = "Default", log: bool = False,
                levels: tuple[float, float] | None = None, floor: float | None = None, *,
                max_voxels: int = MAX_RENDER_VOXELS) -> np.ndarray:
    """An RSM volume as an RGBA uint8 volume of shape ``values.shape + (4,)`` for the 3D Viewer.

    Colour is the colour map at the voxel's place between ``levels`` (clipped); opacity is zero for non-finite
    voxels and at or below ``floor``, then rises linearly to opaque at the upper level. ``levels`` and ``floor``
    are in the displayed space (post-Log when ``log``); by default ``levels`` are Int's percentiles and
    ``floor`` the lower level. The same bytes as ``colour_volume(color_table(colormap), *volume_index(...))``,
    made a slab at a time without the two index volumes."""
    source = np.asarray(values)
    check_render_size(source.shape, max_voxels)
    table = color_table(colormap)
    shown = log_transform(source) if log else source
    low, high, cutoff = _checked_levels(volume_levels(shown, log=log) if levels is None else levels, floor)
    out = np.zeros(source.shape + (4,), dtype=np.uint8)
    for start, stop in _slabs(source.shape):
        index, alpha = _index_slab(shown[start:stop], low, high, cutoff)
        out[start:stop, ..., :3] = table[index, :3]
        out[start:stop, ..., 3] = alpha
    return out


def volume_index(shown: np.ndarray, levels: tuple[float, float],
                 floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each voxel of ``shown`` (already in the displayed space: post-Log when Log is on) as the 3D Viewer colours
    it: its colour-table row, 0–255 (its place between ``levels``, clipped, as ``volume_rgba`` places it), and its
    opacity, 0–255 (zero where empty and at or below ``floor``, the lower level by default; then rising to opaque
    at the upper level). Two uint8 volumes, made a slab at a time. ``colour_volume`` applies a colour table to
    them, so a colour map change applies another table to the same index; with ``color_table(colormap)`` the
    result is ``volume_rgba``'s bytes. Levels or a floor that are not finite, or levels not low below high, are
    refused as ``volume_rgba`` refuses them."""
    source = np.asarray(shown)
    low, high, cutoff = _checked_levels(levels, floor)
    index = np.empty(source.shape, dtype=np.uint8)
    alpha = np.empty(source.shape, dtype=np.uint8)
    for start, stop in _slabs(source.shape):
        index[start:stop], alpha[start:stop] = _index_slab(source[start:stop], low, high, cutoff)
    return index, alpha


def colour_volume(table: np.ndarray, index: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """The RGBA uint8 volume of ``volume_index``'s two volumes under a colour table (``color_table``): each
    voxel's row of the table, then its opacity. A slab at a time, so the only volume-sized array is the RGBA."""
    out = np.empty(index.shape + (4,), dtype=np.uint8)
    for start, stop in _slabs(index.shape):
        out[start:stop, ..., :3] = table[index[start:stop], :3]
        out[start:stop, ..., 3] = alpha[start:stop]
    return out


def _checked_levels(levels, floor) -> tuple[float, float, float]:
    """(low, high, the opacity floor), refused unless finite with low below high."""
    low, high = float(levels[0]), float(levels[1])
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise VolumeRenderRefused("VOLUME_LEVELS_INVALID", "the levels need two finite values, low below high")
    cutoff = low if floor is None else float(floor)
    if not np.isfinite(cutoff):
        raise VolumeRenderRefused("VOLUME_LEVELS_INVALID", "the opacity floor must be finite")
    return low, high, cutoff


def _slabs(shape):
    """(start, stop) along the first axis, about ``_SLAB_VOXELS`` voxels each."""
    plane = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    step = max(1, _SLAB_VOXELS // plane)
    for start in range(0, shape[0], step):
        yield start, min(start + step, shape[0])


def _index_slab(values, low: float, high: float, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """One slab's colour-table rows and opacities (uint8): the one formula of ``volume_rgba`` and
    ``volume_index``."""
    slab = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(slab)
    place = np.clip((np.where(finite, slab, low) - low) / (high - low), 0.0, 1.0)
    index = np.rint(place * 255.0).astype(np.uint8)  # 0 to 255 exactly: the table's rows
    if high > cutoff:
        ramp = np.clip((np.where(finite, slab, cutoff) - cutoff) / (high - cutoff), 0.0, 1.0)
    else:
        ramp = np.ones(slab.shape)
    visible = finite & (slab > cutoff)
    return index, np.where(visible, np.rint(ramp * 255.0), 0).astype(np.uint8)


@dataclass(frozen=True)
class DownsampledVolume:
    """A volume at half resolution: block means of 2×2×2 voxels, the axes as pair means."""

    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: np.ndarray


def downsample_volume_2x(axes, values: np.ndarray) -> DownsampledVolume:
    """The 2× view the 3D Viewer offers for a volume above its limit.

    Each output voxel is the mean of the finite voxels in its 2×2×2 block (NaN when none is finite). An odd
    axis ends in a half-width block: its last output voxel takes the last plane alone, and its last axis value
    is the last centre (the pair means' rule for a lone centre). Slab by slab along the first axis, so only the
    output is volume-sized."""
    source = np.asarray(values)
    if source.ndim != 3 or len(axes) != 3:
        raise VolumeRenderRefused("VOLUME_SHAPE_INVALID", "the 2× view needs one 3-D volume and its three axes")
    centres = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    if tuple(axis.shape for axis in centres) != tuple((n,) for n in source.shape):
        raise VolumeRenderRefused("VOLUME_SHAPE_INVALID", "each axis must hold one centre per voxel")
    half = tuple((n + 1) // 2 for n in source.shape)
    out = np.empty(half, dtype=np.float32)
    n1, n2 = source.shape[1:]
    for index in range(half[0]):
        slab = np.asarray(source[2 * index:2 * index + 2], dtype=np.float64)
        padded = np.full((2, 2 * half[1], 2 * half[2]), np.nan)
        padded[:slab.shape[0], :n1, :n2] = slab
        finite = np.isfinite(padded)
        blocks = np.where(finite, padded, 0.0).reshape(2, half[1], 2, half[2], 2)
        counts = finite.reshape(2, half[1], 2, half[2], 2).sum(axis=(0, 2, 4))
        sums = blocks.sum(axis=(0, 2, 4))
        mean = np.full(sums.shape, np.nan)
        np.divide(sums, counts, out=mean, where=counts > 0)
        out[index] = mean
    pairs = []
    for axis in centres:
        padded_axis = np.append(axis, axis[-1]) if axis.size % 2 else axis
        pairs.append(padded_axis.reshape(-1, 2).mean(axis=1))
    return DownsampledVolume(tuple(pairs), out)


def _finite_minimum(values: np.ndarray) -> float | None:
    best = None
    flat = values.reshape(-1)
    for start in range(0, flat.size, _SLAB_VOXELS):
        chunk = np.asarray(flat[start:start + _SLAB_VOXELS], dtype=np.float64)
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            low = float(chunk.min())
            best = low if best is None else min(best, low)
    return best


def _finite_sample(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    count = 0
    for start in range(0, flat.size, _SLAB_VOXELS):
        count += int(np.count_nonzero(np.isfinite(flat[start:start + _SLAB_VOXELS])))
    stride = max(1, -(-count // _LEVEL_SAMPLE))
    picked, seen = [], 0
    for start in range(0, flat.size, _SLAB_VOXELS):
        chunk = np.asarray(flat[start:start + _SLAB_VOXELS], dtype=np.float64)
        chunk = chunk[np.isfinite(chunk)]
        first = (-seen) % stride
        picked.append(chunk[first::stride].copy())  # a view would keep its whole chunk alive
        seen += chunk.size
    return np.concatenate(picked) if picked else np.empty(0)
