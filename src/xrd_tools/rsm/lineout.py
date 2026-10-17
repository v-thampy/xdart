"""Axis-aligned line-outs through a point of an RSM volume (headless, Qt-free).

A line-out along one named axis keeps every bin of that axis and averages the voxels whose other two
coordinates lie inside an inclusive box around the point: ``abs(coordinate - centre) <= width / 2``, where
``width`` is the FULL bandwidth (the convention of the example notebook ``RSM_process.ipynb``'s
``compute_lineouts``, with which these line-outs agree on its square volumes).

The value of a bin is the unweighted mean of the box's finite voxels (a voxel mean, not a sum or a coordinate
integral). A bin without any finite voxel is unavailable: NaN with count 0, never an observed zero. An infinite
voxel inside a box is refused (scientific reduction stays loud), not excluded. Contiguous boxes are read as views
of the volume and reduced in blocks, so no line-out copies the volume.

Arbitrary-direction line-outs are a separate contract (a later extension of ``custom_slice``), not this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from xrd_tools.rsm.volume import (
    InfiniteVoxelError,
    RSMVolume,
    _box_mean_along,
    _box_view,
    _mask_selector,
)

__all__ = ["RSMLineout", "RSMLineoutRefused", "axis_lineouts"]


class RSMLineoutRefused(ValueError):
    """A named refusal of a line-out (``code``: ``INFINITE_VOXEL``)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class RSMLineout:
    """One axis-aligned line-out.

    ``coordinates`` are the retained axis's bin centres (all of them, in the volume's order); ``intensity`` the
    voxel mean per bin (NaN where unavailable); ``counts`` the number of finite voxels averaged per bin.
    ``centre`` is the point by axis name; ``box`` the inclusive ``(lower, upper)`` bounds applied on each
    transverse axis (``centre -/+ width / 2``), and ``box_bins`` the number of bins each bound selected.
    """

    axis: str
    unit: str | None
    coordinates: np.ndarray
    intensity: np.ndarray
    counts: np.ndarray
    centre: tuple[tuple[str, float], ...]
    box: tuple[tuple[str, tuple[float, float]], ...]
    box_bins: tuple[tuple[str, int], ...]

    @property
    def available(self) -> np.ndarray:
        """True where a bin averaged at least one finite voxel."""
        return self.counts > 0


def _per_axis(value, names: tuple[str, ...], what: str) -> dict[str, float]:
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown:
            raise ValueError(f"{what} names unknown axis {sorted(unknown)!r}; the axes are {names!r}")
        resolved = dict(value)
    elif isinstance(value, Real):
        resolved = dict.fromkeys(names, value)
    elif isinstance(value, Sequence) or isinstance(value, np.ndarray):
        if len(value) != 3:
            raise ValueError(f"{what} needs one value per axis {names!r}")
        resolved = dict(zip(names, value))
    else:
        raise ValueError(f"{what} must be a number, a sequence in axis order or a mapping by axis name")
    out = {}
    for name, item in resolved.items():
        try:
            out[name] = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"{what} for {name!r} is not a number") from None
    return out


def axis_lineouts(
    volume: RSMVolume,
    centre: Mapping[str, float] | Sequence[float],
    width: float | Mapping[str, float] | Sequence[float],
    axes: Sequence[str] | None = None,
) -> dict[str, RSMLineout]:
    """Line-outs through ``centre`` along each named axis of ``volume``.

    Parameters
    ----------
    volume : RSMVolume
        The admitted volume; its coordinate frame names the axes (``h/k/l`` or ``qx/qy/qz``).
    centre : mapping by axis name, or three values in the frame's axis order
        The point every line-out passes through.
    width : number, mapping by axis name, or three values in axis order
        FULL bandwidth on a transverse axis (a single number applies to every axis). A line-out along ``a``
        uses the widths of the two other axes.
    axes : names, optional
        The axes to retain, default all three; the result keeps the frame's axis order.

    Returns
    -------
    dict[str, RSMLineout]
        One line-out per requested axis, keyed by axis name.
    """
    if type(volume) is not RSMVolume:
        raise TypeError("line-outs need an RSMVolume")
    names = volume.coordinate_frame.axis_names
    units = volume.coordinate_frame.axis_units
    requested = names if axes is None else tuple(axes)
    unknown = [name for name in requested if name not in names]
    if unknown or not requested:
        raise ValueError(f"line-out axis must be one of {names!r}; got {tuple(requested)!r}")
    if isinstance(centre, Real):
        raise ValueError(f"centre needs one value per axis {names!r}")
    point = _per_axis(centre, names, "centre")
    if set(point) != set(names) or not all(np.isfinite(value) for value in point.values()):
        raise ValueError(f"centre needs a finite value for every axis {names!r}")
    widths = _per_axis(width, names, "width")
    values = volume.axis_values
    transverse_of = {name: tuple(other for other in names if other != name) for name in requested}
    selectors = {}
    bounds = {}
    for index, name in enumerate(names):
        if not any(name in others for others in transverse_of.values()):
            continue
        if name not in widths:
            raise ValueError(f"width needs a value for the transverse axis {name!r}")
        full = widths[name]
        if not np.isfinite(full) or full < 0:
            raise ValueError(f"width for {name!r} must be finite and non-negative; got {full!r}")
        half = full / 2
        selectors[name] = _mask_selector(np.abs(values[index] - point[name]) <= half)
        bounds[name] = (point[name] - half, point[name] + half)

    result: dict[str, RSMLineout] = {}
    for keep, name in enumerate(names):
        if name not in requested:
            continue
        transverse = transverse_of[name]
        box = [slice(None) if other == name else selectors[other] for other in names]
        view = _box_view(volume.intensity, box)
        try:
            intensity, counts = _box_mean_along(view, keep, refuse_infinite=True)
        except InfiniteVoxelError:
            raise RSMLineoutRefused(
                "INFINITE_VOXEL",
                f"the {name} line-out's box holds an infinite voxel; mask or correct it before a line-out",
            ) from None
        coordinates = np.array(values[keep], dtype=np.float64, copy=True)
        for array in (coordinates, intensity, counts):
            array.flags.writeable = False
        result[name] = RSMLineout(
            axis=name,
            unit=units[keep],
            coordinates=coordinates,
            intensity=intensity,
            counts=counts,
            centre=tuple((axis, point[axis]) for axis in names),
            box=tuple((other, bounds[other]) for other in transverse),
            box_bins=tuple((other, int(view.shape[names.index(other)])) for other in transverse),
        )
    return result
