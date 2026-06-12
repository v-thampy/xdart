"""
RSMVolume container and associated slice/line-cut/VTK utilities.

Contains:
- RSMVolume          — dataclass for gridded H-K-L intensity volumes
- mask_data()        — crop a volume to an H-K-L bounding box
- save_vtk()         — export to VTK rectilinear grid (requires pyevtk)
- extract_line_cut() — 1D projection / line cut along one axis
- extract_2d_slice() — 2D projection / slice by integrating over one axis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional: VTK export
gridToVTK: Any = None
_VTK_AVAILABLE = False
try:
    from pyevtk.hl import gridToVTK  # type: ignore[import-untyped]
    _VTK_AVAILABLE = True
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Core data container
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class RSMVolume:
    """Container for gridded reciprocal-space data on a regular H-K-L grid."""

    h: np.ndarray
    k: np.ndarray
    l: np.ndarray
    intensity: np.ndarray

    def __post_init__(self) -> None:
        self.h = np.asarray(self.h, dtype=float)
        self.k = np.asarray(self.k, dtype=float)
        self.l = np.asarray(self.l, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)

        if self.intensity.ndim != 3:
            raise ValueError("intensity must be a 3D array")

        expected_shape = (len(self.h), len(self.k), len(self.l))
        if self.intensity.shape != expected_shape:
            raise ValueError(
                f"intensity shape {self.intensity.shape} does not match "
                f"axis lengths {expected_shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.intensity.shape

    def get_bounds(self) -> list[list[float]]:
        """
        Return min, max, and approximate step size for each axis.

        Returns
        -------
        list of list of float
            [[hmin, hmax, dh], [kmin, kmax, dk], [lmin, lmax, dl]]
        """
        bounds: list[list[float]] = []
        for axis in (self.h, self.k, self.l):
            step = 0.0 if len(axis) < 2 else float(np.nanmean(np.diff(axis)))
            bounds.append([float(np.nanmin(axis)), float(np.nanmax(axis)), step])
        return bounds

    def save_vtk(self, path: str | Path) -> None:
        """
        Save the volume to VTK format.

        Parameters
        ----------
        path : str or Path
            Output path without extension.
        """
        save_vtk(self.intensity, (self.h, self.k, self.l), path)

    def get_slice(
        self,
        axis: str,
        val_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a 2D slice or projection by integrating over one axis.

        Parameters
        ----------
        axis : {'h', 'k', 'l'}
            Axis to integrate over.
        val_range : tuple of float, optional
            (min, max) range along the integrated axis. If omitted, the full
            axis is integrated, giving a projection.

        Returns
        -------
        axis1 : np.ndarray
            First axis of the returned 2D image.
        axis2 : np.ndarray
            Second axis of the returned 2D image.
        slice_2d : np.ndarray
            Integrated 2D intensity.
        integrated_axis_vals : np.ndarray
            Values of the integrated axis used in the slice/projection.
        """
        axis_key = axis.strip().lower()
        axis_map = {"h": 0, "k": 1, "l": 2}
        if axis_key not in axis_map:
            raise ValueError("axis must be one of 'h', 'k', or 'l'")

        return extract_2d_slice(
            self.h,
            self.k,
            self.l,
            self.intensity,
            integrate_axis=axis_map[axis_key],
            axis_range=val_range,
        )

    def line_cut(
        self,
        axis: str,
        fixed_ranges: dict[int | str, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a 1D line cut or projection along one axis.

        Parameters
        ----------
        axis : {'h', 'k', 'l'}
            Axis to retain as the 1D profile axis.
        fixed_ranges : dict, optional
            Ranges for the other axes. Keys may be 0/1/2 or 'h'/'k'/'l'.
            If omitted, the full ranges of the other axes are integrated,
            giving a projection.

        Returns
        -------
        axis_vals : np.ndarray
            Coordinate values along the retained axis.
        intensity_1d : np.ndarray
            Integrated 1D intensity profile.
        """
        axis_key = axis.strip().lower()
        axis_map = {"h": 0, "k": 1, "l": 2}
        if axis_key not in axis_map:
            raise ValueError("axis must be one of 'h', 'k', or 'l'")

        fr: dict[int, tuple[float, float]] | None = None
        if fixed_ranges is not None:
            fr = {}
            for k, v in fixed_ranges.items():
                if isinstance(k, str):
                    kk = k.strip().lower()
                    if kk not in axis_map:
                        raise ValueError(f"Invalid axis key in fixed_ranges: {k!r}")
                    fr[axis_map[kk]] = v
                else:
                    fr[int(k)] = v

        return extract_line_cut(
            self.h,
            self.k,
            self.l,
            self.intensity,
            axis=axis_map[axis_key],
            fixed_ranges=fr,
        )

    def crop(
        self,
        hrange: tuple[float, float] = (-np.inf, np.inf),
        krange: tuple[float, float] = (-np.inf, np.inf),
        lrange: tuple[float, float] = (-np.inf, np.inf),
    ) -> RSMVolume:
        """
        Crop the volume to an H-K-L box.

        Parameters
        ----------
        hrange, krange, lrange : tuple of float, optional
            (min, max) bounds for each axis.

        Returns
        -------
        RSMVolume
            Cropped volume.
        """
        axes, data = mask_data(
            self.h,
            self.k,
            self.l,
            self.intensity,
            HRange=hrange,
            KRange=krange,
            LRange=lrange,
        )
        return RSMVolume(axes[0], axes[1], axes[2], data)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def mask_data(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    grid_data: np.ndarray,
    HRange: tuple[float, float] = (-np.inf, np.inf),
    KRange: tuple[float, float] = (-np.inf, np.inf),
    LRange: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[list[np.ndarray], np.ndarray]:
    h_idx = (h >= HRange[0]) & (h <= HRange[1])
    k_idx = (k >= KRange[0]) & (k <= KRange[1])
    l_idx = (l >= LRange[0]) & (l <= LRange[1])

    return [h[h_idx], k[k_idx], l[l_idx]], grid_data[np.ix_(h_idx, k_idx, l_idx)]


def save_vtk(
    grid_data: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    fname: Path | str = "test",
) -> None:
    if not _VTK_AVAILABLE or gridToVTK is None:
        raise ImportError("pyevtk is required for save_vtk")

    H, K, L = coords
    x, y, z = np.meshgrid(H, K, L, indexing="ij")

    if np.all(np.isnan(grid_data)):
        data = np.zeros_like(grid_data, dtype=float)
    else:
        data = grid_data - np.nanmin(grid_data)

    gridToVTK(
        str(fname),
        x,
        y,
        z,
        pointData={"Intensity": np.nan_to_num(data, nan=-1)},
    )


def extract_line_cut(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    intensity: np.ndarray,
    axis: int,
    fixed_ranges: dict[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1D line cut or projection along one axis.

    Parameters
    ----------
    h, k, l : np.ndarray
        1D coordinate axes.
    intensity : np.ndarray
        3D intensity array with shape (len(h), len(k), len(l)).
    axis : int
        Axis to retain: 0 for h, 1 for k, 2 for l.
    fixed_ranges : dict, optional
        Ranges for the two other axes. If omitted, full-axis integration is
        used, yielding a projection.

    Returns
    -------
    axis_vals : np.ndarray
        Coordinate values along the retained axis.
    intensity_1d : np.ndarray
        Integrated intensity along that axis.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    axes_arr = [h, k, l]
    slab = intensity

    # Copy once upfront — never mutate caller's dict
    ranges = dict(fixed_ranges) if fixed_ranges is not None else {}
    for dim in (0, 1, 2):
        if dim != axis and dim not in ranges:
            ranges[dim] = (-np.inf, np.inf)

    for dim in sorted((0, 1, 2), reverse=True):
        if dim == axis:
            continue
        mask = (axes_arr[dim] >= ranges[dim][0]) & (
            axes_arr[dim] <= ranges[dim][1])
        slab = np.moveaxis(slab, dim, 0)[mask]
        slab = np.moveaxis(slab, 0, dim)

    slab = np.moveaxis(slab, axis, -1)
    intensity_1d = np.nansum(slab, axis=(0, 1))
    return axes_arr[axis], intensity_1d


def extract_2d_slice(
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    intensity: np.ndarray,
    integrate_axis: int,
    axis_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2D slice or projection by integrating over one axis.

    Parameters
    ----------
    h, k, l : np.ndarray
        1D coordinate axes.
    intensity : np.ndarray
        3D intensity array.
    integrate_axis : int
        Axis to integrate over: 0 for h, 1 for k, 2 for l.
    axis_range : tuple of float, optional
        Range on the integrated axis. If omitted, the full axis is integrated.

    Returns
    -------
    axis1 : np.ndarray
        First axis of the 2D output.
    axis2 : np.ndarray
        Second axis of the 2D output.
    slice_2d : np.ndarray
        Integrated 2D intensity.
    integrated_axis_vals : np.ndarray
        Coordinate values on the integrated axis that contributed.
    """
    if integrate_axis not in (0, 1, 2):
        raise ValueError("integrate_axis must be 0, 1, or 2")

    axes_arr = [h, k, l]

    if axis_range is None:
        axis_range = (-np.inf, np.inf)

    mask = (axes_arr[integrate_axis] >= axis_range[0]) & (
        axes_arr[integrate_axis] <= axis_range[1]
    )

    if integrate_axis == 0:
        slab = intensity[mask, :, :]
        return k, l, np.nansum(slab, axis=0), h[mask]

    if integrate_axis == 1:
        slab = intensity[:, mask, :]
        return h, l, np.nansum(slab, axis=1), k[mask]

    slab = intensity[:, :, mask]
    return h, k, np.nansum(slab, axis=2), l[mask]
