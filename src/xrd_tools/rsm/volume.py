"""
RSMVolume container and associated slice/line-cut/VTK utilities.

Contains:
- RSMVolume          — frame-aware gridded reciprocal-space volume
- mask_data()        — legacy crop helper for an H-K-L bounding box
- save_vtk()         — export to VTK rectilinear grid (requires pyevtk)
- extract_line_cut() — 1D projection / line cut along one axis
- extract_2d_slice() — 2D projection / slice by integrating over one axis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame

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

@dataclass(slots=True, init=False)
class RSMVolume:
    """Container for one closed reciprocal-space coordinate frame.

    The legacy constructor remains H/K/L-only.  New frames must use
    :meth:`from_axes`, which binds ordered names and units through an exact
    :class:`RSMCoordinateFrame`.  Consequently, Qx/Qy/Qz data can never be
    exposed through the legacy ``h``/``k``/``l`` properties.
    """

    _coordinate_frame: RSMCoordinateFrame
    _axis_values: tuple[np.ndarray, np.ndarray, np.ndarray]
    intensity: np.ndarray
    #: optional provenance (the RSMPlan + applied CorrectionStack) — populated by
    #: read_rsm when a persisted volume carries a provenance_json blob.
    provenance: Any = None

    def __init__(
        self,
        h: np.ndarray,
        k: np.ndarray,
        l: np.ndarray,
        intensity: np.ndarray,
        provenance: Any = None,
    ) -> None:
        """Build a legacy-compatible H/K/L volume."""

        self._initialize(
            RSMCoordinateFrame.HKL,
            (h, k, l),
            intensity,
            provenance,
        )

    @classmethod
    def from_axes(
        cls,
        coordinate_frame: RSMCoordinateFrame,
        axes: tuple[
            tuple[str, np.ndarray],
            tuple[str, np.ndarray],
            tuple[str, np.ndarray],
        ],
        intensity: np.ndarray,
        provenance: Any = None,
    ) -> RSMVolume:
        """Build a volume from one exact ordered frame descriptor."""

        if type(coordinate_frame) is not RSMCoordinateFrame:
            raise TypeError("RSM coordinate frame must be exact")
        if (
            type(axes) is not tuple
            or len(axes) != 3
            or any(type(item) is not tuple or len(item) != 2 for item in axes)
            or tuple(item[0] for item in axes) != coordinate_frame.axis_names
        ):
            raise ValueError("RSM axes do not match the coordinate frame")
        result = cls.__new__(cls)
        result._initialize(
            coordinate_frame,
            tuple(item[1] for item in axes),
            intensity,
            provenance,
        )
        return result

    def _initialize(
        self,
        coordinate_frame: RSMCoordinateFrame,
        axis_values: tuple[np.ndarray, np.ndarray, np.ndarray],
        intensity: np.ndarray,
        provenance: Any,
    ) -> None:
        values = tuple(np.asarray(axis, dtype=float) for axis in axis_values)
        if any(axis.ndim != 1 for axis in values):
            raise ValueError("RSM coordinate axes must be 1D arrays")
        volume = np.asarray(intensity, dtype=float)

        if volume.ndim != 3:
            raise ValueError("intensity must be a 3D array")

        expected_shape = tuple(len(axis) for axis in values)
        if volume.shape != expected_shape:
            raise ValueError(
                f"intensity shape {volume.shape} does not match "
                f"axis lengths {expected_shape}"
            )
        self._coordinate_frame = coordinate_frame
        self._axis_values = values
        self.intensity = volume
        self.provenance = provenance

    @property
    def coordinate_frame(self) -> RSMCoordinateFrame:
        return self._coordinate_frame

    @property
    def axes(self) -> tuple[
        tuple[str, np.ndarray],
        tuple[str, np.ndarray],
        tuple[str, np.ndarray],
    ]:
        return tuple(
            zip(
                self._coordinate_frame.axis_names,
                self._axis_values,
                strict=True,
            )
        )  # type: ignore[return-value]

    @property
    def axis_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._axis_values

    @property
    def axis_units(self) -> tuple[
        tuple[str, str | None],
        tuple[str, str | None],
        tuple[str, str | None],
    ]:
        return tuple(
            zip(
                self._coordinate_frame.axis_names,
                self._coordinate_frame.axis_units,
                strict=True,
            )
        )  # type: ignore[return-value]

    def _axis_for_name(self, name: str) -> np.ndarray:
        try:
            index = self._coordinate_frame.axis_names.index(name)
        except ValueError:
            raise AttributeError(
                f"{name} is not an axis in {self._coordinate_frame.value}"
            ) from None
        return self._axis_values[index]

    @property
    def h(self) -> np.ndarray:
        return self._axis_for_name("h")

    @property
    def k(self) -> np.ndarray:
        return self._axis_for_name("k")

    @property
    def l(self) -> np.ndarray:
        return self._axis_for_name("l")

    @property
    def qx(self) -> np.ndarray:
        return self._axis_for_name("qx")

    @property
    def qy(self) -> np.ndarray:
        return self._axis_for_name("qy")

    @property
    def qz(self) -> np.ndarray:
        return self._axis_for_name("qz")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.intensity.shape

    def get_bounds(self) -> list[list[float]]:
        """
        Return min, max, and approximate step size for each axis.

        Returns
        -------
        list of list of float
            One ``[minimum, maximum, step]`` row per ordered frame axis.
        """
        bounds: list[list[float]] = []
        for axis in self._axis_values:
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
        save_vtk(self.intensity, self._axis_values, path)

    def get_slice(
        self,
        axis: str,
        val_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a 2D slice or projection by integrating over one axis.

        Parameters
        ----------
        axis : str
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
        axis_map = {
            name: index
            for index, name in enumerate(self._coordinate_frame.axis_names)
        }
        if axis_key not in axis_map:
            raise ValueError(
                "axis must be one of "
                + ", ".join(repr(name) for name in axis_map)
            )

        return extract_2d_slice(
            *self._axis_values,
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
        axis : str
            Axis to retain as the 1D profile axis.
        fixed_ranges : dict, optional
            Ranges for the other axes. Keys may be 0/1/2 or one of the
            volume's exact axis names.
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
        axis_map = {
            name: index
            for index, name in enumerate(self._coordinate_frame.axis_names)
        }
        if axis_key not in axis_map:
            raise ValueError(
                "axis must be one of "
                + ", ".join(repr(name) for name in axis_map)
            )

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
            *self._axis_values,
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
        if self._coordinate_frame is not RSMCoordinateFrame.HKL:
            raise ValueError(
                "crop(hrange, krange, lrange) is HKL-only; "
                "use crop_by_axes() for this coordinate frame"
            )
        return self.crop_by_axes(
            {"h": hrange, "k": krange, "l": lrange}
        )

    def crop_by_axes(
        self,
        ranges: Mapping[str, tuple[float, float]],
    ) -> RSMVolume:
        """Crop using exact axis names from this volume's frame."""

        if not isinstance(ranges, Mapping):
            raise TypeError("RSM crop ranges must be a mapping")
        names = self._coordinate_frame.axis_names
        unknown = set(ranges) - set(names)
        if unknown:
            raise ValueError(
                f"RSM crop axes are not in {self._coordinate_frame.value}: "
                f"{sorted(unknown)!r}"
            )
        selectors = []
        selected_axes = []
        for name, values in zip(names, self._axis_values, strict=True):
            lower, upper = ranges.get(name, (-np.inf, np.inf))
            selected = (values >= lower) & (values <= upper)
            selectors.append(selected)
            selected_axes.append(values[selected])
        data = self.intensity[np.ix_(*selectors)]
        return RSMVolume.from_axes(
            self._coordinate_frame,
            tuple(zip(names, selected_axes, strict=True)),
            data,
            provenance=self.provenance,
        )


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
    intensity_1d = np.nanmean(slab, axis=(0, 1))
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
        return k, l, np.nanmean(slab, axis=0), h[mask]

    if integrate_axis == 1:
        slab = intensity[:, mask, :]
        return h, l, np.nanmean(slab, axis=1), k[mask]

    slab = intensity[:, :, mask]
    return h, k, np.nanmean(slab, axis=2), l[mask]
