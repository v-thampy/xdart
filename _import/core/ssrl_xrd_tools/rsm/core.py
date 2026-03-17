"""
Reciprocal Space Mapping (RSM) utilities for x-ray diffraction data.

Processes SPEC scans and detector images into 3D HKL volumes, with I/O,
gridding, combination, line cuts, slices. Image I/O and SPEC parsing
live in ssrl_xrd_tools.io; fitting in ssrl_xrd_tools.analysis.fitting.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xrayutilities as xu
from scipy.interpolate import RegularGridInterpolator

from ssrl_xrd_tools.io.image import (
    read_image,
    read_image_stack,
    read_images_parallel,
    find_image_files,
    get_detector_mask,
    apply_rotation,
)
from ssrl_xrd_tools.io.spec import (
    get_scan_path_info,
    get_energy_and_UB,
    get_angles,
)

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
# Helpers
# -----------------------------------------------------------------------------

def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_pickle(path: Path) -> Any | None:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Failed to load pickle: %s", path)
        return None


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


# -----------------------------------------------------------------------------
# Core data containers
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


@dataclass(slots=True)
class ScanInfo:
    spec_path: Path
    img_dir: Path
    h5_path: Path | None = None


@dataclass(slots=True)
class DiffractometerConfig:
    """Geometry configuration for xu.QConversion and xu.HXRD."""

    sample_rot: tuple[str, ...] = ("z-", "y+", "z-")
    detector_rot: tuple[str, ...] = ("z-",)
    r_i: tuple[float, float, float] = (0.0, 1.0, 0.0)

    q_conv_kwargs: dict[str, Any] = field(default_factory=dict)

    hxrd_n: tuple[float, float, float] = (0.0, 1.0, 0.0)
    hxrd_q: tuple[float, float, float] = (1.0, 0.0, 0.0)
    hxrd_geometry: str = "real"
    hxrd_kwargs: dict[str, Any] = field(default_factory=dict)

    init_area_detrot: str = "z-"
    init_area_tiltazimuth: str = "x+"
    ang2q_kwargs: dict[str, Any] = field(default_factory=dict)

    def make_hxrd(self, energy: float) -> xu.HXRD:
        qconversion = xu.QConversion(
            self.sample_rot,
            self.detector_rot,
            self.r_i,
            **self.q_conv_kwargs,
        )
        return xu.HXRD(
            self.hxrd_n,
            self.hxrd_q,
            geometry=self.hxrd_geometry,
            en=energy,
            qconv=qconversion,
            **self.hxrd_kwargs,
        )


@dataclass(slots=True)
class ExperimentConfig:
    """Session-level configuration for an RSM experiment."""

    base_path: Path
    pickle_dir: Path
    header: dict[str, Any]

    img_rel_path: str = "images"
    diff_motors: tuple[str, ...] = ("th", "chi", "phi", "tth")
    diff_config: DiffractometerConfig = field(default_factory=DiffractometerConfig)
    bins: tuple[int, int, int] = (80, 80, 100)
    rotation: int = 0
    h5_glob: str = "{sample}_scan*{scan_num}_master.h5"
    detector: str = "Pilatus300k"

    def __post_init__(self) -> None:
        self.pickle_dir.mkdir(parents=True, exist_ok=True)

    def find_h5(self, spec_dir: Path, sample: str, scan_num: int) -> Path | None:
        pattern = self.h5_glob.format(sample=sample, scan_num=scan_num)
        try:
            return next(spec_dir.glob(pattern))
        except StopIteration:
            return None

    def build_scans(self, scan_dict: dict[str, dict[str, Any]]) -> dict[str, ScanInfo]:
        scans: dict[str, ScanInfo] = {}
        for sample, sdict in scan_dict.items():
            spec_dir = self.base_path / sdict["spec_rel_path"]
            spec_path = spec_dir / sample

            for scan_num in sdict["scan_nums"]:
                h5_file = self.find_h5(spec_dir, sample, scan_num)
                img_dir = spec_dir if h5_file else self.base_path / self.img_rel_path
                scans[f"{sample}_{scan_num}"] = ScanInfo(
                    spec_path=spec_path,
                    h5_path=h5_file,
                    img_dir=img_dir,
                )
        return scans

    def process(
        self,
        scan_name: str,
        scan_info: ScanInfo,
        bins: tuple[int, int, int] | None = None,
        rotation: int | None = None,
        reprocess: bool = False,
        roi: tuple[int, int, int, int] | None = None,
        parallel: bool = True,
        strict: bool = False,
    ) -> RSMVolume | None:
        return process_scan(
            scan_name=scan_name,
            scan_info=scan_info,
            bins=self.bins if bins is None else bins,
            header=self.header,
            diff_motors=self.diff_motors,
            diff_config=self.diff_config,
            pickle_dir=self.pickle_dir,
            rotation=self.rotation if rotation is None else rotation,
            reprocess=reprocess,
            roi=roi,
            parallel=parallel,
            strict=strict,
            detector=self.detector,
        )


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

def load_images(
    scan_name: str,
    scan_info: ScanInfo,
    rotation: int = 0,
    parallel: bool = True,
    detector: str = "Pilatus300k",
) -> np.ndarray | None:
    """Load image stack for a scan (HDF5 or individual image files)."""
    mask = get_detector_mask(detector)
    if scan_info.h5_path:
        return read_image_stack(
            scan_info.h5_path,
            mask=mask,
            rotation=rotation,
        )
    spec_name, scan_num = get_scan_path_info(scan_name)
    img_files = find_image_files(
        scan_info.img_dir,
        stem=f"_{spec_name}_scan{scan_num[:-2]}_",
    )
    if not img_files:
        logger.warning(
            "No image files found for scan %s in %s",
            scan_name, scan_info.img_dir,
        )
        return None
    if parallel and len(img_files) > 1:
        return read_images_parallel(
            img_files, rotation=rotation, mask=mask,
        )
    return np.stack([
        read_image(f, mask=mask, rotation=rotation)
        for f in img_files
    ])


# -----------------------------------------------------------------------------
# Reciprocal-space mapping
# -----------------------------------------------------------------------------

def grid_img_data(
    img: np.ndarray,
    energy: float,
    UB: np.ndarray,
    angles: list[list[float]],
    header: dict[str, Any],
    diff_config: DiffractometerConfig,
    bins: tuple[int, int, int] = (200, 200, 200),
    roi: tuple[int, int, int, int] | None = None,
    mask_static_pixels: bool = True,
) -> RSMVolume:
    """
    Map image stack to reciprocal space and bin onto a 3D HKL grid.
    """
    hxrd = diff_config.make_hxrd(energy)

    header1 = dict(header)
    img = np.array(img, dtype=float, copy=True)

    if img.ndim != 3:
        raise ValueError(f"img must be a 3D stack of shape (n, ny, nx), got {img.shape}")

    if roi is not None:
        r0, r1, c0, c1 = roi
        img = img[:, r0:r1, c0:c1]
        header1["cch1"] = str(int(header["cch1"]) - r0)
        header1["cch2"] = str(int(header["cch2"]) - c0)

    if mask_static_pixels:
        std_dev = np.nanstd(img, axis=0)
        img[:, std_dev == 0] = np.nan

    header1["Nch1"], header1["Nch2"] = img.shape[1], img.shape[2]

    hxrd.Ang2Q.init_area(
        diff_config.init_area_detrot,
        diff_config.init_area_tiltazimuth,
        **header1,
    )  # type: ignore[arg-type]

    qx, qy, qz = hxrd.Ang2Q.area(
        *angles,
        UB=UB,
        **diff_config.ang2q_kwargs,
    )  # type: ignore[misc]

    gridder = xu.Gridder3D(*bins)
    gridder(qx, qy, qz, img)

    return RSMVolume(
        h=np.asarray(gridder.xaxis, dtype=float),
        k=np.asarray(gridder.yaxis, dtype=float),
        l=np.asarray(gridder.zaxis, dtype=float),
        intensity=np.asarray(gridder.data, dtype=float),
    )


def process_scan_data(
    scan_name: str,
    scan_info: ScanInfo,
    header: dict[str, Any],
    diff_motors: list[str] | tuple[str, ...],
    diff_config: DiffractometerConfig,
    bins: tuple[int, int, int],
    rotation: int = 0,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
) -> RSMVolume | None:
    """
    Pure processing path without cache I/O.
    """
    spec_file = scan_info.spec_path
    _, scan_num = get_scan_path_info(scan_name)

    try:
        energy, UB = get_energy_and_UB(spec_file, scan_num)
        angles = get_angles(spec_file, scan_num, diff_motors)
        img_arr = load_images(
            scan_name, scan_info,
            rotation=rotation,
            parallel=parallel,
            detector=detector,
        )

        if img_arr is None or img_arr.size == 0:
            logger.warning("No image data for scan %s", scan_name)
            return None

        return grid_img_data(
            img_arr,
            energy,
            UB,
            angles,
            header,
            diff_config,
            bins=bins,
            roi=roi,
        )
    except Exception:
        logger.exception("Error processing scan %s", scan_name)
        if strict:
            raise
        return None


def process_scan(
    scan_name: str,
    scan_info: ScanInfo,
    bins: tuple[int, int, int],
    header: dict[str, Any],
    diff_motors: list[str] | tuple[str, ...],
    diff_config: DiffractometerConfig,
    pickle_dir: Path,
    rotation: int = 0,
    reprocess: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
) -> RSMVolume | None:
    """
    Process one scan to an RSMVolume, using cache if available.
    """
    sample_name, scan_num = get_scan_path_info(scan_name)
    pickle_file = pickle_dir / f"{sample_name}_{scan_num[:-2]}.pkl"

    if pickle_file.is_file() and not reprocess:
        cached = _load_pickle(pickle_file)
        if isinstance(cached, RSMVolume):
            return cached
        if isinstance(cached, tuple) and len(cached) == 4:
            return RSMVolume(
                h=np.asarray(cached[0]),
                k=np.asarray(cached[1]),
                l=np.asarray(cached[2]),
                intensity=np.asarray(cached[3]),
            )

    volume = process_scan_data(
        scan_name=scan_name,
        scan_info=scan_info,
        header=header,
        diff_motors=diff_motors,
        diff_config=diff_config,
        bins=bins,
        rotation=rotation,
        roi=roi,
        parallel=parallel,
        strict=strict,
        detector=detector,
    )

    if volume is not None:
        _save_pickle(pickle_file, volume)

    return volume


# -----------------------------------------------------------------------------
# Volume combination / interpolation
# -----------------------------------------------------------------------------

def get_common_grid(
    volumes: list[RSMVolume],
    bins: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not volumes:
        raise ValueError("volumes must not be empty")

    hmin = min(np.nanmin(v.h) for v in volumes)
    hmax = max(np.nanmax(v.h) for v in volumes)
    kmin = min(np.nanmin(v.k) for v in volumes)
    kmax = max(np.nanmax(v.k) for v in volumes)
    lmin = min(np.nanmin(v.l) for v in volumes)
    lmax = max(np.nanmax(v.l) for v in volumes)

    h = np.linspace(hmin, hmax, bins[0])
    k = np.linspace(kmin, kmax, bins[1])
    l = np.linspace(lmin, lmax, bins[2])
    return h, k, l


def combine_grids(
    volumes: list[RSMVolume],
    bins: tuple[int, int, int],
) -> RSMVolume:
    """
    Re-grid each volume onto a common grid and sum intensity.
    """
    if not volumes:
        raise ValueError("volumes must not be empty")

    h, k, l = get_common_grid(volumes, bins)
    combined = np.zeros((len(h), len(k), len(l)), dtype=float)

    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    pts = np.column_stack((H.ravel(), K.ravel(), L.ravel()))

    for vol in volumes:
        vals = np.nan_to_num(vol.intensity, nan=0.0)
        rgi = RegularGridInterpolator(
            (vol.h, vol.k, vol.l),
            vals,
            bounds_error=False,
            fill_value=0.0,
        )
        combined += rgi(pts).reshape(H.shape)

    return RSMVolume(h=h, k=k, l=l, intensity=combined)


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
