from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import xrayutilities as xu
from scipy.interpolate import RegularGridInterpolator

from ssrl_xrd_tools.rsm.volume import RSMVolume

if TYPE_CHECKING:
    from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig


logger = logging.getLogger(__name__)


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
