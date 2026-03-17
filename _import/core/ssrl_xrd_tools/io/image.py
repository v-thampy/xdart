# ssrl_xrd_tools/io/image.py
"""
Detector-agnostic image I/O for ssrl_xrd_tools.

Handles: EDF, TIFF, CBF, MarCCD, raw binary, Eiger HDF5.
All reads go through fabio so format detection is automatic.
Masks come from pyFAI's detector registry, not hardcoded arrays.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import fabio
import h5py
import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".edf", ".tif", ".tiff", ".cbf", ".img", ".mar3450",
                  ".h5", ".hdf5", ".raw"}


def get_detector_mask(detector_name: str) -> np.ndarray | None:
    """
    Get bad-pixel mask from pyFAI detector registry.

    Parameters
    ----------
    detector_name : str
        pyFAI detector name e.g. 'Pilatus300k', 'Eiger1M'.

    Returns
    -------
    np.ndarray or None
        Boolean mask, or None if detector not found.
    """
    try:
        import pyFAI.detectors as detectors
        det = detectors.detector_factory(detector_name)
        return det.get_mask()
    except Exception:
        logger.warning("Could not get mask for detector: %s", detector_name)
        return None


def read_image(
    path: Path | str,
    frame: int = 0,
    rotation: int = 0,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Read a single detector image frame.

    Parameters
    ----------
    path : path-like
        Image file. Supported: EDF, TIFF, CBF, raw, HDF5 (NeXus/Eiger).
    frame : int
        Frame index for multi-frame files. Ignored for single-frame files.
    rotation : int
        Clockwise rotation in degrees, must be a multiple of 90.
    mask : ndarray of bool, optional
        Pixels set True are replaced with NaN.
    threshold : float, optional
        Pixels above this value are replaced with NaN.

    Returns
    -------
    np.ndarray
        Float64 image array, NaN where masked or above threshold.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".h5", ".hdf5"}:
        arr = _read_hdf5_frame(path, frame)
    else:
        arr = _read_fabio_frame(path, frame)

    arr = arr.astype(float, copy=False)

    if threshold is not None:
        arr[arr > threshold] = np.nan
    if mask is not None:
        arr[np.asarray(mask, dtype=bool)] = np.nan

    return apply_rotation(arr, rotation)


def read_image_stack(
    path: Path | str,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    rotation: int = 0,
    reduce: str | None = None,       # None | 'mean' | 'sum'
) -> np.ndarray:
    """
    Load all frames from a multi-frame file as a 3D stack.

    Parameters
    ----------
    reduce : {None, 'mean', 'sum'}
        If given, collapse the frame axis before returning.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".h5", ".hdf5"}:
        arr = _read_hdf5_stack(path)
    else:
        with fabio.open(path) as f:
            if f.nframes == 1:
                arr = f.data[np.newaxis]
            else:
                arr = np.stack([f.get_frame(i).data for i in range(f.nframes)])

    arr = arr.astype(float, copy=False)
    if threshold is not None:
        arr[arr > threshold] = np.nan
    if mask is not None:
        arr[:, np.asarray(mask, dtype=bool)] = np.nan

    arr = apply_rotation(arr, rotation)

    if reduce == 'mean':
        return np.nanmean(arr, axis=0)
    if reduce == 'sum':
        return np.nansum(arr, axis=0)
    return arr


def read_images_parallel(
    paths: Sequence[Path | str],
    rotation: int = 0,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Read a list of single-frame image files in parallel.
    Returns a 3D array of shape (n_files, ny, nx).
    """
    if not paths:
        return np.array([])

    frames = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(read_image)(p, mask=mask, threshold=threshold, rotation=rotation)
        for p in paths
    )
    return np.stack(frames)


def find_image_files(
    directory: Path | str,
    stem: str | None = None,
    exts: set[str] | None = None,
) -> list[Path]:
    """
    Find image files in a directory, optionally filtered by stem pattern and extension.
    Returns naturally sorted list (so scan_10 comes after scan_9).
    """
    from natsort import os_sorted   # optional dep, graceful fallback below
    directory = Path(directory)
    exts = exts or SUPPORTED_EXTS
    pattern = f"*{stem}*" if stem else "*"
    candidates = [p for p in directory.glob(pattern) if p.suffix.lower() in exts]
    try:
        return os_sorted(candidates)
    except Exception:
        return sorted(candidates)


def apply_rotation(arr: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate array by `rotation` degrees (multiple of 90), 2D or 3D."""
    if rotation % 90 != 0:
        raise ValueError(f"rotation must be a multiple of 90, got {rotation}")
    k = (rotation // 90) % 4
    if k == 0:
        return arr
    axes = (1, 2) if arr.ndim == 3 else None
    return np.rot90(arr, k, axes=axes) if axes else np.rot90(arr, k)


# --- private helpers ---

def _read_fabio_frame(path: Path, frame: int) -> np.ndarray:
    with fabio.open(path) as f:
        if f.nframes == 1 or frame == 0:
            return np.asarray(f.data)
        return np.asarray(f.get_frame(frame).data)


def _read_hdf5_frame(path: Path, frame: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        dataset = _find_hdf5_image_dataset(f)
        if dataset.ndim == 2:
            return dataset[:]
        return dataset[frame]


def _read_hdf5_stack(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return _find_hdf5_image_dataset(f)[:]


def _find_hdf5_image_dataset(f: h5py.File) -> h5py.Dataset:
    """Try standard NeXus paths, fall back to largest dataset."""
    for candidate in ("/entry/data/data", "/entry/instrument/detector/data", "/data"):
        if candidate in f:
            return f[candidate]
    # fallback: find the largest dataset
    found = {}
    f.visititems(lambda name, obj: found.update({name: obj}) if isinstance(obj, h5py.Dataset) else None)
    return max(found.values(), key=lambda d: d.size)
