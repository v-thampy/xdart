# ssrl_xrd_tools/io/image.py
"""
Detector-agnostic image I/O for ssrl_xrd_tools.

Handles: EDF, TIFF, CBF, MarCCD, raw binary, Eiger HDF5.
All reads go through fabio so format detection is automatic.
Eiger master files (``*_master.h5``) are read via fabio's EigerImage,
which handles external data-file linking transparently.  Other HDF5
files (NeXus, etc.) fall back to raw h5py when fabio cannot open them.
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


def _is_eiger_master(path: Path) -> bool:
    """Return True if *path* looks like an Eiger HDF5 master file."""
    return path.stem.endswith("_master")


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
        mask = det.get_mask()
        return np.asarray(mask) if mask is not None else None
    except Exception:
        logger.warning("Could not get mask for detector: %s", detector_name)
        return None


def read_image(
    path: Path | str,
    frame: int = 0,
    rotation: int = 0,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    detector_shape: tuple[int, int] | None = None,
    raw_dtype: str = "int32",
    raw_header_skip: int = 4096,
) -> np.ndarray:
    """
    Read a single detector image frame.

    Parameters
    ----------
    path : path-like
        Image file. Supported: EDF, TIFF, CBF, raw binary, HDF5 (NeXus/Eiger).
    frame : int
        Frame index for multi-frame files. Ignored for single-frame files.
    rotation : int
        Clockwise rotation in degrees, must be a multiple of 90.
    mask : ndarray of bool, optional
        Pixels set True are replaced with NaN.
    threshold : float, optional
        Pixels above this value are replaced with NaN.
    detector_shape : (rows, cols), optional
        Detector dimensions for raw binary files.  Required when fabio
        cannot auto-detect the format.  Also used as the reshape target
        for the fallback binary reader.
    raw_dtype : str
        NumPy dtype string for raw binary files (default ``'int32'``).
    raw_header_skip : int
        Bytes to skip at the start of raw binary files (default ``4096``).

    Returns
    -------
    np.ndarray
        Float64 image array, NaN where masked or above threshold.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".h5", ".hdf5"} and _is_eiger_master(path):
        # Eiger master files — fabio handles external data-file linking
        arr = _read_fabio_frame(path, frame)
    elif ext in {".h5", ".hdf5"}:
        # Other HDF5 (NeXus, etc.) — try fabio first, fall back to h5py
        try:
            arr = _read_fabio_frame(path, frame)
        except Exception:
            logger.debug("fabio could not open %s, falling back to h5py", path)
            arr = _read_hdf5_frame(path, frame)
    else:
        # EDF, TIFF, CBF, raw, etc. — try fabio, fall back to raw binary
        try:
            arr = _read_fabio_frame(path, frame)
        except Exception:
            if detector_shape is None:
                raise
            logger.debug("fabio could not open %s, falling back to raw binary",
                         path)
            arr = _read_raw_binary(path, detector_shape, raw_dtype,
                                   raw_header_skip)

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

    if ext in {".h5", ".hdf5"} and not _is_eiger_master(path):
        # Non-Eiger HDF5 — try fabio first, fall back to h5py
        try:
            arr = _read_fabio_stack(path)
        except Exception:
            logger.debug("fabio could not stack %s, falling back to h5py", path)
            arr = _read_hdf5_stack(path)
    else:
        # Eiger master files + all non-HDF5 formats go through fabio
        arr = _read_fabio_stack(path)

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
    detector_shape: tuple[int, int] | None = None,
    raw_dtype: str = "int32",
    raw_header_skip: int = 4096,
) -> np.ndarray:
    """
    Read a list of single-frame image files in parallel.
    Returns a 3D array of shape (n_files, ny, nx).
    """
    if not paths:
        return np.array([])

    frames = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(read_image)(
            p, mask=mask, threshold=threshold, rotation=rotation,
            detector_shape=detector_shape, raw_dtype=raw_dtype,
            raw_header_skip=raw_header_skip,
        )
        for p in paths
    )
    return np.stack([f for f in frames if f is not None])


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


def count_frames(path: Path | str) -> int:
    """
    Return the number of frames in an image file.

    Parameters
    ----------
    path : path-like
        Image file (any format supported by fabio, or HDF5).

    Returns
    -------
    int
        Number of frames, or 0 if the file cannot be read.
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in {".h5", ".hdf5"} and not _is_eiger_master(path):
            # Non-Eiger HDF5 — try fabio, fall back to h5py
            try:
                with fabio.open(path) as f:
                    return f.nframes
            except Exception:
                with h5py.File(path, "r") as f:
                    ds = _find_hdf5_image_dataset(f)
                    return ds.shape[0] if ds.ndim >= 3 else 1
        else:
            with fabio.open(path) as f:
                return f.nframes
    except Exception:
        logger.warning("Could not determine frame count for %s", path)
        return 0


# --- private helpers ---

def _read_fabio_frame(path: Path, frame: int) -> np.ndarray:
    """Read a single frame via fabio (works for all formats incl. Eiger)."""
    with fabio.open(path) as f:
        if f.nframes == 1 or frame == 0:
            return np.asarray(f.data)
        return np.asarray(f.get_frame(frame).data)


def _read_fabio_stack(path: Path) -> np.ndarray:
    """Read all frames via fabio as a 3D stack."""
    with fabio.open(path) as f:
        if f.nframes == 1:
            return f.data[np.newaxis]
        return np.stack([f.get_frame(i).data for i in range(f.nframes)])


def _read_hdf5_frame(path: Path, frame: int) -> np.ndarray:
    """Read a single frame via raw h5py (fallback for non-Eiger HDF5)."""
    with h5py.File(path, "r") as f:
        dataset = _find_hdf5_image_dataset(f)
        if dataset.ndim == 2:
            return dataset[:]
        return dataset[frame]


def _read_hdf5_stack(path: Path) -> np.ndarray:
    """Read all frames via raw h5py (fallback for non-Eiger HDF5)."""
    with h5py.File(path, "r") as f:
        return _find_hdf5_image_dataset(f)[:]


def _read_raw_binary(
    path: Path,
    shape: tuple[int, int],
    dtype: str = "int32",
    header_skip: int = 4096,
) -> np.ndarray:
    """
    Read a raw binary detector dump (no fabio support needed).

    Parameters
    ----------
    path : Path
        Raw binary file.
    shape : (rows, cols)
        Detector pixel dimensions.
    dtype : str
        NumPy dtype string.
    header_skip : int
        Bytes to skip before pixel data.
    """
    with open(path, "rb") as fh:
        fh.seek(header_skip)
        arr = np.frombuffer(fh.read(), dtype=dtype)
    return arr.reshape(shape)


def _find_hdf5_image_dataset(f: h5py.File) -> h5py.Dataset:
    """Try standard NeXus paths, fall back to largest dataset."""
    for candidate in ("/entry/data/data", "/entry/instrument/detector/data", "/data"):
        if candidate in f and isinstance(f[candidate], h5py.Dataset):
            return f[candidate]  # type: ignore[return-value]
    # fallback: find the largest dataset
    found = {}
    f.visititems(lambda name, obj: found.update({name: obj}) if isinstance(obj, h5py.Dataset) else None)
    return max(found.values(), key=lambda d: d.size)
