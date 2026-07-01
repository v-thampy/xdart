# xrd_tools/io/image.py
"""
Detector-agnostic image I/O for xrd_tools.

Handles: EDF, TIFF, CBF, MarCCD, raw binary, Eiger HDF5, NeXus (.nxs).
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
                  ".h5", ".hdf5", ".nxs", ".raw"}


def _is_eiger_master(path: Path) -> bool:
    """Return True if *path* looks like an Eiger HDF5 master file."""
    return path.stem.endswith("_master")


def resolve_detector_shape(
    detector: str | tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """
    Resolve a detector identifier to a ``(rows, cols)`` pixel shape.

    Parameters
    ----------
    detector : str, (rows, cols), or None
        * **str** — pyFAI detector name (e.g. ``'pilatus100k'``,
          ``'pilatus300k'``).  Looked up via
          ``pyFAI.detectors.detector_factory``.
        * **tuple** — passed through unchanged.
        * **None** — returns *None*.

    Returns
    -------
    (int, int) or None
    """
    if detector is None:
        return None
    if isinstance(detector, (tuple, list)):
        return tuple(detector)  # type: ignore[return-value]
    try:
        import pyFAI.detectors as detectors
        det = detectors.detector_factory(detector)
        return det.shape  # type: ignore[return-value]
    except Exception:
        logger.warning("Could not resolve detector shape for: %s", detector)
        return None


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


def load_mask(
    mask: np.ndarray | Path | str,
    threshold: float | None = None,
    data: np.ndarray | None = None,
) -> np.ndarray:
    """
    Load or build a boolean bad-pixel mask from various inputs.

    Parameters
    ----------
    mask : ndarray, path-like
        * **ndarray** — used directly.  Boolean arrays are returned as-is.
          Integer/float arrays are converted: non-zero values → ``True`` (bad).
        * **str / Path** — path to a mask file (e.g. ``.edf``, ``.tif``).
          The file is read via :func:`read_image` and non-zero pixels are
          treated as bad.
    threshold : float, optional
        If given **and** *data* is provided, pixels in *data* exceeding
        this value are OR-ed into the mask.
    data : ndarray, optional
        Image data used for the *threshold* mask.  Ignored if *threshold*
        is None.

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` = bad pixel.
    """
    if isinstance(mask, (str, Path)):
        arr = read_image(Path(mask))
        bool_mask = np.asarray(arr, dtype=float) != 0.0
        # read_image returns NaN for bad pixels in some formats;
        # treat NaN as bad too
        bool_mask |= np.isnan(arr)
    elif isinstance(mask, np.ndarray):
        if mask.dtype == bool:
            bool_mask = mask.copy()
        else:
            bool_mask = np.asarray(mask, dtype=float) != 0.0
    else:
        raise TypeError(
            f"mask must be an ndarray or a file path, got {type(mask).__name__}"
        )

    if threshold is not None and data is not None:
        bool_mask = bool_mask | (data > threshold) | np.isnan(data)

    return bool_mask


def read_image(
    path: Path | str,
    frame: int = 0,
    rotation: int = 0,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    detector_shape: tuple[int, int] | None = None,
    detector: str | tuple[int, int] | None = None,
    raw_dtype: str = "int32",
    raw_header_skip: int = 0,
) -> np.ndarray:
    """
    Read a single detector image frame.

    Parameters
    ----------
    path : path-like
        Image file. Supported: EDF, TIFF, CBF, raw binary, HDF5 (NeXus/Eiger),
        NPY (saved NumPy array, e.g. a mask).
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
    detector : str or (rows, cols), optional
        Alternative to *detector_shape*.  If a string (e.g.
        ``'pilatus100k'``, ``'pilatus300k'``), the shape is resolved via
        ``pyFAI.detectors.detector_factory``.  A tuple is treated the
        same as *detector_shape*.  If both *detector* and
        *detector_shape* are given, *detector_shape* takes precedence.
    raw_dtype : str
        NumPy dtype string for raw binary files (default ``'int32'``).
    raw_header_skip : int
        Bytes to skip at the start of raw binary files (default ``0``).

    Returns
    -------
    np.ndarray
        Float64 image array, NaN where masked or above threshold.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Resolve detector shape: explicit tuple wins, then detector name lookup
    shape = detector_shape or resolve_detector_shape(detector)

    if ext == ".npy":
        # Saved NumPy array (e.g. a boolean/integer mask).  fabio can't open
        # these, so load directly; index a stacked array by frame.
        arr = np.load(path)
        if arr.ndim > 2:
            arr = arr[frame]
    elif ext in {".h5", ".hdf5"} and _is_eiger_master(path):
        # Eiger master files — fabio handles external data-file linking
        arr = _read_fabio_frame(path, frame)
    elif ext in {".h5", ".hdf5", ".nxs"}:
        # Reject processed xdart scan files up front (before fabio, which
        # might otherwise pick up a reduced dataset) — they carry no raw
        # detector image; callers should use io.read.get_raw_frame.
        _reject_if_processed_xdart(path)
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
            if shape is None:
                raise
            logger.debug("fabio could not open %s, falling back to raw binary",
                         path)
            arr = _read_raw_binary(path, shape, raw_dtype, raw_header_skip)

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

    if reduce in {"mean", "sum"}:
        if ext in {".h5", ".hdf5", ".nxs"} and not _is_eiger_master(path):
            _reject_if_processed_xdart(path)
        return _reduce_image_stack(
            path,
            mask=mask,
            threshold=threshold,
            rotation=rotation,
            reduce=reduce,
        )

    if ext in {".h5", ".hdf5", ".nxs"} and not _is_eiger_master(path):
        _reject_if_processed_xdart(path)
        # Non-Eiger HDF5 / NeXus — try fabio first, fall back to h5py
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

    return arr


def _reduce_image_stack(
    path: Path,
    *,
    mask: np.ndarray | None,
    threshold: float | None,
    rotation: int,
    reduce: str,
) -> np.ndarray:
    """Fold a multi-frame image file without materializing the full stack."""
    n_frames = count_frames(path)
    if n_frames <= 0:
        raise ValueError(f"Could not determine frame count for {path}")

    total: np.ndarray | None = None
    counts: np.ndarray | None = None
    for frame_idx in range(n_frames):
        frame = read_image(
            path,
            frame=frame_idx,
            mask=mask,
            threshold=threshold,
            rotation=rotation,
        )
        if total is None:
            total = np.zeros_like(frame, dtype=float)
            if reduce == "mean":
                counts = np.zeros(frame.shape, dtype=np.int64)
        valid = ~np.isnan(frame)
        total[valid] += frame[valid]
        if counts is not None:
            counts[valid] += 1

    if total is None:
        return np.empty((0, 0), dtype=float)
    if reduce == "sum":
        return total

    assert counts is not None
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / counts
    mean[counts == 0] = np.nan
    return mean


def read_images_parallel(
    paths: Sequence[Path | str],
    rotation: int = 0,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    n_jobs: int = -1,
    detector_shape: tuple[int, int] | None = None,
    detector: str | tuple[int, int] | None = None,
    raw_dtype: str = "int32",
    raw_header_skip: int = 0,
) -> np.ndarray:
    """
    Read a list of single-frame image files in parallel.
    Returns a 3D array of shape (n_files, ny, nx).

    Parameters
    ----------
    detector : str or (rows, cols), optional
        See :func:`read_image`.
    """
    if not paths:
        return np.empty((0, 0, 0))

    frames = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(read_image)(
            p, mask=mask, threshold=threshold, rotation=rotation,
            detector_shape=detector_shape, detector=detector,
            raw_dtype=raw_dtype, raw_header_skip=raw_header_skip,
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


def read_nexus_frame(
    path: Path | str,
    frame: int = 0,
    dataset_path: str | None = None,
) -> np.ndarray:
    """
    Read a single image frame from a NeXus / HDF5 file.

    This is a convenience wrapper that opens the file, locates the
    image dataset (either via *dataset_path* or automatic discovery),
    and returns a single frame as a 2-D float array.

    Parameters
    ----------
    path : path-like
        NeXus (``.nxs``) or HDF5 file.
    frame : int
        0-based frame index.  Ignored when the dataset is 2-D.
    dataset_path : str, optional
        Explicit internal HDF5 path to the image dataset
        (e.g. ``'/entry/data/data'``).  If *None*, the dataset is
        found automatically via :func:`_find_hdf5_image_dataset`.

    Returns
    -------
    np.ndarray
        Float64 2-D image array.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        if dataset_path is not None:
            ds = f[dataset_path]
        else:
            ds = _find_hdf5_image_dataset(f)
        if ds.ndim == 2:
            return np.asarray(ds[:], dtype=float)
        return np.asarray(ds[frame], dtype=float)


def nexus_info(path: Path | str) -> dict:
    """
    Return metadata about the image dataset in a NeXus / HDF5 file.

    Parameters
    ----------
    path : path-like
        NeXus or HDF5 file.

    Returns
    -------
    dict
        Keys: ``dataset_path`` (str), ``shape`` (tuple), ``dtype``,
        ``nframes`` (int).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        ds = _find_hdf5_image_dataset(f)
        nframes = ds.shape[0] if ds.ndim >= 3 else 1
        return {
            "dataset_path": ds.name,
            "shape": ds.shape,
            "dtype": str(ds.dtype),
            "nframes": nframes,
        }


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
        if ext in {".h5", ".hdf5", ".nxs"} and not _is_eiger_master(path):
            # Non-Eiger HDF5 / NeXus — try fabio, fall back to h5py
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


def _reject_if_processed_xdart(path: Path) -> None:
    """Raise ``ValueError`` if ``path`` is a processed xdart v2 scan file.

    Such files store reduced data (``integrated_1d``/``integrated_2d`` are
    ndim>=2 NXData) and no raw detector frames — reading them as images
    would silently return an integrated-pattern stack.  Raw frames live in
    the original master; use ``io.read.get_raw_frame``.
    """
    try:
        with h5py.File(path, "r") as f:
            processed = any(
                p in f for p in ("entry/integrated_1d", "entry/integrated_2d")
            )
    except OSError:
        return  # not openable as HDF5 here — let the normal path report it
    if processed:
        raise ValueError(
            f"{path} is a processed xdart scan file (no raw detector image "
            "inside); use io.read.get_raw_frame to read raw frames via the "
            "stored source pointer."
        )


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
    header_skip: int = 0,
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
    """Locate the image dataset inside an HDF5/NeXus file.

    Search order:

    1. Well-known fixed paths (NeXus and common beamline conventions).
    2. NXdata groups found via ``NX_class`` attributes — look for a
       dataset whose ``signal`` attribute marks the default data, or
       pick the first 2-D+ dataset.
    3. NXdetector groups — ``/entry/**/NXdetector/data``.
    4. Fallback: the largest 2-D+ dataset anywhere in the file.
    """
    # --- 0. Reject processed xdart v2 files ---------------------------------
    # A processed scan file stores *reduced* data (integrated_1d/2d are
    # NXData with ndim>=2), not raw detector frames — without this guard the
    # search below would return ``integrated_1d`` (shape (N, n_q)) and the
    # caller would display an integrated-pattern stack as if it were an
    # image.  Raw frames for such a file live in the original detector
    # master; use ``xrd_tools.io.read.get_raw_frame`` (it resolves the
    # per-frame source pointer) instead.
    if any(p in f for p in ("entry/integrated_1d", "entry/integrated_2d")):
        raise ValueError(
            f"{f.filename} is a processed xdart scan file (no raw detector "
            "image inside); use io.read.get_raw_frame to read raw frames via "
            "the stored source pointer."
        )

    # --- 1. Fixed candidate paths -------------------------------------------
    _FIXED_PATHS = (
        "/entry/data/data",
        "/entry/instrument/detector/data",
        "/entry/instrument/detector/data_000001",
        "/entry/measurement/data",
        "/entry/data",
        "/data",
        "/entry/instrument/pilatus/data",
        "/entry/instrument/eiger/data",
    )
    for candidate in _FIXED_PATHS:
        if candidate in f:
            obj = f[candidate]
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                return obj  # type: ignore[return-value]

    # --- 2. NXdata groups (signal attribute) ---------------------------------
    def _search_nxdata(grp: h5py.Group) -> h5py.Dataset | None:
        for name, item in grp.items():
            nx_class = item.attrs.get("NX_class", b"")
            if isinstance(nx_class, bytes):
                nx_class = nx_class.decode("utf-8", errors="replace")
            if nx_class == "NXdata" and isinstance(item, h5py.Group):
                signal_name = item.attrs.get("signal", None)
                if signal_name is not None:
                    if isinstance(signal_name, bytes):
                        signal_name = signal_name.decode("utf-8", errors="replace")
                    if signal_name in item and isinstance(item[signal_name], h5py.Dataset):
                        ds = item[signal_name]
                        if ds.ndim >= 2:
                            return ds  # type: ignore[return-value]
                # No signal attribute — pick first ≥2-D dataset
                for sub_name, sub_item in item.items():
                    if isinstance(sub_item, h5py.Dataset) and sub_item.ndim >= 2:
                        return sub_item  # type: ignore[return-value]
            # Recurse into subgroups
            if isinstance(item, h5py.Group):
                result = _search_nxdata(item)
                if result is not None:
                    return result
        return None

    nxdata_result = _search_nxdata(f)
    if nxdata_result is not None:
        return nxdata_result

    # --- 3. NXdetector groups ------------------------------------------------
    def _search_nxdetector(grp: h5py.Group) -> h5py.Dataset | None:
        for name, item in grp.items():
            nx_class = item.attrs.get("NX_class", b"")
            if isinstance(nx_class, bytes):
                nx_class = nx_class.decode("utf-8", errors="replace")
            if nx_class == "NXdetector" and isinstance(item, h5py.Group):
                if "data" in item and isinstance(item["data"], h5py.Dataset):
                    return item["data"]  # type: ignore[return-value]
            if isinstance(item, h5py.Group):
                result = _search_nxdetector(item)
                if result is not None:
                    return result
        return None

    nxdet_result = _search_nxdetector(f)
    if nxdet_result is not None:
        return nxdet_result

    # --- 4. Fallback: largest ≥2-D dataset -----------------------------------
    found: dict[str, h5py.Dataset] = {}

    def _visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            found[name] = obj  # type: ignore[assignment]

    f.visititems(_visitor)
    if not found:
        raise ValueError(f"No 2-D+ dataset found in {f.filename}")
    return max(found.values(), key=lambda d: d.size)
