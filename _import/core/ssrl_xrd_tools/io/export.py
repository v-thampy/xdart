"""
Data export helpers for integrated diffraction results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _as_1d_array(data: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def _default_sigma(ydata: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(ydata))


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def write_xye(
    path: Path | str,
    xdata: np.ndarray | list[float],
    ydata: np.ndarray | list[float],
    variance: np.ndarray | list[float] | None = None,
) -> None:
    """
    Write 1D data to ``.xye`` format.

    Parameters
    ----------
    path : Path or str
        Output file path.
    xdata : ndarray or list of float
        X-axis values (e.g. 2theta or q).
    ydata : ndarray or list of float
        Intensity values.
    variance : ndarray or list of float, optional
        Error or variance-like column to write as the third column. If omitted,
        ``sqrt(abs(ydata))`` is used for compatibility with existing notebooks.
    """
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _as_1d_array(xdata)
    y = _as_1d_array(ydata)
    sigma = _default_sigma(y) if variance is None else _as_1d_array(variance)

    if x.shape != y.shape or sigma.shape != x.shape:
        raise ValueError("xdata, ydata, and variance must have matching shapes")

    np.savetxt(out_path, np.column_stack((x, y, sigma)), delimiter="\t")


def write_csv(
    path: Path | str,
    xdata: np.ndarray | list[float],
    ydata: np.ndarray | list[float],
    variance: np.ndarray | list[float] | None = None,
) -> None:
    """
    Write 1D data to three-column CSV format.

    Parameters
    ----------
    path : Path or str
        Output file path.
    xdata : ndarray or list of float
        X-axis values (e.g. 2theta or q).
    ydata : ndarray or list of float
        Intensity values.
    variance : ndarray or list of float, optional
        Error or variance-like column to write as the third column. If omitted,
        ``sqrt(abs(ydata))`` is used for compatibility with existing notebooks.
    """
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = _as_1d_array(xdata)
    y = _as_1d_array(ydata)
    sigma = _default_sigma(y) if variance is None else _as_1d_array(variance)

    if x.shape != y.shape or sigma.shape != x.shape:
        raise ValueError("xdata, ydata, and variance must have matching shapes")

    np.savetxt(out_path, np.column_stack((x, y, sigma)), delimiter=",")


def write_h5(
    path: Path | str,
    frame: int | str,
    q: np.ndarray | list[float],
    intensity: np.ndarray | list[float],
    iqchi: np.ndarray,
    q_2d: np.ndarray | list[float],
    chi: np.ndarray | list[float],
) -> None:
    """
    Write per-frame integration results to HDF5.

    The output layout matches the workflow described in ``CLAUDE.md`` and the
    existing experimental notebooks: each frame is stored in a group named by
    ``frame`` and contains datasets ``q``, ``I``, ``IQChi``, ``Q``, and ``Chi``.

    Parameters
    ----------
    path : Path or str
        Output HDF5 file path.
    frame : int or str
        Frame index or label used as the group name.
    q : ndarray or list of float
        1D radial axis for the integrated profile.
    intensity : ndarray or list of float
        1D integrated intensity.
    iqchi : np.ndarray
        2D cake intensity array.
    q_2d : ndarray or list of float
        Radial axis associated with ``iqchi``.
    chi : ndarray or list of float
        Azimuthal axis associated with ``iqchi``.
    """
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q_arr = _as_1d_array(q)
    intensity_arr = _as_1d_array(intensity)
    iqchi_arr = np.asarray(iqchi, dtype=float)
    q2_arr = _as_1d_array(q_2d)
    chi_arr = _as_1d_array(chi)

    if q_arr.shape != intensity_arr.shape:
        raise ValueError("q and intensity must have matching shapes")
    if iqchi_arr.ndim != 2:
        raise ValueError(f"iqchi must be 2D, got shape {iqchi_arr.shape}")

    with h5py.File(out_path, "a") as h5file:
        frame_group = h5file.require_group(str(frame))
        _replace_dataset(frame_group, "q", q_arr)
        _replace_dataset(frame_group, "I", intensity_arr)
        _replace_dataset(frame_group, "IQChi", iqchi_arr)
        _replace_dataset(frame_group, "Q", q2_arr)
        _replace_dataset(frame_group, "Chi", chi_arr)

    logger.debug("Wrote integrated data to %s [frame=%s]", out_path, frame)


__all__ = ["write_csv", "write_h5", "write_xye"]
