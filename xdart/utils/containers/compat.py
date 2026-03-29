"""
xdart/utils/containers/compat.py

Backward-compatibility readers for legacy xdart HDF5 files.

Old format (int_1d_data_static / int_2d_data_static) stores plain numpy
arrays under the following dataset names:

  1D: norm, ttheta, q, i_qz, qz, i_qxy, qxy
  2D: i_tthChi, i_qChi, ttheta, q, chi, i_QxyQz, qz, qxy,
      q_from_tth, tth_from_q

Return types are the new ssrl_xrd_tools containers
(IntegrationResult1D / IntegrationResult2D).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

if TYPE_CHECKING:
    import h5py


def _has_data(arr: np.ndarray) -> bool:
    """Return True when *arr* is not the scalar-0 sentinel used by the old format."""
    return arr.ndim > 0 and arr.size > 0


def read_legacy_1d(grp: "h5py.Group") -> IntegrationResult1D:
    """Read an old ``int_1d_data_static`` HDF5 group.

    Prefers ``q`` / ``q_A^-1`` when available; falls back to ``ttheta`` /
    ``2th_deg``.  The legacy format never stored ``sigma``.

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group that was written by ``int_1d_data_static.to_hdf5``.

    Returns
    -------
    IntegrationResult1D
    """
    def _load(key: str) -> np.ndarray:
        if key in grp:
            return np.asarray(grp[key], dtype=float)
        return np.array(0.0)

    norm = _load("norm")
    q = _load("q")
    ttheta = _load("ttheta")

    if _has_data(q):
        radial = q
        unit = "q_A^-1"
    elif _has_data(ttheta):
        radial = ttheta
        unit = "2th_deg"
    else:
        # Empty / uninitialised arch — return an empty result
        return IntegrationResult1D(
            radial=np.array([], dtype=float),
            intensity=np.array([], dtype=float),
            unit="q_A^-1",
        )

    intensity = norm if _has_data(norm) else np.zeros_like(radial)
    return IntegrationResult1D(
        radial=radial,
        intensity=intensity,
        sigma=None,
        unit=unit,
    )


def read_legacy_2d(grp: "h5py.Group") -> IntegrationResult2D:
    """Read an old ``int_2d_data_static`` HDF5 group.

    Chooses the primary radial axis by inspecting which datasets were actually
    populated (non-scalar 0):

    * If ``q`` is present → use ``i_qChi`` as intensity, ``unit="q_A^-1"``
    * Else use ``i_tthChi`` as intensity, ``unit="2th_deg"``

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group written by ``int_2d_data_static.to_hdf5``.

    Returns
    -------
    IntegrationResult2D
        Standard (radial, chi) cake result.  ``azimuthal_unit`` is always
        ``"chi_deg"``.  For the legacy GI QxyQz result use
        :func:`read_legacy_2d_gi` instead.
    """
    def _load(key: str) -> np.ndarray:
        if key in grp:
            return np.asarray(grp[key], dtype=float)
        return np.array(0.0)

    q = _load("q")
    ttheta = _load("ttheta")
    chi = _load("chi")
    i_qChi = _load("i_qChi")
    i_tthChi = _load("i_tthChi")

    if _has_data(q) and _has_data(i_qChi):
        radial = q
        intensity = i_qChi
        unit = "q_A^-1"
    elif _has_data(ttheta) and _has_data(i_tthChi):
        radial = ttheta
        intensity = i_tthChi
        unit = "2th_deg"
    else:
        # Nothing integrated yet
        return IntegrationResult2D(
            radial=np.array([], dtype=float),
            azimuthal=np.array([], dtype=float),
            intensity=np.zeros((0, 0), dtype=float),
            unit="q_A^-1",
            azimuthal_unit="chi_deg",
        )

    azimuthal = chi if _has_data(chi) else np.array([], dtype=float)

    # Intensity shape from the old format: (npt_azim, npt_rad) — the same
    # convention as raw pyFAI output.  ssrl_xrd_tools uses (npt_rad, npt_azim),
    # so transpose here.
    if intensity.ndim == 2:
        intensity = intensity.T

    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity,
        sigma=None,
        unit=unit,
        azimuthal_unit="chi_deg",
    )


def read_legacy_2d_gi(grp: "h5py.Group") -> IntegrationResult2D | None:
    """Read the GI ``i_QxyQz`` dataset from an ``int_2d_data_static`` group.

    Returns ``None`` when the group contains no GI data (i.e. the
    ``i_QxyQz`` field was never populated).

    Parameters
    ----------
    grp : h5py.Group
        HDF5 group written by ``int_2d_data_static.to_hdf5``.

    Returns
    -------
    IntegrationResult2D or None
        ``radial=qxy``, ``azimuthal=qz``, ``unit="qip_A^-1"``,
        ``azimuthal_unit="qoop_A^-1"``.
    """
    def _load(key: str) -> np.ndarray:
        if key in grp:
            return np.asarray(grp[key], dtype=float)
        return np.array(0.0)

    i_QxyQz = _load("i_QxyQz")
    qxy = _load("qxy")
    qz = _load("qz")

    if not (_has_data(i_QxyQz) and _has_data(qxy) and _has_data(qz)):
        return None

    intensity = i_QxyQz
    if intensity.ndim == 2:
        intensity = intensity.T

    return IntegrationResult2D(
        radial=qxy,
        azimuthal=qz,
        intensity=intensity,
        sigma=None,
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )


__all__ = [
    "read_legacy_1d",
    "read_legacy_2d",
    "read_legacy_2d_gi",
]
