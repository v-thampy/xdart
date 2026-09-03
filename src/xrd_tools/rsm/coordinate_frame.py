"""Closed coordinate-frame contract for reciprocal-space maps.

The numerical RSM gridder always consumes three Cartesian arrays.  This
module records what those arrays *mean* at the operation, artifact, and viewer
boundaries so an identity-valued crystal UB can never be mistaken for a
sample-fixed Cartesian-Q request.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class RSMCoordinateFrame(str, Enum):
    """The two production-supported reciprocal-space coordinate systems."""

    HKL = "hkl"
    Q_SAMPLE_CARTESIAN_XU = "q_sample_cartesian_xu"

    @property
    def axis_names(self) -> tuple[str, str, str]:
        if self is RSMCoordinateFrame.HKL:
            return ("h", "k", "l")
        return ("qx", "qy", "qz")

    @property
    def axis_units(self) -> tuple[str | None, str | None, str | None]:
        if self is RSMCoordinateFrame.HKL:
            return (None, None, None)
        return ("q_A^-1", "q_A^-1", "q_A^-1")

    @property
    def matrix_policy(self) -> str:
        if self is RSMCoordinateFrame.HKL:
            return "authenticated-source-ub-f8-v1"
        return "explicit-identity-ub-f8-v1"

    @property
    def display_name(self) -> str:
        if self is RSMCoordinateFrame.HKL:
            return "HKL (crystal reciprocal coordinates)"
        return "Q Cartesian (Qx, Qy, Qz; Å⁻¹)"

    @property
    def axis_symbols(self) -> tuple[str, str, str]:
        if self is RSMCoordinateFrame.HKL:
            return ("H", "K", "L")
        return ("Qx", "Qy", "Qz")

    @property
    def axis_labels(self) -> tuple[str, str, str]:
        if self is RSMCoordinateFrame.HKL:
            return self.axis_symbols
        return ("Qx (Å⁻¹)", "Qy (Å⁻¹)", "Qz (Å⁻¹)")


def rsm_coordinate_frame_from_axes(
    axis_names: tuple[str, str, str],
    axis_units: tuple[str | None, str | None, str | None],
) -> RSMCoordinateFrame:
    """Resolve one exact persisted axis descriptor to its closed frame."""

    if type(axis_names) is not tuple or type(axis_units) is not tuple:
        raise TypeError("RSM axis descriptor must use exact tuples")
    for frame in RSMCoordinateFrame:
        if axis_names == frame.axis_names and axis_units == frame.axis_units:
            return frame
    raise ValueError("RSM axis descriptor is unsupported")


def rsm_coordinate_matrix(
    coordinate_frame: RSMCoordinateFrame,
    source_ub: object,
) -> np.ndarray:
    """Admit and canonicalize the matrix driving one frame conversion.

    Cartesian Q always returns a fresh explicit float64 identity matrix.
    H/K/L requires a finite, numerically nonsingular source UB.  The frame is
    never inferred from the numerical value of the matrix.
    """

    if type(coordinate_frame) is not RSMCoordinateFrame:
        raise TypeError("RSM coordinate frame must be exact")
    if coordinate_frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU:
        if source_ub is not None:
            try:
                supplied = np.asarray(source_ub)
                numeric = np.ascontiguousarray(source_ub, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Cartesian-Q matrix must be omitted or identity"
                ) from error
            if (
                supplied.dtype.kind not in "fiu"
                or numeric.shape != (3, 3)
                or not np.all(np.isfinite(numeric))
                or not np.array_equal(numeric, np.eye(3, dtype=np.float64))
            ):
                raise ValueError(
                    "Cartesian-Q matrix must be omitted or identity"
                )
        return np.eye(3, dtype=np.float64)
    if source_ub is None:
        raise ValueError("H/K/L requires a source UB matrix")
    try:
        supplied = np.asarray(source_ub)
        matrix = np.array(
            source_ub,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError("H/K/L source UB matrix is invalid") from error
    if (
        supplied.dtype.kind not in "fiu"
        or matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or singular_values.shape != (3,)
        or not np.all(np.isfinite(singular_values))
        or singular_values[0] <= 0.0
        or singular_values[-1] <= singular_values[0] * 1e-12
    ):
        raise ValueError("H/K/L source UB matrix is invalid")
    return matrix


__all__ = [
    "RSMCoordinateFrame",
    "rsm_coordinate_frame_from_axes",
    "rsm_coordinate_matrix",
]
