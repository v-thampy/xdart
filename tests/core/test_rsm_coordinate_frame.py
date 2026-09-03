from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pytest

from xrd_tools.rsm.coordinate_frame import (
    RSMCoordinateFrame,
    rsm_coordinate_frame_from_axes,
    rsm_coordinate_matrix,
)


def test_xrayutilities_runtime_is_the_audited_version():
    pytest.importorskip("xrayutilities")
    assert version("xrayutilities") == "1.7.12"


def test_coordinate_frame_is_one_closed_two_value_contract():
    assert tuple(RSMCoordinateFrame) == (
        RSMCoordinateFrame.HKL,
        RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
    )
    assert RSMCoordinateFrame.HKL.axis_names == ("h", "k", "l")
    assert RSMCoordinateFrame.HKL.axis_units == (None, None, None)
    assert RSMCoordinateFrame.HKL.axis_symbols == ("H", "K", "L")
    assert RSMCoordinateFrame.HKL.axis_labels == ("H", "K", "L")
    assert RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.axis_names == (
        "qx",
        "qy",
        "qz",
    )
    assert RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.axis_units == (
        "q_A^-1",
        "q_A^-1",
        "q_A^-1",
    )
    assert RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.axis_symbols == (
        "Qx",
        "Qy",
        "Qz",
    )
    assert RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.axis_labels == (
        "Qx (Å⁻¹)",
        "Qy (Å⁻¹)",
        "Qz (Å⁻¹)",
    )


@pytest.mark.parametrize("frame", tuple(RSMCoordinateFrame))
def test_coordinate_frame_round_trips_only_its_exact_axis_descriptor(frame):
    assert rsm_coordinate_frame_from_axes(frame.axis_names, frame.axis_units) is frame


def test_coordinate_frame_refuses_unknown_or_mixed_axis_descriptors():
    with pytest.raises(ValueError, match="unsupported"):
        rsm_coordinate_frame_from_axes(
            ("qx", "qy", "qz"),
            (None, None, None),
        )
    with pytest.raises(TypeError, match="exact tuples"):
        rsm_coordinate_frame_from_axes(
            ["h", "k", "l"],  # type: ignore[arg-type]
            (None, None, None),
        )


def test_coordinate_matrix_never_infers_frame_from_identity():
    identity = np.eye(3, dtype=np.float64)
    hkl = rsm_coordinate_matrix(RSMCoordinateFrame.HKL, identity)
    q = rsm_coordinate_matrix(
        RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
        None,
    )

    for matrix in (hkl, q):
        assert matrix.dtype == np.float64
        assert matrix.flags.c_contiguous
        np.testing.assert_array_equal(matrix, identity)
    assert hkl is not identity
    assert q is not identity


def test_coordinate_matrix_enforces_the_closed_frame_policies():
    nonorthogonal = np.array(
        ((2.0, 0.4, 0.1), (0.0, 3.0, 0.2), (0.0, 0.0, 4.0)),
        dtype=np.float32,
    )
    admitted = rsm_coordinate_matrix(RSMCoordinateFrame.HKL, nonorthogonal)
    np.testing.assert_array_equal(admitted, nonorthogonal.astype(np.float64))

    with pytest.raises(ValueError, match="requires a source UB"):
        rsm_coordinate_matrix(RSMCoordinateFrame.HKL, None)
    with pytest.raises(ValueError, match="invalid"):
        rsm_coordinate_matrix(RSMCoordinateFrame.HKL, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="invalid"):
        rsm_coordinate_matrix(
            RSMCoordinateFrame.HKL,
            np.diag((1.0, 1.0, np.nan)),
        )
    with pytest.raises(ValueError, match="omitted or identity"):
        rsm_coordinate_matrix(
            RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
            nonorthogonal,
        )
