"""Tests for xrd_tools.io.export."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.io.export import read_xye, write_h5, write_xye


# XYE is written at "%.9g" (io/export.py), so a round trip is lossy in the 9th
# significant digit.  The tolerance below IS the accepted contract: values are
# preserved to better than 5e-9 relative, far beyond any XRD measurement.  The
# whole-number fixtures used elsewhere in this file cannot distinguish "%.9g"
# from the old "%.18e" and so pin nothing -- these values can.
XYE_ROUND_TRIP_RTOL = 5e-9


def test_write_xye(tmp_path):
    out = tmp_path / "test.xye"
    write_xye(out, [1, 2, 3], [4, 5, 6])

    arr = np.loadtxt(out)
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[:, 0], [1, 2, 3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(arr[:, 1], [4, 5, 6], rtol=1e-10, atol=1e-12)


def test_write_xye_round_trip_precision_contract(tmp_path):
    """Pin the accepted "%.9g" round-trip tolerance on rounding-SENSITIVE values."""
    out = tmp_path / "precision.xye"
    # 1.0000000049 is the WORST case for "%.9g": a mantissa just above 1.0
    # loses a full half-ulp of the 9th digit, i.e. 4.9e-9 relative.  Without a
    # value like it the fixture only reaches ~1e-9 and the bound is not pinned.
    x = np.array([1.0000000049, np.pi * 1e3, 1.0 / 3.0, 2.5e-7])
    y = np.array([1.0 / 7.0, 6.02214076e23, 1.2345678901234e4, np.e])
    sigma = np.sqrt(np.abs(y))
    write_xye(out, x, y, variance=sigma)

    rx, ry, rsigma = read_xye(out)
    for original, restored in ((x, rx), (y, ry), (sigma, rsigma)):
        np.testing.assert_allclose(
            restored, original, rtol=XYE_ROUND_TRIP_RTOL, atol=0.0
        )
    # And the tolerance is TIGHT: these values genuinely exercise it, so a
    # future format change that loses more precision cannot pass unnoticed.
    worst = max(
        float(np.max(np.abs((restored - original) / original)))
        for original, restored in ((x, rx), (y, ry), (sigma, rsigma))
    )
    assert worst > 4e-9, f"fixture no longer reaches the bound (worst={worst})"


def test_write_xye_with_variance(tmp_path):
    out = tmp_path / "test_var.xye"
    variance = np.array([0.1, 0.2, 0.3], dtype=float)
    write_xye(out, [1, 2, 3], [4, 5, 6], variance=variance)

    arr = np.loadtxt(out)
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[:, 2], variance, rtol=1e-10, atol=1e-12)


def test_write_h5(tmp_path):
    out = tmp_path / "test.h5"
    q = np.linspace(0, 5, 100)
    intensity = np.random.default_rng(0).random(100)
    iqchi = np.random.default_rng(1).random((50, 60))
    q_2d = np.linspace(0, 5, 50)
    chi = np.linspace(-180, 180, 60)

    write_h5(
        out,
        frame=0,
        q=q,
        intensity=intensity,
        iqchi=iqchi,
        q_2d=q_2d,
        chi=chi,
    )

    with h5py.File(out, "r") as h5:
        assert "0" in h5
        g0 = h5["0"]
        assert {"q", "I", "IQChi", "Q", "Chi"}.issubset(set(g0.keys()))
        assert g0["q"].shape == (100,)
        assert g0["I"].shape == (100,)
        assert g0["IQChi"].shape == (50, 60)
        assert g0["Q"].shape == (50,)
        assert g0["Chi"].shape == (60,)


def test_write_h5_multiple_frames(tmp_path):
    out = tmp_path / "multi.h5"
    rng = np.random.default_rng(2)

    for frame in (0, 1, 2):
        write_h5(
            out,
            frame=frame,
            q=np.linspace(0, 5, 100),
            intensity=rng.random(100),
            iqchi=rng.random((40, 30)),
            q_2d=np.linspace(0, 5, 40),
            chi=np.linspace(-180, 180, 30),
        )

    with h5py.File(out, "r") as h5:
        assert {"0", "1", "2"} == set(h5.keys())


def test_write_h5_overwrite_frame(tmp_path):
    out = tmp_path / "overwrite.h5"
    q = np.linspace(0, 5, 10)
    q_2d = np.linspace(0, 5, 5)
    chi = np.linspace(-180, 180, 6)

    first_i = np.full(10, 1.0)
    second_i = np.full(10, 2.0)
    first_iqchi = np.full((5, 6), 10.0)
    second_iqchi = np.full((5, 6), 20.0)

    write_h5(out, frame=0, q=q, intensity=first_i, iqchi=first_iqchi, q_2d=q_2d, chi=chi)
    write_h5(out, frame=0, q=q, intensity=second_i, iqchi=second_iqchi, q_2d=q_2d, chi=chi)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"0"}  # overwritten, not duplicated groups
        g0 = h5["0"]
        np.testing.assert_allclose(g0["I"][:], second_i, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g0["IQChi"][:], second_iqchi, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("suffix", (".nexus", ".nxs", ".cxi"))
def test_write_h5_refuses_ne_xus_suffixes(tmp_path, suffix):
    out = tmp_path / f"mislabelled{suffix}"
    with pytest.raises(ValueError, match="generic HDF5 export target"):
        write_h5(
            out,
            frame=0,
            q=[0.0],
            intensity=[1.0],
            iqchi=np.ones((1, 1)),
            q_2d=[0.0],
            chi=[0.0],
        )
    assert not out.exists()


def test_write_xye_shape_mismatch(tmp_path):
    out = tmp_path / "bad.xye"
    with pytest.raises(ValueError, match="matching shapes"):
        write_xye(out, [1, 2, 3], [4, 5])
