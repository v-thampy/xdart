"""Tests for ssrl_xrd_tools.io.export."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.io.export import write_csv, write_h5, write_xye


def test_write_xye(tmp_path):
    out = tmp_path / "test.xye"
    write_xye(out, [1, 2, 3], [4, 5, 6])

    arr = np.loadtxt(out)
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[:, 0], [1, 2, 3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(arr[:, 1], [4, 5, 6], rtol=1e-10, atol=1e-12)


def test_write_xye_with_variance(tmp_path):
    out = tmp_path / "test_var.xye"
    variance = np.array([0.1, 0.2, 0.3], dtype=float)
    write_xye(out, [1, 2, 3], [4, 5, 6], variance=variance)

    arr = np.loadtxt(out)
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[:, 2], variance, rtol=1e-10, atol=1e-12)


def test_write_csv(tmp_path):
    out = tmp_path / "test.csv"
    write_csv(out, [1, 2, 3], [4, 5, 6])

    arr = np.loadtxt(out, delimiter=",")
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[:, 0], [1, 2, 3], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(arr[:, 1], [4, 5, 6], rtol=1e-10, atol=1e-12)


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


def test_write_xye_shape_mismatch(tmp_path):
    out = tmp_path / "bad.xye"
    with pytest.raises(ValueError, match="matching shapes"):
        write_xye(out, [1, 2, 3], [4, 5])
