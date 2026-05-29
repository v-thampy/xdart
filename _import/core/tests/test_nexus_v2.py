"""Tests for the v2 NeXus reader (``read_scan`` / ``read_stitched``).

We hand-craft a small NXroot fixture with pure h5py — no xdart
dependency, no captured-file dependency — and round-trip it through
the v2 reader.  The fixture corresponds to a 5-frame psic scan.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.geometry import DiffractometerGeometry
from ssrl_xrd_tools.core.provenance import write_provenance
from ssrl_xrd_tools.io.nexus import read_scan, read_stitched


N_FRAMES = 5
N_Q = 64
N_CHI = 32


@pytest.fixture
def v2_fixture(tmp_path):
    """A minimal v2-conformant NXroot file for a 5-frame psic scan."""
    p = tmp_path / "psic_5frame.nxs"

    rng = np.random.default_rng(0)
    q = np.linspace(0.5, 5.0, N_Q).astype(np.float32)
    chi = np.linspace(-180.0, 180.0, N_CHI, endpoint=False).astype(np.float32)
    frame_index = np.arange(N_FRAMES, dtype=np.int32)

    intensity_1d = rng.random((N_FRAMES, N_Q), dtype=np.float32)
    sigma_1d = np.sqrt(intensity_1d).astype(np.float32)
    intensity_2d = rng.random((N_FRAMES, N_CHI, N_Q), dtype=np.float32)

    # psic motors (sample + detector)
    eta = np.linspace(0.0, 1.0, N_FRAMES, dtype=np.float32)
    chi_m = np.zeros(N_FRAMES, dtype=np.float32)
    phi = np.zeros(N_FRAMES, dtype=np.float32)
    mu = np.zeros(N_FRAMES, dtype=np.float32)
    del_ = np.linspace(10.0, 20.0, N_FRAMES, dtype=np.float32)
    nu = np.linspace(2.0, 4.0, N_FRAMES, dtype=np.float32)

    geom = DiffractometerGeometry.psic()
    derived = geom.derive_per_frame(
        {"nu": nu, "del": del_, "eta": eta}
    )

    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "integrated_1d"

        # integrated_1d ------------------------------------------------
        g1 = entry.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.attrs["signal"] = "intensity"
        g1.attrs["axes"] = ["frame_index", "q"]
        g1.create_dataset("intensity", data=intensity_1d)
        g1.create_dataset("sigma", data=sigma_1d)
        q_ds = g1.create_dataset("q", data=q)
        q_ds.attrs["units"] = "1/angstrom"
        g1.create_dataset("frame_index", data=frame_index)

        # integrated_2d ------------------------------------------------
        g2 = entry.create_group("integrated_2d")
        g2.attrs["NX_class"] = "NXdata"
        g2.attrs["signal"] = "intensity"
        g2.attrs["axes"] = ["frame_index", "chi", "q"]
        g2.create_dataset("intensity", data=intensity_2d)
        g2.create_dataset("q", data=q)
        chi_ds = g2.create_dataset("chi", data=chi)
        chi_ds.attrs["units"] = "deg"
        g2.create_dataset("frame_index", data=frame_index)

        # per_frame_geometry ------------------------------------------
        gg = entry.create_group("per_frame_geometry")
        for key in ("rot1", "rot2", "rot3", "incident_angle"):
            ds = gg.create_dataset(key, data=derived[key].astype(np.float32))
            ds.attrs["units"] = "rad" if key.startswith("rot") else "deg"
        gg.create_dataset("frame_index", data=frame_index)

        # sample positioners ------------------------------------------
        sp = entry.create_group("sample/positioners")
        sp.attrs["NX_class"] = "NXcollection"
        for name, arr in [("eta", eta), ("chi", chi_m), ("phi", phi), ("mu", mu)]:
            pg = sp.create_group(name)
            pg.attrs["NX_class"] = "NXpositioner"
            pg.create_dataset("value", data=arr)
            pg["value"].attrs["units"] = "deg"

        # detector positioners ----------------------------------------
        dp = entry.create_group("instrument/detector/positioners")
        dp.attrs["NX_class"] = "NXcollection"
        for name, arr in [("del", del_), ("nu", nu)]:
            pg = dp.create_group(name)
            pg.attrs["NX_class"] = "NXpositioner"
            pg.create_dataset("value", data=arr)
            pg["value"].attrs["units"] = "deg"

        # stitched_1d (so read_stitched can be tested too) -------------
        st = entry.create_group("stitched_1d")
        st.attrs["NX_class"] = "NXdata"
        st.create_dataset("intensity", data=rng.random(N_Q, dtype=np.float32))
        st.create_dataset("q", data=q)

        # provenance ---------------------------------------------------
        write_provenance(
            f,
            program="xdart",
            program_version="0.37.0-dev0",
            config={
                "bai_1d_args": {"npt": N_Q, "unit": "q_A^-1"},
                "geometry": {
                    "convention": "psic",
                    "mapping_json": geom.to_json(),
                    "motor_sources": {"eta": "eta", "del": "del", "nu": "nu"},
                },
            },
            inputs={"raw_files": [f"frame_{i:04d}.h5" for i in range(N_FRAMES)],
                    "meta_file": "psic_scan.spec"},
            date="2026-05-11T00:00:00Z",
            host="testhost",
        )

    return p, geom, derived


# ---------------------------------------------------------------------------
# read_scan
# ---------------------------------------------------------------------------

class TestReadSphere:
    def test_basic_dims_and_shapes(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        assert ds.sizes["frame"] == N_FRAMES
        assert ds.sizes["q"] == N_Q
        assert ds.sizes["chi"] == N_CHI
        assert ds["intensity_1d"].dims == ("frame", "q")
        # 1D's radial is `q`; 2D has its own `q_2d` so 1D/2D can be
        # at different resolutions without an xarray dim conflict.
        assert ds["intensity_2d"].dims == ("frame", "chi", "q_2d")

    def test_sigma_1d_loaded(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        assert "sigma_1d" in ds.data_vars
        assert ds["sigma_1d"].shape == (N_FRAMES, N_Q)

    def test_groups_filter_skips_2d(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p, groups=("1d",))
        assert "intensity_1d" in ds.data_vars
        assert "intensity_2d" not in ds.data_vars
        # chi dim should not be present either
        assert "chi" not in ds.dims

    def test_per_frame_geometry_loaded(self, v2_fixture):
        p, _, derived = v2_fixture
        ds = read_scan(p)
        for key in ("rot1", "rot2", "rot3", "incident_angle"):
            assert key in ds.data_vars
            np.testing.assert_allclose(
                ds[key].values, derived[key].astype(np.float32), atol=1e-6
            )

    def test_motor_positioners_loaded(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        # Sample motors
        for m in ("eta", "phi", "mu"):
            assert m in ds.data_vars
            assert ds[m].shape == (N_FRAMES,)
        # 'chi' is both a coord (dim) and a sample motor name —
        # the reader stores the motor under a prefixed key to avoid
        # collision.
        assert "sample_chi" in ds.data_vars or "chi" in ds.coords
        # Detector motors
        assert "del" in ds.data_vars
        assert "nu" in ds.data_vars

    def test_q_units_attr(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        assert ds["q"].attrs.get("units") == "1/angstrom"
        assert ds["chi"].attrs.get("units") == "deg"

    def test_reduction_attrs_attached(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        red = ds.attrs.get("reduction", {})
        assert red["program"] == "xdart"
        assert red["version"] == "0.37.0-dev0"
        assert red["versions"]["python"]
        assert red["config"]["geometry"]["convention"] == "psic"

    def test_frame_coord_present(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        np.testing.assert_array_equal(
            ds["frame"].values, np.arange(N_FRAMES)
        )

    def test_missing_entry_raises(self, tmp_path):
        p = tmp_path / "empty.nxs"
        with h5py.File(p, "w") as f:
            f.create_group("not_entry")
        with pytest.raises(KeyError):
            read_scan(p)

    def test_thumbnails_off_by_default(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_scan(p)
        assert "thumbnail" not in ds.data_vars


# ---------------------------------------------------------------------------
# read_stitched
# ---------------------------------------------------------------------------

class TestReadStitched:
    def test_reads_stitched_1d(self, v2_fixture):
        p, _, _ = v2_fixture
        ds = read_stitched(p)
        assert "stitched_1d" in ds.data_vars
        assert ds["stitched_1d"].dims == ("q",)
        assert ds.sizes["q"] == N_Q

    def test_raises_when_no_stitched_present(self, tmp_path):
        p = tmp_path / "no_stitch.nxs"
        with h5py.File(p, "w") as f:
            f.create_group("entry")
        with pytest.raises(KeyError):
            read_stitched(p)


# ---------------------------------------------------------------------------
# Regression: 1D and 2D radial axes can have independent resolutions
# ---------------------------------------------------------------------------

class TestMixedQResolution:
    """xdart often integrates 1D at npt=2000 and 2D at npt_rad=500.

    The reader must surface those as separate dims (``q`` and
    ``q_2d``); putting both under ``q`` raises ValueError in xarray
    because the dim's length is ambiguous.
    """

    def test_separate_q_and_q_2d_when_sizes_differ(self, tmp_path):
        p = tmp_path / "mixed_q.nxs"
        n_frames = 3
        n_q_1d = 2000
        n_q_2d = 500
        n_chi = 16
        rng = np.random.default_rng(2)

        with h5py.File(p, "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"

            g1 = entry.create_group("integrated_1d")
            g1.attrs["NX_class"] = "NXdata"
            g1.create_dataset("intensity",
                              data=rng.random((n_frames, n_q_1d), dtype=np.float32))
            g1.create_dataset("q",
                              data=np.linspace(0.1, 5.0, n_q_1d, dtype=np.float32))
            g1.create_dataset("frame_index",
                              data=np.arange(n_frames, dtype=np.int32))

            g2 = entry.create_group("integrated_2d")
            g2.attrs["NX_class"] = "NXdata"
            g2.create_dataset("intensity",
                              data=rng.random((n_frames, n_chi, n_q_2d),
                                              dtype=np.float32))
            g2.create_dataset("q",
                              data=np.linspace(0.1, 5.0, n_q_2d, dtype=np.float32))
            g2.create_dataset("chi",
                              data=np.linspace(-180.0, 180.0, n_chi,
                                               endpoint=False, dtype=np.float32))

        ds = read_scan(p)
        # Both q axes present with their own sizes
        assert ds.sizes["q"] == n_q_1d
        assert ds.sizes["q_2d"] == n_q_2d
        # 2D's radial dim is q_2d, not q
        assert ds["intensity_2d"].dims == ("frame", "chi", "q_2d")
        # 1D's radial dim stays q
        assert ds["intensity_1d"].dims == ("frame", "q")


# ---------------------------------------------------------------------------
# 1D / 2D frame-label alignment in read_scan
# ---------------------------------------------------------------------------

def _write_1d2d(path, fidx_1d, fidx_2d, *, nq=6, nchi=4):
    """Minimal v2 file with independent 1D/2D frame_index vectors."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=rng.random((len(fidx_1d), nq)).astype("f4"))
        g1.create_dataset("q", data=np.linspace(1, 5, nq))
        g1.create_dataset("frame_index", data=np.asarray(fidx_1d, "i4"))
        g2 = e.create_group("integrated_2d")
        # distinct value per 2D row so mislabeling would be detectable
        i2 = np.stack([np.full((nchi, nq), float(k)) for k in range(len(fidx_2d))])
        g2.create_dataset("intensity", data=i2.astype("f4"))
        g2.create_dataset("q", data=np.linspace(1, 5, nq))
        g2.create_dataset("chi", data=np.linspace(-180, 180, nchi, endpoint=False))
        g2.create_dataset("frame_index", data=np.asarray(fidx_2d, "i4"))


def test_read_scan_shared_frame_when_labels_match(tmp_path):
    p = tmp_path / "match.nxs"
    _write_1d2d(p, [0, 1, 2], [0, 1, 2])
    ds = read_scan(p)
    assert ds["intensity_1d"].dims == ("frame", "q")
    assert ds["intensity_2d"].dims == ("frame", "chi", "q_2d")
    assert "frame_2d" not in ds.coords


def test_read_scan_separate_dim_when_labels_differ(tmp_path):
    """1D and 2D reduced over different frame labels must NOT be forced
    onto one 'frame' coord (silent mislabel) — 2D gets its own dim."""
    p = tmp_path / "mismatch.nxs"
    _write_1d2d(p, [0, 1, 2], [10, 11, 12])
    ds = read_scan(p)
    assert list(ds["frame"].values) == [0, 1, 2]
    assert ds["intensity_2d"].dims == ("frame_2d", "chi", "q_2d")
    assert list(ds["frame_2d"].values) == [10, 11, 12]
    # row k still holds value k (orientation/order preserved, not mislabeled)
    np.testing.assert_allclose(ds["intensity_2d"].values[1], 1.0)
