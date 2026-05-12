"""Tests for v1 (xdart ≤ 0.36.x) schema read support.

Hand-crafts a v1-conformant NXroot fixture with pure h5py (no xdart
dependency) and runs it through the public ``read_sphere`` dispatcher
to verify v1 → canonical xr.Dataset mapping.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.io.nexus import read_sphere


N_FRAMES = 4
N_Q = 32
N_CHI = 16


@pytest.fixture
def v1_fixture(tmp_path):
    """Minimal v1-conformant file: 4-frame 2-circle scan."""
    p = tmp_path / "v1_scan.nxs"

    rng = np.random.default_rng(1)
    radial = np.linspace(0.5, 5.0, N_Q).astype(np.float32)
    azim = np.linspace(-180.0, 180.0, N_CHI, endpoint=False).astype(np.float32)
    frame_indices = np.arange(N_FRAMES)

    # Per-frame integrated patterns
    i_1d = rng.random((N_FRAMES, N_Q), dtype=np.float32)
    s_1d = np.sqrt(i_1d).astype(np.float32)
    # v1 2D shape per-frame: (nq, nchi) — xdart convention
    i_2d = rng.random((N_FRAMES, N_Q, N_CHI), dtype=np.float32)

    # Motors (2-circle)
    th = np.linspace(0.0, 0.3, N_FRAMES, dtype=np.float32)
    tth = np.linspace(10.0, 12.0, N_FRAMES, dtype=np.float32)

    with h5py.File(p, "w") as f:
        # Stamp the v1 marker
        f.attrs["type"] = "EwaldSphere"

        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.attrs["type"] = "EwaldSphere"

        # scan_data/ (NXcollection) ----------------------------------
        sd = e.create_group("scan_data")
        sd.attrs["NX_class"] = "NXcollection"
        sd.create_dataset("th", data=th)
        sd.create_dataset("tth", data=tth)

        # frames/ ----------------------------------------------------
        frames = e.create_group("frames")
        for k in range(N_FRAMES):
            # 1D group: <NNNN>/ with radial, intensity, sigma
            g1 = frames.create_group(f"{k:04d}")
            g1.attrs["NX_class"] = "NXdata"
            g1.attrs["signal"] = "intensity"
            g1.attrs["axes"] = ["radial"]
            r = g1.create_dataset("radial", data=radial)
            r.attrs["units"] = "1/angstrom"
            g1.create_dataset("intensity", data=i_1d[k])
            g1.create_dataset("sigma", data=s_1d[k])

            # 2D group: <NNNN>_2d/ with radial, azimuthal, intensity
            g2 = frames.create_group(f"{k:04d}_2d")
            g2.attrs["NX_class"] = "NXdata"
            g2.attrs["signal"] = "intensity"
            g2.create_dataset("radial", data=radial)
            ds_az = g2.create_dataset("azimuthal", data=azim)
            ds_az.attrs["units"] = "deg"
            g2.create_dataset("intensity", data=i_2d[k])

            # Thumbnail dataset (uncompressed for the fixture; v1
            # uses gzip — the reader doesn't care either way)
            frames.create_dataset(
                f"{k:04d}_thumb",
                data=rng.random((8, 8), dtype=np.float32),
            )

        # integrated_1d (summed — 1D shape, the v1 schema marker) ----
        s1 = e.create_group("integrated_1d")
        s1.attrs["NX_class"] = "NXdata"
        s1.create_dataset("intensity", data=i_1d.sum(axis=0))
        s1.create_dataset("radial", data=radial)

    return p, radial, azim, i_1d, i_2d, th, tth


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

class TestSchemaDetection:
    def test_v1_file_dispatched_to_v1_reader(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p)
        assert ds.attrs.get("schema_version") == "v1"

    def test_explicit_v1_schema_works(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p, schema="v1")
        assert ds.attrs.get("schema_version") == "v1"

    def test_unknown_schema_raises(self, v1_fixture):
        p, *_ = v1_fixture
        with pytest.raises(ValueError, match="Unknown schema"):
            read_sphere(p, schema="v9")


# ---------------------------------------------------------------------------
# v1 reader payload
# ---------------------------------------------------------------------------

class TestReadSphereV1:
    def test_dims_match_fixture(self, v1_fixture):
        p, radial, azim, *_ = v1_fixture
        ds = read_sphere(p)
        assert ds.sizes["frame"] == N_FRAMES
        assert ds.sizes["q"] == N_Q
        assert ds.sizes["chi"] == N_CHI

    def test_intensity_1d_roundtrips(self, v1_fixture):
        p, _, _, i_1d, *_ = v1_fixture
        ds = read_sphere(p)
        np.testing.assert_allclose(ds["intensity_1d"].values, i_1d, atol=1e-6)

    def test_sigma_1d_loaded(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p)
        assert "sigma_1d" in ds.data_vars
        assert ds["sigma_1d"].shape == (N_FRAMES, N_Q)

    def test_intensity_2d_transposed_to_canonical_orientation(self, v1_fixture):
        # v1 stores 2D as (nq, nchi); canonical Dataset is (frame, chi, q)
        p, _, _, _, i_2d, *_ = v1_fixture
        ds = read_sphere(p)
        # Per-frame: stored (nq, nchi) → expected (nchi, nq)
        expected = np.transpose(i_2d, (0, 2, 1))
        np.testing.assert_allclose(ds["intensity_2d"].values, expected, atol=1e-6)

    def test_q_coord_attached(self, v1_fixture):
        p, radial, *_ = v1_fixture
        ds = read_sphere(p)
        np.testing.assert_allclose(ds["q"].values, radial)
        assert ds["q"].attrs.get("units") == "1/angstrom"

    def test_chi_coord_attached(self, v1_fixture):
        p, _, azim, *_ = v1_fixture
        ds = read_sphere(p)
        np.testing.assert_allclose(ds["chi"].values, azim)
        assert ds["chi"].attrs.get("units") == "deg"

    def test_motor_columns_loaded_from_scan_data(self, v1_fixture):
        p, _, _, _, _, th, tth = v1_fixture
        ds = read_sphere(p)
        assert "th" in ds.data_vars
        assert "tth" in ds.data_vars
        np.testing.assert_allclose(ds["th"].values, th)
        np.testing.assert_allclose(ds["tth"].values, tth)

    def test_frame_coord_uses_fixture_indices(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p)
        np.testing.assert_array_equal(
            ds["frame"].values, np.arange(N_FRAMES)
        )

    def test_thumbnails_off_by_default(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p)
        assert "thumbnail" not in ds.data_vars

    def test_thumbnails_loaded_when_requested(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p, include_thumbnails=True)
        assert "thumbnail" in ds.data_vars
        assert ds["thumbnail"].shape == (N_FRAMES, 8, 8)

    def test_groups_filter_skips_2d(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p, groups=("1d",))
        assert "intensity_1d" in ds.data_vars
        assert "intensity_2d" not in ds.data_vars

    def test_v1_has_empty_reduction_attr(self, v1_fixture):
        p, *_ = v1_fixture
        ds = read_sphere(p)
        # v1 files have no NXprocess block — reduction attr is {}
        assert ds.attrs.get("reduction") == {}

    def test_no_per_frame_geometry_vars(self, v1_fixture):
        # v1 has no derived rot1/rot2/rot3/incident_angle
        p, *_ = v1_fixture
        ds = read_sphere(p)
        for key in ("rot1", "rot2", "rot3", "incident_angle"):
            assert key not in ds.data_vars

    def test_v1_no_stitched_raises(self, v1_fixture):
        from ssrl_xrd_tools.io.nexus import read_stitched
        p, *_ = v1_fixture
        with pytest.raises(KeyError):
            read_stitched(p)
