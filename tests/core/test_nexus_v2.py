"""Tests for the v2 NeXus reader (``read_scan`` / ``read_stitched``).

We create a small current record through the production writer — no xdart
dependency, no captured-file dependency — and round-trip it through the v2
reader.  The fixture corresponds to a 5-frame psic scan.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.geometry import DiffractometerGeometry
from xrd_tools.core.provenance import write_provenance
from xrd_tools.io.nexus import read_scan, read_stitched, write_nexus


N_FRAMES = 5
N_Q = 64
N_CHI = 32


def _write_current_results(
    path,
    fidx_1d,
    fidx_2d=None,
    *,
    q_1d=None,
    q_2d=None,
    chi=None,
    intensity_1d=None,
    sigma_1d=None,
    intensity_2d=None,
):
    """Write deterministic current results through the public writer."""
    fidx_1d = [int(value) for value in fidx_1d]
    q_1d = np.asarray(
        np.linspace(0.5, 1.0, 2) if q_1d is None else q_1d,
        dtype=float,
    )
    if intensity_1d is None:
        intensity_1d = np.zeros((len(fidx_1d), len(q_1d)), dtype=np.float32)
    intensity_1d = np.asarray(intensity_1d)
    if sigma_1d is not None:
        sigma_1d = np.asarray(sigma_1d)
    results_1d = {
        label: IntegrationResult1D(
            radial=q_1d,
            intensity=intensity_1d[row],
            sigma=None if sigma_1d is None else sigma_1d[row],
            unit="q_A^-1",
        )
        for row, label in enumerate(fidx_1d)
    }

    results_2d = None
    if fidx_2d is not None:
        fidx_2d = [int(value) for value in fidx_2d]
        q_2d = np.asarray(q_1d if q_2d is None else q_2d, dtype=float)
        chi = np.asarray(chi, dtype=float)
        intensity_2d = np.asarray(intensity_2d)
        results_2d = {
            label: IntegrationResult2D(
                radial=q_2d,
                azimuthal=chi,
                intensity=intensity_2d[row].T,
                unit="q_A^-1",
                azimuthal_unit="chi_deg",
            )
            for row, label in enumerate(fidx_2d)
        }

    write_nexus(
        path,
        results_1d=results_1d,
        results_2d=results_2d,
        compression=None,
        overwrite=True,
    )


@pytest.fixture
def v2_fixture(tmp_path):
    """A minimal v2-conformant NXroot file for a 5-frame psic scan."""
    p = tmp_path / "psic_5frame.nexus"

    rng = np.random.default_rng(0)
    q = np.linspace(0.5, 5.0, N_Q).astype(np.float32)
    chi = np.linspace(-180.0, 180.0, N_CHI, endpoint=False).astype(np.float32)
    frame_index = np.arange(N_FRAMES, dtype=np.int64)

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

    _write_current_results(
        p,
        frame_index,
        frame_index,
        q_1d=q,
        q_2d=q,
        chi=chi,
        intensity_1d=intensity_1d,
        sigma_1d=sigma_1d,
        intensity_2d=intensity_2d,
    )
    with h5py.File(p, "r+") as f:
        entry = f["entry"]
        entry.attrs["default"] = "integrated_1d"

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
        assert ds["q"].attrs.get("units") == "q_A^-1"
        assert ds["chi"].attrs.get("units") == "chi_deg"

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
        p = tmp_path / "empty.nexus"
        with h5py.File(p, "w") as f:
            f.create_group("not_entry")
        with pytest.raises(ValueError, match="not a current xdart"):
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
        p = tmp_path / "no_stitch.nexus"
        _write_current_results(p, [0])
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
        p = tmp_path / "mixed_q.nexus"
        n_frames = 3
        n_q_1d = 2000
        n_q_2d = 500
        n_chi = 16
        rng = np.random.default_rng(2)

        _write_current_results(
            p,
            np.arange(n_frames),
            np.arange(n_frames),
            q_1d=np.linspace(0.1, 5.0, n_q_1d, dtype=np.float32),
            q_2d=np.linspace(0.1, 5.0, n_q_2d, dtype=np.float32),
            chi=np.linspace(
                -180.0, 180.0, n_chi, endpoint=False, dtype=np.float32,
            ),
            intensity_1d=rng.random(
                (n_frames, n_q_1d), dtype=np.float32,
            ),
            intensity_2d=rng.random(
                (n_frames, n_chi, n_q_2d), dtype=np.float32,
            ),
        )

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
    # Distinct value per 2D row so mislabeling would be detectable.
    i2 = np.stack([np.full((nchi, nq), float(k)) for k in range(len(fidx_2d))])
    _write_current_results(
        path,
        fidx_1d,
        fidx_2d,
        q_1d=np.linspace(1, 5, nq, dtype=np.float32),
        q_2d=np.linspace(1, 5, nq, dtype=np.float32),
        chi=np.linspace(-180, 180, nchi, endpoint=False, dtype=np.float32),
        intensity_1d=rng.random((len(fidx_1d), nq)).astype("f4"),
        intensity_2d=i2.astype("f4"),
    )


def test_read_scan_shared_frame_when_labels_match(tmp_path):
    p = tmp_path / "match.nexus"
    _write_1d2d(p, [0, 1, 2], [0, 1, 2])
    ds = read_scan(p)
    assert ds["intensity_1d"].dims == ("frame", "q")
    assert ds["intensity_2d"].dims == ("frame", "chi", "q_2d")
    assert "frame_2d" not in ds.coords


def test_read_scan_separate_dim_when_labels_differ(tmp_path):
    """1D and 2D reduced over different frame labels must NOT be forced
    onto one 'frame' coord (silent mislabel) — 2D gets its own dim."""
    p = tmp_path / "mismatch.nexus"
    _write_1d2d(p, [0, 1, 2], [10, 11, 12])
    ds = read_scan(p)
    assert list(ds["frame"].values) == [0, 1, 2]
    assert ds["intensity_2d"].dims == ("frame_2d", "chi", "q_2d")
    assert list(ds["frame_2d"].values) == [10, 11, 12]
    # row k still holds value k (orientation/order preserved, not mislabeled)
    np.testing.assert_allclose(ds["intensity_2d"].values[1], 1.0)


def test_read_scan_metadata_surfaces_frame_2d_on_mismatch(tmp_path):
    """The lightweight metadata path mirrors read_scan: divergent 2D labels
    appear on a frame_2d coord (so get_metadata doesn't disagree)."""
    from xrd_tools.io.nexus import read_scan_metadata
    p = tmp_path / "meta_mismatch.nexus"
    _write_1d2d(p, [0, 1, 2], [10, 11, 12])
    ds = read_scan_metadata(p)
    assert list(ds["frame"].values) == [0, 1, 2]
    assert list(ds["frame_2d"].values) == [10, 11, 12]
    p2 = tmp_path / "meta_match.nexus"
    _write_1d2d(p2, [0, 1, 2], [0, 1, 2])
    assert "frame_2d" not in read_scan_metadata(p2).coords


def test_write_positioners_and_geometry_roundtrip(tmp_path):
    """ssrl write_positioners + write_per_frame_geometry round-trip through
    read_scan_metadata, aligned to gapped frame ids."""
    import pandas as pd
    from xrd_tools.io.nexus import (
        write_positioners, write_per_frame_geometry, read_scan_metadata,
    )
    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    fis = [1, 2, 4]  # gapped + 1-based to exercise reindex/alignment
    sd = pd.DataFrame({"tth": [10.0, 11.0, 12.0], "th": [0.1, 0.2, 0.3]}, index=fis)

    p = tmp_path / "geom.nexus"
    _write_current_results(
        p,
        fis,
        q_1d=np.linspace(1, 5, 5, dtype=np.float32),
    )
    with h5py.File(p, "r+") as f:
        e = f["entry"]
        write_positioners(e, sd, fis, geom)
        write_per_frame_geometry(e, sd, fis, geom)

    ds = read_scan_metadata(p)
    assert list(ds["frame"].values) == fis
    assert "th" in ds.data_vars
    np.testing.assert_allclose(ds["th"].values, [0.1, 0.2, 0.3], rtol=1e-5)
    assert "rot1" in ds.data_vars and ds["rot1"].shape == (3,)


def test_write_positioners_reindexes_out_of_order(tmp_path):
    """scan_data rows in a different order than frame_indices must be aligned,
    not attached positionally (mirrors the stitch fix)."""
    import pandas as pd
    from xrd_tools.io.nexus import write_positioners, read_scan_metadata
    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    fis = [0, 1, 2]
    # scan_data rows are in REVERSE order vs fis
    sd = pd.DataFrame({"tth": [12.0, 11.0, 10.0], "th": [0.3, 0.2, 0.1]}, index=[2, 1, 0])
    p = tmp_path / "ooo.nexus"
    _write_current_results(
        p,
        fis,
        q_1d=np.linspace(1, 5, 5, dtype=np.float32),
    )
    with h5py.File(p, "r+") as f:
        e = f["entry"]
        write_positioners(e, sd, fis, geom)
    ds = read_scan_metadata(p)
    # th for frame 0 must be 0.1 (its scan_data row), not 0.3 (positional row 0)
    np.testing.assert_allclose(ds["th"].values, [0.1, 0.2, 0.3], rtol=1e-5)


def test_replacement_metadata_writers_clear_empty_authoritative_state(tmp_path):
    import pandas as pd
    from xrd_tools.io.nexus import (
        write_per_frame_geometry,
        write_positioners,
        write_scan_metadata,
    )

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    sd = pd.DataFrame({"tth": [10.0], "th": [0.1], "i0": [1.0]}, index=[0])
    p = tmp_path / "metadata_clear.nexus"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_positioners(e, sd, [0], geom)
        write_per_frame_geometry(e, sd, [0], geom)
        write_scan_metadata(e, sd, [0])
        write_positioners(e, pd.DataFrame(), [], geom)
        write_per_frame_geometry(e, pd.DataFrame(), [], geom)
        write_scan_metadata(e, pd.DataFrame(), [])
        assert "sample/positioners" not in e
        assert "instrument/detector/positioners" not in e
        assert "per_frame_geometry" not in e
        assert "scan_data" not in e


def test_write_scan_metadata_rejects_duplicate_labels(tmp_path):
    import pandas as pd
    import pytest
    from xrd_tools.io.nexus import write_scan_metadata

    p = tmp_path / "duplicate_metadata_write.nexus"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        with pytest.raises(ValueError, match="duplicate labels"):
            write_scan_metadata(
                e,
                pd.DataFrame({"th": [0.1, 0.2]}, index=[0, 0]),
                [0, 0],
            )
        assert "scan_data" not in e


def test_dup_label_write_preserves_existing_metadata(tmp_path):
    """A malformed (duplicate-label) write must validate BEFORE deleting the
    authoritative group, so an existing valid scan_data / per_frame_geometry
    survives the failed call instead of being lost (delete-then-raise)."""
    import pandas as pd
    import pytest
    from xrd_tools.io.nexus import (
        write_per_frame_geometry,
        write_scan_metadata,
    )

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    good = pd.DataFrame({"tth": [10.0, 11.0], "th": [0.1, 0.2], "i0": [1.0, 2.0]},
                        index=[0, 1])
    dup = pd.DataFrame({"tth": [9.0, 9.0], "th": [0.0, 0.0], "i0": [9.0, 9.0]},
                       index=[0, 0])

    p = tmp_path / "preserve.nexus"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_scan_metadata(e, good, [0, 1])
        write_per_frame_geometry(e, good, [0, 1], geom)
        sd_before = e["scan_data/i0"][()].tolist()
        geo_before = e["per_frame_geometry/frame_index"][()].tolist()

        with pytest.raises(ValueError, match="duplicate labels"):
            write_scan_metadata(e, dup, [0, 0])
        with pytest.raises(ValueError, match="duplicate labels"):
            write_per_frame_geometry(e, dup, [0, 0], geom)

        assert "scan_data" in e and "per_frame_geometry" in e
        assert e["scan_data/i0"][()].tolist() == sd_before
        assert e["per_frame_geometry/frame_index"][()].tolist() == geo_before


def test_scan_data_string_columns_survive_roundtrip(tmp_path):
    """A4: non-numeric (string) scan_data columns must survive write->read
    intact, aligned to frame labels, with numeric columns unchanged (the
    `keith_I=0V` drop).  Exercises BOTH writers (full write_scan_metadata +
    incremental upsert) and BOTH readers (xarray read_scan_metadata + dict
    _scan_data_for_frames)."""
    import pandas as pd
    from xrd_tools.io.nexus import (
        write_scan_metadata, upsert_scan_metadata, read_scan_metadata,
    )
    from xrd_tools.io.read import _scan_data_for_frames

    fis = [0, 1, 2]
    sd = pd.DataFrame(
        {"i0": [1.0, 2.0, 3.0], "keith_I": ["0V", "1V", "2V"],
         "temp": [300.0, 301.0, 302.0]}, index=fis)

    # full writer + both readers
    p = tmp_path / "full.nexus"
    _write_current_results(p, fis)
    with h5py.File(p, "r+") as f:
        e = f["entry"]
        write_scan_metadata(e, sd, fis)
    ds = read_scan_metadata(p)
    assert list(np.asarray(ds["keith_I"].values)) == ["0V", "1V", "2V"]
    np.testing.assert_allclose(ds["i0"].values, [1.0, 2.0, 3.0], rtol=1e-5)
    dct = _scan_data_for_frames(p, fis)
    assert list(dct["keith_I"]) == ["0V", "1V", "2V"]

    # incremental upsert (create, then append a row) keeps the string column
    p2 = tmp_path / "upsert.nexus"
    _write_current_results(p2, fis)
    with h5py.File(p2, "r+") as f:
        e = f["entry"]
        upsert_scan_metadata(e, sd.iloc[:2], [0, 1])
        upsert_scan_metadata(
            e, pd.DataFrame({"i0": [3.0], "keith_I": ["2V"], "temp": [302.0]},
                            index=[2]), [2])
    ds2 = read_scan_metadata(p2)
    assert list(np.asarray(ds2["keith_I"].values)) == ["0V", "1V", "2V"]
    np.testing.assert_allclose(ds2["i0"].values, [1.0, 2.0, 3.0], rtol=1e-5)
