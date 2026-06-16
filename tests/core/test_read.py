"""Tests for the notebook-friendly convenience readers (``io.read``).

Hand-crafts a small v2 NXroot with pure h5py (no xdart dependency) and
round-trips it through ``get_1d`` / ``get_2d`` / ``get_thumbnail`` /
``get_metadata`` / ``get_frames`` and the ``Scan`` sugar.

The fixture deliberately uses **gapped, 1-based frame labels**
(``[1, 2, 4, 7, 8]``) so the label→position resolution is actually
exercised rather than coinciding with row positions.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xrd_tools.io import (
    Integrated1D,
    Integrated2D,
    Scan,
    get_1d,
    get_2d,
    get_frames,
    get_metadata,
    get_thumbnail,
    open_scan,
)

N_FRAMES = 5
N_Q = 64
N_CHI = 32
N_THUMB = 16
FRAME_LABELS = np.array([1, 2, 4, 7, 8], dtype=np.int32)  # gapped, 1-based
ENERGY_KEV = 12.0


@pytest.fixture
def scan_file(tmp_path):
    p = tmp_path / "gapped_5frame.nxs"
    rng = np.random.default_rng(1)

    q = np.linspace(0.5, 5.0, N_Q).astype(np.float32)
    q2 = np.linspace(0.5, 4.0, N_Q).astype(np.float32)
    chi = np.linspace(-180.0, 180.0, N_CHI, endpoint=False).astype(np.float32)
    intensity_1d = rng.random((N_FRAMES, N_Q), dtype=np.float32)
    sigma_1d = np.sqrt(intensity_1d).astype(np.float32)
    intensity_2d = rng.random((N_FRAMES, N_CHI, N_Q), dtype=np.float32)
    thumbs = rng.random((N_FRAMES, N_THUMB, N_THUMB), dtype=np.float32)
    eta = np.linspace(0.0, 1.0, N_FRAMES, dtype=np.float32)

    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"

        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=intensity_1d)
        g1.create_dataset("sigma", data=sigma_1d)
        qd = g1.create_dataset("q", data=q)
        qd.attrs["units"] = "1/angstrom"
        g1.create_dataset("frame_index", data=FRAME_LABELS)

        g2 = e.create_group("integrated_2d")
        g2.create_dataset("intensity", data=intensity_2d)
        q2d = g2.create_dataset("q", data=q2)
        q2d.attrs["units"] = "1/angstrom"
        cd = g2.create_dataset("chi", data=chi)
        cd.attrs["units"] = "deg"
        g2.create_dataset("frame_index", data=FRAME_LABELS)

        sp = e.create_group("sample/positioners")
        pg = sp.create_group("eta")
        pg.attrs["NX_class"] = "NXpositioner"
        pg.create_dataset("value", data=eta)

        # monochromator + sample scalars for get_metadata
        e.create_dataset("instrument/monochromator/energy", data=ENERGY_KEV)
        e.create_dataset("sample/name", data=np.bytes_(b"LaB6"))

        # per-frame thumbnails keyed by label (4-digit zero pad)
        for i, lbl in enumerate(FRAME_LABELS):
            fg = e.create_group(f"frames/frame_{int(lbl):04d}")
            fg.create_dataset("thumbnail", data=thumbs[i])

    return p, dict(
        q=q, q2=q2, chi=chi, intensity_1d=intensity_1d, sigma_1d=sigma_1d,
        intensity_2d=intensity_2d, thumbs=thumbs, eta=eta,
    )


def test_get_frames(scan_file):
    p, _ = scan_file
    np.testing.assert_array_equal(get_frames(p), FRAME_LABELS)


def test_get_1d_single_frame_by_label(scan_file):
    p, ref = scan_file
    # label 4 is at row position 2
    r = get_1d(p, frame=4)
    assert isinstance(r, Integrated1D)
    assert r.frames == 4
    assert r.q.shape == (N_Q,)
    assert r.intensity.shape == (N_Q,)
    np.testing.assert_array_equal(r.intensity, ref["intensity_1d"][2])
    np.testing.assert_array_equal(r.sigma, ref["sigma_1d"][2])
    assert r.q_unit == "1/angstrom"


def test_get_1d_all_frames(scan_file):
    p, ref = scan_file
    r = get_1d(p)
    assert r.intensity.shape == (N_FRAMES, N_Q)
    np.testing.assert_array_equal(r.intensity, ref["intensity_1d"])
    np.testing.assert_array_equal(r.frames, FRAME_LABELS)


def test_get_1d_subset_preserves_requested_order(scan_file):
    p, ref = scan_file
    # request out of order + gapped: labels 7,2 -> rows 3,1
    r = get_1d(p, frame=[7, 2])
    np.testing.assert_array_equal(r.frames, [7, 2])
    np.testing.assert_array_equal(r.intensity[0], ref["intensity_1d"][3])
    np.testing.assert_array_equal(r.intensity[1], ref["intensity_1d"][1])


def test_get_2d_single_frame(scan_file):
    p, ref = scan_file
    r = get_2d(p, frame=8)  # last row
    assert isinstance(r, Integrated2D)
    assert r.intensity.shape == (N_CHI, N_Q)
    np.testing.assert_array_equal(r.intensity, ref["intensity_2d"][4])
    assert r.chi.shape == (N_CHI,)
    assert r.chi_unit == "deg"


def test_get_2d_all_frames(scan_file):
    p, _ = scan_file
    r = get_2d(p)
    assert r.intensity.shape == (N_FRAMES, N_CHI, N_Q)


def test_unknown_frame_raises(scan_file):
    p, _ = scan_file
    with pytest.raises(KeyError):
        get_1d(p, frame=3)  # 3 is in the gap


def test_get_thumbnail(scan_file):
    p, ref = scan_file
    img = get_thumbnail(p, 4)
    assert img.shape == (N_THUMB, N_THUMB)
    np.testing.assert_array_equal(img, ref["thumbs"][2])


def test_get_metadata(scan_file):
    p, ref = scan_file
    m = get_metadata(p)
    assert m["n_frames"] == N_FRAMES
    np.testing.assert_array_equal(m["frames"], FRAME_LABELS)
    assert m["has_1d"] and m["has_2d"]
    assert m["sample_name"] == "LaB6"
    assert m["energy_keV"] == pytest.approx(ENERGY_KEV)
    assert m["wavelength_A"] > 0  # derived from energy
    assert "eta" in m["positioners"]
    np.testing.assert_array_equal(m["positioners"]["eta"], ref["eta"])
    assert "q" in m and "chi" in m


def test_get_metadata_positioners_vs_scan_data_split(tmp_path):
    """``positioners`` is geometry motors only (from the NXpositioner
    groups); the full per-frame table (motors + counters like i0) lives in
    ``scan_data`` — so normalization/geometry consumers of positioners
    aren't polluted with counters."""
    import pandas as pd
    from xrd_tools.io import write_scan_metadata, get_metadata

    p = tmp_path / "split.nxs"
    sd = pd.DataFrame(
        {"th": [0.1, 0.2, 0.3], "i0": [1e6, 1.1e6, 1.2e6],
         "mon": [33.0, 34.0, 35.0]},
        index=[0, 1, 2],
    )
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((3, 4)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([0, 1, 2], dtype=np.int64))
        write_scan_metadata(e, sd, [0, 1, 2])          # full table → /entry/scan_data
        pg = e.create_group("sample/positioners/th")   # th ALSO a geometry motor
        pg.attrs["NX_class"] = "NXpositioner"
        pg.create_dataset("value", data=np.array([0.1, 0.2, 0.3], dtype=np.float32))

    m = get_metadata(p)
    # positioners: only the geometry motor th, NOT the counters
    assert set(m["positioners"]) == {"th"}
    assert "i0" not in m["positioners"] and "mon" not in m["positioners"]
    # scan_data: the full table including counters
    assert {"th", "i0", "mon"}.issubset(set(m["scan_data"]))
    np.testing.assert_allclose(m["scan_data"]["i0"], [1e6, 1.1e6, 1.2e6])


def test_mismatched_positioner_length_is_skipped_not_fatal(tmp_path):
    """A per-frame column whose length != frame count (malformed/partial
    file, e.g. 4 integrated frames but 2 'th' positions) must be skipped
    with a warning, not crash the whole reader."""
    from xrd_tools.io.nexus import read_scan, read_scan_metadata

    p = tmp_path / "mismatch.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=np.zeros((4, 8), dtype="f4"))
        g1.create_dataset("q", data=np.linspace(1, 5, 8))
        g1.create_dataset("frame_index", data=np.arange(4, dtype="i4"))
        # 'th' positioner with only 2 values for a 4-frame scan.
        sp = e.create_group("sample/positioners")
        pg = sp.create_group("th")
        pg.attrs["NX_class"] = "NXpositioner"
        pg.create_dataset("value", data=np.array([0.1, 0.2], dtype="f4"))
        # a well-formed positioner of the right length survives.
        pg2 = sp.create_group("samz")
        pg2.attrs["NX_class"] = "NXpositioner"
        pg2.create_dataset("value", data=np.arange(4, dtype="f4"))

    for reader in (read_scan_metadata, lambda x: read_scan(x)):
        ds = reader(p)
        assert ds.sizes["frame"] == 4
        assert "th" not in ds.data_vars          # mismatched → skipped
        assert "samz" in ds.data_vars            # matching → kept


def test_legacy_read_sphere_names_are_gone():
    """The transitional read_sphere* aliases were removed in the rename
    release — only read_scan / read_scan_metadata exist now."""
    import xrd_tools.io.nexus as nexus_io
    import xrd_tools.io as io_pkg

    for legacy in ("read_sphere", "read_sphere_metadata"):
        assert not hasattr(nexus_io, legacy), f"{legacy} should be removed"
        assert not hasattr(io_pkg, legacy), f"{legacy} should not be re-exported"
    assert hasattr(nexus_io, "read_scan")
    assert hasattr(nexus_io, "read_scan_metadata")


def test_get_2d_resolves_against_its_own_frame_labels(tmp_path):
    """When 1D and 2D were reduced over different frame labels, get_2d must
    index integrated_2d's own frame_index (not integrated_1d's)."""
    p = tmp_path / "diff_labels.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=np.zeros((3, 5), dtype="f4"))
        g1.create_dataset("q", data=np.linspace(1, 5, 5))
        g1.create_dataset("frame_index", data=np.array([0, 1, 2], dtype="i4"))
        g2 = e.create_group("integrated_2d")
        # distinct value per 2D row so we can tell which one we got
        i2 = np.stack([np.full((4, 5), float(k)) for k in range(3)]).astype("f4")
        g2.create_dataset("intensity", data=i2)
        g2.create_dataset("q", data=np.linspace(1, 5, 5))
        g2.create_dataset("chi", data=np.linspace(-180, 180, 4, endpoint=False))
        g2.create_dataset("frame_index", data=np.array([10, 11, 12], dtype="i4"))

    # 2D label 11 → row 1 (value 1.0 everywhere); must NOT raise or pick row 0.
    r = get_2d(p, frame=11)
    assert r.frames == 11
    assert np.allclose(r.intensity, 1.0)
    # a label that only exists in 1D is not a valid 2D frame
    with pytest.raises(KeyError):
        get_2d(p, frame=0)
    # 1D still indexes its own labels
    assert get_1d(p, frame=2).intensity.shape == (5,)


def test_scan_sugar(scan_file):
    p, ref = scan_file
    scan = open_scan(p)
    assert isinstance(scan, Scan)
    assert len(scan) == N_FRAMES
    np.testing.assert_array_equal(scan.frames, FRAME_LABELS)
    np.testing.assert_array_equal(scan.get_1d(1).intensity, ref["intensity_1d"][0])
    assert scan.get_2d(2).intensity.shape == (N_CHI, N_Q)
    assert "LaB6" in repr(scan) or "n_frames=5" in repr(scan)


# ---------------------------------------------------------------------------
# get_raw_frame: resolve raw via source pointer, fall back to thumbnail
# ---------------------------------------------------------------------------

def test_get_raw_frame_resolves_source_pointer(tmp_path):
    """A processed file points each frame back to the detector master via
    ``frames/frame_NNNN/source``; get_raw_frame loads the full-res raw."""
    from xrd_tools.io import get_raw_frame

    # synthetic raw master (.h5) in the same dir, two frames
    master = tmp_path / "scan_master.h5"
    raw = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"scan_master.h5"))
        s.create_dataset("frame_index", data=1)  # → master frame 1

    img = get_raw_frame(nxs, frame=0)
    assert img.shape == (8, 8)
    np.testing.assert_allclose(img, raw[1])


def test_get_raw_frame_resolves_absolute_source_pointer(tmp_path):
    from xrd_tools.io import get_raw_frame

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    master = raw_dir / "scan_master.h5"
    raw = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    nxs = processed_dir / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([3], dtype=np.int64))
        s = e.create_group("frames/frame_0003/source")
        s.create_dataset("path", data=np.bytes_(str(master).encode()))
        s.create_dataset("frame_index", data=1)

    np.testing.assert_allclose(get_raw_frame(nxs, frame=3), raw[1])


def test_get_raw_frame_resolves_sibling_basename_when_old_path_is_stale(tmp_path):
    from xrd_tools.io import get_raw_frame

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    master = processed_dir / "scan_master.h5"
    raw = np.arange(4 * 4, dtype=float).reshape(1, 4, 4)
    with h5py.File(master, "w") as f:
        f.create_dataset("entry/data/data", data=raw)

    nxs = processed_dir / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([7], dtype=np.int64))
        s = e.create_group("frames/frame_0007/source")
        s.create_dataset("path", data=np.bytes_(b"old/raw/scan_master.h5"))
        s.create_dataset("frame_index", data=0)

    np.testing.assert_allclose(get_raw_frame(nxs, frame=7), raw[0])


def test_get_raw_frame_falls_back_to_thumbnail(tmp_path):
    """When the source master is missing, get_raw_frame returns the stored
    thumbnail, dequantized to its original intensity range."""
    from xrd_tools.io import get_raw_frame

    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((1, 5)))
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"does_not_exist.h5"))
        s.create_dataset("frame_index", data=0)
        th = e.create_dataset(
            "frames/frame_0000/thumbnail",
            data=(np.ones((4, 4)) * 128).astype(np.uint8),
        )
        th.attrs["vmin"] = 10.0; th.attrs["vmax"] = 20.0; th.attrs["dtype"] = "uint8"

    img = get_raw_frame(nxs, frame=0)
    assert img.shape == (4, 4)
    # 128/255 * (20-10) + 10
    np.testing.assert_allclose(img, 10.0 + (128 / 255) * 10.0, atol=1e-6)


def test_open_scan_strict_raw_does_not_use_thumbnail(tmp_path):
    import pytest
    from xrd_tools.io import get_raw_frame, open_scan

    nxs = tmp_path / "scan.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=np.zeros((1, 5)))
        g1.create_dataset("q", data=np.linspace(0.5, 2.0, 5))
        g1.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
        s = e.create_group("frames/frame_0000/source")
        s.create_dataset("path", data=np.bytes_(b"missing_master.h5"))
        s.create_dataset("frame_index", data=0)
        th = e.create_dataset("frames/frame_0000/thumbnail", data=np.ones((4, 4), dtype=np.uint8))
        th.attrs["vmin"] = 0.0
        th.attrs["vmax"] = 1.0
        th.attrs["dtype"] = "uint8"

    assert get_raw_frame(nxs, frame=0).shape == (4, 4)
    with pytest.raises(KeyError, match="thumbnail fallback disabled"):
        open_scan(nxs).load_frame(0)


def test_open_scan_frame_source_uses_union_labels_and_scan_data(tmp_path):
    from xrd_tools.io import open_scan

    nxs = tmp_path / "mixed_outputs.nxs"
    with h5py.File(nxs, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("intensity", data=np.zeros((2, 5)))
        g1.create_dataset("q", data=np.linspace(0.5, 2.0, 5))
        g1.create_dataset("frame_index", data=np.array([0, 2], dtype=np.int64))
        g2 = e.create_group("integrated_2d")
        g2.create_dataset("intensity", data=np.zeros((2, 3, 5)))
        g2.create_dataset("q", data=np.linspace(0.5, 2.0, 5))
        g2.create_dataset("chi", data=np.linspace(-1.0, 1.0, 3))
        g2.create_dataset("frame_index", data=np.array([1, 2], dtype=np.int64))
        sd = e.create_group("scan_data")
        sd.create_dataset("frame_index", data=np.array([0, 1, 2, 99], dtype=np.int64))
        sd.create_dataset("th", data=np.array([0.1, 0.2, 0.3, 9.9]))

    scan = open_scan(nxs)
    assert scan.frame_indices == [0, 1, 2]
    np.testing.assert_allclose(scan.scan_data["th"], [0.1, 0.2, 0.3])


def test_read_image_rejects_processed_file(tmp_path):
    """read_image must not mis-read a processed scan's integrated_1d as a
    raw detector image — it raises pointing at get_raw_frame instead."""
    from xrd_tools.io.image import read_image

    nxs = tmp_path / "processed.nxs"
    with h5py.File(nxs, "w") as f:
        g = f.create_group("entry/integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((1, 2000)))  # (1, n_q)
        g.create_dataset("frame_index", data=np.array([0], dtype=np.int64))

    with pytest.raises(ValueError, match="processed xdart"):
        read_image(nxs, frame=0)

    from xrd_tools.io.image import read_image_stack
    with pytest.raises(ValueError, match="processed xdart"):
        read_image_stack(nxs)
    with pytest.raises(ValueError, match="processed xdart"):
        read_image_stack(nxs, reduce="mean")


def test_read_image_stack_reduces_hdf5_without_stack_materialization(
    monkeypatch, tmp_path
):
    from xrd_tools.io import image as image_mod
    from xrd_tools.io.image import read_image_stack

    h5_path = tmp_path / "raw_stack.h5"
    stack = np.array(
        [
            [[1.0, np.nan], [3.0, 4.0]],
            [[2.0, np.nan], [5.0, 6.0]],
            [[np.inf, np.nan], [7.0, np.nan]],
        ],
        dtype=np.float32,
    )
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("entry/data/data", data=stack)

    def fail_stack(*args, **kwargs):
        raise AssertionError("read_image_stack(reduce=...) must fold by frame")

    monkeypatch.setattr(image_mod, "_read_hdf5_stack", fail_stack)

    np.testing.assert_allclose(
        read_image_stack(h5_path, reduce="sum"),
        np.nansum(stack, axis=0),
    )
    expected_mean = np.array([[np.inf, np.nan], [5.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(
        read_image_stack(h5_path, reduce="mean"),
        expected_mean,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# write_scan_metadata: full per-frame metadata survives a read-back
# ---------------------------------------------------------------------------

def test_scan_metadata_full_table_round_trips(tmp_path):
    """write_scan_metadata persists every column (not just geometry
    motors), and read_scan surfaces them as per-frame vars — so a reload
    restores the whole metadata table, not just the incidence motor."""
    import pandas as pd
    from xrd_tools.io import write_scan_metadata, read_scan, read_scan_metadata

    p = tmp_path / "meta.nxs"
    sd = pd.DataFrame(
        {"i0": [1e6, 1.1e6, 1.2e6], "mon": [33.6, 34.1, 35.0],
         "th": [0.1, 0.15, 0.2], "TEMP": [300.0, 301.0, 302.0]},
        index=[0, 2, 5],  # gapped labels
    )
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((3, 10)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 10))
        g.create_dataset("frame_index", data=np.array([0, 2, 5], dtype=np.int64))
        write_scan_metadata(e, sd, [0, 2, 5])

    ds = read_scan(p, groups=("1d",))
    for col in ("i0", "mon", "th", "TEMP"):
        assert col in ds.data_vars
    np.testing.assert_allclose(ds["TEMP"].values, [300.0, 301.0, 302.0])
    assert list(ds["frame"].values) == [0, 2, 5]
    # metadata-only reader carries them too
    assert "mon" in read_scan_metadata(p).data_vars


def test_scan_metadata_dedupes_positioners(tmp_path):
    """When a motor is in both /entry/scan_data and a positioners group,
    read_scan loads it once (from scan_data) — no duplicate sample_*/th."""
    import pandas as pd
    from xrd_tools.io import write_scan_metadata, read_scan

    p = tmp_path / "dedup.nxs"
    sd = pd.DataFrame({"th": [0.1, 0.2], "i0": [1e6, 2e6]}, index=[0, 1])
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((2, 4)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([0, 1], dtype=np.int64))
        write_scan_metadata(e, sd, [0, 1])
        # a positioners group that also carries th (geometry motor)
        pg = e.create_group("sample/positioners/th")
        pg.attrs["NX_class"] = "NXpositioner"
        pg.create_dataset("value", data=np.array([0.1, 0.2], dtype=np.float32))

    ds = read_scan(p, groups=("1d",))
    assert "th" in ds.data_vars
    assert "sample_th" not in ds.data_vars   # not duplicated


def test_stale_scan_data_column_falls_back_to_positioner(tmp_path):
    """P2: a scan_data column whose length disagrees with the frame count
    is rejected on read — a valid same-named NXpositioner must still load
    (the rejected key shouldn't suppress the positioner fallback)."""
    import h5py
    from xrd_tools.io import write_scan_metadata, read_scan

    p = tmp_path / "stale.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((3, 4)))   # 3 frames
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([0, 1, 2], dtype=np.int64))
        # scan_data 'th' is STALE — only 2 rows for 3 frames (length mismatch).
        sd_grp = e.create_group("scan_data")
        sd_grp.attrs["NX_class"] = "NXcollection"
        sd_grp.create_dataset("frame_index", data=np.array([0, 1], dtype=np.int64))
        sd_grp.create_dataset("th", data=np.array([0.1, 0.2], dtype=np.float32))
        # A valid full-length positioner with the same name.
        pg = e.create_group("sample/positioners/th")
        pg.attrs["NX_class"] = "NXpositioner"
        pg.create_dataset("value",
                          data=np.array([0.1, 0.2, 0.3], dtype=np.float32))

    ds = read_scan(p, groups=("1d",))
    # th loaded from the positioner (length 3), not dropped because the
    # stale scan_data column was skipped.
    assert "th" in ds.data_vars
    assert ds["th"].shape == (3,)
    np.testing.assert_allclose(ds["th"].values, [0.1, 0.2, 0.3])


def test_scan_data_aligned_by_label_not_position(tmp_path):
    """P2: scan_data is attached BY LABEL, not positionally — integrated
    frames [0, 2] with scan_data stored in a different label order must
    still get each frame's own metadata, not a positional mismatch."""
    import h5py
    from xrd_tools.io import read_scan

    p = tmp_path / "reorder.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.attrs["NX_class"] = "NXdata"; g.attrs["signal"] = "intensity"
        g.create_dataset("intensity", data=np.zeros((2, 4)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([0, 2], dtype=np.int64))
        # scan_data covers the same labels {0, 2} but stored 2-then-0.
        sd = e.create_group("scan_data")
        sd.attrs["NX_class"] = "NXcollection"
        sd.create_dataset("frame_index", data=np.array([2, 0], dtype=np.int64))
        sd.create_dataset("th", data=np.array([0.22, 0.00], dtype=np.float32))

    ds = read_scan(p, groups=("1d",))
    # frame 0 → th 0.00, frame 2 → th 0.22 (aligned by label, not row order)
    assert list(ds["frame"].values) == [0, 2]
    np.testing.assert_allclose(ds["th"].values, [0.00, 0.22], atol=1e-6)


def test_partial_thumbnails_use_independent_frame_coordinate(tmp_path):
    import h5py
    from xrd_tools.io import read_scan

    p = tmp_path / "partial_thumbnails.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((2, 4)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([2, 0], dtype=np.int64))
        fg = e.create_group("frames/frame_0002")
        fg.create_dataset("thumbnail", data=np.ones((3, 4), dtype=np.uint8))

    ds = read_scan(p, groups=("1d",), include_thumbnails=True)
    assert ds["thumbnail"].dims == ("thumbnail_frame", "thumb_y", "thumb_x")
    assert list(ds["thumbnail_frame"].values) == [2]


def test_scan_data_duplicate_labels_rejected(tmp_path):
    import h5py
    import pytest
    from xrd_tools.io import read_scan

    p = tmp_path / "duplicate_metadata.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g = e.create_group("integrated_1d")
        g.create_dataset("intensity", data=np.zeros((2, 4)))
        g.create_dataset("q", data=np.linspace(0.5, 5.0, 4))
        g.create_dataset("frame_index", data=np.array([0, 1], dtype=np.int64))
        sd = e.create_group("scan_data")
        sd.create_dataset("frame_index", data=np.array([0, 0], dtype=np.int64))
        sd.create_dataset("th", data=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="duplicate labels"):
        read_scan(p, groups=("1d",))
