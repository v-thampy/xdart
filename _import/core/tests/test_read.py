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

from ssrl_xrd_tools.io import (
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


def test_mismatched_positioner_length_is_skipped_not_fatal(tmp_path):
    """A per-frame column whose length != frame count (malformed/partial
    file, e.g. 4 integrated frames but 2 'th' positions) must be skipped
    with a warning, not crash the whole reader."""
    from ssrl_xrd_tools.io.nexus import read_scan, read_scan_metadata

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


def test_read_sphere_alias_deprecated(scan_file):
    """The old read_sphere* names still work but emit DeprecationWarning."""
    import warnings
    from ssrl_xrd_tools.io.nexus import (
        read_scan, read_scan_metadata, read_sphere, read_sphere_metadata,
    )
    p, _ = scan_file
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = read_sphere(p)
        meta = read_sphere_metadata(p)
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("read_sphere" in m for m in msgs)
    # aliases return the same thing as the canonical names
    assert set(ds.data_vars) == set(read_scan(p).data_vars)
    assert set(meta.coords) == set(read_scan_metadata(p).coords)


def test_scan_sugar(scan_file):
    p, ref = scan_file
    scan = open_scan(p)
    assert isinstance(scan, Scan)
    assert len(scan) == N_FRAMES
    np.testing.assert_array_equal(scan.frames, FRAME_LABELS)
    np.testing.assert_array_equal(scan.get_1d(1).intensity, ref["intensity_1d"][0])
    assert scan.get_2d(2).intensity.shape == (N_CHI, N_Q)
    assert "LaB6" in repr(scan) or "n_frames=5" in repr(scan)
