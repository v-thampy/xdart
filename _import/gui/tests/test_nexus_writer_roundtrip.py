# -*- coding: utf-8 -*-
"""Round-trip test that locks down the v2 NeXus on-disk format.

Acts as the safety net for the upcoming nexusformat-based writer
migration: we exercise the *current* h5py-based writer end-to-end,
then open the result with ``nexusformat.nxload`` and assert the tree
shape (NX_class, signal/axes, dataset shapes, units, mandatory
groups).  Any future writer that produces the same assertions will
be format-compatible with everything that already reads these files.

Why a synthetic sphere (not a real EwaldArch + pyFAI integrator):
constructing :class:`EwaldArch` pulls in pyFAI's azimuthal
integrator just to satisfy ``setup_integrator()`` in __init__.  The
writer never touches the integrator — it only reads attribute trees
off arches.  Duck-typed objects with the right attrs are enough and
keep the test fast / dep-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# nexusformat is the read-side gate.  If it isn't installed, skip;
# the test is the format-validation contract and only meaningful
# with the library present.
nx = pytest.importorskip("nexusformat.nexus")


# ---------------------------------------------------------------------------
# Synthetic sphere fixture
# ---------------------------------------------------------------------------

N_FRAMES = 4
N_Q = 32
N_CHI = 16


class _DuckResult1D:
    """Duck-typed IntegrationResult1D — only the attrs the writer reads."""

    def __init__(self, radial, intensity, sigma=None, unit="q_A^-1"):
        self.radial = np.asarray(radial, dtype=np.float32)
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.sigma = (
            np.asarray(sigma, dtype=np.float32) if sigma is not None else None
        )
        self.unit = unit


class _DuckResult2D:
    """Duck-typed IntegrationResult2D — note xdart's per-frame shape is
    ``(nq, nchi)`` so the writer transposes to ``(nchi, nq)`` before
    stacking into the (N, nchi, nq) tensor."""

    def __init__(self, radial, azimuthal, intensity,
                 unit="q_A^-1", azimuthal_unit="deg"):
        self.radial = np.asarray(radial, dtype=np.float32)
        self.azimuthal = np.asarray(azimuthal, dtype=np.float32)
        self.intensity = np.asarray(intensity, dtype=np.float32)
        self.unit = unit
        self.azimuthal_unit = azimuthal_unit
        self.sigma = None


class _DuckPONI:
    """Duck-typed PONI — six pyFAI geometry scalars."""

    def __init__(self):
        self.dist = 0.1
        self.poni1 = 0.05
        self.poni2 = 0.05
        self.rot1 = 0.0
        self.rot2 = 0.0
        self.rot3 = 0.0


class _DuckArch:
    """Minimal duck-typed EwaldArch."""

    def __init__(self, idx, *, nq=N_Q, nchi=N_CHI, seed=0):
        rng = np.random.default_rng(seed + idx)
        radial = np.linspace(0.5, 5.0, nq, dtype=np.float32)
        azim = np.linspace(-180.0, 180.0, nchi, endpoint=False, dtype=np.float32)

        self.idx = int(idx)
        self.poni = _DuckPONI()
        self.source_file = f"frame_{idx:04d}.tif"
        self.skip_map_raw = True
        self.map_raw = None
        self.bg_raw = 0
        self.mask = None
        self.int_1d = _DuckResult1D(
            radial=radial,
            intensity=rng.random(nq, dtype=np.float32),
            sigma=rng.random(nq, dtype=np.float32) * 0.1,
        )
        self.int_2d = _DuckResult2D(
            radial=radial,
            azimuthal=azim,
            intensity=rng.random((nq, nchi), dtype=np.float32),  # (nq, nchi)
        )
        self.gi_1d = {}
        self.gi_2d = {}
        # Thumbnail mirrors what arch.make_thumbnail produces — small 2D float.
        self.thumbnail = rng.random((64, 64), dtype=np.float32)


class _DuckArches(list):
    """List that masquerades as ArchSeries for the writer.

    The writer only needs ``list(sphere.arches)``-style iteration and
    ``arches[0].int_1d.radial`` random access; a plain list satisfies
    both.  Real ArchSeries adds lazy-disk-load + in-memory cache —
    irrelevant for a fully-in-memory test.
    """


class _DuckSphere:
    """Minimal duck-typed EwaldSphere."""

    def __init__(self, arches, *, scan_data=None, geometry=None,
                 global_mask=None):
        self.arches = _DuckArches(arches)
        self.scan_data = scan_data if scan_data is not None else pd.DataFrame()
        self.bai_1d_args = {"numpoints": N_Q}
        self.bai_2d_args = {"npt_rad": N_Q, "npt_azim": N_CHI}
        self.mg_args = {"wavelength": 1.0e-10}
        self.geometry = geometry
        self.global_mask = global_mask
        self.stitched_1d = None
        self.stitched_2d = None
        # Stitching writes only happen with finalize=True; not exercised here.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def written_nxs(tmp_path):
    """Build a synthetic 4-frame sphere and run it through the writer."""
    from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus

    arches = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
        "th": np.linspace(0.0, 0.3, N_FRAMES, dtype=np.float32),
        "i0": np.linspace(1e6, 1.2e6, N_FRAMES, dtype=np.float32),
    })
    # Mark a couple of pixels as masked so we can assert the mask group
    # is written.  Indices are flat-relative to a notional (256, 256)
    # detector — the values are arbitrary; the writer just stores them.
    global_mask = np.array([0, 1, 256, 65535], dtype=np.int64)

    sphere = _DuckSphere(arches, scan_data=scan_data, global_mask=global_mask)

    path = tmp_path / "roundtrip.nxs"
    save_sphere_to_nexus(sphere, path, mode="w", finalize=False)
    return path


@pytest.fixture
def written_nxs_with_stitched(tmp_path):
    """4-frame sphere whose ``stitched_1d``/``stitched_2d`` are populated.

    Exercises the ``finalize=True`` path of
    :func:`save_sphere_to_nexus`, which writes the stitched outputs
    only on the final save of the scan.
    """
    from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus

    nq, nchi = 64, 24
    s1 = _DuckResult1D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        intensity=np.random.default_rng(7).random(nq, dtype=np.float32),
        sigma=np.full(nq, 0.05, dtype=np.float32),
    )
    s2 = _DuckResult2D(
        radial=np.linspace(0.5, 5.0, nq, dtype=np.float32),
        azimuthal=np.linspace(-180, 180, nchi, endpoint=False,
                              dtype=np.float32),
        # Stitched 2D intensity is stored in the natural (nchi, nq) shape
        # (no transpose); this matches how stitch_2d returns its result.
        intensity=np.random.default_rng(8).random((nchi, nq), dtype=np.float32),
    )

    arches = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    sphere = _DuckSphere(
        arches,
        scan_data=pd.DataFrame({"tth": np.linspace(10, 14, N_FRAMES,
                                                   dtype=np.float32)}),
    )
    sphere.stitched_1d = s1
    sphere.stitched_2d = s2

    path = tmp_path / "stitched.nxs"
    save_sphere_to_nexus(sphere, path, mode="w", finalize=True)
    return path, nq, nchi


@pytest.fixture
def written_nxs_with_geometry(tmp_path):
    """4-frame sphere WITH a DiffractometerGeometry attached.

    Exercises the positioner + per-frame-geometry write paths that the
    bare ``written_nxs`` fixture doesn't touch.
    """
    from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
    from ssrl_xrd_tools.core.geometry import DiffractometerGeometry

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")

    arches = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
        "th": np.linspace(0.0, 0.3, N_FRAMES, dtype=np.float32),
    })
    sphere = _DuckSphere(arches, scan_data=scan_data, geometry=geom)

    path = tmp_path / "geometry.nxs"
    save_sphere_to_nexus(sphere, path, mode="w", finalize=False)
    return path


# ---------------------------------------------------------------------------
# Tree-shape assertions via nexusformat
# ---------------------------------------------------------------------------

def test_nxload_opens_cleanly(written_nxs):
    """The file is at minimum a valid NeXus tree per nxload."""
    root = nx.nxload(str(written_nxs))
    assert "entry" in root
    assert root["entry"].nxclass == "NXentry"


def test_entry_attrs(written_nxs):
    """NXentry has the expected class + default plot pointer."""
    root = nx.nxload(str(written_nxs))
    e = root["entry"]
    assert e.nxclass == "NXentry"
    # @default points at the auto-plot target — convention says NXdata.
    assert e.attrs.get("default") == "integrated_1d"


def test_integrated_1d_nxdata(written_nxs):
    """The stacked 1D pattern is an NXdata with signal/axes/units."""
    root = nx.nxload(str(written_nxs))
    g = root["entry/integrated_1d"]
    assert g.nxclass == "NXdata"
    assert g.attrs["signal"] == "intensity"
    axes = list(g.attrs["axes"])
    assert axes == ["frame_index", "q"]

    # Shape: (N_FRAMES, N_Q)
    assert g["intensity"].shape == (N_FRAMES, N_Q)
    assert g["q"].shape == (N_Q,)
    assert g["frame_index"].shape == (N_FRAMES,)
    assert g["sigma"].shape == (N_FRAMES, N_Q)

    # Units on the radial axis
    q_units = g["q"].attrs.get("units", "")
    assert q_units in ("1/angstrom", "1/nm")


def test_integrated_2d_nxdata(written_nxs):
    """The stacked 2D cake is an NXdata with axes [frame_index, chi, q]."""
    root = nx.nxload(str(written_nxs))
    g = root["entry/integrated_2d"]
    assert g.nxclass == "NXdata"
    assert g.attrs["signal"] == "intensity"
    axes = list(g.attrs["axes"])
    assert axes == ["frame_index", "chi", "q"]

    # Writer transposes per-frame from xdart's (nq, nchi) into (nchi, nq),
    # so the stacked tensor is (N_FRAMES, N_CHI, N_Q).
    assert g["intensity"].shape == (N_FRAMES, N_CHI, N_Q)
    assert g["q"].shape == (N_Q,)
    assert g["chi"].shape == (N_CHI,)

    assert g["chi"].attrs.get("units") in ("deg", "rad")


def test_frames_group_and_thumbnails(written_nxs):
    """Per-frame metadata lives under NXcollection/frames/frame_NNNN."""
    root = nx.nxload(str(written_nxs))
    frames = root["entry/frames"]
    assert frames.nxclass == "NXcollection"
    for i in range(N_FRAMES):
        sub = frames[f"frame_{i:04d}"]
        assert sub.nxclass == "NXcollection"
        # Thumbnail is uncompressed uint8/16 with vmin/vmax/dtype attrs.
        thumb = sub["thumbnail"]
        assert "vmin" in thumb.attrs
        assert "vmax" in thumb.attrs
        assert thumb.attrs["dtype"] in ("uint8", "uint16")


def test_instrument_groups(written_nxs):
    """Instrument tree: NXinstrument/NXdetector with mask + PONI scalars."""
    root = nx.nxload(str(written_nxs))
    instr = root["entry/instrument"]
    assert instr.nxclass == "NXinstrument"
    det = instr["detector"]
    assert det.nxclass == "NXdetector"

    # PONI scalars from arches[0] (six numbers)
    for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
        assert k in det

    # Global mask (flat pixel indices)
    assert "mask" in det
    assert det["mask"].nxdata.tolist() == [0, 1, 256, 65535]

    # Source/wavelength
    src = instr["source"]
    assert src.nxclass == "NXsource"
    assert "wavelength_A" in src


def test_reduction_provenance(written_nxs):
    """/entry/reduction is an NXprocess with program + versions + config."""
    root = nx.nxload(str(written_nxs))
    red = root["entry/reduction"]
    assert red.nxclass == "NXprocess"
    assert "program" in red
    assert "version" in red
    assert "date" in red
    assert red["versions"].nxclass == "NXcollection"
    assert "config" in red


def test_positioners_groups(written_nxs_with_geometry):
    """With a geometry set, motor positioners get NXsample / NXdetector trees.

    Locks down:
    * ``/entry/sample`` is :class:`NXsample` (not :class:`NXcollection`).
    * ``/entry/instrument/detector`` is :class:`NXdetector` (not
      :class:`NXinstrument` — the old h5py code had this stamped
      wrongly; the typed constructor port fixed it).
    * Each motor is an :class:`NXpositioner` with a ``value`` field
      whose ``@units`` attr is set.
    """
    root = nx.nxload(str(written_nxs_with_geometry))
    e = root["entry"]

    sample = e["sample"]
    assert sample.nxclass == "NXsample"
    assert sample["positioners"].nxclass == "NXcollection"
    th = sample["positioners/th"]
    assert th.nxclass == "NXpositioner"
    assert th["value"].shape == (N_FRAMES,)
    assert th["value"].attrs.get("units") == "deg"

    det = e["instrument/detector"]
    assert det.nxclass == "NXdetector"
    assert det["positioners"].nxclass == "NXcollection"
    tth = det["positioners/tth"]
    assert tth.nxclass == "NXpositioner"
    assert tth["value"].shape == (N_FRAMES,)


def test_per_frame_geometry(written_nxs_with_geometry):
    """``/entry/per_frame_geometry`` is an :class:`NXcollection` of
    derived rot1/rot2/rot3 + optional incident_angle, plus
    ``frame_index``.  Each derived array has a ``@units`` attr.
    """
    root = nx.nxload(str(written_nxs_with_geometry))
    g = root["entry/per_frame_geometry"]
    assert g.nxclass == "NXcollection"
    assert "frame_index" in g
    assert g["frame_index"].shape == (N_FRAMES,)
    # At minimum rot1/rot2/rot3 (the pyFAI per-frame rotations).
    for k in ("rot1", "rot2", "rot3"):
        assert k in g
        assert g[k].shape == (N_FRAMES,)
        assert g[k].attrs.get("units") == "rad"


def test_stitched_outputs(written_nxs_with_stitched):
    """``finalize=True`` writes ``stitched_1d`` / ``stitched_2d`` as
    NXdata with the same signal/axes conventions as their per-frame
    siblings (intensity signal; q on the 1D axis; chi+q on 2D).
    """
    path, nq, nchi = written_nxs_with_stitched
    root = nx.nxload(str(path))
    e = root["entry"]

    s1 = e["stitched_1d"]
    assert s1.nxclass == "NXdata"
    assert s1["intensity"].shape == (nq,)
    assert s1["q"].shape == (nq,)
    assert s1["q"].attrs.get("units") in ("1/angstrom", "1/nm")
    assert s1["sigma"].shape == (nq,)

    s2 = e["stitched_2d"]
    assert s2.nxclass == "NXdata"
    assert s2["intensity"].shape == (nchi, nq)
    assert s2["q"].shape == (nq,)
    assert s2["chi"].shape == (nchi,)
    assert s2["chi"].attrs.get("units") in ("deg", "rad")


def test_h5py_layout_is_idempotent(tmp_path):
    """Calling save_sphere_to_nexus twice on the same file is safe — the
    stacked datasets get replaced cleanly with no leftover ghost entries.

    This guards the idempotent ``_replace_ds`` pattern that the live-
    mode + batch flushes both rely on.
    """
    from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus

    arches = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
    })
    sphere = _DuckSphere(arches, scan_data=scan_data)

    path = tmp_path / "idempotent.nxs"
    save_sphere_to_nexus(sphere, path, mode="w")
    save_sphere_to_nexus(sphere, path, mode="a")

    # Second write should not have produced duplicate keys or shape drift.
    root = nx.nxload(str(path))
    assert root["entry/integrated_1d/intensity"].shape == (N_FRAMES, N_Q)
    assert len(list(root["entry/frames"])) == N_FRAMES
