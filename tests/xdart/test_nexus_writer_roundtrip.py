# -*- coding: utf-8 -*-
"""Round-trip test that locks down the v2 NeXus on-disk format.

Acts as the safety net for the upcoming nexusformat-based writer
migration: we exercise the *current* h5py-based writer end-to-end,
then open the result with ``nexusformat.nxload`` and assert the tree
shape (NX_class, signal/axes, dataset shapes, units, mandatory
groups).  Any future writer that produces the same assertions will
be format-compatible with everything that already reads these files.

Why a synthetic scan (not a real LiveFrame + pyFAI integrator):
constructing :class:`LiveFrame` pulls in pyFAI's azimuthal
integrator just to satisfy ``setup_integrator()`` in __init__.  The
writer never touches the integrator — it only reads attribute trees
off frames.  Duck-typed objects with the right attrs are enough and
keep the test fast / dep-free.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# nexusformat is the read-side gate.  If it isn't installed, skip;
# the test is the format-validation contract and only meaningful
# with the library present.
nx = pytest.importorskip("nexusformat.nexus")


# ---------------------------------------------------------------------------
# Synthetic scan fixture
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
    """Minimal duck-typed LiveFrame."""

    def __init__(self, idx, *, nq=N_Q, nchi=N_CHI, seed=0):
        rng = np.random.default_rng(seed + idx)
        radial = np.linspace(0.5, 5.0, nq, dtype=np.float32)
        azim = np.linspace(-180.0, 180.0, nchi, endpoint=False, dtype=np.float32)

        self.idx = int(idx)
        self.poni = _DuckPONI()
        self.source_file = f"frame_{idx:04d}.tif"
        # R2: per-file frame index — for single-frame TIFs always 0.
        # Test cases for multi-frame sources (e.g. Eiger external links)
        # set this to the offset within the source file.
        self.source_frame_idx = 0
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
        # Thumbnail mirrors what frame.make_thumbnail produces — small 2D float.
        self.thumbnail = rng.random((64, 64), dtype=np.float32)


class _DuckArches:
    """Minimal LiveFrameSeries stand-in for the writer tests.

    Provides the three interfaces the writer needs:

    * ``.index`` — list of frame indices, matching
      :class:`LiveFrameSeries.index` (an attribute, not the inherited
      ``list.index`` method that the previous fixture exposed).
    * ``frames[idx]`` — fetch the frame with that *frame* index.
    * ``list(frames)`` — iterate frames in index order.

    Real LiveFrameSeries adds lazy-disk-load + an in-memory cache; this
    duck stores everything in a dict for in-memory speed.
    """

    def __init__(self, frames):
        self._by_idx = {a.idx: a for a in frames}
        self.index = [a.idx for a in frames]
        # Mirror LiveFrameSeries._in_memory — the writer's R4 helpers
        # (_representative_poni, _write_per_frame_metadata) read this
        # directly to avoid materialising the full frame list on every
        # save.  Tests use the same dict as _by_idx so every frame is
        # always "in memory" (real LiveFrameSeries caps this at 64 with
        # FIFO eviction; tests don't need to exercise that path).
        self._in_memory = self._by_idx

    def __iter__(self):
        return (self._by_idx[i] for i in self.index)

    def __getitem__(self, idx):
        return self._by_idx[idx]

    def __len__(self):
        return len(self.index)

    def append(self, frame):
        """Extend the series — used by the new write-then-extend test."""
        self._by_idx[frame.idx] = frame
        if frame.idx not in self.index:
            self.index.append(frame.idx)


class _DuckSphere:
    """Minimal duck-typed LiveScan."""

    def __init__(self, frames, *, scan_data=None, geometry=None,
                 global_mask=None, detector_shape=None, gi=False, skip_2d=False):
        self.frames = _DuckArches(frames)
        self.scan_data = scan_data if scan_data is not None else pd.DataFrame()
        self.bai_1d_args = {"numpoints": N_Q}
        self.bai_2d_args = {"npt_rad": N_Q, "npt_azim": N_CHI}
        self.mg_args = {"wavelength": 1.0e-10}
        self.geometry = geometry
        self.global_mask = global_mask
        self.detector_shape = detector_shape
        self.gi = gi
        self.skip_2d = skip_2d
        self.stitched_1d = None
        self.stitched_2d = None
        # Stitching writes only happen with finalize=True; not exercised here.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def written_nxs(tmp_path):
    """Build a synthetic 4-frame scan and run it through the writer."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
        "th": np.linspace(0.0, 0.3, N_FRAMES, dtype=np.float32),
        "i0": np.linspace(1e6, 1.2e6, N_FRAMES, dtype=np.float32),
    })
    # Mark a couple of pixels as masked so we can assert the mask group
    # is written.  Indices are flat-relative to a notional (256, 256)
    # detector — the values are arbitrary; the writer just stores them.
    global_mask = np.array([0, 1, 256, 65535], dtype=np.int64)

    scan = _DuckSphere(frames, scan_data=scan_data, global_mask=global_mask)

    path = tmp_path / "roundtrip.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    return path


@pytest.fixture
def written_nxs_with_stitched(tmp_path):
    """4-frame scan whose ``stitched_1d``/``stitched_2d`` are populated.

    Exercises the ``finalize=True`` path of
    :func:`save_scan_to_nexus`, which writes the stitched outputs
    only on the final save of the scan.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

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
        # stitch_2d returns IntegrationResult2D.intensity as (npt_rad,
        # npt_azim) == (nq, nchi); write_stitched stores it AS-IS with
        # dims (q, chi).  (The old xdart writer mislabelled the axes
        # [chi, q] over (nq, nchi) data — the delegated primitive fixes
        # that, so the fixture now uses the real (nq, nchi) orientation.)
        intensity=np.random.default_rng(8).random((nq, nchi), dtype=np.float32),
    )

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(
        frames,
        scan_data=pd.DataFrame({"tth": np.linspace(10, 14, N_FRAMES,
                                                   dtype=np.float32)}),
    )
    scan.stitched_1d = s1
    scan.stitched_2d = s2

    path = tmp_path / "stitched.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=True)
    return path, nq, nchi


@pytest.fixture
def written_nxs_with_geometry(tmp_path):
    """4-frame scan WITH a DiffractometerGeometry attached.

    Exercises the positioner + per-frame-geometry write paths that the
    bare ``written_nxs`` fixture doesn't touch.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.core.geometry import DiffractometerGeometry

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
        "th": np.linspace(0.0, 0.3, N_FRAMES, dtype=np.float32),
    })
    scan = _DuckSphere(frames, scan_data=scan_data, geometry=geom)

    path = tmp_path / "geometry.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    return path


# ---------------------------------------------------------------------------
# Tree-shape assertions via nexusformat
# ---------------------------------------------------------------------------

def test_nxload_opens_cleanly(written_nxs):
    """The file is at minimum a valid NeXus tree per nxload."""
    root = nx.nxload(str(written_nxs))
    assert "entry" in root
    assert root["entry"].nxclass == "NXentry"


def test_overwrite_save_is_atomic_on_failure(tmp_path, monkeypatch):
    import h5py
    from xdart.modules.ewald import nexus_writer

    path = tmp_path / "atomic.nxs"
    original = _DuckSphere([_DuckArch(idx=i) for i in range(2)])
    nexus_writer.save_scan_to_nexus(original, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        original_intensity = f["entry/integrated_1d/intensity"][()].copy()
        original_frame_index = f["entry/integrated_1d/frame_index"][()].copy()

    def fail_reduction(*args, **kwargs):
        raise RuntimeError("forced writer failure")

    monkeypatch.setattr(nexus_writer, "_write_reduction", fail_reduction)
    replacement = _DuckSphere([_DuckArch(idx=i, seed=100) for i in range(3)])

    with pytest.raises(RuntimeError, match="forced writer failure"):
        nexus_writer.save_scan_to_nexus(replacement, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f["entry/integrated_1d/frame_index"][()],
            original_frame_index,
        )
        np.testing.assert_allclose(
            f["entry/integrated_1d/intensity"][()],
            original_intensity,
        )
    assert not list(tmp_path.glob(".atomic.nxs.tmp-*"))


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

    # Units on the radial axis.  The shared write_integrated_stack
    # primitive stores the IntegrationResult's raw unit string verbatim
    # ("q_A^-1") rather than the old xdart "1/angstrom" remap — this
    # matches the headless write path so files are interchangeable.
    q_units = g["q"].attrs.get("units", "")
    assert q_units in ("q_A^-1", "1/angstrom", "1/nm")


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


def test_gui_integrated_stacks_use_configured_filter_never_lzf(written_nxs):
    # The GUI writer compresses its stacks with the resolved default -- lz4+shuffle
    # (hdf5plugin filter 32004) when available, else portable gzip+shuffle -- and
    # NEVER emits raw lzf (h5py-only filter, ARM64-macOS bus error).
    import h5py
    from xdart.modules.ewald.nexus_writer import INTEGRATED_STACK_COMPRESSION

    with h5py.File(written_nxs, "r") as f:
        for path in ("entry/integrated_1d/intensity",
                     "entry/integrated_1d/sigma",
                     "entry/integrated_2d/intensity"):
            ds = f[path]
            filters = dict(ds._filters)
            assert "lzf" not in filters, (path, filters)     # never raw lzf
            assert ds.shuffle is True, path
            if INTEGRATED_STACK_COMPRESSION == "lz4":
                assert "32004" in filters, (path, filters)   # hdf5plugin LZ4
            else:
                assert ds.compression == "gzip", (path, ds.compression)


def test_gi_nonuniform_stacked_2d_writer_stays_strict(tmp_path):
    """GI 2D outputs still need one shared axis before stacked writing."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=0), _DuckArch(idx=1)]
    frames[1].int_2d = _DuckResult2D(
        radial=np.linspace(0.6, 5.1, N_Q, dtype=np.float32),
        azimuthal=np.linspace(-170.0, 170.0, N_CHI, endpoint=False,
                              dtype=np.float32),
        intensity=np.ones((N_Q, N_CHI), dtype=np.float32),
    )
    scan = _DuckSphere(frames, gi=True)

    path = tmp_path / "gi_nonuniform_2d.nxs"
    with pytest.raises(ValueError, match="different q/chi axis"):
        save_scan_to_nexus(scan, path, mode="w", finalize=False)


def test_publication_validation_skips_dummy_gi_2d_but_keeps_1d(tmp_path, caplog):
    """All-dummy GI cakes should not block the valid 1D stack."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    caplog.set_level(logging.WARNING, logger="xdart.modules.ewald.nexus_writer")
    frame = _DuckArch(idx=0)
    frame.gi = True
    frame.int_2d = _DuckResult2D(
        radial=np.linspace(-1.0, 1.0, N_Q, dtype=np.float32),
        azimuthal=np.linspace(0.0, 3.0, N_CHI, dtype=np.float32),
        intensity=np.full((N_Q, N_CHI), -1.0, dtype=np.float32),
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )
    scan = _DuckSphere([frame], gi=True)

    path = tmp_path / "dummy_gi_2d.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        assert "integrated_1d" in f["entry"]
        np.testing.assert_array_equal(f["entry/integrated_1d/frame_index"][()], [0])
        assert "integrated_2d" not in f["entry"]
    assert "Skipping frame 0 2D write" in caplog.text


def test_publication_validation_filters_bad_2d_rows_per_frame(tmp_path):
    """A bad cake should not lose neighboring good 2D rows or any 1D rows."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(3)]
    frames[1].gi = True
    frames[1].int_2d = _DuckResult2D(
        radial=np.linspace(-1.0, 1.0, N_Q, dtype=np.float32),
        azimuthal=np.linspace(0.0, 3.0, N_CHI, dtype=np.float32),
        intensity=np.full((N_Q, N_CHI), -1.0, dtype=np.float32),
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )
    scan = _DuckSphere(frames, gi=True)

    path = tmp_path / "one_bad_gi_2d.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f["entry/integrated_1d/frame_index"][()], [0, 1, 2],
        )
        np.testing.assert_array_equal(
            f["entry/integrated_2d/frame_index"][()], [0, 2],
        )


def test_publication_validation_filters_bad_1d_without_blocking_2d(tmp_path):
    """1D and 2D validation are independent persistence gates."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(2)]
    frames[1].int_1d = _DuckResult1D(
        radial=np.asarray(frames[1].int_1d.radial),
        intensity=np.full(N_Q, np.nan, dtype=np.float32),
        sigma=frames[1].int_1d.sigma,
    )
    scan = _DuckSphere(frames)

    path = tmp_path / "one_bad_1d.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(
            f["entry/integrated_1d/frame_index"][()], [0],
        )
        np.testing.assert_array_equal(
            f["entry/integrated_2d/frame_index"][()], [0, 1],
        )


class _DuckArchReloadable1D(_DuckArch):
    """1D-only frame whose raw is reloadable from source (PERF-5).

    Carries no precomputed thumbnail and reports it can be skipped, mirroring
    a real ``LiveFrame`` in a ``skip_2d`` scan whose source master resolves.
    """

    def __init__(self, idx, **kw):
        super().__init__(idx, **kw)
        self.thumbnail = None

    def can_skip_thumbnail(self, skip_2d):
        return bool(skip_2d)

    def make_thumbnail(self, global_mask=None):  # must never be called when skipped
        raise AssertionError(
            "make_thumbnail called for a frame whose thumbnail should be skipped"
        )


def test_thumbnail_skipped_for_reloadable_1d_frames(tmp_path):
    """PERF-5: a 1D-only (skip_2d) scan whose frames are reloadable from source
    writes NO per-frame thumbnail (and never lazily computes one), but still
    writes the source ref so the Image Viewer can reload the raw on demand."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArchReloadable1D(idx=i) for i in range(2)]
    scan = _DuckSphere(frames, skip_2d=True)

    path = tmp_path / "skip2d_no_thumb.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    root = nx.nxload(str(path))
    for i in range(2):
        sub = root[f"entry/frames/frame_{i:04d}"]
        # No thumbnail dataset (the make_thumbnail AssertionError also guards
        # against a lazy save-time recompute).
        assert "thumbnail" not in sub
        # But the source ref is present so the viewer reloads raw on demand.
        assert "source" in sub
        assert "path" in sub["source"]
        assert "frame_index" in sub["source"]


def test_thumbnail_kept_for_2d_scan_even_if_reloadable(tmp_path):
    """PERF-5 guard: a 2D scan (skip_2d=False) keeps the thumbnail even when the
    frame would otherwise be reloadable — it is the 2D preview."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArchReloadable1D(idx=i) for i in range(2)]
    # thumbnail is None and make_thumbnail raises, so the writer must NOT try to
    # make one — for a 2D scan can_skip_thumbnail(False) is False, so the writer
    # would call make_thumbnail.  Give it a thumbnail so the 2D path persists it.
    for fr in frames:
        fr.thumbnail = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    scan = _DuckSphere(frames, skip_2d=False)

    path = tmp_path / "twod_keeps_thumb.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    root = nx.nxload(str(path))
    for i in range(2):
        assert "thumbnail" in root[f"entry/frames/frame_{i:04d}"]


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
        # R2 source ref: path + frame_index always present when the
        # frame carried a source_file (every test frame does).
        assert "source" in sub
        src = sub["source"]
        assert src.nxclass == "NXcollection"
        assert "path" in src
        assert "frame_index" in src
        # Source paths are persisted as absolute paths so Image Viewer can
        # resolve raw masters even when processed files live elsewhere.
        path = src["path"].nxdata
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        assert path == str((written_nxs.parent / f"frame_{i:04d}.tif").resolve())
        assert int(src["frame_index"].nxdata) == 0


def test_source_ref_multiframe(tmp_path):
    """Eiger-shaped source ref: same source_file, varying frame_index."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    # Mimic a NeXus wrangler stamp: every frame points to the same
    # master file, source_frame_idx = global index within the master.
    for i, a in enumerate(frames):
        a.source_file = "scan_001_master.h5"
        a.source_frame_idx = 1000 + i  # arbitrary global offset

    scan = _DuckSphere(frames)
    path = tmp_path / "multiframe.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    with h5py.File(path, "r") as f:
        for i in range(N_FRAMES):
            src = f[f"entry/frames/frame_{i:04d}/source"]
            v = src["path"][()]
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            assert v == str((path.parent / "scan_001_master.h5").resolve())
            assert int(src["frame_index"][()]) == 1000 + i


def test_replace_frame_indices_updates_only_targets(tmp_path):
    """Replace mode: slice-assign new int_1d/int_2d over the listed
    frames; leave all other rows byte-identical.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    n = 5
    frames = [_DuckArch(idx=i) for i in range(n)]
    scan = _DuckSphere(frames)
    path = tmp_path / "replace.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    # Snapshot original on-disk rows so we can verify untouched frames.
    with h5py.File(path, "r") as f:
        orig_1d = np.asarray(f["entry/integrated_1d/intensity"][()])
        orig_2d = np.asarray(f["entry/integrated_2d/intensity"][()])
        orig_frame_idx = list(f["entry/integrated_1d/frame_index"][()])
        assert orig_frame_idx == list(range(n))

    # Simulate a GUI reintegration: change int_1d/int_2d on frames 1 and 3.
    rng = np.random.default_rng(999)
    new_1d_arr = rng.random(N_Q, dtype=np.float32)
    new_2d_arr = rng.random((N_Q, N_CHI), dtype=np.float32)
    targets = [1, 3]
    for fi in targets:
        a = frames[fi]
        a.int_1d = _DuckResult1D(
            radial=np.asarray(a.int_1d.radial),
            intensity=new_1d_arr.copy(),
            sigma=a.int_1d.sigma,
        )
        a.int_2d = _DuckResult2D(
            radial=np.asarray(a.int_2d.radial),
            azimuthal=np.asarray(a.int_2d.azimuthal),
            intensity=new_2d_arr.copy(),
        )

    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=targets,
    )

    with h5py.File(path, "r") as f:
        # Frame_index order is unchanged (replace doesn't reorder).
        assert list(f["entry/integrated_1d/frame_index"][()]) == list(range(n))
        new_on_disk_1d = np.asarray(f["entry/integrated_1d/intensity"][()])
        new_on_disk_2d = np.asarray(f["entry/integrated_2d/intensity"][()])

    # Targets: rows updated to the new values.
    for fi in targets:
        assert np.allclose(new_on_disk_1d[fi], new_1d_arr)
        # 2D layout on disk is (frame, chi, q); frame int_2d is (q, chi).
        assert np.allclose(new_on_disk_2d[fi], new_2d_arr.T)

    # Non-targets: rows byte-identical to original.
    untouched = [i for i in range(n) if i not in targets]
    for fi in untouched:
        assert np.array_equal(new_on_disk_1d[fi], orig_1d[fi])
        assert np.array_equal(new_on_disk_2d[fi], orig_2d[fi])


def test_replace_filtered_2d_row_removes_stale_disk_row(tmp_path, caplog):
    """A rejected replace-mode 2D row must not leave old cake data behind."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    caplog.set_level(logging.WARNING, logger="xdart.modules.ewald.nexus_writer")
    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "replace_filtered_2d.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    with h5py.File(path, "r") as f:
        old_2d = np.asarray(f["entry/integrated_2d/intensity"][()])
        np.testing.assert_array_equal(
            f["entry/integrated_2d/frame_index"][()], [0, 1, 2],
        )

    new_1d = np.full(N_Q, 42.0, dtype=np.float32)
    frames[1].int_1d = _DuckResult1D(
        radial=np.asarray(frames[1].int_1d.radial),
        intensity=new_1d,
        sigma=frames[1].int_1d.sigma,
    )
    frames[1].int_2d = _DuckResult2D(
        radial=np.asarray(frames[1].int_2d.radial),
        azimuthal=np.asarray(frames[1].int_2d.azimuthal),
        intensity=np.full((N_Q, N_CHI), -1.0, dtype=np.float32),
    )

    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=[1],
    )

    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(
            f["entry/integrated_1d/intensity"][1], new_1d,
        )
        np.testing.assert_array_equal(
            f["entry/integrated_2d/frame_index"][()], [0, 2],
        )
        new_2d = np.asarray(f["entry/integrated_2d/intensity"][()])
        np.testing.assert_array_equal(new_2d[0], old_2d[0])
        np.testing.assert_array_equal(new_2d[1], old_2d[2])
    assert "Skipping frame 1 2D write" in caplog.text


def test_replace_with_no_existing_file_degrades_to_append(tmp_path):
    """First save with replace_frame_indices set should still succeed —
    no on-disk dataset means there's nothing to replace, so the writer
    falls back to append mode for those frames."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "replace_first.nxs"
    # User somehow calls replace before any append save existed.
    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=[0, 1, 2],
    )
    # Result: the file was created via the append fallback; all rows
    # present.
    with h5py.File(path, "r") as f:
        assert "entry/integrated_1d/intensity" in f
        assert f["entry/integrated_1d/intensity"].shape == (3, N_Q)


def test_reload_only_flag_round_trips(tmp_path):
    """Frames loaded from disk via _load_frame_v2 should carry
    is_reload_only=True; freshly-integrated frames handed from the
    wrangler should not.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.frame_series import _load_frame_v2
    import h5py

    frames = [_DuckArch(idx=i) for i in range(2)]
    scan = _DuckSphere(frames)
    path = tmp_path / "reload_flag.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    # Fresh frames (from the wrangler) — default is_reload_only=False.
    from xdart.modules.ewald.frame import LiveFrame
    fresh = LiveFrame(idx=0)
    assert fresh.is_reload_only is False

    # Reloaded frames — the loader stamps the flag.
    with h5py.File(path, "r") as f:
        loaded = _load_frame_v2(f, 0, static=False, gi=False)
    assert loaded.is_reload_only is True


def test_has_reload_only_frames_scan_helper(tmp_path):
    """LiveScan.has_reload_only_frames mirrors the in-memory cache."""
    from xdart.modules.ewald.scan import LiveScan
    from xdart.modules.ewald.frame import LiveFrame

    scan = LiveScan(name='t', data_file=str(tmp_path / 'none.nxs'))

    # No frames → False (nothing to re-integrate, but nothing flagged).
    assert scan.has_reload_only_frames() is False

    # A fresh frame (wrangler hand-off) — flag stays False.
    fresh = LiveFrame(idx=0)
    scan.frames.stash(fresh)
    scan.frames.index.append(0)
    assert scan.has_reload_only_frames() is False

    # Mark one as reload-only → True.
    fresh.is_reload_only = True
    assert scan.has_reload_only_frames() is True


def test_acquire_save_reload_reintegrate_save_reload(tmp_path):
    """End-to-end: simulate the acquire → save → reload → reintegrate
    → save → reload workflow that the GUI exercises, verifying that
    the second reload sees the post-reintegration values on disk.

    Uses the real LiveFrameSeries (not the duck) so the test covers the
    actual lazy-load + replace-frames code path.  Re-integration
    here is faked by mutating the int_1d values on the in-memory
    frames — we're testing the writer's replace mode, not pyFAI.
    """
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.frame_series import _load_frame_v2

    # --- Phase 1: acquire + save -------------------------------------
    frames = [_DuckArch(idx=i) for i in range(4)]
    scan = _DuckSphere(frames)
    path = tmp_path / "lifecycle.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=True)

    # --- Phase 2: reload --------------------------------------------
    with h5py.File(path, "r") as f:
        original_intensity = f["entry/integrated_1d/intensity"][1].copy()

    # --- Phase 3: simulate reintegration on frames 1 + 2 ------------
    new_intensity_frame1 = np.full(N_Q, 7.0, dtype=np.float32)
    new_intensity_frame2 = np.full(N_Q, 9.0, dtype=np.float32)
    frames[1].int_1d = _DuckResult1D(
        radial=np.asarray(frames[1].int_1d.radial),
        intensity=new_intensity_frame1.copy(),
        sigma=frames[1].int_1d.sigma,
    )
    frames[2].int_1d = _DuckResult1D(
        radial=np.asarray(frames[2].int_1d.radial),
        intensity=new_intensity_frame2.copy(),
        sigma=frames[2].int_1d.sigma,
    )

    # --- Phase 4: save replace --------------------------------------
    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=[1, 2],
    )

    # --- Phase 5: reload again, confirm updates persisted -----------
    with h5py.File(path, "r") as f:
        assert np.allclose(f["entry/integrated_1d/intensity"][1],
                           new_intensity_frame1)
        assert np.allclose(f["entry/integrated_1d/intensity"][2],
                           new_intensity_frame2)
        # Frame 1's old value should be different from the new one.
        assert not np.allclose(original_intensity, new_intensity_frame1)
        # The replace save also re-wrote /entry/reduction so the
        # persisted args reflect the (possibly tweaked) integration
        # parameters.
        assert "entry/reduction" in f


def test_replace_with_shape_change_falls_back_to_full_rewrite(tmp_path):
    """C3: when reintegration changes numpoints/npt_*, slice-assign
    can't work — writer should delete the group and rewrite from
    scratch so the q/chi axes refresh too.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    n = 3
    frames = [_DuckArch(idx=i, nq=N_Q, nchi=N_CHI) for i in range(n)]
    scan = _DuckSphere(frames)
    path = tmp_path / "shape_change.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    with h5py.File(path, "r") as f:
        assert f["entry/integrated_1d/intensity"].shape == (n, N_Q)
        assert f["entry/integrated_2d/intensity"].shape == (n, N_CHI, N_Q)

    # Simulate user changing numpoints from N_Q to N_Q * 2.
    NQ2 = N_Q * 2
    NCHI2 = N_CHI * 2
    rng = np.random.default_rng(7)
    radial_new = np.linspace(0.5, 5.0, NQ2, dtype=np.float32)
    chi_new = np.linspace(-180.0, 180.0, NCHI2, endpoint=False,
                          dtype=np.float32)
    for a in frames:
        a.int_1d = _DuckResult1D(
            radial=radial_new,
            intensity=rng.random(NQ2, dtype=np.float32),
            sigma=rng.random(NQ2, dtype=np.float32) * 0.1,
        )
        a.int_2d = _DuckResult2D(
            radial=radial_new,
            azimuthal=chi_new,
            intensity=rng.random((NQ2, NCHI2), dtype=np.float32),
        )

    # Trigger replace mode for all frames.  Shape change should be
    # detected and the writer should rewrite the stacked datasets +
    # refresh the q/chi axes.
    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=[0, 1, 2],
    )

    with h5py.File(path, "r") as f:
        # 1D: new shape, new q axis.
        assert f["entry/integrated_1d/intensity"].shape == (n, NQ2)
        assert f["entry/integrated_1d/q"].shape == (NQ2,)
        # 2D: new (chi, q) row shape, new chi axis.
        assert f["entry/integrated_2d/intensity"].shape == (n, NCHI2, NQ2)
        assert f["entry/integrated_2d/chi"].shape == (NCHI2,)


def test_replace_unknown_frame_idx_is_silent(tmp_path):
    """Listing a frame_idx that doesn't exist on disk: no error, no
    write, other targets still processed."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "replace_unknown.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    # Mutate frame 1 only.
    new_arr = np.full(N_Q, 99.0, dtype=np.float32)
    frames[1].int_1d = _DuckResult1D(
        radial=np.asarray(frames[1].int_1d.radial),
        intensity=new_arr.copy(),
        sigma=frames[1].int_1d.sigma,
    )
    # Pass valid + invalid frame idx.
    save_scan_to_nexus(
        scan, path, entry="entry", finalize=False,
        replace_frame_indices=[1, 999],
    )
    with h5py.File(path, "r") as f:
        assert np.allclose(f["entry/integrated_1d/intensity"][1], new_arr)


def test_source_ref_omitted_when_no_source_file(tmp_path):
    """No source/ subgroup is written when frame.source_file is empty."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    frames = [_DuckArch(idx=i) for i in range(2)]
    for a in frames:
        a.source_file = ""  # no source
    scan = _DuckSphere(frames)
    path = tmp_path / "no_source.nxs"
    save_scan_to_nexus(scan, path, entry="entry", finalize=False)

    with h5py.File(path, "r") as f:
        for i in range(2):
            assert "source" not in f[f"entry/frames/frame_{i:04d}"]


def test_late_out_of_order_frame_is_written(tmp_path):
    """P2: a late frame whose label lands *between* already-saved labels
    (save [0, 2], then frame 1 arrives) must get its integrated row written
    — the old positional-tail selection skipped it."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    # First save with frames 0 and 2 only.
    scan = _DuckSphere([_DuckArch(idx=0), _DuckArch(idx=2)])
    path = tmp_path / "late.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    with h5py.File(path, "r") as f:
        assert sorted(f["entry/integrated_1d/frame_index"][()]) == [0, 2]

    # Frame 1 arrives late; rebuild the scan with [0, 1, 2] in order.
    scan2 = _DuckSphere([_DuckArch(idx=0), _DuckArch(idx=1), _DuckArch(idx=2)])
    save_scan_to_nexus(scan2, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        on_disk = sorted(int(x) for x in f["entry/integrated_1d/frame_index"][()])
        assert on_disk == [0, 1, 2]            # frame 1 now present
        assert f["entry/integrated_1d/intensity"].shape[0] == 3


def test_scan_data_grows_on_incremental_saves(tmp_path):
    """P1: scan_data (and positioners/geometry) must track the integrated
    frame count across per-frame saves — not freeze at the first save's
    length (live mode never passes finalize=True)."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.core.geometry import DiffractometerGeometry
    import h5py

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")

    def _scan(n):
        frames = [_DuckArch(idx=i) for i in range(n)]
        sd = pd.DataFrame({
            "tth": np.linspace(10.0, 14.0, n, dtype=np.float32),
            "th": np.linspace(0.0, 0.3, n, dtype=np.float32),
            "i0": np.linspace(1e6, 1.2e6, n, dtype=np.float32),
        }, index=range(n))
        return _DuckSphere(frames, scan_data=sd, geometry=geom)

    path = tmp_path / "grow.nxs"
    save_scan_to_nexus(_scan(2), path, mode="w", finalize=False)   # 2 frames
    save_scan_to_nexus(_scan(5), path, mode="a", finalize=False)   # grows to 5

    with h5py.File(path, "r") as f:
        assert f["entry/integrated_1d/intensity"].shape[0] == 5
        # scan_data + positioners + geometry all caught up to 5 (not stuck at 2)
        assert f["entry/scan_data/frame_index"].shape[0] == 5
        assert f["entry/scan_data/i0"].shape[0] == 5
        assert f["entry/sample/positioners/th/value"].shape[0] == 5
        assert f["entry/per_frame_geometry/frame_index"].shape[0] == 5


def test_load_frame_v2_uses_independent_2d_rows_units_and_sigma(tmp_path):
    """1D and 2D stacks may have different label order and units."""
    import h5py
    from xdart.modules.ewald.frame_series import _load_frame_v2

    p = tmp_path / "independent_2d.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("frame_index", data=np.array([0, 2]))
        g1.create_dataset("q", data=np.array([1.0, 2.0]))
        g1.create_dataset("intensity", data=np.array([[10.0, 11.0], [20.0, 21.0]]))
        g2 = e.create_group("integrated_2d")
        g2.create_dataset("frame_index", data=np.array([2, 0]))
        q = g2.create_dataset("q", data=np.array([3.0, 4.0]))
        q.attrs["units"] = "2th_deg"
        chi = g2.create_dataset("chi", data=np.array([-1.0, 1.0]))
        chi.attrs["units"] = "deg"
        g2.create_dataset(
            "intensity",
            data=np.array([[[200.0, 201.0], [202.0, 203.0]],
                           [[100.0, 101.0], [102.0, 103.0]]]),
        )
        g2.create_dataset(
            "sigma",
            data=np.array([[[20.0, 21.0], [22.0, 23.0]],
                           [[10.0, 11.0], [12.0, 13.0]]]),
        )

    with h5py.File(p, "r") as f:
        frame = _load_frame_v2(f, 0, static=False, gi=False)
    np.testing.assert_array_equal(frame.int_2d.intensity,
                                  [[100.0, 102.0], [101.0, 103.0]])
    np.testing.assert_array_equal(frame.int_2d.sigma,
                                  [[10.0, 12.0], [11.0, 13.0]])
    assert frame.int_2d.unit == "2th_deg"


def test_load_frame_v2_restores_scan_info_for_normalization(tmp_path):
    """Reloaded frames need scan_info for GUI/headless monitor normalization."""
    import h5py
    from xdart.modules.ewald.frame_series import _load_frame_v2

    p = tmp_path / "scan_info.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        g1 = e.create_group("integrated_1d")
        g1.create_dataset("frame_index", data=np.array([10, 17], dtype=np.int64))
        q = g1.create_dataset("q", data=np.array([1.0, 2.0]))
        q.attrs["units"] = "q_A^-1"
        g1.create_dataset("intensity", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        sd = e.create_group("scan_data")
        sd.create_dataset("frame_index", data=np.array([10, 17], dtype=np.int64))
        sd.create_dataset("i0", data=np.array([100.0, 250.0]))
        sd.create_dataset("th", data=np.array([0.1, 0.2]))

    with h5py.File(p, "r") as f:
        frame = _load_frame_v2(f, 17, static=False, gi=False)

    assert frame.scan_info["i0"] == 250.0
    assert frame.scan_info["th"] == pytest.approx(0.2)


def test_frame_position_cache_separates_groups_and_clears_recreated_rows(tmp_path):
    import h5py
    from xdart.modules.ewald.frame_series import (
        _frame_position,
        clear_frame_position_cache,
    )

    p = tmp_path / "row_cache.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        for name, labels in (("integrated_1d", [0, 1, 2]),
                             ("integrated_2d", [2, 1, 0])):
            g = e.create_group(name)
            g.create_dataset("frame_index", data=np.asarray(labels))
        assert _frame_position(f, 0, "integrated_1d") == 0
        assert _frame_position(f, 0, "integrated_2d") == 2
        del e["integrated_1d"]
        g = e.create_group("integrated_1d")
        g.create_dataset("frame_index", data=np.array([0, 9, 2]))
        clear_frame_position_cache(f.filename)
        assert _frame_position(f, 1, "integrated_1d") is None
        assert _frame_position(f, 9, "integrated_1d") == 1


def test_partial_same_size_axis_change_requires_full_reintegration(tmp_path):
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "partial_axis_change.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    frame = frames[1]
    frame.int_1d = _DuckResult1D(
        radial=np.linspace(1.0, 6.0, N_Q, dtype=np.float32),
        intensity=np.asarray(frame.int_1d.intensity),
    )
    with pytest.raises(ValueError, match="Recompute and save every frame"):
        save_scan_to_nexus(
            scan, path, replace_frame_indices=[1], finalize=False,
        )


def test_write_cursor_fast_path_reads_only_tail_labels():
    """The trusted periodic-save path must not materialize all disk labels."""
    from xdart.modules.ewald.nexus_writer import (
        NexusWriteCursor,
        _index_structure_signature,
        _new_frames_for_write,
    )
    from xdart.modules.ewald.frame_series import _IndexedList

    class _NoFullReadLabels:
        shape = (2,)

        def __getitem__(self, key):
            if key == ():
                raise AssertionError("fast path read the full frame_index")
            return [0, 1][key]

    class _Intensity:
        shape = (2, N_Q)

    class _FakeH5(dict):
        pass

    group_path = "entry/integrated_1d"
    h5f = _FakeH5({group_path: {"intensity": _Intensity(),
                                "frame_index": _NoFullReadLabels()}})
    scan = _DuckSphere([_DuckArch(idx=i) for i in range(2)])
    scan.frames.index = _IndexedList(scan.frames.index)
    cursor = NexusWriteCursor(
        path="fake",
        groups={group_path: (2, 1, _index_structure_signature(scan.frames.index, 2))},
    )
    scan.frames.append(_DuckArch(idx=2))
    frames, existing_n = _new_frames_for_write(scan, h5f, group_path, cursor)
    assert existing_n == 2
    assert [frame.idx for frame in frames] == [2]


def test_same_scan_metadata_cursor_appends_new_tail(tmp_path):
    from xrd_tools.core.geometry import DiffractometerGeometry
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    scan = _DuckSphere(
        [_DuckArch(idx=0), _DuckArch(idx=1)],
        scan_data=pd.DataFrame({"tth": [10.0, 11.0], "th": [0.1, 0.2]},
                               index=[0, 1]),
        geometry=geom,
    )
    path = tmp_path / "metadata_tail.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    scan.frames.append(_DuckArch(idx=2))
    scan.scan_data.loc[2] = {"tth": 12.0, "th": 0.3}
    save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entry/scan_data/frame_index"][()], [0, 1, 2])
        np.testing.assert_allclose(f["entry/scan_data/tth"][()], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(
            f["entry/sample/positioners/frame_index"][()], [0, 1, 2],
        )
        np.testing.assert_array_equal(
            f["entry/per_frame_geometry/frame_index"][()], [0, 1, 2],
        )


def test_resume_missing_metadata_groups_rebuilds_full_history(tmp_path):
    from xrd_tools.core.geometry import DiffractometerGeometry
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    path = tmp_path / "resume_metadata.nxs"
    scan = _DuckSphere(
        [_DuckArch(idx=0), _DuckArch(idx=1)],
        scan_data=pd.DataFrame({"tth": [10.0, 11.0], "th": [0.1, 0.2]},
                               index=[0, 1]),
        geometry=None,
    )
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    scan.geometry = DiffractometerGeometry.two_circle(tth="tth", th="th")
    scan.frames.append(_DuckArch(idx=2))
    scan.scan_data.loc[2] = {"tth": 12.0, "th": 0.3}
    save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entry/scan_data/frame_index"][()], [0, 1, 2])
        np.testing.assert_array_equal(
            f["entry/sample/positioners/frame_index"][()], [0, 1, 2],
        )
        np.testing.assert_array_equal(
            f["entry/per_frame_geometry/frame_index"][()], [0, 1, 2],
        )


def test_xdart_save_preflights_1d_2d_before_mutation(tmp_path):
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    path = tmp_path / "xdart_no_half_commit.nxs"
    scan = _DuckSphere([_DuckArch(idx=0), _DuckArch(idx=1)])
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    frame = _DuckArch(idx=2)
    frame.int_2d.radial = np.linspace(9.0, 10.0, N_Q, dtype=np.float32)
    scan.frames.append(frame)
    with pytest.raises(ValueError, match="Integration settings changed"):
        save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entry/integrated_1d/frame_index"][()], [0, 1])
        np.testing.assert_array_equal(f["entry/integrated_2d/frame_index"][()], [0, 1])


def test_write_cursor_reconciles_after_interior_relabel(tmp_path):
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    path = tmp_path / "cursor_relabel.nxs"
    scan = _DuckSphere([_DuckArch(idx=0), _DuckArch(idx=1), _DuckArch(idx=2)])
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    frame = scan.frames._by_idx.pop(1)
    frame.idx = 9
    scan.frames._by_idx[9] = frame
    scan.frames.index[1] = 9
    scan.frames.append(_DuckArch(idx=3))
    with pytest.raises(ValueError, match="persisted frame labels"):
        save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entry/integrated_1d/frame_index"][()], [0, 1, 2])
        np.testing.assert_array_equal(f["entry/integrated_2d/frame_index"][()], [0, 1, 2])


def test_instrument_mask_rewrites_when_live_mask_changes(tmp_path):
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    path = tmp_path / "instrument_dirty.nxs"
    scan = _DuckSphere([_DuckArch(idx=0)], global_mask=np.array([0, 1]))
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    scan.global_mask = np.array([7, 8, 9])
    scan.frames.append(_DuckArch(idx=1))
    save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["entry/instrument/detector/mask"][()], [7, 8, 9])


def test_instrument_detector_shape_rewrites_when_live_shape_changes(tmp_path):
    """Regression (review P2): detector_shape is part of the instrument
    fingerprint, so changing ONLY the shape on a later append re-writes the
    instrument group instead of leaving a stale value on disk."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    path = tmp_path / "dshape_dirty.nxs"
    scan = _DuckSphere([_DuckArch(idx=0)], detector_shape=(100, 100))
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    scan.detector_shape = (200, 250)            # ONLY the shape changes
    scan.frames.append(_DuckArch(idx=1))
    save_scan_to_nexus(scan, path, mode="a", finalize=False)

    with h5py.File(path, "r") as f:
        assert (f["entry/instrument/detector/detector_shape"][()].tolist()
                == [200, 250])


def test_detector_shape_persisted_and_reloads(tmp_path):
    """detector_shape (full-res raw shape) round-trips: written as an NXdetector
    child and restored onto scan.detector_shape by the reader.  Lets a reloaded
    thumbnail-only scan map the detector gap mask into thumbnail coordinates
    without a resident full-res frame (codex P2 / cold-reload edge)."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan
    import h5py

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(frames, global_mask=np.array([0, 7, 64], dtype=np.int64),
                       detector_shape=(2167, 2070))
    path = tmp_path / "dshape.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    # persisted as a plain NXdetector child (external readers ignore it)
    with h5py.File(path, "r") as f:
        assert (f["entry/instrument/detector/detector_shape"][()].tolist()
                == [2167, 2070])

    # restored onto a real LiveScan by the reader
    rescan = LiveScan(data_file=str(path))
    rescan.load_from_h5()
    assert rescan.detector_shape == (2167, 2070)


def test_no_detector_shape_writes_no_dataset(written_nxs):
    """Additive / byte-compat: a scan without detector_shape writes NO such
    dataset (so existing files + the byte-compat fixtures are unchanged)."""
    import h5py
    with h5py.File(str(written_nxs), "r") as f:
        assert "detector_shape" not in f["entry/instrument/detector"]


def test_integrated_compression_env_override(monkeypatch):
    """XDART_INTEGRATED_COMPRESSION lets the user A/B the integrated-stack
    compression from the shell before launching xdart (default lz4 when hdf5plugin
    is available, else gzip; read once at import)."""
    from xdart.modules.ewald import nexus_writer as nw
    monkeypatch.delenv("XDART_INTEGRATED_COMPRESSION", raising=False)
    assert nw._resolve_integrated_compression() in ("lz4", "gzip")  # unset -> default
    for off in ("none", "None", "OFF", "0", "false", "no", "", "  none  "):
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", off)
        assert nw._resolve_integrated_compression() is None, off
    monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "gzip")
    assert nw._resolve_integrated_compression() == "gzip"
    monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "blosc")     # unknown -> gzip
    assert nw._resolve_integrated_compression() == "gzip"


def test_instrument_groups(written_nxs):
    """Instrument tree: NXinstrument/NXdetector with mask + PONI scalars."""
    root = nx.nxload(str(written_nxs))
    instr = root["entry/instrument"]
    assert instr.nxclass == "NXinstrument"
    det = instr["detector"]
    assert det.nxclass == "NXdetector"

    # PONI scalars from frames[0] (six numbers)
    for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
        assert k in det

    # Global mask (flat pixel indices)
    assert "mask" in det
    assert det["mask"].nxdata.tolist() == [0, 1, 256, 65535]

    # Source/wavelength: the fixture has no integrator and only the mg_args
    # 1e-10 (1.0 Å) sentinel, so the writer must NOT persist a misleading
    # wavelength_A (A2).  A real wavelength is exercised in the dedicated
    # wavelength tests below.
    assert "source" not in instr


def test_wavelength_not_written_for_sentinel_only(tmp_path):
    # A2: with no integrator and only the mg_args 1e-10 (1.0 Å) sentinel,
    # the writer must skip source/wavelength_A rather than persist 1.0 Å.
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    scan = _DuckSphere([_DuckArch(idx=0)])
    assert scan.mg_args["wavelength"] == 1.0e-10        # the sentinel
    assert getattr(scan, "_cached_integrator", None) is None

    path = tmp_path / "no_wl.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    root = nx.nxload(str(path))
    assert "source" not in root["entry/instrument"]


def test_wavelength_prefers_integrator_over_mg_args(tmp_path):
    # A2: a real integrator wavelength wins over mg_args (and over the
    # sentinel); the persisted wavelength_A is the integrator value in Å.
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    scan = _DuckSphere([_DuckArch(idx=0)])
    scan.mg_args = {"wavelength": 1.0e-10}              # sentinel-ish, must lose
    scan._cached_integrator = SimpleNamespace(wavelength=0.7293e-10)

    path = tmp_path / "wl_from_ai.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    root = nx.nxload(str(path))
    src = root["entry/instrument/source"]
    assert src.nxclass == "NXsource"
    assert float(src["wavelength_A"].nxdata) == pytest.approx(0.7293)


def test_reload_restores_persisted_wavelength_into_mg_args(tmp_path):
    # G1: reloaded scans have no live integrator.  The loader must restore the
    # real v2 wavelength stamp so display Q↔2θ conversion does not fall back to
    # the LiveScan constructor's 1e-10 m sentinel.
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan

    scan = _DuckSphere([_DuckArch(idx=0)])
    scan.mg_args = {"wavelength": 1.0e-10}
    scan._cached_integrator = SimpleNamespace(wavelength=0.7293e-10)

    path = tmp_path / "wl_reload.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    loaded = LiveScan(name="wl", data_file=str(path))
    loaded.load_from_h5(replace=True, mode="r")

    assert loaded.mg_args["wavelength"] == pytest.approx(0.7293e-10)


def test_reload_clears_stale_persisted_wavelength_when_missing(tmp_path):
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan

    with_wl = _DuckSphere([_DuckArch(idx=0)])
    with_wl._cached_integrator = SimpleNamespace(wavelength=0.7293e-10)
    path_with_wl = tmp_path / "with_wl.nxs"
    save_scan_to_nexus(with_wl, path_with_wl, mode="w", finalize=False)

    no_wl = _DuckSphere([_DuckArch(idx=0)])
    path_no_wl = tmp_path / "no_wl.nxs"
    save_scan_to_nexus(no_wl, path_no_wl, mode="w", finalize=False)

    loaded = LiveScan(name="wl", data_file=str(path_with_wl))
    loaded.load_from_h5(replace=True, mode="r")
    assert loaded.mg_args["wavelength"] == pytest.approx(0.7293e-10)

    loaded.data_file = str(path_no_wl)
    loaded.load_from_h5(replace=True, mode="r")

    assert getattr(loaded, "_persisted_wavelength_m", None) is None
    assert loaded.mg_args["wavelength"] == pytest.approx(1.0e-10)


def test_reload_restores_wavelength_in_append_mode(tmp_path):
    # T0-2: Append runs load with replace=False, mode='a' while load_from_h5
    # holds the file open — the restore must read through the already-open
    # handle, not a second h5py.File open of the same path.
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan

    scan = _DuckSphere([_DuckArch(idx=0)])
    scan.mg_args = {"wavelength": 1.0e-10}
    scan._cached_integrator = SimpleNamespace(wavelength=0.7293e-10)
    path = tmp_path / "wl_append.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    loaded = LiveScan(name="wl", data_file=str(path))
    loaded.load_from_h5(replace=False, mode="a")

    assert loaded.mg_args["wavelength"] == pytest.approx(0.7293e-10)
    assert loaded._persisted_wavelength_m == pytest.approx(0.7293e-10)


def test_reset_clears_persisted_wavelength(tmp_path):
    # T0-1: reset() is a data-identity change without a v2 load — the
    # wavelength restored from the previously loaded file must not survive it
    # (it short-circuits _get_wavelength ahead of the current file's stamp).
    from types import SimpleNamespace
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan

    scan = _DuckSphere([_DuckArch(idx=0)])
    scan._cached_integrator = SimpleNamespace(wavelength=0.7293e-10)
    path = tmp_path / "wl_reset.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    loaded = LiveScan(name="wl", data_file=str(path))
    loaded.load_from_h5(replace=True, mode="r")
    assert loaded._persisted_wavelength_m == pytest.approx(0.7293e-10)

    loaded.reset()

    assert loaded._persisted_wavelength_m is None
    assert loaded.mg_args["wavelength"] == pytest.approx(1.0e-10)


def test_source_base_stamp_failure_fails_save_loudly(tmp_path, monkeypatch):
    # G2: relative per-frame source paths without entry/@source_base are
    # misleadingly portable-looking but unresolvable.  The save must fail loudly
    # if the attr stamp cannot be written.
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    original_setitem = h5py._hl.attrs.AttributeManager.__setitem__

    def fail_source_base(self, name, value):
        if name == "source_base":
            raise OSError("attr write blocked")
        return original_setitem(self, name, value)

    monkeypatch.setattr(h5py._hl.attrs.AttributeManager, "__setitem__",
                        fail_source_base)
    scan = _DuckSphere([_DuckArch(idx=0)])
    scan.source_base = str(tmp_path)

    with pytest.raises(RuntimeError, match="@source_base"):
        save_scan_to_nexus(scan, tmp_path / "source_base_fail.nxs",
                           mode="w", finalize=False)


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
    assert s1["q"].attrs.get("units") in ("q_A^-1", "1/angstrom", "1/nm")
    assert s1["sigma"].shape == (nq,)

    s2 = e["stitched_2d"]
    assert s2.nxclass == "NXdata"
    # stitched_2d is stored AS-IS (q, chi) == (nq, nchi) — unlike the
    # per-frame integrated_2d stack which is (frame, chi, q).
    assert s2["intensity"].shape == (nq, nchi)
    assert s2["q"].shape == (nq,)
    assert s2["chi"].shape == (nchi,)
    assert s2["chi"].attrs.get("units") in ("deg", "rad")


def test_h5py_layout_is_idempotent(tmp_path):
    """Calling save_scan_to_nexus twice on the same file is safe — the
    stacked datasets get replaced cleanly with no leftover ghost entries.

    This guards the idempotent ``_replace_ds`` pattern that the live-
    mode + batch flushes both rely on.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
    })
    scan = _DuckSphere(frames, scan_data=scan_data)

    path = tmp_path / "idempotent.nxs"
    save_scan_to_nexus(scan, path, mode="w")
    save_scan_to_nexus(scan, path, mode="a")

    # Second write should not have produced duplicate keys or shape drift.
    root = nx.nxload(str(path))
    assert root["entry/integrated_1d/intensity"].shape == (N_FRAMES, N_Q)
    assert len(list(root["entry/frames"])) == N_FRAMES


def test_incremental_save_appends_only_new_frames(tmp_path):
    """P1: per-save Python prep should be O(new frames), not O(all frames).

    Writes K=4 frames, then *extends* the scan with another 4 and
    re-saves.  Verifies:

    * Final on-disk shape is (8, nq) — both batches landed.
    * The first 4 rows are bit-identical between the two checkpoints
      (i.e. the second save didn't rewrite them — append-only).
    * The frame_index dataset extends correctly.

    Without the P1 incremental refactor the second save still wrote
    correct *values* (full-array rewrite was idempotent), but it
    re-stacked all N frames every time.  This test pins down the
    append-only behaviour so future refactors don't regress.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
    })
    # Phase 1 — 4 frames + initial save.
    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(frames, scan_data=scan_data)
    path = tmp_path / "incremental.nxs"
    save_scan_to_nexus(scan, path, mode="w")

    # Snapshot row 0 of intensity from the first checkpoint — must
    # survive the second save unchanged.
    root_a = nx.nxload(str(path))
    intensity_a = np.asarray(root_a["entry/integrated_1d/intensity"])
    frame_index_a = np.asarray(root_a["entry/integrated_1d/frame_index"])
    assert intensity_a.shape == (N_FRAMES, N_Q)
    intensity_a_rows = intensity_a.copy()

    # Phase 2 — extend the scan by another N_FRAMES frames + re-save.
    for i in range(N_FRAMES, 2 * N_FRAMES):
        scan.frames.append(_DuckArch(idx=i, seed=100))
    save_scan_to_nexus(scan, path, mode="a")

    # Verify the extended file.
    root_b = nx.nxload(str(path))
    intensity_b = np.asarray(root_b["entry/integrated_1d/intensity"])
    frame_index_b = np.asarray(root_b["entry/integrated_1d/frame_index"])
    assert intensity_b.shape == (2 * N_FRAMES, N_Q)
    assert frame_index_b.shape == (2 * N_FRAMES,)
    # The pre-existing rows are byte-identical to the first checkpoint.
    np.testing.assert_array_equal(intensity_b[:N_FRAMES], intensity_a_rows)
    np.testing.assert_array_equal(frame_index_b[:N_FRAMES], frame_index_a)
    # The new rows have the new frames' frame indices.
    assert frame_index_b[N_FRAMES:].tolist() == list(range(N_FRAMES, 2 * N_FRAMES))

    # 2D side also extends correctly (parallel append).
    intensity_2d_b = np.asarray(root_b["entry/integrated_2d/intensity"])
    assert intensity_2d_b.shape == (2 * N_FRAMES, N_CHI, N_Q)


def test_append_with_shape_change_requires_full_reintegration(tmp_path):
    """O3: a follow-up save whose new frames have a different
    integration row shape than the on-disk dataset must NOT silently
    drop the previously-saved rows or rebuild from stale frames.  The append
    path must stop and require an explicit full reintegration.

    Pre-O3 ``_append_new_rows`` recreated the dataset with only
    the new tail when shape changed — so e.g. saving 4 frames at
    nq=32 then 4 more at nq=64 produced a 4-row file (the first
    4 lost).  Post-O3 the file ends at 8 rows of nq=64 (all
    frames explicitly supplied through replace mode).
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    import h5py

    nq1 = N_Q          # 32
    nq2 = N_Q * 2      # 64 — different!
    nchi1 = N_CHI      # 16
    nchi2 = N_CHI + 4  # 20 — different!

    # Phase 1: 4 frames at original shape.
    frames = [_DuckArch(idx=i, nq=nq1, nchi=nchi1) for i in range(N_FRAMES)]
    scan = _DuckSphere(
        frames,
        scan_data=pd.DataFrame({"tth": np.linspace(10, 14, N_FRAMES,
                                                   dtype=np.float32)}),
    )
    path = tmp_path / "shape_drift.nxs"
    save_scan_to_nexus(scan, path, mode="w")

    with h5py.File(path, "r") as f:
        assert f["entry/integrated_1d/intensity"].shape == (N_FRAMES, nq1)
        assert f["entry/integrated_2d/intensity"].shape == (N_FRAMES,
                                                            nchi1, nq1)

    # Phase 2: replace the in-memory frames with new ones at a
    # different shape — and *also* extend the series.  This is the
    # bug scenario: a user reintegrating with new numpoints mid-scan,
    # then continuing to acquire frames.
    new_frames = [
        _DuckArch(idx=i, nq=nq2, nchi=nchi2, seed=50)
        for i in range(2 * N_FRAMES)
    ]
    # Re-seat scan.frames to point at the reshaped frames.
    scan.frames = _DuckArches(new_frames)
    with pytest.raises(ValueError, match="Integration settings changed"):
        save_scan_to_nexus(scan, path, mode="a")

    # The failed append leaves the original stack intact. A caller can now
    # reintegrate every frame and save them together via replace mode.
    with h5py.File(path, "r") as f:
        assert f["entry/integrated_1d/intensity"].shape == (N_FRAMES, nq1)
        assert f["entry/integrated_2d/intensity"].shape == (N_FRAMES, nchi1, nq1)


def test_scan_data_index_aligns_with_gapped_frame_ids(tmp_path):
    """O1 / N2 regression: after writer → loader round-trip, the
    reloaded ``scan.scan_data`` DataFrame must be indexed by the
    actual frame IDs (``frame.idx``), not a default 0..N-1 range
    index.

    Live acquisition does ``scan_data.loc[frame.idx] = ser``.  If
    reload built a default RangeIndex, ``loc[frame.idx]`` after
    reload would silently misalign whenever the IDs were not
    ``range(N)`` — e.g. 1-based SPEC, gapped Eiger external links,
    or post-deletion holes.

    Test gaps the IDs to ``[10, 12, 17, 22]`` and asserts they
    appear as the reloaded ``scan_data`` index.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan
    from xrd_tools.core.geometry import DiffractometerGeometry

    gapped = [10, 12, 17, 22]
    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")

    frames = [_DuckArch(idx=i) for i in gapped]
    # Real LiveScan indexes scan_data by frame.idx (live:
    # ``scan_data.loc[frame.idx] = ser``; reload:
    # ``pd.DataFrame(motor_cols, index=frame_index)``), so the fixture
    # must too — the writer aligns positioners/geometry to the frame
    # set by *reindexing on that id index*.  (The pre-#18 writer used a
    # "reindex only when lengths differ" heuristic that silently fell
    # back to positional alignment; that tolerated a mis-indexed
    # scan_data but would mis-pair motors to frames after a delete /
    # reorder.  The shared ssrl primitive aligns by id instead.)
    scan_data = pd.DataFrame(
        {
            "tth": np.array([11.0, 12.5, 14.0, 15.5], dtype=np.float32),
            "th":  np.array([0.10, 0.15, 0.20, 0.25], dtype=np.float32),
        },
        index=gapped,
    )
    scan = _DuckSphere(frames, scan_data=scan_data, geometry=geom)

    path = tmp_path / "gapped.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    # Round-trip through the real LiveScan loader.
    reloaded = LiveScan(name="gapped", data_file=str(path))
    reloaded.load_from_h5(replace=True, mode="r")

    # The reloaded scan_data index *is* the gapped frame IDs.
    assert list(reloaded.scan_data.index) == gapped
    # And the columns survived with the right values.
    assert "tth" in reloaded.scan_data.columns
    assert "th" in reloaded.scan_data.columns
    np.testing.assert_allclose(
        reloaded.scan_data.loc[17, "tth"], 14.0, atol=1e-5,
    )
    # Most importantly: a ``.loc[frame.idx]`` lookup pattern that
    # the GUI uses produces the right row, not row-position 17.
    np.testing.assert_allclose(
        reloaded.scan_data.loc[22, "th"], 0.25, atol=1e-5,
    )


def test_incremental_save_with_no_new_frames_is_noop(tmp_path):
    """Re-saving with no new frames must not change anything on disk.

    Specifically: dataset shapes stay constant, contents stay
    byte-identical.  Guards the ``if not new_frames: return`` early
    exit in :func:`_write_integrated_1d` / :func:`_write_integrated_2d`.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(
        frames,
        scan_data=pd.DataFrame({"tth": np.zeros(N_FRAMES, dtype=np.float32)}),
    )
    path = tmp_path / "noop.nxs"
    save_scan_to_nexus(scan, path, mode="w")

    before = np.asarray(nx.nxload(str(path))["entry/integrated_1d/intensity"])
    save_scan_to_nexus(scan, path, mode="a")  # no new frames added
    after = np.asarray(nx.nxload(str(path))["entry/integrated_1d/intensity"])

    assert after.shape == before.shape == (N_FRAMES, N_Q)
    np.testing.assert_array_equal(after, before)


def test_positioners_align_to_frame_count_when_metadata_partial(tmp_path):
    """Batch frames with missing metadata leave ``scan_data`` shorter than
    the integrated frame set.  Positioners (and per-frame geometry) must
    still be written at the full integrated-frame length — NaN-padded for
    the metadata-less frames — so the per-frame dim agrees across groups
    and ``read_scan`` doesn't raise "conflicting sizes for dimension
    'frame'" (the Combi angle-dependence batch-mode bug).
    """
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.core.geometry import DiffractometerGeometry

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]  # 4 integrated frames
    # Only the first 2 frames carried metadata → scan_data has 2 rows.
    scan_data = pd.DataFrame(
        {"th": [0.0, 0.1], "tth": [10.0, 11.0]}, index=[0, 1],
    )
    scan = _DuckSphere(frames, scan_data=scan_data, geometry=geom)

    path = tmp_path / "partial_meta.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        e = f["entry"]
        n_int = e["integrated_1d/frame_index"].shape[0]
        th = np.asarray(e["sample/positioners/th/value"])
        assert n_int == N_FRAMES
        assert th.shape[0] == N_FRAMES        # aligned, not 2
        assert not np.isnan(th[:2]).any()     # real metadata preserved
        assert np.isnan(th[2:]).all()         # missing frames NaN-padded


# ---------------------------------------------------------------------------
# scan_data READ round-trip (the GUI batch->reload path, headless)
# ---------------------------------------------------------------------------

def test_scan_data_survives_save_then_load_from_h5(tmp_path):
    """The batch flow writes scan_data then reloads the .nxs for display via
    LiveScan.load_from_h5.  scan_data (motor metadata incl. the GI ``th``
    incidence motor) must come back populated — an empty scan_data leaves the
    metadata panel blank AND collapses the GI-2D geometry.  Reproduces the
    reported regression headlessly: write via the real writer, reload via the
    real loader, assert the table survives.
    """
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald.scan import LiveScan
    from xrd_tools.core.geometry import DiffractometerGeometry

    geom = DiffractometerGeometry.two_circle(tth="tth", th="th")
    scan_data = pd.DataFrame({
        "tth": np.linspace(10.0, 14.0, N_FRAMES, dtype=np.float32),
        "th": np.linspace(0.10, 0.40, N_FRAMES, dtype=np.float32),
        "i0": np.linspace(1e6, 1.2e6, N_FRAMES, dtype=np.float32),
    })
    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(frames, scan_data=scan_data, geometry=geom)

    path = tmp_path / "scan_data_rt.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    loaded = LiveScan(name="rt", data_file=str(path))
    loaded.load_from_h5()

    sd = loaded.scan_data
    assert sd is not None and len(sd.index) == N_FRAMES, (
        f"scan_data empty after reload (got {None if sd is None else len(sd.index)} rows)"
    )
    # The GI incidence motor must be recoverable for the geometry/panel.
    assert "th" in sd.columns, f"'th' missing from reloaded scan_data {list(sd.columns)}"
    np.testing.assert_allclose(
        np.asarray(sd["th"].values, dtype=float),
        scan_data["th"].values, rtol=1e-4)
    # A pure (non-geometry) motor must survive too.
    assert "i0" in sd.columns


def test_coerce_scan_info_keeps_non_numeric_columns():
    """N2: heterogeneous per-frame metadata survives ingestion.

    ``_coerce_scan_info`` keeps every field -- numbers (and numeric strings,
    since SPEC stores numbers as text) coerced to float, genuinely non-numeric
    values kept as strings -- instead of dropping the whole column, so the
    writer can persist it (numeric -> float32, non-numeric -> vlen string)."""
    from xdart.modules.ewald.scan import _coerce_scan_info
    info = {"th": 0.2, "i0": 1.0e6, "tth": 12.5,
            "keith_I": "0V", "sample": "Combi4", "date": "2026-03-27"}
    assert _coerce_scan_info(info) == {
        "th": 0.2, "i0": 1.0e6, "tth": 12.5,
        "keith_I": "0V", "sample": "Combi4", "date": "2026-03-27",
    }
    # numeric strings still coerce to float (motors / monitors stay numeric)
    assert _coerce_scan_info({"th": "0.30"}) == {"th": 0.30}
    # the assembled scan_data keeps the string column heterogeneous (no float64)
    sd = pd.DataFrame([_coerce_scan_info({"th": 0.2, "keith_I": "0V"})], index=[0])
    sd.loc[1] = pd.Series(_coerce_scan_info({"th": 0.3, "keith_I": "1V"}))
    assert list(sd["keith_I"]) == ["0V", "1V"]
    assert sd["th"].tolist() == [0.2, 0.3]


def test_heterogeneous_scan_data_roundtrips_to_nexus(tmp_path):
    """N2 gate: a mixed numeric+string scan_data table round-trips -- the
    string column persists as a vlen UTF-8 column (``ssrl_dtype='string'``)
    and numeric columns are unchanged."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.io.read import get_metadata

    sd = pd.DataFrame(
        {"i0": [100.0, 101.0, 102.0], "keith_I": ["0V", "1V", "2V"]},
        index=[0, 1, 2],
    )
    scan = _DuckSphere([_DuckArch(idx=i) for i in range(3)], scan_data=sd)
    path = tmp_path / "hetero_scan_data.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=True)

    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["entry/scan_data/i0"][()], [100.0, 101.0, 102.0])
        ki = f["entry/scan_data/keith_I"]
        assert ki.attrs["ssrl_dtype"] == "string"
        assert list(ki.asstr()[()]) == ["0V", "1V", "2V"]

    meta = get_metadata(path)
    assert list(np.asarray(meta["scan_data"]["keith_I"]).astype(str)) == ["0V", "1V", "2V"]
    np.testing.assert_allclose(
        np.asarray(meta["scan_data"]["i0"], dtype=float), [100.0, 101.0, 102.0],
    )


def test_gi_config_roundtrips_to_reduction_config(tmp_path):
    """N3: the GI output mode persists as a first-class
    /entry/reduction/config/gi_config field, recoverable on read without
    sniffing the q-unit string."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    scan = _DuckSphere(
        [_DuckArch(idx=i) for i in range(2)],
        scan_data=pd.DataFrame({"i0": [1.0, 2.0]}, index=[0, 1]),
        gi=True,
    )
    scan.gi_config = {"gi_mode_1d": "q_oop", "gi_mode_2d": "qip_qoop",
                      "incidence_motor": "th", "tilt_angle": 0.0,
                      "sample_orientation": 1}
    path = tmp_path / "gi_config.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=True)

    # read_scan's metadata recovers it as a parsed, first-class field.
    from xrd_tools.io.read import get_metadata
    gi = get_metadata(path)["reduction"]["config"]["gi_config"]
    assert gi["gi_mode_1d"] == "q_oop"
    assert gi["gi_mode_2d"] == "qip_qoop"

    from xdart.modules.ewald.scan import LiveScan
    loaded = LiveScan(name="gi", data_file=str(path))
    loaded.load_from_h5()
    assert loaded.gi_config["gi_mode_1d"] == "q_oop"
    assert loaded.gi_config["gi_mode_2d"] == "qip_qoop"


def test_gi_freeze_diagnostic_persisted_in_provenance(tmp_path):
    # Codex P2: the T0-4 first-chunk-freeze advisory must survive in the
    # output file's reduction provenance, not just as a transient GUI label.
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.io.nexus import read_scan_metadata

    scan = _DuckSphere([_DuckArch(idx=0)])
    scan.gi_freeze_diagnostic = (
        "GI: test reason — output grid will be frozen from the first frames")
    path = tmp_path / "gi_diag.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    ds = read_scan_metadata(path)
    config = (ds.attrs.get("reduction") or {}).get("config") or {}
    assert "frozen from the first frames" in config.get("gi_freeze_diagnostic", "")


def test_one_resultless_frame_does_not_block_later_2d_writes(tmp_path):
    """H1 (fresh-eyes review): a frame whose int_2d is missing (publication-
    dropped row lazy-reloaded as None, reload-only frame, ...) must be skipped
    PER FRAME — the old all-or-nothing check silently skipped the ENTIRE 2D
    write for that save and every save after it (1D complete, 2D truncated,
    no error).  The dropped label is remembered on the write cursor so later
    saves don't re-select (and re-lazy-load) it."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    scan = _DuckSphere([_DuckArch(idx=i) for i in range(3)])
    scan.frames[1].int_2d = None                       # the bad frame

    path = tmp_path / "h1.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        idx_2d = sorted(int(x) for x in f["entry/integrated_2d/frame_index"][()])
        idx_1d = sorted(int(x) for x in f["entry/integrated_1d/frame_index"][()])
    assert idx_2d == [0, 2], "valid frames' 2D rows must still be written"
    assert idx_1d == [0, 1, 2], "1D unaffected (filtered independently)"
    cursor = scan._nexus_write_cursor
    assert 1 in cursor.dropped.get("entry/integrated_2d", set())

    # Later frames + a second save: the dropped label is NOT re-selected,
    # new frames write normally.
    scan.frames.append(_DuckArch(idx=3))
    save_scan_to_nexus(scan, path, mode="a", finalize=False)
    with h5py.File(path, "r") as f:
        idx_2d = sorted(int(x) for x in f["entry/integrated_2d/frame_index"][()])
    assert idx_2d == [0, 2, 3]


def test_gi_freeze_diagnostic_persists_on_later_save(tmp_path):
    # P2 follow-up (codex review): on the REAL path the first save
    # (initialize_scan) writes provenance BEFORE the prepass stamps the
    # diagnostic, and the GUI never passes finalize=True -- the cursor-deduped
    # re-fire must persist it on a later periodic save.
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.io.nexus import read_scan_metadata

    scan = _DuckSphere([_DuckArch(idx=0)])
    path = tmp_path / "gi_diag_late.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)   # pre-stamp save

    scan.gi_freeze_diagnostic = "GI: output grid set from the first frames (test)"
    scan.frames.append(_DuckArch(idx=1))
    save_scan_to_nexus(scan, path, mode="a", finalize=False)   # periodic save

    ds = read_scan_metadata(path)
    config = (ds.attrs.get("reduction") or {}).get("config") or {}
    assert "set from the first frames" in config.get("gi_freeze_diagnostic", "")


# ---------------------------------------------------------------------------
# Pre-ship sweep regressions: cursor.dropped vs the replace recovery path
# ---------------------------------------------------------------------------

def _all_dummy_2d(frame):
    """Make a frame's 2D result publication-invalid (all-dummy cake)."""
    frame.int_2d.intensity = np.full_like(frame.int_2d.intensity, -1.0)


def test_replace_recovers_group_that_was_never_created(tmp_path):
    """If EVERY row of integrated_2d was publication-rejected during the run
    (group never created on disk; all labels in cursor.dropped), a later
    reintegrate-all replace save with now-valid results must WRITE the group.
    The stale dropped bookkeeping previously excluded every recomputed frame
    on the append fallback (`group_path in h5f` is False for a group that
    never existed) -- the designed recovery path silently wrote nothing."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "recover.nxs"

    for fr in frames:
        _all_dummy_2d(fr)
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    with h5py.File(path, "r") as f:
        assert "entry/integrated_2d" not in f          # all rows rejected
    assert scan._nexus_write_cursor.dropped            # drops recorded

    # Reintegration fixed the cakes; replace-all must write them.
    for i, fr in enumerate(frames):
        rng = np.random.default_rng(100 + i)
        fr.int_2d.intensity = rng.random(fr.int_2d.intensity.shape).astype(
            np.float32)
    save_scan_to_nexus(scan, path, mode="a", finalize=False,
                       replace_frame_indices=[0, 1, 2])
    with h5py.File(path, "r") as f:
        assert "entry/integrated_2d" in f
        assert f["entry/integrated_2d/intensity"].shape[0] == 3


def test_replace_with_shape_change_survives_one_publication_drop(tmp_path):
    """Replace mode + changed row shape + ONE publication-rejected frame must
    not abort the whole-scan save: the rejected frame's stale on-disk row is
    dropped first, then validation sees a covered batch.  (Previously
    _require_batch_covers_existing raised over the uncovered stale row and
    NOTHING was persisted.)"""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus

    frames = [_DuckArch(idx=i) for i in range(3)]
    scan = _DuckSphere(frames)
    path = tmp_path / "shape_change.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    # Reintegrate everything at a different npt; frame 1 comes out invalid.
    new_nq = N_Q + 7
    for i in range(3):
        fresh = _DuckArch(idx=i, nq=new_nq, seed=50)
        frames[i].int_1d = fresh.int_1d
        frames[i].int_2d = fresh.int_2d
    scan.bai_1d_args["numpoints"] = new_nq
    scan.bai_2d_args["npt_rad"] = new_nq
    _all_dummy_2d(frames[1])

    save_scan_to_nexus(scan, path, mode="a", finalize=False,
                       replace_frame_indices=[0, 1, 2])
    with h5py.File(path, "r") as f:
        # 1D (all valid): full new-shape stack
        assert f["entry/integrated_1d/intensity"].shape == (3, new_nq)
        # 2D: frame 1 dropped per frame, frames 0+2 written at the new
        # shape (rows stored (nchi, nq) -- nq is the trailing axis)
        i2d = f["entry/integrated_2d"]
        assert i2d["intensity"].shape == (2, N_CHI, new_nq)
        assert sorted(int(x) for x in i2d["frame_index"][()]) == [0, 2]


# ---------------------------------------------------------------------------
# Calibration round-trip — the detector identity must survive write→reload so a
# reloaded scan can be re-integrated with its OWN geometry (2026-06-18 fix for
# the `_pixel1 is None` crash: the .nxs used to persist only the 6 geometry
# scalars, never the detector name/pixel sizes, so a reload had no usable
# integrator).
# ---------------------------------------------------------------------------

def test_calibration_round_trips_through_nxs(tmp_path):
    """Writer persists the detector NAME + pixel sizes; reader rebuilds a
    pixel-bearing integrator + PONI from the file ALONE (no GUI PONI)."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald import LiveScan
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.calibration import poni_to_integrator

    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    real_poni = PONI(dist=0.15, poni1=0.02, poni2=0.02,
                     rot1=0.0, rot2=0.0, rot3=0.0,
                     wavelength=1.0e-10, detector="Pilatus100k")
    for fr in frames:
        fr.poni = real_poni            # the representative-poni source
    scan = _DuckSphere(frames)
    scan._cached_integrator = poni_to_integrator(real_poni)   # pixel-size source

    path = tmp_path / "calib_roundtrip.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    # WRITER: detector identity persisted under instrument/detector.
    with h5py.File(path, "r") as f:
        det = f["entry/instrument/detector"]
        name = det["detector_name"][()]
        assert (name.decode() if isinstance(name, bytes) else str(name)) \
            == "Pilatus100k"
        assert float(det["x_pixel_size"][()]) > 0.0
        assert float(det["y_pixel_size"][()]) > 0.0

    # READER: a reloaded scan carries a pixel-bearing integrator + PONI.
    reloaded = LiveScan(data_file=str(path))
    reloaded.load_from_h5()
    assert reloaded._cached_integrator is not None
    assert reloaded._cached_integrator.detector is not None
    assert reloaded._cached_integrator.detector.pixel1 is not None
    assert reloaded._cached_poni is not None
    assert reloaded._cached_poni.detector == "Pilatus100k"


def test_old_nxs_without_detector_identity_leaves_cache_none(tmp_path):
    """Backward-compat: a .nxs written WITHOUT a detector name/pixel sizes (the
    pre-fix shape — only the 6 geometry scalars) reloads with _cached_integrator
    left None, so the Reintegrate guard surfaces a clear 're-process' message
    instead of building a pixel-less integrator that crashes pyFAI."""
    import h5py
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xdart.modules.ewald import LiveScan

    # _DuckPONI has no `detector`; no _cached_integrator → no pixel sizes.
    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    scan = _DuckSphere(frames)
    path = tmp_path / "old_style.nxs"
    save_scan_to_nexus(scan, path, mode="w", finalize=False)

    with h5py.File(path, "r") as f:
        det = f["entry/instrument/detector"]
        assert "detector_name" not in det        # nothing to round-trip
        assert "x_pixel_size" not in det

    reloaded = LiveScan(data_file=str(path))
    reloaded.load_from_h5()
    assert reloaded._cached_integrator is None    # guard will prompt re-process


def _written_calibrated_scan(tmp_path, name="cal.nxs"):
    """Write a 4-frame scan carrying a named-detector calibration; return path."""
    from xdart.modules.ewald.nexus_writer import save_scan_to_nexus
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.calibration import poni_to_integrator
    frames = [_DuckArch(idx=i) for i in range(N_FRAMES)]
    real_poni = PONI(dist=0.15, poni1=0.02, poni2=0.02, rot1=0.0, rot2=0.0,
                     rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")
    for fr in frames:
        fr.poni = real_poni
    scan = _DuckSphere(frames)
    scan._cached_integrator = poni_to_integrator(real_poni)
    path = tmp_path / name
    save_scan_to_nexus(scan, path, mode="w", finalize=False)
    return str(path)


def test_ensure_calibration_loaded_self_heals_without_full_load(tmp_path):
    """Reintegrate self-heal: a scan that never got a full reload (e.g. only a
    live data_only refresh reached it) has _cached_integrator=None;
    ensure_calibration_loaded re-reads it from the scan's OWN .nxs on demand."""
    from xdart.modules.ewald import LiveScan
    path = _written_calibrated_scan(tmp_path, "selfheal.nxs")

    s = LiveScan(data_file=path)               # constructed, never load_from_h5'd
    assert getattr(s, "_cached_integrator", None) is None
    assert s.ensure_calibration_loaded() is True
    assert s._cached_integrator.detector.pixel1 is not None
    assert s._cached_poni.detector == "Pilatus100k"
    assert s.ensure_calibration_loaded() is True   # idempotent


def test_data_only_refresh_restores_when_empty_but_never_clobbers(tmp_path):
    """A live data_only refresh restores calibration ONLY when the cache is
    empty; it must never overwrite the wrangler's live integrator mid-run."""
    from xdart.modules.ewald import LiveScan
    path = _written_calibrated_scan(tmp_path, "dataonly.nxs")

    # Empty cache → data_only refresh fills it from disk.
    s = LiveScan(data_file=path)
    s.load_from_h5(replace=False, data_only=True)
    assert s._cached_integrator is not None
    assert s._cached_integrator.detector.pixel1 is not None

    # Live integrator already cached → data_only refresh leaves it untouched.
    s2 = LiveScan(data_file=path)
    sentinel = object()
    s2._cached_integrator = sentinel
    s2.load_from_h5(replace=False, data_only=True)
    assert s2._cached_integrator is sentinel
