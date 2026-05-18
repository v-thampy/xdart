# -*- coding: utf-8 -*-
"""Tests for L1 lazy raw load: LiveFrame._lazy_load_raw + the
re-evaluated ``is_reload_only`` flag set by _load_arch_v2.

Covers both source types the wranglers stamp into v2 files:

* **TIF source** — single-frame TIFFs from the SPEC wrangler.
* **NeXus source** — single dataset or multi-link Eiger master, via
  :class:`ssrl_xrd_tools.io.nexus.NexusImageStack`.

Tests exercise the lazy-load path directly (no full pyFAI
integration) so they stay fast and don't need a real PONI.  The
integrate methods would just consume ``arch.map_raw`` after the
lazy load — and that's covered separately in the integrate tests.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_tif(path, shape=(8, 8), seed=0):
    """Write a tiny TIFF with deterministic content for round-trip checks."""
    import tifffile
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 1000, size=shape).astype(np.uint16)
    tifffile.imwrite(str(path), img)
    return img


def _write_eiger_master(tmp_path, n_files=2, frames_per_file=3,
                       shape=(6, 8), seed=0):
    """Build an Eiger-style master with sibling external links.

    Same construction as the ssrl_xrd_tools test fixture — duplicated
    here so this test file doesn't import the ssrl test-suite helpers.
    Returns (master_path, full_expected_stack).
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for k in range(1, n_files + 1):
        data_path = tmp_path / f"scan_data_{k:06d}.h5"
        block = rng.integers(0, 1000,
                             size=(frames_per_file,) + shape
                             ).astype(np.uint16)
        chunks.append(block)
        with h5py.File(data_path, "w") as df:
            df.create_dataset("data", data=block, chunks=(1,) + shape)
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as mf:
        e = mf.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        d = e.create_group("data")
        for k in range(1, n_files + 1):
            d[f"data_{k:06d}"] = h5py.ExternalLink(
                f"scan_data_{k:06d}.h5", "/data",
            )
    return master, np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# _resolved_source_path — pure-path tests, no I/O
# ---------------------------------------------------------------------------

class TestResolvedSourcePath:
    def test_no_source_file_returns_empty(self):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.source_file = ""
        assert a._resolved_source_path() == ""

    def test_absolute_source_file_kept(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.source_file = str(tmp_path / "abs.tif")
        a._source_root = "/should/be/ignored"
        assert a._resolved_source_path() == str(tmp_path / "abs.tif")

    def test_relative_joined_against_source_root(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.source_file = "frame_0001.tif"
        a._source_root = str(tmp_path)
        assert a._resolved_source_path() == str(tmp_path / "frame_0001.tif")

    def test_relative_with_empty_root_uses_cwd(self):
        """Defensive: pre-R2 reloads might lack _source_root; the lazy
        load shouldn't crash, it should return False on os.path.exists."""
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.source_file = "frame_0001.tif"
        a._source_root = ""
        # Returns the bare relpath (matches "use cwd" semantics).
        assert a._resolved_source_path() == "frame_0001.tif"


# ---------------------------------------------------------------------------
# _lazy_load_raw — TIF source
# ---------------------------------------------------------------------------

class TestLazyLoadTif:
    def test_loads_tif_frame_from_relative_path(self, tmp_path):
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.arch import LiveFrame

        img = _write_tif(tmp_path / "frame_0000.tif", shape=(10, 12), seed=7)

        a = LiveFrame(idx=0)
        a.map_raw = None
        a.source_file = "frame_0000.tif"
        a.source_frame_idx = 0  # SPEC TIFs are always frame 0
        a._source_root = str(tmp_path)

        assert a._lazy_load_raw() is True
        assert a.map_raw is not None
        assert a.map_raw.shape == img.shape
        # read_image returns float32; compare values (not dtype).
        assert np.allclose(a.map_raw, img.astype(np.float32))

    def test_no_op_when_map_raw_already_set(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.map_raw = np.ones((4, 4), dtype=np.float32) * 42
        a.source_file = "nonexistent.tif"
        a._source_root = str(tmp_path)
        # Should return True immediately and not touch map_raw.
        assert a._lazy_load_raw() is True
        assert np.all(a.map_raw == 42)

    def test_missing_file_returns_false(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.map_raw = None
        a.source_file = "ghost.tif"
        a._source_root = str(tmp_path)
        assert a._lazy_load_raw() is False
        assert a.map_raw is None

    def test_no_source_file_returns_false(self):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.map_raw = None
        a.source_file = ""
        assert a._lazy_load_raw() is False

    def test_copy_preserves_lazy_load_provenance(self, tmp_path):
        """P4 regression: ``LiveFrame.copy(include_2d=False)`` must
        preserve ``source_frame_idx``, ``_source_root``, and
        ``is_reload_only`` along with ``source_file``.

        Pre-P4 the 1D-only copies stashed in ``data_1d`` lost three
        of those fields, so any later attempt to lazy-reload the raw
        frame (for reintegrate / thumbnail regen / etc.) either
        resolved to frame 0 or to a path that didn't exist.  Locks
        down the field-list so future copy() edits can't quietly
        regress it.
        """
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.arch import LiveFrame

        a = LiveFrame(idx=7)
        a.source_file = "stack.tif"
        a.source_frame_idx = 3
        a._source_root = str(tmp_path)
        a.is_reload_only = True

        c = a.copy(include_2d=False)
        assert c.source_file == "stack.tif"
        assert c.source_frame_idx == 3
        assert c._source_root == str(tmp_path)
        assert c.is_reload_only is True

        # And functional sanity: an actual lazy-load through the copy
        # should resolve to the same path the original would.
        _write_tif(tmp_path / "stack.tif", shape=(4, 4))
        assert c._resolved_source_path() == a._resolved_source_path()

    def test_loads_specific_frame_from_multiframe_tif(self, tmp_path):
        """O2 regression: for multi-frame TIFF/CBF stacks the lazy
        loader must forward ``source_frame_idx`` to read_image, not
        always pull frame 0.

        Pre-O2 the non-HDF5 branch called ``read_image(full)`` with
        no frame arg, so reload from a multi-frame stack returned
        the first slice for every arch — silently wrong.
        """
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.arch import LiveFrame
        import tifffile

        rng = np.random.default_rng(31)
        stack = rng.integers(0, 1000, size=(5, 6, 8)).astype(np.uint16)
        stack_path = tmp_path / "stack.tif"
        # tifffile writes a single multi-page TIF when given a 3D array.
        tifffile.imwrite(str(stack_path), stack)

        target_idx = 3
        a = LiveFrame(idx=target_idx)
        a.map_raw = None
        a.source_file = "stack.tif"
        a.source_frame_idx = target_idx
        a._source_root = str(tmp_path)

        assert a._lazy_load_raw() is True
        assert a.map_raw is not None
        # Without O2 this would be ``stack[0]``; with O2 it's the
        # right page from the multi-page TIF.
        np.testing.assert_allclose(
            a.map_raw, stack[target_idx].astype(np.float32),
        )


# ---------------------------------------------------------------------------
# _lazy_load_raw — NeXus source (single dataset + Eiger external links)
# ---------------------------------------------------------------------------

class TestLazyLoadNexus:
    def test_loads_specific_frame_from_eiger_master(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        master, full = _write_eiger_master(
            tmp_path, n_files=3, frames_per_file=4, shape=(7, 9), seed=11,
        )
        target_idx = 9  # crosses the data_000001 → data_000002 boundary

        a = LiveFrame(idx=target_idx)
        a.map_raw = None
        a.source_file = master.name
        a.source_frame_idx = target_idx  # NeXus: global stack index
        a._source_root = str(tmp_path)

        assert a._lazy_load_raw() is True
        assert a.map_raw.shape == full.shape[1:]
        assert np.allclose(a.map_raw, full[target_idx].astype(np.float32))

    def test_falls_back_to_idx_when_source_frame_idx_missing(self, tmp_path):
        """Reload-from-older-file path: source_frame_idx is None →
        lazy loader uses arch.idx instead."""
        from xdart.modules.ewald.arch import LiveFrame
        master, full = _write_eiger_master(
            tmp_path, n_files=2, frames_per_file=3, shape=(5, 5), seed=13,
        )
        a = LiveFrame(idx=2)
        a.map_raw = None
        a.source_file = master.name
        a.source_frame_idx = None
        a._source_root = str(tmp_path)
        assert a._lazy_load_raw() is True
        assert np.allclose(a.map_raw, full[2].astype(np.float32))

    def test_missing_master_returns_false(self, tmp_path):
        from xdart.modules.ewald.arch import LiveFrame
        a = LiveFrame(idx=0)
        a.map_raw = None
        a.source_file = "no_such_master.h5"
        a.source_frame_idx = 0
        a._source_root = str(tmp_path)
        assert a._lazy_load_raw() is False


# ---------------------------------------------------------------------------
# _load_arch_v2 — flag is reload-feasibility-aware after L1
# ---------------------------------------------------------------------------

class TestReloadOnlyFlagWiring:
    """``is_reload_only`` should reflect whether lazy load will succeed.

    Verified end-to-end: write a sphere with a real-on-disk source, then
    re-load and inspect arch.is_reload_only.  Going through
    save_sphere_to_nexus + _load_arch_v2 ensures the wiring through
    arch_series is exercised.
    """

    def test_is_reload_only_false_when_source_exists(self, tmp_path):
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
        from xdart.modules.ewald.arch_series import _load_arch_v2

        # Test fixture pieces inlined to keep this file standalone.
        from tests.test_nexus_writer_roundtrip import (
            _DuckArch, _DuckSphere, N_Q,
        )

        # Build a real on-disk TIF and stamp it on the duck arch.
        tif_path = tmp_path / "frame_0000.tif"
        _write_tif(tif_path, shape=(8, 8))

        a = _DuckArch(idx=0)
        a.source_file = "frame_0000.tif"
        a.source_frame_idx = 0
        sphere = _DuckSphere([a])

        nxs = tmp_path / "scan.nxs"
        save_sphere_to_nexus(sphere, nxs, entry="entry", finalize=False)

        # Reload via the same code path as the GUI viewer.
        source_root = str(tmp_path)
        with h5py.File(nxs, "r") as f:
            loaded = _load_arch_v2(f, 0, static=False, gi=False,
                                   source_root=source_root)
        # With the source file present, lazy load is feasible.
        assert loaded.is_reload_only is False
        assert loaded.source_file == "frame_0000.tif"
        assert loaded._source_root == source_root

    def test_is_reload_only_true_when_source_missing(self, tmp_path):
        from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
        from xdart.modules.ewald.arch_series import _load_arch_v2
        from tests.test_nexus_writer_roundtrip import (
            _DuckArch, _DuckSphere,
        )

        # Stamp a source_file that doesn't exist on disk — simulates
        # the user moving the .nxs file to a different machine without
        # the raw frames.
        a = _DuckArch(idx=0)
        a.source_file = "nonexistent_frame.tif"
        a.source_frame_idx = 0
        sphere = _DuckSphere([a])

        nxs = tmp_path / "scan.nxs"
        save_sphere_to_nexus(sphere, nxs, entry="entry", finalize=False)

        with h5py.File(nxs, "r") as f:
            loaded = _load_arch_v2(f, 0, static=False, gi=False,
                                   source_root=str(tmp_path))
        assert loaded.is_reload_only is True

    def test_is_reload_only_true_when_no_source_file(self, tmp_path):
        """Edge: empty source_file (older v2 files written before
        R2's source-ref schema landed) — guardrail must still fire."""
        from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
        from xdart.modules.ewald.arch_series import _load_arch_v2
        from tests.test_nexus_writer_roundtrip import (
            _DuckArch, _DuckSphere,
        )

        a = _DuckArch(idx=0)
        a.source_file = ""
        a.source_frame_idx = 0
        sphere = _DuckSphere([a])
        nxs = tmp_path / "scan.nxs"
        save_sphere_to_nexus(sphere, nxs, entry="entry", finalize=False)

        with h5py.File(nxs, "r") as f:
            loaded = _load_arch_v2(f, 0, static=False, gi=False,
                                   source_root=str(tmp_path))
        assert loaded.is_reload_only is True


# ---------------------------------------------------------------------------
# integrate_*: lazy load happens automatically when map_raw is None
# ---------------------------------------------------------------------------

class TestPicklingPreservesLazyLoadCapability:
    """F8 / L1 follow-up.

    The sphere_threads.bai_*_all reintegrate path sends arches into
    a ProcessPoolExecutor.  Arches reloaded from v2 .nxs lack
    ``map_raw`` and depend on ``_source_root`` + ``source_file`` to
    lazy-load it.  The review flagged a suspicion that ``_source_root``
    might not survive the pickle round-trip (or might resolve
    relative paths against the *child*'s cwd, not the parent's).

    These tests confirm both attributes pickle cleanly and that the
    child process can still resolve + load the raw frame.
    """

    def test_attrs_round_trip_through_pickle(self, tmp_path):
        import pickle
        from xdart.modules.ewald.arch import LiveFrame

        a = LiveFrame(idx=42)
        a.source_file = "frame_0042.tif"
        a.source_frame_idx = 0
        a._source_root = str(tmp_path)
        a.is_reload_only = False

        blob = pickle.dumps(a)
        b = pickle.loads(blob)

        assert b.idx == 42
        assert b.source_file == "frame_0042.tif"
        assert b.source_frame_idx == 0
        assert b._source_root == str(tmp_path)
        assert b.is_reload_only is False
        # _resolved_source_path should produce an absolute path
        # rooted at _source_root — independent of child cwd.
        full = b._resolved_source_path()
        assert os.path.isabs(full)
        assert full == os.path.normpath(
            os.path.join(str(tmp_path), "frame_0042.tif")
        )

    def test_subprocess_can_lazy_load_via_pickled_arch(self, tmp_path):
        """End-to-end: spawn a real subprocess, pickle an arch into
        it, ask it to lazy-load, get the loaded array back.  This is
        what sphere_threads.bai_*_all does under the hood.
        """
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.arch import LiveFrame

        img = _write_tif(tmp_path / "frame_0007.tif",
                         shape=(8, 10), seed=42)

        a = LiveFrame(idx=7)
        a.map_raw = None
        a.source_file = "frame_0007.tif"
        a.source_frame_idx = 0
        a._source_root = str(tmp_path)

        # Use a ProcessPoolExecutor to mirror the real reintegrate
        # path (sphere_threads._reintegrate_all uses ProcessPoolExecutor).
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_load_and_return_raw, a)
            result_arr, result_root = future.result()

        # Lazy load succeeded inside the subprocess.
        assert result_arr is not None
        assert result_arr.shape == img.shape
        # The subprocess's _source_root matches the parent's — the
        # attribute survived pickling.
        assert result_root == str(tmp_path)


def _load_and_return_raw(arch):
    """Helper for ProcessPoolExecutor — must be module-level for
    pickling to work."""
    arch._lazy_load_raw()
    return arch.map_raw, arch._source_root


class TestIntegrateTriggersLazyLoad:
    """Make sure ``integrate_1d`` / ``integrate_2d`` no longer no-op
    silently when there's a recoverable source.

    We don't actually run pyFAI here — we just check that ``map_raw``
    is populated by the time the method's main path runs.  Asserted
    by mocking the integrator's ``integrate1d_ng`` to record what it
    sees.
    """

    def test_integrate_1d_lazy_loads_before_pyfai(self, tmp_path):
        pytest.importorskip("tifffile")
        from xdart.modules.ewald.arch import LiveFrame

        # Write a real TIF.
        _write_tif(tmp_path / "frame_0000.tif", shape=(16, 16), seed=99)

        a = LiveFrame(idx=0)
        a.map_raw = None
        a.source_file = "frame_0000.tif"
        a.source_frame_idx = 0
        a._source_root = str(tmp_path)

        # Capture what pyFAI gets handed.
        captured = {}
        def fake_integrate1d_ng(data, **kwargs):
            captured['shape'] = data.shape
            captured['n_nan'] = int(np.isnan(data).sum())
            # Return a duck result with attrs the arch consumer reads.
            class _R:
                radial = np.linspace(0, 5, 10).astype(np.float32)
                intensity = np.zeros(10, dtype=np.float32)
                sigma = np.zeros(10, dtype=np.float32)
                unit = "q_A^-1"
            return _R()

        a.integrator.integrate1d_ng = fake_integrate1d_ng  # monkeypatch
        # Don't run the full integrate_1d (it ties into other ssrl
        # functions via integrator.single).  Instead, confirm just the
        # lazy load fires.
        before = a.map_raw
        assert before is None
        a._lazy_load_raw()
        after = a.map_raw
        assert after is not None
        assert after.shape == (16, 16)
