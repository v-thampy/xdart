"""Tests for xrd_tools.io.nexus — reader and writer."""

from __future__ import annotations

import numpy as np
import pytest
import h5py

import xrd_tools.io.nexus as _nexus_io
from xrd_tools.io.nexus import (
    NexusImageStack,
    _comp_kwargs,
    _require_uniform_axes_1d,
    _require_uniform_axes_2d,
    find_nexus_image_dataset,
    list_entries,
    open_nexus_image_stack,
    open_nexus_writer,
    read_nexus,
    read_scan_metadata,
    resolve_stack_compression,
    write_integrated_stack,
    write_nexus,
    write_nexus_frame,
)
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.metadata import ScanMetadata


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nexus_file(tmp_path):
    """Minimal NeXus file with instrument, sample, data groups."""
    p = tmp_path / "scan_001.h5"
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        inst = entry.create_group("instrument")
        mono = inst.create_group("monochromator")
        mono.create_dataset("energy", data=12.0)
        mono.create_dataset("wavelength", data=1.033)

        det = inst.create_group("detector")
        det.create_dataset("data", data=np.random.default_rng(0).random((5, 20, 30)))

        sample = entry.create_group("sample")
        sample.create_dataset("name", data="test_sample")
        sample.create_dataset("ub_matrix", data=np.eye(3).flatten())

        data = entry.create_group("data")
        data.create_dataset("th",      data=np.linspace(10, 20, 5))
        data.create_dataset("tth",     data=np.linspace(20, 40, 5))
        data.create_dataset("i0",      data=np.ones(5) * 1e5)
        data.create_dataset("i1",      data=np.ones(5) * 500)
        data.create_dataset("seconds", data=np.ones(5) * 1.0)
    return p


@pytest.fixture
def sample_metadata():
    return ScanMetadata(
        scan_id="scan_042",
        energy=12.0,
        wavelength=1.033,
        angles={"th": np.linspace(0, 5, 10), "phi": np.zeros(10)},
        counters={"i0": np.ones(10) * 1e5, "i1": np.ones(10) * 500},
        ub_matrix=np.eye(3),
        sample_name="my_film",
        source="test",
    )


@pytest.fixture
def result_1d():
    q = np.linspace(0.1, 5.0, 200)
    intensity = np.exp(-((q - 2.5) ** 2) / 0.1)
    sigma = np.sqrt(np.abs(intensity) + 1) * 0.01
    return IntegrationResult1D(radial=q, intensity=intensity, sigma=sigma, unit="q_A^-1")


@pytest.fixture
def result_2d():
    q = np.linspace(0.1, 5.0, 100)
    chi = np.linspace(-180, 180, 80)
    intensity = np.random.default_rng(1).random((100, 80))
    return IntegrationResult2D(radial=q, azimuthal=chi, intensity=intensity, unit="q_A^-1")


def _minimal_nexus(tmp_path, name="scan_min.h5", **overrides):
    """Write a bare-minimum NeXus file; caller passes datasets to omit."""
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        inst = entry.create_group("instrument")
        mono = inst.create_group("monochromator")
        if "energy" not in overrides.get("omit", set()):
            mono.create_dataset("energy", data=12.0)
        if "wavelength" not in overrides.get("omit", set()):
            mono.create_dataset("wavelength", data=1.033)
        sample = entry.create_group("sample")
        if "ub_matrix" not in overrides.get("omit", set()):
            sample.create_dataset("ub_matrix", data=np.eye(3))
        if "name" not in overrides.get("omit", set()):
            sample.create_dataset("name", data="sample")
        entry.create_group("data")
    return p


# ---------------------------------------------------------------------------
# TestReadNexus
# ---------------------------------------------------------------------------

class TestReadNexus:
    def test_returns_scan_metadata(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert isinstance(meta, ScanMetadata)

    def test_energy(self, nexus_file):
        meta = read_nexus(nexus_file)
        np.testing.assert_allclose(meta.energy, 12.0)

    def test_wavelength(self, nexus_file):
        meta = read_nexus(nexus_file)
        np.testing.assert_allclose(meta.wavelength, 1.033, rtol=1e-4)

    def test_ub_matrix(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert meta.ub_matrix is not None
        assert meta.ub_matrix.shape == (3, 3)
        np.testing.assert_allclose(meta.ub_matrix, np.eye(3))

    def test_sample_name(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert meta.sample_name == "test_sample"

    def test_source(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert meta.source == "nexus"

    def test_scan_id(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert meta.scan_id == "scan_001"

    def test_h5_path(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert meta.h5_path == nexus_file

    def test_angles(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert "th" in meta.angles
        assert "tth" in meta.angles
        assert meta.angles["th"].shape == (5,)
        assert meta.angles["tth"].shape == (5,)
        np.testing.assert_allclose(meta.angles["th"], np.linspace(10, 20, 5))

    def test_counters(self, nexus_file):
        meta = read_nexus(nexus_file)
        assert "i0" in meta.counters
        assert "i1" in meta.counters
        assert "seconds" in meta.counters
        assert "i0" not in meta.angles

    def test_custom_motor_names(self, nexus_file):
        meta = read_nexus(nexus_file, motor_names=["th"], counter_names=["i0"])
        assert set(meta.angles.keys()) == {"th"}
        assert set(meta.counters.keys()) == {"i0"}

    def test_missing_entry(self, nexus_file):
        with pytest.raises(KeyError):
            read_nexus(nexus_file, entry="nonexistent")

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_nexus(tmp_path / "ghost.h5")

    def test_harvests_scan_data_and_positioners(self, tmp_path):
        """F6: motors live under entry/scan_data (the SPEC-style all-motors
        per-point table) and entry/sample/positioners/<motor>/value (the scanned
        NXpositioner), not just entry/data.  read_nexus harvests all three so the
        GI incidence motor (possibly a non-scanned positioner) reaches the
        metadata.  Non-positioner datasets under positioners (e.g. frame_index)
        are NOT treated as motors."""
        p = tmp_path / "scan_nexusstyle.h5"
        with h5py.File(p, "w") as f:
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            sd = entry.create_group("scan_data")          # no entry/data here
            sd.create_dataset("samz", data=np.linspace(4.0, 5.0, 5))
            sd.create_dataset("phi", data=np.zeros(5))
            sd.create_dataset("i0", data=np.ones(5) * 1e5)   # a counter
            # xdart's own writer emits frame_index into scan_data -- it's a
            # bookkeeping index, NOT a motor, and must be skipped.
            sd.create_dataset("frame_index", data=np.arange(5))
            # SPEC-style tables carry string timestamp/label columns: a
            # fixed-length-string (|S) column AND a variable-length (object)
            # string column must be SKIPPED, not crash the cast (h5py raises
            # OSError casting a string dataset to float).
            sd.create_dataset("timestamp",
                              data=np.array([b"t0", b"t1", b"t2", b"t3", b"t4"]))
            sd.create_dataset("label", data=["a", "b", "c", "d", "e"],
                              dtype=h5py.string_dtype())
            sample = entry.create_group("sample")
            positioners = sample.create_group("positioners")
            th = positioners.create_group("th")              # NXpositioner
            th.attrs["NX_class"] = "NXpositioner"
            th.create_dataset("value", data=np.linspace(2.0, 2.4, 5))
            positioners.create_dataset("frame_index", data=np.arange(5))

        meta = read_nexus(p)                      # must NOT raise on |S columns
        assert "th" in meta.angles                # scanned positioner
        assert "samz" in meta.angles              # scan_data motor
        assert "phi" in meta.angles
        assert "i0" in meta.counters              # scan_data counter
        # frame_index (bookkeeping index our own writer emits into scan_data AND
        # under positioners) is never a motor/counter.
        assert "frame_index" not in meta.angles
        assert "frame_index" not in meta.counters
        # string columns skipped entirely (the OSError-on-cast regression)
        for s in ("timestamp", "label"):
            assert s not in meta.angles and s not in meta.counters
        np.testing.assert_allclose(
            meta.angles["th"], np.linspace(2.0, 2.4, 5))


# ---------------------------------------------------------------------------
# TestFindNexusImageDataset
# ---------------------------------------------------------------------------

class TestFindNexusImageDataset:
    def test_finds_detector_data(self, nexus_file):
        path = find_nexus_image_dataset(nexus_file)
        assert path is not None
        assert "detector" in path
        assert "data" in path

    def test_returns_none_no_images(self, tmp_path):
        p = tmp_path / "no_images.h5"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"
            e.create_dataset("scalar", data=1.0)
        assert find_nexus_image_dataset(p) is None

    def test_fallback_data_data(self, tmp_path):
        p = tmp_path / "dd.h5"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.create_group("data").create_dataset("data", data=np.zeros((3, 8, 8)))
        assert find_nexus_image_dataset(p) == "/entry/data/data"

    def test_fallback_instrument_subgroup(self, tmp_path):
        p = tmp_path / "sg.h5"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.create_group("instrument/eiger4m").create_dataset(
                "data", data=np.zeros((4, 8, 8))
            )
        assert find_nexus_image_dataset(p) == "/entry/instrument/eiger4m/data"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_nexus_image_dataset(tmp_path / "ghost.h5")


# ---------------------------------------------------------------------------
# TestNexusImageStack — Eiger external-link iterator
# ---------------------------------------------------------------------------


def _make_eiger_master(
    tmp_path,
    n_files: int,
    frames_per_file: int,
    img_shape: tuple[int, int] = (12, 16),
    dtype=np.uint16,
    master_name: str = "master.h5",
) -> tuple:
    """Create an Eiger-style master file with external links to data files.

    Each ``_data_NNNNNN.h5`` carries ``/data`` of shape
    ``(frames_per_file, H, W)``; the master file links them as
    ``/entry/data/data_NNNNNN``.

    Returns ``(master_path, expected_full_stack)`` where the stack is
    the deterministic per-frame content used for byte-equality checks.
    """
    rng = np.random.default_rng(42)
    chunks = []
    data_paths = []
    for k in range(1, n_files + 1):
        # Vary per-frame content so concatenation order matters.
        block = (rng.integers(0, 1000, size=(frames_per_file,) + img_shape)
                 .astype(dtype))
        chunks.append(block)
        data_path = tmp_path / f"scan_data_{k:06d}.h5"
        with h5py.File(data_path, "w") as df:
            df.create_dataset("data", data=block, chunks=(1,) + img_shape)
        data_paths.append(data_path)

    master_path = tmp_path / master_name
    with h5py.File(master_path, "w") as mf:
        e = mf.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        data_grp = e.create_group("data")
        for k, dp in enumerate(data_paths, start=1):
            data_grp[f"data_{k:06d}"] = h5py.ExternalLink(dp.name, "/data")

    full = np.concatenate(chunks, axis=0)
    return master_path, full


class TestNexusImageStack:
    """Functional tests for the external-link aware image stack proxy."""

    def test_single_dataset_round_trip(self, nexus_file):
        """A file with one detector dataset works as a 1-segment stack."""
        with open_nexus_image_stack(nexus_file) as stack:
            assert isinstance(stack, NexusImageStack)
            assert stack.ndim == 3
            assert stack.shape == (5, 20, 30)
            assert len(stack) == 5
            frame_0 = np.asarray(stack[0])
            assert frame_0.shape == (20, 30)
            full = np.asarray(stack[:])
            assert full.shape == (5, 20, 30)
            assert np.array_equal(full[0], frame_0)

    def test_eiger_external_links_concatenated(self, tmp_path):
        """Stack should equal the per-file concatenation along axis 0."""
        master, expected = _make_eiger_master(
            tmp_path, n_files=3, frames_per_file=4,
        )
        with open_nexus_image_stack(master) as stack:
            assert stack.shape == expected.shape
            assert stack.dtype == expected.dtype
            full = np.asarray(stack[:])
            assert np.array_equal(full, expected)

    def test_slice_crosses_file_boundary(self, tmp_path):
        """Slicing across a segment boundary stitches correctly."""
        master, expected = _make_eiger_master(
            tmp_path, n_files=4, frames_per_file=5,
        )
        # Total = 20 frames; this slice spans segments 0, 1, 2.
        with open_nexus_image_stack(master) as stack:
            block = np.asarray(stack[3:13])
            assert block.shape == (10,) + expected.shape[1:]
            assert np.array_equal(block, expected[3:13])

    def test_int_index_at_each_segment(self, tmp_path):
        """Indexing the first and last frame of each segment works."""
        master, expected = _make_eiger_master(
            tmp_path, n_files=3, frames_per_file=4,
        )
        with open_nexus_image_stack(master) as stack:
            for i in (0, 3, 4, 7, 8, 11):
                assert np.array_equal(np.asarray(stack[i]), expected[i])

    def test_negative_index(self, tmp_path):
        master, expected = _make_eiger_master(
            tmp_path, n_files=2, frames_per_file=3,
        )
        with open_nexus_image_stack(master) as stack:
            assert np.array_equal(np.asarray(stack[-1]), expected[-1])
            assert np.array_equal(np.asarray(stack[-6]), expected[0])

    def test_out_of_range(self, tmp_path):
        master, _ = _make_eiger_master(
            tmp_path, n_files=2, frames_per_file=3,
        )
        with open_nexus_image_stack(master) as stack:
            with pytest.raises(IndexError):
                stack[6]
            with pytest.raises(IndexError):
                stack[-7]

    def test_iteration_yields_each_frame(self, tmp_path):
        master, expected = _make_eiger_master(
            tmp_path, n_files=2, frames_per_file=3,
        )
        with open_nexus_image_stack(master) as stack:
            frames = [np.asarray(f) for f in stack]
        assert len(frames) == expected.shape[0]
        for i, f in enumerate(frames):
            assert np.array_equal(f, expected[i])

    def test_slice_with_step(self, tmp_path):
        master, expected = _make_eiger_master(
            tmp_path, n_files=3, frames_per_file=4,
        )
        with open_nexus_image_stack(master) as stack:
            sub = np.asarray(stack[1:11:3])
            assert np.array_equal(sub, expected[1:11:3])

    def test_empty_slice(self, tmp_path):
        master, _ = _make_eiger_master(
            tmp_path, n_files=2, frames_per_file=3,
        )
        with open_nexus_image_stack(master) as stack:
            out = stack[5:5]
            assert out.shape == (0,) + stack.shape[1:]

    def test_single_2d_detector_frame_is_a_one_frame_stack(self, tmp_path):
        """F6 (Codex review): a lone 2-D detector dataset (one-exposure
        acquisition) must open as a logical (1, H, W) stack — the classifier
        and display reader already accept ndim >= 2; this used to raise
        '... is 2-D; NexusImageStack expects 3-D'."""
        img = np.arange(63, dtype=np.uint32).reshape(7, 9)
        p = tmp_path / "single.nxs"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"
            d = e.create_group("data")
            d.attrs["NX_class"] = "NXdata"
            d.create_dataset("data", data=img)
        with open_nexus_image_stack(p) as stack:
            assert stack.shape == (1, 7, 9)
            assert stack.ndim == 3
            assert len(stack) == 1
            assert np.array_equal(np.asarray(stack[0]), img)
            assert np.array_equal(np.asarray(stack[-1]), img)
            with pytest.raises(IndexError):
                stack[1]
            block = np.asarray(stack[0:1])
            assert block.shape == (1, 7, 9)
            assert np.array_equal(block[0], img)
            assert stack[1:1].shape == (0, 7, 9)
            frames = [np.asarray(fr) for fr in stack]
            assert len(frames) == 1
            assert np.array_equal(frames[0], img)

    def test_multi_segment_2d_still_rejected(self, tmp_path):
        """The (1, H, W) normalization is single-dataset only — a 2-D dataset
        among Eiger external-link segments remains a hard error."""
        a = tmp_path / "a.h5"
        b = tmp_path / "b.h5"
        for path in (a, b):
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=np.zeros((10, 10), dtype=np.uint16))
        master = tmp_path / "master.h5"
        with h5py.File(master, "w") as mf:
            d = mf.create_group("entry").create_group("data")
            d["data_000001"] = h5py.ExternalLink(a.name, "/data")
            d["data_000002"] = h5py.ExternalLink(b.name, "/data")
        with pytest.raises(ValueError, match="expects 3-D"):
            open_nexus_image_stack(master)

    def test_inconsistent_per_frame_shapes_raise(self, tmp_path):
        """Mismatched (H, W) across segments must be rejected."""
        # Two data files: one (4, 10, 10), the other (4, 12, 10).
        a = tmp_path / "a.h5"
        b = tmp_path / "b.h5"
        with h5py.File(a, "w") as f:
            f.create_dataset("data", data=np.zeros((4, 10, 10), dtype=np.uint16))
        with h5py.File(b, "w") as f:
            f.create_dataset("data", data=np.zeros((4, 12, 10), dtype=np.uint16))
        master = tmp_path / "master.h5"
        with h5py.File(master, "w") as mf:
            e = mf.create_group("entry")
            d = e.create_group("data")
            d["data_000001"] = h5py.ExternalLink(a.name, "/data")
            d["data_000002"] = h5py.ExternalLink(b.name, "/data")
        with pytest.raises(ValueError, match="Inconsistent per-frame"):
            open_nexus_image_stack(master)

    def test_lexical_sort_is_numeric_for_eiger_names(self, tmp_path):
        """Zero-padded Eiger names sort identically lexical vs numeric."""
        # Build 12 files; only ordering matters here.  If we accidentally
        # used string-sort on un-padded names "data_2" would come before
        # "data_10".  Eiger always pads, so this should be a no-op test.
        master, expected = _make_eiger_master(
            tmp_path, n_files=12, frames_per_file=2,
        )
        with open_nexus_image_stack(master) as stack:
            full = np.asarray(stack[:])
            assert np.array_equal(full, expected)

    def test_context_manager_closes_file(self, tmp_path):
        master, _ = _make_eiger_master(
            tmp_path, n_files=2, frames_per_file=3,
        )
        with open_nexus_image_stack(master) as stack:
            # File handle is live inside the with-block.
            assert stack._h5.id.valid
            assert len(stack) == 6
        # After exit, the file handle is gone.
        assert stack._h5 is None

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_nexus_image_stack(tmp_path / "ghost.h5")

    def test_no_image_dataset_raises(self, tmp_path):
        p = tmp_path / "barren.h5"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.create_dataset("scalar", data=1.0)
        # (message wording updated with F6: 2-D single frames now count, so the
        # error no longer claims a "3-D" requirement)
        with pytest.raises(KeyError, match="No image dataset"):
            open_nexus_image_stack(p)


# ---------------------------------------------------------------------------
# TestListEntries
# ---------------------------------------------------------------------------

class TestReadSphereMetadata:
    """Locks down the C5 lean reader's contract.

    The metadata loader exists so the GUI open path can spin up
    an :class:`EwaldSphere` viewer in O(few KB) reads instead of
    O(N * nchi * nq * 4B).  Per-frame intensity arrays stay on disk
    and ArchSeries lazy-loads them.
    """

    @pytest.fixture
    def synth_v2_sphere(self, tmp_path):
        """Hand-craft a minimal v2 NXroot with frame_index, q/chi,
        positioners, and an ``integrated_2d`` that is intentionally
        big so 'metadata-only' has something to refuse to load."""
        p = tmp_path / "sphere.nxs"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"

            # 1D stack — 100 frames × 32 q.  Frame IDs 1-based to
            # mimic SPEC (the C4 alignment case).
            g1 = e.create_group("integrated_1d")
            g1.create_dataset(
                "intensity",
                data=np.ones((100, 32), dtype=np.float32),
            )
            g1.create_dataset(
                "frame_index",
                data=np.arange(1, 101, dtype=np.int32),
            )
            q = g1.create_dataset("q",
                                  data=np.linspace(0.5, 5.0, 32,
                                                   dtype=np.float32))
            q.attrs["units"] = b"1/angstrom"

            # 2D stack — 100 × 16 × 32.  Bigger ndim; this is the
            # one the metadata loader must NOT pull into memory.
            g2 = e.create_group("integrated_2d")
            g2.create_dataset(
                "intensity",
                data=np.ones((100, 16, 32), dtype=np.float32),
            )
            g2.create_dataset("q",
                              data=np.linspace(0.5, 5.0, 32,
                                               dtype=np.float32))
            chi = g2.create_dataset(
                "chi", data=np.linspace(-180.0, 180.0, 16,
                                        endpoint=False,
                                        dtype=np.float32),
            )
            chi.attrs["units"] = b"deg"

            # Positioner.
            samp = e.create_group("sample")
            samp.attrs["NX_class"] = "NXsample"
            pos = samp.create_group("positioners")
            pos.attrs["NX_class"] = "NXcollection"
            th = pos.create_group("th")
            th.attrs["NX_class"] = "NXpositioner"
            v = th.create_dataset("value",
                                  data=np.linspace(0.0, 9.9, 100,
                                                   dtype=np.float32))
            v.attrs["units"] = b"deg"
        return p

    def test_returns_frame_coord_and_axes(self, synth_v2_sphere):
        ds = read_scan_metadata(synth_v2_sphere)
        assert "frame" in ds.coords
        assert "q" in ds.coords
        assert "q_2d" in ds.coords
        assert "chi" in ds.coords
        # Frame IDs preserved (1-based, not arange).
        assert list(ds["frame"].values) == list(range(1, 101))

    def test_omits_intensity_arrays(self, synth_v2_sphere):
        ds = read_scan_metadata(synth_v2_sphere)
        assert "intensity_1d" not in ds.data_vars
        assert "intensity_2d" not in ds.data_vars
        assert "sigma_1d" not in ds.data_vars
        assert "thumbnail" not in ds.data_vars

    def test_includes_positioners(self, synth_v2_sphere):
        ds = read_scan_metadata(synth_v2_sphere)
        assert "th" in ds.data_vars
        assert ds["th"].dims == ("frame",)
        assert ds["th"].shape == (100,)

    def test_units_round_trip(self, synth_v2_sphere):
        ds = read_scan_metadata(synth_v2_sphere)
        assert ds["q"].attrs.get("units") == "1/angstrom"
        assert ds["chi"].attrs.get("units") == "deg"


class TestListEntries:
    def test_single_entry(self, nexus_file):
        entries = list_entries(nexus_file)
        assert entries == ["entry"]

    def test_multiple_entries(self, tmp_path):
        p = tmp_path / "multi.h5"
        with h5py.File(p, "w") as f:
            for name in ("entry_1", "entry_2"):
                g = f.create_group(name)
                g.attrs["NX_class"] = "NXentry"
        entries = list_entries(p)
        assert set(entries) == {"entry_1", "entry_2"}

    def test_fallback_to_all_groups(self, tmp_path):
        p = tmp_path / "plain.h5"
        with h5py.File(p, "w") as f:
            f.create_group("entry_a")
            f.create_group("entry_b")
        entries = list_entries(p)
        assert set(entries) == {"entry_a", "entry_b"}

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_entries(tmp_path / "ghost.h5")


# ---------------------------------------------------------------------------
# TestReadNexusMissingFields
# ---------------------------------------------------------------------------

class TestReadNexusMissingFields:
    def test_missing_energy(self, tmp_path):
        p = _minimal_nexus(tmp_path, name="no_energy.h5", omit={"energy", "wavelength"})
        meta = read_nexus(p)
        assert isinstance(meta, ScanMetadata)
        assert np.isnan(meta.energy)
        assert np.isnan(meta.wavelength)

    def test_missing_energy_with_wavelength_present(self, tmp_path):
        p = _minimal_nexus(tmp_path, name="no_en_wl.h5", omit={"energy"})
        meta = read_nexus(p)
        assert np.isnan(meta.energy)
        np.testing.assert_allclose(meta.wavelength, 1.033, rtol=1e-4)

    def test_missing_ub_matrix(self, tmp_path):
        p = _minimal_nexus(tmp_path, name="no_ub.h5", omit={"ub_matrix"})
        meta = read_nexus(p)
        assert meta.ub_matrix is None

    def test_missing_sample_name(self, tmp_path):
        p = _minimal_nexus(tmp_path, name="no_name.h5", omit={"name"})
        meta = read_nexus(p)
        assert meta.sample_name == ""

    def test_no_data_group(self, tmp_path):
        p = tmp_path / "no_data_grp.h5"
        with h5py.File(p, "w") as f:
            e = f.create_group("entry")
            e.attrs["NX_class"] = "NXentry"
            mono = e.create_group("instrument/monochromator")
            mono.create_dataset("energy", data=12.0)
        meta = read_nexus(p)
        assert meta.angles == {}
        assert meta.counters == {}


# ---------------------------------------------------------------------------
# TestWriteNexus — file-level single-call write
# ---------------------------------------------------------------------------

class TestWriteNexus:
    def test_returns_path(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "out.h5", results_1d={0: result_1d})
        assert p == tmp_path / "out.h5"
        assert p.exists()

    def test_creates_parent_dirs(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "subdir" / "deep" / "out.h5", results_1d={0: result_1d})
        assert p.exists()

    def test_nxentry_class(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert f["entry"].attrs["NX_class"] == "NXentry"

    def test_reduction_group(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert "reduction" in f["entry"]
            assert f["entry/reduction"].attrs["NX_class"] == "NXprocess"
            # Persisted-format literal: stays "ssrl_xrd_tools" across the
            # monorepo rename (stable on-disk output).
            assert f["entry/reduction"].attrs["program"] == "ssrl_xrd_tools"

    def test_overwrite(self, tmp_path, result_1d):
        p = tmp_path / "over.h5"
        write_nexus(p, results_1d={0: result_1d})
        write_nexus(p, results_1d={1: result_1d}, overwrite=True)
        with h5py.File(p, "r") as f:
            frames = list(f["entry/integrated_1d/frame_index"][()])
        assert frames == [1]  # old frame gone

    def test_append_default(self, tmp_path, result_1d):
        p = tmp_path / "app.h5"
        write_nexus(p, results_1d={0: result_1d})
        write_nexus(p, results_1d={1: result_1d})  # append
        with h5py.File(p, "r") as f:
            frames = list(f["entry/integrated_1d/frame_index"][()])
        assert frames == [0, 1]

    def test_compression_none(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "nc.h5", results_1d={0: result_1d}, compression=None)
        with h5py.File(p, "r") as f:
            assert f["entry/integrated_1d/intensity"].compression is None

    def test_compression_lzf_is_aliased_to_portable_gzip(self, tmp_path, result_1d):
        # Converged policy: lzf is never emitted (h5py-only filter, ARM64-macOS
        # bus error) -- it is a backward-compatible alias for gzip on EVERY
        # platform, so the written file is always gzip and stock-h5py readable.
        p = write_nexus(tmp_path / "lzf.h5", results_1d={0: result_1d}, compression="lzf")
        with h5py.File(p, "r") as f:
            assert f["entry/integrated_1d/intensity"].compression == "gzip"

    def test_comp_kwargs_single_policy_gzip_shuffle(self):
        # gzip and lzf both resolve to the one portable policy (gzip + shuffle +
        # fast level 1) on all platforms; None means uncompressed.
        expected = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
        assert _comp_kwargs("gzip") == expected
        assert _comp_kwargs("lzf") == expected
        assert _comp_kwargs(None) == {}

    def test_comp_kwargs_lz4_returns_hdf5plugin_codec_when_safe(self, monkeypatch):
        # On a machine where the lz4 plugin is available, "lz4" emits hdf5plugin's
        # LZ4 (filter 32004) + the HDF5 shuffle filter -- the reader needs hdf5plugin.
        hdf5plugin = pytest.importorskip("hdf5plugin")
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: False)
        kw = _comp_kwargs("lz4")
        assert kw == dict(shuffle=True, **hdf5plugin.LZ4())
        assert kw["compression"] == 32004
        assert kw["shuffle"] is True

    def test_comp_kwargs_lz4_falls_back_to_gzip_when_native_unsafe(self, monkeypatch):
        # Defensive: even if asked for lz4 on ARM64-macOS / no-plugin, never emit
        # an unreadable filter -- degrade to the portable gzip+shuffle policy.
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: True)
        assert _comp_kwargs("lz4") == {
            "compression": "gzip",
            "compression_opts": 1,
            "shuffle": True,
        }


class TestResolveStackCompression:
    """XDART_INTEGRATED_COMPRESSION is resolved identically for the GUI writer
    and the headless reduction (one source of truth), so a shell override applies
    to both.  Falsey -> None; lz4 (the DEFAULT) -> hdf5plugin LZ4 when available
    else gzip; gzip/lzf -> portable gzip; anything unrecognized -> gzip + warn."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("XDART_INTEGRATED_COMPRESSION", raising=False)

    def test_unset_defaults_to_lz4_when_plugin_available(self, monkeypatch):
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: False)
        assert resolve_stack_compression() == "lz4"

    def test_unset_falls_back_to_gzip_without_plugin(self, monkeypatch):
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: True)
        assert resolve_stack_compression() == "gzip"

    def test_unset_honors_explicit_default(self):
        assert resolve_stack_compression(default=None) is None

    @pytest.mark.parametrize(
        "val", ["none", "off", "0", "false", "no", "NONE", "  none  "]
    )
    def test_falsey_values_disable_compression(self, monkeypatch, val):
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", val)
        assert resolve_stack_compression() is None

    def test_empty_env_is_treated_as_unset_not_off(self, monkeypatch):
        # A stale ``export XDART_INTEGRATED_COMPRESSION=`` must NOT silently
        # disable compression: empty == unset -> default (lz4), not off.
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: False)
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "")
        assert resolve_stack_compression() == "lz4"

    def test_explicit_env_disable_warns(self, monkeypatch, caplog):
        # Opting into uncompressed output via the env var is loud (WARNING), so a
        # ~4x-bigger-file surprise is visible in the log.
        import logging
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "none")
        with caplog.at_level(logging.WARNING):
            assert resolve_stack_compression() is None
        assert any("DISABLED" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("val", ["gzip", "GZIP", "lzf", "  gzip  "])
    def test_gzip_and_lzf_alias_to_portable_gzip(self, monkeypatch, val):
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", val)
        assert resolve_stack_compression() == "gzip"

    def test_lz4_selected_when_native_filters_safe(self, monkeypatch):
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: False)
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "lz4")
        assert resolve_stack_compression() == "lz4"

    def test_lz4_falls_back_to_gzip_when_native_unsafe(self, monkeypatch):
        monkeypatch.setattr(_nexus_io, "_native_filter_unsafe", lambda: True)
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "lz4")
        assert resolve_stack_compression() == "gzip"

    def test_unknown_filter_falls_back_to_gzip(self, monkeypatch):
        # A typo'd / unknown codec must NOT be honored verbatim (that crashes
        # every write at create_dataset) -- degrade loudly to the portable gzip.
        monkeypatch.setenv("XDART_INTEGRATED_COMPRESSION", "blosc")
        assert resolve_stack_compression() == "gzip"

    def test_lz4_round_trips_when_natively_supported(self, tmp_path, result_1d):
        # Real write+read of an lz4 stack -- runs wherever hdf5plugin is importable
        # (incl. arm64-macOS: the old bus error was h5py's bundled LZF, not LZ4).
        pytest.importorskip("hdf5plugin")
        if _nexus_io._native_filter_unsafe():
            pytest.skip("hdf5plugin not importable here")
        p = write_nexus(tmp_path / "lz4.h5", results_1d={0: result_1d}, compression="lz4")
        with h5py.File(p, "r") as f:
            ds = f["entry/integrated_1d/intensity"]
            # h5py reports plugin filters as 'unknown'; the LZ4 filter id 32004 is
            # in the filter pipeline (it does NOT show up as ds.compression).
            assert "32004" in dict(ds._filters)   # hdf5plugin LZ4 engaged
            assert ds.shuffle is True             # shuffle filter engaged
            # lz4 is lossless; the only diff is the float64->float32 store dtype.
            np.testing.assert_allclose(ds[0], result_1d.intensity, rtol=1e-6)

    def test_compression_gzip(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "gz.h5", results_1d={0: result_1d}, compression="gzip")
        with h5py.File(p, "r") as f:
            ds = f["entry/integrated_1d/intensity"]
            assert ds.compression == "gzip"
            assert ds.shuffle is True            # shuffle filter engaged (better ratio)

    def test_custom_entry_name(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "e.h5", results_1d={0: result_1d}, entry="scan_1")
        with h5py.File(p, "r") as f:
            assert "scan_1" in f
            assert "entry" not in f


# ---------------------------------------------------------------------------
# TestWriteNexusResult1D
# ---------------------------------------------------------------------------

class TestWriteNexusResult1D:
    """Stacked /entry/integrated_1d layout (read_scan-compatible)."""

    def test_group_created(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert "integrated_1d" in f["entry"]
            assert "intensity" in f["entry/integrated_1d"]

    def test_nxdata_class(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            grp = f["entry/integrated_1d"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            assert list(grp.attrs["axes"]) == ["frame_index", "q"]

    def test_q_axis_values(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/integrated_1d/q"][()], result_1d.radial, rtol=1e-6,
            )

    def test_intensity_values_stacked(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            stored = f["entry/integrated_1d/intensity"][()]
        assert stored.shape == (1, result_1d.intensity.shape[0])
        np.testing.assert_allclose(stored[0], result_1d.intensity, rtol=1e-6)

    def test_sigma_written_when_present(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert "sigma" in f["entry/integrated_1d"]
            np.testing.assert_allclose(
                f["entry/integrated_1d/sigma"][0], result_1d.sigma, rtol=1e-6,
            )

    def test_sigma_absent_when_none(self, tmp_path):
        r = IntegrationResult1D(
            radial=np.linspace(0, 5, 50), intensity=np.ones(50), unit="q_A^-1",
        )
        p = write_nexus(tmp_path / "a.h5", results_1d={0: r})
        with h5py.File(p, "r") as f:
            assert "sigma" not in f["entry/integrated_1d"]

    def test_unit_on_q_axis(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert f["entry/integrated_1d/q"].attrs["units"] == "q_A^-1"

    def test_non_integer_frame_key_raises(self, tmp_path, result_1d):
        # The stacked frame_index requires integer frame labels.
        with pytest.raises(ValueError):
            write_nexus(tmp_path / "a.h5", results_1d={"frame_5": result_1d})

    def test_multiple_frames_stacked(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={i: result_1d for i in range(5)})
        with h5py.File(p, "r") as f:
            assert list(f["entry/integrated_1d/frame_index"][()]) == [0, 1, 2, 3, 4]
            assert f["entry/integrated_1d/intensity"].shape[0] == 5


# ---------------------------------------------------------------------------
# TestWriteNexusResult2D
# ---------------------------------------------------------------------------

class TestWriteNexusResult2D:
    """Stacked /entry/integrated_2d layout (read_scan-compatible)."""

    def test_group_created(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            assert "integrated_2d" in f["entry"]

    def test_nxdata_class(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            grp = f["entry/integrated_2d"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            assert list(grp.attrs["axes"]) == ["frame_index", "chi", "q"]

    def test_axes_values(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/integrated_2d/q"][()], result_2d.radial, rtol=1e-6,
            )
            np.testing.assert_allclose(
                f["entry/integrated_2d/chi"][()], result_2d.azimuthal, rtol=1e-6,
            )

    def test_intensity_shape_and_orientation(self, tmp_path, result_2d):
        # IntegrationResult2D.intensity is (n_q, n_chi); stored (frame, chi, q).
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            stored = f["entry/integrated_2d/intensity"][()]
        n_q, n_chi = result_2d.intensity.shape
        assert stored.shape == (1, n_chi, n_q)
        np.testing.assert_allclose(stored[0], result_2d.intensity.T, rtol=1e-6)

    def test_intensity_chunked(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d}, compression="lzf")
        with h5py.File(p, "r") as f:
            assert f["entry/integrated_2d/intensity"].chunks is not None

    def test_sigma_written_when_present(self, tmp_path):
        q = np.linspace(0.1, 5, 30)
        chi = np.linspace(-90, 90, 20)
        r = IntegrationResult2D(radial=q, azimuthal=chi, intensity=np.ones((30, 20)),
                                sigma=np.ones((30, 20)) * 0.1, unit="q_A^-1")
        p = write_nexus(tmp_path / "a.h5", results_2d={0: r})
        with h5py.File(p, "r") as f:
            assert "sigma" in f["entry/integrated_2d"]

    def test_unit_on_q_axis(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            assert f["entry/integrated_2d/q"].attrs["units"] == "q_A^-1"


# ---------------------------------------------------------------------------
# TestWriteNexusMetadata
# ---------------------------------------------------------------------------

class TestWriteNexusMetadata:
    def test_energy_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            e = f["entry/instrument/monochromator/energy"][()]
        assert float(e) == pytest.approx(12.0)

    def test_wavelength_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            wl = f["entry/instrument/monochromator/wavelength"][()]
        assert float(wl) == pytest.approx(1.033)

    def test_sample_name_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            raw = f["entry/sample/name"][()]
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        assert name == "my_film"

    def test_ub_matrix_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            ub = f["entry/sample/ub_matrix"][()]
        np.testing.assert_allclose(ub, np.eye(3))

    def test_scan_id_attribute(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            assert f["entry"].attrs["scan_id"] == "scan_042"

    def test_motor_arrays_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/data/th"][()], sample_metadata.angles["th"]
            )

    def test_counter_arrays_written(self, tmp_path, sample_metadata):
        p = write_nexus(tmp_path / "a.h5", metadata=sample_metadata)
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/data/i0"][()], sample_metadata.counters["i0"]
            )

    def test_no_ub_matrix_skipped(self, tmp_path):
        meta = ScanMetadata(
            scan_id="x", energy=12.0, wavelength=1.033,
            angles={}, counters={}, ub_matrix=None,
        )
        p = write_nexus(tmp_path / "a.h5", metadata=meta)
        with h5py.File(p, "r") as f:
            assert "ub_matrix" not in f["entry/sample"]

    def test_empty_sample_name_skipped(self, tmp_path):
        meta = ScanMetadata(
            scan_id="x", energy=12.0, wavelength=1.033,
            angles={}, counters={}, sample_name="",
        )
        p = write_nexus(tmp_path / "a.h5", metadata=meta)
        with h5py.File(p, "r") as f:
            assert "name" not in f["entry/sample"]

    def test_roundtrip_metadata(self, tmp_path, sample_metadata):
        """write_nexus + read_nexus should recover metadata."""
        p = write_nexus(tmp_path / "scan_042.h5", metadata=sample_metadata)
        recovered = read_nexus(p)
        assert recovered.scan_id == "scan_042"
        np.testing.assert_allclose(recovered.energy, sample_metadata.energy)
        np.testing.assert_allclose(recovered.wavelength, sample_metadata.wavelength)
        assert recovered.sample_name == sample_metadata.sample_name
        np.testing.assert_allclose(recovered.ub_matrix, sample_metadata.ub_matrix)


# ---------------------------------------------------------------------------
# TestOpenNexusWriter + write_nexus_frame
# ---------------------------------------------------------------------------

class TestOpenNexusWriter:
    def test_returns_open_file(self, tmp_path):
        h5 = open_nexus_writer(tmp_path / "live.h5")
        try:
            assert isinstance(h5, h5py.File)
            assert h5.id.valid
        finally:
            h5.close()

    def test_nxentry_created(self, tmp_path):
        h5 = open_nexus_writer(tmp_path / "live.h5")
        try:
            assert "entry" in h5
            assert h5["entry"].attrs["NX_class"] == "NXentry"
        finally:
            h5.close()

    def test_reduction_group_created(self, tmp_path):
        h5 = open_nexus_writer(tmp_path / "live.h5")
        try:
            assert "entry/reduction" in h5
        finally:
            h5.close()

    def test_metadata_written_on_open(self, tmp_path, sample_metadata):
        h5 = open_nexus_writer(tmp_path / "live.h5", metadata=sample_metadata)
        try:
            assert "entry/instrument/monochromator/energy" in h5
            assert float(h5["entry/instrument/monochromator/energy"][()]) == pytest.approx(12.0)
        finally:
            h5.close()

    def test_overwrite(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        write_nexus_frame(h5, 0, result_1d=result_1d)
        h5.close()

        h5 = open_nexus_writer(p, overwrite=True)
        try:
            assert "0" not in h5.get("entry/reduction", {})
        finally:
            h5.close()

    def test_custom_entry(self, tmp_path):
        h5 = open_nexus_writer(tmp_path / "live.h5", entry="run_1")
        try:
            assert "run_1" in h5
        finally:
            h5.close()


class TestWriteNexusFrame:
    def test_writes_1d_result(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/integrated_1d/intensity"][0], result_1d.intensity, rtol=1e-6,
            )

    def test_writes_2d_result(self, tmp_path, result_2d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_2d=result_2d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            stored = f["entry/integrated_2d/intensity"][0]   # (chi, q)
        np.testing.assert_allclose(stored, result_2d.intensity.T, rtol=1e-6)

    def test_writes_multiple_frames(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            for i in range(4):
                write_nexus_frame(h5, i, result_1d=result_1d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert list(f["entry/integrated_1d/frame_index"][()]) == [0, 1, 2, 3]
            assert f["entry/integrated_1d/intensity"].shape[0] == 4

    def test_appends_in_call_order(self, tmp_path, result_1d):
        """Stacked write is append-only — frames land in the order written.

        Frames in one scan share the q axis (the second frame reuses
        ``result_1d.radial``); see ``test_append_rejects_divergent_axis``
        for the case where it differs."""
        r_new = IntegrationResult1D(
            radial=result_1d.radial, intensity=np.full(200, 2.0), unit="q_A^-1")
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d)
            write_nexus_frame(h5, 1, result_1d=r_new)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert list(f["entry/integrated_1d/frame_index"][()]) == [0, 1]
            np.testing.assert_allclose(
                f["entry/integrated_1d/intensity"][1], r_new.intensity, rtol=1e-6,
            )

    def test_append_rejects_divergent_axis(self, tmp_path, result_1d):
        """Per-frame append (NexusSink path) can't refresh the shared q axis,
        so a frame whose radial grid/unit differs from disk is rejected
        rather than stored under the stale axis (silent corruption)."""
        import pytest
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d)        # q_A^-1
            tth = IntegrationResult1D(  # same length, different axis + unit
                radial=np.linspace(1, 6, 200),
                intensity=np.full(200, 2.0), unit="2th_deg")
            with pytest.raises(ValueError, match="axis"):
                write_nexus_frame(h5, 1, result_1d=tth)
        finally:
            h5.close()

    def test_frame_with_both_1d_and_2d(self, tmp_path, result_1d, result_2d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d, result_2d=result_2d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert "integrated_1d" in f["entry"]
            assert "integrated_2d" in f["entry"]

    def test_flush_does_not_corrupt(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d)
            h5.flush()
            write_nexus_frame(h5, 1, result_1d=result_1d)
            h5.flush()
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert list(f["entry/integrated_1d/frame_index"][()]) == [0, 1]

    def test_non_integer_frame_key_raises(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            with pytest.raises(ValueError):
                write_nexus_frame(h5, "frame_007", result_1d=result_1d)
        finally:
            h5.close()

    def test_no_result_is_noop(self, tmp_path):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0)  # no result_1d, no result_2d
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert "integrated_1d" not in f["entry"]


# ---------------------------------------------------------------------------
# TestRoundtrip — write then read back raw HDF5 structure
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_1d_full_roundtrip(self, tmp_path, sample_metadata, result_1d):
        p = write_nexus(
            tmp_path / "scan_042.h5",
            metadata=sample_metadata,
            results_1d={0: result_1d, 1: result_1d},
        )
        with h5py.File(p, "r") as f:
            g = f["entry/integrated_1d"]
            assert list(g["frame_index"][()]) == [0, 1]
            np.testing.assert_allclose(g["q"][()], result_1d.radial, rtol=1e-6)
            for row in (0, 1):
                np.testing.assert_allclose(g["intensity"][row], result_1d.intensity, rtol=1e-6)
                np.testing.assert_allclose(g["sigma"][row], result_1d.sigma, rtol=1e-6)
            assert g["q"].attrs["units"] == "q_A^-1"

    def test_2d_full_roundtrip(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "scan.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            g = f["entry/integrated_2d"]
            np.testing.assert_allclose(g["q"][()], result_2d.radial, rtol=1e-6)
            np.testing.assert_allclose(g["chi"][()], result_2d.azimuthal, rtol=1e-6)
            np.testing.assert_allclose(g["intensity"][0], result_2d.intensity.T, rtol=1e-6)

    def test_list_entries_after_write(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "scan.h5", results_1d={0: result_1d}, entry="entry_001")
        entries = list_entries(p)
        assert "entry_001" in entries

    def test_incremental_same_as_batch(self, tmp_path, result_1d, result_2d):
        """open_nexus_writer + write_nexus_frame produces the same stacked
        structure as a single write_nexus call."""
        p_batch = tmp_path / "batch.h5"
        write_nexus(p_batch, results_1d={0: result_1d}, results_2d={0: result_2d})

        p_incr = tmp_path / "incr.h5"
        h5 = open_nexus_writer(p_incr)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d, result_2d=result_2d)
        finally:
            h5.close()

        with h5py.File(p_batch, "r") as fb, h5py.File(p_incr, "r") as fi:
            for key in ("integrated_1d/q", "integrated_1d/intensity",
                        "integrated_2d/q", "integrated_2d/chi", "integrated_2d/intensity"):
                np.testing.assert_allclose(fb[f"entry/{key}"][()], fi[f"entry/{key}"][()])


class TestRequireUniformAxesGuard:
    """Lock the stacked-write uniform-axes invariant in isolation.

    A batch ``integrated_1d`` / ``integrated_2d`` stack stores ONE shared axis
    for every frame, so all frames must agree on it.  ``_require_uniform_axes``
    is the contract a multi-frame writer (and any per-frame-axis freeze that
    feeds it — e.g. the grazing-incidence common-grid freeze in xdart's
    ``image_wrangler_thread``) must satisfy: identical (frozen) per-frame axes
    are accepted; axis/unit drift beyond ``rtol=1e-5`` is rejected.

    These run without any detector data (unlike xdart's real-data GI matrix,
    which is skipped in CI), so they are the always-on guard for the invariant.
    """

    @staticmethod
    def _r1d(radial, *, unit="q_A^-1"):
        radial = np.asarray(radial, float)
        return IntegrationResult1D(
            radial=radial, intensity=np.ones(radial.shape[0]), unit=unit)

    @staticmethod
    def _r2d(radial, azimuthal, *, unit="q_A^-1", azimuthal_unit="deg"):
        radial = np.asarray(radial, float)
        azimuthal = np.asarray(azimuthal, float)
        r = IntegrationResult2D(
            radial=radial, azimuthal=azimuthal,
            intensity=np.ones((radial.shape[0], azimuthal.shape[0])), unit=unit)
        # azimuthal_unit is optional on the container; set when supported so
        # the unit-drift branch of the 2D validator is exercised.
        try:
            r.azimuthal_unit = azimuthal_unit
        except Exception:
            pass
        return r

    # ── 1D ───────────────────────────────────────────────────────────────────
    def test_uniform_axes_1d_accepts_frozen_grid(self):
        q = np.linspace(0.1, 5.0, 200)
        # Two frames on the SAME frozen q grid — the freeze's whole job.
        _require_uniform_axes_1d([self._r1d(q), self._r1d(q.copy())])

    def test_uniform_axes_1d_accepts_within_tolerance_jitter(self):
        q = np.linspace(0.1, 5.0, 200)
        # Tiny numerical jitter strictly inside rtol=1e-5 must NOT trip it
        # (the frozen grid need only match to the writer's tolerance).
        _require_uniform_axes_1d([self._r1d(q), self._r1d(q * (1.0 + 1e-7))])

    def test_uniform_axes_1d_rejects_drift(self):
        q = np.linspace(0.1, 5.0, 200)
        drifted = np.linspace(0.12, 5.05, 200)  # per-frame auto-range drift
        with pytest.raises(ValueError,
                           match=r"results_1d\[1\] has a different radial axis"):
            _require_uniform_axes_1d([self._r1d(q), self._r1d(drifted)])

    def test_uniform_axes_1d_rejects_unit_change(self):
        q = np.linspace(0.1, 5.0, 200)
        with pytest.raises(ValueError, match=r"results_1d\[1\] has a different"):
            _require_uniform_axes_1d(
                [self._r1d(q, unit="q_A^-1"), self._r1d(q, unit="2th_deg")])

    # ── 2D ───────────────────────────────────────────────────────────────────
    def test_uniform_axes_2d_accepts_frozen_grid(self):
        q = np.linspace(0.1, 5.0, 100)
        chi = np.linspace(-180.0, 180.0, 80)
        _require_uniform_axes_2d(
            [self._r2d(q, chi), self._r2d(q.copy(), chi.copy())])

    def test_uniform_axes_2d_rejects_azimuthal_drift(self):
        # The grazing-incidence out-of-plane (qoop / chi / exit-angle) axis is
        # the one that drifts per incidence — the exact crash the freeze fixes.
        q = np.linspace(0.1, 5.0, 100)
        chi = np.linspace(-180.0, 180.0, 80)
        chi_drift = np.linspace(-175.0, 178.0, 80)
        with pytest.raises(ValueError,
                           match=r"results_2d\[1\] has a different q/chi axis"):
            _require_uniform_axes_2d([self._r2d(q, chi), self._r2d(q, chi_drift)])

    def test_uniform_axes_2d_rejects_radial_drift(self):
        q = np.linspace(0.1, 5.0, 100)
        q_drift = np.linspace(0.15, 5.1, 100)
        chi = np.linspace(-180.0, 180.0, 80)
        with pytest.raises(ValueError, match=r"results_2d\[1\] has a different"):
            _require_uniform_axes_2d([self._r2d(q, chi), self._r2d(q_drift, chi)])

    # ── end-to-end through the writer (mirrors xdart's real-data matrix) ──────
    def test_write_integrated_stack_rejects_drifted_axes(self, tmp_path):
        q = np.linspace(0.1, 5.0, 200)
        drifted = np.linspace(0.12, 5.05, 200)
        with h5py.File(tmp_path / "drift_1d.h5", "w") as h5:
            entry = h5.create_group("entry")
            with pytest.raises(ValueError,
                               match=r"results_1d\[1\] has a different"):
                write_integrated_stack(
                    entry, frame_indices=[0, 1],
                    results_1d=[self._r1d(q), self._r1d(drifted)])

        q2 = np.linspace(0.1, 5.0, 100)
        chi = np.linspace(-180.0, 180.0, 80)
        chi_drift = np.linspace(-175.0, 178.0, 80)
        with h5py.File(tmp_path / "drift_2d.h5", "w") as h5:
            entry = h5.create_group("entry")
            with pytest.raises(ValueError,
                               match=r"results_2d\[1\] has a different"):
                write_integrated_stack(
                    entry, frame_indices=[0, 1],
                    results_2d=[self._r2d(q2, chi), self._r2d(q2, chi_drift)])

    def test_write_integrated_stack_accepts_frozen_grid(self, tmp_path):
        q = np.linspace(0.1, 5.0, 200)
        q2 = np.linspace(0.1, 5.0, 100)
        chi = np.linspace(-180.0, 180.0, 80)
        with h5py.File(tmp_path / "frozen.h5", "w") as h5:
            entry = h5.create_group("entry")
            # Frozen grid → writes cleanly, no ValueError.
            write_integrated_stack(
                entry, frame_indices=[0, 1],
                results_1d=[self._r1d(q), self._r1d(q.copy())],
                results_2d=[self._r2d(q2, chi), self._r2d(q2.copy(), chi.copy())])
        with h5py.File(tmp_path / "frozen.h5", "r") as f:
            assert list(f["entry/integrated_1d/frame_index"][()]) == [0, 1]
            assert list(f["entry/integrated_2d/frame_index"][()]) == [0, 1]
