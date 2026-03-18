"""Tests for ssrl_xrd_tools.io.nexus — reader and writer."""

from __future__ import annotations

import numpy as np
import pytest
import h5py

from ssrl_xrd_tools.io.nexus import (
    find_nexus_image_dataset,
    list_entries,
    open_nexus_writer,
    read_nexus,
    write_nexus,
    write_nexus_frame,
)
from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.metadata import ScanMetadata


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
# TestListEntries
# ---------------------------------------------------------------------------

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
            assert f["entry/reduction"].attrs["program"] == "ssrl_xrd_tools"

    def test_overwrite(self, tmp_path, result_1d):
        p = tmp_path / "over.h5"
        write_nexus(p, results_1d={0: result_1d})
        write_nexus(p, results_1d={1: result_1d}, overwrite=True)
        with h5py.File(p, "r") as f:
            frames = list(f["entry/reduction"].keys())
        assert "0" not in frames  # old frame gone
        assert "1" in frames

    def test_append_default(self, tmp_path, result_1d):
        p = tmp_path / "app.h5"
        write_nexus(p, results_1d={0: result_1d})
        write_nexus(p, results_1d={1: result_1d})  # append
        with h5py.File(p, "r") as f:
            frames = list(f["entry/reduction"].keys())
        assert "0" in frames
        assert "1" in frames

    def test_compression_none(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "nc.h5", results_1d={0: result_1d}, compression=None)
        with h5py.File(p, "r") as f:
            ds = f["entry/reduction/0/int_1d/intensity"]
            assert ds.compression is None

    def test_compression_lzf(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "lzf.h5", results_1d={0: result_1d}, compression="lzf")
        with h5py.File(p, "r") as f:
            ds = f["entry/reduction/0/int_1d/intensity"]
            assert ds.compression == "lzf"

    def test_compression_gzip(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "gz.h5", results_1d={0: result_1d}, compression="gzip")
        with h5py.File(p, "r") as f:
            ds = f["entry/reduction/0/int_1d/intensity"]
            assert ds.compression == "gzip"

    def test_custom_entry_name(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "e.h5", results_1d={0: result_1d}, entry="scan_1")
        with h5py.File(p, "r") as f:
            assert "scan_1" in f
            assert "entry" not in f


# ---------------------------------------------------------------------------
# TestWriteNexusResult1D
# ---------------------------------------------------------------------------

class TestWriteNexusResult1D:
    def test_frame_group_created(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert "0" in f["entry/reduction"]
            assert "int_1d" in f["entry/reduction/0"]

    def test_nxdata_class(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            grp = f["entry/reduction/0/int_1d"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            assert list(grp.attrs["axes"]) == ["radial"]

    def test_radial_values(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/reduction/0/int_1d/radial"][()], result_1d.radial
            )

    def test_intensity_values(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/reduction/0/int_1d/intensity"][()], result_1d.intensity
            )

    def test_sigma_written_when_present(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert "sigma" in f["entry/reduction/0/int_1d"]
            np.testing.assert_allclose(
                f["entry/reduction/0/int_1d/sigma"][()], result_1d.sigma
            )

    def test_sigma_absent_when_none(self, tmp_path):
        r = IntegrationResult1D(
            radial=np.linspace(0, 5, 50),
            intensity=np.ones(50),
            unit="q_A^-1",
        )
        p = write_nexus(tmp_path / "a.h5", results_1d={0: r})
        with h5py.File(p, "r") as f:
            assert "sigma" not in f["entry/reduction/0/int_1d"]

    def test_unit_attribute(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={0: result_1d})
        with h5py.File(p, "r") as f:
            assert f["entry/reduction/0/int_1d"].attrs["unit"] == "q_A^-1"

    def test_string_frame_key(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "a.h5", results_1d={"frame_5": result_1d})
        with h5py.File(p, "r") as f:
            assert "frame_5" in f["entry/reduction"]

    def test_multiple_frames(self, tmp_path, result_1d):
        results = {i: result_1d for i in range(5)}
        p = write_nexus(tmp_path / "a.h5", results_1d=results)
        with h5py.File(p, "r") as f:
            frames = set(f["entry/reduction"].keys())
        assert frames == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# TestWriteNexusResult2D
# ---------------------------------------------------------------------------

class TestWriteNexusResult2D:
    def test_frame_group_created(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            assert "int_2d" in f["entry/reduction/0"]

    def test_nxdata_class(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            grp = f["entry/reduction/0/int_2d"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            assert list(grp.attrs["axes"]) == ["radial", "azimuthal"]

    def test_axes_values(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["entry/reduction/0/int_2d/radial"][()], result_2d.radial
            )
            np.testing.assert_allclose(
                f["entry/reduction/0/int_2d/azimuthal"][()], result_2d.azimuthal
            )

    def test_intensity_shape(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            stored = f["entry/reduction/0/int_2d/intensity"][()]
        assert stored.shape == result_2d.intensity.shape
        np.testing.assert_allclose(stored, result_2d.intensity)

    def test_intensity_chunked(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d}, compression="lzf")
        with h5py.File(p, "r") as f:
            ds = f["entry/reduction/0/int_2d/intensity"]
            assert ds.chunks is not None

    def test_sigma_written_when_present(self, tmp_path):
        q = np.linspace(0.1, 5, 30)
        chi = np.linspace(-90, 90, 20)
        intensity = np.ones((30, 20))
        sigma = intensity * 0.1
        r = IntegrationResult2D(radial=q, azimuthal=chi, intensity=intensity,
                                sigma=sigma, unit="q_A^-1")
        p = write_nexus(tmp_path / "a.h5", results_2d={0: r})
        with h5py.File(p, "r") as f:
            assert "sigma" in f["entry/reduction/0/int_2d"]

    def test_unit_attribute(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "a.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            assert f["entry/reduction/0/int_2d"].attrs["unit"] == "q_A^-1"


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
                f["entry/reduction/0/int_1d/intensity"][()], result_1d.intensity
            )

    def test_writes_2d_result(self, tmp_path, result_2d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_2d=result_2d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            stored = f["entry/reduction/0/int_2d/intensity"][()]
        np.testing.assert_allclose(stored, result_2d.intensity)

    def test_writes_multiple_frames(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            for i in range(4):
                write_nexus_frame(h5, i, result_1d=result_1d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            frames = set(f["entry/reduction"].keys())
        assert frames == {"0", "1", "2", "3"}

    def test_overwrites_existing_frame(self, tmp_path, result_1d):
        """Writing the same frame key twice replaces the data (_replace semantics)."""
        q_new = np.linspace(1, 6, 200)
        r_new = IntegrationResult1D(radial=q_new, intensity=np.ones(200), unit="q_A^-1")
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d)
            write_nexus_frame(h5, 0, result_1d=r_new)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            radial = f["entry/reduction/0/int_1d/radial"][()]
        np.testing.assert_allclose(radial, q_new)

    def test_frame_with_both_1d_and_2d(self, tmp_path, result_1d, result_2d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d, result_2d=result_2d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert "int_1d" in f["entry/reduction/0"]
            assert "int_2d" in f["entry/reduction/0"]

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
            assert "0" in f["entry/reduction"]
            assert "1" in f["entry/reduction"]

    def test_string_frame_key(self, tmp_path, result_1d):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, "frame_007", result_1d=result_1d)
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            assert "frame_007" in f["entry/reduction"]

    def test_no_result_is_noop(self, tmp_path):
        p = tmp_path / "live.h5"
        h5 = open_nexus_writer(p)
        try:
            write_nexus_frame(h5, 0)  # no result_1d, no result_2d
        finally:
            h5.close()
        with h5py.File(p, "r") as f:
            # frame group may or may not exist — just no crash
            pass


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
            for frame in ("0", "1"):
                grp = f[f"entry/reduction/{frame}/int_1d"]
                np.testing.assert_allclose(grp["radial"][()], result_1d.radial)
                np.testing.assert_allclose(grp["intensity"][()], result_1d.intensity)
                np.testing.assert_allclose(grp["sigma"][()], result_1d.sigma)
                assert grp.attrs["unit"] == "q_A^-1"

    def test_2d_full_roundtrip(self, tmp_path, result_2d):
        p = write_nexus(tmp_path / "scan.h5", results_2d={0: result_2d})
        with h5py.File(p, "r") as f:
            grp = f["entry/reduction/0/int_2d"]
            np.testing.assert_allclose(grp["radial"][()], result_2d.radial)
            np.testing.assert_allclose(grp["azimuthal"][()], result_2d.azimuthal)
            np.testing.assert_allclose(grp["intensity"][()], result_2d.intensity)

    def test_list_entries_after_write(self, tmp_path, result_1d):
        p = write_nexus(tmp_path / "scan.h5", results_1d={0: result_1d}, entry="entry_001")
        entries = list_entries(p)
        assert "entry_001" in entries

    def test_incremental_same_as_batch(self, tmp_path, result_1d, result_2d):
        """open_nexus_writer + write_nexus_frame should produce the same structure
        as a single write_nexus call."""
        p_batch = tmp_path / "batch.h5"
        write_nexus(p_batch, results_1d={0: result_1d}, results_2d={0: result_2d})

        p_incr = tmp_path / "incr.h5"
        h5 = open_nexus_writer(p_incr)
        try:
            write_nexus_frame(h5, 0, result_1d=result_1d, result_2d=result_2d)
        finally:
            h5.close()

        with h5py.File(p_batch, "r") as fb, h5py.File(p_incr, "r") as fi:
            for key in ("int_1d/radial", "int_1d/intensity",
                        "int_2d/radial", "int_2d/azimuthal", "int_2d/intensity"):
                np.testing.assert_allclose(
                    fb[f"entry/reduction/0/{key}"][()],
                    fi[f"entry/reduction/0/{key}"][()],
                )
