"""Tests for ssrl_xrd_tools.core (containers and metadata)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.metadata import ScanMetadata


# ---------------------------------------------------------------------------
# PONI
# ---------------------------------------------------------------------------

class TestPONI:
    def test_creation_all_fields(self):
        poni = PONI(
            dist=0.2,
            poni1=0.081,
            poni2=0.0775,
            rot1=0.001,
            rot2=-0.002,
            rot3=0.0,
            wavelength=1.0e-10,
            detector="eiger4m",
        )
        assert poni.dist == pytest.approx(0.2)
        assert poni.poni1 == pytest.approx(0.081)
        assert poni.poni2 == pytest.approx(0.0775)
        assert poni.rot1 == pytest.approx(0.001)
        assert poni.rot2 == pytest.approx(-0.002)
        assert poni.rot3 == pytest.approx(0.0)
        assert poni.wavelength == pytest.approx(1.0e-10)
        assert poni.detector == "eiger4m"

    def test_defaults(self):
        poni = PONI(dist=0.3, poni1=0.05, poni2=0.05)
        assert poni.rot1 == pytest.approx(0.0)
        assert poni.rot2 == pytest.approx(0.0)
        assert poni.rot3 == pytest.approx(0.0)
        assert poni.wavelength == pytest.approx(0.0)
        assert poni.detector == ""

    def test_slots_no_dict(self):
        poni = PONI(dist=0.2, poni1=0.0, poni2=0.0)
        assert not hasattr(poni, "__dict__")

    def test_slots_rejects_new_attribute(self):
        poni = PONI(dist=0.2, poni1=0.0, poni2=0.0)
        with pytest.raises(AttributeError):
            poni.nonexistent_field = 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# IntegrationResult1D
# ---------------------------------------------------------------------------

class TestIntegrationResult1D:
    def test_creation_matching_shapes(self):
        q = np.linspace(0.1, 5.0, 100)
        I = np.ones(100)
        result = IntegrationResult1D(radial=q, intensity=I, unit="q_A^-1")
        np.testing.assert_array_equal(result.radial, q)
        np.testing.assert_array_equal(result.intensity, I)
        assert result.sigma is None
        assert result.unit == "q_A^-1"

    def test_arrays_coerced_to_float64(self):
        q = list(range(50))          # list → ndarray
        I = np.ones(50, dtype=int)   # int array → float
        result = IntegrationResult1D(radial=q, intensity=I)
        assert result.radial.dtype == np.float64
        assert result.intensity.dtype == np.float64

    def test_mismatched_radial_intensity_raises(self):
        with pytest.raises(ValueError, match="radial shape"):
            IntegrationResult1D(radial=np.ones(100), intensity=np.ones(99))

    def test_sigma_none(self):
        result = IntegrationResult1D(
            radial=np.linspace(0, 1, 50),
            intensity=np.zeros(50),
        )
        assert result.sigma is None

    def test_sigma_matching_shape(self):
        q = np.linspace(0, 1, 80)
        I = np.ones(80)
        sig = np.sqrt(I)
        result = IntegrationResult1D(radial=q, intensity=I, sigma=sig)
        assert result.sigma is not None
        assert result.sigma.shape == (80,)
        assert result.sigma.dtype == np.float64

    def test_sigma_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="sigma shape"):
            IntegrationResult1D(
                radial=np.ones(80),
                intensity=np.ones(80),
                sigma=np.ones(79),
            )

    def test_slots_no_dict(self):
        result = IntegrationResult1D(radial=np.ones(10), intensity=np.ones(10))
        assert not hasattr(result, "__dict__")

    def test_default_unit(self):
        result = IntegrationResult1D(radial=np.ones(10), intensity=np.ones(10))
        assert result.unit == "2th_deg"


# ---------------------------------------------------------------------------
# IntegrationResult2D
# ---------------------------------------------------------------------------

class TestIntegrationResult2D:
    def test_creation_correct_shape(self):
        radial = np.linspace(0.1, 5.0, 100)
        azimuthal = np.linspace(-180, 180, 50)
        intensity = np.ones((100, 50))
        result = IntegrationResult2D(
            radial=radial,
            azimuthal=azimuthal,
            intensity=intensity,
            unit="q_A^-1",
        )
        assert result.intensity.shape == (100, 50)
        assert result.radial.shape == (100,)
        assert result.azimuthal.shape == (50,)
        assert result.sigma is None
        assert result.unit == "q_A^-1"

    def test_arrays_coerced_to_float64(self):
        result = IntegrationResult2D(
            radial=list(range(10)),
            azimuthal=list(range(5)),
            intensity=np.ones((10, 5), dtype=int),
        )
        assert result.radial.dtype == np.float64
        assert result.azimuthal.dtype == np.float64
        assert result.intensity.dtype == np.float64

    def test_wrong_intensity_shape_raises(self):
        with pytest.raises(ValueError, match="intensity shape"):
            IntegrationResult2D(
                radial=np.ones(100),
                azimuthal=np.ones(50),
                intensity=np.ones((100, 49)),   # wrong azimuthal count
            )

    def test_transposed_intensity_raises(self):
        """Caller must transpose; (naz, nrad) instead of (nrad, naz) is rejected."""
        with pytest.raises(ValueError, match="intensity shape"):
            IntegrationResult2D(
                radial=np.ones(100),
                azimuthal=np.ones(50),
                intensity=np.ones((50, 100)),   # transposed — wrong convention
            )

    def test_non_2d_intensity_raises(self):
        with pytest.raises(ValueError, match="2D"):
            IntegrationResult2D(
                radial=np.ones(10),
                azimuthal=np.ones(5),
                intensity=np.ones(50),          # 1D — rejected
            )

    def test_sigma_matching_shape(self):
        radial = np.ones(30)
        azimuthal = np.ones(20)
        intensity = np.ones((30, 20))
        sigma = np.full((30, 20), 0.1)
        result = IntegrationResult2D(
            radial=radial, azimuthal=azimuthal,
            intensity=intensity, sigma=sigma,
        )
        assert result.sigma is not None
        assert result.sigma.shape == (30, 20)

    def test_sigma_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="sigma shape"):
            IntegrationResult2D(
                radial=np.ones(30),
                azimuthal=np.ones(20),
                intensity=np.ones((30, 20)),
                sigma=np.ones((30, 19)),        # wrong sigma shape
            )

    def test_slots_no_dict(self):
        result = IntegrationResult2D(
            radial=np.ones(10),
            azimuthal=np.ones(5),
            intensity=np.ones((10, 5)),
        )
        assert not hasattr(result, "__dict__")

    def test_default_unit(self):
        result = IntegrationResult2D(
            radial=np.ones(10),
            azimuthal=np.ones(5),
            intensity=np.ones((10, 5)),
        )
        assert result.unit == "2th_deg"


# ---------------------------------------------------------------------------
# ScanMetadata
# ---------------------------------------------------------------------------

class TestScanMetadata:
    def _make(self, **overrides):
        defaults = dict(
            scan_id="sample_scan12",
            energy=12.4,
            wavelength=1.0,
            angles={"del": [0.0, 0.1, 0.2], "nu": [0.0, 0.0, 0.0]},
            counters={"i0": [1000, 1001, 999], "i1": [950, 948, 952]},
        )
        defaults.update(overrides)
        return ScanMetadata(**defaults)

    def test_creation_required_fields(self):
        meta = self._make()
        assert meta.scan_id == "sample_scan12"
        assert meta.energy == pytest.approx(12.4)
        assert meta.wavelength == pytest.approx(1.0)

    def test_angles_coerced_to_ndarray(self):
        meta = self._make()
        assert isinstance(meta.angles["del"], np.ndarray)
        assert meta.angles["del"].dtype == np.float64
        np.testing.assert_array_equal(meta.angles["del"], [0.0, 0.1, 0.2])

    def test_counters_coerced_to_ndarray(self):
        meta = self._make()
        assert isinstance(meta.counters["i0"], np.ndarray)
        assert meta.counters["i0"].dtype == np.float64

    def test_optional_field_defaults(self):
        meta = self._make()
        assert meta.ub_matrix is None
        assert meta.sample_name == ""
        assert meta.scan_type == ""
        assert meta.source == ""
        assert meta.image_paths == []
        assert meta.h5_path is None
        assert meta.extra == {}

    def test_ub_matrix_coerced(self):
        ub = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        meta = self._make(ub_matrix=ub)
        assert isinstance(meta.ub_matrix, np.ndarray)
        assert meta.ub_matrix.shape == (3, 3)
        assert meta.ub_matrix.dtype == np.float64

    def test_image_paths_coerced_to_path_objects(self):
        meta = self._make(image_paths=["/tmp/img_001.edf", "/tmp/img_002.edf"])
        assert all(isinstance(p, Path) for p in meta.image_paths)
        assert meta.image_paths[0] == Path("/tmp/img_001.edf")

    def test_image_paths_already_path_objects_unchanged(self):
        paths = [Path("/tmp/a.edf"), Path("/tmp/b.edf")]
        meta = self._make(image_paths=paths)
        assert meta.image_paths == paths

    def test_h5_path_coerced_from_string(self):
        meta = self._make(h5_path="/data/scan001_master.h5")
        assert isinstance(meta.h5_path, Path)
        assert meta.h5_path == Path("/data/scan001_master.h5")

    def test_h5_path_none_stays_none(self):
        meta = self._make(h5_path=None)
        assert meta.h5_path is None

    def test_extra_dict_stored(self):
        meta = self._make(extra={"beamline": "BL7-2", "ring_current": 499.8})
        assert meta.extra["beamline"] == "BL7-2"
        assert meta.extra["ring_current"] == pytest.approx(499.8)

    def test_source_and_scan_type(self):
        meta = self._make(source="spec", scan_type="ascan del 0 1 100 1.0")
        assert meta.source == "spec"
        assert meta.scan_type == "ascan del 0 1 100 1.0"

    def test_sample_name(self):
        meta = self._make(sample_name="LSMO_on_STO")
        assert meta.sample_name == "LSMO_on_STO"

    def test_slots_no_dict(self):
        meta = self._make()
        assert not hasattr(meta, "__dict__")

    def test_slots_rejects_new_attribute(self):
        meta = self._make()
        with pytest.raises(AttributeError):
            meta.nonexistent = "oops"  # type: ignore[attr-defined]
