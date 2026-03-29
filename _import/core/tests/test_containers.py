"""Tests for ssrl_xrd_tools.core.containers."""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
    _pyfai_unit_to_nexus,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_PONI = PONI(
    dist=0.1234,
    poni1=0.05,
    poni2=0.06,
    rot1=0.001,
    rot2=0.002,
    rot3=0.0,
    wavelength=1.0e-10,
    detector="Pilatus300k",
)

# Minimal .poni YAML (pyFAI v2 format)
_PONI_YAML = textwrap.dedent("""\
    poni_version: 2
    Detector: Pilatus300k
    Detector_config: {}
    Distance: 0.1234
    Poni1: 0.05
    Poni2: 0.06
    Rot1: 0.001
    Rot2: 0.002
    Rot3: 0.0
    Wavelength: 1.0e-10
""")


@pytest.fixture()
def poni_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.poni"
    p.write_text(_PONI_YAML)
    return p


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

def test_to_dict_keys():
    d = SAMPLE_PONI.to_dict()
    expected_keys = {"dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength", "detector"}
    assert set(d.keys()) == expected_keys


def test_to_dict_values():
    d = SAMPLE_PONI.to_dict()
    assert d["dist"] == pytest.approx(0.1234)
    assert d["poni1"] == pytest.approx(0.05)
    assert d["poni2"] == pytest.approx(0.06)
    assert d["rot1"] == pytest.approx(0.001)
    assert d["rot2"] == pytest.approx(0.002)
    assert d["rot3"] == pytest.approx(0.0)
    assert d["wavelength"] == pytest.approx(1.0e-10)
    assert d["detector"] == "Pilatus300k"


# ---------------------------------------------------------------------------
# from_dict — lowercase keys
# ---------------------------------------------------------------------------

def test_from_dict_lowercase_roundtrip():
    d = SAMPLE_PONI.to_dict()
    poni2 = PONI.from_dict(d)
    assert poni2.dist == pytest.approx(SAMPLE_PONI.dist)
    assert poni2.poni1 == pytest.approx(SAMPLE_PONI.poni1)
    assert poni2.poni2 == pytest.approx(SAMPLE_PONI.poni2)
    assert poni2.rot1 == pytest.approx(SAMPLE_PONI.rot1)
    assert poni2.rot2 == pytest.approx(SAMPLE_PONI.rot2)
    assert poni2.rot3 == pytest.approx(SAMPLE_PONI.rot3)
    assert poni2.wavelength == pytest.approx(SAMPLE_PONI.wavelength)
    assert poni2.detector == SAMPLE_PONI.detector


def test_from_dict_capitalised_keys():
    d = {
        "Distance": 0.1234,
        "Poni1": 0.05,
        "Poni2": 0.06,
        "Rot1": 0.001,
        "Rot2": 0.002,
        "Rot3": 0.0,
        "Wavelength": 1.0e-10,
        "Detector": "Pilatus300k",
    }
    poni = PONI.from_dict(d)
    assert poni.dist == pytest.approx(0.1234)
    assert poni.detector == "Pilatus300k"


def test_from_dict_unknown_keys_ignored():
    d = {"dist": 0.1, "poni1": 0.0, "poni2": 0.0, "unknown_key": 99}
    poni = PONI.from_dict(d)
    assert poni.dist == pytest.approx(0.1)


def test_from_dict_wavelength_string():
    d = {"dist": 0.1, "poni1": 0.0, "poni2": 0.0, "Wavelength": "6.2e-11"}
    poni = PONI.from_dict(d)
    assert poni.wavelength == pytest.approx(6.2e-11)


def test_from_dict_missing_fields_default_to_zero():
    poni = PONI.from_dict({"dist": 0.5})
    assert poni.dist == pytest.approx(0.5)
    assert poni.rot1 == pytest.approx(0.0)
    assert poni.wavelength == pytest.approx(0.0)
    assert poni.detector == ""


# ---------------------------------------------------------------------------
# from_poni_file
# ---------------------------------------------------------------------------

def test_from_poni_file(poni_file: Path):
    poni = PONI.from_poni_file(poni_file)
    assert poni.dist == pytest.approx(0.1234)
    assert poni.poni1 == pytest.approx(0.05)
    assert poni.wavelength == pytest.approx(1.0e-10)
    assert poni.detector == "Pilatus300k"


def test_from_poni_file_str_path(poni_file: Path):
    poni = PONI.from_poni_file(str(poni_file))
    assert poni.dist == pytest.approx(0.1234)


def test_from_poni_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        PONI.from_poni_file(tmp_path / "nonexistent.poni")


# ---------------------------------------------------------------------------
# to_poni_file + round-trip
# ---------------------------------------------------------------------------

def test_to_poni_file_roundtrip(tmp_path: Path):
    out = tmp_path / "out.poni"
    SAMPLE_PONI.to_poni_file(out)
    assert out.exists()
    restored = PONI.from_poni_file(out)
    assert restored.dist == pytest.approx(SAMPLE_PONI.dist)
    assert restored.poni1 == pytest.approx(SAMPLE_PONI.poni1)
    assert restored.poni2 == pytest.approx(SAMPLE_PONI.poni2)
    assert restored.rot1 == pytest.approx(SAMPLE_PONI.rot1)
    assert restored.rot2 == pytest.approx(SAMPLE_PONI.rot2)
    assert restored.rot3 == pytest.approx(SAMPLE_PONI.rot3)
    assert restored.wavelength == pytest.approx(SAMPLE_PONI.wavelength)
    assert restored.detector == SAMPLE_PONI.detector


def test_to_poni_file_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "subdir" / "nested" / "out.poni"
    SAMPLE_PONI.to_poni_file(out)
    assert out.exists()


def test_to_poni_file_str_path(tmp_path: Path):
    out = tmp_path / "str_path.poni"
    SAMPLE_PONI.to_poni_file(str(out))
    assert out.exists()


# ===========================================================================
# IntegrationResult1D
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def r1d_q():
    q = np.linspace(0.5, 5.0, 200)
    I = np.random.default_rng(0).random(200)
    sig = np.sqrt(I)
    return IntegrationResult1D(radial=q, intensity=I, sigma=sig, unit="q_A^-1")


@pytest.fixture()
def r1d_tth():
    tth = np.linspace(5.0, 60.0, 200)
    I = np.ones(200)
    return IntegrationResult1D(radial=tth, intensity=I, unit="2th_deg")


# ---------------------------------------------------------------------------
# Basic construction / validation (complement test_core.py)
# ---------------------------------------------------------------------------

class TestIntegrationResult1DNew:
    def test_default_azimuthal_unit_absent(self):
        """1D result has no azimuthal_unit field."""
        r = IntegrationResult1D(radial=np.ones(10), intensity=np.ones(10))
        assert not hasattr(r, "azimuthal_unit")

    def test_unit_stored(self):
        r = IntegrationResult1D(
            radial=np.ones(10), intensity=np.ones(10), unit="qoop_A^-1"
        )
        assert r.unit == "qoop_A^-1"

    # ------------------------------------------------------------------
    # to_unit — scale conversions
    # ------------------------------------------------------------------

    def test_to_unit_same_unit_returns_copy(self, r1d_q):
        r2 = r1d_q.to_unit("q_A^-1")
        np.testing.assert_array_equal(r2.radial, r1d_q.radial)
        assert r2.unit == "q_A^-1"
        assert r2 is not r1d_q

    def test_to_unit_q_a_to_q_nm(self, r1d_q):
        r2 = r1d_q.to_unit("q_nm^-1")
        np.testing.assert_allclose(r2.radial, r1d_q.radial * 10.0)
        assert r2.unit == "q_nm^-1"

    def test_to_unit_q_nm_to_q_a(self, r1d_q):
        r_nm = r1d_q.to_unit("q_nm^-1")
        r_a = r_nm.to_unit("q_A^-1")
        np.testing.assert_allclose(r_a.radial, r1d_q.radial, rtol=1e-12)

    def test_to_unit_gi_scale(self):
        r = IntegrationResult1D(
            radial=np.array([1.0, 2.0, 3.0]), intensity=np.ones(3), unit="qoop_A^-1"
        )
        r2 = r.to_unit("qoop_nm^-1")
        np.testing.assert_allclose(r2.radial, [10.0, 20.0, 30.0])
        assert r2.unit == "qoop_nm^-1"

    def test_to_unit_cross_axis_gi_raises(self):
        r = IntegrationResult1D(
            radial=np.ones(5), intensity=np.ones(5), unit="qip_A^-1"
        )
        with pytest.raises(ValueError, match="Cross-axis|not supported"):
            r.to_unit("qoop_A^-1")

    def test_to_unit_tth_deg_to_rad(self, r1d_tth):
        r2 = r1d_tth.to_unit("2th_rad")
        np.testing.assert_allclose(r2.radial, np.deg2rad(r1d_tth.radial))
        assert r2.unit == "2th_rad"

    def test_to_unit_2th_to_q_requires_wavelength(self, r1d_tth):
        with pytest.raises(ValueError, match="wavelength"):
            r1d_tth.to_unit("q_A^-1")

    def test_to_unit_2th_to_q_roundtrip(self, r1d_tth):
        wl_A = 1.0  # 1 Å
        r_q = r1d_tth.to_unit("q_A^-1", wavelength=wl_A)
        assert r_q.unit == "q_A^-1"
        r_back = r_q.to_unit("2th_deg", wavelength=wl_A)
        np.testing.assert_allclose(r_back.radial, r1d_tth.radial, rtol=1e-10)

    def test_to_unit_sigma_preserved(self, r1d_q):
        r2 = r1d_q.to_unit("q_nm^-1")
        assert r2.sigma is not None
        np.testing.assert_allclose(r2.sigma, r1d_q.sigma)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def test_add_intensity(self, r1d_q):
        r2 = r1d_q + r1d_q
        np.testing.assert_allclose(r2.intensity, 2 * r1d_q.intensity)
        assert r2.unit == r1d_q.unit

    def test_add_sigma_propagation(self, r1d_q):
        r2 = r1d_q + r1d_q
        expected_sigma = np.sqrt(2) * r1d_q.sigma
        np.testing.assert_allclose(r2.sigma, expected_sigma)

    def test_add_no_sigma_gives_none(self, r1d_tth):
        r2 = r1d_tth + r1d_tth
        assert r2.sigma is None

    def test_add_mixed_sigma_gives_none(self, r1d_q, r1d_tth):
        # Different units → should raise before sigma question arises
        with pytest.raises(ValueError, match="Unit mismatch"):
            r1d_q + r1d_tth

    def test_add_unit_mismatch_raises(self, r1d_q, r1d_tth):
        with pytest.raises(ValueError, match="Unit mismatch"):
            r1d_q + r1d_tth

    def test_sub_intensity(self, r1d_q):
        r_zero = r1d_q - r1d_q
        np.testing.assert_allclose(r_zero.intensity, 0.0)

    def test_sub_sigma_propagation(self, r1d_q):
        r2 = r1d_q - r1d_q
        expected_sigma = np.sqrt(2) * r1d_q.sigma
        np.testing.assert_allclose(r2.sigma, expected_sigma)

    def test_mul_scales_intensity(self, r1d_q):
        r2 = r1d_q * 3.0
        np.testing.assert_allclose(r2.intensity, 3.0 * r1d_q.intensity)

    def test_mul_scales_sigma(self, r1d_q):
        r2 = r1d_q * 2.5
        np.testing.assert_allclose(r2.sigma, 2.5 * r1d_q.sigma)

    def test_mul_negative_scalar_abs_sigma(self, r1d_q):
        r2 = r1d_q * (-2.0)
        np.testing.assert_allclose(r2.intensity, -2.0 * r1d_q.intensity)
        np.testing.assert_allclose(r2.sigma, 2.0 * r1d_q.sigma)

    def test_rmul(self, r1d_q):
        r2 = 2.0 * r1d_q
        np.testing.assert_allclose(r2.intensity, 2.0 * r1d_q.intensity)

    # ------------------------------------------------------------------
    # from_pyfai
    # ------------------------------------------------------------------

    def test_from_pyfai_standard(self):
        fake = types.SimpleNamespace(
            radial=np.linspace(0, 5, 100),
            intensity=np.ones(100),
            sigma=np.full(100, 0.1),
            unit="q_A^-1",
        )
        r = IntegrationResult1D.from_pyfai(fake)
        assert r.unit == "q_A^-1"
        assert r.radial.shape == (100,)
        assert r.sigma is not None

    def test_from_pyfai_no_sigma(self):
        fake = types.SimpleNamespace(
            radial=np.ones(50),
            intensity=np.ones(50),
            sigma=None,
            unit="2th_deg",
        )
        r = IntegrationResult1D.from_pyfai(fake)
        assert r.sigma is None

    def test_from_pyfai_unit_override(self):
        fake = types.SimpleNamespace(
            radial=np.ones(50),
            intensity=np.ones(50),
            sigma=None,
            unit="q_A^-1",
        )
        r = IntegrationResult1D.from_pyfai(fake, unit="qoop_A^-1")
        assert r.unit == "qoop_A^-1"

    # ------------------------------------------------------------------
    # HDF5 round-trip
    # ------------------------------------------------------------------

    def test_to_from_hdf5_roundtrip(self, r1d_q, tmp_path):
        import h5py
        p = tmp_path / "test.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("result")
            r1d_q.to_hdf5(grp)
        with h5py.File(p, "r") as f:
            r2 = IntegrationResult1D.from_hdf5(f["result"])
        np.testing.assert_allclose(r2.radial, r1d_q.radial)
        np.testing.assert_allclose(r2.intensity, r1d_q.intensity)
        np.testing.assert_allclose(r2.sigma, r1d_q.sigma)
        assert r2.unit == r1d_q.unit

    def test_to_hdf5_no_sigma(self, r1d_tth, tmp_path):
        import h5py
        p = tmp_path / "test.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("result")
            r1d_tth.to_hdf5(grp)
        with h5py.File(p, "r") as f:
            assert "sigma" not in f["result"]
            r2 = IntegrationResult1D.from_hdf5(f["result"])
        assert r2.sigma is None

    def test_from_hdf5_missing_unit_defaults(self, tmp_path):
        import h5py
        q = np.linspace(0, 1, 20)
        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("r")
            grp.create_dataset("radial", data=q)
            grp.create_dataset("intensity", data=q)
            # no "unit" attr
        with h5py.File(p, "r") as f:
            r = IntegrationResult1D.from_hdf5(f["r"])
        assert r.unit == "2th_deg"

    # ------------------------------------------------------------------
    # to_nexus
    # ------------------------------------------------------------------

    def test_to_nexus_attributes(self, r1d_q, tmp_path):
        import h5py
        p = tmp_path / "nx.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("nxdata")
            r1d_q.to_nexus(grp)
        with h5py.File(p, "r") as f:
            grp = f["nxdata"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            assert list(grp.attrs["axes"]) == ["radial"]
            assert grp["radial"].attrs["units"] == "angstrom^-1"
            assert grp["radial"].attrs["long_name"] == "Q"
            assert "intensity" in grp

    def test_to_nexus_sigma_written(self, r1d_q, tmp_path):
        import h5py
        p = tmp_path / "nx.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("nxdata")
            r1d_q.to_nexus(grp)
        with h5py.File(p, "r") as f:
            assert "sigma" in f["nxdata"]
            assert f["nxdata"]["sigma"].attrs["long_name"] == "Uncertainty"

    def test_to_nexus_no_sigma_not_written(self, r1d_tth, tmp_path):
        import h5py
        p = tmp_path / "nx.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("nxdata")
            r1d_tth.to_nexus(grp)
        with h5py.File(p, "r") as f:
            assert "sigma" not in f["nxdata"]


# ===========================================================================
# IntegrationResult2D
# ===========================================================================

@pytest.fixture()
def r2d_q():
    radial = np.linspace(0.5, 5.0, 100)
    azimuthal = np.linspace(-180.0, 180.0, 72)
    rng = np.random.default_rng(1)
    intensity = rng.random((100, 72))
    sigma = np.sqrt(intensity)
    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity,
        sigma=sigma,
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )


@pytest.fixture()
def r2d_gi():
    """GI qip/qoop result."""
    radial = np.linspace(0.0, 3.0, 60)
    azimuthal = np.linspace(0.0, 2.0, 40)
    intensity = np.ones((60, 40))
    return IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity,
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
    )


class TestIntegrationResult2DNew:
    def test_azimuthal_unit_default(self):
        r = IntegrationResult2D(
            radial=np.ones(10),
            azimuthal=np.ones(5),
            intensity=np.ones((10, 5)),
        )
        assert r.azimuthal_unit == "chi_deg"

    def test_azimuthal_unit_stored(self, r2d_gi):
        assert r2d_gi.azimuthal_unit == "qoop_A^-1"

    def test_slots_no_dict(self, r2d_q):
        assert not hasattr(r2d_q, "__dict__")

    # ------------------------------------------------------------------
    # to_unit (radial)
    # ------------------------------------------------------------------

    def test_to_unit_q_a_to_nm(self, r2d_q):
        r2 = r2d_q.to_unit("q_nm^-1")
        np.testing.assert_allclose(r2.radial, r2d_q.radial * 10.0)
        assert r2.unit == "q_nm^-1"
        assert r2.azimuthal_unit == "chi_deg"  # unchanged

    def test_to_unit_preserves_azimuthal(self, r2d_q):
        r2 = r2d_q.to_unit("q_nm^-1")
        np.testing.assert_array_equal(r2.azimuthal, r2d_q.azimuthal)

    def test_to_unit_gi_radial(self, r2d_gi):
        r2 = r2d_gi.to_unit("qip_nm^-1")
        np.testing.assert_allclose(r2.radial, r2d_gi.radial * 10.0)
        assert r2.unit == "qip_nm^-1"
        assert r2.azimuthal_unit == "qoop_A^-1"  # unchanged

    # ------------------------------------------------------------------
    # to_azimuthal_unit
    # ------------------------------------------------------------------

    def test_to_azimuthal_unit_chi_deg_to_rad(self, r2d_q):
        r2 = r2d_q.to_azimuthal_unit("chi_rad")
        np.testing.assert_allclose(r2.azimuthal, np.deg2rad(r2d_q.azimuthal))
        assert r2.azimuthal_unit == "chi_rad"
        assert r2.unit == r2d_q.unit  # unchanged

    def test_to_azimuthal_unit_roundtrip(self, r2d_q):
        r2 = r2d_q.to_azimuthal_unit("chi_rad").to_azimuthal_unit("chi_deg")
        np.testing.assert_allclose(r2.azimuthal, r2d_q.azimuthal, rtol=1e-12)

    def test_to_azimuthal_unit_gi_oop(self, r2d_gi):
        r2 = r2d_gi.to_azimuthal_unit("qoop_nm^-1")
        np.testing.assert_allclose(r2.azimuthal, r2d_gi.azimuthal * 10.0)
        assert r2.azimuthal_unit == "qoop_nm^-1"

    def test_to_azimuthal_unit_cross_axis_raises(self, r2d_gi):
        with pytest.raises(ValueError):
            r2d_gi.to_azimuthal_unit("qip_A^-1")  # oop → ip not allowed

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def test_add_intensity(self, r2d_q):
        r2 = r2d_q + r2d_q
        np.testing.assert_allclose(r2.intensity, 2 * r2d_q.intensity)

    def test_add_sigma_propagation(self, r2d_q):
        r2 = r2d_q + r2d_q
        expected = np.sqrt(2) * r2d_q.sigma
        np.testing.assert_allclose(r2.sigma, expected)

    def test_add_unit_mismatch_raises(self, r2d_q, r2d_gi):
        with pytest.raises(ValueError, match="Unit mismatch"):
            r2d_q + r2d_gi

    def test_sub_gives_zero(self, r2d_q):
        r2 = r2d_q - r2d_q
        np.testing.assert_allclose(r2.intensity, 0.0)

    def test_mul_scales(self, r2d_q):
        r2 = r2d_q * 3.0
        np.testing.assert_allclose(r2.intensity, 3.0 * r2d_q.intensity)
        np.testing.assert_allclose(r2.sigma, 3.0 * r2d_q.sigma)

    def test_rmul(self, r2d_q):
        r2 = 2.0 * r2d_q
        np.testing.assert_allclose(r2.intensity, 2.0 * r2d_q.intensity)

    # ------------------------------------------------------------------
    # from_pyfai
    # ------------------------------------------------------------------

    def test_from_pyfai_standard(self):
        nrad, nazi = 50, 30
        fake = types.SimpleNamespace(
            radial=np.linspace(0, 5, nrad),
            azimuthal=np.linspace(-180, 180, nazi),
            intensity=np.ones((nazi, nrad)),  # pyFAI shape: (npt_azim, npt_rad)
            sigma=None,
            unit="q_A^-1",
        )
        r = IntegrationResult2D.from_pyfai(fake)
        assert r.intensity.shape == (nrad, nazi)  # transposed
        assert r.unit == "q_A^-1"
        assert r.azimuthal_unit == "chi_deg"

    def test_from_pyfai_tuple_unit(self):
        nrad, nazi = 40, 20
        fake = types.SimpleNamespace(
            radial=np.ones(nrad),
            azimuthal=np.ones(nazi),
            intensity=np.ones((nazi, nrad)),
            sigma=None,
            unit=("q_A^-1", "chi_deg"),
        )
        r = IntegrationResult2D.from_pyfai(fake)
        assert r.unit == "q_A^-1"
        assert r.azimuthal_unit == "chi_deg"

    def test_from_pyfai_gi_result(self):
        nip, noop = 50, 40
        fake = types.SimpleNamespace(
            inplane=np.linspace(0, 3, nip),
            outofplane=np.linspace(0, 2, noop),
            intensity=np.ones((noop, nip)),  # pyFAI shape: (npt_oop, npt_ip)
            sigma=None,
            ip_unit="qip_A^-1",
            oop_unit="qoop_A^-1",
            unit=None,
        )
        r = IntegrationResult2D.from_pyfai(fake)
        assert r.intensity.shape == (nip, noop)
        assert r.unit == "qip_A^-1"
        assert r.azimuthal_unit == "qoop_A^-1"

    def test_from_pyfai_unit_override(self):
        nrad, nazi = 30, 20
        fake = types.SimpleNamespace(
            radial=np.ones(nrad),
            azimuthal=np.ones(nazi),
            intensity=np.ones((nazi, nrad)),
            sigma=None,
            unit="2th_deg",
        )
        r = IntegrationResult2D.from_pyfai(
            fake, unit="qip_A^-1", azimuthal_unit="qoop_A^-1"
        )
        assert r.unit == "qip_A^-1"
        assert r.azimuthal_unit == "qoop_A^-1"

    def test_from_pyfai_invalid_raises(self):
        fake = types.SimpleNamespace(intensity=np.ones((10, 5)), sigma=None, unit=None)
        with pytest.raises(ValueError, match="Cannot parse pyFAI result"):
            IntegrationResult2D.from_pyfai(fake)

    # ------------------------------------------------------------------
    # extract_1d
    # ------------------------------------------------------------------

    def test_extract_1d_radial_sum(self, r2d_q):
        r1 = r2d_q.extract_1d(axis="radial")
        assert r1.unit == r2d_q.unit
        assert r1.radial.shape == (100,)
        np.testing.assert_allclose(r1.intensity, r2d_q.intensity.sum(axis=1))

    def test_extract_1d_azimuthal_sum(self, r2d_q):
        r1 = r2d_q.extract_1d(axis="azimuthal")
        assert r1.unit == r2d_q.azimuthal_unit
        assert r1.radial.shape == (72,)
        np.testing.assert_allclose(r1.intensity, r2d_q.intensity.sum(axis=0))

    def test_extract_1d_radial_index(self, r2d_q):
        r1 = r2d_q.extract_1d(axis="radial", index=5)
        np.testing.assert_array_equal(r1.intensity, r2d_q.intensity[:, 5])

    def test_extract_1d_azimuthal_index(self, r2d_q):
        r1 = r2d_q.extract_1d(axis="azimuthal", index=10)
        np.testing.assert_array_equal(r1.intensity, r2d_q.intensity[10, :])

    def test_extract_1d_radial_range(self, r2d_q):
        # sum chi in [-30, 30]
        mask = (r2d_q.azimuthal >= -30) & (r2d_q.azimuthal <= 30)
        r1 = r2d_q.extract_1d(axis="radial", range_=(-30.0, 30.0))
        expected = r2d_q.intensity[:, mask].sum(axis=1)
        np.testing.assert_allclose(r1.intensity, expected)

    def test_extract_1d_sigma_propagated(self, r2d_q):
        r1 = r2d_q.extract_1d(axis="radial")
        expected_sigma = np.sqrt((r2d_q.sigma ** 2).sum(axis=1))
        np.testing.assert_allclose(r1.sigma, expected_sigma)

    def test_extract_1d_invalid_axis_raises(self, r2d_q):
        with pytest.raises(ValueError, match="axis must be"):
            r2d_q.extract_1d(axis="diagonal")

    # ------------------------------------------------------------------
    # HDF5 round-trip
    # ------------------------------------------------------------------

    def test_to_from_hdf5_roundtrip(self, r2d_q, tmp_path):
        import h5py
        p = tmp_path / "test.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("result")
            r2d_q.to_hdf5(grp)
        with h5py.File(p, "r") as f:
            r2 = IntegrationResult2D.from_hdf5(f["result"])
        np.testing.assert_allclose(r2.radial, r2d_q.radial)
        np.testing.assert_allclose(r2.azimuthal, r2d_q.azimuthal)
        np.testing.assert_allclose(r2.intensity, r2d_q.intensity)
        np.testing.assert_allclose(r2.sigma, r2d_q.sigma)
        assert r2.unit == r2d_q.unit
        assert r2.azimuthal_unit == r2d_q.azimuthal_unit

    def test_to_hdf5_gi_units_preserved(self, r2d_gi, tmp_path):
        import h5py
        p = tmp_path / "gi.h5"
        with h5py.File(p, "w") as f:
            r2d_gi.to_hdf5(f.create_group("r"))
        with h5py.File(p, "r") as f:
            r2 = IntegrationResult2D.from_hdf5(f["r"])
        assert r2.unit == "qip_A^-1"
        assert r2.azimuthal_unit == "qoop_A^-1"

    def test_from_hdf5_missing_azimuthal_unit_defaults(self, tmp_path):
        import h5py
        p = tmp_path / "t.h5"
        radial = np.ones(10)
        azimuthal = np.ones(5)
        with h5py.File(p, "w") as f:
            grp = f.create_group("r")
            grp.create_dataset("radial", data=radial)
            grp.create_dataset("azimuthal", data=azimuthal)
            grp.create_dataset("intensity", data=np.ones((10, 5)))
            grp.attrs["unit"] = "q_A^-1"
            # azimuthal_unit attr absent — should default to "chi_deg"
        with h5py.File(p, "r") as f:
            r = IntegrationResult2D.from_hdf5(f["r"])
        assert r.azimuthal_unit == "chi_deg"

    # ------------------------------------------------------------------
    # to_nexus
    # ------------------------------------------------------------------

    def test_to_nexus_attributes(self, r2d_q, tmp_path):
        import h5py
        p = tmp_path / "nx.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("nxdata")
            r2d_q.to_nexus(grp)
        with h5py.File(p, "r") as f:
            grp = f["nxdata"]
            assert grp.attrs["NX_class"] == "NXdata"
            assert grp.attrs["signal"] == "intensity"
            axes = list(grp.attrs["axes"])
            assert axes == ["radial", "azimuthal"]
            assert grp["radial"].attrs["units"] == "angstrom^-1"
            assert grp["radial"].attrs["long_name"] == "Q"
            assert grp["azimuthal"].attrs["units"] == "degrees"
            assert grp["azimuthal"].attrs["long_name"] == "Chi"
            assert "intensity" in grp

    def test_to_nexus_gi_units(self, r2d_gi, tmp_path):
        import h5py
        p = tmp_path / "gi_nx.h5"
        with h5py.File(p, "w") as f:
            grp = f.create_group("nxdata")
            r2d_gi.to_nexus(grp)
        with h5py.File(p, "r") as f:
            grp = f["nxdata"]
            assert grp["radial"].attrs["long_name"] == "Q_ip"
            assert grp["azimuthal"].attrs["long_name"] == "Q_oop"

    def test_to_nexus_sigma_written_when_present(self, r2d_q, tmp_path):
        import h5py
        p = tmp_path / "nx.h5"
        with h5py.File(p, "w") as f:
            r2d_q.to_nexus(f.create_group("g"))
        with h5py.File(p, "r") as f:
            assert "sigma" in f["g"]


# ===========================================================================
# _pyfai_unit_to_nexus helper
# ===========================================================================

class TestPyfaiUnitToNexus:
    @pytest.mark.parametrize("unit, expected_units, expected_long", [
        ("q_A^-1", "angstrom^-1", "Q"),
        ("q_nm^-1", "nm^-1", "Q"),
        ("2th_deg", "degrees", "2Theta"),
        ("2th_rad", "radians", "2Theta"),
        ("chi_deg", "degrees", "Chi"),
        ("chi_rad", "radians", "Chi"),
        ("qip_A^-1", "angstrom^-1", "Q_ip"),
        ("qip_nm^-1", "nm^-1", "Q_ip"),
        ("qoop_A^-1", "angstrom^-1", "Q_oop"),
        ("qoop_nm^-1", "nm^-1", "Q_oop"),
        ("qtot_A^-1", "angstrom^-1", "Q_total"),
        ("chigi_deg", "degrees", "Chi_GI"),
        ("chigi_rad", "radians", "Chi_GI"),
    ])
    def test_known_units(self, unit, expected_units, expected_long):
        nexus_u, long = _pyfai_unit_to_nexus(unit)
        assert nexus_u == expected_units
        assert long == expected_long

    def test_unknown_unit_fallback(self):
        nexus_u, long = _pyfai_unit_to_nexus("custom_unit")
        assert nexus_u == "a.u."
        assert long == "custom_unit"
