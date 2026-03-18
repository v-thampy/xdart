"""Tests for ssrl_xrd_tools.integrate.calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.integrate.calibration import (
    get_detector,
    get_detector_mask,
    load_poni,
    poni_to_fiber_integrator,
    poni_to_integrator,
    save_poni,
)

# ---------------------------------------------------------------------------
# Module-level: check whether FiberIntegrator is available in this install
# ---------------------------------------------------------------------------

def _fiber_integrator_available() -> bool:
    try:
        from pyFAI.integrator.fiber import FiberIntegrator  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_FIBER_INTEGRATOR = _fiber_integrator_available()


# ---------------------------------------------------------------------------
# load_poni / save_poni
# ---------------------------------------------------------------------------

class TestLoadSavePoni:
    def test_roundtrip_geometry(self, poni_fixture: PONI, tmp_poni_file: Path):
        """Numerical geometry fields survive a save→load round-trip."""
        loaded = load_poni(tmp_poni_file)
        assert loaded.dist == pytest.approx(poni_fixture.dist, rel=1e-6)
        assert loaded.poni1 == pytest.approx(poni_fixture.poni1, rel=1e-6)
        assert loaded.poni2 == pytest.approx(poni_fixture.poni2, rel=1e-6)
        assert loaded.rot1 == pytest.approx(poni_fixture.rot1, abs=1e-10)
        assert loaded.rot2 == pytest.approx(poni_fixture.rot2, abs=1e-10)
        assert loaded.rot3 == pytest.approx(poni_fixture.rot3, abs=1e-10)

    def test_roundtrip_wavelength(self, poni_fixture: PONI, tmp_poni_file: Path):
        """Wavelength (stored in metres) survives a save→load round-trip."""
        loaded = load_poni(tmp_poni_file)
        assert loaded.wavelength == pytest.approx(poni_fixture.wavelength, rel=1e-6)

    def test_roundtrip_returns_poni_instance(self, tmp_poni_file: Path):
        loaded = load_poni(tmp_poni_file)
        assert isinstance(loaded, PONI)

    def test_roundtrip_detector_name_preserved(self, poni_fixture: PONI, tmp_poni_file: Path):
        """
        pyFAI normalises the detector name to its canonical long form on
        save/load (e.g. ``"eiger4m"`` → ``"Eiger 4M"``).  Verify that the
        original name is at least contained in the loaded name
        case-insensitively.
        """
        loaded = load_poni(tmp_poni_file)
        # Both should refer to the same detector family
        assert poni_fixture.detector.replace(" ", "").lower() in \
               loaded.detector.replace(" ", "").lower()

    def test_save_creates_file(self, poni_fixture: PONI, tmp_path: Path):
        out = tmp_path / "out.poni"
        assert not out.exists()
        save_poni(poni_fixture, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_accepts_str_path(self, poni_fixture: PONI, tmp_path: Path):
        out = str(tmp_path / "str_path.poni")
        save_poni(poni_fixture, out)
        assert Path(out).exists()

    def test_save_creates_parent_dirs(self, poni_fixture: PONI, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "dir" / "cal.poni"
        save_poni(poni_fixture, out)
        assert out.exists()

    def test_load_accepts_str_path(self, tmp_poni_file: Path):
        loaded = load_poni(str(tmp_poni_file))
        assert isinstance(loaded, PONI)


# ---------------------------------------------------------------------------
# poni_to_integrator
# ---------------------------------------------------------------------------

class TestPoniToIntegrator:
    def test_returns_azimuthal_integrator(self, poni_fixture: PONI):
        ai = poni_to_integrator(poni_fixture)
        assert isinstance(ai, AzimuthalIntegrator)

    def test_geometry_fields_match(self, poni_fixture: PONI):
        ai = poni_to_integrator(poni_fixture)
        assert ai.dist == pytest.approx(poni_fixture.dist, rel=1e-10)
        assert ai.poni1 == pytest.approx(poni_fixture.poni1, rel=1e-10)
        assert ai.poni2 == pytest.approx(poni_fixture.poni2, rel=1e-10)
        assert ai.rot1 == pytest.approx(poni_fixture.rot1, abs=1e-12)
        assert ai.rot2 == pytest.approx(poni_fixture.rot2, abs=1e-12)
        assert ai.rot3 == pytest.approx(poni_fixture.rot3, abs=1e-12)

    def test_wavelength_matches(self, poni_fixture: PONI):
        ai = poni_to_integrator(poni_fixture)
        assert ai.wavelength == pytest.approx(poni_fixture.wavelength, rel=1e-10)

    def test_detector_set_when_name_given(self, poni_fixture: PONI):
        """When poni.detector is a non-empty string, ai.detector must be set."""
        assert poni_fixture.detector != ""
        ai = poni_to_integrator(poni_fixture)
        assert ai.detector is not None

    def test_no_detector_when_empty_string(self):
        poni = PONI(dist=0.3, poni1=0.0, poni2=0.0, wavelength=1e-10, detector="")
        ai = poni_to_integrator(poni)
        # No detector specified: pyFAI uses a generic detector placeholder
        # — we just check the call doesn't raise
        assert isinstance(ai, AzimuthalIntegrator)

    def test_ai_fixture_matches_poni_fixture(self, poni_fixture: PONI, ai_fixture):
        """The session-scoped ai_fixture must be consistent with poni_fixture."""
        assert ai_fixture.dist == pytest.approx(poni_fixture.dist, rel=1e-10)
        assert ai_fixture.poni1 == pytest.approx(poni_fixture.poni1, rel=1e-10)
        assert ai_fixture.poni2 == pytest.approx(poni_fixture.poni2, rel=1e-10)


# ---------------------------------------------------------------------------
# get_detector
# ---------------------------------------------------------------------------

class TestGetDetector:
    def test_known_detector_returns_object(self):
        det = get_detector("Pilatus300k")
        assert det is not None

    def test_known_detector_correct_shape(self):
        det = get_detector("Pilatus300k")
        assert det.max_shape == (619, 487)

    def test_eiger4m_shape(self):
        det = get_detector("eiger4m")
        assert det.max_shape == (2167, 2070)

    def test_unknown_detector_raises_value_error(self):
        with pytest.raises(ValueError, match="NotADetector"):
            get_detector("NotADetector")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_detector("")


# ---------------------------------------------------------------------------
# get_detector_mask
# ---------------------------------------------------------------------------

class TestGetDetectorMask:
    def test_known_detector_returns_ndarray(self):
        mask = get_detector_mask("eiger4m")
        # Some detectors have no mask → None is a valid return.
        # Eiger 4M has a module gap mask so it should be an array.
        assert mask is not None
        assert isinstance(mask, np.ndarray)

    def test_known_detector_correct_shape(self):
        mask = get_detector_mask("eiger4m")
        assert mask is not None
        assert mask.shape == (2167, 2070)

    def test_mask_is_boolean_like(self):
        """Mask values must be 0 (good) or non-zero (bad); castable to bool."""
        mask = get_detector_mask("eiger4m")
        assert mask is not None
        bool_mask = mask.astype(bool)
        assert bool_mask.dtype == bool
        # Module gaps exist → at least some pixels should be masked
        assert bool_mask.any()

    def test_pilatus300k_mask_shape(self):
        mask = get_detector_mask("Pilatus300k")
        # Pilatus 300k may or may not have a mask; if present, check shape
        if mask is not None:
            assert mask.shape == (619, 487)

    def test_unknown_detector_returns_none(self):
        """Unknown detector names should return None gracefully (no exception)."""
        mask = get_detector_mask("NotADetector_XYZ_9999")
        assert mask is None


# ---------------------------------------------------------------------------
# poni_to_fiber_integrator
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_FIBER_INTEGRATOR,
    reason="FiberIntegrator requires pyFAI >= 2025.01",
)
class TestPoniToFiberIntegrator:
    def test_returns_fiber_integrator(self, poni_fixture: PONI):
        from pyFAI.integrator.fiber import FiberIntegrator

        fi = poni_to_fiber_integrator(poni_fixture, incident_angle=0.5)
        assert isinstance(fi, FiberIntegrator)

    def test_incident_angle_cached(self, poni_fixture: PONI):
        """create_fiber_integrator caches angles on the instance."""
        import numpy as np

        fi = poni_to_fiber_integrator(poni_fixture, incident_angle=0.5, angle_unit="deg")
        cached = getattr(fi, "_gi_incident_angle", None)
        assert cached is not None
        assert cached == pytest.approx(np.deg2rad(0.5), rel=1e-6)

    def test_tilt_angle_default_zero(self, poni_fixture: PONI):
        fi = poni_to_fiber_integrator(poni_fixture, incident_angle=0.2)
        cached_tilt = getattr(fi, "_gi_tilt_angle", None)
        assert cached_tilt is not None
        assert cached_tilt == pytest.approx(0.0, abs=1e-12)

    def test_sample_orientation_stored(self, poni_fixture: PONI):
        fi = poni_to_fiber_integrator(
            poni_fixture, incident_angle=0.3, sample_orientation=3
        )
        cached_orient = getattr(fi, "_gi_sample_orientation", None)
        assert cached_orient == 3

    def test_rad_angle_unit(self, poni_fixture: PONI):
        """When angle_unit='rad', angles are stored unchanged."""
        import numpy as np

        inc_rad = np.deg2rad(0.5)
        fi = poni_to_fiber_integrator(
            poni_fixture, incident_angle=inc_rad, angle_unit="rad"
        )
        cached = getattr(fi, "_gi_incident_angle", None)
        assert cached == pytest.approx(inc_rad, rel=1e-10)


class TestPoniToFiberIntegratorFallback:
    """Verify behaviour when FiberIntegrator is unavailable."""

    def test_raises_import_error_when_unavailable(self, poni_fixture: PONI, monkeypatch):
        """
        Simulate an environment without FiberIntegrator by patching
        create_fiber_integrator to raise ImportError.
        """
        import ssrl_xrd_tools.integrate.calibration as cal_mod

        def _fake_create(*args, **kwargs):
            raise ImportError("FiberIntegrator not available (simulated)")

        monkeypatch.setattr(
            "ssrl_xrd_tools.integrate.gid.create_fiber_integrator",
            _fake_create,
        )
        with pytest.raises(ImportError, match="FiberIntegrator"):
            poni_to_fiber_integrator(poni_fixture, incident_angle=0.5)
