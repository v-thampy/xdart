"""Tests for xrd_tools.core.geometry.

Covers the new flexible diffractometer geometry primitives used by the
xdart 0.37+ v2 NeXus writer:

* :class:`AngleMapping` linear transform + ``is_active`` semantics
* :class:`DiffractometerGeometry` convention factories
* per-frame derivation of pyFAI rotations + GI incidence angle
* JSON round-trip for NeXus persistence

The pre-existing :class:`DiffractometerConfig` (xrayutilities/RSM) is
covered in ``test_rsm.py`` — not retested here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from xrd_tools.core.geometry import (
    AngleMapping,
    DiffractometerGeometry,
)


# ---------------------------------------------------------------------------
# AngleMapping
# ---------------------------------------------------------------------------

class TestAngleMapping:
    def test_default_is_inactive(self):
        m = AngleMapping()
        assert m.source_motor == ""
        assert m.sign == 1.0
        assert m.offset == 0.0
        assert m.is_active is False

    def test_is_active_when_source_set(self):
        assert AngleMapping(source_motor="th").is_active is True

    def test_apply_identity(self):
        m = AngleMapping(source_motor="del")
        out = m.apply(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(out, [10.0, 20.0, 30.0])

    def test_apply_sign_and_offset(self):
        m = AngleMapping(source_motor="th", sign=-1.0, offset=2.0)
        out = m.apply(np.array([5.0, 10.0]))
        # -1 * [5, 10] + 2 = [-3, -8]
        np.testing.assert_allclose(out, [-3.0, -8.0])

    def test_apply_scalar_returned_as_array(self):
        m = AngleMapping(source_motor="th")
        out = m.apply(15.0)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, [15.0])

    def test_inactive_apply_returns_zeros(self):
        m = AngleMapping()  # inactive
        out = m.apply(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_frozen(self):
        m = AngleMapping(source_motor="th")
        with pytest.raises((AttributeError, Exception)):
            m.source_motor = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DiffractometerGeometry factories
# ---------------------------------------------------------------------------

class TestFactories:
    def test_two_circle_defaults(self):
        g = DiffractometerGeometry.two_circle()
        assert g.convention == "two_circle"
        assert g.rot1.source_motor == "tth"
        assert g.rot2.is_active is False
        assert g.rot3.is_active is False
        assert g.incident_angle.source_motor == "th"
        assert g.sample_motors == ("th",)
        assert g.detector_motors == ("tth",)

    def test_two_circle_with_gonchi(self):
        g = DiffractometerGeometry.two_circle(
            tth="tth", th="th", gonchi="gonchi"
        )
        assert g.sample_motors == ("th", "gonchi")
        assert g.detector_motors == ("tth",)

    def test_two_circle_custom_names(self):
        g = DiffractometerGeometry.two_circle(tth="del", th="eta")
        assert g.rot1.source_motor == "del"
        assert g.incident_angle.source_motor == "eta"

    def test_psic_defaults(self):
        g = DiffractometerGeometry.psic()
        assert g.convention == "psic"
        assert g.rot1.source_motor == "nu"
        assert g.rot2.source_motor == "del"
        assert g.rot3.is_active is False
        assert g.incident_angle.source_motor == "eta"
        assert g.sample_motors == ("eta", "chi", "phi", "mu")
        assert g.detector_motors == ("del", "nu")

    def test_psic_halpha(self):
        g = DiffractometerGeometry.psic_halpha()
        assert g.convention == "psic_halpha"
        assert g.rot1.source_motor == "nu"
        assert g.rot2.source_motor == "del"
        # halpha replaces eta as the incidence axis
        assert g.incident_angle.source_motor == "halpha"
        assert "halpha" in g.sample_motors
        assert "eta" not in g.sample_motors

    def test_custom_geometry_construction(self):
        g = DiffractometerGeometry(
            convention="custom",
            rot1=AngleMapping(source_motor="theta_arm"),
            incident_angle=AngleMapping(source_motor="omega", offset=-0.5),
            sample_motors=("omega",),
            detector_motors=("theta_arm",),
        )
        assert g.convention == "custom"
        assert g.rot2.is_active is False
        assert g.incident_angle.offset == -0.5


# ---------------------------------------------------------------------------
# derive_per_frame
# ---------------------------------------------------------------------------

class TestDerivePerFrame:
    def test_two_circle_simple(self):
        g = DiffractometerGeometry.two_circle()
        motors = {
            "tth": np.array([10.0, 20.0, 30.0]),
            "th":  np.array([1.0,  1.0,  1.0]),
        }
        out = g.derive_per_frame(motors)
        # rot1 = deg2rad(tth)
        np.testing.assert_allclose(out["rot1"], np.deg2rad([10.0, 20.0, 30.0]))
        # rot2, rot3 inactive -> zeros
        np.testing.assert_array_equal(out["rot2"], np.zeros(3))
        np.testing.assert_array_equal(out["rot3"], np.zeros(3))
        # incident_angle stays in degrees
        np.testing.assert_array_equal(out["incident_angle"], [1.0, 1.0, 1.0])

    def test_psic_two_active_rotations(self):
        g = DiffractometerGeometry.psic()
        motors = {
            "nu":  np.array([2.0, 4.0]),
            "del": np.array([15.0, 30.0]),
            "eta": np.array([0.5, 0.5]),
        }
        out = g.derive_per_frame(motors)
        np.testing.assert_allclose(out["rot1"], np.deg2rad([2.0, 4.0]))
        np.testing.assert_allclose(out["rot2"], np.deg2rad([15.0, 30.0]))
        np.testing.assert_array_equal(out["rot3"], [0.0, 0.0])
        np.testing.assert_array_equal(out["incident_angle"], [0.5, 0.5])

    def test_psic_halpha_incidence_uses_halpha(self):
        g = DiffractometerGeometry.psic_halpha()
        motors = {
            "nu":     np.array([0.0]),
            "del":    np.array([10.0]),
            "halpha": np.array([3.0]),
        }
        out = g.derive_per_frame(motors)
        np.testing.assert_array_equal(out["incident_angle"], [3.0])

    def test_sign_and_offset_applied_before_deg2rad(self):
        g = DiffractometerGeometry(
            rot1=AngleMapping(source_motor="m", sign=-1.0, offset=10.0),
        )
        motors = {"m": np.array([5.0])}
        out = g.derive_per_frame(motors)
        # sign*m + offset = -5 + 10 = 5 deg -> deg2rad(5)
        np.testing.assert_allclose(out["rot1"], [np.deg2rad(5.0)])

    def test_missing_active_motor_raises(self):
        g = DiffractometerGeometry.psic()
        # Missing 'nu' (which rot1 needs)
        with pytest.raises(KeyError, match="nu"):
            g.derive_per_frame({"del": np.array([1.0]), "eta": np.array([1.0])})

    def test_inconsistent_length_raises(self):
        g = DiffractometerGeometry.psic()
        with pytest.raises(ValueError, match="inconsistent"):
            g.derive_per_frame({
                "nu":  np.array([1.0, 2.0]),
                "del": np.array([1.0, 2.0, 3.0]),
                "eta": np.array([1.0, 2.0]),
            })

    def test_no_active_mappings_gives_length_1_zeros(self):
        g = DiffractometerGeometry()  # all defaults inactive
        out = g.derive_per_frame({})
        for key in ("rot1", "rot2", "rot3", "incident_angle"):
            assert out[key].shape == (1,)
            assert out[key][0] == 0.0


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_two_circle_roundtrip(self):
        g = DiffractometerGeometry.two_circle(gonchi="gonchi")
        g2 = DiffractometerGeometry.from_json(g.to_json())
        assert g == g2

    def test_psic_roundtrip(self):
        g = DiffractometerGeometry.psic()
        g2 = DiffractometerGeometry.from_json(g.to_json())
        assert g == g2

    def test_psic_halpha_roundtrip(self):
        g = DiffractometerGeometry.psic_halpha()
        g2 = DiffractometerGeometry.from_json(g.to_json())
        assert g == g2

    def test_custom_with_offsets_roundtrip(self):
        g = DiffractometerGeometry(
            convention="custom",
            rot1=AngleMapping(source_motor="a", sign=-1.0, offset=0.25),
            rot2=AngleMapping(source_motor="b", sign=1.0, offset=-0.5),
            incident_angle=AngleMapping(source_motor="omega", offset=1.0),
            sample_motors=("omega", "phi"),
            detector_motors=("a", "b"),
        )
        g2 = DiffractometerGeometry.from_json(g.to_json())
        assert g == g2

    def test_json_is_compact(self):
        # Compact (no whitespace) JSON keeps HDF5 attribute size sane.
        g = DiffractometerGeometry.psic()
        s = g.to_json()
        assert " " not in s
        # Re-parseable as plain dict, just to verify it's actually valid JSON
        parsed = json.loads(s)
        assert parsed["convention"] == "psic"


# ---------------------------------------------------------------------------
# all_referenced_motors
# ---------------------------------------------------------------------------

class TestReferencedMotors:
    def test_two_circle(self):
        g = DiffractometerGeometry.two_circle()
        refs = g.all_referenced_motors()
        assert set(refs) == {"tth", "th"}

    def test_psic(self):
        g = DiffractometerGeometry.psic()
        refs = g.all_referenced_motors()
        assert set(refs) == {"nu", "del", "eta", "chi", "phi", "mu"}

    def test_psic_halpha(self):
        g = DiffractometerGeometry.psic_halpha()
        refs = g.all_referenced_motors()
        assert "halpha" in refs
        assert "eta" not in refs
        assert "del" in refs and "nu" in refs

    def test_no_duplicates(self):
        # sample_motors and detector_motors may overlap with active source
        # motors; the result should have each motor exactly once.
        g = DiffractometerGeometry.two_circle()
        refs = g.all_referenced_motors()
        assert len(refs) == len(set(refs))
