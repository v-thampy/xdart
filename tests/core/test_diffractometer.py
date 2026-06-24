"""Tests for the canonical ``Diffractometer`` (one object, two derived views).

Covers design ``design_diffractometer_geometry_jun2026.md`` §6 steps 0 + 1:

* preset authoring of BOTH halves + structural preset-consistency (§5.1)
* JSON round-trip for persistence
* ``to_pyfai_per_frame`` byte-equal to ``DiffractometerGeometry.derive_per_frame``
* ``to_qconversion`` / ``to_hxrd`` equivalent to ``DiffractometerConfig.make_hxrd``
  (real-xrayutilities, importorskip-guarded)

The two legacy classes (``DiffractometerConfig`` / ``DiffractometerGeometry``)
remain the value-preserving reference: the new object must reproduce them
exactly so the step-2 compat shims are trivial.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from xrd_tools.core.geometry import (
    AngleMapping,
    Diffractometer,
    DiffractometerConfig,
    DiffractometerGeometry,
)


# Presets that have an exact legacy ``DiffractometerGeometry`` counterpart —
# their pyFAI per-frame view must be byte-equal.
_LEGACY_PARITY_PRESETS = ("two_circle", "psic", "psic_halpha")

_ALL_PRESETS = ("two_circle", "fourc", "psic", "sixc", "psic_halpha")


def _build(preset: str) -> Diffractometer:
    return getattr(Diffractometer, preset)()


# ---------------------------------------------------------------------------
# Presets author both halves + structural consistency (§5.1)
# ---------------------------------------------------------------------------

class TestPresets:
    def test_preset_tag_recorded(self):
        # ``sixc`` is an alias of ``psic`` so it reports the psic preset tag.
        assert Diffractometer.two_circle().preset == "two_circle"
        assert Diffractometer.fourc().preset == "fourc"
        assert Diffractometer.psic().preset == "psic"
        assert Diffractometer.sixc().preset == "psic"
        assert Diffractometer.psic_halpha().preset == "psic_halpha"

    def test_two_circle_matches_legacy_pyfai_half(self):
        d = Diffractometer.two_circle(gonchi="gonchi")
        g = DiffractometerGeometry.two_circle(gonchi="gonchi")
        assert d.rot1 == g.rot1
        assert d.rot2 == g.rot2
        assert d.rot3 == g.rot3
        assert d.incident_angle == g.incident_angle
        assert d.sample_motors == g.sample_motors
        assert d.detector_motors == g.detector_motors

    def test_psic_matches_legacy_pyfai_half(self):
        d = Diffractometer.psic()
        g = DiffractometerGeometry.psic()
        assert (d.rot1, d.rot2, d.incident_angle) == (g.rot1, g.rot2, g.incident_angle)
        assert d.sample_motors == g.sample_motors
        assert d.detector_motors == g.detector_motors

    def test_psic_halpha_matches_legacy_pyfai_half(self):
        d = Diffractometer.psic_halpha()
        g = DiffractometerGeometry.psic_halpha()
        assert d.incident_angle == g.incident_angle
        assert d.sample_motors == g.sample_motors

    def test_psic_xu_half_is_validated_stack(self):
        # The SSRL Pilatus-300k-w psic arm (xu_geometry_del_nu.json).
        d = Diffractometer.psic()
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
        assert d.detector_circles == ("x+", "z-")
        assert d.camera == ("x-", "z+")

    @pytest.mark.parametrize("preset", _ALL_PRESETS)
    def test_preset_consistency_structural(self, preset):
        """The two derived views must agree on the motor wiring (§5.1).

        A mis-authored preset (e.g. ``rot1`` driven by a motor the circle
        stack never lists) is exactly the drift this object removes.
        """
        d = _build(preset)
        refs = set(d.all_referenced_motors())

        # (1) every active pyFAI detector rotation is driven by a detector motor
        for rot in (d.rot1, d.rot2, d.rot3):
            if rot.is_active:
                assert rot.source_motor in d.detector_motors, (
                    f"{preset}: {rot.source_motor!r} drives a rot but is not a "
                    f"detector motor {d.detector_motors}"
                )
        # (2) the GI incidence is driven by a sample motor
        if d.incident_angle.is_active:
            assert d.incident_angle.source_motor in d.sample_motors

        # (3) circle_motors align 1:1 with the combined circle stack
        n_circles = len(d.sample_circles) + len(d.detector_circles)
        assert len(d.circle_motors) == n_circles, (
            f"{preset}: {len(d.circle_motors)} circle motors for {n_circles} circles"
        )
        # sample circles first, then detector circles
        sample_cms = d.circle_motors[: len(d.sample_circles)]
        detector_cms = d.circle_motors[len(d.sample_circles):]
        detector_cm_motors = {cm.source_motor for cm in detector_cms if cm.is_active}
        sample_cm_motors = {cm.source_motor for cm in sample_cms if cm.is_active}
        # the detector rotations come from detector circles
        for rot in (d.rot1, d.rot2, d.rot3):
            if rot.is_active:
                assert rot.source_motor in detector_cm_motors
        # the incidence comes from a sample circle
        if d.incident_angle.is_active:
            assert d.incident_angle.source_motor in sample_cm_motors

        # (4) every circle motor is one of the referenced motors (no orphans)
        for cm in d.circle_motors:
            if cm.is_active:
                assert cm.source_motor in refs


# ---------------------------------------------------------------------------
# Adapter 1 — to_pyfai_per_frame (byte-equal to derive_per_frame)
# ---------------------------------------------------------------------------

class TestPyfaiPerFrame:
    @pytest.mark.parametrize("preset", _LEGACY_PARITY_PRESETS)
    def test_byte_equal_to_legacy(self, preset):
        d = _build(preset)
        g = getattr(DiffractometerGeometry, preset)()
        motors = {
            "tth": np.array([10.0, 20.0, 30.0]),
            "th": np.array([1.0, 1.0, 1.0]),
            "nu": np.array([2.0, 4.0, 6.0]),
            "del": np.array([15.0, 30.0, 45.0]),
            "eta": np.array([0.5, 0.5, 0.5]),
            "halpha": np.array([3.0, 3.0, 3.0]),
        }
        # restrict to the motors each geometry needs
        need = {m: motors[m] for m in d.all_referenced_motors() if m in motors}
        out_new = d.to_pyfai_per_frame(need)
        out_old = g.derive_per_frame(need)
        assert out_new.keys() == out_old.keys()
        for k in out_old:
            np.testing.assert_array_equal(out_new[k], out_old[k], err_msg=f"{preset}:{k}")

    def test_rot_radians_incidence_degrees(self):
        d = Diffractometer.psic()
        out = d.to_pyfai_per_frame({
            "nu": np.array([2.0, 4.0]),
            "del": np.array([15.0, 30.0]),
            "eta": np.array([0.5, 0.5]),
        })
        np.testing.assert_allclose(out["rot1"], np.deg2rad([2.0, 4.0]))
        np.testing.assert_allclose(out["rot2"], np.deg2rad([15.0, 30.0]))
        np.testing.assert_array_equal(out["rot3"], [0.0, 0.0])
        np.testing.assert_array_equal(out["incident_angle"], [0.5, 0.5])

    def test_fitted_scale_and_offset(self):
        # AngleMapping.sign as a fitted scale: rot = deg2rad(sign*motor+offset).
        d = Diffractometer(
            rot2=AngleMapping(source_motor="del", sign=0.9976885327329215,
                              offset=0.4449878175901234),
        )
        out = d.to_pyfai_per_frame({"del": np.array([5.0, 25.0, 45.0])})
        # matches pyFAI ROBL_v1 rot2 ground truth (radians)
        np.testing.assert_allclose(
            out["rot2"], [0.09483125157611508, 0.44309024768772914, 0.7913492437993431],
            rtol=0, atol=1e-12,
        )

    def test_missing_active_motor_raises(self):
        d = Diffractometer.psic()
        with pytest.raises(KeyError, match="nu"):
            d.to_pyfai_per_frame({"del": np.array([1.0]), "eta": np.array([1.0])})

    def test_inconsistent_length_raises(self):
        d = Diffractometer.psic()
        with pytest.raises(ValueError, match="inconsistent"):
            d.to_pyfai_per_frame({
                "nu": np.array([1.0, 2.0]),
                "del": np.array([1.0, 2.0, 3.0]),
                "eta": np.array([1.0, 2.0]),
            })

    def test_no_active_gives_length_1_zeros(self):
        d = Diffractometer()
        out = d.to_pyfai_per_frame({})
        for k in ("rot1", "rot2", "rot3", "incident_angle"):
            assert out[k].shape == (1,)
            assert out[k][0] == 0.0


# ---------------------------------------------------------------------------
# JSON round-trip (persistence)
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    @pytest.mark.parametrize("preset", _ALL_PRESETS)
    def test_preset_roundtrip(self, preset):
        d = _build(preset)
        assert d == Diffractometer.from_json(d.to_json())

    def test_custom_with_offsets_and_kwargs_roundtrip(self):
        d = Diffractometer(
            preset="custom",
            rot1=AngleMapping(source_motor="a", sign=-1.0, offset=0.25),
            rot2=AngleMapping(source_motor="b", sign=0.99, offset=-0.5),
            incident_angle=AngleMapping(source_motor="om", offset=1.0),
            sample_circles=("x+", "z-"),
            detector_circles=("x+",),
            r_i=(0.0, 1.0, 0.0),
            camera=("x-", "z+"),
            circle_motors=(AngleMapping(source_motor="om"),
                           AngleMapping(source_motor="b"),
                           AngleMapping(source_motor="a")),
            sample_motors=("om", "phi"),
            detector_motors=("a", "b"),
            qconv_kwargs={"wl": 1.2},
            ang2q_kwargs={"foo": "bar"},
        )
        assert d == Diffractometer.from_json(d.to_json())

    def test_circle_motors_roundtrip(self):
        d = Diffractometer.psic()
        d2 = Diffractometer.from_json(d.to_json())
        assert d.circle_motors == d2.circle_motors
        assert all(isinstance(cm, AngleMapping) for cm in d2.circle_motors)

    def test_json_is_compact(self):
        s = Diffractometer.psic().to_json()
        assert " " not in s
        parsed = json.loads(s)
        assert parsed["preset"] == "psic"


# ---------------------------------------------------------------------------
# Adapter 2 — to_qconversion / to_hxrd (real xrayutilities)
# ---------------------------------------------------------------------------

class TestXuAdapters:
    def test_to_qconversion_axis_counts(self):
        xu = pytest.importorskip("xrayutilities")
        d = Diffractometer.psic()
        qc = d.to_qconversion()
        assert isinstance(qc, xu.QConversion)

    def test_to_hxrd_builds_instance(self):
        xu = pytest.importorskip("xrayutilities")
        d = Diffractometer.psic()
        hxrd = d.to_hxrd(17000.0)
        assert isinstance(hxrd, xu.HXRD)

    def test_equivalent_to_diffractometer_config_make_hxrd(self):
        """Strong byte-equality: identical fields → identical q (real xu).

        Build a ``Diffractometer`` and a ``DiffractometerConfig`` with the
        SAME axis stacks / camera / refs, run the full
        ``make_hxrd → init_area → Ang2Q.area`` path on both, and assert the
        per-pixel q is bit-identical.  This is the step-1 'numeric equality
        vs the old class' gate that lets step 2 alias the old class onto the
        new object.
        """
        pytest.importorskip("xrayutilities")
        sample = ("x+", "z-", "y+", "z-")
        detector = ("x+", "z-")
        camera = ("x-", "z+")
        r_i = (0.0, 1.0, 0.0)

        cfg = DiffractometerConfig(
            sample_rot=sample, detector_rot=detector, r_i=r_i,
            init_area_detrot=camera[0], init_area_tiltazimuth=camera[1],
        )
        diff = Diffractometer(
            sample_circles=sample, detector_circles=detector, r_i=r_i,
            camera=camera,
        )

        energy = 17000.0
        # synthetic angles: 4 sample axes + 2 detector axes, 2 frames
        angles = [
            np.array([0.0, 0.0]), np.array([1.0, 2.0]),
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            np.array([5.0, 6.0]), np.array([10.0, 12.0]),
        ]
        cch1, cch2 = 100.0, 200.0
        pwidth, distance = 0.172, 390.0
        Nch1, Nch2 = 50, 60

        def run(builder):
            hxrd = builder(energy)
            hxrd.Ang2Q.init_area(
                camera[0], camera[1],
                cch1=cch1, cch2=cch2, pwidth1=pwidth, pwidth2=pwidth,
                distance=distance, Nch1=Nch1, Nch2=Nch2,
            )
            return hxrd.Ang2Q.area(*angles)

        qx0, qy0, qz0 = run(cfg.make_hxrd)
        qx1, qy1, qz1 = run(diff.to_hxrd)
        np.testing.assert_array_equal(qx0, qx1)
        np.testing.assert_array_equal(qy0, qy1)
        np.testing.assert_array_equal(qz0, qz1)
