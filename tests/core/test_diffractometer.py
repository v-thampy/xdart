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
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import (
    AngleMapping,
    DetectorCalibration,
    Diffractometer,
    DiffractometerConfig,
    DiffractometerGeometry,
    ImageOrientation,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_GONIO_V1 = _FIXTURES / "gonio_robl_v1.json"   # del-only, rot2 pos-linear
_GONIO_V2 = _FIXTURES / "gonio_robl_v2.json"   # del-only, rot1 AND rot2 pos-linear
_GONIO_DELNU = _FIXTURES / "gonio_del_nu_object.json"  # real del/nu (rot1<-nu, rot2<-del)


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
        # The production RSM psic values (RSM_process.ipynb).
        d = Diffractometer.psic()
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
        assert d.detector_circles == ("x+", "z-")
        assert d.camera == ("z-", "x-")
        assert d.hxrd_n == (0.0, 1.0, 0.0)
        assert d.hxrd_q == (0.0, 0.0, 1.0)

    def test_default_is_psic_oriented(self):
        # a bare Diffractometer() is psic-oriented (the house default); the
        # motors are unwired until a preset is applied.
        d = Diffractometer()
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
        assert d.detector_circles == ("x+", "z-")
        assert d.camera == ("z-", "x-")
        assert d.rot1.is_active is False  # orientation only, no motor wiring

    # design-validated golden xu stacks — a swapped/garbage stack must fail,
    # not pass vacuously (the order-blind set check alone cannot catch that).
    _GOLDEN_XU = {
        "two_circle": (("z-",), ("z-",), ("z-", "x+")),
        "fourc": (("z-", "y+", "z-"), ("z-",), ("z-", "x+")),
        "psic": (("x+", "z-", "y+", "z-"), ("x+", "z-"), ("z-", "x-")),
        "sixc": (("x+", "z-", "y+", "z-"), ("x+", "z-"), ("z-", "x-")),
        "psic_halpha": (("x+", "z-", "y+", "z-"), ("x+", "z-"), ("z-", "x-")),
    }

    @pytest.mark.parametrize("preset", _ALL_PRESETS)
    def test_preset_consistency_structural(self, preset):
        """The two derived views must agree on the motor wiring (§5.1).

        A mis-authored preset (e.g. ``rot1`` driven by a motor the circle
        stack never lists) is exactly the drift this object removes.
        """
        d = _build(preset)
        refs = set(d.all_referenced_motors())

        # (0) the xu circle stack + camera are the design-validated literals —
        # an order-blind membership check alone would pass a swapped stack.
        sample_g, detector_g, camera_g = self._GOLDEN_XU[preset]
        assert d.sample_circles == sample_g
        assert d.detector_circles == detector_g
        assert d.camera == camera_g

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

    def test_derive_per_frame_alias(self):
        # the legacy-compat name a duck-typed Scan.geometry consumer calls
        d = Diffractometer.psic()
        motors = {"nu": np.array([2.0]), "del": np.array([15.0]),
                  "eta": np.array([0.5])}
        a = d.derive_per_frame(motors)
        b = d.to_pyfai_per_frame(motors)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


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
        # the QConversion really carries the preset's 4 sample + 2 detector axes
        assert len(qc.sampleAxis) == len(d.sample_circles) == 4
        assert len(qc.detectorAxis) == len(d.detector_circles) == 2

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
        # match the legacy config's HXRD defaults so the only thing under test
        # is the adapter, not a default mismatch
        diff = Diffractometer(
            sample_circles=sample, detector_circles=detector, r_i=r_i,
            camera=camera, hxrd_n=cfg.hxrd_n, hxrd_q=cfg.hxrd_q,
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


# ---------------------------------------------------------------------------
# ImageOrientation (GAP E — raw-array mount transform)
# ---------------------------------------------------------------------------

class TestImageOrientation:
    def test_identity(self):
        o = ImageOrientation()
        assert o.is_identity
        assert not o.swaps_axes
        img = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(o.apply(img), img)

    def test_rejects_bad_rotation(self):
        with pytest.raises(ValueError, match="0/90/180/270"):
            ImageOrientation(rotation=45)

    def test_180_rotation(self):
        o = ImageOrientation(rotation=180)
        img = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(o.apply(img), img[::-1, ::-1])
        assert not o.swaps_axes

    def test_90_swaps_axes(self):
        o = ImageOrientation(rotation=90)
        assert o.swaps_axes
        img = np.arange(6).reshape(2, 3)  # (2,3) -> (3,2)
        assert o.apply(img).shape == (3, 2)

    def test_transpose_xor_rotation_swaps(self):
        # transpose + 90 cancel back to no swap
        assert not ImageOrientation(rotation=90, transpose=True).swaps_axes
        assert ImageOrientation(transpose=True).swaps_axes

    def test_applies_to_trailing_axes_of_stack(self):
        o = ImageOrientation(rotation=180)
        stack = np.arange(2 * 2 * 3).reshape(2, 2, 3)
        out = o.apply(stack)
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out[0], stack[0, ::-1, ::-1])

    def test_dict_roundtrip(self):
        o = ImageOrientation(rotation=270, flip_vertical=True, transpose=True)
        assert ImageOrientation.from_dict(o.to_dict()) == o


# ---------------------------------------------------------------------------
# DetectorCalibration (PONI + Detector_config + mount)
# ---------------------------------------------------------------------------

class TestDetectorCalibration:
    def test_carries_detector_config_and_mount(self):
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.03, poni2=0.05, rot1=0.002,
                      wavelength=7.7e-11, detector="Pilatus300kw"),
            detector_config={"orientation": 3},
            image_orientation=ImageOrientation(rotation=180),
        )
        assert cal.detector_config["orientation"] == 3
        assert cal.image_orientation.rotation == 180

    def test_json_roundtrip(self):
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.03, poni2=0.05, rot1=0.002,
                      rot2=0.0, rot3=0.0, wavelength=7.7e-11,
                      detector="Pilatus300kw"),
            detector_config={"orientation": 3},
            image_orientation=ImageOrientation(rotation=180),
        )
        cal2 = DetectorCalibration.from_json(cal.to_json())
        assert cal2.poni == cal.poni
        assert cal2.detector_config == cal.detector_config
        assert cal2.image_orientation == cal.image_orientation


# ---------------------------------------------------------------------------
# from_pyfai_goniometer (real-data gate — closes stitching GAP D)
# ---------------------------------------------------------------------------

def _full_rot(diff: Diffractometer, motor: str, pos: float):
    """Reconstruct full per-frame pyFAI (rot1, rot2, rot3) at a scalar pos:

    full rotN = base calibration rotN + the per-frame mapping rotN.
    """
    pf = diff.to_pyfai_per_frame({motor: np.array([float(pos)])})
    cal = diff.calibration
    assert cal is not None
    return (
        cal.poni.rot1 + pf["rot1"][0],
        cal.poni.rot2 + pf["rot2"][0],
        cal.poni.rot3 + pf["rot3"][0],
    )


class TestFromPyfaiGoniometer:
    def test_v1_base_calibration(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del", base=Diffractometer.psic())
        cal = d.calibration
        assert cal is not None
        # base poni (constant) from ROBL_v1
        assert cal.poni.dist == pytest.approx(0.3935276273297399)
        assert cal.poni.poni1 == pytest.approx(0.03312099826826781)
        assert cal.poni.poni2 == pytest.approx(0.04978509374677992)
        # rot1 is a CONSTANT in v1 -> lives on the base poni, mapping inactive
        assert cal.poni.rot1 == pytest.approx(0.0027275019657882526)
        assert d.rot1.is_active is False
        # rot2 is pos-linear -> base rot2 == 0, mapping carries it all
        assert cal.poni.rot2 == 0.0
        assert d.rot2.is_active is True
        assert d.rot2.source_motor == "del"
        # rot3 == 0 constant -> inactive
        assert cal.poni.rot3 == 0.0
        assert d.rot3.is_active is False
        # wavelength (meters) + detector + Detector_config preserved
        assert cal.poni.wavelength == pytest.approx(7.748757264459935e-11)
        assert "Pilatus" in cal.poni.detector
        assert cal.detector_config.get("orientation") == 3

    def test_v1_recovered_scale_and_offset(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del")
        # rot2 = deg2rad(sign*del + offset); the fitted scale is ~0.998·deg2rad
        assert d.rot2.sign == pytest.approx(0.9976885327329215, rel=1e-12)
        assert d.rot2.offset == pytest.approx(0.4449878175901234, rel=1e-12)

    def test_v1_reproduces_pyfai_rotations(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del", base=Diffractometer.psic())
        # ground truth from pyFAI Goniometer.get_ai(pos) (radians)
        expected = {
            5: (0.0027275019657882526, 0.09483125157611509, 0.0),
            25: (0.0027275019657882526, 0.4430902476877291, 0.0),
            45: (0.0027275019657882526, 0.791349243799343, 0.0),
        }
        for pos, (r1, r2, r3) in expected.items():
            got = _full_rot(d, "del", pos)
            np.testing.assert_allclose(got, (r1, r2, r3), rtol=0, atol=1e-12)

    def test_v2_rot1_also_pos_linear(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V2, source_motors="del")
        # in v2 rot1_expr = 'rot1_scale*pos + rot1' -> ACTIVE mapping
        assert d.rot1.is_active is True
        assert d.rot1.sign == pytest.approx(-0.008853094908028594, rel=1e-9)
        assert d.rot1.offset == pytest.approx(0.6567280316039488, rel=1e-9)
        assert d.calibration.poni.rot1 == 0.0  # whole rotation in the mapping

    def test_cross_check_against_pyfai_goniometer(self):
        """Strongest gate: reconstruction == pyFAI Goniometer.get_ai per frame."""
        gonio = pytest.importorskip("pyFAI.goniometer")
        for path in (_GONIO_V1, _GONIO_V2):
            g = gonio.Goniometer.sload(str(path))
            d = Diffractometer.from_pyfai_goniometer(path, source_motors="del")
            for pos in (3.0, 17.5, 41.0):
                ai = g.get_ai(pos)
                got = _full_rot(d, "del", pos)
                np.testing.assert_allclose(
                    got, (ai.rot1, ai.rot2, ai.rot3), rtol=0, atol=1e-12)
                # base geometry matches too
                assert d.calibration.poni.dist == pytest.approx(ai.dist)
                assert d.calibration.poni.poni1 == pytest.approx(ai.poni1)
                assert d.calibration.poni.poni2 == pytest.approx(ai.poni2)

    def test_real_del_nu_two_axis(self):
        """Real two-position (del, nu) goniometer: rot1<-nu, rot2<-del,
        axis-separable — reproduces pyFAI get_ai at arbitrary (del, nu)."""
        gonio = pytest.importorskip("pyFAI.goniometer")
        g = gonio.Goniometer.sload(str(_GONIO_DELNU))
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_DELNU, source_motors={"del_value": "del", "nu_value": "nu"},
            base=Diffractometer.psic())
        assert d.rot1.source_motor == "nu" and d.rot1.is_active
        assert d.rot2.source_motor == "del" and d.rot2.is_active
        cal = d.calibration
        for de, nu in [(6.0, 0.0), (20.0, 5.0), (45.0, 29.0)]:
            ai = g.get_ai((de, nu))
            pf = d.to_pyfai_per_frame({"del": np.array([de]), "nu": np.array([nu])})
            got = (cal.poni.rot1 + pf["rot1"][0], cal.poni.rot2 + pf["rot2"][0],
                   cal.poni.rot3 + pf["rot3"][0])
            np.testing.assert_allclose(
                got, (ai.rot1, ai.rot2, ai.rot3), rtol=0, atol=1e-12)

    def test_xu_half_donated_from_base(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del", base=Diffractometer.psic())
        # the gonio JSON has no xu info; it comes from the base preset
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
        assert d.camera == ("z-", "x-")
        assert "del" in d.detector_motors

    def test_image_orientation_threaded(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del",
            image_orientation=ImageOrientation(rotation=180))
        assert d.calibration.image_orientation.rotation == 180

    def test_full_json_roundtrip_with_calibration(self):
        d = Diffractometer.from_pyfai_goniometer(
            _GONIO_V1, source_motors="del", base=Diffractometer.psic(),
            image_orientation=ImageOrientation(rotation=180))
        d2 = Diffractometer.from_json(d.to_json())
        assert d2 == d

    def test_custom_subclass_rejected(self):
        # a record with no trans_function (StackedArmGoniometer etc.)
        with pytest.raises(NotImplementedError, match="trans_function"):
            Diffractometer.from_pyfai_goniometer(
                {"content": "StackedArmGoniometer", "param": [], "param_names": []})

    def test_nonlinear_rejected(self):
        gonio = {
            "content": "Goniometer calibration v2",
            "detector": "Pilatus300kw",
            "wavelength": 7.7e-11,
            "param_names": ["dist", "poni1", "poni2", "k"],
            "param": [0.39, 0.03, 0.05, 0.01],
            "trans_function": {
                "content": "GeometryTransformation",
                "pos_names": ["pos"],
                "param_names": ["dist", "poni1", "poni2", "k"],
                "dist_expr": "dist", "poni1_expr": "poni1", "poni2_expr": "poni2",
                "rot1_expr": "k * pos**2", "rot2_expr": "0.0", "rot3_expr": "0.0",
                "constants": {"pi": 3.141592653589793},
            },
        }
        with pytest.raises(NotImplementedError, match="non-linear"):
            Diffractometer.from_pyfai_goniometer(gonio, source_motors="del")

    def _two_pos_gonio(self, *, dist_expr="dist", rot3_expr="0.0"):
        return {
            "content": "Goniometer calibration v2",
            "detector": "Pilatus300kw", "wavelength": 7.7e-11,
            "param_names": ["dist", "poni1", "poni2", "r2s", "c", "drift"],
            "param": [0.39, 0.03, 0.05, 0.0174, 0.001, 0.001],
            "trans_function": {
                "content": "GeometryTransformation",
                "pos_names": ["p", "q"],
                "param_names": ["dist", "poni1", "poni2", "r2s", "c", "drift"],
                "dist_expr": dist_expr, "poni1_expr": "poni1", "poni2_expr": "poni2",
                "rot1_expr": "0.0", "rot2_expr": "r2s * p", "rot3_expr": rot3_expr,
                "constants": {"pi": 3.141592653589793},
            },
        }

    def test_cross_term_rejected(self):
        # a pure bilinear p*q term would otherwise read as constant 0 and be
        # silently dropped (verified to grow to tens of degrees off-axis).
        g = self._two_pos_gonio(rot3_expr="c * p * q")
        with pytest.raises(NotImplementedError, match="cross-term"):
            Diffractometer.from_pyfai_goniometer(g, source_motors={"p": "nu", "q": "del"})

    def test_moving_base_rejected(self):
        # a position-dependent dist (moving detector) cannot be frozen at pos=0.
        g = self._two_pos_gonio(dist_expr="dist + drift * p")
        with pytest.raises(NotImplementedError, match="dist_expr depends on position"):
            Diffractometer.from_pyfai_goniometer(g, source_motors={"p": "nu", "q": "del"})

    def test_additive_multi_axis_still_rejected(self):
        # additive (non-cross) multi-axis dependence is still out of scope, but
        # must be rejected by the multi-axis branch, not the cross-term one.
        g = self._two_pos_gonio(rot3_expr="0.01 * p + 0.02 * q")
        with pytest.raises(NotImplementedError, match="multiple position axes"):
            Diffractometer.from_pyfai_goniometer(g, source_motors={"p": "nu", "q": "del"})


# ---------------------------------------------------------------------------
# Serialization robustness (review fixes — numpy scalars, tuples, partial JSON)
# ---------------------------------------------------------------------------

class TestSerializationRobustness:
    def test_numpy_rotation_coerced(self):
        o = ImageOrientation(rotation=np.int64(180))
        assert isinstance(o.rotation, int)
        assert json.dumps(o.to_dict())  # would raise TypeError on a numpy scalar

    def test_numpy_scalar_in_detector_config_does_not_crash(self):
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.0, poni2=0.0),
            detector_config={"orientation": np.int64(3)},
            image_orientation=ImageOrientation(rotation=np.int64(180)))
        # to_json must not raise on the numpy scalar, and must round-trip
        assert DetectorCalibration.from_json(cal.to_json()) == cal
        assert cal.detector_config["orientation"] == 3

    def test_tuple_valued_kwargs_roundtrip(self):
        d = Diffractometer(preset="custom", hxrd_kwargs={"sampleor": ("z+",)})
        assert d == Diffractometer.from_json(d.to_json())

    def test_tuple_in_detector_config_roundtrip(self):
        cal = DetectorCalibration(
            poni=PONI(dist=0.39, poni1=0.0, poni2=0.0),
            detector_config={"max_shape": (195, 1475)})
        assert DetectorCalibration.from_json(cal.to_json()) == cal

    def test_partial_json_uses_canonical_defaults(self):
        # a blob missing the xu circle stacks reconstructs the dataclass
        # defaults (the psic orientation)
        d = Diffractometer.from_json(json.dumps({"preset": "custom"}))
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
        assert d.detector_circles == ("x+", "z-")
        assert d == Diffractometer()


# ---------------------------------------------------------------------------
# Legacy interop bridges (step 2 — lift/lower, value-preserving)
# ---------------------------------------------------------------------------

class TestLegacyInterop:
    @pytest.mark.parametrize("preset", _LEGACY_PARITY_PRESETS)
    def test_geometry_lift_lower_roundtrip(self, preset):
        g = getattr(DiffractometerGeometry, preset)()
        d = Diffractometer.from_diffractometer_geometry(g)
        assert d.to_diffractometer_geometry() == g

    @pytest.mark.parametrize("preset", _LEGACY_PARITY_PRESETS)
    def test_lifted_geometry_derive_is_byte_equal(self, preset):
        g = getattr(DiffractometerGeometry, preset)()
        d = Diffractometer.from_diffractometer_geometry(g)
        motors = {
            "tth": np.array([10.0, 20.0]), "th": np.array([1.0, 2.0]),
            "nu": np.array([2.0, 4.0]), "del": np.array([15.0, 30.0]),
            "eta": np.array([0.5, 0.5]), "halpha": np.array([3.0, 3.0]),
        }
        need = {m: motors[m] for m in g.all_referenced_motors() if m in motors}
        out_new = d.to_pyfai_per_frame(need)
        out_old = g.derive_per_frame(need)
        for k in out_old:
            np.testing.assert_array_equal(out_new[k], out_old[k])

    def test_config_lift_lower_roundtrip(self):
        c = DiffractometerConfig(
            sample_rot=("x+", "z-", "y+", "z-"),
            detector_rot=("x+", "z-"),
            init_area_detrot="x-", init_area_tiltazimuth="z+",
            q_conv_kwargs={"wl": 1.2}, ang2q_kwargs={"foo": "bar"},
        )
        d = Diffractometer.from_diffractometer_config(c)
        assert d.to_diffractometer_config() == c

    def test_lifted_config_qconversion_is_byte_equal(self):
        pytest.importorskip("xrayutilities")
        c = DiffractometerConfig(
            sample_rot=("x+", "z-", "y+", "z-"),
            detector_rot=("x+", "z-"),
            init_area_detrot="x-", init_area_tiltazimuth="z+",
        )
        d = Diffractometer.from_diffractometer_config(c)
        angles = [np.array([0.0, 0.0]), np.array([1.0, 2.0]),
                  np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                  np.array([5.0, 6.0]), np.array([10.0, 12.0])]

        def run(builder):
            hxrd = builder(17000.0)
            hxrd.Ang2Q.init_area(c.init_area_detrot, c.init_area_tiltazimuth,
                                 cch1=100.0, cch2=200.0, pwidth1=0.172,
                                 pwidth2=0.172, distance=390.0,
                                 Nch1=40, Nch2=50)
            return hxrd.Ang2Q.area(*angles)

        for a, b in zip(run(c.make_hxrd), run(d.to_hxrd)):
            np.testing.assert_array_equal(a, b)

    def test_pixelqmap_dropin_equivalence(self):
        """A Diffractometer is a transparent PixelQMap.diff_config: same q as
        the equivalent DiffractometerConfig through the full RSM pixel_q path."""
        pytest.importorskip("xrayutilities")
        from xrd_tools.core.geometry import DetectorHeader, PixelQMap
        sample = ("x+", "z-", "y+", "z-")
        detector = ("x+", "z-")
        camera = ("z-", "x-")
        cfg = DiffractometerConfig(
            sample_rot=sample, detector_rot=detector, r_i=(0.0, 1.0, 0.0),
            init_area_detrot=camera[0], init_area_tiltazimuth=camera[1],
            hxrd_n=(0.0, 1.0, 0.0), hxrd_q=(0.0, 0.0, 1.0))
        diff = Diffractometer.from_diffractometer_config(cfg)
        header = DetectorHeader(cch1=100.0, cch2=200.0, pwidth1=0.172,
                                pwidth2=0.172, distance=390.0, Nch1=40, Nch2=50)
        angles = [np.array([0.0, 0.0]), np.array([1.0, 2.0]),
                  np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                  np.array([5.0, 6.0]), np.array([10.0, 12.0])]
        UB = np.eye(3)
        q_cfg = PixelQMap(cfg, header).pixel_q(angles, 17000.0, UB=UB)
        q_diff = PixelQMap(diff, header).pixel_q(angles, 17000.0, UB=UB)
        for a, b in zip(q_cfg, q_diff):
            np.testing.assert_array_equal(a, b)

    def test_lift_with_overrides_donates_xu_half(self):
        # a writer holds a DiffractometerGeometry (pyFAI half only); donate the
        # xu half from a config so the lifted object also feeds RSM.
        g = DiffractometerGeometry.psic()
        c = DiffractometerConfig(sample_rot=("x+", "z-", "y+", "z-"),
                                 detector_rot=("x+", "z-"))
        donor = Diffractometer.from_diffractometer_config(c)
        d = Diffractometer.from_diffractometer_geometry(
            g, sample_circles=donor.sample_circles,
            detector_circles=donor.detector_circles, camera=donor.camera)
        assert d.rot1 == g.rot1 and d.rot2 == g.rot2
        assert d.sample_circles == ("x+", "z-", "y+", "z-")
