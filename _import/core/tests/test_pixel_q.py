"""Tests for ssrl_xrd_tools.core.geometry.pixel_q.

Covers the per-pixel q-space mapping primitives added for the post-0.37
RSM track:

* :class:`DetectorHeader` field semantics, ROI / image-shape transforms,
  JSON round-trip.
* :class:`PixelQMap` shape contract, ROI shifting of the beam centre,
  image_shape override, error on missing Nch1/Nch2, UB propagation,
  energy plumbing into ``make_hxrd``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from ssrl_xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)

# ---------------------------------------------------------------------------
# DetectorHeader
# ---------------------------------------------------------------------------

class TestDetectorHeader:
    def test_basic_fields(self):
        h = DetectorHeader(
            cch1=100.0, cch2=200.0,
            pwidth1=0.075, pwidth2=0.075,
            distance=830.0,
            Nch1=512, Nch2=1024,
        )
        assert h.cch1 == 100.0
        assert h.Nch1 == 512
        assert h.Nch2 == 1024

    def test_frozen(self):
        h = DetectorHeader(0, 0, 1, 1, 100, 10, 10)
        with pytest.raises((AttributeError, Exception)):
            h.cch1 = 5  # type: ignore[misc]

    def test_with_roi_positive_bounds(self):
        h = DetectorHeader(
            cch1=100.0, cch2=200.0,
            pwidth1=0.075, pwidth2=0.075,
            distance=830.0,
            Nch1=512, Nch2=1024,
        )
        out = h.with_roi((10, 110, 20, 220))  # 100 x 200 ROI
        # Beam centre shifts by (-r0, -c0)
        assert out.cch1 == pytest.approx(90.0)
        assert out.cch2 == pytest.approx(180.0)
        # New Nch* = r1 - r0, c1 - c0
        assert out.Nch1 == 100
        assert out.Nch2 == 200
        # pwidth + distance preserved
        assert out.pwidth1 == h.pwidth1
        assert out.distance == h.distance

    def test_with_roi_negative_bounds(self):
        """Negative r1/c1 follow Python-slice semantics."""
        h = DetectorHeader(0.0, 0.0, 1.0, 1.0, 100.0, Nch1=100, Nch2=200)
        out = h.with_roi((10, -10, 20, -20))
        # New Nch1 = (Nch1 + r1) - r0 = (100 - 10) - 10 = 80
        assert out.Nch1 == 80
        assert out.Nch2 == 160

    def test_with_image_shape(self):
        h = DetectorHeader(0.0, 0.0, 1.0, 1.0, 100.0, Nch1=0, Nch2=0)
        out = h.with_image_shape((3, 514, 1030))  # accept stacked shape
        assert out.Nch1 == 514
        assert out.Nch2 == 1030
        # Beam centre preserved
        assert out.cch1 == h.cch1
        assert out.cch2 == h.cch2

    def test_with_image_shape_2d(self):
        h = DetectorHeader(0.0, 0.0, 1.0, 1.0, 100.0, Nch1=0, Nch2=0)
        out = h.with_image_shape((42, 84))
        assert out.Nch1 == 42
        assert out.Nch2 == 84

    def test_with_image_shape_too_short(self):
        h = DetectorHeader(0.0, 0.0, 1.0, 1.0, 100.0, Nch1=0, Nch2=0)
        with pytest.raises(ValueError, match=">= 2 dims"):
            h.with_image_shape((5,))

    def test_json_roundtrip(self):
        h = DetectorHeader(
            cch1=257.5, cch2=515.0,
            pwidth1=0.075, pwidth2=0.075,
            distance=830.0,
            Nch1=514, Nch2=1030,
        )
        s = h.to_json()
        # JSON keys are predictable / lexicographic so this is stable
        parsed = json.loads(s)
        assert set(parsed.keys()) == {
            "cch1", "cch2", "pwidth1", "pwidth2",
            "distance", "Nch1", "Nch2",
        }
        roundtripped = DetectorHeader.from_json(s)
        assert roundtripped == h


# ---------------------------------------------------------------------------
# DetectorHeader.from_poni
# ---------------------------------------------------------------------------

class _FakePoni:
    """Stand-in for ssrl_xrd_tools.core.containers.PONI (SI units)."""
    def __init__(
        self,
        dist: float = 0.830,        # 830 mm
        poni1: float = 0.0193125,   # = 257.5 * 75 μm
        poni2: float = 0.0386250,   # = 515.0 * 75 μm
        detector: str = "",
    ):
        self.dist = dist
        self.poni1 = poni1
        self.poni2 = poni2
        self.detector = detector


class TestDetectorHeaderFromPoni:
    def test_explicit_pixel_sizes(self):
        poni = _FakePoni()
        h = DetectorHeader.from_poni(
            poni, pixel1=75e-6, pixel2=75e-6, image_shape=(514, 1030),
        )
        # cch = poni / pixel (in pixels)
        assert h.cch1 == pytest.approx(257.5)
        assert h.cch2 == pytest.approx(515.0)
        # pwidth = pixel * 1000 (m → mm)
        assert h.pwidth1 == pytest.approx(0.075)
        assert h.pwidth2 == pytest.approx(0.075)
        # distance = dist * 1000 (m → mm)
        assert h.distance == pytest.approx(830.0)
        # Nch from image_shape
        assert h.Nch1 == 514
        assert h.Nch2 == 1030

    def test_image_shape_optional(self):
        poni = _FakePoni()
        h = DetectorHeader.from_poni(poni, pixel1=75e-6, pixel2=75e-6)
        assert h.Nch1 == 0 and h.Nch2 == 0  # placeholder until user fills in
        # User can chain with_image_shape
        h2 = h.with_image_shape((128, 256))
        assert h2.Nch1 == 128 and h2.Nch2 == 256

    def test_stacked_image_shape_uses_last_two(self):
        poni = _FakePoni()
        h = DetectorHeader.from_poni(
            poni, pixel1=75e-6, pixel2=75e-6, image_shape=(10, 64, 128),
        )
        assert h.Nch1 == 64 and h.Nch2 == 128

    def test_no_pixel_size_or_detector_raises(self):
        poni = _FakePoni(detector="")
        with pytest.raises(ValueError, match="pixel1/pixel2 are required"):
            DetectorHeader.from_poni(poni)

    def test_detector_argument_overrides_poni_detector(self, monkeypatch):
        """Explicit detector= wins over poni.detector when both are present."""
        called: dict[str, str] = {}

        class _StubDet:
            pixel1 = 100e-6
            pixel2 = 200e-6

        class _StubDetectorsModule:
            @staticmethod
            def detector_factory(name):
                called["name"] = name
                return _StubDet()

        import sys
        import types
        # Need to register the parent pyFAI package too so `import pyFAI.detectors`
        # resolves under sandbox conditions where pyFAI isn't actually installed.
        fake_pkg = types.ModuleType("pyFAI")
        fake_pkg.detectors = _StubDetectorsModule  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyFAI", fake_pkg)
        monkeypatch.setitem(sys.modules, "pyFAI.detectors", _StubDetectorsModule)

        poni = _FakePoni(detector="Pilatus300k")
        h = DetectorHeader.from_poni(poni, detector="Eiger1M")
        assert called["name"] == "Eiger1M"
        assert h.pwidth1 == pytest.approx(0.1)   # 100 μm → 0.1 mm
        assert h.pwidth2 == pytest.approx(0.2)

    def test_poni_detector_used_as_fallback(self, monkeypatch):
        called: dict[str, str] = {}

        class _StubDet:
            pixel1 = 75e-6
            pixel2 = 75e-6

        class _StubDetectorsModule:
            @staticmethod
            def detector_factory(name):
                called["name"] = name
                return _StubDet()

        import sys
        import types
        fake_pkg = types.ModuleType("pyFAI")
        fake_pkg.detectors = _StubDetectorsModule  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyFAI", fake_pkg)
        monkeypatch.setitem(sys.modules, "pyFAI.detectors", _StubDetectorsModule)

        poni = _FakePoni(detector="Pilatus300k")
        h = DetectorHeader.from_poni(poni)  # no explicit detector arg
        assert called["name"] == "Pilatus300k"
        assert h.pwidth1 == pytest.approx(0.075)

    def test_zero_pixel_size_rejected(self):
        poni = _FakePoni()
        with pytest.raises(ValueError, match="pixel sizes must be > 0"):
            DetectorHeader.from_poni(poni, pixel1=0.0, pixel2=75e-6)

    def test_pyfai_lookup_failure_raises_helpful_error(self, monkeypatch):
        class _FailingDetectorsModule:
            @staticmethod
            def detector_factory(name):
                raise RuntimeError(f"no detector named {name!r}")

        import sys
        import types
        fake_pkg = types.ModuleType("pyFAI")
        fake_pkg.detectors = _FailingDetectorsModule  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyFAI", fake_pkg)
        monkeypatch.setitem(sys.modules, "pyFAI.detectors", _FailingDetectorsModule)

        poni = _FakePoni(detector="NotARealDetector")
        with pytest.raises(ValueError, match="cannot resolve detector"):
            DetectorHeader.from_poni(poni)

    def test_round_trip_via_pixelqmap(self, monkeypatch):
        """from_poni → PixelQMap → pixel_q (mocked) preserves the header."""
        captured: dict = {}

        class _Tracker:
            def init_area(self, *args, **kwargs):
                captured.update(kwargs)
            def area(self, *args, **kwargs):
                N = len(np.atleast_1d(args[0]))
                Nch1 = captured["Nch1"]
                Nch2 = captured["Nch2"]
                return tuple(np.zeros((N, Nch1, Nch2)) for _ in range(3))

        class _TrackerHXRD:
            Ang2Q = _Tracker()

        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            lambda self, energy: _TrackerHXRD(),
        )

        poni = _FakePoni()
        header = DetectorHeader.from_poni(
            poni, pixel1=75e-6, pixel2=75e-6, image_shape=(514, 1030),
        )
        mapper = PixelQMap(DiffractometerConfig(), header)
        mapper.pixel_q([np.array([0.0])], energy=12000.0)
        assert captured["cch1"] == pytest.approx(257.5)
        assert captured["pwidth1"] == pytest.approx(0.075)
        assert captured["distance"] == pytest.approx(830.0)
        assert captured["Nch1"] == 514


# ---------------------------------------------------------------------------
# PixelQMap — needs xrayutilities (mocked when not installed)
# ---------------------------------------------------------------------------

class _FakeAng2Q:
    """Records init_area kwargs; area() returns zero arrays of the configured shape."""

    def __init__(self):
        self.init_area_kwargs: dict = {}
        self.init_area_args: tuple = ()
        self.area_args: tuple = ()
        self.area_kwargs: dict = {}

    def init_area(self, *args, **kwargs):
        self.init_area_args = args
        self.init_area_kwargs = kwargs

    def area(self, *args, **kwargs):
        self.area_args = args
        self.area_kwargs = kwargs
        # angles is the positional args, take N from first sample axis
        N = len(np.atleast_1d(args[0]))
        Nch1 = self.init_area_kwargs.get("Nch1")
        Nch2 = self.init_area_kwargs.get("Nch2")
        shape = (N, Nch1, Nch2)
        return (
            np.zeros(shape, dtype=float),
            np.ones(shape, dtype=float),
            np.full(shape, 2.0, dtype=float),
        )


class _FakeHXRD:
    def __init__(self):
        self.Ang2Q = _FakeAng2Q()


def _fake_make_hxrd_factory(captured: dict):
    """Returns a make_hxrd replacement that records its energy and returns a _FakeHXRD."""
    def _make_hxrd(self, energy):
        captured["energy"] = energy
        h = _FakeHXRD()
        captured["hxrd"] = h
        return h
    return _make_hxrd


class TestPixelQMap:
    """``make_hxrd`` is monkeypatched, so these tests run without xrayutilities."""
    def test_shape_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        mapper = PixelQMap(
            DiffractometerConfig(),
            DetectorHeader(
                cch1=50, cch2=80, pwidth1=0.1, pwidth2=0.1,
                distance=500, Nch1=128, Nch2=256,
            ),
        )
        angles = [
            np.array([0.0, 0.1, 0.2]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
        ]
        qx, qy, qz = mapper.pixel_q(angles, energy=11205.0)
        assert qx.shape == (3, 128, 256)
        assert qy.shape == (3, 128, 256)
        assert qz.shape == (3, 128, 256)
        assert captured["energy"] == 11205.0

    def test_init_area_receives_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        header = DetectorHeader(
            cch1=10.0, cch2=20.0, pwidth1=0.075, pwidth2=0.075,
            distance=830.0, Nch1=100, Nch2=200,
        )
        mapper = PixelQMap(DiffractometerConfig(), header)
        mapper.pixel_q([np.array([0.0])], energy=12000.0)

        kw = captured["hxrd"].Ang2Q.init_area_kwargs
        assert kw["cch1"] == 10.0
        assert kw["cch2"] == 20.0
        assert kw["pwidth1"] == 0.075
        assert kw["pwidth2"] == 0.075
        assert kw["distance"] == 830.0
        assert kw["Nch1"] == 100
        assert kw["Nch2"] == 200

    def test_roi_shifts_beam_centre_and_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        header = DetectorHeader(0.0, 0.0, 0.075, 0.075, 830.0,
                                Nch1=514, Nch2=1030)
        # Pre-shift cch so we can see the change
        header = DetectorHeader(257.0, 515.0, 0.075, 0.075, 830.0,
                                Nch1=514, Nch2=1030)

        mapper = PixelQMap(DiffractometerConfig(), header)
        mapper.pixel_q([np.array([0.0])], energy=12000.0,
                       roi=(50, 450, 100, 900))

        kw = captured["hxrd"].Ang2Q.init_area_kwargs
        assert kw["cch1"] == pytest.approx(257.0 - 50)
        assert kw["cch2"] == pytest.approx(515.0 - 100)
        assert kw["Nch1"] == 400  # 450 - 50
        assert kw["Nch2"] == 800  # 900 - 100

    def test_image_shape_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        # Header says 514x1030 but the actual image is 256x512 (e.g. detector binning)
        header = DetectorHeader(0, 0, 1, 1, 100, Nch1=514, Nch2=1030)
        mapper = PixelQMap(DiffractometerConfig(), header)
        mapper.pixel_q([np.array([0.0])], energy=12000.0,
                       image_shape=(256, 512))

        kw = captured["hxrd"].Ang2Q.init_area_kwargs
        assert kw["Nch1"] == 256
        assert kw["Nch2"] == 512

    def test_missing_detector_size_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        # Nch1/Nch2 are placeholder zeros — pixel_q should refuse to call init_area
        header = DetectorHeader(0, 0, 1, 1, 100, Nch1=0, Nch2=0)
        mapper = PixelQMap(DiffractometerConfig(), header)
        with pytest.raises(ValueError, match="Detector size not set"):
            mapper.pixel_q([np.array([0.0])], energy=12000.0)

    def test_ub_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            DiffractometerConfig, "make_hxrd",
            _fake_make_hxrd_factory(captured),
        )

        header = DetectorHeader(0, 0, 1, 1, 100, Nch1=8, Nch2=8)
        mapper = PixelQMap(DiffractometerConfig(), header)
        UB = np.diag([2.0, 3.0, 4.0])
        mapper.pixel_q([np.array([0.0])], energy=12000.0, UB=UB)
        np.testing.assert_allclose(captured["hxrd"].Ang2Q.area_kwargs["UB"], UB)
