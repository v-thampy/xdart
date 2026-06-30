"""Regression: a fresh **Run** on a reloaded processed ``.nxs`` with a GENERIC
(unnamed) detector must NOT lose the detector pixel size.

The reload restore (``LiveScan._restore_calibration_from_group``) builds a
pixel-bearing ``AzimuthalIntegrator`` from persisted ``x_pixel_size`` /
``y_pixel_size`` for an unnamed/generic detector.  A fresh Run click adopts that
scan's geometry but seeds only the pixel-LESS ``PONI`` dataclass (detector NAME
only).  The wrangler thread's poni-identity rebuild block used to CLOBBER the
restored integrator with ``poni_to_integrator(self.poni)`` — which yields a pyFAI
detector with ``_pixel1`` = None for a generic detector — and the multi-worker
reduction then crashed in ``calc_cartesian_positions``
(``TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'``).

The fix carries the restored integrator (keyed on the adopted poni) from
``image_wrangler._adopt_loaded_scan_run_inputs`` to the thread, and
``imageThread._install_run_integrator`` REUSES it instead of rebuilding.  A
genuinely-new user-loaded ``.poni`` is a different object, so it still rebuilds.

Qt-free: ``_install_run_integrator`` touches only the scan's cache attributes and
``poni_to_integrator``; we drive it as an unbound method on a stand-in carrier.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pyFAI")


def _generic_pixel_bearing_ai():
    """A restored integrator for a GENERIC detector: real pixel size, no name."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    det = Detector(pixel1=73.242e-6, pixel2=73.242e-6, max_shape=(3072, 3072))
    ai = AzimuthalIntegrator(
        dist=0.15, poni1=0.1, poni2=0.1, detector=det, wavelength=1e-10)
    assert ai.detector._pixel1 == pytest.approx(73.242e-6)
    return ai


def _pixel_less_poni():
    """The seeded PONI a reloaded generic-detector scan adopts: NAME only."""
    from xrd_tools.core.containers import PONI

    return PONI(dist=0.15, poni1=0.1, poni2=0.1, rot1=0.0, rot2=0.0, rot3=0.0,
                wavelength=1e-10, detector="")  # '' => generic, no pixel info


def _carrier(poni, adopted_poni, adopted_ai, adopted_fi=None):
    """A stand-in for the wrangler thread carrying only the attributes
    ``_install_run_integrator`` reads."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import imageThread

    obj = SimpleNamespace(
        poni=poni,
        _adopted_poni=adopted_poni,
        _adopted_integrator=adopted_ai,
        _adopted_fiber_integrator=adopted_fi,
    )
    # Bind the real unbound method to the stand-in.
    obj._install_run_integrator = imageThread._install_run_integrator.__get__(obj)
    return obj


def test_run_reuses_adopted_pixel_bearing_integrator():
    """When the Run adopted the loaded scan's geometry, the helper REUSES the
    restored pixel-bearing AI instead of rebuilding a pixel-less one."""
    ai = _generic_pixel_bearing_ai()
    poni = _pixel_less_poni()

    carrier = _carrier(poni=poni, adopted_poni=poni, adopted_ai=ai)
    # A fresh LiveScan would start with no usable integrator; simulate the
    # worst case (already clobbered to a pixel-less rebuild).
    from xrd_tools.integrate.calibration import poni_to_integrator
    scan = SimpleNamespace(
        name="reload_generic",
        _cached_integrator=poni_to_integrator(poni),  # pixel-less (pre-fix state)
        _cached_poni=None,
        _cached_fiber_integrator=None,
    )
    assert scan._cached_integrator.detector._pixel1 is None  # pre-fix: would crash

    carrier._install_run_integrator(scan)

    # The fix: the restored pixel-bearing AI is reinstated; pixel size survives.
    assert scan._cached_integrator is ai
    assert scan._cached_integrator.detector._pixel1 == pytest.approx(73.242e-6)
    assert scan._cached_integrator.detector._pixel2 == pytest.approx(73.242e-6)
    assert scan._cached_poni is poni


def test_user_loaded_poni_still_rebuilds():
    """A genuinely-new user-loaded .poni (a DIFFERENT object than the adopted
    poni) must NOT reuse the adopted integrator — it rebuilds from that poni."""
    ai = _generic_pixel_bearing_ai()
    adopted_poni = _pixel_less_poni()
    # The user then loads their own calibration — a distinct object.
    user_poni = _pixel_less_poni()
    assert user_poni is not adopted_poni

    carrier = _carrier(poni=user_poni, adopted_poni=adopted_poni, adopted_ai=ai)
    scan = SimpleNamespace(
        name="user_poni", _cached_integrator=None, _cached_poni=None,
        _cached_fiber_integrator=object(),
    )
    carrier._install_run_integrator(scan)

    # Rebuilt from the user's poni, NOT the adopted integrator.
    assert scan._cached_integrator is not ai
    assert scan._cached_poni is user_poni
    # Fiber cache invalidated on a genuine rebuild.
    assert scan._cached_fiber_integrator is None


def _readiness_host(scan, *, poni=None, img_file="/tmp/raw_0001.tif"):
    """Minimal Qt-free host carrying just what _inputs_valid touches."""
    from types import MethodType

    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler

    status = {"text": None}
    host = SimpleNamespace(
        poni=poni,
        img_file=img_file,
        img_dir="",
        img_ext="",
        scan=scan,
        thread=SimpleNamespace(),
        parameters=SimpleNamespace(names=[]),
        stitch_mode=False,
        _set_status_text=lambda t: status.__setitem__("text", t),
    )
    host._adopt_loaded_scan_run_inputs = MethodType(
        imageWrangler._adopt_loaded_scan_run_inputs, host)
    host._inputs_valid = MethodType(imageWrangler._inputs_valid, host)
    return host, status


def test_inputs_valid_blocks_pixel_less_reloaded_scan():
    """A reloaded scan with usable geometry but NO restored integrator (the
    pixel-less / generic-no-pixel-size case) is REFUSED up front with a clear
    message — it must not crash a multi-worker Run mid-write."""

    class Frames:
        index = [1]

        def __getitem__(self, idx):
            return SimpleNamespace(
                source_file="/tmp/raw_0001.tif",
                _resolved_source_path=lambda: "/tmp/raw_0001.tif",
            )

    pixel_less_poni = _pixel_less_poni()
    # Restore that bailed: poni present, but NO usable integrator.
    scan = SimpleNamespace(
        _cached_poni=pixel_less_poni, _cached_integrator=None, frames=Frames())
    host, status = _readiness_host(scan, poni=None)

    assert host._inputs_valid() is False
    assert "without detector pixel sizes" in status["text"]


def test_inputs_valid_allows_reloaded_scan_with_integrator():
    """The same adoption path SUCCEEDS when the reloaded scan restored a
    pixel-bearing integrator (the normal generic-detector reload)."""
    import os

    raw = "/tmp/xrd_test_run_integrator_carry_raw_0001.tif"
    with open(raw, "wb") as fh:
        fh.write(b"placeholder")

    class Frames:
        index = [1]

        def __getitem__(self, idx):
            return SimpleNamespace(
                source_file=raw, _resolved_source_path=lambda: raw)

    try:
        poni = _pixel_less_poni()
        scan = SimpleNamespace(
            _cached_poni=poni, _cached_integrator=_generic_pixel_bearing_ai(),
            frames=Frames())
        host, status = _readiness_host(scan, poni=None, img_file="")

        assert host._inputs_valid() is True
        assert host.poni is poni
        # Adoption carried the restored integrator to the thread, keyed on poni.
        assert host.thread._adopted_poni is poni
        assert host.thread._adopted_integrator is scan._cached_integrator
    finally:
        os.path.exists(raw) and os.remove(raw)


def test_no_adoption_falls_back_to_poni_rebuild():
    """Normal poni-file Run (no adoption): rebuild from self.poni as before."""
    from pyFAI.detectors import detector_factory
    from xrd_tools.core.containers import PONI

    # A NAMED detector resolves via the pyFAI registry on rebuild.
    named = detector_factory("Eiger4M")
    poni = PONI(dist=0.15, poni1=0.1, poni2=0.1, wavelength=1e-10,
                detector="Eiger4M")

    carrier = _carrier(poni=poni, adopted_poni=None, adopted_ai=None)
    scan = SimpleNamespace(
        name="named", _cached_integrator=None, _cached_poni=None,
        _cached_fiber_integrator=None,
    )
    carrier._install_run_integrator(scan)

    assert scan._cached_integrator is not None
    # Eiger4M has a known pixel size from the registry -> not None.
    assert scan._cached_integrator.detector._pixel1 == pytest.approx(named.pixel1)
    assert scan._cached_poni is poni
