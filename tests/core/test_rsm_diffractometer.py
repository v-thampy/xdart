"""P6.4 — RSM consumes the canonical Diffractometer (ADR-0007), a byte-equal
drop-in for the legacy DiffractometerConfig.

The unification gate: ``PixelQMap`` driven by a ``Diffractometer`` produces
**bit-identical** per-pixel q to the same geometry expressed as the legacy
``DiffractometerConfig`` — so swapping RSM's geometry source onto the one
canonical object changes nothing.  The ``from_diffractometer_config`` /
``to_diffractometer_config`` bridges carry every load-bearing field (the circle
stacks, ``r_i``, the camera detrot/tiltazimuth, the HXRD refs, the qconv/ang2q
kwargs); this pins them through ``pixel_q`` (the actual RSM call), not just at
the QConversion layer.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.geometry import (
    DetectorHeader,
    Diffractometer,
    DiffractometerConfig,
    PixelQMap,
)


def _header():
    return DetectorHeader(cch1=30, cch2=40, pwidth1=0.172, pwidth2=0.172,
                          distance=500.0, Nch1=48, Nch2=60)


def _angles(n_circ):
    """One distinct sample angle + zeros — n_circ arrays for n_circ circles."""
    return [np.array([10.0, 20.0, 30.0])] + [np.zeros(3) for _ in range(n_circ - 1)]


# legacy configs spanning the convention knobs the bridge must carry
_CONFIGS = [
    DiffractometerConfig(),  # the default 3-sample / 1-detector
    DiffractometerConfig(init_area_detrot="z-", init_area_tiltazimuth="x-",
                         hxrd_q=(0.0, 0.0, 1.0)),
    DiffractometerConfig(sample_rot=("z-", "y+"), detector_rot=("z-",),
                         r_i=(0.0, 1.0, 0.0)),
    DiffractometerConfig(sample_rot=("x+", "z-", "y+", "z-"),
                         detector_rot=("x+", "z-")),  # psic-shaped
]


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_diffractometer_pixel_q_is_byte_equal_to_legacy_config(cfg):
    pytest.importorskip("xrayutilities")
    diff = Diffractometer.from_diffractometer_config(cfg)
    h = _header()
    UB = np.eye(3)
    n = len(cfg.sample_rot) + len(cfg.detector_rot)
    angles = _angles(n)
    qa = PixelQMap(diff_config=cfg, header=h).pixel_q(angles, 10000.0, UB=UB)
    qb = PixelQMap(diff_config=diff, header=h).pixel_q(angles, 10000.0, UB=UB)
    for a, b in zip(qa, qb):
        np.testing.assert_array_equal(a, b)   # byte-equal, not just close


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_roundtrip_lower_then_lift_preserves_pixel_q(cfg):
    """``cfg → Diffractometer → cfg'`` round-trips the geometry exactly."""
    pytest.importorskip("xrayutilities")
    diff = Diffractometer.from_diffractometer_config(cfg)
    cfg2 = diff.to_diffractometer_config()
    h = _header()
    UB = np.eye(3)
    angles = _angles(len(cfg.sample_rot) + len(cfg.detector_rot))
    qa = PixelQMap(diff_config=cfg, header=h).pixel_q(angles, 10000.0, UB=UB)
    qc = PixelQMap(diff_config=cfg2, header=h).pixel_q(angles, 10000.0, UB=UB)
    for a, c in zip(qa, qc):
        np.testing.assert_array_equal(a, c)


def test_canonical_psic_diffractometer_drives_pixel_q():
    """A bare canonical ``Diffractometer.psic()`` is a working PixelQMap geometry
    (the drop-in interface: make_hxrd / init_area_detrot / ang2q_kwargs)."""
    pytest.importorskip("xrayutilities")
    diff = Diffractometer.psic()  # 4 sample + 2 detector circles
    h = _header()
    angles = [np.array([1.0, 2.0]), np.array([0.3, 0.3]), np.zeros(2),
              np.zeros(2), np.array([3.0, 6.0]), np.array([10.0, 20.0])]
    qx, qy, qz = PixelQMap(diff_config=diff, header=h).pixel_q(
        angles, 10000.0, UB=np.eye(3))
    assert qx.shape == (2, h.Nch1, h.Nch2)
    assert np.isfinite(qx).all() and np.isfinite(qz).all()
