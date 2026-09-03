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
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session


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


def test_pixel_q_omitted_ub_forwards_explicit_contiguous_float64_identity():
    observed = []

    class Ang2Q:
        def init_area(self, *_args, **_kwargs):
            return None

        def area(self, *_angles, UB=None, **_kwargs):
            observed.append(UB)
            shape = (1, 2, 3)
            return tuple(np.zeros(shape, dtype=np.float64) for _ in range(3))

    class HXRD:
        pass

    hxrd = HXRD()
    hxrd.Ang2Q = Ang2Q()

    class Diff:
        init_area_detrot = "x+"
        init_area_tiltazimuth = "z+"
        ang2q_kwargs = {}

        def make_hxrd(self, _energy):
            return hxrd

    header = DetectorHeader(
        cch1=1.0,
        cch2=1.0,
        pwidth1=0.1,
        pwidth2=0.1,
        distance=1.0,
        Nch1=2,
        Nch2=3,
    )
    PixelQMap(Diff(), header).pixel_q((np.array([0.0]),), 10_000.0)
    assert len(observed) == 1
    assert observed[0].dtype == np.dtype(np.float64)
    assert observed[0].flags.c_contiguous
    np.testing.assert_array_equal(observed[0], np.eye(3, dtype=np.float64))


def _direct_point_and_area(cfg, header, energy, angles, *, UB_marker):
    hxrd = cfg.make_hxrd(energy)
    point_kwargs = dict(cfg.ang2q_kwargs)
    area_kwargs = dict(cfg.ang2q_kwargs)
    if UB_marker is not None:
        point_kwargs["UB"] = UB_marker
        area_kwargs["UB"] = UB_marker
    point = tuple(
        np.asarray(value, dtype=np.float64)
        for value in hxrd.Ang2Q.point(*angles, **point_kwargs)
    )
    hxrd.Ang2Q.init_area(
        cfg.init_area_detrot,
        cfg.init_area_tiltazimuth,
        cch1=float(header.cch1),
        cch2=float(header.cch2),
        pwidth1=float(header.pwidth1),
        pwidth2=float(header.pwidth2),
        distance=float(header.distance),
        Nch1=int(header.Nch1),
        Nch2=int(header.Nch2),
    )
    area = tuple(
        np.asarray(value, dtype=np.float64)
        for value in hxrd.Ang2Q.area(
            *(np.asarray([angle], dtype=np.float64) for angle in angles),
            **area_kwargs,
        )
    )
    return point, area


def test_pinned_xu_omitted_ub_equals_identity_and_q_norm_is_analytic():
    pytest.importorskip("xrayutilities")
    cfg = DiffractometerConfig()
    header = _header()
    energy_eV = 10_000.0
    angles = (1.5, -2.0, 0.75, 24.0)
    with xu_runtime_session():
        omitted_point, omitted_area = _direct_point_and_area(
            cfg,
            header,
            energy_eV,
            angles,
            UB_marker=None,
        )
        identity_point, identity_area = _direct_point_and_area(
            cfg,
            header,
            energy_eV,
            angles,
            UB_marker=np.eye(3, dtype=np.float64),
        )
    for omitted, identity in zip(
        (*omitted_point, *omitted_area),
        (*identity_point, *identity_area),
        strict=True,
    ):
        np.testing.assert_array_equal(omitted, identity)

    wavelength_angstrom = 12_398.419843320026 / energy_eV
    expected_q = (
        4.0
        * np.pi
        * np.sin(np.deg2rad(abs(angles[-1])) / 2.0)
        / wavelength_angstrom
    )
    observed_q = float(np.linalg.norm(np.asarray(identity_point)))
    assert observed_q == pytest.approx(expected_q, rel=5e-13, abs=5e-13)


def test_pinned_xu_nonorthogonal_matrix_maps_hkl_back_to_cartesian_q():
    pytest.importorskip("xrayutilities")
    cfg = DiffractometerConfig()
    header = _header()
    matrix = np.array(
        (
            (1.2, 0.15, -0.08),
            (0.04, 0.91, 0.23),
            (-0.17, 0.06, 1.08),
        ),
        dtype=np.float64,
    )
    angles = (2.0, -1.0, 0.5, 17.0)
    with xu_runtime_session():
        q_point, q_area = _direct_point_and_area(
            cfg,
            header,
            12_000.0,
            angles,
            UB_marker=np.eye(3, dtype=np.float64),
        )
        hkl_point, hkl_area = _direct_point_and_area(
            cfg,
            header,
            12_000.0,
            angles,
            UB_marker=matrix,
        )
    for q_values, hkl_values in (
        (q_point, hkl_point),
        (q_area, hkl_area),
    ):
        q = np.stack(q_values)
        hkl = np.stack(hkl_values)
        np.testing.assert_allclose(
            np.einsum("ij,j...->i...", matrix, hkl),
            q,
            rtol=2e-13,
            atol=2e-13,
        )
