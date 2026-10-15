"""COVERAGE-SUPPORT-20261013 frozen headless oracle (Claude, October 13, after Codex's ratification).

Rows K1-K9 of the kickoff (handoffs/coverage_support_2026-10-13.md, "Finite oracle"). Every expected value is
computed here independently of the kernel: exact circle/rectangle arc measures for perpendicular flat detectors,
an independent grazing-incidence rotation sampled densely around each ring, and a sub-pixel raster union built
from pyFAI's own pixel geometry for moving Stitch poses. New APIs are imported inside each row, so on the parent a
missing API fails only its own row (an interface failure, recorded as such). Tolerances are the kickoff's: a
fraction error <= 0.005 of the requested window, one further oversampling step changes <= 0.0025, complete or
absent bins <= 1e-6.
"""
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

TOL = 0.005
CONVERGE = 0.0025
EXACT = 1e-6


# ------------------------------------------------------------------------------------------- independent models

def _planar(shape=(400, 500), pixel=1e-4, dist=0.1, poni=(0.020013, 0.025007), wavelength=1e-10, detector=None):
    """A perpendicular flat detector (rot = 0): the beam meets it at the PONI."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    det = detector if detector is not None else Detector(pixel1=pixel, pixel2=pixel, max_shape=shape)
    return AzimuthalIntegrator(dist=dist, poni1=poni[0], poni2=poni[1], detector=det, wavelength=wavelength)


def _lab(ai, rows, cols):
    """Lab (x, y) of pixel-corner coordinates through pyFAI's own forward geometry."""
    _z, y, x = ai.calc_pos_zyx(d1=np.asarray(rows, float) - .5, d2=np.asarray(cols, float) - .5)
    return np.asarray(x), np.asarray(y)


def _rect(ai, r0, r1, c0, c1):
    """Lab rectangle covered by pixels [r0, r1) x [c0, c1) of a perpendicular detector."""
    x, y = _lab(ai, [r0, r0, r1, r1], [c0, c1, c0, c1])
    assert np.ptp(x[[0, 2]]) < 1e-12 and np.ptp(y[[0, 1]]) < 1e-12, "fixture must be perpendicular"
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _inside(rect, x, y):
    x0, x1, y0, y1 = rect
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _arcs(radius, rect, holes=(), windows=None):
    """Exact measure (radians) of the beam-centred circle inside ``rect``, outside ``holes``, within ``windows``."""
    cuts = [-math.pi, math.pi]
    for x0, x1, y0, y1 in (rect, *holes):
        for value in (x0, x1):
            if abs(value) < radius:
                angle = math.acos(value / radius)
                cuts += [angle, -angle]
        for value in (y0, y1):
            if abs(value) < radius:
                angle = math.asin(value / radius)
                cuts += [angle, math.pi - angle if angle >= 0 else -math.pi - angle]
    for lo, hi in windows or ():
        cuts += [value for value in (lo, hi) if -math.pi <= value <= math.pi]
    cuts = np.unique(np.clip(cuts, -math.pi, math.pi))
    middle = .5 * (cuts[:-1] + cuts[1:])
    x, y = radius * np.cos(middle), radius * np.sin(middle)
    keep = _inside(rect, x, y)
    for hole in holes:
        keep &= ~_inside(hole, x, y)
    if windows is not None:
        keep &= np.any([(middle >= lo) & (middle <= hi) for lo, hi in windows], axis=0)
    return float(np.sum(np.diff(cuts)[keep]))


def _two_theta(values, unit, wavelength_m):
    values = np.asarray(values, float)
    if unit == "2th_deg":
        return np.radians(values)
    if unit == "2th_rad":
        return values
    scale = {"q_A^-1": 1e10, "q_nm^-1": 1e9}[unit]
    return 2 * np.arcsin(values * wavelength_m * scale / (4 * np.pi))


def _expected(edges, unit, ai, rect, holes=(), windows=None, samples=400):
    """Bin mean (degrees) over each native bin, sampled uniformly in the native unit."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        native = lo + (np.arange(samples) + .5) * (hi - lo) / samples
        radii = ai.dist * np.tan(_two_theta(native, unit, ai.wavelength))
        out.append(math.degrees(np.mean([_arcs(r, rect, holes, windows) for r in radii])))
    return np.asarray(out)


def _centre(edges, unit, ai, rect, holes=(), windows=None):
    centres = .5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    radii = ai.dist * np.tan(_two_theta(centres, unit, ai.wavelength))
    return np.asarray([math.degrees(_arcs(r, rect, holes, windows)) for r in radii])


def _deg(*pairs):
    return [(math.radians(lo), math.radians(hi)) for lo, hi in pairs]


def _orient(x, y, orientation):
    """pyFAI's EXIF sample orientation (pyFAI.units.rotate_sample_orientation), written out independently."""
    return {1: (x, y), 2: (-x, y), 3: (-x, -y), 4: (x, -y),
            5: (-y, -x), 6: (-y, x), 7: (y, x), 8: (y, -x)}[orientation]


def _gi_chi(x, y, z, incident, tilt, orientation):
    """chi_GI = atan2(sign(q_horz)·sqrt(q_beam²+q_horz²), q_vert) after Ry(incident) then Rx(-tilt)."""
    h, v = _orient(x, y, orientation)
    r = np.sqrt(x * x + y * y + z * z)
    qb, qh, qv = z / r - 1.0, h / r, v / r
    ca, sa, ct, st = math.cos(incident), math.sin(incident), math.cos(-tilt), math.sin(-tilt)
    b1, h1, v1 = ca * qb + sa * qv, qh, -sa * qb + ca * qv
    b2, h2, v2 = b1, ct * h1 - st * v1, st * h1 + ct * v1
    return np.arctan2(np.sign(h2) * np.sqrt(b2 * b2 + h2 * h2), v2), np.sign(h2)


def _gi_ring(ai, mask, two_theta, incident, tilt, orientation, window=None, n=100_000):
    """Degrees of chi_GI reached by unmasked detector points on one 2θ cone (perpendicular detector)."""
    psi = (np.arange(n) + .5) * 2 * np.pi / n - np.pi
    radius = ai.dist * math.tan(two_theta)
    x, y = radius * np.cos(psi), radius * np.sin(psi)
    pixel1, pixel2 = ai.detector.pixel1, ai.detector.pixel2
    rows, cols = (y + ai.poni1) / pixel1, (x + ai.poni2) / pixel2
    height, width = ai.detector.shape
    covered = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if mask is not None:
        safe = covered.copy()
        covered[safe] = ~mask[rows[safe].astype(int), cols[safe].astype(int)]
    chi, side = _gi_chi(x, y, np.full_like(x, ai.dist), incident, tilt, orientation)
    a, b = chi, np.roll(chi, -1)
    pair = covered & np.roll(covered, -1) & (side == np.roll(side, -1)) & (side != 0)
    lo, hi = np.minimum(a, b)[pair], np.maximum(a, b)[pair]
    if window is not None:
        lo, hi = np.maximum(lo, window[0]), np.minimum(hi, window[1])
    return math.degrees(float(np.sum(np.clip(hi - lo, 0, None))))


def _gi_expected(edges, unit, ai, mask, incident, tilt, orientation, window=None, radii=48):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        native = lo + (np.arange(radii) + .5) * (hi - lo) / radii
        out.append(np.mean([_gi_ring(ai, mask, t, incident, tilt, orientation, window)
                            for t in _two_theta(native, unit, ai.wavelength)]))
    return np.asarray(out)


def _ring_union(poses, edges, unit, wavelength_m, bins, lines=12, samples=50_000):
    """Independent union estimate for flat detector poses: rays around each ring meet each pose's detector plane.

    For every (ai, mask) pose the detector plane comes from pyFAI's forward geometry at three pixel corners; each
    ray (2θ, χ) is intersected with it, mapped to pixel coordinates and kept when it lands on an unmasked pixel.
    The union is the share of χ samples any pose reaches; ``lines`` radii per bin, uniform in the native unit.
    """
    chi = (np.arange(samples) + .5) * 2 * np.pi / samples - np.pi
    planes = []
    for ai, mask in poses:
        _z, y, x = ai.calc_pos_zyx(d1=np.array([-.5, .5, -.5]), d2=np.array([-.5, -.5, .5]))
        z = _z
        p00 = np.array([x[0], y[0], z[0]])
        e_row = np.array([x[1], y[1], z[1]]) - p00
        e_col = np.array([x[2], y[2], z[2]]) - p00
        planes.append((p00, e_row, e_col, np.cross(e_row, e_col), ai.detector.shape, mask))
    out = np.zeros(len(bins))
    for slot, b in enumerate(bins):
        lo, hi = edges[b], edges[b + 1]
        native = lo + (np.arange(lines) + .5) * (hi - lo) / lines
        values = []
        for tth in _two_theta(native, unit, wavelength_m):
            ray = np.stack((np.sin(tth) * np.cos(chi), np.sin(tth) * np.sin(chi), np.full_like(chi, np.cos(tth))), 1)
            covered = np.zeros(samples, bool)
            for p00, e_row, e_col, normal, (height, width), mask in planes:
                reach = ray @ normal
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = (p00 @ normal) / reach
                ahead = np.isfinite(scale) & (scale > 0)
                point = ray[ahead] * scale[ahead, None] - p00
                gram = np.array([[e_row @ e_row, e_row @ e_col], [e_row @ e_col, e_col @ e_col]])
                row, col = np.linalg.solve(gram, np.stack((point @ e_row, point @ e_col))).reshape(2, -1)
                inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
                hit = np.flatnonzero(ahead)[inside]
                if mask is not None:
                    hit = hit[~mask[row[inside].astype(int), col[inside].astype(int)]]
                covered[hit] = True
            values.append(360.0 * covered.mean())
        out[slot] = np.mean(values)
    return out


def _request(ai, mask, unit, edges, window, *, convention="detector_chi", gi=None, poses=None):
    from xrd_tools.analysis.angular_support import AngularSupportRequest, SupportPose
    poses = poses or (SupportPose.from_integrator(ai, mask, gi=gi),)
    return AngularSupportRequest(poses=tuple(poses), convention=convention, radial_unit=unit,
                                 bin_edges=np.asarray(edges, float), wavelength_m=ai.wavelength, window=window)


def _gap_holes(ai):
    """Lab rectangles of every fully masked detector row/column run (module gaps)."""
    mask = ai.detector.mask.astype(bool)
    height, width = mask.shape
    holes = []
    for axis, full in ((0, mask.all(axis=1)), (1, mask.all(axis=0))):
        index = np.flatnonzero(full)
        if not len(index):
            continue
        for run in np.split(index, np.flatnonzero(np.diff(index) > 1) + 1):
            lo, hi = int(run[0]), int(run[-1]) + 1
            holes.append(_rect(ai, lo, hi, 0, width) if axis == 0 else _rect(ai, 0, height, lo, hi))
    return holes


# --------------------------------------------------------------------------------------------------------- K1

def test_k1_full_partial_zero_unknown():
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    ai = _planar()
    rect = _rect(ai, 0, 400, 0, 500)
    edges = np.linspace(2.0, 24.0, 23)
    full_turn = admitted_window(None, reducer="int_standard")
    assert full_turn.denominator_deg == 360.0 and full_turn.reason is None
    profile = angular_support(_request(ai, None, "2th_deg", edges, full_turn))
    assert profile.available and profile.origin is SupportOrigin.RUN and profile.window_deg == 360.0
    expected = _expected(edges, "2th_deg", ai, rect)
    full, empty = expected >= 360 - 1e-9, expected <= 1e-9
    partial = ~full & ~empty
    assert full.sum() >= 3 and empty.sum() >= 3 and partial.sum() >= 4, "fixture spans all three regimes"
    got = profile.supported_deg
    assert got.shape == (22,) and not got.flags.writeable
    assert np.all(np.abs(got[full] - 360.0) <= 360 * EXACT)
    assert np.all(got[empty] <= 360 * EXACT)
    assert np.all(np.abs(got[partial] - expected[partial]) <= TOL * 360)
    np.testing.assert_allclose(profile.fraction, got / 360.0)
    explicit = angular_support(_request(ai, None, "2th_deg", edges, admitted_window((-180., 180.), reducer="int_standard")))
    np.testing.assert_allclose(explicit.supported_deg, got, atol=1e-9)

    # unknown is a named reason, never a zero profile
    unknown = angular_support(_request(ai, None, "r_mm", np.linspace(1., 20., 11), full_turn))
    assert not unknown.available and unknown.supported_deg is None and unknown.fraction is None
    assert unknown.reason is SupportReason.UNSUPPORTED_UNIT

    # the requested window is the denominator (D2)
    quarter = admitted_window((-45.0, 45.0), reducer="int_standard")
    assert quarter.denominator_deg == 90.0
    windowed = angular_support(_request(ai, None, "2th_deg", edges, quarter))
    assert windowed.window_deg == 90.0
    assert np.all(np.abs(windowed.supported_deg - _expected(edges, "2th_deg", ai, rect, windows=_deg((-45, 45))))
                  <= TOL * 90)
    np.testing.assert_allclose(windowed.fraction, windowed.supported_deg / 90.0)

    # equal or reversed endpoints are degenerate, not pyFAI's reinterpreted full turn
    for bad in ((30., 30.), (40., 30.)):
        window = admitted_window(bad, reducer="int_standard")
        assert window.reason is SupportReason.DEGENERATE_WINDOW
        refused = angular_support(_request(ai, None, "2th_deg", edges, window))
        assert refused.reason is SupportReason.DEGENERATE_WINDOW and refused.supported_deg is None

    # D1: the bin mean, not the centre sample, across a tangency (the detector's right edge)
    edge_x = rect[1]
    lo = math.degrees(math.atan((edge_x - 2e-4) / ai.dist))
    hi = math.degrees(math.atan((edge_x + 1e-3) / ai.dist))
    tangent = np.array([lo, hi])
    mean = _expected(tangent, "2th_deg", ai, rect, samples=4000)
    centre = _centre(tangent, "2th_deg", ai, rect)
    assert abs(mean[0] - centre[0]) > 2 * TOL * 360, "fixture discriminates the two metrics"
    binned = angular_support(_request(ai, None, "2th_deg", tangent, full_turn)).supported_deg
    assert abs(binned[0] - mean[0]) <= TOL * 360 < abs(binned[0] - centre[0])


def test_k1b_units_weight_the_native_bin():
    """Each unit's bins are a reducer's uniform grid in that unit; a q bin is a q average, not a relabelled 2θ one."""
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    ai = _planar()
    rect = _rect(ai, 0, 400, 0, 500)
    window = admitted_window(None, reducer="int_standard")
    two_theta = np.linspace(9.0, 21.0, 7)

    def q_of(tth, scale):
        return 4 * np.pi * np.sin(np.radians(tth) / 2) / (ai.wavelength * scale)

    for unit, edges in (("2th_deg", two_theta), ("2th_rad", np.radians(two_theta)),
                        ("q_A^-1", np.linspace(q_of(9.0, 1e10), q_of(21.0, 1e10), 7)),
                        ("q_nm^-1", np.linspace(q_of(9.0, 1e9), q_of(21.0, 1e9), 7))):
        got = angular_support(_request(ai, None, unit, edges, window)).supported_deg
        assert np.all(np.abs(got - _expected(edges, unit, ai, rect)) <= TOL * 360), unit
    # one physical bin, two measures: the q mean weights by d(sin θ), the 2θ mean by d(2θ)
    wide_tth = np.array([9.0, 21.0])
    wide_q = q_of(wide_tth, 1e10)
    expected_q = _expected(wide_q, "q_A^-1", ai, rect, samples=4000)[0]
    expected_tth = _expected(wide_tth, "2th_deg", ai, rect, samples=4000)[0]
    assert abs(expected_q - expected_tth) > 0.05, "fixture: the two measures differ"
    got_q = angular_support(_request(ai, None, "q_A^-1", wide_q, window)).supported_deg[0]
    got_tth = angular_support(_request(ai, None, "2th_deg", wide_tth, window)).supported_deg[0]
    assert np.sign(got_q - got_tth) == np.sign(expected_q - expected_tth)
    assert abs((got_q - got_tth) - (expected_q - expected_tth)) <= .5 * abs(expected_q - expected_tth)


# --------------------------------------------------------------------------------------------------------- K2

def test_k2_mask_and_value_independence():
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    ai = _planar(dist=0.08, poni=(0.050013, 0.040007), detector=detector_factory("Pilatus300k"))
    height, width = ai.detector.shape
    rect = _rect(ai, 0, height, 0, width)
    holes = _gap_holes(ai)
    assert len(holes) == 2, "Pilatus 300K has two horizontal module gaps"
    user = np.zeros((height, width), bool)
    user[:, 300:306] = True
    holes_user = holes + [_rect(ai, 0, height, 300, 306)]
    edges = np.linspace(20.0, 40.0, 11)
    window = admitted_window(None, reducer="int_standard")
    applied = user | ai.detector.mask.astype(bool)
    got = angular_support(_request(ai, applied, "2th_deg", edges, window)).supported_deg
    assert np.all(np.abs(got - _expected(edges, "2th_deg", ai, rect, holes_user)) <= TOL * 360)
    without_user = angular_support(_request(ai, ai.detector.mask.astype(bool), "2th_deg", edges, window))
    assert np.all(np.abs(without_user.supported_deg - _expected(edges, "2th_deg", ai, rect, holes)) <= TOL * 360)
    assert np.max(without_user.supported_deg - got) > TOL * 360, "the user band is resolved"


def _int_scan(ai, images, name="k", mask=None):
    from xrd_tools.reduction import Frame, Scan
    return Scan(name, [Frame(index, image=image, mask=mask) for index, image in enumerate(images)], integrator=ai)


#: pass no angular support policy at all, so the API's own default applies (INT-PERF-REGRESSION-20261014 A4)
_OMITTED = object()


def _explicit(policy):
    """Coverage science asks for recording explicitly (Off by default since INT-PERF-REGRESSION-20261014): None is
    explicit On, ``_OMITTED`` passes nothing, any other policy is passed as given."""
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    if policy is _OMITTED:
        return {}
    return {"angular_support": AngularSupportPolicy(enabled=True) if policy is None else policy}


def _run(plan, scan, policy=None):
    from xrd_tools.reduction import run_reduction
    return run_reduction(plan, scan, execution="chunked", executor=False, **_explicit(policy))


def test_k2b_value_conditioning_never_reduces_support_and_intensity_is_unchanged():
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.reduction import Integration1DPlan, ReductionPlan
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    static = np.zeros((120, 160), bool)
    static[:, 40:44] = True
    base = np.full((120, 160), 50.0, np.float32)
    rough = base.copy()
    rough[10:30, 10:60] = 0.0
    rough[60:70, 90:140] = -5.0
    rough[80:100, 20:40] = 65535.0
    flat = np.ones((120, 160), np.float32)
    flat[100:110, 100:150] = np.nan
    dark = np.zeros((120, 160), np.float32)
    dark[0:10, 120:160] = np.nan
    plain = ReductionPlan(integration_1d=Integration1DPlan(npt=48, unit="2th_deg", method="csr"), mask=static)
    plain_support = _run(plain, _int_scan(ai, [base])).frames[0].angular_support_1d
    assert plain_support.available
    for plan, images in (
        (plain, [rough]),
        (replace(plain, threshold_min=1.0, threshold_max=1000.0), [rough]),
        (replace(plain, mask_saturation=True, saturation=65535), [rough]),
        (replace(plain, flat=flat), [base]),
        (replace(plain, dark=dark), [base]),
    ):
        enabled = _run(plan, _int_scan(ai, images))
        disabled = _run(plan, _int_scan(ai, images), AngularSupportPolicy(enabled=False))
        support = enabled.frames[0].angular_support_1d
        np.testing.assert_array_equal(support.supported_deg, plain_support.supported_deg)
        for field in ("radial", "intensity"):
            assert getattr(enabled.frames[0].result_1d, field).tobytes() == \
                getattr(disabled.frames[0].result_1d, field).tobytes()
        from xrd_tools.core.angular_support import SupportReason
        assert disabled.frames[0].angular_support_1d.reason is SupportReason.RECORDING_DISABLED  # decision 6


def test_k2c_the_applied_mask_includes_bound_and_degraded_cases():
    from pyFAI.detectors import Detector
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    from xrd_tools.reduction import Integration1DPlan, ReductionPlan
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    band = np.zeros((120, 160), bool)
    band[:, 100:110] = True
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="2th_deg", method="csr"), mask=band)
    image = np.full((120, 160), 10.0, np.float32)
    bound = _run(plan, _int_scan(ai, [image])).frames[0].angular_support_1d
    unbound = _run(replace(plan, mask=None), _int_scan(ai, [image], mask=band)).frames[0].angular_support_1d
    np.testing.assert_allclose(bound.supported_deg, unbound.supported_deg, atol=1e-9)
    edges = bound.edges
    direct = angular_support(_request(ai, band, "2th_deg", edges, admitted_window(None, reducer="int_standard")))
    np.testing.assert_allclose(bound.supported_deg, direct.supported_deg, atol=1e-9)
    # a detector mask the reducer cannot union (shape mismatch) is not applied, so it is not geometry either
    degraded = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=(120, 160))
    block = np.zeros((121, 160), np.int8)
    block[:, :80] = 1
    degraded.mask = block
    odd = _planar(shape=(120, 160), poni=(0.006013, 0.008007), detector=degraded)
    captured = _run(plan, _int_scan(odd, [image])).frames[0].angular_support_1d
    np.testing.assert_allclose(captured.supported_deg, bound.supported_deg, atol=1e-9)


# --------------------------------------------------------------------------------------------------------- K3

def test_k3_periodic_footprints_and_tangency():
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    from xrd_tools.core.angular_support import SupportReason
    ai = _planar()
    rect = _rect(ai, 0, 400, 0, 500)
    strip = _rect(ai, 0, 400, 140, 143)
    edges = np.linspace(4.0, 22.0, 19)
    full = admitted_window(None, reducer="int_standard")
    band = np.zeros((400, 500), bool)
    band[:, 140:143] = True
    got = angular_support(_request(ai, band, "2th_deg", edges, full)).supported_deg
    assert np.all(np.abs(got - _expected(edges, "2th_deg", ai, rect, [strip])) <= TOL * 360)
    # tangent bins at the strip's near edge discriminate the bin mean from the centre sample
    near = strip[1] if abs(strip[1]) < abs(strip[0]) else strip[0]
    lo = math.degrees(math.atan((abs(near) - 1e-4) / ai.dist))
    hi = math.degrees(math.atan((abs(near) + 5e-4) / ai.dist))
    tangent = np.array([lo, hi])
    mean = _expected(tangent, "2th_deg", ai, rect, [strip], samples=4000)
    assert abs(angular_support(_request(ai, band, "2th_deg", tangent, full)).supported_deg[0] - mean[0]) \
        <= TOL * 360

    # the χ seam at ±180°: a ring crossing it is one continuous footprint, no fake gap
    seam = admitted_window((90., 180.), reducer="int_standard")
    seamed = angular_support(_request(ai, None, "2th_deg", edges, seam)).supported_deg
    assert np.all(np.abs(seamed - _expected(edges, "2th_deg", ai, rect, windows=_deg((90, 180)))) <= TOL * 90)

    # a nonzero χ offset shifts the requested window into pyFAI's raw frame before pyFAI normalises it
    shifted = admitted_window((10., 100.), reducer="int_standard", offset_deg=30.)
    assert shifted.reason is None and shifted.denominator_deg == 90.
    got = angular_support(_request(ai, None, "2th_deg", edges, shifted)).supported_deg
    assert np.all(np.abs(got - _expected(edges, "2th_deg", ai, rect, windows=_deg((-20, 70)))) <= TOL * 90)

    # an offset-shifted full turn collapses in pyFAI (p01): reachable directions are lost → named refusal
    collapsed = admitted_window((-180., 180.), reducer="int_standard", offset_deg=30.)
    refused = angular_support(_request(ai, None, "2th_deg", edges, collapsed))
    assert refused.reason is SupportReason.WINDOW_TRUNCATED and refused.supported_deg is None
    # beyond ±180°: pyFAI keeps [160°, 180°) of (-200°, -160°); the lost [-180°, -160°) is reachable here
    beyond = admitted_window((-200., -160.), reducer="int_standard")
    assert angular_support(_request(ai, None, "2th_deg", edges, beyond)).reason is SupportReason.WINDOW_TRUNCATED
    assert angular_support(_request(ai, None, "2th_deg", edges, admitted_window(
        (170., 190.), reducer="int_standard"))).reason is SupportReason.WINDOW_TRUNCATED
    # ...but a detector that never reaches the lost directions loses nothing: harmless, still the 40° denominator
    side = _planar(shape=(200, 300), poni=(-0.002013, 0.031007))
    side_rect = _rect(side, 0, 200, 0, 300)
    assert side_rect[1] < 0 < side_rect[2], "detector lies at negative lab x, above the beam (χ in (90°, 180°))"
    side_edges = np.linspace(4.0, 15.0, 12)
    assert np.max(_expected(side_edges, "2th_deg", side, side_rect, windows=_deg((-180, -160)))) == 0.0
    kept_expected = _expected(side_edges, "2th_deg", side, side_rect, windows=_deg((160, 180)))
    assert np.max(kept_expected) > 5.0, "fixture reaches the admitted part"
    kept = angular_support(_request(side, None, "2th_deg", side_edges,
                                    admitted_window((160., 200.), reducer="int_standard")))
    assert kept.available and kept.window_deg == 40.
    assert np.all(np.abs(kept.supported_deg - kept_expected) <= TOL * 40)


def test_k3b_oversampling_converges():
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    ai = _planar()
    band = np.zeros((400, 500), bool)
    band[:, 140:143] = True
    band[300:303, :] = True
    edges = np.linspace(4.0, 22.0, 37)
    request = _request(ai, band, "2th_deg", edges, admitted_window(None, reducer="int_standard"))
    coarse = angular_support(request, oversample=8).supported_deg
    fine = angular_support(request, oversample=16).supported_deg
    assert np.max(np.abs(coarse - fine)) <= CONVERGE * 360


# --------------------------------------------------------------------------------------------------------- K4

@pytest.mark.parametrize("orientation", (1, 4))
def test_k4_gi_rotation_wedge_and_energy(orientation):
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    from xrd_tools.core.angular_support import SupportReason
    import pyFAI.units as units
    ai = _planar(shape=(300, 400), poni=(0.005013, 0.020007))
    incident, tilt = math.radians(2.0), math.radians(0.0)
    # the independent rotation is pyFAI's own convention (checked on random points)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-0.02, 0.02, 64), rng.uniform(-0.02, 0.02, 64)
    z = np.full(64, ai.dist)
    mine, _side = _gi_chi(x, y, z, incident, math.radians(1.5), orientation)
    theirs = units.eq_chi_gi(x, y, z, ai.wavelength, incident, math.radians(1.5), orientation)
    np.testing.assert_allclose(mine, theirs, atol=1e-12)

    # the column that straddles q_horz = 0 (the beam's vertical) stays unmasked: the missing wedge stays empty
    mask = np.zeros((300, 400), bool)
    mask[:, 300:305] = True
    edges = np.linspace(6.0, 14.0, 9)
    gi = (2.0, 0.0, orientation)
    window = admitted_window(None, reducer="int_gi_total_q")
    got = angular_support(_request(ai, mask, "2th_deg", edges, window, convention="gi_chi", gi=gi))
    assert got.available and got.convention == "gi_chi" and got.window_deg == 360.
    expected = _gi_expected(edges, "2th_deg", ai, mask, incident, tilt, orientation)
    assert np.all(np.abs(got.supported_deg - expected) <= TOL * 360)
    detector_chi = angular_support(_request(ai, mask, "2th_deg", edges, admitted_window(None, reducer="int_standard")))
    assert np.max(np.abs(detector_chi.supported_deg - got.supported_deg)) > TOL * 360, "GI χ is not detector χ"

    # the slow path's χ_GI window, and refusal outside its ±180° domain
    slow = admitted_window((-60., 60.), reducer="int_gi_total_q")
    assert slow.denominator_deg == 120.
    windowed = angular_support(_request(ai, mask, "2th_deg", edges, slow, convention="gi_chi", gi=gi))
    assert np.all(np.abs(windowed.supported_deg - _gi_expected(edges, "2th_deg", ai, mask, incident, tilt,
                                                               orientation, (math.radians(-60), math.radians(60))))
                  <= TOL * 120)
    for outside in ((-200., 20.), (100., 190.)):
        assert admitted_window(outside, reducer="int_gi_total_q").reason is SupportReason.WINDOW_OUTSIDE_DOMAIN

    # energies: equal at the same physical 2θ, different at the same q
    other = _planar(shape=(300, 400), poni=(0.005013, 0.020007), wavelength=0.7e-10)
    same_tth = angular_support(_request(other, mask, "2th_deg", edges, window, convention="gi_chi", gi=gi))
    np.testing.assert_allclose(same_tth.supported_deg, got.supported_deg, atol=1e-9)
    q_edges = np.linspace(*(4 * np.pi * np.sin(np.radians(edges[[0, -1]]) / 2) / (ai.wavelength * 1e10)), edges.size)
    at_q = angular_support(_request(ai, mask, "q_A^-1", q_edges, window, convention="gi_chi", gi=gi))
    other_q = angular_support(_request(other, mask, "q_A^-1", q_edges, window, convention="gi_chi", gi=gi))
    assert np.max(np.abs(at_q.supported_deg - other_q.supported_deg)) > TOL * 360


def test_k4b_gi_tilt_moves_the_cut():
    from xrd_tools.analysis.angular_support import admitted_window, angular_support
    ai = _planar(shape=(300, 400), poni=(0.005013, 0.020007))
    edges = np.linspace(6.0, 14.0, 9)
    for tilt_deg in (0.0, 4.0):
        got = angular_support(_request(ai, None, "2th_deg", edges, admitted_window(None, reducer="int_gi_total_q"),
                                       convention="gi_chi", gi=(1.0, tilt_deg, 1)))
        expected = _gi_expected(edges, "2th_deg", ai, None, math.radians(1.0), math.radians(tilt_deg), 1)
        assert np.all(np.abs(got.supported_deg - expected) <= TOL * 360), tilt_deg


# --------------------------------------------------------------------------------------------------------- K5

def test_k5_capture_and_availability(tmp_path):
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, admitted_window, angular_support
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.reduction import GIMode, Integration1DPlan, ReductionPlan
    from xrd_tools.core.containers import PONI
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    band = np.zeros((120, 160), bool)
    band[:, 100:110] = True
    images = [np.full((120, 160), 10.0 + index, np.float32) for index in range(3)]
    rect = _rect(ai, 0, 120, 0, 160)
    for unit in ("q_A^-1", "q_nm^-1", "2th_deg", "2th_rad"):
        plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit=unit, method="csr",
                                                              azimuth_range=(-150., 120.)), mask=band)
        result = _run(plan, _int_scan(ai, images))
        supports = [result.frames[index].angular_support_1d for index in range(3)]
        for reduction, support in zip(result.frames.values(), supports):
            assert support.origin is SupportOrigin.RUN and support.available and support.radial_unit == unit
            assert support.requested_window_deg == (-150., 120.) and support.window_deg == 270.
            assert support.supported_deg.shape == reduction.result_1d.radial.shape
            assert not support.supported_deg.flags.writeable
            assert support.provenance["pyfai_version"]
        edges = supports[0].edges
        np.testing.assert_allclose(.5 * (edges[1:] + edges[:-1]), result.frames[0].result_1d.radial, rtol=1e-12)
        expected = _expected(edges, unit, ai, rect, [_rect(ai, 0, 120, 100, 110)], windows=_deg((-150, 120)))
        assert np.all(np.abs(supports[0].supported_deg - expected) <= TOL * 270), unit
        before = supports[0].supported_deg.copy()
        band[:5] = ~band[:5]
        np.testing.assert_array_equal(supports[0].supported_deg, before)
        band[:5] = ~band[:5]

    # ineligible profiles carry their own reasons
    chi_profile = ReductionPlan(integration_1d=Integration1DPlan(npt=36, unit="chi_deg", radial_range=(1., 3.)))
    assert _run(chi_profile, _int_scan(ai, images[:1])).frames[0].angular_support_1d.reason is \
        SupportReason.NOT_RADIAL_PROFILE
    poni = PONI(dist=ai.dist, poni1=ai.poni1, poni2=ai.poni2, wavelength=ai.wavelength, detector="Detector")
    for mode, reason in (("q_total", None), ("q_ip", SupportReason.NOT_RADIAL_PROFILE)):
        gi_plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="q_A^-1"),
                                gi=GIMode(incident_angle=1.0, mode_1d=mode, sample_orientation=1))
        from xrd_tools.reduction import Frame, Scan
        gi_scan = Scan("gi", [Frame(0, image=images[0])], integrator=ai, poni=poni)
        support = _run(gi_plan, gi_scan).frames[0].angular_support_1d
        if reason is not None:
            assert support.reason is reason
        else:
            assert support.available and support.convention == "gi_chi" and support.window_deg == 360.
            direct = angular_support(_request(ai, None, "q_A^-1", support.edges,
                                              admitted_window(None, reducer="int_gi_total_q"),
                                              convention="gi_chi", gi=(1.0, 0.0, 1)))
            np.testing.assert_allclose(support.supported_deg, direct.supported_deg, atol=1e-9)

    # unqualified geometry refuses with a reason, before any detector-sized work (parallax; a curved detector)
    from pyFAI.detectors import detector_factory
    full_turn = admitted_window(None, reducer="int_standard")
    sensor = detector_factory("Pilatus100k", {"orientation": 3, "sensor": {"material": "Si", "thickness": 0.00045}})
    parallax = _planar(dist=0.1, poni=(0.010013, 0.040007), detector=sensor)
    parallax.enable_parallax(True)
    assert angular_support(_request(parallax, None, "2th_deg", np.linspace(2., 12., 11), full_turn)).reason is \
        SupportReason.UNQUALIFIED_GEOMETRY
    curved = _planar(detector=detector_factory("Aarhus"))
    assert angular_support(_request(curved, None, "2th_deg", np.linspace(2., 12., 11), full_turn)).reason is \
        SupportReason.UNQUALIFIED_GEOMETRY

    # the singular direction: bins starting at 2θ = 0 with the beam on an unmasked pixel follow their integral
    edges = np.linspace(0.0, 2.0, 11)
    singular = angular_support(_request(ai, None, "2th_deg", edges, admitted_window(None, reducer="int_standard")))
    assert np.all(np.isfinite(singular.supported_deg)) and abs(singular.supported_deg[0] - 360.) <= TOL * 360

    # D3: an over-budget Run keeps its intensity and names its reason
    over = _run(ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="2th_deg")), _int_scan(ai, images),
                AngularSupportPolicy(enabled=True, max_geometry_keys=0))
    plain_run = _run(ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit="2th_deg")), _int_scan(ai, images),
                     AngularSupportPolicy(enabled=False))
    for index in range(3):
        assert over.frames[index].angular_support_1d.reason is SupportReason.OVER_BUDGET
        assert over.frames[index].result_1d.intensity.tobytes() == plain_run.frames[index].result_1d.intensity.tobytes()


def _energy_scan_views(energies, unit):
    """BRR's bound_session (test_predicted_ring_overlays.py), with the plan carrying the scan's own energy selection
    as the production plan does (readiness.native_int_reduction_plan): only then does Int reduce frame by frame.
    Test-only oracle repair: bound_session's plan omits it, so its reduction ran at one wavelength."""
    import xrd_tools.session.readiness as readiness
    from tests.core.test_predicted_ring_overlays import calibration
    from xrd_tools.reduction import Frame, Integration1DPlan, Integration2DPlan, MemorySink, ReductionPlan, Scan
    from xrd_tools.session import FrameRecordStore
    from xrd_tools.session.headless_scan import open_headless_scan_session
    from xrd_tools.session.readiness import bind_native_int_source_energy
    frames = [Frame(label, image=np.full((195, 487), 100., dtype=np.float32), metadata={"energy": energy})
              for label, energy in zip((7, 11, 15, 19), energies)]
    scan = Scan("rings", frames)
    accepted = bind_native_int_source_energy(scan, calibration(wavelength=12.398419843320026 / 15 * 1e-10))
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=64, unit=unit),
                         integration_2d=Integration2DPlan(npt_rad=64, npt_azim=36, unit=unit),
                         extra={"energy_selection": scan.extra["energy_selection"]})
    store = FrameRecordStore()
    # coverage science asks for recording explicitly (Off by default since INT-PERF-REGRESSION-20261014), through an
    # exact validated session policy, as R3 does
    from dataclasses import replace
    from types import SimpleNamespace
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.session.policy import requirements_from, resolve_session_policy
    image = frames[0].image
    policy = resolve_session_policy(
        requirements_from(SimpleNamespace(frame_shape=image.shape, dtype=image.dtype), plan),
        envelope_bytes=8 * 1024 ** 3, env={})
    session = open_headless_scan_session(
        scan, plan, sink=MemorySink(), record_store=store, executor=1,
        ring_geometry=readiness.effective_ring_geometry(accepted, integrator=scan.integrator),
        policy=replace(policy, angular_support=AngularSupportPolicy(enabled=True)))
    for frame in frames:
        session.submit(frame)
    session.finish()
    return [next(iter(store.get(frame.index).results_1d.values())) for frame in frames]


def test_k5b_publication_carries_the_runs_support():
    """The published FrameView carries the Run's own immutable support object; mutation of inputs cannot reach it."""
    from xrd_tools.core.angular_support import SupportOrigin
    views = _energy_scan_views((16., 17.), "q_A^-1")
    published = [view.angular_support for view in views]
    assert all(item is not None and item.origin is SupportOrigin.RUN and item.available for item in published)
    assert published[0].provenance["wavelength_m"] != published[1].provenance["wavelength_m"]
    assert np.max(np.abs(published[0].supported_deg - published[1].supported_deg)) > 0, "energy scan: per-frame"


# --------------------------------------------------------------------------------------------------------- K6

def _saved_int(tmp_path, *, frames=3, enabled=True, unit="q_A^-1"):
    """A real headless Run written through NexusSink; returns the output and each frame's captured support."""
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.reduction import CompositeSink, Frame, Integration1DPlan, MemorySink, NexusSink, ReductionPlan, \
        Scan, run_reduction
    tmp_path.mkdir(parents=True, exist_ok=True)
    ai = _planar(shape=(120, 160), poni=(0.006013, 0.008007))
    band = np.zeros((120, 160), bool)
    band[:, 100:110] = True
    labels = list(range(1, frames + 1))
    scan = Scan("k6", [Frame(label, image=np.full((120, 160), 10.0 + label, np.float32)) for label in labels],
                integrator=ai)
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, unit=unit, method="csr"), mask=band)
    target = tmp_path / "k6.nexus"
    memory = MemorySink()
    run_reduction(plan, scan, CompositeSink((memory, NexusSink(target, overwrite=True))), execution="chunked",
                  executor=False, angular_support=AngularSupportPolicy(enabled=enabled))
    return target, {label: memory.frames[label].angular_support_1d for label in labels}


def test_k6_saved_support_contract(tmp_path):
    import h5py
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    from xrd_tools.io.angular_support import AngularSupportFormatError, read_angular_support
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS, ANGULAR_SUPPORT_VERSION_ATTR, INTEGRATED_ROW_ALIGNED, SCHEMA
    assert ANGULAR_SUPPORT_DATASETS == ("angular_support_deg", "angular_support_meta", "angular_support_rows",
                                        "angular_support_row_digest")
    assert ANGULAR_SUPPORT_VERSION_ATTR == "ssrl_angular_support_version"
    assert INTEGRATED_ROW_ALIGNED == {"frame_index", "intensity", "sigma"}
    assert set(SCHEMA.groups["integrated_1d"].datasets) == {"intensity", "axis_1", "frame_index", "sigma"}
    target, captured = _saved_int(tmp_path)
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert all(name in group for name in ANGULAR_SUPPORT_DATASETS)
        assert "coverage" not in group and not any(isinstance(group[name], h5py.Group) for name in group
                                                    if name.startswith("angular_support"))
        table = group["angular_support_deg"]
        assert table.dtype == np.float32 and table.shape[1] == group["axis_1"].shape[0]
        assert int(table.attrs[ANGULAR_SUPPORT_VERSION_ATTR]) == 1 and table.attrs["units"] == "deg"
        assert group["angular_support_rows"].dtype == np.int64 and group["angular_support_rows"].shape[1] == 2
        assert group["angular_support_row_digest"].dtype == np.uint8
        assert group["angular_support_row_digest"].shape[1] == 16
        assert all(len(item) <= 4096 for item in group["angular_support_meta"].asstr()[()])
    with FrameViewReader(target) as reader:
        for label, run in captured.items():
            saved = reader.read(label).angular_support
            assert saved.origin is SupportOrigin.SAVED and saved.available
            np.testing.assert_allclose(saved.supported_deg, run.supported_deg.astype(np.float32),
                                       rtol=0, atol=0)
            assert saved.window_deg == run.window_deg and saved.radial_unit == run.radial_unit
            assert saved.bin_edges == run.bin_edges and saved.convention == run.convention
    # a row rewritten by a path that does not maintain support is stale, never another row's support
    stale = tmp_path / "stale.nexus"
    stale.write_bytes(target.read_bytes())
    with h5py.File(stale, "r+") as handle:
        handle["entry/integrated_1d/intensity"][1] = handle["entry/integrated_1d/intensity"][1] * 2
    with FrameViewReader(stale) as reader:
        assert reader.read(2).angular_support.reason is SupportReason.STALE_SAVED_SUPPORT
        assert reader.read(1).angular_support.available
    # row surgery moves rows with their bytes: survivors stay bound, a removed label's entry is ignored
    from xrd_tools.io.nexus_record import drop_integrated_rows
    dropped = tmp_path / "dropped.nexus"
    dropped.write_bytes(target.read_bytes())
    with h5py.File(dropped, "r+") as handle:
        drop_integrated_rows(handle, "entry/integrated_1d", [2])
    with FrameViewReader(dropped) as reader:
        assert reader.read(3).angular_support.available
    # present but malformed is an error, not legacy absence
    for name, edit in (
        ("link", lambda g: (g.__delitem__("angular_support_deg"),
                            g.__setitem__("angular_support_deg", h5py.SoftLink("/entry/integrated_1d/intensity")))),
        ("dtype", lambda g: (g.__delitem__("angular_support_rows"),
                             g.create_dataset("angular_support_rows", data=np.zeros((3, 2), np.float64)))),
        ("version", lambda g: g["angular_support_deg"].attrs.__setitem__(ANGULAR_SUPPORT_VERSION_ATTR, 99)),
        ("shape", lambda g: (g.__delitem__("angular_support_row_digest"),
                             g.create_dataset("angular_support_row_digest", data=np.zeros((3, 7), np.uint8)))),
        ("meta", lambda g: (g.__delitem__("angular_support_meta"),
                            g.create_dataset("angular_support_meta", data=["x" * 5000] * 3,
                                             dtype=h5py.string_dtype()))),
    ):
        broken = tmp_path / f"broken-{name}.nexus"
        broken.write_bytes(target.read_bytes())
        with h5py.File(broken, "r+") as handle:
            edit(handle["entry/integrated_1d"])
        with h5py.File(broken, "r") as handle:
            group = handle["entry/integrated_1d"]
            with pytest.raises(AngularSupportFormatError):
                read_angular_support(group, labels=(1, 2, 3), intensity_rows=group["intensity"][()])
    # absence is a fact for the strict reader: no saved support (Browse alone may reconstruct a sufficient Run).
    # Decision 6 (October 13): a Run with recording off now records that it was off, so the older-output form is a
    # written output with its support removed (D5's form).
    legacy = tmp_path / "legacy.nexus"
    legacy.write_bytes(target.read_bytes())
    with h5py.File(legacy, "r+") as handle:
        for name in ANGULAR_SUPPORT_DATASETS:
            del handle["entry/integrated_1d"][name]
    with h5py.File(legacy, "r") as handle:
        assert not any(name in handle["entry/integrated_1d"] for name in ANGULAR_SUPPORT_DATASETS)
    with FrameViewReader(legacy) as reader:
        assert reader.read(1).angular_support.reason is SupportReason.NO_SAVED_SUPPORT


def test_k6b_current_admission_reads_the_extension_unchanged(tmp_path):
    """Strict admission and every existing reader keep their meaning; the extension is invisible to them."""
    import h5py
    from xrd_tools.io.read import ProcessedScan
    from xrd_tools.io.schema import read_current_mode_layout
    with_support, _ = _saved_int(tmp_path / "a")
    without, _ = _saved_int(tmp_path / "b", enabled=False)
    for path in (with_support, without):
        with h5py.File(path, "r") as handle:
            read_current_mode_layout(handle["entry/integrated_1d"], "1d")
    a, b = ProcessedScan(with_support), ProcessedScan(without)
    try:
        np.testing.assert_array_equal(a.get_1d(1)[1], b.get_1d(1)[1])
    finally:
        for scan in (a, b):
            close = getattr(scan, "close", None)
            if close is not None:
                close()


def test_k6c_average_outputs_carry_no_support(tmp_path):
    import h5py
    from tests.core.test_p36_average import _run_average_scan, _series
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues
    source = _series(tmp_path, [np.full((8, 8), value, np.uint16) for value in (5, 6, 7)])
    calibration = CalibrationState(PoniValues(0.05, 0.0004, 0.0004, 0.0, 0.0, 0.0, 1e-10), "Detector",
                                   {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [8, 8], "orientation": 3},
                                   status=FactStatus.PRESENT)
    result = _run_average_scan(AverageScanRecipe(source, tmp_path / "avg.nexus", ReductionPlan(
        integration_1d=Integration1DPlan(npt=4, method="numpy", unit="2th_deg")), calibration=calibration))
    assert result.disposition == "COMMITTED", result
    with h5py.File(result.target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert not any(name in group for name in ANGULAR_SUPPORT_DATASETS)
        label = int(group["frame_index"][0])
    with FrameViewReader(result.target) as reader:
        assert reader.read(label).angular_support.reason is SupportReason.NO_SAVED_SUPPORT


# --------------------------------------------------------------------------------------------------------- K7

def test_k7_legacy_record_sufficiency(tmp_path):
    """BRR's saved basis, extended: an older Int Run's own record reconstructs support, or names what is missing."""
    from tests.core.test_browse_reference_rings import _saved_configuration, _selection
    from xrd_tools.analysis.angular_support import (SavedSupportMask, admitted_window, angular_support,
                                                    support_basis_from_saved)
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason
    frozen, saved = _saved_configuration(tmp_path, mode="Int 1D", one={"npt": 40, "method": "csr",
                                                                       "unit": "2th_deg",
                                                                       "radial_range": [2.0, 18.0]})
    labels = (1, 2, 3)
    selection = _selection(labels, [16., 16., 16.])
    signed_none = SavedSupportMask(signed=True, sha256=None, indices=None)
    basis = support_basis_from_saved(saved, selection, labels=labels, epochs=(labels,), mask=signed_none,
                                     shape=(195, 487))
    assert hasattr(basis, "profile"), basis
    integrator = basis.integrator_for(1)
    axis = integrator.integrate1d(np.ones((195, 487), np.float32), 40, unit="2th_deg", radial_range=(2.0, 18.0),
                                  method="csr").radial.astype(np.float32)
    profile = basis.profile(1, axis_values=axis, axis_unit="2th_deg")
    assert profile.origin is SupportOrigin.RECONSTRUCTED and profile.available
    assert profile.provenance["pyfai_version"]
    detector_mask = integrator.detector.mask
    direct = angular_support(_request(integrator, None if detector_mask is None else detector_mask.astype(bool),
                                      "2th_deg", np.linspace(2.0, 18.0, 41),
                                      admitted_window(None, reducer="int_standard")))
    np.testing.assert_allclose(profile.supported_deg, direct.supported_deg, atol=1e-9)

    # each missing or contradicting fact names itself; nothing is guessed
    def reason(**changes):
        kwargs = dict(labels=labels, epochs=(labels,), mask=signed_none, shape=(195, 487))
        kwargs.update(changes)
        result = support_basis_from_saved(saved, kwargs.pop("selection", selection), **kwargs)
        if hasattr(result, "profile"):
            result = result.profile(1, axis_values=axis, axis_unit="2th_deg")
        return result.reason
    assert reason(mask=SavedSupportMask(signed=False, sha256=None, indices=None)) is \
        SupportReason.LEGACY_INSUFFICIENT, "an unsigned record cannot say whether a mask applied"
    assert reason(mask=SavedSupportMask(signed=True, sha256="0" * 64, indices=None)) is \
        SupportReason.LEGACY_INSUFFICIENT, "a digest without its pixels"
    assert reason(mask=SavedSupportMask(signed=True, sha256="0" * 64,
                                        indices=np.array([195 * 487 + 3], np.int64))) is \
        SupportReason.LEGACY_INSUFFICIENT, "indices beyond the detector: the builder could have ignored it"
    assert reason(shape=(196, 487)) is SupportReason.LEGACY_INSUFFICIENT, "shape contradicts the detector"
    assert reason(selection={**selection, "wavelength_m": None}) is SupportReason.LEGACY_INSUFFICIENT
    assert reason(kind="average") is SupportReason.AVERAGE_RESULT
    assert reason(kind="reintegrate") is SupportReason.LEGACY_REINTEGRATE
    auto_saved = json.loads(json.dumps(saved))
    auto_saved["bai_1d_args"].pop("radial_range", None)
    auto = support_basis_from_saved(auto_saved, selection, labels=labels, epochs=(labels,), mask=signed_none,
                                    shape=(195, 487))
    mismatched = auto.profile(1, axis_values=np.float32(np.linspace(1.0, 9.0, 40)), axis_unit="2th_deg")
    assert mismatched.reason is SupportReason.LEGACY_INSUFFICIENT, "Auto bins must reproduce the saved axis bytes"
    # a digest with its embedded pixels is applied under the reducer's union with the detector mask
    indices = np.flatnonzero(np.pad(np.ones((195, 10), bool), ((0, 0), (200, 277))))
    masked = support_basis_from_saved(saved, selection, labels=labels, epochs=(labels,), shape=(195, 487),
                                      mask=SavedSupportMask(signed=True, sha256="a" * 64, indices=indices))
    with_band = masked.profile(1, axis_values=axis, axis_unit="2th_deg")
    assert with_band.available and np.max(profile.supported_deg - with_band.supported_deg) > TOL * 360


# --------------------------------------------------------------------------------------------------------- K8

def _stitch_source(tmp_path, poses, images):
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleKind, ModuleSourceReceipt
    from xrd_tools.analysis.scan_operations import AnalysisDisposition, MetadataTablePlan, run_metadata_table
    from xrd_tools.sources.selection import image_series_spec
    import tifffile
    root = tmp_path / "source"
    root.mkdir(parents=True)
    for index, ((delta, nu), image) in enumerate(zip(poses, images)):
        path = root / f"scan_{index + 1:04d}.tif"
        tifffile.imwrite(path, image)
        path.with_suffix(".txt").write_text(f"# Counters\nI0 = 10\n# Motors\ndel = {delta}, nu = {nu}\n"
                                            "User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n",
                                            encoding="utf-8")
    table = run_metadata_table(MetadataTablePlan(image_series_spec(root / "scan_0001.tif", metadata_format="txt")))
    assert table.disposition is AnalysisDisposition.COMPLETED
    return ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=tuple(range(1, len(poses) + 1)),
        resolved_selectors=(MetadataColumnSelector("I0"), MetadataColumnSelector("del"),
                            MetadataColumnSelector("nu")))


def _stitch(tmp_path, poses, images, *, azimuth=None, unit="q_A^-1", policy=None, conditioning=None):
    import hashlib
    from tests.core.test_stitch_operation import _goniometer_record
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleOutputRequest
    from xrd_tools.analysis.stitch_operation import (StitchGeometryInput, StitchGeometryKind, StitchOperationPlan,
                                                     capture_stitch_geometry, prepare_stitch_operation,
                                                     run_stitch_operation)
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    source = _stitch_source(tmp_path, poses, images)
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(_goniometer_record()), encoding="utf-8")
    geometry = capture_stitch_geometry(StitchGeometryInput(
        geometry_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        expected_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        source_motors=(("del_angle", "del"), ("nu_angle", "nu"))))
    kwargs = dict(mode="1d", npt_1d=96, npt_rad_2d=48, npt_azim_2d=32, unit=unit, native_records=True,
                  radial_range=(0.5, 5.0) if unit == "q_A^-1" else (5.0, 50.0), azimuth_range_1d=azimuth,
                  monitor_selector=MetadataColumnSelector("I0"), max_frame_bytes=64 * 1024 * 1024)
    if conditioning is not None:
        kwargs["conditioning"] = conditioning
    plan = StitchOperationPlan(geometry, **kwargs)
    from xrd_tools.io.output_path import resolve_finite_output_target
    target = resolve_finite_output_target(tmp_path, "native", operation_token="stitch-1d")
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, artifact_family="native")
    request = prepare_stitch_operation(source, output, plan, project_root=tmp_path)
    return request, run_stitch_operation(request, **_explicit(policy))


def _pose_ai(delta, nu):
    from tests.core.test_stitch_operation import _goniometer_record
    from pyFAI.detectors import detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    record = _goniometer_record()
    dist, poni1, poni2 = record["param"][:3]
    return AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2, rot1=math.radians(nu), rot2=math.radians(delta),
                               rot3=0.0, detector=detector_factory("Pilatus100k"), wavelength=1e-10)


def test_k8_stitch_contributor_union(tmp_path):
    from xrd_tools.core.angular_support import SupportOrigin
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    shape = (195, 487)
    ones = np.full(shape, 200, np.uint16)
    poses = ((0.0, 0.0), (0.0, 0.0), (8.0, 0.0), (8.0, 0.0), (24.0, 0.0))
    images = [ones, ones, ones, ones, np.zeros(shape, np.uint16)]
    request, result = _stitch(tmp_path / "union", poses, images)
    support = result.angular_support
    assert support is not None and support.available and support.origin is SupportOrigin.RUN
    assert support.convention == "detector_chi" and support.window_deg == 360.0
    assert support.provenance["backend"] == "multigeometry"
    assert support.provenance["included_contributors"] == 4, "the screened-out empty frame is not geometry"
    assert support.provenance["unique_poses"] == 2, "repeated poses are computed once"
    edges = support.edges
    compared = list(range(2, 40, 2))
    expected = _ring_union([(_pose_ai(0.0, 0.0), None), (_pose_ai(8.0, 0.0), None)], edges, "q_A^-1", 1e-10,
                           compared)
    assert np.all(np.abs(support.supported_deg[compared] - expected) <= TOL * 360)
    with_excluded = _ring_union([(_pose_ai(0.0, 0.0), None), (_pose_ai(8.0, 0.0), None),
                                 (_pose_ai(24.0, 0.0), None)], edges, "q_A^-1", 1e-10, compared)
    assert np.max(with_excluded - expected) > TOL * 360, "fixture: the excluded pose would have added support"
    # value-invalid pixels inside an included frame stay geometrically covered; intensity is unchanged
    hot = ones.copy()
    hot[40:120, 100:300] = 65000
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    thresholded, thresholded_result = _stitch(tmp_path / "hot", poses[:4], [hot, hot, ones, ones],
                                              conditioning=StitchConditioning("manual", threshold_max=1000.0))
    np.testing.assert_allclose(thresholded_result.angular_support.supported_deg, support.supported_deg, atol=1e-9)
    _off_request, off = _stitch(tmp_path / "off", poses[:4], [hot, hot, ones, ones],
                                conditioning=StitchConditioning("manual", threshold_max=1000.0),
                                policy=AngularSupportPolicy(enabled=False))
    from xrd_tools.core.angular_support import SupportReason
    assert off.angular_support.reason is SupportReason.RECORDING_DISABLED, "decision 6: off records why"
    np.testing.assert_array_equal(thresholded_result.payload.intensity, off.payload.intensity)
    # the profile's own window: explicit (normalised by pyFAI) versus scouted (denominator stays 360°)
    _w_request, windowed = _stitch(tmp_path / "window", poses[:4], [ones] * 4, azimuth=(-30.0, 30.0))
    assert windowed.angular_support.window_deg == 60.0
    assert np.all(windowed.angular_support.supported_deg <= 60.0 + 1e-9)


def test_k8b_xu_corners_and_support(tmp_path):
    from tests.core.test_stitch_operation import _xu_prepared
    from xrd_tools.analysis.angular_support import xu_corner_grid
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import XuPowderQProvider, resolve_xu_stitch_effective_geometry
    request = _xu_prepared(tmp_path)
    asset = json.loads(canonical_surface_resource_bytes())
    calibration = request.plan.calibration
    with xu_runtime_session() as runtime:
        geometry = resolve_xu_stitch_effective_geometry(calibration, runtime)
        provider = XuPowderQProvider(geometry, asset["acquisition"])
        with provider.frame(nu=-10.0, del_=14.0) as lease:
            q_centre = np.array(lease.q_magnitude())
            chi_centre = np.radians(lease.chi())
        tth_corner, chi_corner = xu_corner_grid(geometry, asset["acquisition"], nu=-10.0, del_=14.0)
    wavelength = geometry.projection.xu_mapping_wavelength_A
    tth_centre = 2 * np.arcsin(q_centre * wavelength / (4 * np.pi))
    assert tth_corner.shape == (tth_centre.shape[0] + 1, tth_centre.shape[1] + 1)

    def vectors(tth, chi):
        return np.stack((np.cos(tth), np.sin(tth) * np.cos(chi), np.sin(tth) * np.sin(chi)), axis=-1)

    corners = vectors(tth_corner, chi_corner)
    mean = corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:]
    mean /= np.linalg.norm(mean, axis=-1, keepdims=True)
    angle = np.arccos(np.clip(np.sum(mean * vectors(tth_centre, chi_centre), axis=-1), -1, 1))
    assert float(np.max(angle)) <= 2e-6, "XU corners are half-pixel exact about XU's own centres"
    result = run_stitch_operation(request, **_explicit(None))  # coverage science asks for recording explicitly
    support = result.angular_support
    assert support.available and support.provenance["backend"] == "xu_hist"
    assert np.all((support.supported_deg >= 0) & (support.supported_deg <= support.window_deg + 1e-9))


# --------------------------------------------------------------------------------------------------------- K9

def test_k9_stitch_saved_support_and_legacy(tmp_path):
    import h5py
    from xrd_tools.core.angular_support import SupportOrigin, SupportReason, REASON_TEXT
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    from xrd_tools.io.stitch_record import read_stitch_record
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    shape = (195, 487)
    ones = np.full(shape, 200, np.uint16)
    poses = ((0.0, 0.0), (8.0, 0.0), (16.0, 0.0))
    request, result = _stitch(tmp_path / "saved", poses, [ones] * 3)
    target = request.module.output.target
    payload = read_analysis_artifact(target)
    saved = payload.angular_support
    assert saved.origin is SupportOrigin.SAVED and saved.available
    np.testing.assert_allclose(saved.supported_deg, result.angular_support.supported_deg.astype(np.float32),
                               rtol=0, atol=0)
    record = read_stitch_record(target)
    assert record.angular_support.origin is SupportOrigin.SAVED
    np.testing.assert_array_equal(record.angular_support.supported_deg, saved.supported_deg)
    with h5py.File(target, "r") as handle:
        group = handle["entry/integrated_1d"]
        assert all(name in group for name in ANGULAR_SUPPORT_DATASETS)
        assert int(group["angular_support_rows"][0, 0]) == 1
        assert not any(name in handle["entry/stitch"] for name in ANGULAR_SUPPORT_DATASETS)
    _off_request, off = _stitch(tmp_path / "plain", poses, [ones] * 3, policy=AngularSupportPolicy(enabled=False))
    plain = read_analysis_artifact(_off_request.module.output.target)
    np.testing.assert_array_equal(payload.intensity, plain.intensity)
    np.testing.assert_array_equal(payload.coverage, plain.coverage)
    # an older Stitch result (no support datasets) is refused, never reconstructed (D6). Decision 6: recording off
    # now records that it was off, so the older form is a written result with its support removed — at its own path,
    # since a stitched record reads only where it was written (its file_name; test-only oracle repair).
    with h5py.File(target, "r+") as handle:
        for name in ANGULAR_SUPPORT_DATASETS:
            del handle["entry/integrated_1d"][name]
    older_payload = read_analysis_artifact(target)
    assert older_payload.angular_support.reason is SupportReason.LEGACY_STITCH
    # the same result with and without its support: only support differs, so the fingerprint must not (two Stitches
    # in different directories always differ by their own provenance — test-only oracle repair)
    assert older_payload.result_fingerprint == payload.result_fingerprint, "support stays outside the fingerprint"
    assert REASON_TEXT[SupportReason.LEGACY_STITCH] == \
        "Older Stitch result: reprocess it to record angular coverage."
    # membership replacement: a new run's provenance names its own membership
    _second, again = _stitch(tmp_path / "fewer", poses[:2], [ones] * 2)
    assert again.angular_support.provenance["included_contributors"] == 2
