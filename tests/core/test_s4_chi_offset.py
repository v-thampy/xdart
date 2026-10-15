"""S-4 — chi_offset flows to Integration1DPlan.azimuth_offset (mirror the 2D),
NOT a shift of the input azimuth range.

GI-1 "auto == explicit -180..180" held in GI mode but FAILED in STANDARD mode
because chi_offset (default 90 deg) shifted the 1D INPUT range and left the 1D
chi OUTPUT axis in the raw pyFAI frame, 90 deg out of frame with the 2D cake chi
(which re-adds the offset to its output).  These drive the real plan builder.
"""

import numpy as np
import pytest

from xrd_tools.session.readiness import (
    build_native_int_reduction_plan_from_args as build,
)


def test_standard_chi_offset_becomes_1d_azimuth_offset_range_unshifted():
    a1 = {"unit": "chi_deg", "chi_offset": 90.0,
          "azimuth_range": (-180.0, 180.0), "npt": 180}
    plan = build(a1, {"unit": "q_A^-1"}, gi_enabled=False)
    p1 = plan.integration_1d
    assert p1.azimuth_offset == 90.0             # carried onto the plan...
    assert p1.azimuth_range == (-180.0, 180.0)   # ...NOT popped-and-shifted


def test_standard_1d_and_2d_offsets_agree():
    plan = build({"unit": "chi_deg", "chi_offset": 90.0},
                 {"unit": "q_A^-1", "azimuth_offset": 90.0}, gi_enabled=False)
    assert (plan.integration_1d.azimuth_offset
            == plan.integration_2d.azimuth_offset == 90.0)


def test_zero_chi_offset_is_a_noop():
    plan = build({"unit": "chi_deg", "azimuth_range": (-180.0, 180.0)},
                 {"unit": "q_A^-1"}, gi_enabled=False)
    assert plan.integration_1d.azimuth_offset == 0.0
    assert plan.integration_1d.azimuth_range == (-180.0, 180.0)


# ----------------------------------------------------------- CHI-WINDOW-SEAM-TRUNCATION (W rows)
# PRE-RELEASE-CORRECTNESS-20261015, frozen on ae6f0453 before any production edit. An explicit
# panel window (lo, hi) under offset o selects the periodic preimage of its directions; a full
# turn selects every valid pixel once. Every row drives the real calibration, the native plan
# builder, run_reduction and pyFAI on a test-owned 200x200 detector (beam inside). References are
# native results of the same reducer: Auto, an exact one-degree permutation of the raw full turn,
# or one call on a second test-owned integrator whose chi discontinuity is at zero. No test-side
# integration and no interior split; test_diag_* rows only record behavior.

SHAPE = (200, 200)
Q_BAND = (0.1, 0.6)
HOT = 1.0e7
OUTS = ("q", "chi", "cake")
SEAMS = {"lower": ((-170.0, -110.0), 30.0), "upper": ((110.0, 170.0), -30.0)}
DIAGNOSTICS = {"w5": ((-180.0, 180.0), 30.5, 71), "wide": ((0.0, 400.0), 0.0, 40),
               "equal": ((0.0, 0.0), 0.0, 40)}
SEAM_PONI = ('poni_version: 2\nDetector: Detector\n'
             'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [200, 200]}\n'
             'Distance: 0.1\nPoni1: 0.010013\nPoni2: 0.010007\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\n'
             'Wavelength: 1.0e-10\n')


@pytest.fixture(scope="module")
def seam_poni(tmp_path_factory):
    path = tmp_path_factory.mktemp("seam") / "seam.poni"
    path.write_text(SEAM_PONI)
    return path


def _integrator(poni, at_pi=True):
    from xrd_tools.integrate.calibration import (
        detector_calibration_to_integrator, load_detector_calibration)
    ai = detector_calibration_to_integrator(load_detector_calibration(poni))
    if not at_pi:
        ai.setChiDiscAtZero()
    return ai


def _image(poni):
    """A raw chi ramp (discontinuous at the seam) on a ring, legitimate zeros and excluded hot pixels."""
    ai = _integrator(poni)
    chi = np.degrees(ai.center_array(SHAPE, "chi_rad"))
    q = ai.center_array(SHAPE, "q_A^-1")
    image = (200.0 + chi) * (1.0 + 4.0 * np.exp(-((q - 0.35) / 0.04) ** 2))
    image[(chi > 60.0) & (chi < 70.0)] = 0.0
    image[(np.abs(chi) > 172.0) & (np.arange(chi.size).reshape(SHAPE) % 5 == 0)] = HOT
    return image.astype(np.float32)


def _reduce(poni, out, window, offset, *, npt, method="no", at_pi=True, support=False):
    """One real Run of the frame: its q, I(chi) or cake result and its captured 1-D support."""
    from xrd_tools.analysis.angular_support import AngularSupportPolicy
    from xrd_tools.reduction import Frame, MemorySink, Scan, run_reduction
    args = {"method": method, "radial_range": Q_BAND, "polarization_factor": 0.99,
            "monitor": "i0", "chi_offset": offset}
    if window is not None:
        args["azimuth_range"] = window
    if out == "cake":
        args.update(npt_rad=25, npt_azim=npt, unit="q_A^-1", error_model="poisson")
    elif out == "chi":
        args.update(npt=npt, unit="chi_deg", chi_npt_rad=100)
    else:
        args.update(npt=npt, unit="q_A^-1", error_model="poisson")
    one, two = (None, args) if out == "cake" else (args, None)
    plan = build(one, two, integrate_1d=one is not None, integrate_2d=two is not None,
                 threshold_max=HOT / 10, detector_shape=SHAPE)
    scan = Scan("seam", [Frame(1, image=_image(poni), metadata={"i0": 2.0})],
                integrator=_integrator(poni, at_pi))
    sink = MemorySink()
    run_reduction(plan, scan, sink, execution="chunked", executor=False,
                  angular_support=AngularSupportPolicy(enabled=support))
    frame = sink.frames[1]
    return (frame.result_2d if two is not None else frame.result_1d), frame.angular_support_1d


def _npt(out, width):
    return 50 if out == "q" else int(round(width))


def _same(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    assert actual.shape == expected.shape
    gaps = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), gaps)
    np.testing.assert_allclose(actual[~gaps], expected[~gaps], rtol=1e-6, atol=1e-8)


def _check(out, actual, expected, *, axis=None, roll=0):
    """``actual`` is the native ``expected``, chi bins rolled by ``roll``, chi axis relabelled ``axis``."""
    chi = 1 if out == "cake" else 0
    if out == "cake":
        _same(actual.radial, expected.radial)
    observed = actual.azimuthal if out == "cake" else actual.radial
    reference = expected.azimuthal if out == "cake" else expected.radial
    _same(observed, reference if axis is None else axis)
    _same(actual.intensity, np.roll(expected.intensity, roll, axis=chi))
    if expected.sigma is None:
        assert out == "chi" and actual.sigma is None, "I(chi) keeps no sigma; q and cake requested it"
    else:
        _same(actual.sigma, np.roll(expected.sigma, roll, axis=chi))


def _zero_reference(poni, out, window, offset, **kwargs):
    """One native call on a chi-discontinuity-at-zero integrator, the raw window moved into [0, 360)."""
    lo, hi = window[0] - offset, window[1] - offset
    turns = -360.0 * np.floor(lo / 360.0)
    result, support = _reduce(poni, out, (lo + turns, hi + turns), 0.0, at_pi=False, **kwargs)
    return result, support, turns


def _seam_row(poni, out, window, offset, **kwargs):
    actual, _ = _reduce(poni, out, window, offset, **kwargs)
    expected, _, turns = _zero_reference(poni, out, window, offset, **kwargs)
    axis = None if out == "q" else (
        expected.azimuthal if out == "cake" else expected.radial) - turns + offset
    _check(out, actual, expected, axis=axis)


@pytest.mark.parametrize("out", OUTS)
def test_w1_shifted_full_turn_selects_every_direction(seam_poni, out):
    actual, _ = _reduce(seam_poni, out, (-180.0, 180.0), 30.0, npt=_npt(out, 360))
    if out == "q":
        expected, _ = _reduce(seam_poni, out, None, 30.0, npt=50)
        _check(out, actual, expected)
        return
    # One-degree bins: panel bin k is raw bin k - 30 of the raw full turn. The cake's Auto chi
    # extent is pixel-derived, so its native full turn is the explicit raw (-180, 180).
    window = None if out == "chi" else (-180.0, 180.0)
    expected, _ = _reduce(seam_poni, out, window, 0.0, npt=360)
    _check(out, actual, expected, roll=30)


@pytest.mark.parametrize("offset", (0.0, -30.0), ids=("offset0", "offset-30"))
def test_w2_q_full_turn_is_offset_invariant(seam_poni, offset):
    actual, _ = _reduce(seam_poni, "q", (-180.0, 180.0), offset, npt=50)
    expected, _ = _reduce(seam_poni, "q", None, offset, npt=50)
    _check("q", actual, expected)


@pytest.mark.parametrize("out", OUTS)
@pytest.mark.parametrize("seam", SEAMS)
def test_w3_seam_crossing_panel_windows_select_both_sides(seam_poni, seam, out):
    _seam_row(seam_poni, out, *SEAMS[seam], npt=_npt(out, 60))


@pytest.mark.parametrize("out", OUTS)
def test_w4_non_crossing_window_is_preserved(seam_poni, out):
    actual, _ = _reduce(seam_poni, out, (20.0, 80.0), 30.0, npt=_npt(out, 60))
    expected, _ = _reduce(seam_poni, out, (-10.0, 50.0), 0.0, npt=_npt(out, 60))
    axis = None if out == "q" else (
        expected.azimuthal if out == "cake" else expected.radial) + 30.0
    _check(out, actual, expected, axis=axis)


def test_w5_non_aligned_offset_q_full_turn(seam_poni):
    actual, _ = _reduce(seam_poni, "q", (-180.0, 180.0), 30.5, npt=50)
    expected, _ = _reduce(seam_poni, "q", None, 30.5, npt=50)
    _check("q", actual, expected)


@pytest.mark.parametrize("method", ("bbox", "csr"))
@pytest.mark.parametrize("out", ("q", "cake"))
def test_w6_split_methods_keep_pyfai_footprints_across_the_seam(seam_poni, out, method):
    _seam_row(seam_poni, out, *SEAMS["lower"], npt=_npt(out, 60), method=method)


def test_w7_seam_support_describes_the_directions_reduced(seam_poni):
    window, offset = SEAMS["lower"]
    _, support = _reduce(seam_poni, "q", window, offset, npt=50, support=True)
    _, reference, _ = _zero_reference(seam_poni, "q", window, offset, npt=50, support=True)
    assert support.available and reference.available, (support.reason, reference.reason)
    widths = sum(hi - lo for lo, hi in support.effective_window_deg)
    np.testing.assert_allclose(widths, 60.0, rtol=0, atol=1e-9)
    _same(support.supported_deg, reference.supported_deg)


def test_w7_non_crossing_and_degenerate_support_are_unchanged(seam_poni):
    from xrd_tools.core.angular_support import SupportReason
    _, kept = _reduce(seam_poni, "q", (20.0, 80.0), 30.0, npt=50, support=True)
    assert kept.available and kept.window_deg == 60.0
    np.testing.assert_allclose(np.array(kept.effective_window_deg), [[-10.0, 50.0]], rtol=0,
                               atol=1e-9)
    _, degenerate = _reduce(seam_poni, "q", (50.0, 50.0), 0.0, npt=50, support=True)
    assert degenerate.reason is SupportReason.DEGENERATE_WINDOW


@pytest.mark.parametrize("out", OUTS)
def test_w8_zero_offset_seam_window(seam_poni, out):
    _seam_row(seam_poni, out, (160.0, 200.0), 0.0, npt=_npt(out, 40))


@pytest.mark.parametrize("case,out", [(case, out) for case in DIAGNOSTICS for out in OUTS
                                      if (case, out) != ("w5", "q")])
def test_diag_window_shapes_record_parent_behavior(seam_poni, case, out):
    """Diagnosis only (W5 chi/cake and two shapes): print the outcome; no oracle, no credit."""
    window, offset, npt = DIAGNOSTICS[case]
    try:
        result, _ = _reduce(seam_poni, out, window, offset, npt=_npt(out, npt))
    except Exception as error:  # recorded, not judged
        print(f"DIAG {case} {out}: {type(error).__name__}: {error}")
        return
    axis = np.asarray(result.azimuthal if out == "cake" else result.radial, float)
    values = np.asarray(result.intensity, float)
    print(f"DIAG {case} {out}: axis {axis.size} [{axis.min():.4f}, {axis.max():.4f}] "
          f"finite {int(np.isfinite(values).sum())} nan {int(np.isnan(values).sum())} "
          f"sum {np.nansum(values):.9g}")
