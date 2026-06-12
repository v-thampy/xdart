"""Headless guards for the batch-GI pre-release blocker.

These cover the three behaviours requested for ``harden/gi-batch``:

1. ``LiveFrame._get_incident_angle`` never silently returns 0° — it
   resolves a numeric (Manual) value or a metadata motor, and otherwise
   raises :class:`IncidenceAngleUnresolved` so callers surface "set Manual
   theta" instead of integrating a degenerate 0° (blank) GI cake.
2. The GI 2D auto-range freeze refuses a collapsed/degenerate grid rather
   than propagating a squashed range to every frame in the scan.
3. A blank (all-dummy) GI 2D result is detectable, and the scout's
   incidence resolution matches per-frame processing (live ↔ batch
   equivalence at the incidence layer — both build the frame from the same
   ``img_meta`` + ``incidence_motor``).

All are pure / Qt-free and don't require pyFAI integration or the real
detector data — they exercise the decision logic that turned an
unresolved/degenerate incidence into a silent blank cake.
"""

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult2D
from xdart.modules.live import LiveFrame, IncidenceAngleUnresolved
from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
    _padded_axis_range,
    _freeze_gi_2d_ranges_from_result,
    _result_intensity_all_dummy,
)


def _gi_frame(scan_info, incidence_motor):
    """Minimal GI LiveFrame for incidence-resolution tests."""
    return LiveFrame(
        0, None, scan_info=dict(scan_info), gi=True,
        th_mtr=incidence_motor,
    )


# --- (1) _get_incident_angle never silent-0.0 -------------------------------

def test_incidence_resolves_manual_numeric():
    # Manual-Theta path: incidence_motor is the entered angle string.
    frame = _gi_frame({}, "0.30")
    assert frame._get_incident_angle() == pytest.approx(0.30)


def test_incidence_resolves_from_metadata_motor():
    # Motor-name path: case-insensitive lookup in scan_info.
    frame = _gi_frame({"TH": 0.42, "i0": 1000}, "th")
    assert frame._get_incident_angle() == pytest.approx(0.42)


def test_incidence_raises_when_motor_missing():
    # Eiger / no-metadata regression: motor name not numeric and absent from
    # metadata must NOT silently default to 0° (degenerate GI cake).
    frame = _gi_frame({"i0": 1000}, "th")
    with pytest.raises(IncidenceAngleUnresolved):
        frame._get_incident_angle()


def test_incidence_raises_when_metadata_value_non_numeric():
    # Motor present but its value can't be coerced -> still unresolved, not 0°.
    frame = _gi_frame({"th": "open"}, "th")
    with pytest.raises(IncidenceAngleUnresolved):
        frame._get_incident_angle()


def test_zero_is_only_returned_when_explicitly_manual_zero():
    # A genuine, explicit 0.0 Manual entry is honoured (caller's choice);
    # the guard is specifically against the *defaulted* 0°.
    frame = _gi_frame({}, "0.0")
    assert frame._get_incident_angle() == 0.0


# --- (2) degenerate-range freeze guard --------------------------------------

def test_padded_axis_range_pads_real_axis():
    rng = _padded_axis_range(np.linspace(0.0, 2.0, 50))
    assert rng is not None
    lo, hi = rng
    assert lo < 0.0 and hi > 2.0           # padded outward
    assert (hi - lo) > 2.0                  # non-degenerate span


def test_padded_axis_range_refuses_collapsed_axis():
    # Every finite value identical -> collapsed -> refuse (None), so we never
    # freeze a squashed grid that would blank the whole scan.
    assert _padded_axis_range(np.full(64, 0.1)) is None
    assert _padded_axis_range(np.array([np.nan, np.nan])) is None
    assert _padded_axis_range(None) is None


def test_freeze_gi_2d_freezes_non_degenerate_qoop():
    args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None}
    result = IntegrationResult2D(
        radial=np.linspace(-1.0, 3.0, 40),        # qip  -> x_range
        azimuthal=np.linspace(0.0, 2.5, 30),      # qoop -> y_range
        intensity=np.ones((40, 30)),
    )
    changed = _freeze_gi_2d_ranges_from_result(args, result)
    assert changed
    # qoop (y_range) must come back non-degenerate.
    ylo, yhi = args["y_range"]
    assert (yhi - ylo) > 2.0


def test_freeze_gi_2d_refuses_degenerate_qoop():
    # Collapsed qoop axis (degenerate incidence): y_range must stay unfrozen
    # rather than freezing a squashed range onto every frame.
    args = {"gi_mode_2d": "qip_qoop", "x_range": None, "y_range": None}
    result = IntegrationResult2D(
        radial=np.linspace(-1.0, 3.0, 40),
        azimuthal=np.full(30, 0.05),              # collapsed qoop
        intensity=np.ones((40, 30)),
    )
    _freeze_gi_2d_ranges_from_result(args, result)
    assert args["x_range"] is not None            # qip still freezes
    assert args["y_range"] is None                # degenerate qoop refused


# --- (3) all-dummy detection + live<->batch incidence equivalence -----------

def test_gi_2d_not_all_dummy_detector():
    blank = IntegrationResult2D(
        radial=np.linspace(0, 1, 10), azimuthal=np.linspace(0, 1, 8),
        intensity=np.full((10, 8), -1.0),
    )
    real = IntegrationResult2D(
        radial=np.linspace(0, 1, 10), azimuthal=np.linspace(0, 1, 8),
        intensity=np.where(np.eye(10, 8) > 0, 5.0, -1.0),
    )
    empty = IntegrationResult2D(
        radial=np.array([]), azimuthal=np.array([]),
        intensity=np.empty((0, 0)),
    )
    assert _result_intensity_all_dummy(blank) is True
    assert _result_intensity_all_dummy(empty) is True
    assert _result_intensity_all_dummy(real) is False


def test_scout_and_per_frame_incidence_equivalent():
    # live (per-frame) and batch (scout) build the frame from the SAME
    # img_meta + incidence_motor, so the resolved incidence must match.
    img_meta = {"th": 0.37, "i0": 1234}
    motor = "th"
    scout = _gi_frame(img_meta, motor)            # _freeze_gi_2d_auto_ranges scratch
    per_frame = _gi_frame(img_meta, motor)        # _process_one frame
    assert scout._get_incident_angle() == per_frame._get_incident_angle()

    # And both refuse identically when incidence is unresolvable.
    img_meta_no_th = {"i0": 1234}
    scout2 = _gi_frame(img_meta_no_th, motor)
    per_frame2 = _gi_frame(img_meta_no_th, motor)
    for f in (scout2, per_frame2):
        with pytest.raises(IncidenceAngleUnresolved):
            f._get_incident_angle()
