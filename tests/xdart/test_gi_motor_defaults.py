# -*- coding: utf-8 -*-
"""The GI θ-motor dropdown default-selection policy + the stale-gi_config guard.

Pins the maintainer-specified rule (recurs across sessions): show every real
motor, default-pick the named incidence axis (th/eta/halpha/gonth/theta), else a
rotation-sounding motor, else Manual — and NEVER carry a dead motor across a
source switch (the ``th``-leak: a reused LiveScan's stale ``gi_config``).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xdart.gui.tabs.static_scan.gi_motor_defaults import (  # noqa: E402
    GI_MOTOR_PREFERENCE,
    pick_default_gi_motor,
)


# ---- default-selection policy ------------------------------------------------

def test_named_preference_wins_when_present():
    motors = ["halpha", "detx", "dety", "hx", "hy", "hz", "sbsx", "sbsy", "sfpx"]
    # halpha is the only named preference present -> it wins over the others.
    assert pick_default_gi_motor(motors) == "halpha"


def test_preference_order_is_honoured():
    # 'th' precedes 'halpha' in GI_MOTOR_PREFERENCE, so it wins when both exist.
    assert pick_default_gi_motor(["halpha", "eta", "th"]) == "th"
    assert "th" == GI_MOTOR_PREFERENCE[0]


def test_preference_is_case_insensitive():
    assert pick_default_gi_motor(["HALPHA", "detx"]) == "HALPHA"


def test_rotation_heuristic_when_no_named_preference():
    # No named preference present; a phi-bearing motor reads as a rotation axis.
    assert pick_default_gi_motor(["detx", "samphi", "dety"]) == "samphi"
    # 'omega'/'chi'/'gonio' style names too.
    assert pick_default_gi_motor(["detx", "chiR"]) == "chiR"
    assert pick_default_gi_motor(["stagex", "gonio"]) == "gonio"


def test_manual_when_nothing_looks_like_rotation():
    # Pure translation stages -> no incidence axis -> Manual (do not pick detx).
    assert pick_default_gi_motor(["detx", "dety", "sbsx", "sbsy"]) == "Manual"


def test_manual_when_empty():
    assert pick_default_gi_motor([]) == "Manual"
    assert pick_default_gi_motor(None) == "Manual"


def test_never_returns_a_name_not_in_the_list():
    # The result is always either a member of the input or the literal 'Manual'.
    for motors in ([], ["detx"], ["th"], ["samomega", "detx"]):
        out = pick_default_gi_motor(motors)
        assert out == "Manual" or out in motors


# ---- the stale-gi_config guard (the th-leak root cause) ----------------------

def test_reset_clears_gi_config():
    """LiveScan.reset() must clear gi_config so the single reused scan object
    does not carry a previous file's incidence motor into the next load."""
    from xdart.modules.live import LiveScan

    scan = LiveScan("null_main")
    # Simulate a prior processed scan having stamped a GI incidence motor.
    scan.gi_config = {"incidence_motor": "th", "tilt_angle": 0.0}
    scan.reset()
    assert scan.gi_config == {}
