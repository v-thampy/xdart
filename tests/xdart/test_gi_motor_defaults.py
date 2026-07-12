# -*- coding: utf-8 -*-
"""The GI θ-motor dropdown default-selection policy + the stale-gi_config guard.

Pins the maintainer-specified rule (recurs across sessions): show every real
motor, default-pick the named incidence axis (th/eta/halpha/gonth/theta, plus
alpha_i/mu/incidence since F3), else a TOKEN-AWARE rotation-sounding motor,
else Manual — and NEVER carry a dead motor across a source switch (the
``th``-leak: a reused LiveScan's stale ``gi_config``).
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


# ---- F3 (Codex review 2026-07-11): the substring leak -------------------------

def test_f3_width_home_translation_stages_fall_to_manual():
    """The 2-char hints matched THROUGH words: slit_wid**th**, sample_h**om**e.
    A source with only translation stages must default to Manual, never silently
    pick a slit or a homing stage as the GI incidence axis."""
    assert pick_default_gi_motor(
        ["slit_width", "sample_home", "slit_length", "beam_width"]) == "Manual"
    # Each one alone must also miss (isolate the four false-positive names).
    for name in ("slit_width", "sample_home", "slit_length", "beam_width"):
        assert pick_default_gi_motor([name, "detx"]) == "Manual", name


def test_f3_more_embedded_hint_names_rejected():
    # r**ang**e / ma**chi**ne / gra**phi**te / fl**ang**e: token-aware matching
    # must not fire on hints buried inside unrelated words.
    assert pick_default_gi_motor(
        ["home_x", "range_z", "machine_x", "graphite_x", "flange_z"]) == "Manual"


def test_f3_rotation_heuristic_still_catches_decorated_names():
    # Tightening must NOT lose the intended catches (affix + camelCase + digits).
    assert pick_default_gi_motor(["detx", "samomega"]) == "samomega"
    assert pick_default_gi_motor(["detx", "theta2"]) == "theta2"
    assert pick_default_gi_motor(["detx", "rot_z"]) == "rot_z"


def test_f3_new_incidence_aliases_recognized():
    # Maintainer decision 2026-07-12: alpha_i / mu / incidence are real
    # incidence-axis names — pick them rather than falling to Manual.
    assert pick_default_gi_motor(["detx", "alpha_i"]) == "alpha_i"
    assert pick_default_gi_motor(["detx", "mu"]) == "mu"
    assert pick_default_gi_motor(["detx", "incidence"]) == "incidence"
    # Token-aware, so a decorated form is caught too...
    assert pick_default_gi_motor(["detx", "sample_mu"]) == "sample_mu"
    # ...but never as a mid-word substring.
    assert pick_default_gi_motor(["muffin_x", "detx"]) == "Manual"


def test_f3_original_preference_outranks_new_aliases():
    # th/eta/halpha/gonth/theta keep their historical priority.
    assert pick_default_gi_motor(["alpha_i", "th"]) == "th"
    assert pick_default_gi_motor(["mu", "theta"]) == "theta"


def test_f3_decorated_halpha_recognized():
    # Bare 'halpha' wins via the preference; decorated forms must be caught by
    # the token fallback too (maintainer request 2026-07-12).
    assert pick_default_gi_motor(["detx", "sam_halpha"]) == "sam_halpha"
    assert pick_default_gi_motor(["detx", "halpha2"]) == "halpha2"
    assert pick_default_gi_motor(["HALPHA", "detx"]) == "HALPHA"


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
