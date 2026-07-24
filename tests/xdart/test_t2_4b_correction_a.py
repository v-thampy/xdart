"""O-1a-T2.4b (§17.10) — Correction-A completion, beyond the real-click matrix.

Covers §17.5 (Config-Save veto fails closed on BOTH a typed refusal and a
raising hook, emitting the structured ``config_save_refused`` precondition
event), §17.7 (revision-aware refusal dedupe — a new invalid revision is its own
visible message), §17.10-5 (the accepted/invalid strict-schema corpus at the
user-reachable ``stage_controls_transaction`` seam), and the §17.8 pure-builder
mutation sentinels, committed RED (xfail-strict) as they are explicitly assigned
to T-2.5's ownership rework.
"""

from __future__ import annotations

import copy

import pytest
from pyqtgraph.Qt import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


# --------------------------------------------------------------------------- #
# §17.5 — Config-Save veto fails closed on BOTH a typed refusal and a raise.
# --------------------------------------------------------------------------- #

def test_config_save_typed_veto_writes_no_file(widget, tmp_path):
    """A pending INVALID edit makes the checked veto return True — the Config
    Save writes no file (the normal typed-refusal path)."""
    widget._controls_v2_record_edit(
        ("Int1D", "points"), "4.5", origin="draft")
    target = tmp_path / "typed-veto-no-save.json"

    widget.h5viewer.defaultWidget.save_defaults(fname=str(target))

    assert not target.exists()


def test_config_save_raising_hook_fails_closed_with_event(
        widget, monkeypatch, tmp_path):
    """§17.5 / Codex T-2.4 test 2: a veto hook whose validation owner RAISES must
    fail CLOSED — no file — and surface a structured ``config_save_refused``
    event (phase ``precondition``) plus one user-visible status."""
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    def fail_commit():
        raise RuntimeError("injected veto failure")

    monkeypatch.setattr(widget, "_commit_controls_v2_pending_edits", fail_commit)
    events = []
    monkeypatch.setattr(
        module, "run_config_debug_log",
        lambda _logger, event, **kw: events.append((event, kw)))
    messages = []
    monkeypatch.setattr(widget, "_controls_v2_status_message", messages.append)
    target = tmp_path / "raising-veto-no-save.json"

    widget.h5viewer.defaultWidget.save_defaults(fname=str(target))

    assert not target.exists()
    refused = [kw for event, kw in events if event == "config_save_refused"]
    assert refused and refused[0].get("phase") == "precondition"
    assert messages and "no file" in messages[0].lower()


# --------------------------------------------------------------------------- #
# §17.7 — refusal dedupe is revision-aware.
# --------------------------------------------------------------------------- #

def test_new_invalid_revision_gets_its_own_visible_refusal(widget, monkeypatch):
    """Codex T-2.4 test 3: two distinct invalid focus-loss edits (a higher
    revision each) produce two visible messages — dedupe suppresses only the
    same-revision action replay, never a new bad edit."""
    messages = []
    monkeypatch.setattr(widget, "_controls_v2_status_message", messages.append)
    monkeypatch.setattr(widget, "_refresh_controls_v2_profile", lambda *a, **k: None)

    widget._on_controls_v2_field_changed(("Int1D", "points"), "4.5")
    widget._on_controls_v2_field_changed(("Int1D", "points"), "5.5")

    assert len(messages) == 2


def test_same_revision_action_replay_is_deduped(widget, monkeypatch):
    """The immediate action replay of the SAME journal revision stays one
    message (the invalid focus-loss edit followed by its Advanced click)."""
    from xrd_tools.session.readiness import ControlAction

    messages = []
    monkeypatch.setattr(widget, "_controls_v2_status_message", messages.append)
    monkeypatch.setattr(widget, "_refresh_controls_v2_profile", lambda *a, **k: None)
    monkeypatch.setattr(widget, "_show_integration_advanced", lambda: None)

    widget._on_controls_v2_field_changed(("Int1D", "points"), "4.5")
    widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

    assert len(messages) == 1


# --------------------------------------------------------------------------- #
# §17.10-5 — accepted/invalid strict-schema corpus at the user-reachable seam.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("Int1D", "points"), "500"),               # integral
        (("Int1D", "unit"), "q"),                    # unit alias
        (("Int1D", "axis"), "q"),                    # axis (non-GI -> unit path)
        (("Int1D", "method"), "csr"),                # method whitelist
        (("Source", "energy_preference"), "metadata"),  # source-preference
        (("Int1D", "correctSolidAngle"), "true"),    # boolean (total schema)
        (("Int1D", "correctSolidAngle"), "false"),
    ],
)
def test_strict_schema_corpus_accepts_valid(widget, path, value):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        StagedControlsTransaction,
    )

    result = widget.stage_controls_transaction([(path, value)])

    assert isinstance(result, StagedControlsTransaction)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("Int1D", "points"), "4.5"),                # non-integral
        (("Int1D", "points"), "0"),                  # below minimum
        (("Int1D", "unit"), "2garbage"),             # fuzzy-garbage unit
        (("Int1D", "unit"), "machine"),
        (("Int1D", "axis"), "2garbage"),             # fuzzy-garbage axis
        (("Int1D", "method"), "not_a_method"),       # unsupported method
        (("Source", "energy_preference"), "xyz"),    # unknown source-preference
    ],
)
def test_strict_schema_corpus_refuses_invalid(widget, path, value):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError,
    )

    result = widget.stage_controls_transaction([(path, value)])

    assert isinstance(result, ControlsTransactionError)
    assert result.path == path


# --------------------------------------------------------------------------- #
# §17.8 — the "pure" builder/serializer still mutates.  ASSIGNED TO T-2.5:
# mutation sentinels — committed RED (xfail-strict) at T-2.4b; T-2.5 makes the
# builders pure / keeps the contradiction journal-only and removes the markers.
# --------------------------------------------------------------------------- #

def test_native_plan_builder_is_side_effect_free(widget, monkeypatch):
    from xdart.gui.tabs.static_scan import static_scan_widget as module
    from xrd_tools.session.run_configuration import GIIntent

    intent = widget._controls_v2_ensure_run_intent()
    intent.gi = GIIntent(
        enabled=True,
        incidence_motor="Manual",
        th_val=0.2,
        sample_orientation=4,
        tilt_angle=0.0,
    )
    widget.scan.gi = False
    widget.scan.gi_config = {"sentinel": "unchanged"}
    scan_before = {
        "gi": widget.scan.gi,
        "gi_config": copy.deepcopy(widget.scan.gi_config),
        "incidence_motor": copy.deepcopy(widget.scan.incidence_motor),
    }
    monkeypatch.setattr(
        module,
        "build_native_int_reduction_plan_from_scan",
        lambda *_a, **_k: object(),
    )

    widget._controls_v2_native_reduction_plan()

    assert widget.scan.gi == scan_before["gi"]
    assert widget.scan.gi_config == scan_before["gi_config"]
    assert widget.scan.incidence_motor == scan_before["incidence_motor"]


def test_transient_cross_field_state_is_not_serialized_as_committed(
        widget, monkeypatch):
    from xrd_tools.session.run_configuration import FrozenThresholdPolicy

    monkeypatch.setattr(widget, "_refresh_controls_v2_profile", lambda *a, **k: None)
    widget._on_controls_v2_field_changed(("Mask", "max"), "100")
    widget._on_controls_v2_field_changed(("Mask", "min"), "200")

    state = widget._controls_v2_int_session_state()["threshold_config"]
    FrozenThresholdPolicy(**state)
