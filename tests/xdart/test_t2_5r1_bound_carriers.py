"""O-1a-T2.5R.1 (§19.5/§19.6/§19.7) — bound-handle ownership, verified self-state,
exception-total recovery.

§19.5: after preflight the PATH is a diagnostic and an identity guard only — never
the data authority.  Every forward write and every restore goes through the handle
bound at preflight, so an object newly installed at the same path is left untouched.

§19.6: the three self-state fields are snapshotted, restored independently, read
back, and a failed restore is a NAMED recovery failure.

§19.7: no ordinary recovery exception escapes the transaction — every wrapper is
total, later recovery classes are still attempted, and the typed refusal carries
the full ordered ``recovery_failed_paths``.

Production-wired: real ``staticWidget``, real pyqtgraph ``Parameter`` handles, the
real staging/commit engine.
"""

from __future__ import annotations

import copy

import pytest
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.parametertree import Parameter


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


def _replacement(value="/tmp/replacement-must-stay.edf"):
    return Parameter.create(name="replacement-mask", type="str", value=value)


# ---------------------------------------------------------------------------
# §19.5 — the prepared carrier is the write authority
# ---------------------------------------------------------------------------

def test_prepared_carrier_is_frozen():
    """§19.5 req 1: a `__slots__` class whose attributes stay assignable is not
    an immutable transaction plan."""
    from xdart.gui.tabs.static_scan.static_scan_widget import _PreparedLegacyCarrier

    carrier = _PreparedLegacyCarrier(
        ("Signal", "mask_file"), object(), "v", "p", "e")
    for name, value in (
        ("path", ("BG", "File")),
        ("param", object()),
        ("value", "x"),
        ("prior", "y"),
        ("expected", "replacement"),
    ):
        with pytest.raises((AttributeError, TypeError)):
            setattr(carrier, name, value)


def test_prepared_carrier_payload_is_isolated_from_the_caller():
    """§19.5 req 1: a mutable payload handed to the carrier is normalised, so a
    later mutation of the caller's object cannot rewrite the plan."""
    from xdart.gui.tabs.static_scan.static_scan_widget import _PreparedLegacyCarrier

    source = ["a"]
    carrier = _PreparedLegacyCarrier(
        ("Signal", "mask_file"), object(), source, list(source), list(source))
    source.append("b")
    assert list(carrier.value) == ["a"]
    assert list(carrier.prior) == ["a"]
    assert list(carrier.expected) == ["a"]


def test_forward_write_never_mutates_a_replacement_at_the_same_path(
        widget, monkeypatch):
    """§19.5 req 2-4: the forward writer writes the BOUND handle.  A replacement
    resolved by path inside the writer must stay untouched."""
    path = ("Signal", "mask_file")
    params = widget.wrangler.parameters
    original = widget._controls_v2_param(path)
    prior = original.value()
    replacement = _replacement()
    replacement_prior = replacement.value()
    staged = widget.stage_controls_transaction(
        [(path, "/tmp/bound-forward.edf")])

    original_child = params.child
    calls = {"target": 0}

    def switching_child(*parts):
        if tuple(parts) != path:
            return original_child(*parts)
        calls["target"] += 1
        if calls["target"] == 3:
            return replacement
        return original

    monkeypatch.setattr(params, "child", switching_child)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert original.value() == prior
    assert replacement.value() == replacement_prior


def test_replacement_before_write_refuses_without_touching_either_object(
        widget, monkeypatch):
    """§19.5 req 3: a replacement detected BEFORE the write refuses; neither the
    bound original nor the replacement is written."""
    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    replacement = _replacement()
    replacement_prior = replacement.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/before.edf")])

    real_lookup = widget._controls_v2_param
    seen = {"n": 0}

    def lookup(p):
        if tuple(p) != path:
            return real_lookup(p)
        seen["n"] += 1
        # preflight binds the ORIGINAL; the pre-apply identity guard then sees a
        # replacement and must refuse before any write.
        return original if seen["n"] == 1 else replacement

    monkeypatch.setattr(widget, "_controls_v2_param", lookup)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.failed_path == path
    assert original.value() == prior
    assert replacement.value() == replacement_prior


def test_setter_that_mutates_then_raises_is_rolled_back_on_the_bound_handle(
        widget, monkeypatch):
    """§19.5 req 4: the carrier is pushed onto the attempted stack BEFORE the
    setter runs, so a setter that mutates and then raises is still restored."""
    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/mutate-raise.edf")])

    real_set = type(original).setValue

    def mutate_then_raise(self, value, *a, **k):
        if self is original and value == "/tmp/mutate-raise.edf":
            real_set(self, "/tmp/half-written.edf")
            raise RuntimeError("injected setter failure after mutation")
        return real_set(self, value, *a, **k)

    monkeypatch.setattr(type(original), "setValue", mutate_then_raise)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert original.value() == prior


def test_silent_readback_no_op_is_a_failure_not_a_success(widget, monkeypatch):
    """§19.5 req 4: the setter's return is never the authority — a write that
    silently does nothing must fail the transaction."""
    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/no-op.edf")])

    real_set = type(original).setValue

    def silent_no_op(self, value, *a, **k):
        if self is original and value == "/tmp/no-op.edf":
            return None
        return real_set(self, value, *a, **k)

    monkeypatch.setattr(type(original), "setValue", silent_no_op)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert original.value() == prior


def test_later_failure_rolls_back_an_earlier_bound_carrier(widget, monkeypatch):
    """§19.5 req 5: a later failure restores the earlier carrier through ITS bound
    handle, leaving any object newly installed at that path untouched."""
    first_path = ("Signal", "mask_file")
    second_path = ("BG", "File")
    first = widget._controls_v2_param(first_path)
    second = widget._controls_v2_param(second_path)
    first_prior = first.value()
    second_prior = second.value()
    replacement = _replacement("/tmp/replacement-stays.edf")
    replacement_prior = replacement.value()
    staged = widget.stage_controls_transaction(
        [(first_path, "/tmp/first-forward.edf"),
         (second_path, "/tmp/second-forward.edf")])

    real_set = type(second).setValue

    def fail_second(self, value, *a, **k):
        if self is second and value == "/tmp/second-forward.edf":
            return None                       # silent no-op -> readback fails
        return real_set(self, value, *a, **k)

    monkeypatch.setattr(type(second), "setValue", fail_second)
    # From this point the path resolves to a REPLACEMENT: the rollback of the
    # earlier carrier must not find it.
    real_lookup = widget._controls_v2_param
    monkeypatch.setattr(
        widget, "_controls_v2_param",
        lambda p: (replacement if tuple(p) == first_path and first.value() ==
                   "/tmp/first-forward.edf" else real_lookup(p)))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert first.value() == first_prior
    assert second.value() == second_prior
    assert replacement.value() == replacement_prior


def test_no_post_preflight_write_targets_a_path(widget, monkeypatch):
    """§19.5 req 6 / §22.10.A.2: ONE mechanism.  Every carrier write inside the
    transaction is handed the PREPARED CARRIER itself — never a path, and never a
    key into a widget-level registry that a second authority could redirect — so
    no post-preflight write can reach a replacement."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)

    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    staged = widget.stage_controls_transaction([(path, "/tmp/one-mechanism.edf")])
    targets = []
    real_writer = widget._controls_v2_write_legacy_carrier

    def recording_writer(parameters, carrier, value):
        targets.append(carrier)
        return real_writer(parameters, carrier, value)

    monkeypatch.setattr(
        widget, "_controls_v2_write_legacy_carrier", recording_writer)
    # Force a failure AFTER the legacy carrier is installed, so both the forward
    # write and the full rollback run.
    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert targets, "the forward carrier write did not reach the strict writer"
    for target in targets:
        assert isinstance(target, _PreparedLegacyCarrier)
        assert target.param is original
        assert not isinstance(target, (tuple, list))


# ---------------------------------------------------------------------------
# §19.6 — self-state restore is verified and named
# ---------------------------------------------------------------------------

def _stage_energy_pref(widget):
    prior = getattr(widget, "_controls_v2_source_energy_preference", "poni")
    requested = "metadata" if str(prior) != "metadata" else "poni"
    staged = widget.stage_controls_transaction(
        [(("Source", "energy_preference"), requested)])
    return prior, staged


def test_silently_refusing_self_state_setter_is_named(widget, monkeypatch):
    """§19.6 req 3-4: a restore that silently does nothing is a NAMED failure."""
    prior, staged = _stage_energy_pref(widget)
    assert staged.source_energy_preference != prior
    state = {"value": prior, "sets": 0}

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_preference",
            property(lambda _s: state["value"],
                     lambda _s, v: state.update(
                         value=v if state["sets"] == 0 else state["value"],
                         sets=state["sets"] + 1)),
            raising=False)
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Intent", "source_energy_preference") in result.recovery_failed_paths


def test_raising_self_state_setter_is_named_and_others_still_restore(
        widget, monkeypatch):
    """§19.6 req 2/5: fields restore INDEPENDENTLY — a raising setter is named and
    the remaining self-state fields are still restored."""
    prior_threshold = copy.deepcopy(widget._controls_v2_threshold_state)
    prior_explicit = widget._controls_v2_gi_selection_explicit
    prior, staged = _stage_energy_pref(widget)

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_preference",
            property(lambda _s: prior,
                     lambda _s, v: (_ for _ in ()).throw(
                         RuntimeError("injected self-state setter failure"))),
            raising=False)
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Intent", "source_energy_preference") in result.recovery_failed_paths
    assert widget._controls_v2_gi_selection_explicit == prior_explicit
    assert widget._controls_v2_threshold_state == prior_threshold


def test_multiple_self_state_failures_are_all_reported_in_order(
        widget, monkeypatch):
    """§19.6 req 5/7: every failing field is reported, in a deterministic order."""
    prior, staged = _stage_energy_pref(widget)

    def _boom(_s, _v):
        raise RuntimeError("injected")

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_preference",
            property(lambda _s: prior, _boom), raising=False)
        patch.setattr(
            type(widget), "_controls_v2_gi_selection_explicit",
            property(lambda _s: None, _boom), raising=False)
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
        result = widget.commit_controls_transaction(staged)

    named = [p for p in result.recovery_failed_paths if p[0] == "Intent"]
    assert ("Intent", "gi_selection_explicit") in named
    assert ("Intent", "source_energy_preference") in named
    assert named.index(("Intent", "gi_selection_explicit")) < named.index(
        ("Intent", "source_energy_preference"))


# ---------------------------------------------------------------------------
# §19.7 — recovery is exception-total
# ---------------------------------------------------------------------------

def test_display_backstop_exception_does_not_escape_and_later_classes_run(
        widget, monkeypatch):
    """§19.7 req 1-2/7: a raising display backstop is COLLECTED, the transaction
    returns a typed refusal, and the later (legacy) recovery class still runs."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    # A display-projecting edit is staged alongside the legacy carrier so the
    # display snapshot is non-empty and its recovery class really runs.
    staged = widget.stage_controls_transaction(
        [(path, "/tmp/display-fail.edf"), (("Int1D", "points"), "444")])

    def partial_then_raise(_snapshot):
        # Mutate the display scan, THEN raise: the projection is left half
        # applied, so recovery must really restore it.
        widget.scan.bai_1d_args["numpoints"] = 999_999
        raise RuntimeError("injected forward display failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan", partial_then_raise)
        patch.setattr(
            widget, "_controls_v2_restore_display_scan", lambda *_a, **_k: None)
        patch.setattr(
            staticWidget, "_controls_v2_apply_display_snapshot",
            staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected backstop failure"))))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    # later class attempted: the legacy carrier was restored on its bound handle
    assert original.value() == prior


def test_intent_backstop_exception_does_not_escape_and_later_classes_run(
        widget, monkeypatch):
    """§19.7 req 1-2/7: the same for the intent wrapper's backstop."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    # A display-projecting edit is staged alongside the legacy carrier so the
    # display snapshot is non-empty and its recovery class really runs.
    staged = widget.stage_controls_transaction(
        [(path, "/tmp/intent-fail.edf"), (("Int1D", "points"), "444")])

    with monkeypatch.context() as patch:
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected forward display failure")))
        patch.setattr(
            widget, "_controls_v2_restore_intent_values", lambda *_a, **_k: None)
        patch.setattr(
            staticWidget, "_controls_v2_apply_intent_snapshot",
            staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected intent backstop failure"))))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Intent",) in result.recovery_failed_paths
    assert original.value() == prior


def test_recovery_wrapper_regression_cannot_suppress_later_classes(
        widget, monkeypatch):
    """§19.7 req 2: the wrapper INVOCATIONS are wrapped too, so a regression
    inside one class cannot skip the recovery of later classes."""
    path = ("Signal", "mask_file")
    original = widget._controls_v2_param(path)
    prior = original.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/wrapper-boom.edf")])

    with monkeypatch.context() as patch:
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
        patch.setattr(
            widget, "_controls_v2_restore_display_scan_verified",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected wrapper regression")))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    assert original.value() == prior          # the later legacy class still ran


def test_typed_refusal_carries_reason_and_ordered_failed_paths(
        widget, monkeypatch):
    """§19.7 req 3/4/6: the original forward failure is preserved separately and
    the refusal carries `reason` plus the full ordered `recovery_failed_paths`
    with no duplicates or overwrites."""
    path = ("Signal", "mask_file")
    staged = widget.stage_controls_transaction([(path, "/tmp/ordered.edf")])

    with monkeypatch.context() as patch:
        patch.setattr(
            widget, "_controls_v2_apply_snapshot_to_scan",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected forward display failure")))
        patch.setattr(
            widget, "_controls_v2_restore_display_scan_verified",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
        patch.setattr(
            widget, "_controls_v2_restore_intent_values_verified",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "install"
    assert result.reason
    paths = list(result.recovery_failed_paths)
    assert ("Display",) in paths and ("Intent",) in paths
    # deterministic recovery order: display is restored before intent
    assert paths.index(("Display",)) < paths.index(("Intent",))
