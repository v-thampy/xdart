"""O-1a-T2.5R.3 (§24) — cache and cleanup residuals of T-2.5R.2.

Codex's exact-object review of `c05c314d` found one P1 and three P2 residuals
adjacent to the closed §22 findings:

* §24.2 **P1** — energy-preference invalidation is correctly inside the typed
  forward funnel, but recovery restored the source caches only under
  ``if source_reconciled:``.  An energy-preference-only transaction never sets
  ``source_selection_touched``, so a cache setter that installed ``None`` and THEN
  raised lost the prior cache with ``recovery_failed_paths == ()``.  Cache mutation
  and source-owner reconciliation are separate facts: the caches are now restored
  whenever ANY cache mutation was attempted, source-owner-first when both.
* §24.3 **P2** — ``collect()`` materialized a wrapper result inside its ``try`` but
  MERGED it into the global ``failures`` list outside that containment, with no
  segment validation.  A returned path holding an object whose ``__eq__`` raises
  escaped during the global dedup and skipped every later recovery class.
* §24.4 **P2** — the strict writer restored the root/child signal-block states in a
  ``finally`` that caught unblock exceptions and only logged them, so a write could
  report SUCCESS with signals still blocked.
* §24.5 **P2** — ``_controls_v2_capture_source_receipt()`` was evaluated
  unconditionally and outside containment in the ``ctx`` literal.

Production-wired: real ``staticWidget``, real pyqtgraph ``Parameter`` handles and
real Qt ``blockSignals``/``signalsBlocked``, the real staging/commit engine.
Cases 1-2 are the production-wired promotion of
``/private/tmp/test_codex_t25r2_residuals.py``.
"""

from __future__ import annotations

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


MASK = ("Signal", "mask_file")
#: a SOURCE-SELECTION legacy carrier — the only kind that reconciles the owner
INCLUDE_SUBDIR = ("Signal", "include_subdir")
ENERGY_PREF = ("Source", "energy_preference")


def _stage_energy_pref(widget):
    prior = getattr(widget, "_controls_v2_source_energy_preference", "poni")
    requested = "metadata" if str(prior) != "metadata" else "poni"
    return prior, widget.stage_controls_transaction([(ENERGY_PREF, requested)])


def _failing_unblock(owner, message):
    """A real ``blockSignals`` that BLOCKS fine but fails to RESTORE.

    Installed on the instance so the production write path calls it exactly as it
    calls Qt's own, and ``signalsBlocked()`` keeps reporting the true state."""
    real = type(owner).blockSignals

    def block(state, _real=real, _owner=owner):
        if not state:
            raise RuntimeError(message)
        return _real(_owner, state)

    return block


# ---------------------------------------------------------------------------
# §24.2 P1 — case 1: cache mutation is recovered without source reconciliation
# ---------------------------------------------------------------------------

def test_energy_cache_mutate_then_raise_restores_preference_and_exact_cache(
        widget, monkeypatch):
    """§24.7 case 1.  The energy-cache setter installs ``None`` and THEN raises.
    Both the preference AND the exact prior cache must come back, and the cache
    class must be NAMED — at `c05c314d` the cache restore was gated on
    `source_reconciled`, which an energy-preference-only transaction never sets."""
    prior_preference, staged = _stage_energy_pref(widget)
    prior_cache = {"prior": 1}
    state = {"cache": prior_cache, "raise_once": True}

    def get_cache(_self):
        return state["cache"]

    def set_cache(_self, value):
        state["cache"] = value
        if value is None and state["raise_once"]:
            state["raise_once"] = False
            raise RuntimeError("mutated cache, then raised")

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_cache",
            property(get_cache, set_cache), raising=False)
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "install"
        assert widget._controls_v2_source_energy_preference == prior_preference
        # the EXACT prior cache object's value, not None and not a replacement
        assert state["cache"] == prior_cache


def test_cache_recovery_does_not_require_source_reconciliation(
        widget, monkeypatch):
    """§24.2: the cache-touch fact is INDEPENDENT of `source_reconciled`.  A
    transaction that never touches the source selection still reaches the cache
    recovery class when it mutated a cache."""
    prior_preference, staged = _stage_energy_pref(widget)
    assert not staged.source_selection_touched, (
        "an energy-preference edit must not be a source-selection edit")
    widget._controls_v2_metadata_probe_cache = ("prior-probe", 2)
    seen = {"cache_class": False}
    state = {"cache": ("prior-energy", 1), "raise_once": True}

    def get_cache(_self):
        return state["cache"]

    def set_cache(_self, value):
        state["cache"] = value
        if value is None and state["raise_once"]:
            state["raise_once"] = False
            raise RuntimeError("mutated cache, then raised")

    real = widget._controls_v2_restore_source_caches_verified

    def watched(ctx):
        seen["cache_class"] = True
        return real(ctx)

    with monkeypatch.context() as patch:
        patch.setattr(
            type(widget), "_controls_v2_source_energy_cache",
            property(get_cache, set_cache), raising=False)
        patch.setattr(
            widget, "_controls_v2_restore_source_caches_verified", watched)
        result = widget.commit_controls_transaction(staged)

    assert not result.ok
    # the ONLY post-mutation failure point an energy-preference-only transaction
    # has is the invalidation itself, which is exactly the §24.2 P1 shape.
    assert seen["cache_class"], "the cache recovery class was never attempted"
    assert widget._controls_v2_source_energy_preference == prior_preference
    assert state["cache"] == ("prior-energy", 1)
    assert widget._controls_v2_metadata_probe_cache == ("prior-probe", 2)


def test_source_owner_is_recovered_before_the_caches_when_both_apply(
        widget, monkeypatch):
    """§24.2: splitting the two facts must NOT reorder them — the owner reconcile
    still runs BEFORE the caches are restored from the preflight snapshot."""
    param = widget._controls_v2_param(INCLUDE_SUBDIR)
    staged = widget.stage_controls_transaction(
        [(INCLUDE_SUBDIR, not bool(param.value()))])
    assert staged.source_selection_touched
    order = []
    real_owner = widget._controls_v2_restore_source_owner_verified
    real_caches = widget._controls_v2_restore_source_caches_verified

    monkeypatch.setattr(
        widget, "_controls_v2_restore_source_owner_verified",
        lambda *a, **k: (order.append("owner"), real_owner(*a, **k))[1])
    monkeypatch.setattr(
        widget, "_controls_v2_restore_source_caches_verified",
        lambda *a, **k: (order.append("cache"), real_caches(*a, **k))[1])
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert order == ["owner", "cache"]


# ---------------------------------------------------------------------------
# §24.3 P2 — case 2: recovery-result aggregation is exception-total
# ---------------------------------------------------------------------------

def test_equality_poison_recovery_path_is_contained_and_later_classes_run(
        widget, monkeypatch):
    """§24.7 case 2.  A wrapper returns a path segment whose ``__eq__`` raises.
    At `c05c314d` the global dedup comparison raised out of the collector and
    skipped every later recovery class."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])
    later = {"legacy": False}

    class Poison:
        def __eq__(self, _other):
            raise RuntimeError("poison path equality")

        def __hash__(self):
            return 0

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: [(Poison(),)])

    def later_legacy(*_args, **_kwargs):
        later["legacy"] = True
        return []

    monkeypatch.setattr(widget, "_controls_v2_rollback_legacy_all", later_legacy)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    # reduced to the class's stable fallback label ...
    assert ("Display",) in result.recovery_failed_paths
    # ... and NO attacker-controlled object entered the reported result
    for path in result.recovery_failed_paths:
        for segment in path:
            assert type(segment) is str and segment
    assert later["legacy"], "a later recovery class was skipped"


@pytest.mark.parametrize("malformed", [
    [()],                       # empty path
    [("Display", "")],          # empty segment
    [("Display", 3)],           # non-str segment
    [("Display", b"bytes")],    # bytes segment
    ["Display"],                # a bare string is not a path
    [42],
])
def test_malformed_recovery_paths_reduce_to_the_class_label(
        widget, monkeypatch, malformed):
    """§24.3: a valid recovery path is a NONEMPTY tuple/list of NONEMPTY ``str``
    segments; anything else becomes the class's stable fallback label."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])
    later = {"legacy": False}

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: malformed)

    def later_legacy(*_args, **_kwargs):
        later["legacy"] = True
        return []

    monkeypatch.setattr(widget, "_controls_v2_rollback_legacy_all", later_legacy)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    for path in result.recovery_failed_paths:
        for segment in path:
            assert type(segment) is str and segment
    assert later["legacy"]


def test_str_subclass_segment_with_poison_equality_is_rejected(
        widget, monkeypatch):
    """§24.3: ``type(segment) is not str`` is deliberately exact — a ``str``
    SUBCLASS may override ``__eq__``, which is the same hazard."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])

    class PoisonStr(str):
        def __eq__(self, _other):
            raise RuntimeError("poison str equality")

        def __hash__(self):
            return 0

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: [(PoisonStr("Display"), PoisonStr("x"))])
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    for path in result.recovery_failed_paths:
        for segment in path:
            assert type(segment) is str


# ---------------------------------------------------------------------------
# §24.4 P2 — cases 3-5, 8: signal-state restoration is part of the checked write
# ---------------------------------------------------------------------------

def test_child_unblock_failure_cannot_return_transaction_success(
        widget, monkeypatch):
    """§24.7 case 3 / §24.4 req 3-4.  The setter succeeds but the bound child's
    ``blockSignals(False)`` raises — the child is left BLOCKED.  At `c05c314d`
    that was logged and the transaction reported success."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-child.edf")])
    monkeypatch.setattr(
        param, "blockSignals", _failing_unblock(param, "child unblock failed"),
        raising=False)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == MASK
    assert "signal state not restored" in result.reason
    # req 7: the still-blocked child is NAMED at its stable path
    assert ("Signal", "child_signals") in result.recovery_failed_paths
    # the carrier itself was rolled back
    assert param.value() == prior


def test_root_unblock_failure_cannot_return_success_nor_suppress_child_cleanup(
        widget, monkeypatch):
    """§24.7 case 4 / §24.4 req 2.  The ROOT restore raises; the transaction must
    still fail AND the CHILD must still have been cleaned up."""
    root = widget.wrangler.parameters
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-root.edf")])
    monkeypatch.setattr(
        root, "blockSignals", _failing_unblock(root, "root unblock failed"),
        raising=False)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert "signal state not restored" in result.reason
    assert ("Signal", "root_signals") in result.recovery_failed_paths
    # the child was cleaned up independently of the root failure
    assert param.signalsBlocked() is False
    assert ("Signal", "child_signals") not in result.recovery_failed_paths
    assert param.value() == prior


def test_setter_and_unblock_failure_keeps_setter_primary_and_reports_cleanup(
        widget, monkeypatch):
    """§24.7 case 5 / §24.4 req 5.  When BOTH the setter and the child unblock
    fail, the SETTER stays the primary forward reason and the unresolved cleanup
    is still reported in ``recovery_failed_paths``."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    requested = "/tmp/t25r3-both.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            raise RuntimeError("setter exploded")
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    monkeypatch.setattr(
        param, "blockSignals", _failing_unblock(param, "child unblock failed"),
        raising=False)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    # req 5: the SETTER is the primary reason, not the cleanup
    assert "setter exploded" in result.reason
    assert "signal state not restored" not in result.reason
    # ... and the unresolved cleanup is still reported
    assert ("Signal", "child_signals") in result.recovery_failed_paths
    assert param.value() == prior


def test_silent_noop_unblock_is_caught_by_the_check_not_by_an_exception(
        widget, monkeypatch):
    """§24.4 req 3: the CHECK is the authority.  A restore that returns normally
    but does not change the state is caught through ``signalsBlocked()``."""
    param = widget._controls_v2_param(MASK)
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-noop.edf")])
    real = type(param).blockSignals

    def block(state, _real=real, _p=param):
        if not state:
            return True          # silently refuse to unblock
        return _real(_p, state)

    monkeypatch.setattr(param, "blockSignals", block, raising=False)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Signal", "child_signals") in result.recovery_failed_paths


def test_recovery_reattempt_that_succeeds_is_not_reported_as_outstanding(
        widget, monkeypatch):
    """§24.4 req 6-7: recovery REATTEMPTS and RE-VERIFIES.  An owner the retry
    restores is genuinely restored and must NOT be named as outstanding."""
    param = widget._controls_v2_param(MASK)
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-retry.edf")])
    real = type(param).blockSignals
    state = {"fail_unblock": True}

    def block(want, _real=real, _p=param):
        if not want and state["fail_unblock"]:
            state["fail_unblock"] = False       # fails once, then works
            raise RuntimeError("child unblock failed once")
        return _real(_p, want)

    monkeypatch.setattr(param, "blockSignals", block, raising=False)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok                        # the write is still a failure
    assert "signal state not restored" in result.reason
    assert param.signalsBlocked() is False      # a later attempt restored it
    assert ("Signal", "child_signals") not in result.recovery_failed_paths


def test_valid_write_blocks_both_sources_and_restores_exact_prior_states(
        widget, monkeypatch):
    """§24.7 case 8.  An ordinary VALID strict write must still block BOTH signal
    sources for the duration of the setter and restore their EXACT prior states."""
    root = widget.wrangler.parameters
    param = widget._controls_v2_param(MASK)
    assert root.signalsBlocked() is False and param.signalsBlocked() is False
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-valid.edf")])
    during = {}
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param:
            during["root"] = root.signalsBlocked()
            during["child"] = param.signalsBlocked()
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    result = widget.commit_controls_transaction(staged)

    assert result.ok
    assert during == {"root": True, "child": True}
    assert root.signalsBlocked() is False
    assert param.signalsBlocked() is False
    assert param.value() == "/tmp/t25r3-valid.edf"


def test_valid_write_restores_an_already_blocked_root_to_blocked(
        widget, monkeypatch):
    """§24.4 req 1: "EXACT prior states" means the CAPTURED states, not ``False``.
    A root that was already blocked before the write stays blocked afterwards."""
    root = widget.wrangler.parameters
    param = widget._controls_v2_param(MASK)
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-preblocked.edf")])
    root.blockSignals(True)
    try:
        during = {}
        real_set = type(param).setValue

        def setter(self, value, *args, **kwargs):
            if self is param:
                during["root"] = root.signalsBlocked()
                during["child"] = param.signalsBlocked()
            return real_set(self, value, *args, **kwargs)

        monkeypatch.setattr(type(param), "setValue", setter)
        result = widget.commit_controls_transaction(staged)

        assert result.ok
        assert during == {"root": True, "child": True}
        assert root.signalsBlocked() is True        # restored to its PRIOR state
        assert param.signalsBlocked() is False
    finally:
        root.blockSignals(False)


# ---------------------------------------------------------------------------
# §24.5 P2 — cases 6-7: receipt capture is conditional and typed
# ---------------------------------------------------------------------------

def test_non_source_transaction_captures_zero_source_receipts(
        widget, monkeypatch):
    """§24.7 case 6.  The receipt is needed only when the transaction can reach
    source reconciliation; an unrelated edit captures it ZERO times."""
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r3-noreceipt.edf")])
    assert not staged.source_selection_touched
    calls = {"n": 0}
    real = widget._controls_v2_capture_source_receipt

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(
        widget, "_controls_v2_capture_source_receipt", counting)
    result = widget.commit_controls_transaction(staged)

    assert result.ok
    assert calls["n"] == 0


def test_source_selection_transaction_captures_exactly_one_receipt(
        widget, monkeypatch):
    """§24.5 positive control, so case 6 is not vacuous: a source-selection
    transaction captures the receipt exactly once, and it is the typed record."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        SourceRecoveryReceipt)

    param = widget._controls_v2_param(INCLUDE_SUBDIR)
    staged = widget.stage_controls_transaction(
        [(INCLUDE_SUBDIR, not bool(param.value()))])
    assert staged.source_selection_touched
    seen = {"n": 0, "receipt": None}
    real = widget._controls_v2_capture_source_receipt

    def counting():
        seen["n"] += 1
        seen["receipt"] = real()
        return seen["receipt"]

    monkeypatch.setattr(
        widget, "_controls_v2_capture_source_receipt", counting)
    widget.commit_controls_transaction(staged)

    assert seen["n"] == 1
    assert isinstance(seen["receipt"], SourceRecoveryReceipt)


def test_receipt_capture_failure_is_a_typed_preflight_refusal_with_zero_writes(
        widget, monkeypatch):
    """§24.7 case 7.  A raising receipt capture on a source-selection transaction
    is a typed ``preflight`` refusal with ZERO writes — never an untyped escape
    out of the action/Run boundary, and never a manufactured empty receipt."""
    param = widget._controls_v2_param(INCLUDE_SUBDIR)
    prior = bool(param.value())
    staged = widget.stage_controls_transaction([(INCLUDE_SUBDIR, not prior)])
    monkeypatch.setattr(
        widget, "_controls_v2_capture_source_receipt",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("host property exploded")))

    try:
        result = widget.commit_controls_transaction(staged)
    except Exception as exc:                      # pragma: no cover - the defect
        pytest.fail(f"untyped exception escaped the transaction: {exc!r}")

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == ("Source",)
    assert result.recovery_failed_paths == ()
    assert bool(param.value()) == prior           # ZERO writes


def test_absent_receipt_still_cannot_certify_source_recovery(widget):
    """§24.5 + §22.10.E.6 retained: an empty receipt is never manufactured, and
    absence fails CLOSED rather than certifying restoration."""
    assert widget._controls_v2_source_restore_verified(None) is False
    assert widget._controls_v2_restore_source_owner_verified(
        {"source_receipt": None}, []) == [("Source",)]


def test_global_failure_merge_lives_inside_collect_containment():
    """§24.3 / §24.7 mutation 3: the MERGE into the global ``failures`` list must
    itself be inside ``collect()``'s exception boundary.

    Once per-segment validation is in place, no *behavioral* probe can separate
    "merge inside" from "merge outside": nothing that can raise on comparison
    survives validation, so the containment is defense in depth (recorded as a
    discrepancy in Boundary 7).  §24.3 nevertheless requires the whole operation
    to remain within containment, so this pins it structurally — at least one
    ``failures.append(...)`` must live in the ``try`` BODY, not only after it."""
    import ast
    import inspect
    import textwrap

    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    source = textwrap.dedent(
        inspect.getsource(staticWidget._controls_v2_recover_all))
    outer = ast.parse(source).body[0]
    collect = next(
        node for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef) and node.name == "collect")
    tries = [node for node in collect.body if isinstance(node, ast.Try)]
    assert len(tries) == 1, "collect() must have exactly one exception boundary"

    def _merges(nodes):
        found = 0
        for node in nodes:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "append"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "failures"):
                    found += 1
        return found

    assert _merges(tries[0].body) >= 1, (
        "the merge into `failures` is OUTSIDE collect()'s try body")
