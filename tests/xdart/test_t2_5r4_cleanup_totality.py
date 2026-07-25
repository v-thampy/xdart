"""O-1a-T2.5R.4 (§25) — recovery cleanup is exception-total.

Codex's exact-object review of `5dfc1776` reproduced five defects:

* §25.3 **P1** — `_controls_v2_validated_recovery_path` accepted `tuple`/`list`
  SUBCLASSES through `isinstance()`, validated ONE iteration, then took a SECOND
  iteration in `tuple(item)`.  A stateful container yielded exact strings while
  being validated and a non-string / equality-poison object while being converted;
  the poison then reached the FALLBACK membership merge, which sat OUTSIDE
  `collect()`'s containment, so it escaped and skipped every later recovery class.
* §25.4 **P1** — both rollback paths swallowed their unblock exceptions, and the
  final signal recovery was seeded ONLY from the forward writer's
  `_ControlsStrictWriteError`.  A clean forward write + a later failure + a raising
  ROLLBACK unblock therefore left the child permanently blocked and unnamed.
* §25.5 **P2** — a restore that put the state back and THEN raised was logged and
  erased: readback matched, `unrestored` was empty, and a good setter returned
  `ok=True`, violating §24.4 req 4.
* §25.6 **P2** — the prior state came from `blockSignals(True)`'s RETURN value, so a
  call that blocked the owner and then raised lost it and cleanup skipped the owner
  as "never blocked".
* §25.7 **P2** — the captured receipt was never type-checked: `None`, `object()`,
  and an equal-shaped plain tuple all proceeded past preflight.

Cases 1-2, 6, and 9's first form are the production-wired promotion of
`/Users/vthampy/repos/tmp/test_codex_t25r3_exact_review.py`; the root, force-restore,
acquire-then-raise, and transient/permanent variants are added here for full
ownership coverage (§25.10).

Production-wired: real ``staticWidget``, real pyqtgraph ``Parameter`` handles, real
Qt ``blockSignals``/``signalsBlocked``, the real staging/commit engine.
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
INCLUDE_SUBDIR = ("Signal", "include_subdir")
CHILD = ("Signal", "child_signals")
ROOT = ("Signal", "root_signals")


def _stage_mask(widget, value="/tmp/t25r4.edf"):
    return widget.stage_controls_transaction([(MASK, value)])


def _force_recovery_after_clean_write(widget, monkeypatch, state):
    """Make the transaction fail AFTER a clean forward legacy write."""
    def boom(*_a, **_k):
        state["rollback"] = True
        raise RuntimeError("force post-write recovery")

    monkeypatch.setattr(widget, "_controls_v2_apply_snapshot_to_scan", boom)


# ---------------------------------------------------------------------------
# §25.3 P1 / §25.10 cases 1-2 — single-snapshot, exact-type recovery paths
# ---------------------------------------------------------------------------

def _flip_path_factory(second):
    """A ``list`` SUBCLASS that yields safe strings on the FIRST iteration and
    *second* on every later one — the exact time-of-check/time-of-use adversary."""
    class FlipPath(list):
        def __init__(self):
            super().__init__(["placeholder"])
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations == 1:
                return iter(("Display", "safe"))
            return iter(second)

    return FlipPath


def test_changing_container_cannot_emit_a_non_string_after_validation(
        widget, monkeypatch):
    """§25.10 case 1.  A container that validates as strings and then yields a
    non-string must not get a non-string segment into the reported result."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])
    flip = _flip_path_factory(("Display", 42))

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: [flip()])
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.recovery_failed_paths
    for path in result.recovery_failed_paths:
        assert type(path) is tuple and path
        for segment in path:
            assert type(segment) is str and segment
    # reduced to the class's stable fallback label
    assert ("Display",) in result.recovery_failed_paths


def test_changing_path_cannot_escape_the_fallback_merge_or_skip_a_later_class(
        widget, monkeypatch):
    """§25.10 case 2.  The poison used to reach the FALLBACK membership test, which
    sat outside `collect()`'s containment, and skip every later recovery class."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])
    later = {"legacy": False}

    class Poison:
        def __eq__(self, _other):
            raise RuntimeError("late poison equality")

        def __hash__(self):
            return 0

    flip = _flip_path_factory(("Intent", Poison()))

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: [flip()])
    monkeypatch.setattr(
        widget, "_controls_v2_restore_intent_values_verified",
        lambda *_a, **_k: [])
    # a LATER class then takes the fallback path, exercising the fallback merge
    monkeypatch.setattr(
        widget, "_controls_v2_restore_self_state_verified",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force fallback")))

    def later_legacy(*_args, **_kwargs):
        later["legacy"] = True
        return []

    monkeypatch.setattr(widget, "_controls_v2_rollback_legacy_all", later_legacy)

    try:
        result = widget.commit_controls_transaction(staged)
    except Exception as exc:                     # pragma: no cover - the defect
        pytest.fail(f"recovery exception escaped the typed boundary: {exc!r}")

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    assert ("Intent", "self_state") in result.recovery_failed_paths
    assert later["legacy"], "a later recovery class was skipped"
    for path in result.recovery_failed_paths:
        for segment in path:
            assert type(segment) is str and segment


@pytest.mark.parametrize("container", ["tuple_subclass", "list_subclass"])
def test_tuple_and_list_subclasses_are_refused_outright(widget, monkeypatch,
                                                        container):
    """§25.3 req 1: only the EXACT built-in containers are admitted, so `__iter__`
    cannot be overridden at all."""
    intent = widget._controls_v2_ensure_run_intent()
    current = intent.bai_1d_args["numpoints"]
    staged = widget.stage_controls_transaction(
        [(("Int1D", "points"), str(int(current) + 1))])

    base = tuple if container == "tuple_subclass" else list

    class Sneaky(base):
        pass

    payload = Sneaky(("Display", "looks-fine"))
    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force recovery")))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan_verified",
        lambda *_a, **_k: [payload])
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Display",) in result.recovery_failed_paths
    for path in result.recovery_failed_paths:
        assert type(path) is tuple


def test_validated_path_is_the_exact_snapshot_that_is_returned():
    """§25.3 req 2: the returned object is the snapshot taken BEFORE inspection."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_validated_recovery_path)

    assert _controls_v2_validated_recovery_path(["A", "b"]) == ("A", "b")
    assert type(_controls_v2_validated_recovery_path(["A", "b"])) is tuple
    for bad in (["A", ""], ["A", 3], [], ("A", b"b"), "A", 4, {"A"}):
        with pytest.raises((TypeError, ValueError)):
            _controls_v2_validated_recovery_path(bad)


# ---------------------------------------------------------------------------
# §25.4 P1 / §25.10 cases 3-5 — rollback-created signal strands
# ---------------------------------------------------------------------------

def _armed_unblock(param, message, permanent):
    """A real ``blockSignals`` that fails to RESTORE while ``state['armed']``.

    ``permanent=False`` disarms after one failure (the TRANSIENT form, which final
    recovery repairs); ``permanent=True`` keeps failing (the PERMANENT form)."""
    real = type(param).blockSignals
    state = {"armed": False}

    def block(wanted):
        if not wanted and state["armed"]:
            if not permanent:
                state["armed"] = False
            raise RuntimeError(message)
        return real(param, wanted)

    return block, state, real


def _arm_from(widget, monkeypatch, method_name, state):
    """Arm the injected failure from the moment *method_name* is entered.

    It is deliberately NOT disarmed on exit: the TRANSIENT injection disarms itself
    after its single failure, while the PERMANENT one must keep failing through the
    final signal-state recovery backstop as well — otherwise "permanent" would only
    describe the rollback window and final recovery would always repair it."""
    real = getattr(widget, method_name)

    def wrapped(*args, **kwargs):
        state["armed"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(widget, method_name, wrapped)


@pytest.mark.parametrize("permanent", [False, True])
def test_normal_rollback_unblock_failure_is_observed_and_handled(
        widget, monkeypatch, permanent):
    """§25.10 cases 3-4.  A clean forward write, a later failure, and a raising
    NORMAL rollback unblock.  At `5dfc1776` nothing was registered at all, so the
    child stayed BLOCKED and unnamed.

    TRANSIENT form: final recovery repairs it -> prior state restored, and per
    §25.4 req 6 it is NOT an outstanding recovery_failed_path.
    PERMANENT form: it remains mismatched -> named at the stable child path."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = _stage_mask(widget, "/tmp/t25r4-normal-rollback.edf")
    block, state, real_block = _armed_unblock(
        param, "rollback unblock failed", permanent)
    monkeypatch.setattr(param, "blockSignals", block, raising=False)
    _arm_from(widget, monkeypatch, "_controls_v2_restore_carrier_value", state)
    _force_recovery_after_clean_write(widget, monkeypatch, {"rollback": False})

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert param.value() == prior            # the carrier itself rolled back
        if permanent:
            assert param.signalsBlocked() is True
            assert CHILD in result.recovery_failed_paths
        else:
            assert param.signalsBlocked() is False   # the leak is REPAIRED
            assert CHILD not in result.recovery_failed_paths
    finally:
        real_block(param, False)


@pytest.mark.parametrize("permanent", [False, True])
def test_force_rollback_unblock_failure_is_observed_and_handled(
        widget, monkeypatch, permanent):
    """§25.10 case 5.  The FORCE-restore backstop has the same guarantees."""
    param = widget._controls_v2_param(MASK)
    staged = _stage_mask(widget, "/tmp/t25r4-force-rollback.edf")
    shared = {"rollback": False}
    block, state, real_block = _armed_unblock(
        param, "force rollback unblock failed", permanent)
    monkeypatch.setattr(param, "blockSignals", block, raising=False)
    _arm_from(widget, monkeypatch, "_controls_v2_force_restore_carrier", state)
    _force_recovery_after_clean_write(widget, monkeypatch, shared)
    # make the NORMAL restore look unsuccessful during rollback only, so the
    # authoritative force backstop really runs
    real_readback = widget._controls_v2_carrier_readback_ok
    monkeypatch.setattr(
        widget, "_controls_v2_carrier_readback_ok",
        lambda *a, **k: (False if shared["rollback"] else real_readback(*a, **k)))

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        if permanent:
            assert param.signalsBlocked() is True
            assert CHILD in result.recovery_failed_paths
        else:
            assert param.signalsBlocked() is False
            assert CHILD not in result.recovery_failed_paths
    finally:
        real_block(param, False)


def test_repaired_cleanup_error_keeps_structured_diagnostic_evidence(widget):
    """§25.4 req 6: the TRANSIENT form is not an outstanding failure, but the
    cleanup error is NOT erased — it stays on the registry entry (and is emitted as
    a ``controls_signal_cleanup_repaired`` structured event)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_register_signal_owner)

    param = widget._controls_v2_param(MASK)
    registry = {}
    entry = _controls_v2_register_signal_owner(registry, param, CHILD)
    entry["acquired"] = True
    entry["errors"].append(RuntimeError("transient unblock failure"))
    param.blockSignals(True)          # leave it mismatched for the retry to fix
    try:
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})
    finally:
        param.blockSignals(False)

    assert failures == []                  # repaired => not outstanding
    assert entry["repaired"] is True       # ... but the evidence survives
    assert entry["mismatched"] is False
    assert entry["errors"]


def test_permanently_mismatched_owner_is_named_and_not_marked_repaired(widget):
    """§25.4 req 6-7: an owner that stays mismatched is named at its stable path."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_register_signal_owner)

    param = widget._controls_v2_param(MASK)
    registry = {}
    entry = _controls_v2_register_signal_owner(registry, param, CHILD)
    entry["acquired"] = True
    real_block = type(param).blockSignals
    real_block(param, True)
    try:
        object.__setattr__  # no-op reference so the intent is explicit
        entry["owner"] = _RefusingOwner(param)
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})
        assert failures == [CHILD]
        assert entry["mismatched"] is True
        assert entry["repaired"] is False
    finally:
        real_block(param, False)


class _RefusingOwner:
    """A signal owner whose block state can never be restored."""

    def __init__(self, real):
        self._real = real

    def blockSignals(self, _state):
        raise RuntimeError("owner refuses every restore")

    def signalsBlocked(self):
        return True


def test_registry_keeps_the_earliest_prior_state_per_owner_identity(widget):
    """§25.4 req 1 / §25.6 req 3: a later acquisition must not overwrite the
    earliest prior state, or cleanup would "restore" the owner to BLOCKED."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_acquire_signal_block, _controls_v2_release_signal_block)

    param = widget._controls_v2_param(MASK)
    registry = {}
    try:
        first = _controls_v2_acquire_signal_block(registry, param, CHILD)
        assert first["prior"] is False
        # a second acquisition happens while the owner is already blocked
        second = _controls_v2_acquire_signal_block(registry, param, CHILD)
        assert second is first
        assert second["prior"] is False        # NOT overwritten with True
        _controls_v2_release_signal_block(second)
        assert param.signalsBlocked() is False
    finally:
        type(param).blockSignals(param, False)


# ---------------------------------------------------------------------------
# §25.5 P2 / §25.10 cases 6-7 — restore-then-raise
# ---------------------------------------------------------------------------

def _restore_then_raise(owner, message):
    """A ``blockSignals`` that RESTORES the state and only then raises."""
    real = type(owner).blockSignals
    done = {"raised": False}

    def block(wanted):
        result = real(owner, wanted)
        if not wanted and not done["raised"]:
            done["raised"] = True
            raise RuntimeError(message)
        return result

    return block, real


def test_child_restore_then_raise_is_still_a_typed_write_failure(
        widget, monkeypatch):
    """§25.10 case 6.  Readback matches, so `unrestored` was empty and a good setter
    returned ok=True.  Any cleanup exception is a typed forward failure (§24.4
    req 4 / §25.5 req 2)."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = _stage_mask(widget, "/tmp/t25r4-child-restore-raise.edf")
    block, real_block = _restore_then_raise(param, "restored, then raised")
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "legacy_apply"
        assert "signal" in result.reason
        assert param.value() == prior
        # the state DID match, so this is not an outstanding recovery failure
        assert param.signalsBlocked() is False
    finally:
        real_block(param, False)


def test_root_restore_then_raise_is_typed_and_does_not_suppress_child_cleanup(
        widget, monkeypatch):
    """§25.10 case 7."""
    root = widget.wrangler.parameters
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = _stage_mask(widget, "/tmp/t25r4-root-restore-raise.edf")
    block, real_block = _restore_then_raise(root, "root restored, then raised")
    monkeypatch.setattr(root, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "legacy_apply"
        assert "signal" in result.reason
        assert param.signalsBlocked() is False   # child cleanup still happened
        assert param.value() == prior
    finally:
        real_block(root, False)


def test_setter_failure_stays_primary_over_a_cleanup_exception(
        widget, monkeypatch):
    """§25.5 req 4: the setter remains the primary reason when both fail."""
    param = widget._controls_v2_param(MASK)
    requested = "/tmp/t25r4-both-fail.edf"
    staged = _stage_mask(widget, requested)
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            raise RuntimeError("setter exploded")
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    block, real_block = _restore_then_raise(param, "restored, then raised")
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert "setter exploded" in result.reason
        assert "signal state not restored" not in result.reason
    finally:
        real_block(param, False)


# ---------------------------------------------------------------------------
# §25.6 P2 / §25.10 case 8 — acquire-then-raise
# ---------------------------------------------------------------------------

def _acquire_then_raise(owner, message, restore_fails):
    """A ``blockSignals`` that BLOCKS and then raises on acquisition."""
    real = type(owner).blockSignals
    done = {"raised": False}

    def block(wanted):
        if wanted:
            result = real(owner, wanted)       # really blocks ...
            if not done["raised"]:
                done["raised"] = True
                raise RuntimeError(message)    # ... and then raises
            return result
        if restore_fails:
            raise RuntimeError(message + " (restore)")
        return real(owner, wanted)

    return block, real


@pytest.mark.parametrize("owner_name", ["child", "root"])
def test_acquire_then_raise_leaves_the_owner_registered_and_restored(
        widget, monkeypatch, owner_name):
    """§25.10 case 8.  The prior state used to come from `blockSignals(True)`'s
    RETURN, so a block-then-raise lost it and cleanup skipped the owner as "never
    blocked".  The owner must be registered, restored, and never stranded."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    owner = param if owner_name == "child" else root
    staged = _stage_mask(widget, "/tmp/t25r4-acquire-raise.edf")
    block, real_block = _acquire_then_raise(
        owner, "acquired, then raised", restore_fails=False)
    monkeypatch.setattr(owner, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert result.phase == "legacy_apply"
        assert owner.signalsBlocked() is False      # immediate restore worked
    finally:
        real_block(owner, False)


@pytest.mark.parametrize("owner_name,path", [("child", CHILD), ("root", ROOT)])
def test_acquire_then_raise_with_failing_restore_is_named(
        widget, monkeypatch, owner_name, path):
    """§25.6 req 4: acquire-then-raise leaves the owner REGISTERED, so when the
    immediate restore also fails it is named — never silently unregistered."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    owner = param if owner_name == "child" else root
    staged = _stage_mask(widget, "/tmp/t25r4-acquire-raise-stuck.edf")
    block, real_block = _acquire_then_raise(
        owner, "acquired, then raised", restore_fails=True)
    monkeypatch.setattr(owner, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert owner.signalsBlocked() is True       # genuinely stuck
        assert path in result.recovery_failed_paths
    finally:
        real_block(owner, False)


# ---------------------------------------------------------------------------
# §25.7 P2 / §25.10 case 9 — the receipt must BE the typed receipt
# ---------------------------------------------------------------------------

def _malformed_receipts():
    return [
        pytest.param(None, id="none"),
        pytest.param(object(), id="object"),
        # an equal-SHAPED plain tuple: same arity as SourceRecoveryReceipt
        pytest.param((None, 0, None, False, True), id="equal_shaped_tuple"),
    ]


@pytest.mark.parametrize("receipt", _malformed_receipts())
def test_malformed_source_receipt_refuses_at_preflight_with_zero_writes(
        widget, monkeypatch, receipt):
    """§25.10 case 9.  A wrong-type receipt cannot establish the rollback proof, so
    it is treated EXACTLY like a capture failure: typed preflight refusal, zero
    writes, and NO coercion of a tuple into the receipt type."""
    param = widget._controls_v2_param(INCLUDE_SUBDIR)
    prior = bool(param.value())
    mask = widget._controls_v2_param(MASK)
    mask_prior = mask.value()
    staged = widget.stage_controls_transaction(
        [(INCLUDE_SUBDIR, not prior), (MASK, "/tmp/t25r4-receipt.edf")])
    assert staged.source_selection_touched
    monkeypatch.setattr(
        widget, "_controls_v2_capture_source_receipt", lambda: receipt)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == ("Source",)
    assert result.recovery_failed_paths == ()
    # ZERO writes: neither the source-selection carrier nor any other carrier
    assert bool(param.value()) == prior
    assert mask.value() == mask_prior


def test_a_real_receipt_still_proceeds(widget):
    """Positive control so the refusal above is not vacuous."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        SourceRecoveryReceipt)

    assert isinstance(
        widget._controls_v2_capture_source_receipt(), SourceRecoveryReceipt)
    param = widget._controls_v2_param(INCLUDE_SUBDIR)
    staged = widget.stage_controls_transaction(
        [(INCLUDE_SUBDIR, not bool(param.value()))])
    assert widget.commit_controls_transaction(staged).ok


def test_validator_iterates_the_admitted_container_exactly_once():
    """§25.3 req 2, supplementary LOCATION pin.

    The behavioral proof of the invariant is the changing-iterator case above; this
    pins the shape so the validate-then-convert split cannot be reintroduced.  It is
    a supplement, never a substitute (§25.8's E2 ruling): the validator must never
    re-iterate the CALLER's object — it validates the snapshot it returns."""
    import ast
    import inspect
    import textwrap

    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_validated_recovery_path)

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(_controls_v2_validated_recovery_path))).body[0]
    incoming = tree.args.args[0].arg
    reiterations = [node for node in ast.walk(tree)
                    if isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Name)
                    and node.iter.id == incoming]
    assert reiterations == [], (
        "the validator re-iterates the caller's object instead of the snapshot")
    snapshots = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "tuple"
                 and node.args
                 and isinstance(node.args[0], ast.Name)
                 and node.args[0].id == incoming]
    assert len(snapshots) == 1, (
        "expected EXACTLY one snapshot of the caller's object")


def _carrier_with_registry(widget, registry):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)

    param = widget._controls_v2_param(MASK)
    prior = param.value()
    return param, prior, _PreparedLegacyCarrier(
        MASK, param, prior, prior, prior, registry)


def test_normal_rollback_registers_its_cleanup_outcome(widget, monkeypatch):
    """§25.4 req 3: the NORMAL rollback must REGISTER its cleanup outcome instead of
    swallowing it, so final recovery can see an owner only the rollback stranded."""
    registry = {}
    param, prior, carrier = _carrier_with_registry(widget, registry)
    block, state, real_block = _armed_unblock(
        param, "rollback unblock failed", True)
    state["armed"] = True
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        widget._controls_v2_restore_carrier_value(carrier, prior)
        entry = registry[id(param)]
        assert entry["path"] == CHILD
        assert entry["errors"], "the rollback unblock failure was swallowed"
        assert entry["mismatched"] is True
        assert entry["prior"] is False
    finally:
        state["armed"] = False
        real_block(param, False)


def test_force_rollback_registers_its_cleanup_outcome(widget, monkeypatch):
    """§25.4 req 3: the FORCE-restore backstop has the same duty."""
    registry = {}
    param, prior, carrier = _carrier_with_registry(widget, registry)
    block, state, real_block = _armed_unblock(
        param, "force rollback unblock failed", True)
    state["armed"] = True
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        widget._controls_v2_force_restore_carrier(carrier, prior)
        entry = registry[id(param)]
        assert entry["path"] == CHILD
        assert entry["errors"], "the force-restore unblock failure was swallowed"
        assert entry["mismatched"] is True
    finally:
        state["armed"] = False
        real_block(param, False)
