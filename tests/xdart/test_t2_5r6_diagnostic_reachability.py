"""O-1a-T2.5R.6 (§27) — diagnostic reachability severed; error formatting total.

Codex's exact-object review of `e1ab5794` reproduced two defects:

* §27.3 **P1** — `diagnostic_copy()` returned another `_PreparedLegacyCarrier`.  Its
  public `signals` property did return detached copies, but the twin still stored the
  ONE live registry in the ordinary slot `_signals` (so `twin._signals.clear()`
  cleared real recovery authority mid-transaction) and still stored the live Qt
  `Parameter` in `param` (so `twin.param.setValue(rogue)` wrote to production outside
  the checked transaction).  Renaming an attribute closes nothing: **a leading
  underscore is not a capability boundary**, and §26.3 req 8 is reachability-based.
* §27.4 **P2** — both diagnostic projection sites used a bare `repr(error)`.  An
  exception may define a raising `__repr__`; in the detached snapshot that discarded
  the ENTIRE owner entry, and in repaired-event publication it escaped the
  classification loop so the collector falsely named a REPAIRED transient as
  `("Signal", "signal_state")` — violating the accepted §26.2/E1 rule.

Both Codex modules and the independent verifier's SHA-agnostic attack are promoted
here.  Where a probe dereferenced an attribute that §27.3 req 3 *mandates be
omitted* (`twin._signals`, `twin.param`), its INTENT is promoted in ABSENCE form,
exactly as §27.2/§27.5 step 2 direct for the analogous `signals` case.

Production-wired: real `staticWidget`, real pyqtgraph `Parameter` handles, real Qt
`blockSignals`/`signalsBlocked`, the real staging/commit engine.
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
CHILD = ("Signal", "child_signals")
ROOT = ("Signal", "root_signals")
SIGNAL_STATE = ("Signal", "signal_state")


def _find_live_registry(record, owner):
    """The FIRST attribute of *record* yielding a dict holding a LIVE entry.

    The independent verifier's SHA-agnostic attack: it names neither `signals` nor
    `_signals`, so it finds whichever spelling exists.  A live entry is identified by
    its `owner` member, which the detached projection omits by design."""
    for name in dir(record):
        if name.startswith("__"):
            continue
        try:
            value = getattr(record, name)
        except Exception:
            continue
        if not isinstance(value, dict) or id(owner) not in value:
            continue
        entry = value[id(owner)]
        if isinstance(entry, dict) and entry.get("owner") is owner:
            return name, value
    return None, None


def _one_shot_unblock(owner, message):
    """A real `blockSignals` whose FIRST restore attempt raises, leaving it blocked."""
    real = type(owner).blockSignals
    state = {"failed": False}

    def block(wanted):
        if not wanted and not state["failed"]:
            state["failed"] = True
            raise RuntimeError(message)
        return real(owner, wanted)

    return block, real


class _HostileError(RuntimeError):
    """An exception whose representation attacks the diagnostic projection."""

    def __repr__(self):
        raise RuntimeError("repr escaped")


# ---------------------------------------------------------------------------
# §27.3 P1 / §27.6 cases 1-7 — no live authority reachable from the publication
# ---------------------------------------------------------------------------

def test_no_published_slot_holds_the_registry_entry_or_parameter(widget):
    """§27.6 cases 1-3 / §27.3 req 6.  Walk EVERY slot of EVERY published record and
    assert no value is the live registry, a live entry, or the bound Parameter.

    The §26 version asserted only over a chosen "public surface"; that was the hole
    §27.3 found, because `_signals` was an ordinary slot holding the live registry."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsCarrierDiagnostic, _PreparedLegacyCarrier,
        _controls_v2_carrier_registry)

    param = widget._controls_v2_param(MASK)
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r6-slots.edf")])
    seen = {}
    real_writer = widget._controls_v2_write_legacy_carrier

    def capture(params, carrier, value):
        seen["registry"] = _controls_v2_carrier_registry(carrier)
        seen["published"] = dict(widget._controls_v2_bound_carriers)
        return real_writer(params, carrier, value)

    patch = pytest.MonkeyPatch()
    patch.setattr(widget, "_controls_v2_write_legacy_carrier", capture)
    try:
        assert widget.commit_controls_transaction(staged).ok
    finally:
        patch.undo()

    registry = seen["registry"]
    assert registry, "the live registry should hold the root and child entries"
    assert seen["published"], "nothing was published"
    for path, record in seen["published"].items():
        assert isinstance(record, _ControlsCarrierDiagnostic)
        assert not isinstance(record, _PreparedLegacyCarrier)
        assert _controls_v2_carrier_registry(record) is None
        slots = []
        for klass in type(record).__mro__:
            slots.extend(getattr(klass, "__slots__", ()) or ())
        assert slots, "the record must declare __slots__ so this walk is total"
        for name in slots:
            value = getattr(record, name, None)
            assert value is not registry, f"slot {name!r} IS the live registry"
            assert value is not param, f"slot {name!r} IS the bound Parameter"
            for entry in registry.values():
                assert value is not entry, f"slot {name!r} IS a live entry"
                assert value is not entry.get("owner"), (
                    f"slot {name!r} IS a live Qt owner")
        # §27.6 case 2: `_signals` is ABSENT, never the transaction registry
        assert not hasattr(record, "_signals")
        # §27.6 case 3: no live param/Qt owner to write through
        assert not hasattr(record, "param")
        assert _find_live_registry(record, param) == (None, None)


def test_published_record_cannot_write_through_to_the_parameter(widget):
    """§27.6 case 3 (promoted from Codex `..._cannot_write_through_after_commit`).

    That probe wrote `twin.param.setValue(rogue)` and asserted production was
    unchanged; against the §27.3-mandated record there is no `param` to dereference
    at all, so the promoted form asserts ABSENCE of the write seam — a strictly
    stronger property than "the write did not land"."""
    param = widget._controls_v2_param(MASK)
    requested = "/tmp/t25r6-write-through.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    assert widget.commit_controls_transaction(staged).ok
    assert param.value() == requested

    record = widget._controls_v2_bound_carriers[MASK]
    assert not hasattr(record, "param")
    with pytest.raises(AttributeError):
        record.param.setValue("/tmp/t25r6-rogue.edf")
    # no slot yields anything with a setValue at all
    slots = []
    for klass in type(record).__mro__:
        slots.extend(getattr(klass, "__slots__", ()) or ())
    for name in slots:
        assert not hasattr(getattr(record, name, None), "setValue")
    assert param.value() == requested          # production untouched


def test_setter_cannot_clear_registry_through_the_published_record(
        widget, monkeypatch):
    """§27.6 cases 4-5 (promoted from Codex `..._does_not_retain_raw_registry_authority`
    and the verifier's SHA-agnostic clear).

    A real setter hunts for the live registry by ANY attribute name, clears whatever
    it finds, then a ONE-SHOT child unblock fails.  At `e1ab5794` this cleared the
    real registry and left the child blocked and unnamed."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    requested = "/tmp/t25r6-registry-erase.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    seen = {"attr": None, "cleared": False}
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            record = widget._controls_v2_bound_carriers[MASK]
            name, live = _find_live_registry(record, param)
            seen["attr"] = name
            if live is not None:
                live.clear()
                seen["cleared"] = True
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    block, real_block = _one_shot_unblock(param, "one-shot child unblock failure")
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert seen["attr"] is None, (
            f"the live registry was reachable via {seen['attr']!r}")
        assert seen["cleared"] is False
        assert not result.ok                    # the one-shot cleanup still failed
        assert param.value() == prior           # carrier rolled back
        # earliest prior survived => repaired transient, unblocked and unnamed
        assert (param.signalsBlocked(),
                CHILD in result.recovery_failed_paths) == (False, False)
    finally:
        real_block(param, False)


def test_root_scenario_retains_independent_child_cleanup(widget, monkeypatch):
    """§27.6 case 6.  The same attack with a failing ROOT cleanup must not suppress
    the child's independent cleanup."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    prior = param.value()
    requested = "/tmp/t25r6-root-erase.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    seen = {"attr": None}
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            record = widget._controls_v2_bound_carriers[MASK]
            name, live = _find_live_registry(record, root)
            seen["attr"] = name
            if live is not None:
                live.clear()
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    block, real_block = _one_shot_unblock(root, "one-shot root unblock failure")
    monkeypatch.setattr(root, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert seen["attr"] is None
        assert not result.ok
        assert param.value() == prior
        assert (root.signalsBlocked(),
                ROOT in result.recovery_failed_paths) == (False, False)
        assert param.signalsBlocked() is False       # child cleanup independent
    finally:
        real_block(root, False)


@pytest.mark.parametrize("owner_name", ["child", "root"])
def test_forging_a_prior_by_any_attribute_name_cannot_leak(
        widget, monkeypatch, owner_name):
    """§27.6 cases 4/7 (promoted from the verifier's SHA-agnostic forge).

    At `e1ab5794` forging `prior=True` through `_signals` made cleanup BLOCK the
    owner, verify against the forged value, and return `ok=True` — a leak reported as
    success, for both owners."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    owner = param if owner_name == "child" else root
    requested = f"/tmp/t25r6-forge-{owner_name}.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    seen = {"attr": None}
    real_set = type(param).setValue
    real_block = type(owner).blockSignals

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            record = widget._controls_v2_bound_carriers[MASK]
            name, live = _find_live_registry(record, owner)
            seen["attr"] = name
            if live is not None:
                live[id(owner)]["prior"] = True
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    try:
        result = widget.commit_controls_transaction(staged)

        assert seen["attr"] is None
        assert result.ok
        assert owner.signalsBlocked() is False
        assert param.value() == requested
    finally:
        real_block(owner, False)


def test_replacing_the_published_mapping_still_redirects_nothing(
        widget, monkeypatch):
    """§27.6 case 7.  Swapping the whole published mapping — now for value records —
    must not redirect a forward write or a recovery target."""
    from types import MappingProxyType

    from pyqtgraph.parametertree import Parameter
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsCarrierDiagnostic)

    param = widget._controls_v2_param(MASK)
    requested = "/tmp/t25r6-map-swap.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    replacement = Parameter.create(
        name="replacement-mask", type="str", value="/tmp/replacement-prior.edf")
    replacement_prior = replacement.value()
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            forged = {MASK: _ControlsCarrierDiagnostic(
                MASK, value, replacement_prior, value)}
            object.__setattr__(
                widget, "_controls_v2_bound_carriers", MappingProxyType(forged))
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    result = widget.commit_controls_transaction(staged)

    assert result.ok
    assert param.value() == requested
    assert replacement.value() == replacement_prior
    assert param.signalsBlocked() is False


# ---------------------------------------------------------------------------
# §27.4 P2 / §27.6 cases 8-10 — diagnostic formatting is per-error and total
# ---------------------------------------------------------------------------

def test_hostile_error_repr_retains_the_owner_entry_and_stable_path():
    """§27.6 cases 8-9 (promoted from Codex `..._snapshot_is_total_for_hostile_error_repr`).

    One raising `__repr__` used to discard the whole owner entry — path, prior,
    mismatch and repaired flags included.  The entry must survive with a nonempty
    SAFE error string."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_detached_signal_snapshot)

    registry = {
        1: {
            "path": CHILD,
            "errors": [_HostileError("cleanup failed")],
            "prior": False, "probed": True, "acquired": True,
            "mismatched": False, "repaired": True,
        }
    }

    snapshot = _controls_v2_detached_signal_snapshot(registry)

    assert 1 in snapshot                            # the entry was NOT discarded
    assert snapshot[1]["path"] == CHILD             # stable path retained
    assert snapshot[1]["repaired"] is True          # flags retained
    assert snapshot[1]["prior"] is False
    assert snapshot[1]["errors"]                    # nonempty ...
    text = snapshot[1]["errors"][0]
    assert type(text) is str and text.strip()       # ... and a safe exact string
    assert "_HostileError" in text                  # the SAFE type name


def test_error_formatter_prefers_repr_and_falls_back_safely():
    """§27.4 req 2-3: `repr` preferred; on failure a stable exact string built ONLY
    from the safe built-in type name, never re-invoking hostile formatting."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_safe_error_text)

    ordinary = RuntimeError("plain")
    assert _controls_v2_safe_error_text(ordinary) == repr(ordinary)

    text = _controls_v2_safe_error_text(_HostileError("hostile"))
    assert type(text) is str and text.strip()
    assert "_HostileError" in text

    class Nameless:
        """Even a hostile type name must not defeat the formatter."""

        def __repr__(self):
            raise RuntimeError("repr escaped")

    assert _controls_v2_safe_error_text(Nameless()).strip()
    assert _controls_v2_safe_error_text(None) == "None"


def test_snapshot_survives_a_hostile_error_container():
    """§27.4 req 4: no projection failure discards the owner entry."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_detached_signal_snapshot)

    class HostileList:
        def __iter__(self):
            raise RuntimeError("iteration escaped")

        def __bool__(self):
            return True

    registry = {7: {"path": ROOT, "errors": HostileList(), "prior": True,
                    "probed": True, "acquired": True, "mismatched": True,
                    "repaired": False}}

    snapshot = _controls_v2_detached_signal_snapshot(registry)

    assert 7 in snapshot
    assert snapshot[7]["path"] == ROOT
    assert snapshot[7]["prior"] is True
    assert snapshot[7]["errors"] == []


def test_repaired_transient_with_hostile_repr_stays_unnamed_in_recovery(
        widget, monkeypatch):
    """§27.6 case 10 (promoted from both Codex hostile-repr probes).

    The owner is repaired, so E1 says it is NOT outstanding.  At `e1ab5794` the
    repaired-event publication's raw `repr()` raised out of the classification loop
    and the collector named `("Signal", "signal_state")` instead."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_register_signal_owner)

    param = widget._controls_v2_param(MASK)
    registry = {}
    entry = _controls_v2_register_signal_owner(registry, param, CHILD)
    entry["acquired"] = True
    real_block = type(param).blockSignals
    real_block(param, True)
    raised = {"done": False}

    def restore_then_raise(wanted):
        result = real_block(param, wanted)
        if not wanted and not raised["done"]:
            raised["done"] = True
            raise _HostileError("restored then raised")
        return result

    monkeypatch.setattr(param, "blockSignals", restore_then_raise, raising=False)
    try:
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})

        assert failures == []                     # repaired => not outstanding
        assert SIGNAL_STATE not in failures
        assert param.signalsBlocked() is False
        assert entry["repaired"] is True
        assert entry["mismatched"] is False
        assert entry["errors"]                    # the raw exception is retained
    finally:
        real_block(param, False)


def test_transaction_does_not_name_a_repaired_transient_when_repr_raises(
        widget, monkeypatch):
    """§27.6 case 10, through the real action boundary."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r6-hostile.edf")])
    real_block = type(param).blockSignals
    raised = {"done": False}

    def first_release_fails(wanted):
        if not wanted and not raised["done"]:
            raised["done"] = True
            raise _HostileError("transient release failure")
        return real_block(param, wanted)

    monkeypatch.setattr(param, "blockSignals", first_release_fails, raising=False)
    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert param.value() == prior
        assert param.signalsBlocked() is False
        assert SIGNAL_STATE not in result.recovery_failed_paths
        assert CHILD not in result.recovery_failed_paths
    finally:
        real_block(param, False)


def test_published_projection_is_total_for_hostile_errors(widget, monkeypatch):
    """§27.4 req 5/7: the published record's projection uses the SAME formatter, so a
    hostile cleanup error cannot make the diagnostic view raise for a reader."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsCarrierDiagnostic)

    registry = {
        3: {"path": CHILD, "errors": [_HostileError("boom")], "prior": False,
            "probed": True, "acquired": True, "mismatched": False,
            "repaired": True},
    }
    record = _ControlsCarrierDiagnostic(MASK, "v", "p", "e", registry)

    projection = record.signals
    assert 3 in projection
    assert projection[3]["path"] == CHILD
    assert projection[3]["errors"] and type(projection[3]["errors"][0]) is str
    # ... and it is a point-in-time value, never a live view
    registry[3]["repaired"] = False
    assert record.signals[3]["repaired"] is True
    assert "owner" not in projection[3]


# ---------------------------------------------------------------------------
# §27.6 case 11 — the accepted T-2.5R.5 retry-evidence behavior is unchanged
# ---------------------------------------------------------------------------

def test_retry_helper_still_records_its_exception_in_order(widget, monkeypatch):
    """§27.6 case 11 / mutation row 8: the accepted §26.4 retry helper stays pinned."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_register_signal_owner, _controls_v2_release_signal_block)

    param = widget._controls_v2_param(MASK)
    registry = {}
    entry = _controls_v2_register_signal_owner(registry, param, CHILD)
    entry["acquired"] = True
    real_block = type(param).blockSignals
    real_block(param, True)
    calls = {"n": 0}

    def block(wanted):
        if not wanted:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first: release failure")
            if calls["n"] == 2:
                real_block(param, wanted)
                raise RuntimeError("second: retry restored, then raised")
        return real_block(param, wanted)

    monkeypatch.setattr(param, "blockSignals", block, raising=False)
    try:
        _controls_v2_release_signal_block(entry)
        assert len(entry["errors"]) == 1
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})

        assert failures == []
        assert len(entry["errors"]) == 2          # ordered, neither replaced
        assert "first: release failure" in repr(entry["errors"][0])
        assert "second: retry restored" in repr(entry["errors"][1])
        assert entry["repaired"] is True
    finally:
        real_block(param, False)
