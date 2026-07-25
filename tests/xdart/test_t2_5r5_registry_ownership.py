"""O-1a-T2.5R.5 (§26) — recovery authority is hidden; retry evidence is retained.

Codex's exact-object review of `1266ef57` reproduced two defects:

* §26.3 **P1** — the transaction-local signal registry was a plain mutable dict, and
  each LIVE `_PreparedLegacyCarrier` stored that same dict in its PUBLIC `signals`
  attribute while the live carriers themselves were published through
  `_controls_v2_bound_carriers`.  `MappingProxyType` froze only the outer
  path->carrier mapping, so a real setter could reach a live carrier and call
  `carrier.signals.clear()` — erasing recovery authority after the earliest prior
  state was recorded.  Rollback then re-registered the already-blocked child with
  `prior=True`, and final recovery faithfully "restored" that forged value and
  certified a leaked Qt block.
* §26.4 **P2** — the final-retry backstop was the bare lambda
  `entry["owner"].blockSignals(entry["prior"])`.  The member driver caught its
  exception but never recorded it, so a retry that restored the state and THEN
  raised left `errors == []`, `repaired == False`, and no
  `controls_signal_cleanup_repaired` event — contradicting the repaired-transient
  contract that §26.2 confirmed.

Both Codex reproducers under `/Users/vthampy/repos/tmp/` are promoted here as
committed coverage, with the root and ordering variants §26.6 requires added.

Production-wired: real `staticWidget`, real pyqtgraph `Parameter` handles, real Qt
`blockSignals`/`signalsBlocked`, the real staging/commit engine.
"""

from __future__ import annotations

from types import MappingProxyType

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


@pytest.fixture
def events(monkeypatch):
    """Capture the structured `run_config_debug` events my code emits.

    `run_config_debug_log` returns early unless tracing is enabled, so the module
    symbol my code calls is the seam to observe."""
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    captured = []
    real = module.run_config_debug_log

    def capture(logger, event, **fields):
        captured.append((event, fields))
        return real(logger, event, **fields)

    monkeypatch.setattr(module, "run_config_debug_log", capture)
    return captured


def _find_live_registry(record, owner):
    """The FIRST attribute of *record* yielding a dict that holds a LIVE entry.

    §27.3 req 6 / §27.5: the invariant is REACHABILITY, not attribute naming — "a
    leading underscore is not a capability boundary".  This walks every attribute
    (the independent verifier's SHA-agnostic pattern), so it finds `signals` at
    `1266ef57` and `_signals` at `e1ab5794` without naming either.  A live entry is
    identified by its `owner` member, which the detached projection omits."""
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


def _registered_entry(widget, owner, path, *, blocked=True):
    """A registry + entry for *owner*, left blocked so a retry has work to do."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_register_signal_owner)

    registry = {}
    entry = _controls_v2_register_signal_owner(registry, owner, path)
    entry["acquired"] = True
    if blocked:
        type(owner).blockSignals(owner, True)
    return registry, entry


# ---------------------------------------------------------------------------
# §26.3 P1 / §26.6 cases 1-4 — the published diagnostic cannot reach authority
# ---------------------------------------------------------------------------

def test_published_map_exposes_no_live_registry_object(widget):
    """§26.3 req 8 — THE ARCHITECTURE ASSERTION.

    Nothing reachable through the public surface of `_controls_v2_bound_carriers`
    may be the live registry, a live entry, or the live execution carrier, and the
    internal registry accessor must refuse every published twin."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsCarrierDiagnostic, _PreparedLegacyCarrier,
        _controls_v2_carrier_registry)

    staged = widget.stage_controls_transaction([(MASK, "/tmp/t25r5-arch.edf")])
    captured = {}
    real_set = type(widget._controls_v2_param(MASK)).setValue
    param = widget._controls_v2_param(MASK)
    real_writer = widget._controls_v2_write_legacy_carrier

    def capture_writer(params, carrier, value):
        # the LIVE execution carrier is the only holder of the one registry
        captured["live_registry"] = _controls_v2_carrier_registry(carrier)
        return real_writer(params, carrier, value)

    def setter(self, value, *args, **kwargs):
        if self is param and "t25r5-arch" in str(value):
            published = widget._controls_v2_bound_carriers
            captured["published"] = published
            captured["twin"] = published[MASK]
            captured["live_param"] = param
            captured["snapshot_a"] = published[MASK].signals
            captured["snapshot_b"] = published[MASK].signals
        return real_set(self, value, *args, **kwargs)

    monkeypatch_setattr = pytest.MonkeyPatch()
    monkeypatch_setattr.setattr(type(param), "setValue", setter)
    monkeypatch_setattr.setattr(
        widget, "_controls_v2_write_legacy_carrier", capture_writer)
    try:
        result = widget.commit_controls_transaction(staged)
    finally:
        monkeypatch_setattr.undo()

    assert result.ok
    assert isinstance(captured["published"], MappingProxyType)
    record = captured["twin"]
    live_registry = captured["live_registry"]
    live_param = captured["live_param"]

    # §27.6 case 1/3 + §27.3 req 6: walk EVERY slot of EVERY published record and
    # assert no value is the live registry, a live entry, or the bound Parameter.
    # This replaces the §26 "public surface" assertion, which was the hole.
    assert live_registry, "the transaction registry should have been populated"
    for path, published in captured["published"].items():
        assert isinstance(published, _ControlsCarrierDiagnostic)
        assert not isinstance(published, _PreparedLegacyCarrier)
        assert _controls_v2_carrier_registry(published) is None
        slots = []
        for klass in type(published).__mro__:
            slots.extend(getattr(klass, "__slots__", ()) or ())
        assert slots, "the record must declare __slots__ so this walk is total"
        for name in slots:
            value = getattr(published, name, None)
            assert value is not live_registry, (
                f"published record slot {name!r} IS the live registry")
            assert value is not live_param, (
                f"published record slot {name!r} IS the bound Parameter")
            for entry in live_registry.values():
                assert value is not entry, (
                    f"published record slot {name!r} IS a live registry entry")
        # no attribute under ANY spelling yields a live entry
        assert _find_live_registry(published, live_param) == (None, None)
        # and there is no parameter/Qt owner to write through
        assert not hasattr(published, "param")
        assert not hasattr(published, "_signals")

    # each read of the inert projection is a FRESH detached object
    assert captured["snapshot_a"] == captured["snapshot_b"] == {}
    assert captured["snapshot_a"] is not captured["snapshot_b"]
    # the mapping itself still refuses item assignment (§22.10.A.3 retained)
    with pytest.raises(TypeError):
        captured["published"][MASK] = record


def test_setter_clearing_the_published_signals_cannot_erase_authority(
        widget, monkeypatch, events):
    """§26.6 cases 1-2 (promoted from Codex `test_setter_cannot_erase_...`).

    A setter clears `carrier.signals` after root/child registration but before
    forward cleanup, then a ONE-SHOT child unblock fails.  The live registry must
    survive, so the child ends unblocked, unnamed, and represented as repaired
    diagnostic evidence."""
    param = widget._controls_v2_param(MASK)
    prior = param.value()
    requested = "/tmp/t25r5-registry-erase.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    seen = {}
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            # §27.2 RESHAPE: the original form asserted live entries were VISIBLE
            # through the published view.  That pinned the exploit path's
            # availability, not a product contract — §27.2 ruled no production
            # consumer reads `_controls_v2_bound_carriers[*].signals`.  The
            # assertion is now ABSENCE of authority, by any attribute spelling.
            record = widget._controls_v2_bound_carriers[MASK]
            name, live = _find_live_registry(record, param)
            seen["live_attr"] = name
            seen["visible"] = len(record.signals)
            record.signals.clear()               # impotent: a throwaway copy
            seen["after_clear"] = len(record.signals)
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    block, real_block = _one_shot_unblock(param, "one-shot child unblock failure")
    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert param.value() == prior
        # §27.6 case 1/4: NO attribute of the published record yields a live entry,
        # and the projection it does expose is inert (empty, and clearable without
        # effect) — the authority is absent, not merely copy-protected.
        assert seen["live_attr"] is None
        assert seen["visible"] == 0
        assert seen["after_clear"] == 0
        # the earliest prior survived, so the child is repaired and NOT named
        assert (param.signalsBlocked(),
                CHILD in result.recovery_failed_paths) == (False, False)
        assert any(event == "controls_signal_cleanup_repaired"
                   for event, _fields in events)
    finally:
        real_block(param, False)


def test_setter_clearing_published_signals_cannot_lose_the_root_owner(
        widget, monkeypatch):
    """§26.6 case 3 (promoted).  Rollback touches only children, so once the
    registry was erased a failed ROOT cleanup had no chance to re-register.  Child
    cleanup must not be suppressed either."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    prior = param.value()
    requested = "/tmp/t25r5-root-registry-erase.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            widget._controls_v2_bound_carriers[MASK].signals.clear()
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    block, real_block = _one_shot_unblock(root, "one-shot root unblock failure")
    monkeypatch.setattr(root, "blockSignals", block, raising=False)

    try:
        result = widget.commit_controls_transaction(staged)

        assert not result.ok
        assert param.value() == prior
        assert (root.signalsBlocked(),
                ROOT in result.recovery_failed_paths) == (False, False)
        assert param.signalsBlocked() is False      # child cleanup still ran
    finally:
        real_block(root, False)


@pytest.mark.parametrize("owner_name", ["child", "root"])
def test_rewriting_a_published_prior_cannot_forge_a_successful_commit(
        widget, monkeypatch, owner_name):
    """§26.6 case 1 (promoted from Codex `..._rewrite_registered_prior_...`).

    Rewriting the published entry's `prior` to `True` used to make cleanup BLOCK the
    owner, verify against the forged value, and return `ok=True` — a leak reported as
    success.  The rewrite must now mutate a throwaway only."""
    param = widget._controls_v2_param(MASK)
    root = widget.wrangler.parameters
    owner = param if owner_name == "child" else root
    requested = f"/tmp/t25r5-{owner_name}-prior-rewrite.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    real_set = type(param).setValue
    real_block = type(owner).blockSignals
    seen = {}

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            # §27.2 RESHAPE: the original indexed the live entry to forge its
            # `prior`.  There is now no entry to index under ANY attribute name, so
            # the assertion is absence; the attempt itself must stay harmless.
            record = widget._controls_v2_bound_carriers[MASK]
            name, live = _find_live_registry(record, owner)
            seen["live_attr"] = name
            projection = record.signals
            seen["entry_present"] = id(owner) in projection
            projection[id(owner)] = {"prior": True}   # throwaway
            seen["still_absent"] = id(owner) not in record.signals
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    try:
        result = widget.commit_controls_transaction(staged)

        assert seen["live_attr"] is None
        assert seen["entry_present"] is False
        assert seen["still_absent"] is True      # the injection did not stick
        assert result.ok
        assert owner.signalsBlocked() is False
    finally:
        real_block(owner, False)


def test_replacing_the_whole_published_mapping_redirects_nothing(
        widget, monkeypatch):
    """§26.6 case 4.  Swapping the entire published mapping — carriers and all —
    must not redirect a forward write or a recovery target."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier)
    from pyqtgraph.parametertree import Parameter

    param = widget._controls_v2_param(MASK)
    requested = "/tmp/t25r5-map-swap.edf"
    staged = widget.stage_controls_transaction([(MASK, requested)])
    replacement = Parameter.create(
        name="replacement-mask", type="str", value="/tmp/replacement-prior.edf")
    replacement_prior = replacement.value()
    real_set = type(param).setValue

    def setter(self, value, *args, **kwargs):
        if self is param and value == requested:
            forged = {MASK: _PreparedLegacyCarrier(
                MASK, replacement, value, replacement_prior, value)}
            object.__setattr__(
                widget, "_controls_v2_bound_carriers", MappingProxyType(forged))
        return real_set(self, value, *args, **kwargs)

    monkeypatch.setattr(type(param), "setValue", setter)
    result = widget.commit_controls_transaction(staged)

    assert result.ok
    assert param.value() == requested                  # the real target was written
    assert replacement.value() == replacement_prior    # never touched
    assert param.signalsBlocked() is False


def test_diagnostic_copy_is_a_value_record_not_an_execution_carrier(widget):
    """§27.3 req 2-3 (RESHAPED from the §26 twin form): what `diagnostic_copy()`
    returns is a VALUE-ONLY record of a different type entirely — it cannot reach the
    live registry and cannot be mistaken for an execution carrier."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _ControlsCarrierDiagnostic, _PreparedLegacyCarrier,
        _controls_v2_carrier_registry)

    param = widget._controls_v2_param(MASK)
    registry = {}
    live = _PreparedLegacyCarrier(MASK, param, "a", "b", "c", registry)
    record = live.diagnostic_copy()

    assert isinstance(record, _ControlsCarrierDiagnostic)
    assert not isinstance(record, _PreparedLegacyCarrier)
    # the execution carrier still owns the ONE registry ...
    assert _controls_v2_carrier_registry(live) is registry
    # ... and the value record is refused by the accessor and holds no param
    assert _controls_v2_carrier_registry(record) is None
    assert not hasattr(record, "param")
    assert record.signals == {}
    # values round-trip, detached and fresh per read
    assert (record.path, record.value, record.prior, record.expected) == (
        MASK, "a", "b", "c")
    with pytest.raises(AttributeError):
        record.signals = {}


def test_five_positional_carrier_construction_still_works(widget):
    """§27.3 req 5 / the 8-case depth oracle constructs carriers 5-positionally."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _PreparedLegacyCarrier, _controls_v2_carrier_registry)

    carrier = _PreparedLegacyCarrier(MASK, widget._controls_v2_param(MASK),
                                     "v", "p", "e")
    assert _controls_v2_carrier_registry(carrier) is None
    assert (carrier.value, carrier.prior, carrier.expected) == ("v", "p", "e")


# ---------------------------------------------------------------------------
# §26.4 P2 / §26.6 cases 5-8 — the final retry keeps its evidence
# ---------------------------------------------------------------------------

def _retry_restore_then_raise(owner, message):
    """A `blockSignals` that RESTORES and only then raises, once."""
    real = type(owner).blockSignals
    state = {"raised": False}

    def block(wanted):
        result = real(owner, wanted)
        if not wanted and not state["raised"]:
            state["raised"] = True
            raise RuntimeError(message)
        return result

    return block, real


@pytest.mark.parametrize("owner_name,path", [("child", CHILD), ("root", ROOT)])
def test_final_retry_restore_then_raise_retains_evidence_and_is_not_outstanding(
        widget, monkeypatch, events, owner_name, path):
    """§26.6 cases 5-6 (promoted from both Codex retry probes, plus the root form).

    The retry repairs the state and THEN raises.  The state repair is correct, so the
    owner is NOT outstanding — but the exception must be retained on the entry, the
    entry marked repaired, and the structured event emitted."""
    owner = (widget._controls_v2_param(MASK) if owner_name == "child"
             else widget.wrangler.parameters)
    registry, entry = _registered_entry(widget, owner, path)
    block, real_block = _retry_restore_then_raise(
        owner, "final retry restored, then raised")
    monkeypatch.setattr(owner, "blockSignals", block, raising=False)

    try:
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})

        assert failures == []
        assert owner.signalsBlocked() is False
        assert entry["mismatched"] is False
        assert entry["repaired"] is True
        assert any("final retry restored, then raised" in repr(error)
                   for error in entry["errors"])
        repaired = [fields for event, fields in events
                    if event == "controls_signal_cleanup_repaired"]
        assert repaired, "the structured repaired event was not emitted"
        assert repaired[-1]["signal_owner"] == list(path)
        assert repaired[-1]["cleanup_errors"]
    finally:
        real_block(owner, False)


def test_final_retry_that_raises_before_restoring_is_still_named(
        widget, monkeypatch, events):
    """§26.6 case 7.  A retry that raises WITHOUT repairing leaves the owner
    mismatched, so it is named and must NOT be reported as repaired."""
    param = widget._controls_v2_param(MASK)
    registry, entry = _registered_entry(widget, param, CHILD)
    real_block = type(param).blockSignals

    def block(wanted):
        if not wanted:
            raise RuntimeError("retry raised before restoring")
        return real_block(param, wanted)

    monkeypatch.setattr(param, "blockSignals", block, raising=False)

    try:
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})

        assert failures == [CHILD]
        assert entry["mismatched"] is True
        assert entry["repaired"] is False
        assert any("retry raised before restoring" in repr(error)
                   for error in entry["errors"])
        assert not any(event == "controls_signal_cleanup_repaired"
                       for event, _fields in events)
    finally:
        real_block(param, False)


def test_multiple_cleanup_exceptions_stay_ordered_on_the_same_entry(
        widget, monkeypatch):
    """§26.6 case 8 / §26.4 req 6.  An earlier release failure and a later retry
    failure must BOTH survive, in order — the retry never replaces the first."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_release_signal_block)

    param = widget._controls_v2_param(MASK)
    registry, entry = _registered_entry(widget, param, CHILD)
    real_block = type(param).blockSignals
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
        _controls_v2_release_signal_block(entry)      # records the FIRST
        assert len(entry["errors"]) == 1
        failures = widget._controls_v2_restore_signal_state_verified(
            {"signal_registry": registry})           # records the SECOND

        assert failures == []
        assert len(entry["errors"]) == 2
        assert "first: release failure" in repr(entry["errors"][0])
        assert "second: retry restored" in repr(entry["errors"][1])
        assert entry["repaired"] is True
    finally:
        real_block(param, False)


def test_post_acquisition_restore_failure_is_captured_not_only_logged(widget):
    """§26.4 closing paragraph: the immediate restore attempted after an
    acquire-then-raise must CAPTURE its exception into the ordered list too."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        _controls_v2_acquire_signal_block)

    registry = {}

    class Stuck:
        """Starts unblocked; acquisition BLOCKS and then raises; every restore
        refuses — so the owner is genuinely stranded and BOTH cleanup exceptions
        must survive, in order."""

        def __init__(self):
            self.blocked = False

        def signalsBlocked(self):
            return self.blocked

        def blockSignals(self, wanted):
            if wanted:
                self.blocked = True                  # the side effect lands ...
                raise RuntimeError("acquire raised")  # ... and then it raises
            raise RuntimeError("immediate restore raised")

    owner = Stuck()
    with pytest.raises(RuntimeError):
        _controls_v2_acquire_signal_block(registry, owner, CHILD)

    entry = registry[id(owner)]
    assert entry["prior"] is False            # sampled BEFORE acquisition
    assert len(entry["errors"]) == 2          # ordered, neither discarded
    assert "acquire raised" in repr(entry["errors"][0])
    assert "immediate restore raised" in repr(entry["errors"][1])
    assert entry["mismatched"] is True


def test_repaired_transient_is_never_an_outstanding_failure(widget, monkeypatch):
    """§26.2 ruling, retained: repaired-transient semantics are UNCHANGED — a
    cleanup error that final recovery repairs is evidence, never an outstanding
    `recovery_failed_path`."""
    param = widget._controls_v2_param(MASK)
    registry, entry = _registered_entry(widget, param, CHILD)
    entry["errors"].append(RuntimeError("earlier transient"))

    failures = widget._controls_v2_restore_signal_state_verified(
        {"signal_registry": registry})

    assert failures == []
    assert entry["repaired"] is True
    assert entry["mismatched"] is False
    assert param.signalsBlocked() is False
