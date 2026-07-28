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
# §27.3 req 5 — the local execution carrier's construction contract
# ---------------------------------------------------------------------------

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
