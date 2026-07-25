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
SIGNAL_STATE = ("Signal", "signal_state")


class _HostileError(RuntimeError):
    """An exception whose representation attacks the diagnostic projection."""

    def __repr__(self):
        raise RuntimeError("repr escaped")


# ---------------------------------------------------------------------------
# §27.4 P2 / §27.6 cases 8-10 — diagnostic formatting is per-error and total
# ---------------------------------------------------------------------------


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
