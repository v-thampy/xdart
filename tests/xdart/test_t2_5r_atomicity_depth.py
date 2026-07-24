"""O-1a-T2.5R (§18 atomicity-depth) — bind carrier handles + verify recovery.

The nine required red-before/green-after behaviors for the checked commit engine
that the T-2.5 review (§18.4-18.7) exposed:

* the forward readback reads the BOUND ORIGINAL handle, never a path-re-resolved
  replacement (§18.4 req 2);
* a handle replacement detected BEFORE any write fails without touching the
  replacement or the rollback stack; a replacement detected AFTER a write
  restores the bound original and leaves the replacement untouched (§18.4 req 3);
* a silent no-op display / intent / source-cache restore is a NAMED recovery
  failure, not trusted because the helper returned (§18.5 req 4-6);
* an early legacy failure recovers ONLY the touched classes (§18.5 req 7);
* a source recovery reconciles the PRIOR selection, not the new one twice
  (§18.6 req 8);
* both the Run and non-run structured refusal events carry the exact ``reason``
  and the COMPLETE ordered ``recovery_failed_paths`` (§18.7 req 9).

Production-wired: a real ``staticWidget`` with its real wrangler parameter tree
and shared scan; the fixture mirrors the §14 oracle (no profile refresh) so
``widget.scan`` and the parameters are the production objects.
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
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


# --- req 2: readback reads the bound handle, not a path-resolved replacement -- #
def test_readback_reads_bound_handle_not_path_replacement(widget, monkeypatch):
    path = ("Signal", "mask_file")
    real = widget._controls_v2_param(path)
    prior = real.value()
    expected = "/tmp/t25r-bound.edf"
    staged = widget.stage_controls_transaction([(path, expected)])
    original = widget._controls_v2_param
    calls = {"n": 0}

    class Replacement:
        def value(self):
            return expected  # would falsely satisfy a path-resolved readback

    replacement = Replacement()

    def switching(candidate):
        candidate = tuple(candidate)
        if candidate != path:
            return original(candidate)
        calls["n"] += 1
        # preflight + the two identity checks see the stable carrier; only a
        # (buggy) path-resolved readback would see the replacement.
        return real if calls["n"] <= 3 else replacement

    monkeypatch.setattr(widget, "_controls_v2_param", switching)
    monkeypatch.setattr(
        widget, "_mirror_wrangler_parameter_values", lambda *_a, **_k: None)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert real.value() == prior  # the bound original was never actually changed


# --- req 3a: replacement BEFORE write is untouched, not on the rollback stack - #
def test_replacement_before_write_is_untouched(widget, monkeypatch):
    path = ("Signal", "mask_file")
    real = widget._controls_v2_param(path)
    prior = real.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25r-before.edf")])
    original = widget._controls_v2_param
    calls = {"n": 0}
    replacement_writes = []

    class Replacement:
        def value(self):
            return "/tmp/should-not-be-read"

        def blockSignals(self, _v):
            return False

        def setValue(self, value):
            replacement_writes.append(value)

    replacement = Replacement()

    def switching(candidate):
        candidate = tuple(candidate)
        if candidate != path:
            return original(candidate)
        calls["n"] += 1
        # preflight sees the real handle; the pre-apply identity check sees a
        # replacement -> refuse BEFORE any write.
        return real if calls["n"] == 1 else replacement

    monkeypatch.setattr(widget, "_controls_v2_param", switching)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.reason == "legacy carrier handle replaced before apply"
    assert real.value() == prior            # original never written
    assert replacement_writes == []         # replacement never touched
    assert result.recovery_failed_paths == ()  # nothing was on the rollback stack


# --- req 3b: replacement AFTER write restores the original only -------------- #
def test_replacement_after_write_restores_original_only(widget, monkeypatch):
    path = ("Signal", "mask_file")
    real = widget._controls_v2_param(path)
    prior = real.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25r-after.edf")])
    original = widget._controls_v2_param
    calls = {"n": 0}
    replacement_writes = []

    class Replacement:
        def value(self):
            return "/tmp/t25r-after.edf"  # would falsely satisfy readback

        def blockSignals(self, _v):
            return False

        def setValue(self, value):
            replacement_writes.append(value)

    replacement = Replacement()

    def switching(candidate):
        candidate = tuple(candidate)
        if candidate != path:
            return original(candidate)
        calls["n"] += 1
        # preflight + the pre-apply identity check see the real handle; the write
        # lands on the real handle (via the mirror by path); the POST-write
        # identity check and everything after see the replacement.
        return real if calls["n"] <= 2 else replacement

    monkeypatch.setattr(widget, "_controls_v2_param", switching)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.reason == "legacy carrier handle replaced after write"
    assert real.value() == prior        # the bound ORIGINAL handle was restored
    assert replacement_writes == []     # the replacement was never written


# --- req 4: a silent no-op display restore is a named recovery failure ------- #
def test_silent_display_restore_noop_is_named(widget, monkeypatch):
    path = ("Signal", "mask_file")
    staged = widget.stage_controls_transaction([(path, "/tmp/t25r-display.edf")])
    before = copy.deepcopy(widget.scan.bai_1d_args)

    def partial_then_raise(_snapshot):
        widget.scan.bai_1d_args["numpoints"] = 987_654
        raise RuntimeError("injected display install failure")

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan", partial_then_raise)
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan", lambda *_a, **_k: None)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "install"
    assert ("Display",) in result.recovery_failed_paths
    assert widget.scan.bai_1d_args == before  # backstop still left clean state


# --- req 5: a silent no-op intent restore is a named recovery failure -------- #
def test_silent_intent_restore_noop_is_named(widget, monkeypatch):
    path = ("Int1D", "points")
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 222
    widget.scan.bai_1d_args["numpoints"] = 222
    before = copy.deepcopy(intent.bai_1d_args)
    staged = widget.stage_controls_transaction([(path, "500")])

    def raise_after_intent(_snapshot):
        raise RuntimeError("injected display install failure")

    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan", raise_after_intent)
    monkeypatch.setattr(
        widget, "_controls_v2_restore_intent_values", lambda *_a, **_k: None)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert ("Intent",) in result.recovery_failed_paths
    assert intent.bai_1d_args == before  # backstop still left clean state


# --- req 6: a silent no-op source-cache restore is a named recovery failure -- #
def test_silent_source_cache_restore_noop_is_named(widget, monkeypatch):
    path = ("Signal", "include_subdir")
    current = bool(widget._controls_v2_param(path).value())
    prior_energy = ("orig-source", 7.77)
    prior_probe = ("orig-source", "probe")
    widget._controls_v2_source_energy_cache = prior_energy
    widget._controls_v2_metadata_probe_cache = prior_probe
    staged = widget.stage_controls_transaction([(path, not current)])

    def raise_sync():
        raise RuntimeError("injected source failure")

    monkeypatch.setattr(widget, "_sync_controls_v2_source_index", raise_sync)
    # a helper that returns without restoring is NOT proof of restoration.
    monkeypatch.setattr(
        widget, "_controls_v2_restore_source_caches",
        lambda *_a, **_k: None, raising=False)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "source_reconcile"
    assert ("Source", "cache") in result.recovery_failed_paths
    # the backstop still left the caches at their prior values.
    assert widget._controls_v2_source_energy_cache == prior_energy
    assert widget._controls_v2_metadata_probe_cache == prior_probe


# --- req 7: an early legacy failure recovers only the touched classes -------- #
def test_early_legacy_failure_skips_untouched_classes(widget, monkeypatch):
    first, second = ("Signal", "mask_file"), ("BG", "File")
    first_param = widget._controls_v2_param(first)
    first_before = first_param.value()
    staged = widget.stage_controls_transaction(
        [(first, "/tmp/t25r-a.edf"), (second, "/tmp/t25r-b.edf")])

    display_calls = []
    intent_calls = []
    monkeypatch.setattr(
        widget, "_controls_v2_restore_display_scan",
        lambda *_a, **_k: display_calls.append(1))
    monkeypatch.setattr(
        widget, "_controls_v2_restore_intent_values",
        lambda *_a, **_k: intent_calls.append(1))

    # the SECOND legacy write no-ops so its readback fails BEFORE install; the
    # display/intent classes were never installed and must not be "recovered".
    second_param = widget._controls_v2_param(second)
    monkeypatch.setattr(second_param, "setValue", lambda *_a, **_k: None)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert display_calls == []
    assert intent_calls == []
    assert first_param.value() == first_before  # the older carrier was restored


# --- req 8: a source recovery reconciles the PRIOR selection ----------------- #
def test_source_recovery_reconciles_prior_selection(widget, monkeypatch):
    path = ("Signal", "include_subdir")
    param = widget._controls_v2_param(path)
    prior = bool(param.value())
    staged = widget.stage_controls_transaction([(path, not prior)])
    seen = []

    def observe_then_raise():
        seen.append(bool(widget._controls_v2_param(path).value()))
        raise RuntimeError("injected source failure")

    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index", observe_then_raise)

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "source_reconcile"
    # forward reconcile sees the new selection; the recovery reconcile sees the
    # PRIOR selection (the source-selection carrier was restored first).
    assert seen == [not prior, prior]


# --- req 9: both refusal events keep reason + the complete recovery paths ---- #
def test_nonrun_refusal_event_retains_reason_and_all_recovery_paths(
        widget, monkeypatch):
    from xdart.gui.tabs.static_scan import static_scan_widget as module
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsCommitResult

    events = []
    monkeypatch.setattr(
        module, "run_config_debug_log",
        lambda _logger, event, **payload: events.append((event, payload)))
    monkeypatch.setattr(widget, "_controls_v2_status_message", lambda *_a: None)
    result = ControlsCommitResult(
        False, failed_path=("Signal", "mask_file"),
        reason="precise readback mismatch",
        recovery_failed_paths=(("Display",), ("Intent",)),
        phase="legacy_apply")

    widget._controls_v2_report_pending_refusal(result, "reintegrate1D")

    _, payload = next(
        item for item in events
        if item[0] == "controls_v2_action_refused_pending")
    assert payload["reason"] == "precise readback mismatch"
    assert payload["recovery_failed_paths"] == [["Display"], ["Intent"]]
    assert payload["recovery_failed_path"] == ["Display"]  # convenience kept


def test_run_refusal_event_retains_reason_and_all_recovery_paths(
        widget, monkeypatch):
    from xdart.gui.tabs.static_scan import static_scan_widget as module
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsCommitResult,
        DeferredRunEditsPendingError,
    )

    events = []
    monkeypatch.setattr(
        module, "run_config_debug_log",
        lambda _logger, event, **payload: events.append((event, payload)))
    monkeypatch.setattr(
        widget, "_controls_v2_fold_deferred_edits_into_intent",
        lambda: ControlsCommitResult(
            False, failed_path=("Signal", "mask_file"),
            reason="precise readback mismatch",
            recovery_failed_paths=(("Display",), ("Intent",)),
            phase="legacy_apply"))

    with pytest.raises(DeferredRunEditsPendingError):
        widget._prepare_controls_v2_run_configuration()

    _, payload = next(
        item for item in events
        if item[0] == "run_prepare_aborted_invalid_fold")
    assert payload["reason"] == "precise readback mismatch"
    assert payload["recovery_failed_paths"] == [["Display"], ["Intent"]]
    assert payload["recovery_failed_path"] == ["Display"]  # convenience kept
