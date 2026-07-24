"""O-1a-T2.5 (§15.12 Correction B) — the complete atomic commit/recovery engine.

The nine required red-before/green-after tests for ``commit_controls_transaction``:
preflight is zero-write and typed (missing root, getter/coercion failures,
uncopyable display snapshot); the forward apply uses stable preflighted handles
and detects replacement by identity; the recovery collector attempts every
carrier class independently in deterministic reverse-application order; the
energy/probe caches are snapshotted before invalidation; and the engine's exact
``reason`` survives into both the Run and non-run visible refusals.

Fixture mirrors the §14 oracle (no profile refresh) so ``widget.scan`` and the
wrangler parameters are the real production objects.
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


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("uncopyable display state")


def _boom(*_a, **_k):
    raise RuntimeError("injected failure")


# 1 ------------------------------------------------------------------------- #
def test_missing_parameter_root_refuses_with_zero_writes(widget):
    path = ("Signal", "mask_file")
    param = widget._controls_v2_param(path)
    prior = param.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25-missing.edf")])
    wrangler = widget.wrangler
    saved = wrangler.parameters
    wrangler.parameters = None
    try:
        result = widget.commit_controls_transaction(staged)
    finally:
        wrangler.parameters = saved

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == path
    assert param.value() == prior


# 2 ------------------------------------------------------------------------- #
def test_preflight_first_getter_failure_is_typed_zero_write(widget, monkeypatch):
    first, second = ("Signal", "mask_file"), ("BG", "File")
    first_param = widget._controls_v2_param(first)
    first_before = first_param.value()
    staged = widget.stage_controls_transaction(
        [(first, "/tmp/t25-a.edf"), (second, "/tmp/t25-b.edf")])

    bg = widget._controls_v2_param(second)
    monkeypatch.setattr(bg, "value", _boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == second
    assert first_param.value() == first_before


def test_preflight_coercion_failure_is_typed_zero_write(widget, monkeypatch):
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    first, second = ("Signal", "mask_file"), ("BG", "File")
    first_param = widget._controls_v2_param(first)
    first_before = first_param.value()
    staged = widget.stage_controls_transaction(
        [(first, "/tmp/t25-a.edf"), (second, "/tmp/t25-coerce-boom.edf")])

    original = module.coerce_control_edit_value

    def flaky(current, incoming):
        if incoming == "/tmp/t25-coerce-boom.edf":
            raise ValueError("coercion refused")
        return original(current, incoming)

    monkeypatch.setattr(module, "coerce_control_edit_value", flaky)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert result.failed_path == second
    assert first_param.value() == first_before


# 3 ------------------------------------------------------------------------- #
def test_carrier_replaced_between_preflight_and_apply(widget, monkeypatch):
    path = ("Signal", "mask_file")
    real = widget._controls_v2_param(path)
    prior = real.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25-replaced.edf")])
    original = widget._controls_v2_param
    calls = {"n": 0}

    class Ghost:
        def value(self):
            return prior

        def blockSignals(self, _v):
            return False

        def setValue(self, _v):
            return None

    def flaky_param(candidate_path):
        candidate_path = tuple(candidate_path)
        if candidate_path == path:
            calls["n"] += 1
            return real if calls["n"] == 1 else Ghost()
        return original(candidate_path)

    monkeypatch.setattr(widget, "_controls_v2_param", flaky_param)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "legacy_apply"
    assert result.failed_path == path
    assert real.value() == prior  # the real carrier was never mutated


# 4 ------------------------------------------------------------------------- #
def test_display_snapshot_exception_is_contained_legacy_unwritten(
        widget, monkeypatch):
    path = ("Signal", "mask_file")
    param = widget._controls_v2_param(path)
    prior = param.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25-snap.edf")])

    monkeypatch.setattr(widget, "_controls_v2_snapshot_display_scan", _boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert param.value() == prior  # snapshot is before the first write


# 5 ------------------------------------------------------------------------- #
def test_uncopyable_display_value_zero_writes_typed_refusal(widget):
    path = ("Signal", "mask_file")
    param = widget._controls_v2_param(path)
    prior = param.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25-uncopy.edf")])
    widget.scan.sample_orientation = _Uncopyable()

    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "preflight"
    assert param.value() == prior


# 6 ------------------------------------------------------------------------- #
def test_recovery_continues_when_display_restore_throws(widget, monkeypatch):
    path = ("Signal", "mask_file")
    param = widget._controls_v2_param(path)
    before = param.value()
    staged = widget.stage_controls_transaction([(path, "/tmp/t25-recover.edf")])

    # Install fails AFTER the legacy write, triggering recovery; the display
    # restore then throws, but the legacy carrier must still be restored and the
    # display failure recorded (recovery never aborts on the first exception).
    monkeypatch.setattr(widget, "_controls_v2_apply_snapshot_to_scan", _boom)
    monkeypatch.setattr(widget, "_controls_v2_restore_display_scan", _boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "install"
    assert ("Display",) in result.recovery_failed_paths
    assert param.value() == before  # older carrier restored despite display throw


# 7 ------------------------------------------------------------------------- #
def test_mixed_energy_pref_and_source_failure_restores_original_caches(
        widget, monkeypatch):
    path = ("Signal", "include_subdir")
    current = bool(widget._controls_v2_param(path).value())
    prior_energy = ("orig-source", 9.99)
    prior_probe = ("orig-source", "probe")
    widget._controls_v2_source_energy_cache = prior_energy
    widget._controls_v2_metadata_probe_cache = prior_probe
    staged = widget.stage_controls_transaction(
        [(("Source", "energy_preference"), "metadata"), (path, not current)])

    monkeypatch.setattr(widget, "_sync_controls_v2_source_index", _boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    assert result.phase == "source_reconcile"
    assert widget._controls_v2_source_energy_cache == prior_energy
    assert widget._controls_v2_metadata_probe_cache == prior_probe
    assert result.recovery_failed_path == ("Source",)


# 8 ------------------------------------------------------------------------- #
def test_all_recovery_failures_reported_in_reverse_order(widget, monkeypatch):
    path = ("Signal", "include_subdir")
    current = bool(widget._controls_v2_param(path).value())
    staged = widget.stage_controls_transaction([(path, not current)])

    monkeypatch.setattr(widget, "_sync_controls_v2_source_index", _boom)
    monkeypatch.setattr(widget, "_controls_v2_restore_display_scan", _boom)
    monkeypatch.setattr(widget, "_controls_v2_restore_intent_values", _boom)
    result = widget.commit_controls_transaction(staged)

    assert not result.ok
    # reverse-application order: source owner -> display -> intent/self-state.
    assert result.recovery_failed_paths == (
        ("Source",), ("Display",), ("Intent",))


# 9 ------------------------------------------------------------------------- #
def test_specific_reason_survives_into_run_and_non_run_refusal(
        widget, monkeypatch):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsCommitResult,
        DeferredRunEditsPendingError,
    )

    reason = "exact carrier readback mismatch (unit sentinel)"

    monkeypatch.setattr(
        widget,
        "_controls_v2_fold_deferred_edits_into_intent",
        lambda: ControlsCommitResult(
            False, failed_path=("Signal", "mask_file"), reason=reason,
            phase="legacy_apply"),
    )

    with pytest.raises(DeferredRunEditsPendingError) as excinfo:
        widget._prepare_controls_v2_run_configuration()
    assert reason in str(excinfo.value)

    messages = []
    monkeypatch.setattr(widget, "_controls_v2_status_message", messages.append)
    result = ControlsCommitResult(
        False, failed_path=("Signal", "mask_file"), reason=reason,
        phase="legacy_apply")
    widget._controls_v2_report_pending_refusal(result, "reintegrate1D")
    assert messages and reason in messages[0]


# ========================================================================== #
# §17.4 — the valid-edit installation boundary (idle edits install ONLY
# through the same atomic engine; the retired second setter is unreachable).
# ========================================================================== #

def test_valid_idle_edit_install_raise_is_contained(widget, monkeypatch):
    """A VALID native idle edit whose display projection raises: no exception
    escapes the Qt slot, intent/scan are unchanged, and the journal is retained
    (the atomic engine is the installer, not an unchecked second setter)."""
    path = ("Int1D", "points")
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    widget.scan.bai_1d_args["numpoints"] = 321
    intent_before = copy.deepcopy(intent.bai_1d_args)
    scan_before = copy.deepcopy(widget.scan.bai_1d_args)
    escaped = []

    monkeypatch.setattr(widget, "_controls_v2_apply_snapshot_to_scan", _boom)
    try:
        widget._on_controls_v2_field_changed(path, "400")
    except Exception as exc:  # the production Qt slot MUST contain this
        escaped.append(exc)

    assert escaped == []
    assert intent.bai_1d_args == intent_before
    assert widget.scan.bai_1d_args == scan_before
    assert (path, "400") in widget._controls_v2_journal_winners()


def test_valid_legacy_edit_readback_failure_is_typed_refusal(widget, monkeypatch):
    """A VALID legacy idle edit whose carrier write cannot read back is a typed
    refusal — the carrier is rolled back and NO false ``controls_field_applied``
    event fires."""
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    path = ("Signal", "mask_file")
    param = widget._controls_v2_param(path)
    before = param.value()
    events = []
    original_log = module.run_config_debug_log

    def capture(_logger, event, **kw):
        events.append(event)
        return original_log(_logger, event, **kw)

    monkeypatch.setattr(module, "run_config_debug_log", capture)
    monkeypatch.setattr(widget, "_controls_v2_status_message", lambda *_a: None)
    # The forward legacy write no-ops, so the readback cannot match -> the engine
    # rolls the carrier back and returns a typed legacy_apply failure.
    # §22.10.B.1: the transaction no longer delegates to the permissive
    # compatibility mirror, so the no-op is injected at the STRICT writer.
    monkeypatch.setattr(
        widget, "_controls_v2_write_legacy_carrier", lambda *_a, **_k: None)

    widget._on_controls_v2_field_changed(path, "/tmp/t25-legacy-fail.edf")

    assert "controls_field_applied" not in events
    assert param.value() == before  # carrier restored, not half-applied


def test_threshold_cross_field_stays_journal_only_then_commits(widget, monkeypatch):
    """min entered before max (a transient cross-field contradiction) stays
    JOURNAL-OWNED and is not serialized as committed; a later resolving edit
    stages the complete winner set and commits it."""
    from xrd_tools.session.run_configuration import FrozenThresholdPolicy

    monkeypatch.setattr(widget, "_refresh_controls_v2_profile", lambda *a, **k: None)
    # min=200 then max=100: min>max is a contradiction -> the second edit pends.
    widget._on_controls_v2_field_changed(("Mask", "min"), "200")
    widget._on_controls_v2_field_changed(("Mask", "max"), "100")

    # The contradiction is never serialized as committed session state.
    state = widget._controls_v2_int_session_state()["threshold_config"]
    FrozenThresholdPolicy(**state)  # raises if 200/100 were installed
    winners = dict(widget._controls_v2_journal_winners())
    assert ("Mask", "max") in winners  # the contradictory edit is journal-only

    # Resolving edit: max=500 (>200) completes a valid winner set -> commit.
    widget._on_controls_v2_field_changed(("Mask", "max"), "500")
    winners_after = dict(widget._controls_v2_journal_winners())
    assert ("Mask", "max") not in winners_after
    assert ("Mask", "min") not in winners_after


def test_gi_boolean_and_int2d_points_round_trip_through_atomic_owner(
        widget, monkeypatch):
    """The GI-boolean and Int2D-points round trips that motivated the retired
    deviation now go THROUGH the atomic owner — exact final values, ONE
    projection and ONE generation bump per committed idle edit."""
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    projections = []
    original_proj = widget._controls_v2_apply_snapshot_to_scan
    monkeypatch.setattr(
        widget, "_controls_v2_apply_snapshot_to_scan",
        lambda snapshot, *a, **k: (projections.append(1),
                                   original_proj(snapshot, *a, **k))[1])
    bumps = []
    monkeypatch.setattr(
        module, "bump_run_config_debug_generation",
        lambda owner, kind: bumps.append(kind))

    projections.clear()
    bumps.clear()
    widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
    assert widget.scan.gi is True                 # GI-boolean round-trip
    assert projections.count(1) == 1              # one projection
    assert bumps == ["config"]                    # one generation bump

    projections.clear()
    bumps.clear()
    widget._on_controls_v2_field_changed(("Int2D", "radial_points"), "1234")
    assert widget.scan.bai_2d_args["npt_rad"] == 1234   # Int2D-points round-trip
    assert projections.count(1) == 1
    assert bumps == ["config"]


def test_incomplete_contradiction_snapshot_persists_last_valid(widget, monkeypatch):
    """A session snapshot taken DURING an incomplete contradiction reflects the
    last valid COMMITTED state — the draft does not masquerade as committed."""
    from xrd_tools.session.run_configuration import FrozenThresholdPolicy

    monkeypatch.setattr(widget, "_refresh_controls_v2_profile", lambda *a, **k: None)
    # A valid committed edit first, then a contradicting draft.
    widget._on_controls_v2_field_changed(("Mask", "min"), "10")
    committed = widget._controls_v2_int_session_state()["threshold_config"]
    assert committed["threshold_min"] == 10

    widget._on_controls_v2_field_changed(("Mask", "max"), "5")  # 5 < min=10 -> pends
    snapshot = widget._controls_v2_int_session_state()["threshold_config"]
    FrozenThresholdPolicy(**snapshot)               # valid: draft not serialized
    assert snapshot["threshold_min"] == 10          # last valid committed persists


def test_no_user_signal_reaches_retired_setter():
    """Architecture guard: the retired ``_apply_controls_v2_field_value`` has NO
    call sites — no user idle signal can reach the second setter authority."""
    import pathlib

    from xdart.gui.tabs.static_scan import static_scan_widget as module

    source = pathlib.Path(module.__file__).read_text()
    assert "self._apply_controls_v2_field_value(" not in source
