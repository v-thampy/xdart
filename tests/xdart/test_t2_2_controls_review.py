"""In-tree acceptance tests for O-1a-T2.2 (strict candidate schema + draft
overlay + frozen journal).

These are the committed regression guards for the Fifteenth-amendment T-2.2
contract — the Codex §12/§13/§14 oracle modules are NOT committed, so the
behaviours they exercise are pinned here against the REAL production seams
(real ``staticWidget`` offscreen, real ``_on_controls_v2_action`` /
``_controls_v2_click_integrator_button`` handlers, real ``RangeRow`` editors,
the real ``sigGIMotorOptions`` owner).  §13.11 tests 1-4, item 7 (RangeRow
committed), item 10 (E-9 regression), item 11 (idle-PONI parity), item 12
(hydration hardening), item 5 (frozen journal), item 2 (committed-legacy) and
item 6 (dirty-only harvest)."""

from __future__ import annotations

import copy

import pytest
from pyqtgraph.Qt import QtTest, QtWidgets


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


def _form_row(widget, path):
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import FormRow

    path = tuple(path)
    for row in widget.controls_v2.findChildren(FormRow):
        if tuple(row.path) == path:
            return row
    return None


def _range_row(widget, low_path):
    from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import RangeRow

    low_path = tuple(low_path)
    for row in widget.controls_v2.findChildren(RangeRow):
        if tuple(getattr(row, "_low_path", ())) == low_path:
            return row
    return None


class _SpyButton:
    """A stand-in for an integratorTree.ui action button that records clicks
    WITHOUT performing an integration (so a refusal proves zero action)."""

    def __init__(self):
        self.clicks = 0

    def click(self):
        self.clicks += 1


# ---------------------------------------------------------------------------
# §13.11 tests 1-2 — production actions refuse an invalid focused native value.
# ---------------------------------------------------------------------------

def test_advanced_action_refuses_invalid_focused_edit(widget, monkeypatch):
    from xdart.gui.tabs.static_scan.controls_logic import ControlAction

    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    before = copy.deepcopy(intent.bai_1d_args)
    row = _form_row(widget, ("Int1D", "points"))
    assert row is not None
    # A genuinely-invalid (non-integral) focused draft — strict reduce refuses it.
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "4.5")

    opened = []
    monkeypatch.setattr(
        widget, "_show_integration_advanced", lambda: opened.append(1))

    widget._on_controls_v2_action(ControlAction.ADVANCED_PROCESSING)

    assert opened == []                                   # zero delegated action
    assert intent.bai_1d_args == before                  # live state unchanged
    entry = widget._controls_v2_edit_journal_dict().get(("Int1D", "points"))
    assert entry is not None and entry["value"] == "4.5"  # journal retained


@pytest.mark.parametrize("button", ["reintegrate1D", "reintegrate2D"])
def test_reintegrate_action_refuses_invalid_focused_edit(
        widget, monkeypatch, button):
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    scan_before = copy.deepcopy(widget.scan.bai_1d_args)
    row = _form_row(widget, ("Int1D", "points"))
    assert row is not None
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "4.5")

    spy = _SpyButton()
    monkeypatch.setattr(widget.integratorTree.ui, button, spy, raising=False)

    widget._controls_v2_click_integrator_button(button)

    assert spy.clicks == 0                                # zero integration click
    assert widget.scan.bai_1d_args == scan_before        # live scan unchanged
    entry = widget._controls_v2_edit_journal_dict().get(("Int1D", "points"))
    assert entry is not None and entry["value"] == "4.5"  # journal retained


def test_commit_pending_helper_never_raises_on_invalid(widget):
    """The NON-run helper returns a typed refusal, it does NOT raise (§14.11.C.4:
    a bare call cannot leak an exception through a Qt slot)."""
    from xdart.gui.tabs.static_scan.static_scan_widget import ControlsCommitResult

    row = _form_row(widget, ("Int1D", "points"))
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "4.5")

    result = widget._commit_controls_v2_pending_edits()          # must not raise
    assert isinstance(result, ControlsCommitResult)
    assert not result.ok


# ---------------------------------------------------------------------------
# §13.11 test 4 (invalid variant) — an INVALID pending draft survives a forced
# panel rebuild, visible and recoverable (item 4 rebuild overlay).
# ---------------------------------------------------------------------------

def test_invalid_pending_draft_survives_forced_rebuild(widget, qapp):
    from xdart.gui.tabs.static_scan.controls_logic import build_control_panel_state

    path = ("Int1D", "points")
    row = _form_row(widget, path)
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "not-an-int")
    qapp.processEvents()
    assert widget._controls_v2_edit_journal_dict()[path]["value"] == "not-an-int"

    render_state = build_control_panel_state(
        widget._controls_v2_state(),
        widget._controls_v2_field_values(),
        widget._controls_v2_field_choices(),
    )
    widget.controls_v2.set_state(render_state)
    qapp.processEvents()

    rebuilt = _form_row(widget, path)
    assert rebuilt.editor.text() == "not-an-int"   # invalid draft still visible
    # and still recoverable from the journal (revision-stable).
    assert widget._controls_v2_edit_journal_dict()[path]["value"] == "not-an-int"


# ---------------------------------------------------------------------------
# Item 7 — RangeRow committed test (my own; the §13 oracle's disabled-row shape
# is Codex's to fix).  Typing into an ENABLED range editor revisions a draft.
# ---------------------------------------------------------------------------

def test_enabled_range_row_draft_is_revisioned(widget, qapp):
    # Turn auto-range OFF (a concrete radial_range enables the low/high editors).
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["radial_range"] = (0.5, 4.0)
    widget._refresh_controls_v2_profile_now()
    qapp.processEvents()

    row = _range_row(widget, ("Int1D", "radial_low"))
    assert row is not None
    assert row._low.isEnabled(), "auto-range should be off so the editor is live"
    path = tuple(row._low_path)
    row._low.setFocus()
    row._low.selectAll()
    QtTest.QTest.keyClicks(row._low, "1.234")
    qapp.processEvents()

    entry = widget._controls_v2_edit_journal_dict().get(path)
    assert entry is not None
    assert entry["value"] == "1.234"
    assert entry["origin"] == "draft"


# ---------------------------------------------------------------------------
# Item 10 — E-9 regression (ported from the §14 oracle, green at a4a54160 and
# here): a newer same-path revision recorded between commit and clear survives.
# ---------------------------------------------------------------------------

def test_newer_same_path_revision_survives_successful_clear(widget, monkeypatch):
    path = ("Signal", "mask_file")
    widget._controls_v2_record_edit(path, "/tmp/t2-old.edf", origin="deferred")
    original_commit = widget.commit_controls_transaction

    def commit_then_record_newer(staged):
        result = original_commit(staged)
        assert result.ok
        widget._controls_v2_record_edit(path, "/tmp/t2-new.edf", origin="draft")
        return result

    monkeypatch.setattr(
        widget, "commit_controls_transaction", commit_then_record_newer)
    assert widget._controls_v2_fold_deferred_edits_into_intent() is None

    entry = widget._controls_v2_edit_journal_dict()[path]
    assert entry["value"] == "/tmp/t2-new.edf"


# ---------------------------------------------------------------------------
# Item 11 — idle-PONI parity: after a PONI file change, the frozen run carries
# poni_values MATCHING poni_file (not a stale calibration).
# ---------------------------------------------------------------------------

def test_idle_poni_edit_reparses_intent_values(widget, tmp_path):
    from xrd_tools.core.containers import PONI

    poni_path = tmp_path / "idle.poni"
    PONI(
        dist=0.1794,
        poni1=0.01,
        poni2=0.02,
        detector="RayonixMx225",
        wavelength=0.7293e-10,
    ).to_poni_file(poni_path)
    expected = PONI.from_poni_file(str(poni_path)).to_dict()

    intent = widget._controls_v2_ensure_run_intent()
    # Simulate a prior stale calibration, then an idle file-only change.
    intent.poni_values = {"stale": True}
    param = widget._controls_v2_param(("Signal", "poni_file"))
    assert param is not None
    param.setValue(str(poni_path))

    frozen = widget._prepare_controls_v2_run_configuration()
    assert frozen is not None
    assert str(intent.poni_file) == str(poni_path)
    assert intent.poni_values == expected     # re-parsed to MATCH the file
    assert "stale" not in (intent.poni_values or {})


# ---------------------------------------------------------------------------
# Item 12 — hydration hardening: an async GI result requested under source A but
# delivered after the source changed to B carries A's request-start identity and
# is rejected by the owner (never seeds B with A's motors).
# ---------------------------------------------------------------------------

def test_async_gi_hydration_under_changed_source_is_rejected(widget, tmp_path):
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import (
        GIMotorHydration,
    )

    wrangler = widget.wrangler
    captured = []
    wrangler.sigGIMotorOptions.connect(captured.append)

    dir_a = tmp_path / "sourceA"
    dir_b = tmp_path / "sourceB"
    dir_a.mkdir()
    dir_b.mkdir()

    # Request STARTS under source A: bump epoch + capture the A token.
    wrangler.inp_type = "Image Directory"
    wrangler.img_dir = str(dir_a)
    fp_a = wrangler._gi_source_fingerprint()
    wrangler._next_gi_hydration_generation()

    # The source changes to B before A's metadata result comes back.
    wrangler.img_dir = str(dir_b)
    assert wrangler._gi_source_fingerprint() != fp_a

    # A's delayed result emits — it must carry A's request-start fingerprint,
    # NOT the current source B, and the owner must reject it as stale.
    wrangler._emit_gi_hydration(["halpha"], proved=True)

    assert captured, "sigGIMotorOptions did not emit"
    payload = captured[-1]
    assert isinstance(payload, GIMotorHydration)
    assert payload.source_fingerprint == fp_a          # request-start identity
    assert wrangler.gi_hydration_is_current(payload) is False   # owner rejects


def test_known_empty_gi_motor_choices_stay_known_empty(widget):
    """Item 12 tripwire (mirrors the AMENDED §12 oracle, mtime 09:50): a
    set_gi_motor_options() whose empty motor list came from a TARGETED inspection
    (proof=True set explicitly, as the discovery paths do) is KNOWN_EMPTY.  Per
    §15.11-4 / T-2.6b, an UNPROVED bare empty emit is UNKNOWN, so proof is stated
    here explicitly rather than relied on from the fresh-wrangler default."""
    from xdart.gui.tabs.static_scan.static_scan_widget import GIMotorObservation

    widget.wrangler.motors = []
    widget.wrangler._gi_motor_knowledge_proved = True
    widget.wrangler.set_gi_motor_options()
    obs = widget._controls_v2_capture_gi_motor_observation()
    assert obs.state == GIMotorObservation.KNOWN_EMPTY
    assert widget._controls_v2_gi_motor_choices_for_freeze() == ()


# ---------------------------------------------------------------------------
# Item 5 (§13.9) — the journal stores a frozen, deep-copied entry: a caller
# mutating a list/dict value in place cannot change the journal without a new
# revision, and outward reads hand back copies.
# ---------------------------------------------------------------------------

def test_journal_entry_is_immutable_and_deepcopied(widget):
    path = ("Int1D", "radial_range")
    payload = [0.5, 4.0]
    widget._controls_v2_record_edit(path, payload, origin="draft")

    # Mutating the caller's original does not reach into the journal.
    payload.append(999)
    stored = widget._controls_v2_edit_journal_dict()[path]["value"]
    assert stored == [0.5, 4.0]

    # Mutating an outward-projected copy does not reach into the journal either.
    stored.append(123)
    assert widget._controls_v2_edit_journal_dict()[path]["value"] == [0.5, 4.0]


# ---------------------------------------------------------------------------
# Item 2 — committed-legacy capture: the pure stage compares/derives against a
# snapshot captured at stage ENTRY, never a live Qt parameter mutated later.
# ---------------------------------------------------------------------------

def test_stage_uses_committed_legacy_snapshot_not_live_param(widget):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        StagedControlsTransaction,
    )

    # A same-value include_subdir edit (coerces back to the live value) must NOT
    # be treated as a source-selection change even if the live param is mutated
    # AFTER the stage captured its committed snapshot.
    param = widget._controls_v2_param(("Signal", "include_subdir"))
    assert param is not None
    current = bool(param.value())

    original_derive = widget._controls_v2_candidate_source_spec
    seen = {}

    def spy_derive(cand):
        # Mutate the LIVE param mid-stage; a committed-snapshot reducer ignores it.
        param.setValue(not current)
        seen["committed"] = dict(cand.committed_legacy)
        return original_derive(cand)

    widget._controls_v2_candidate_source_spec = spy_derive
    try:
        staged = widget.stage_controls_transaction(
            [(("Signal", "include_subdir"), current)])
    finally:
        widget._controls_v2_candidate_source_spec = original_derive
        param.setValue(current)

    assert isinstance(staged, StagedControlsTransaction)
    # The committed snapshot saw the ORIGINAL live value, and the same-value edit
    # reconciles nothing despite the live param having been flipped mid-stage.
    assert seen["committed"][("Signal", "include_subdir")] == current
    assert staged.source_selection_touched is False


# ---------------------------------------------------------------------------
# Item 6 — dirty-only harvest: the collector consumes the journal plus AT MOST
# one focused-editor flush; a stale committed value is never re-journaled.
# ---------------------------------------------------------------------------

def test_focused_flush_is_the_only_form_source(widget, qapp):
    path = ("Int1D", "points")
    row = _form_row(widget, path)
    # A focused, programmatically-set editor value (no textEdited) is flushed as
    # exactly one revision by the collector.
    row.editor.setFocus()
    row.editor.setText("1234")

    winners = dict(widget._controls_v2_collect_pending_edits())
    assert winners.get(path) == "1234"
    entry = widget._controls_v2_edit_journal_dict()[path]
    assert entry["origin"] == "form"
    assert entry["value"] == "1234"
