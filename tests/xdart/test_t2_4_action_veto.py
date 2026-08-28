"""In-tree acceptance tests for O-1a-T2.4 (idle checked commit, action veto,
strict staging — §15.12-A).

Committed guards for Correction A: an INVALID focused idle edit mutates no
carrier and is refused by every production action (Advanced / Reintegrate 1D-2D
/ Calibrate / Make Mask / Config Save); a value that validates still applies;
and the strict staging path rejects fuzzy garbage while still accepting the
legitimate unit spellings.  Driven through the real ``_on_controls_v2_action`` /
``_controls_v2_click_integrator_button`` / ``save_defaults`` seams with real
``FormRow`` editors."""

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
    from xdart.gui.widgets.controls_panel import FormRow

    path = tuple(path)
    for row in widget.controls_v2.findChildren(FormRow):
        if tuple(row.path) == path:
            return row
    return None


def _type_invalid_points(widget, qapp, text="4.5"):
    """Type a non-integral value into the (focused) Int1D points editor."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    widget.scan.bai_1d_args["numpoints"] = 321
    row = _form_row(widget, ("Int1D", "points"))
    assert row is not None
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, text)
    qapp.processEvents()
    return copy.deepcopy(intent.bai_1d_args), copy.deepcopy(widget.scan.bai_1d_args)


class _SpyButton:
    def __init__(self):
        self.clicks = 0

    def click(self):
        self.clicks += 1


@pytest.mark.parametrize("button", ["reintegrate1D", "reintegrate2D"])
def test_reintegrate_click_refuses_focused_invalid(widget, qapp, monkeypatch, button):
    intent_before, scan_before = _type_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(widget.integratorTree.ui, button, spy, raising=False)

    performed = widget._controls_v2_click_integrator_button(button)

    assert performed is False                       # §15.12-A.5 refused
    assert spy.clicks == 0                           # zero integration click
    assert widget.scan.bai_1d_args == scan_before    # live scan unchanged
    assert widget._controls_v2_ensure_run_intent().bai_1d_args == intent_before
    entry = widget._controls_v2_edit_journal_dict().get(("Int1D", "points"))
    assert entry is not None and entry["value"] == "4.5"   # journal retained


def test_calibrate_refuses_and_skips_autofill_on_focused_invalid(
        widget, qapp, monkeypatch):
    from xrd_tools.session.readiness import ControlAction

    _type_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(widget.integratorTree.ui, "pyfai_calib", spy, raising=False)
    autofilled = []
    monkeypatch.setattr(
        widget, "_autofill_poni_after_calibrate",
        lambda *a, **k: autofilled.append(1))

    widget._on_controls_v2_action(ControlAction.CALIBRATE)

    assert spy.clicks == 0            # no calibrate click
    assert autofilled == []           # §15.12-A.5: no post-action PONI autofill


def test_make_mask_refuses_focused_invalid(widget, qapp, monkeypatch):
    from xrd_tools.session.readiness import ControlAction

    _type_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(widget.integratorTree.ui, "get_mask", spy, raising=False)

    widget._on_controls_v2_action(ControlAction.MAKE_MASK)

    assert spy.clicks == 0


def test_config_save_refuses_focused_invalid_writes_no_file(
        widget, qapp, tmp_path):
    _type_invalid_points(widget, qapp)
    target = tmp_path / "must-not-save.json"

    widget.h5viewer.defaultWidget.save_defaults(fname=str(target))

    assert not target.exists()        # §15.12-A.4: pre-save veto, no file


def test_config_save_writes_when_valid(widget, qapp, tmp_path):
    # A valid pending edit commits and the save proceeds.
    row = _form_row(widget, ("Int1D", "points"))
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "512")
    qapp.processEvents()
    target = tmp_path / "ok.json"

    widget.h5viewer.defaultWidget.save_defaults(fname=str(target))

    assert target.exists()            # valid -> committed -> saved


def test_focus_loss_of_invalid_edit_mutates_no_carrier(widget, qapp):
    """The idle focus-loss commit of an invalid value refuses without mutating
    intent or scan; the journal keeps the draft (§15.12-A.1/A.2 / §15.3)."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    widget.scan.bai_1d_args["numpoints"] = 321
    intent_before = copy.deepcopy(intent.bai_1d_args)
    scan_before = copy.deepcopy(widget.scan.bai_1d_args)

    # Simulate the committed field change that a focus transfer emits.
    widget._on_controls_v2_field_changed(("Int1D", "points"), "4.5")

    assert intent.bai_1d_args == intent_before
    assert widget.scan.bai_1d_args == scan_before
    entry = widget._controls_v2_edit_journal_dict().get(("Int1D", "points"))
    assert entry is not None and entry["value"] == "4.5"


def test_valid_idle_edit_applies_through_checked_path(widget):
    widget._on_controls_v2_field_changed(("Int1D", "points"), "654")
    assert widget.scan.bai_1d_args["numpoints"] == 654
    assert widget._controls_v2_ensure_run_intent().bai_1d_args["numpoints"] == 654


def test_below_minimum_idle_edit_is_refused_not_clamped(widget):
    """A value the permissive setter would CLAMP (points below the minimum) is
    refused, not silently clamped (§15.3)."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    widget.scan.bai_1d_args["numpoints"] = 321

    widget._on_controls_v2_field_changed(("Int1D", "points"), "0")

    assert widget.scan.bai_1d_args["numpoints"] == 321     # not clamped-applied


@pytest.mark.parametrize(
    "value", ["Q (Å⁻¹)", "2θ (°)", "q", "2th", "chi", "tth"])
def test_strict_unit_aliases_still_accepted(widget, value):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        StagedControlsTransaction,
    )

    result = widget.stage_controls_transaction([(("Int1D", "unit"), value)])
    assert isinstance(result, StagedControlsTransaction)


@pytest.mark.parametrize("value", ["2garbage", "machine", "qq!!", "theta-x"])
def test_strict_unit_fuzzy_garbage_refused(widget, value):
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError,
    )

    result = widget.stage_controls_transaction([(("Int1D", "unit"), value)])
    assert isinstance(result, ControlsTransactionError)
    assert result.path == ("Int1D", "unit")
