"""O-1a-T2.4b (§15.12-A completion): the REAL action focus-transfer matrix.

The committed ``test_t2_4_action_veto.py`` drives the action-owner seams
DIRECTLY (calling the handlers while the editor is still focused), so it never
exercises Qt's real focus-transfer-before-``clicked`` ordering (§15.3): a mouse
click on an action button transfers focus OUT of the line edit — emitting the
committed ``valueChanged`` for the invalid draft through the permissive idle
path — BEFORE the button's ``clicked`` fires.  These tests use a REAL
``QTest.mouseClick`` on the production buttons with a REAL focused editor holding
an invalid draft (``hasFocus()`` asserted before the click) and prove every
action owner (Advanced, Reintegrate 1D/2D, Calibrate, Make-Mask, Config Save)
refuses without a permissive mutation.

No test pre-transfers focus programmatically before the click: the ONLY focus
change is the mouse click itself.  Adapted and extended from the verification
agent's T-2.4 probe (proven 4/4 red at a0c3fd99, 4/4 green at da68388d).
"""

from __future__ import annotations

import copy

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets


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


def _action_button(widget, action):
    from xdart.gui.widgets.controls_panel import ActionButton

    for candidate in widget.controls_v2.findChildren(ActionButton):
        if candidate.spec.action == action:
            return candidate
    return None


class _SpyButton:
    def __init__(self):
        self.clicks = 0

    def click(self):
        self.clicks += 1


def _focused_invalid_points(widget, qapp, text="4.5"):
    """Type an invalid draft into Int1D points and LEAVE the editor focused."""
    intent = widget._controls_v2_ensure_run_intent()
    intent.bai_1d_args["numpoints"] = 321
    widget.scan.bai_1d_args["numpoints"] = 321
    row = _form_row(widget, ("Int1D", "points"))
    assert row is not None
    widget.show()
    qapp.processEvents()
    row.editor.setFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, text)
    qapp.processEvents()
    assert row.editor.hasFocus(), "precondition: editor must hold focus pre-click"
    return (
        copy.deepcopy(intent.bai_1d_args),
        copy.deepcopy(widget.scan.bai_1d_args),
    )


def _real_click(widget, qapp, button):
    assert button is not None
    if not button.isEnabled():
        # UI readiness gating is orthogonal to the veto under test; the click
        # still routes through the REAL clicked->actionRequested->handler chain.
        button.setEnabled(True)
    QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
    qapp.processEvents()


def _assert_journal_retains_invalid_points(widget):
    entry = widget._controls_v2_edit_journal_dict().get(("Int1D", "points"))
    assert entry is not None and entry["value"] == "4.5", "journal lost the draft"


@pytest.mark.parametrize(
    ("action_name", "ui_button"),
    [("REINTEGRATE_1D", "reintegrate1D"), ("REINTEGRATE_2D", "reintegrate2D")],
)
def test_real_reintegrate_click_focused_invalid(
        widget, qapp, monkeypatch, action_name, ui_button):
    from xrd_tools.session.readiness import ControlAction

    intent_before, scan_before = _focused_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(widget.integratorTree.ui, ui_button, spy, raising=False)
    button = _action_button(widget, getattr(ControlAction, action_name))

    _real_click(widget, qapp, button)

    assert spy.clicks == 0, "delegated reintegrate ran despite invalid draft"
    assert widget.scan.bai_1d_args == scan_before, "scan mutated by real click"
    assert (widget._controls_v2_ensure_run_intent().bai_1d_args
            == intent_before), "intent mutated by real click"
    _assert_journal_retains_invalid_points(widget)


def test_real_calibrate_click_focused_invalid_skips_autofill(
        widget, qapp, monkeypatch):
    from xrd_tools.session.readiness import ControlAction

    intent_before, scan_before = _focused_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(
        widget.integratorTree.ui, "pyfai_calib", spy, raising=False)
    autofilled = []
    monkeypatch.setattr(
        widget, "_autofill_poni_after_calibrate",
        lambda *a, **k: autofilled.append(1))
    button = _action_button(widget, ControlAction.CALIBRATE)

    _real_click(widget, qapp, button)

    assert spy.clicks == 0, "calibrate delegated despite invalid draft"
    assert autofilled == [], "PONI autofill ran after a refused calibrate"
    assert widget.scan.bai_1d_args == scan_before, "scan mutated by real click"
    assert (widget._controls_v2_ensure_run_intent().bai_1d_args
            == intent_before), "intent mutated by real click"
    _assert_journal_retains_invalid_points(widget)


def test_real_make_mask_click_focused_invalid(widget, qapp, monkeypatch):
    from xrd_tools.session.readiness import ControlAction

    intent_before, scan_before = _focused_invalid_points(widget, qapp)
    spy = _SpyButton()
    monkeypatch.setattr(
        widget.integratorTree.ui, "get_mask", spy, raising=False)
    button = _action_button(widget, ControlAction.MAKE_MASK)

    _real_click(widget, qapp, button)

    assert spy.clicks == 0, "make-mask delegated despite invalid draft"
    assert widget.scan.bai_1d_args == scan_before, "scan mutated by real click"
    assert (widget._controls_v2_ensure_run_intent().bai_1d_args
            == intent_before), "intent mutated by real click"
    _assert_journal_retains_invalid_points(widget)


def test_real_advanced_click_focused_invalid_does_not_open(
        widget, qapp, monkeypatch):
    from xrd_tools.session.readiness import ControlAction

    intent_before, scan_before = _focused_invalid_points(widget, qapp)
    opened = []
    monkeypatch.setattr(
        widget, "_show_integration_advanced", lambda: opened.append(1))
    button = _action_button(widget, ControlAction.ADVANCED_PROCESSING)

    _real_click(widget, qapp, button)

    assert opened == [], "Advanced dialog opened despite invalid draft"
    assert widget.scan.bai_1d_args == scan_before, "scan mutated by real click"
    assert (widget._controls_v2_ensure_run_intent().bai_1d_args
            == intent_before), "intent mutated by real click"
    _assert_journal_retains_invalid_points(widget)


def test_real_config_save_click_focused_invalid_writes_no_file(
        widget, qapp, monkeypatch, tmp_path):
    from pyqtgraph import Qt as _Qt

    intent_before, scan_before = _focused_invalid_points(widget, qapp)
    target = tmp_path / "clicked-save.json"
    # Keep the dialog non-interactive; the click itself stays real.
    monkeypatch.setattr(
        _Qt.QtWidgets.QFileDialog, "getSaveFileName",
        lambda *a, **k: (str(target), "*.json"))
    button = widget.h5viewer.defaultWidget.saveButton

    _real_click(widget, qapp, button)

    assert not target.exists(), "Config Save wrote a file despite refusal"
    assert widget.scan.bai_1d_args == scan_before, "scan mutated by real click"
    assert (widget._controls_v2_ensure_run_intent().bai_1d_args
            == intent_before), "intent mutated by real click"
    _assert_journal_retains_invalid_points(widget)


def test_config_save_veto_that_raises_fails_closed(widget, tmp_path):
    """§15.12-A.4 (verifier finding 2): a pre-save veto hook that RAISES must
    FAIL CLOSED — the save is refused and no file is written, rather than the
    old fail-open path that swallowed the exception and serialized anyway."""
    default_widget = widget.h5viewer.defaultWidget
    default_widget._pre_save_veto = (
        lambda: (_ for _ in ()).throw(RuntimeError("veto hook boom")))
    target = tmp_path / "veto-raise-must-not-save.json"

    default_widget.save_defaults(fname=str(target))

    assert not target.exists(), "raising veto fell open and wrote the file"
