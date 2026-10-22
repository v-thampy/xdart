"""REFERENCE-PHASES-20261011 frozen owner oracle BV1-BV3 and FR1 (Claude, October 11).

Int's shared list (BrowserView) gains opt-in per-row numeric editors, and FormRow a decimal-spinbox kind. Real
Qt widgets and real key events drive both; every existing profile and FormRow kind must stay as it is.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering import browser_view as owner
from xdart.gui.tabs.scattering.browser_view import BrowserProfile, BrowserResult, BrowserView, keyed_list_profile

_ROWS = (BrowserResult("a", "Alpha", "first row"), BrowserResult("b", "Beta", "second row"))


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _spec():
    spec = getattr(owner, "ResultValueSpec", None)
    assert spec is not None, "BrowserView must offer opt-in per-row values"
    return spec(suffix=" %", minimum=0., maximum=100., decimals=1)


def _value_view(qapp):
    view = BrowserView(profile=keyed_list_profile("Phases", checkable=True, values=_spec()))
    view.reconcile_checked_results(_ROWS, ("a",))
    view.reconcile_result_values({"a": (5., True, "Alpha minimum"), "b": (2.5, False, "Beta has no intensities")})
    view.show()
    qapp.processEvents()
    return view


def _visible_editors(view):
    return [editor for editor in view.scans.findChildren(QtWidgets.QDoubleSpinBox)
            if editor.isVisibleTo(view.scans)]


def _close(qapp, *widgets):
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


def test_bv1_row_editors_are_opt_in_and_existing_profiles_build_none(qapp):
    from xdart.gui.tabs.rsm.page import _BROWSER as rsm_profile
    existing = (
        BrowserProfile(),  # Int's own Data Browser
        keyed_list_profile("Signal ROIs"),
        keyed_list_profile("Y (overlay)", checkable=True),
        keyed_list_profile("Phases", checkable=True),
        keyed_list_profile("Help topics"),
        BrowserProfile(results=True, result_selection="extended", result_frames=True,
                       actions=("Open…", "Metadata", "Raw image")),  # Stitch's results list
        rsm_profile,
    )
    views = [BrowserView(profile=profile) for profile in existing]
    try:
        for view in views:
            if view.profile.results:
                view.reconcile_results(_ROWS, None)
            view.show()
        qapp.processEvents()
        assert not any(view.scans.findChildren(QtWidgets.QDoubleSpinBox) for view in views)
        valued = _value_view(qapp)
        views.append(valued)
        alpha, beta = valued.result_value_editor("a"), valued.result_value_editor("b")
        assert isinstance(alpha, QtWidgets.QDoubleSpinBox) and isinstance(beta, QtWidgets.QDoubleSpinBox)
        assert (alpha.value(), alpha.suffix(), alpha.decimals(), alpha.minimum(), alpha.maximum()) == (
            5., " %", 1, 0., 100.)
        assert beta.value() == 2.5
        assert alpha.isEnabled() and not beta.isEnabled()
        assert alpha.toolTip() == "Alpha minimum" and beta.toolTip() == "Beta has no intensities"
        assert len(_visible_editors(valued)) == 2
        assert valued.checked_results() == ("a",)
        with pytest.raises(ValueError):
            BrowserView(profile=BrowserProfile(results=False, result_values=_spec()))
    finally:
        _close(qapp, *views)


def test_bv2_editors_survive_a_real_rebuild_and_leave_no_stale_row(qapp):
    view = _value_view(qapp)
    try:
        # A changed label or tooltip makes reconcile_results rebuild the list from scratch (_fill).
        view.reconcile_checked_results((BrowserResult("a", "Alpha (edited)", "changed detail"), _ROWS[1]), ("a",))
        qapp.processEvents()
        assert view.scans.item(0).text() == "Alpha (edited)", "the rebuild really happened"
        alpha, beta = view.result_value_editor("a"), view.result_value_editor("b")
        assert alpha is not None and beta is not None
        assert alpha.value() == 5. and alpha.isEnabled() and alpha.toolTip() == "Alpha minimum"
        assert beta.value() == 2.5 and not beta.isEnabled() and beta.toolTip() == "Beta has no intensities"
        view.reconcile_checked_results((_ROWS[1],), ())
        view.reconcile_result_values({"b": (2.5, False, "Beta has no intensities")})
        qapp.processEvents()
        assert view.result_value_editor("a") is None
        assert len(_visible_editors(view)) == 1 and view.result_value_editor("b").value() == 2.5
    finally:
        _close(qapp, view)


def test_bv3_projection_is_silent_and_one_user_edit_commits_once(qapp):
    view = _value_view(qapp)
    values, checks = [], []
    view.resultValueChanged.connect(lambda identifier, value: values.append((identifier, value)))
    view.checksChanged.connect(checks.append)
    try:
        view.reconcile_result_values({"a": (7., True, "Alpha minimum"), "b": (2.5, False, "Beta has no intensities")})
        view.reconcile_checked_results(_ROWS, ("a", "b"))
        view.reconcile_checked_results((BrowserResult("a", "Alpha again", "x"), _ROWS[1]), ("a",))
        qapp.processEvents()
        assert not values and not checks, (values, checks)
        editor = view.result_value_editor("a")
        assert editor.value() == 7.
        view.activateWindow()
        editor.setFocus()
        qapp.processEvents()
        editor.selectAll()
        QtTest.QTest.keyClicks(editor, "12")
        qapp.processEvents()
        assert not values, "typing alone commits nothing"
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert values == [("a", 12.)], values
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
        view.reconcile_result_values({"a": (12., True, "Alpha minimum"), "b": (2.5, False, "x")})
        qapp.processEvents()
        assert values == [("a", 12.)] and not checks, (values, checks)
    finally:
        _close(qapp, view)


def test_fr1_float_form_row_is_a_numeric_spinbox_and_other_kinds_are_unchanged(qapp):
    from xdart.gui.widgets.controls_panel import FormRow
    row = FormRow(label="Minimum intensity", path=("phases", "minimum"), value=5., kind="float",
                  minimum=0., maximum=100., decimals=1, suffix=" %")
    edits = []
    row.valueChanged.connect(lambda path, value: edits.append((path, value)))
    others = []
    try:
        assert isinstance(row.editor, QtWidgets.QDoubleSpinBox)
        assert (row.editor.suffix(), row.editor.decimals(), row.editor.minimum(), row.editor.maximum()) == (
            " %", 1, 0., 100.)
        assert row.current_value() == 5. and type(row.current_value()) is float
        row.set_number(7.5)
        assert row.current_value() == 7.5 and not edits, "a projection is silent"
        row.show()
        row.activateWindow()
        row.editor.setFocus()
        qapp.processEvents()
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, "12")
        QtTest.QTest.keyClick(row.editor, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert edits == [(("phases", "minimum"), 12.)], edits
        QtTest.QTest.keyClick(row.editor, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        assert len(edits) == 1, "an unchanged value commits nothing"
        row.editor.selectAll()
        QtTest.QTest.keyClicks(row.editor, "30")
        qapp.processEvents()
        row.set_number(12.)  # equal to the committed value: the focused draft is kept
        assert row.editor.cleanText().startswith("30"), row.editor.cleanText()
        row.set_number(40.)  # a changed value is authoritative
        assert row.current_value() == 40. and len(edits) == 1
        row.set_purpose_tooltip("Applies to every CIF phase without its own minimum.")
        assert row.editor.toolTip() == "Applies to every CIF phase without its own minimum."
        row.editor.setEnabled(False)
        assert not row.editor.isEnabled() and row.editor.toolTip()
        line = FormRow(label="Name", path=("x", "name"), value="text")
        whole = FormRow(label="Bins", path=("x", "bins"), value="12", kind="int", minimum=2)
        toggle = FormRow(label="On", path=("x", "on"), value=True, kind="bool")
        combo = FormRow(label="Mode", path=("x", "mode"), value="b", kind="combo", choices=("a", "b"))
        others = [line, whole, toggle, combo]
        assert type(line.editor) is QtWidgets.QLineEdit and line.current_value() == "text"
        assert whole.current_value() == "12" and whole.refusal() == ""
        assert type(toggle.editor) is QtWidgets.QPushButton and toggle.current_value() is True
        assert combo.current_value() == "b"
    finally:
        _close(qapp, row, *others)
