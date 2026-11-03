"""NEXT-128 background degrees: the GUI rows G1-G8 on both fit windows (Claude, 2026-11-03).

The maintainer, 2026-11-03, during his 1.2.8 live check, of the Peak Fitter's Background (Linear, Constant, None,
Chebyshev (3), Template, Template + linear, Template + Chebyshev (3)): "also, there should be more options for
background - Polynomial (this can replace Linear, constant). when polynomial is chosen, another box to choose the
polynomial degree should show in the same row. same with the chebyshev and Template + Linear (replace with template
with polynomial) and template + Chebyshev - add degree here as well"; for Phase Analysis his "Same in both
(Recommended)".

Frozen before the product change (red on 57fca340). The windows are the real dialogs on the shared rows (hard rule 14);
settings go through the shared ``PageSettings``; notebooks are exported with ``write_fit_notebook`` and executed,
viewer cell included. Only the settings file chooser is answered. New APIs are reached inside the test bodies.
"""
import importlib.util
import json

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_fit_tools_preservation import _peaks, _trace

PEAK_CHOICES = (("None", "none"), ("Polynomial", "polynomial"), ("Chebyshev", "chebyshev"), ("Template", "template"),
                ("Template + polynomial", "template+polynomial"), ("Template + Chebyshev", "template+chebyshev"))
PHASE_CHOICES = (("None", "none"), ("Polynomial", "polynomial"), ("Chebyshev", "chebyshev"), ("Spline", "spline"),
                 ("Substrate", "template"), ("Substrate + polynomial", "template+polynomial"),
                 ("Substrate + Chebyshev", "template+chebyshev"))
DEGREE_CHOICES = {"polynomial", "chebyshev", "spline", "template+polynomial", "template+chebyshev"}


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))


def _dialog(kind, submitted=None):
    from xdart.gui.analysis.peak_fit_dialog import PeakFitDialog
    from xdart.gui.analysis.phase_fit_dialog import PhaseFitDialog
    submit = (lambda *args: None) if submitted is None else (lambda *args: submitted.append(args))
    return (PeakFitDialog if kind == "peak" else PhaseFitDialog)(vnext_submit=submit)


def _close(qapp, dialog):
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def _rows(dialog, kind):
    """The background choice's row, its combo and its degree box."""
    if kind == "peak":
        return dialog.bkg_row, dialog.bkg_combo, dialog.bkg_degree_row
    return dialog.fit_background_row, dialog.fit_background_row.editor, dialog.fit_background_degree_row


def _choose(qapp, combo, value):
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)
    qapp.processEvents()


def _type(qapp, row, text):
    """Type into the whole-number box as the user does (the text kept as typed, refused text shown so)."""
    row.editor.setFocus()
    row.editor.selectAll()
    row.editor.insert(text)
    qapp.processEvents()


def _right(widget, ancestor):
    return widget.mapTo(ancestor, QtCore.QPoint(widget.width(), 0)).x()


def _left(widget, ancestor):
    return widget.mapTo(ancestor, QtCore.QPoint(0, 0)).x()


def _card(row):
    from xdart.gui.widgets.controls_panel import SectionCard
    widget = row
    while widget is not None and not isinstance(widget, SectionCard):
        widget = widget.parentWidget()
    return widget


def _load(qapp, dialog, path, fields):
    """Load a settings document with only ``fields`` (an earlier file's), through the shared PageSettings."""
    kind = dialog.settings_io.kind
    path.write_text(json.dumps({"kind": kind.kind, "version": kind.version, "fields": fields}), encoding="utf-8")
    dialog._settings_path_chooser = lambda action, start: str(path)
    outcome = dialog.settings_io.load()
    qapp.processEvents()
    return outcome.__class__.__name__


# ---- G1 the Peak Fitter's choices, and Polynomial 1 (the earlier Linear) by default ------------------------------

def test_g1_the_peak_fitter_lists_polynomial_and_chebyshev_and_starts_at_polynomial_one(qapp):
    dialog = _dialog("peak")
    try:
        row, combo, degree = _rows(dialog, "peak")
        assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == list(PEAK_CHOICES)
        assert combo.currentData() == "polynomial" and degree.editor.value() == 1
        assert dialog.vnext_fit_values()["background"] == "polynomial1"
        assert not degree.isHidden() and degree.editor.isEnabled()
    finally:
        _close(qapp, dialog)


# ---- G2 the degree box sits in the choice's row, before its "?", shown only for a choice that takes one -----------

@pytest.mark.parametrize("kind", ("peak", "phase"))
def test_g2_the_degree_box_sits_in_the_row_before_its_question_mark_and_shows_only_for_a_degree(qapp, kind):
    from xdart.gui.widgets.controls_panel import FormRow
    topic = "peak-fit.background" if kind == "peak" else "phase-analysis.background"
    dialog = _dialog(kind)
    try:
        dialog.show()
        qapp.processEvents()
        row, combo, degree = _rows(dialog, kind)
        assert isinstance(degree, FormRow) and row.isAncestorOf(degree), "the shared row, in the choice's row"
        assert degree.property("helpTopic") == topic, "F1 opens the background's help"
        assert not hasattr(degree, "help_button"), "the row's own '?' serves it"
        card = _card(row)
        for value in (value for _text, value in (PEAK_CHOICES if kind == "peak" else PHASE_CHOICES)):
            _choose(qapp, combo, value)
            takes = value in DEGREE_CHOICES
            assert degree.isHidden() is not takes, value
            assert degree.isVisibleTo(dialog) is takes, value
            if takes:
                assert degree.editor.isEnabled(), "shown and editable, never shown disabled"
                assert degree.toolTip() and degree.editor.toolTip(), value
                assert _right(degree, card) <= _left(row.help_button, card), (value, "the '?' stays rightmost")
                assert _right(combo, card) <= _left(degree, card), (value, "after the choice")
        rights = [_right(other.help_button, card) for other in card.findChildren(FormRow)
                  if hasattr(other, "help_button") and other.help_button.isVisible()]
        assert rights and max(rights) - min(rights) <= 1, ("every '?' in one column", rights)  # as PF2/PF3 allow
    finally:
        _close(qapp, dialog)


# ---- G3 each family keeps its own degree, within its range, named in the box's tooltip ----------------------------

def test_g3_the_peak_fitters_families_keep_their_degrees_within_their_ranges(qapp):
    submitted = []
    dialog = _dialog("peak", submitted)
    try:
        _row, combo, degree = _rows(dialog, "peak")
        trace, _anchor = _trace(*_peaks())
        dialog.set_vnext_trace(trace)
        assert "0 to 7" in degree.toolTip(), degree.toolTip()
        _type(qapp, degree, "8")
        assert degree.refusal(), "Polynomial stops at 7 (lmfit's PolynomialModel)"
        dialog.fit_btn.click()
        assert not any(action == "fit" for action, _values in submitted)
        assert "Degree" in dialog.status.text(), dialog.status.text()
        with pytest.raises(ValueError, match="Degree"):
            dialog.export_fit_values()  # a notebook must fit what the window would
        _type(qapp, degree, "3")
        assert dialog.vnext_fit_values()["background"] == "polynomial3"
        _choose(qapp, combo, "chebyshev")
        assert degree.editor.value() == 3 and "0 to 15" in degree.toolTip(), degree.toolTip()
        _type(qapp, degree, "6")
        assert dialog.vnext_fit_values()["background"] == "chebyshev6"
        _choose(qapp, combo, "template+polynomial")
        assert dialog.vnext_fit_values()["background"] == "template+polynomial3", "the polynomial's own degree"
        _choose(qapp, combo, "template+chebyshev")
        assert dialog.vnext_fit_values()["background"] == "template+chebyshev6"
        _choose(qapp, combo, "polynomial")
        _type(qapp, degree, "99")
        _choose(qapp, combo, "none")
        assert dialog.vnext_fit_values()["background"] == "none"
        dialog.fit_btn.click()
        assert submitted and submitted[-1][0] == "fit" and submitted[-1][1]["background"] == "none", (
            "a hidden box's text never refuses a fit")
    finally:
        _close(qapp, dialog)


def test_g3_phase_analysis_families_keep_their_degrees_within_their_ranges(qapp):
    dialog = _dialog("phase")
    try:
        _row, combo, degree = _rows(dialog, "phase")
        assert dialog.vnext_fit_values()["fit_background_degree"] == 3, "no degree: the plan's own default"
        _choose(qapp, combo, "polynomial")
        assert degree.editor.value() == 1 and degree.label.text() == "Degree", "the earlier Linear"
        assert "0 to 15" in degree.toolTip(), degree.toolTip()
        values = dialog.vnext_fit_values()
        assert (values["fit_background"], values["fit_background_degree"]) == ("polynomial", 1)
        _type(qapp, degree, "16")
        assert degree.refusal() and "Degree" in dialog.fit_refusal(), dialog.fit_refusal()
        with pytest.raises(ValueError, match="Degree"):
            dialog.export_fit_values()
        _type(qapp, degree, "2")
        _choose(qapp, combo, "chebyshev")
        assert degree.editor.value() == 3
        _choose(qapp, combo, "spline")
        assert degree.editor.value() == 4 and degree.label.text() == "Knots"
        assert "4 to 15" in degree.toolTip(), degree.toolTip()
        _type(qapp, degree, "3")
        assert degree.refusal(), "a spline needs at least four knots"
        _type(qapp, degree, "6")
        _choose(qapp, combo, "template+polynomial")
        assert degree.editor.value() == 2 and degree.label.text() == "Degree", "the polynomial's own degree"
        _type(qapp, degree, "x")
        _choose(qapp, combo, "none")
        assert dialog.fit_refusal() == "", "a hidden box's text never refuses a fit"
        assert dialog.vnext_fit_values()["fit_background_degree"] == 3
        _choose(qapp, combo, "spline")
        assert degree.editor.value() == 6
    finally:
        _close(qapp, dialog)


# ---- G4 the Peak Fitter saves the degree in its background and loads an earlier file's choices --------------------

def test_g4_the_peak_fitter_saves_the_degree_in_its_background_and_loads_earlier_files(qapp, tmp_path):
    dialog = _dialog("peak")
    path = tmp_path / "peak-fit.json"
    try:
        _row, combo, degree = _rows(dialog, "peak")
        _choose(qapp, combo, "chebyshev")
        _type(qapp, degree, "6")
        dialog._settings_path_chooser = lambda action, start: str(path)
        assert dialog.settings_io.save().__class__.__name__ == "ActionCompleted"
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["fields"]["background"] == "chebyshev6", "no new key: the core's own string"
        _choose(qapp, combo, "none")
        assert dialog.settings_io.load().__class__.__name__ == "ActionCompleted"
        assert combo.currentData() == "chebyshev" and degree.editor.value() == 6 and not degree.isHidden()
        for background, (choice, value) in {
                "linear": ("polynomial", 1), "constant": ("polynomial", 0), "chebyshev3": ("chebyshev", 3),
                "template+linear": ("template+polynomial", 1), "template+chebyshev3": ("template+chebyshev", 3),
                "polynomial5": ("polynomial", 5), "template": ("template", None), "none": ("none", None)}.items():
            _choose(qapp, combo, "chebyshev" if choice != "chebyshev" else "polynomial")
            assert _load(qapp, dialog, path, {"background": background}) == "ActionCompleted", background
            assert combo.currentData() == choice, background
            assert degree.isHidden() is (value is None), background
            if value is not None:
                assert degree.editor.value() == value, background
            expected = choice if value is None else f"{choice}{value}"
            assert dialog.vnext_fit_values()["background"] == expected, background
        before = dialog.vnext_fit_values()
        for background in ("polynomial8", "chebyshev16", "cubic", "spline4", "polynomial", 3):
            assert _load(qapp, dialog, path, {"background": background}) == "ActionRefused", background
            assert dialog.vnext_fit_values() == before, background
            assert "Background" in dialog.status.text(), dialog.status.text()
    finally:
        _close(qapp, dialog)


# ---- G5/G6 Phase Analysis's choices and its earlier files ------------------------------------------------------

def test_g5_phase_analysis_lists_polynomial_for_linear_and_no_separate_degree_row(qapp):
    dialog = _dialog("phase")
    try:
        row, combo, degree = _rows(dialog, "phase")
        assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == list(PHASE_CHOICES)
        assert combo.currentData() == "none" and degree.isHidden()
        assert _card(row) is dialog.background_card
        assert degree.parentWidget() is row, "no separate Degree row: the box is in the Fitted background row"
    finally:
        _close(qapp, dialog)


def test_g6_phase_analysis_loads_an_earlier_files_background_into_the_new_choices(qapp, tmp_path):
    dialog = _dialog("phase")
    path = tmp_path / "phase-analysis.json"
    try:
        _row, combo, degree = _rows(dialog, "phase")
        cases = (
            ({"fit_background": "linear", "fit_background_degree": 3}, ("polynomial", 1)),
            ({"fit_background": "template+linear", "fit_background_degree": 5}, ("template+polynomial", 1)),
            ({"fit_background": "spline", "fit_background_degree": 3}, ("spline", 4)),  # the four knots it fitted
            ({"fit_background": "chebyshev", "fit_background_degree": 5}, ("chebyshev", 5)),
            ({"fit_background_degree": 6, "fit_background": "polynomial"}, ("polynomial", 6)),  # either order
            ({"fit_background": "none", "fit_background_degree": 3}, ("none", None)),
        )
        for fields, (choice, value) in cases:
            _choose(qapp, combo, "chebyshev" if choice != "chebyshev" else "spline")
            assert _load(qapp, dialog, path, fields) == "ActionCompleted", (fields, dialog.status.text())
            assert combo.currentData() == choice, fields
            assert degree.isHidden() is (value is None), fields
            values = dialog.vnext_fit_values()
            assert values["fit_background"] == choice, fields
            assert values["fit_background_degree"] == (3 if value is None else value), fields
        before = dialog.vnext_fit_values()
        for fields in ({"fit_background": "polynomial", "fit_background_degree": 16},
                       {"fit_background": "spline", "fit_background_degree": 1},
                       {"fit_background": "cubic", "fit_background_degree": 3},
                       {"fit_background": "polynomial", "fit_background_degree": "two"}):
            assert _load(qapp, dialog, path, fields) == "ActionRefused", fields
            assert dialog.vnext_fit_values() == before, fields
    finally:
        _close(qapp, dialog)


# ---- G7/G8 notebooks exported with a degree open and run, viewer included ---------------------------------------

def _run_notebook(path, format):
    """Execute an exported notebook as a kernel does, its viewer cell included; its fit and its viewer."""
    if format == "jupyter":
        namespace = {}
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "code":
                exec(cell["source"], namespace)
        return namespace["outcome"], namespace.get("fit_viewer"), namespace
    spec = importlib.util.spec_from_file_location(f"bgdeg_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _outputs, definitions = module.app.run()
    return definitions["outcome"], None, definitions


@pytest.mark.parametrize("format", ("jupyter", "marimo"))
@pytest.mark.parametrize("choice, value, names", (
    ("polynomial", "3", {"bg_c0", "bg_c1", "bg_c2", "bg_c3"}),
    ("chebyshev", "6", {f"bg_c{index}" for index in range(7)}),
))
def test_g7_a_peak_notebook_exported_with_a_degree_opens_and_runs(qapp, tmp_path, format, choice, value, names):
    from xdart.gui.analysis.fit_export import write_fit_notebook
    dialog = _dialog("peak")
    viewer = None
    try:
        _row, combo, degree = _rows(dialog, "peak")
        trace, _anchor = _trace(*_peaks())
        dialog.set_vnext_trace(trace)
        _choose(qapp, combo, choice)
        _type(qapp, degree, value)
        settings = dialog.export_fit_values()
        assert settings["background"] == f"{choice}{value}"
        path = tmp_path / ("peak.ipynb" if format == "jupyter" else "peak.py")
        write_fit_notebook(path, dialog.export_snapshot(), "peak", settings)
        outcome, viewer, namespace = _run_notebook(path, format)
        assert outcome.ok and outcome.result.code == "OK", outcome.message
        assert {name for name in outcome.result.parameter_names if name.startswith("bg")} == names
        if format == "jupyter":
            assert json.loads(viewer.controls._displayed_editor.value)["background"] == f"{choice}{value}"
            viewer.controls._fit_button.click()
            assert viewer.outcome is not None and viewer.outcome.ok, viewer.outcome
            assert viewer.current_settings()["background"] == f"{choice}{value}"
        else:
            assert json.loads(namespace["settings_editor"].value)["background"] == f"{choice}{value}"
    finally:
        if viewer is not None:
            viewer.dispose()
        _close(qapp, dialog)


@pytest.mark.parametrize("format", ("jupyter", "marimo"))
def test_g8_a_phase_notebook_exported_with_polynomial_one_opens_and_runs(qapp, tmp_path, format):
    from tests.xdart.scattering.test_phase_reference_overlays import _cif
    from tests.xdart.scattering.test_phase_source_exports import _trace as phase_trace
    from xdart.gui.analysis.fit_export import write_fit_notebook
    dialog = _dialog("phase")
    viewer = None
    try:
        _row, combo, degree = _rows(dialog, "phase")
        dialog.phase_source.add_cif_paths((_cif(tmp_path),))
        dialog.set_vnext_trace(phase_trace(tmp_path))
        dialog._export_wavelength = 1.5406
        dialog.max_nfev.setValue(1)
        _choose(qapp, combo, "polynomial")
        assert degree.editor.value() == 1
        settings = dialog.export_fit_values()
        assert (settings["fit_background"], settings["fit_background_degree"]) == ("polynomial", 1)
        path = tmp_path / ("phase.ipynb" if format == "jupyter" else "phase.py")
        write_fit_notebook(path, dialog.export_snapshot(), "phase", settings)
        outcome, viewer, _namespace = _run_notebook(path, format)
        assert outcome.result is not None and outcome.result.code == "OK", outcome.message
        assert {name for name in outcome.result.parameter_names if name.startswith("bg_c")} == {"bg_c0", "bg_c1"}
        if viewer is not None:
            assert json.loads(viewer.controls._displayed_editor.value)["fit_background_degree"] == 1
    finally:
        if viewer is not None:
            viewer.dispose()
        _close(qapp, dialog)
