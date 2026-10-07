"""Real worker, numerical owner and artifact round trips for the native page."""

from dataclasses import replace
import json
import re
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import (
    StitchFrameSelector, StitchSourceSelection, StitchToolForm, prepare_stitch_tool,
)
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchGeometryKind, run_stitch_operation
from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xdart.gui.widgets.scientific_plot import scaled_intensity
from tests.core.test_stitch_operation import _form, _goniometer_record, _ring


class _Status:
    def show(self, text, timeout_ms=0):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(page, qapp, timeout=30):
    until = time.monotonic() + timeout
    qapp.processEvents()
    # a shown result the page dropped from memory is reloaded between its other work (C12)
    while (page._owner.busy or page._source_observer.computing or page._raw_pending or page._queued_jobs
           or getattr(page, "_reload_pending", None)) and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
    qapp.processEvents()
    assert not page._owner.busy, page.run_bar.readinessLabel.full_text()




def _as_native(form, scans="1"):
    """The same synthetic source as a native one-scan Spec selection."""
    return replace(form, source_selection=StitchSourceSelection(
        "Spec", (str(form.spec_path),), scans=scans, image_dir=str(form.image_dir),
        image_stem=form.image_stem), q_range=None, conditioning=StitchConditioning("auto"))


def _fill_native(page, form):
    page.load_form(_as_native(form))
    page.set_field("output_path", str(Path(form.output_path).parent))
    page.set_sources((form.spec_path,))
    page.set_field("radial_auto", True)
    page.set_field("threshold", "")


def test_native_preview_displays_conditional_affine_warning_and_keeps_run_available(qapp, tmp_path):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1")
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page.preview_button.click()
        _wait(page, qapp)
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        assert page._owner.prepared.warnings
        text = page.preview_text.toPlainText()
        assert "Warnings" in text
        assert "If del is mounted on nu" in text
        assert "stacked calibration" in text
        assert not list(Path(form.output_path).parent.glob("*.nexus"))
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("grouped", [False, True])
def test_metadata_shows_experimental_columns_and_separate_provenance(qapp, tmp_path, grouped):
    from xdart.gui.tabs.scattering.analysis_mount import MetadataResultDialog
    form = _form(tmp_path, "multigeometry")
    spec = Path(form.spec_path)
    content = spec.read_text().replace("17000.018", "12398.41984")
    content = content.replace("#N 3", "#N 4").replace("#L del  nu  energy", "#L del  nu  energy  diode")
    content = content.replace("14 -10 12398.41984\n", "14 -10 12398.41984 123\n")
    content = content.replace("15 -10 12398.41984\n", "15 -10 12398.41984 456\n")
    if grouped:
        content += "\n" + content[content.index("#S 1"):].replace("#S 1", "#S 2").replace(" 123\n", " 789\n")
        for index in range(2):
            image = Path(form.image_dir) / f"sample_scan1_{index:04d}.raw"
            image.with_name(f"sample_scan2_{index:04d}.raw").write_bytes(image.read_bytes())
    spec.write_text(content)
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        if grouped:
            page.set_field("image_stem", "sample_scan{scan}_")
            page.set_field("scan", "[1,2]")
            page.set_field("output_family", "combined")
        page.begin_preview(); _wait(page, qapp)
        assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click(); _wait(page, qapp)
        assert page._current_payload() is not None
        page.metadata_button.click(); _wait(page, qapp)
        dialog = page._metadata_dialog
        assert isinstance(dialog, MetadataResultDialog)
        assert dialog.source_selector.count() == (2 if grouped else 1)
        headers = [dialog.table.horizontalHeaderItem(col).text() for col in range(dialog.table.columnCount())]
        diode = headers.index("diode")
        assert dialog.table.rowCount() == 2
        assert float(dialog.table.item(0, diode).text()) == 123
        assert dialog.table.verticalHeaderItem(0).text() == ("1-0" if grouped else "0")
        if grouped:
            dialog.source_selector.setCurrentIndex(1)
            assert float(dialog.table.item(0, diode).text()) == 789
            assert dialog.table.verticalHeaderItem(0).text() == "2-0"
        dialog.provenance_button.click()
        assert json.loads(dialog.provenance_text.toPlainText()) == json.loads(page._current_payload().provenance_json)
        assert dialog.provenance_text.isReadOnly()
        dialog.close()
        # Reopening does not pass stale cached metadata off as saved experimental data.
        spec.write_text(content + "\n#C changed since stitching\n")
        page.metadata_button.click(); _wait(page, qapp)
        assert page._metadata_dialog.table.rowCount() == 0
        assert "changed" in page._metadata_dialog.status.text().lower()
        assert page._metadata_dialog.provenance_button.isEnabled()
    finally:
        until = time.monotonic() + 5
        while page.shutdown().status is PageCleanup.PENDING and time.monotonic() < until:
            qapp.processEvents(); time.sleep(.01)


def test_native_source_controls_ranges_and_threshold_semantics(qapp, tmp_path):
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, _form(tmp_path, "multigeometry"))
        assert page.build_form().q_range is None
        assert page.build_form().conditioning.mode == "auto"
        assert not page.radial_range._toggle[1].isChecked()
        page.radial_range._toggle[1].click()
        assert page.build_form().q_range is not None
        page.set_field("unit", "2θ (°)")
        assert page.build_form().unit == "2th_deg"
        assert page.radial_range.label.text() == "2θ (°)"
        page.set_field("threshold", "125")
        page.set_field("threshold_min", "4")
        policy = page.build_form().conditioning
        assert (policy.mode, policy.threshold_min, policy.threshold_max) == ("manual", 4., 125.)
        page.threshold_range._toggle[1].click()
        assert page.build_form().conditioning.mode == "off"
        page.threshold_range._toggle[1].click()
        assert page.build_form().conditioning.mode == "manual"
        page.set_field("threshold", "")
        assert page.build_form().conditioning.mode == "auto"
        assert page._value("threshold_min") == 0.
        page.set_field("scan", "1,2,[2-5,8]")
        assert page.build_form().source_selection.scans == "1,2,[2-5,8]"
        page.set_field("output_family", "combined")
        assert page.build_form().output_family == "combined"
        page.set_field("output_family", ";;combined")
        assert page.build_form().output_families == ("", "", "combined")
        assert page.build_form().output_family is None
        restored = page.build_form()
        page.load_form(restored)
        assert page.build_form().fingerprint == restored.fingerprint
        page.set_field("scan_mode", "Stitch All")
        assert page.build_form().source_selection.stitch_all
        assert page._rows["scan"].isHidden()
        page.set_field("source_kind", "NeXus")
        files = (str(tmp_path / "one.nxs"), str(tmp_path / "two.nxs"))
        page.set_sources(files)
        assert page.build_form().source_selection.paths == files
        assert page._rows["image_dir"].isHidden()
        page.set_field("source_kind", "Directory")
        page.set_sources((str(tmp_path),))
        assert page.build_form().source_selection.kind == "Directory"
        assert page._rows["scan_mode"].isHidden()
    finally:
        page.shutdown()


def test_grouped_metadata_plot_uses_only_displayed_contributors(qapp, tmp_path):
    from xdart.gui.analysis.scan_plot_dialog import ScanPlotDialog
    form = replace(_form(tmp_path, "multigeometry"),
                   frame_selector=StitchFrameSelector(0, 1))
    spec = Path(form.spec_path)
    content = spec.read_text().replace("17000.018", "12398.41984")
    content = content.replace("del 14 15 1", "del 14 16 2")
    content = content.replace("#N 3", "#N 4").replace("#L del  nu  energy", "#L del  nu  energy  diode")
    content = content.replace("14 -10 12398.41984\n", "14 -10 12398.41984 123\n")
    content = content.replace("15 -10 12398.41984\n", "15 -10 12398.41984 456\n")
    content += "16 -10 12398.41984 999\n"
    second = content[content.index("#S 1"):].replace("#S 1", "#S 2").replace(" 123\n", " 789\n")
    spec.write_text(content + "\n" + second)
    images = Path(form.image_dir)
    (images / "sample_scan1_0002.raw").write_bytes((images / "sample_scan1_0001.raw").read_bytes())
    for index in range(3):
        image = images / f"sample_scan1_{index:04d}.raw"
        image.with_name(f"sample_scan2_{index:04d}.raw").write_bytes(image.read_bytes())
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("image_stem", "sample_scan{scan}_")
        page.set_field("scan", "[1,2]")
        page.set_field("output_family", "combined")
        page.begin_preview(); _wait(page, qapp)
        page.run_bar.startButton.click(); _wait(page, qapp)
        assert page._current_payload() is not None
        page.metadata_button.click()
        popup = page._metadata_dialog
        assert not popup.plot_button.isEnabled()
        _wait(page, qapp)
        assert popup.table.rowCount() == 2
        popup.plot_button.click()
        plot = page._metadata_plot_dialog
        assert isinstance(plot, ScanPlotDialog) and plot.parentWidget() is page
        np.testing.assert_array_equal(plot._table["diode"], [123, 456])
        popup.source_selector.setCurrentIndex(1)
        popup.plot_button.click()
        assert page._metadata_plot_dialog is plot
        np.testing.assert_array_equal(plot._table["diode"], [789, 456])
        plot._write_csv(tmp_path / "metadata.csv")
        import csv
        with (tmp_path / "metadata.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [float(row["diode"]) for row in rows] == [789, 456]
        popup.close(); qapp.processEvents()
        assert plot.isVisible()
        plot.norm_combo.setCurrentText("diode")
        assert plot.isVisible() and len(plot._table["diode"]) == 2
        spec.write_text(content + "\n" + second + "\n#C changed\n")
        page.metadata_button.click(); _wait(page, qapp)
        assert not page._metadata_dialog.plot_button.isEnabled()
        assert "changed" in page._metadata_dialog.status.text().lower()
        page.shutdown(); qapp.processEvents()
        assert page._metadata_plot_dialog is None
    finally:
        until = time.monotonic() + 5
        while page.shutdown().status is PageCleanup.PENDING and time.monotonic() < until:
            qapp.processEvents(); time.sleep(.01)


def _commit_source_box(page, qapp, text, *, key="Tab"):
    """Edit the source box as a user does: select all, type (or Backspace), commit with Tab/Return."""
    from PySide6.QtTest import QTest
    from pyqtgraph.Qt import QtCore
    editor = page._rows["spec_path"].editor
    editor.setFocus()
    qapp.processEvents()
    editor.selectAll()
    if text:
        QTest.keyClicks(editor, text)
    elif text == "":
        QTest.keyClick(editor, QtCore.Qt.Key.Key_Backspace)
    QTest.keyClick(editor, getattr(QtCore.Qt.Key, f"Key_{key}"))
    qapp.processEvents()


def test_clearing_the_source_box_clears_the_selection_and_refuses_run(qapp, tmp_path):
    """Codex R19-F1: an emptied source box kept the previous source, which Run, Preview and the
    saved settings still used, with Run ready. Clearing it empties the selection (SPEC, NeXus,
    Directory), drops a prepared plan and the tooltip; a typed path is the selection at once."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    page.show()
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        page.preview_button.click()
        _wait(page, qapp)
        assert page._prepared_revision == page._revision and page.run_bar.startButton.isEnabled()
        page.preview_dialog.close()
        page.activateWindow()
        spec, row = page._source_paths, page._rows["spec_path"]
        _commit_source_box(page, qapp, "")
        assert row.current_value() == "" and row.editor.text() == ""
        assert page._source_paths == () and "Spec" not in page.settings()["sources"]
        assert page._prepared_revision is None and not page._queued_jobs
        assert not page.run_bar.startButton.isEnabled()
        assert page.run_bar.readinessLabel.full_text() == "Needs SPEC files"
        assert all(not widget.toolTip() for widget in (row, row.editor, row.label))
        page.preview_button.click()  # nothing is checked without a source
        assert "source path" in page.run_bar.readinessLabel.full_text() and not page._owner.busy
        # a typed path is the selection, and the readiness line follows it at once
        _commit_source_box(page, qapp, spec[0], key="Return")
        _wait(page, qapp)
        assert page._source_paths == spec and page.settings()["sources"]["Spec"] == list(spec)
        assert (page.run_bar.readinessLabel.full_text()  # with its frames, as Int's (maintainer, 2026-09-30)
                == "sample, scan 1 · 2 frames · pyFAI MultiGeometry")
        assert page.run_bar.startButton.isEnabled() and row.toolTip() == spec[0]
        for kind, paths in (("NeXus", (str(tmp_path / "one.nxs"), str(tmp_path / "two.nxs"))),
                            ("Directory", (str(tmp_path),))):
            page.set_field("source_kind", kind)
            page.set_sources(paths)
            _commit_source_box(page, qapp, None, key="Return")  # its own display is not a path
            assert page.build_form().source_selection.paths == paths
            _commit_source_box(page, qapp, "")
            assert page._source_paths == () and kind not in page.settings()["sources"]
            with pytest.raises(ValueError, match="source path"):
                page.build_form()
            assert not page.run_bar.startButton.isEnabled()
    finally:
        page.shutdown()


def test_run_right_after_typing_in_a_field_checks_and_runs(qapp, tmp_path):
    """Live: typing a scan into Selection and clicking Run (on macOS the click leaves the focus in
    the field) came back "Inputs changed while checking · Run again": disabling the controls for
    the check committed the typing mid-check. The typing now commits before the check starts."""
    from PySide6.QtTest import QTest
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    page.show()
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        page.run_bar.writeModeButton.click()  # Replace: the same inputs stitch again (Append skips them)
        page.activateWindow()
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page.run_bar.readinessLabel.full_text().startswith("Complete"), page.run_bar.readinessLabel.full_text()
        editor = page._rows["scan"].editor
        editor.setFocus()
        qapp.processEvents()
        editor.selectAll()
        QTest.keyClicks(editor, "1")  # typed, not committed: no Return or Tab
        assert editor.hasFocus()
        page.run_bar.startButton.click()  # a click that leaves the focus in the field
        _wait(page, qapp)
        assert page.run_bar.readinessLabel.full_text().startswith("Complete"), page.run_bar.readinessLabel.full_text()
        assert page.build_form().source_selection.scans == "1"
    finally:
        page.shutdown()


@pytest.mark.parametrize("dtype", ("int32", "uint32"))
def test_native_minimum_only_threshold_edit_excludes_native_ceiling(qapp, tmp_path, dtype):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    from xrd_tools.integrate.stitch_conditioning import condition_stitch_frame
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), raw_dtype=dtype)
    for path in Path(form.image_dir).glob("*.raw"):
        np.fromfile(path, dtype=np.int32).astype(dtype).tofile(path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        page.begin_preview()
        _wait(page, qapp)
        assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
        ceiling = np.iinfo(dtype).max
        # Decision 39: the PILATUS calibration's counter limit is Auto's ceiling, shown as a hint only
        assert page._value("threshold") is None
        assert page.threshold_range._high.placeholderText() == "Auto · 1048575 (saturation)"
        page.set_field("threshold_min", "4")
        policy = page.build_form().conditioning
        assert policy == StitchConditioning("manual", 4.0, 1048574.0)  # Auto's kept values, from 4
        raw = np.array([[3, 4, 1048574, 1048575, ceiling - 1, ceiling]], dtype=dtype)
        original = raw.copy()
        actual = condition_stitch_frame(raw, policy)
        np.testing.assert_array_equal(np.isfinite(actual), [[False, True, True, False, False, False]])
        np.testing.assert_array_equal(raw, original)
    finally:
        page.shutdown()


def test_native_saved_frame_popup_keeps_curve_and_shows_thumbnail_fallback(qapp, tmp_path):
    from tests.core.test_analysis_stitch_auxiliary import _publish
    path, raw, project, _receipt = _publish(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        assert [page.frames.item(i).text() for i in range(page.frames.count())] == ["Full", "2-0", "2-1", "3-0"]
        original_y = page.curve.listDataItems()[0].yData.copy()
        page.scientific.plot_axis.setCurrentIndex(page.scientific.plot_axis.findData("2theta"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, original_y)
        assert page.curve.getPlotItem().getAxis("bottom").labelText == "2θ"
        page._show_raw()
        _wait(page, qapp)
        assert page._raw_frame_view.thumbnail is not None
        page.scientific.raw_popup_full.click()
        _wait(page, qapp)
        np.testing.assert_array_equal(page._raw_frame_view.raw, raw)
        page.frames.setCurrentRow(3)
        assert page.scientific.raw_popup_image.canvas.raw_image.size == 0
        assert page.scientific.raw_popup_status.text() == "Loading selected frame…"
        _wait(page, qapp)
        assert "3-0" in page.scientific.raw_popup_status.text()
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, original_y)
        project.rename(tmp_path / "unavailable")
        page.frames.setCurrentRow(2)
        _wait(page, qapp)
        assert page._raw_frame_view.raw is None
        assert "Thumbnail · raw source unavailable" in page.scientific.raw_popup_status.text()
        page.scientific.raw_popup_dialog.close()
        assert page._raw_frame_view is None
        assert page.scientific.raw_popup_image.canvas.raw_image.size == 0
    finally:
        page.shutdown()


def _stitched_variants(tmp_path):
    """Four Stitch 1D results of the fixture's scan: a; b and d with other polarization factors (a's grid);
    c with 80 points (another grid). Returns their paths and payloads."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1")
    paths, payloads = [], []
    for name, options in (("a", {}), ("b", {"polarization_factor": .5}), ("c", {"npt_1d": 80}),
                          ("d", {"polarization_factor": .7})):
        (tmp_path / name).mkdir()
        request = prepare_stitch_tool(replace(form, output_path=tmp_path / name / f"{name}_stitch1d.nexus",
                                              **options)).request
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        paths.append(request.module.output.target)
        payloads.append(result.payload)
    return paths, payloads


def test_set_bg_subtracts_the_selected_results_mean_until_cleared_or_another_selection(qapp, tmp_path):
    """Maintainer, 2026-09-30: Int's plot-only Set BG in Stitch ("would the plot only -set as background like in
    Int work here easily?" -- "Yes"): the selected results' mean subtracted from each shown; Clear BG, or
    another selection, drops it (as Int's at every boundary); the records are untouched."""
    paths, payloads = _stitched_variants(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        for path in paths:
            page.open_result(path)
            _wait(page, qapp)
        view, button = page.scientific, page.scientific.background
        view.plot_mode.setCurrentText("Overlay")
        page.results.setCurrentRow(0)
        page.results.clearSelection()
        for row in (0, 1):  # a and b
            page.results.item(row).setSelected(True)
        _wait(page, qapp)
        assert not button.isHidden() and button.text() == "Set 1D BG"
        raw = [np.asarray(payload.intensity, dtype=np.float64) for payload in payloads[:2]]
        first = lambda: np.asarray(page.curve.listDataItems()[0].yData, dtype=np.float64)  # noqa: E731 (Overlay
        # stacks the others with an offset)
        button.click()
        assert button.text() == "Clear 1D BG", page.run_bar.readinessLabel.full_text()
        domain, rows, _key = page._background.projection()  # what the view is given
        mean = (raw[0] + raw[1]) / 2
        assert domain == "integrated_1d" and len(rows) == 2
        subtracted = sorted((values for _frame, values in rows), key=lambda values: values[0])
        for values, expected in zip(subtracted, sorted((row - mean for row in raw), key=lambda row: row[0])):
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9 * np.abs(mean).max())
        assert any(np.array_equal(first(), values) for values in subtracted)  # and shows them
        button.click()  # Clear BG
        assert button.text() == "Set 1D BG" and page._background.projection() is None
        assert any(np.array_equal(first(), row) for row in raw)
        button.click()
        assert button.text() == "Clear 1D BG"
        page.results.item(2).setSelected(True)  # another selection releases it
        _wait(page, qapp)
        assert button.text() == "Set 1D BG" and page._background.phase in {"EMPTY", "RELEASED"}
        stored = read_analysis_artifact(paths[0], expected_kind=AnalysisArtifactKind.STITCH_1D)
        np.testing.assert_array_equal(stored.intensity, payloads[0].intensity)  # display only
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()


def test_average_sum_and_waterfall_show_the_selected_results_as_ints(qapp, tmp_path):
    """Maintainer, 2026-09-30: Average/Sum/Waterfall in Stitch ("this should be straighforward"): Int's view
    modes on the selected results. Average and Sum fold results on one grid; on several they say so (Int's
    view then shows the first), while Waterfall takes each on the first's axis from the fourth result on, as
    Int's."""
    paths, payloads = _stitched_variants(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        for path in paths:
            page.open_result(path)
            _wait(page, qapp)
        view = page.scientific
        view.plot_mode.setCurrentText("Overlay")
        page.results.clearSelection()
        for row in (0, 1):  # a and b: one grid
            page.results.item(row).setSelected(True)
        _wait(page, qapp)
        first, second = payloads[0].intensity, payloads[1].intensity
        assert not np.array_equal(first, second)  # b's polarization factor changed it
        view.plot_mode.setCurrentText("Average")
        _wait(page, qapp)
        (curve,) = page.curve.listDataItems()
        both = np.stack([first, second]).astype(np.float64)  # the record's float32, folded as the view does
        np.testing.assert_allclose(curve.yData, np.nanmean(both, axis=0), rtol=1e-6)
        view.plot_mode.setCurrentText("Sum")
        _wait(page, qapp)
        (curve,) = page.curve.listDataItems()
        np.testing.assert_allclose(curve.yData, np.nansum(both, axis=0), rtol=1e-6)
        assert "one grid" not in view.status.text()
        for row in (2, 3):  # c (80 points) and d
            page.results.item(row).setSelected(True)
        view.plot_mode.setCurrentText("Waterfall")
        _wait(page, qapp)
        assert view.bottom_stack.currentWidget() is view.waterfall
        # Maintainer, 2026-09-30: "can this be changed and made more flexible, at least for sum? This is only
        # for display" -- "bin them together on a common grid": on one grid spanning them, each interpolated
        curves = [(np.asarray(payload.axes[0][1], dtype=float), np.asarray(payload.intensity, dtype=float))
                  for payload in payloads]
        for mode in ("Average", "Sum"):
            view.plot_mode.setCurrentText(mode)
            _wait(page, qapp)
            (curve,) = page.curve.listDataItems()
            x = np.asarray(curve.xData, dtype=float)
            assert x.size == max(radial.size for radial, _values in curves)  # as fine as the finest
            rows = np.stack([np.interp(x, radial, values, left=np.nan, right=np.nan) for radial, values in curves])
            expected = np.nanmean(rows, axis=0) if mode == "Average" else np.nansum(rows, axis=0)
            np.testing.assert_allclose(curve.yData, expected, rtol=1e-6)
            assert "one axis" not in view.status.text()
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_native_preview_run_repeat_reopen_and_overlay_use_real_owners(qapp, tmp_path, backend):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, backend, scans="1")
    (tmp_path / "reference").mkdir()
    reference_request = prepare_stitch_tool(
        replace(form, output_path=tmp_path / "reference" / "reference_stitch1d.nexus")).request
    reference = run_stitch_operation(reference_request)
    assert reference.terminal.disposition is ModuleDisposition.COMMITTED
    page = StitchPage(empty_host_services(_Status()))
    reopened = None
    try:
        page.load_form(form)
        page.preview_button.click()
        _wait(page, qapp)
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        target = Path(page._owner.prepared.request.module.output.target)
        shown, summary = [], page.run_bar.set_readiness_summary
        page.run_bar.set_readiness_summary = lambda text, **k: (shown.append(str(text)), summary(text, **k))[1]
        page.run_bar.startButton.click()
        _wait(page, qapp)
        page.run_bar.set_readiness_summary = summary
        # the run bar says what it is doing, frames while stitching; never the raw stage names
        assert any(text.startswith("Stitching · frame ") for text in shown), shown
        assert not any("science" in text or re.fullmatch(r"[a-z]+ · \d+/\d+", text) for text in shown), shown
        assert page._prepared_revision is None  # a run uses its plan once: the next Run checks again
        payload = read_analysis_artifact(target, expected_kind=AnalysisArtifactKind.STITCH_1D)
        for name in ("intensity", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(payload, name), getattr(reference.payload, name))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, payload.axis("q"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.intensity)
        assert page.results.count() == 1

        page.preview_dialog.close()
        page.set_field("npt_1d", "80")  # changed inputs: Run checks them itself, as Int's Run
        assert page._prepared_revision is None
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click()
        _wait(page, qapp)
        replacement = read_analysis_artifact(target)
        assert replacement.intensity.shape == (80,)
        assert page.results.count() == 1
        assert not page.preview_dialog.isVisible()  # Preview stays optional
        assert page.run_bar.readinessLabel.full_text().startswith("Complete · ")
        assert page.shutdown().status is PageCleanup.CLEAN
        (tmp_path / "source").rename(tmp_path / "source-unavailable")

        reopened = StitchPage(empty_host_services(_Status()))
        reopened.open_result(target)
        _wait(reopened, qapp)
        assert reopened._owner.form is None
        assert reopened._owner.prepared is None
        assert not reopened.run_bar.startButton.isEnabled()
        np.testing.assert_array_equal(reopened.curve.listDataItems()[0].yData, replacement.intensity)
        reopened.open_result(reference_request.module.output.target)
        _wait(reopened, qapp)
        reopened.scientific.plot_mode.setCurrentText("Overlay")
        reopened.results.selectAll()
        _wait(reopened, qapp)  # the result Single let go is read back (C12)
        assert len(reopened.curve.listDataItems()) == 2
        assert reopened.curve.listDataItems()[0].opts["pen"].color().getRgb() == (31, 119, 180, 255)
        reopened.scientific.previous_frame.click()
        assert reopened.results.currentRow() == 0
        assert len(reopened.results.selectedItems()) == 2
        assert len(reopened.curve.listDataItems()) == 2
        reopened.scientific.next_frame.click()
        assert reopened.results.currentRow() == 1
        assert len(reopened.results.selectedItems()) == 2
        reopened.scientific.color_map.setCurrentText("viridis")
        reopened.scientific.log_scale.click()  # as on Int: the image and the curve's intensity scale
        assert reopened._view.log_scale is True and reopened._view.color_map == "viridis"
        assert reopened._view.plot_options.intensity_scale == "Log"
        assert len(reopened.curve.listDataItems()) == 2
        options = reopened.scientific.plot_options_dialog  # Int's dialog, with Stitch's Diagnostic
        assert options.isAncestorOf(reopened._diagnostic_combo)
        reopened.scientific.plot_mode.setCurrentText("Single")
        shown = reopened._current_payload()
        np.testing.assert_array_equal(reopened.curve.listDataItems()[0].yData,
                                      scaled_intensity(shown.intensity, "Log"))
        reopened._diagnostic_combo.setCurrentText("Coverage")  # Stitch's row and a view option,
        options.intensity_scale.setCurrentText("Sqrt")  # accepted together
        options.accept_button.click()
        assert reopened._diagnostic == "Coverage"  # the plot follows, not a cached trace
        assert reopened._view.plot_options.intensity_scale == "Sqrt"
        assert reopened._diagnostic_combo.currentText() == "Coverage"
        np.testing.assert_array_equal(reopened.curve.listDataItems()[0].yData,
                                      scaled_intensity(shown.coverage, "Sqrt"))
        reopened._diagnostic_combo.setCurrentText("Normalization")
        options.cancel_button.click()  # Cancel keeps the accepted Diagnostic
        assert reopened._diagnostic == "Coverage" and reopened._diagnostic_combo.currentText() == "Coverage"
        reopened._diagnostic_combo.setCurrentText("Intensity")
        options.intensity_scale.setCurrentText("Linear")
        options.accept_button.click()
        np.testing.assert_array_equal(reopened.curve.listDataItems()[0].yData, shown.intensity)
        reopened.scientific.plot_mode.setCurrentText("Single")
        assert len(reopened.results.selectedItems()) == 1
        assert len(reopened.curve.listDataItems()) == 1
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        if reopened is not None:
            assert reopened.shutdown().status is PageCleanup.CLEAN
            reopened.deleteLater()
        qapp.processEvents()


def test_reopen_corrupt_file_cannot_grant_run_or_replace_display(qapp, tmp_path):
    malformed = tmp_path / "corrupt.nexus"
    malformed.write_text("not a NeXus artifact")
    before = malformed.read_bytes()
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(malformed)
        _wait(page, qapp)
        assert page.results.count() == 0
        assert not page.run_bar.startButton.isEnabled()
        assert malformed.read_bytes() == before
        assert page._owner.prepared is None
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_selecting_existing_result_fences_pending_reopen(qapp, tmp_path, monkeypatch):
    from threading import Event
    from tests.core.test_analysis_stitch_auxiliary import _publish
    from xdart.gui.tools import stitch_owner
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first, *_ = _publish(first_dir, target=tmp_path / "first.nexus")
    second, *_ = _publish(second_dir, target=tmp_path / "second.nexus")
    page = StitchPage(empty_host_services(_Status()))
    entered, release = Event(), Event()
    original = stitch_owner.read_analysis_artifact
    def select(path):
        page.results.setCurrentItem(next(page.results.item(index) for index in range(page.results.count())
            if page.results.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == str(path)))
    try:
        page.open_result(first)
        _wait(page, qapp)
        page.open_result(second)
        _wait(page, qapp)
        expected_y = page.curve.listDataItems()[0].yData.copy()  # the second result's
        select(first)

        def delayed(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        monkeypatch.setattr(stitch_owner, "read_analysis_artifact", delayed)
        page.open_result(first)
        assert entered.wait(5)
        select(second)  # read back once the pending reopen is done (C12)
        release.set()
        _wait(page, qapp)
        assert page._current_payload().inspection.path == str(second)
        assert page.browser.current_result() == str(second)
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, expected_y)
    finally:
        release.set()
        _wait(page, qapp)
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_native_xu_displays_admitted_mapping_and_rotation_read_only(qapp, tmp_path):
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.xu_stitch_calibration import authored_surface_calibration
    form = _form(tmp_path, "xu_hist")
    asset = authored_surface_calibration(detector=detector_factory("Pilatus300kw"),
        cch1=220., cch2=235., distance_m=.395, del_offset_deg=3., nu_offset_deg=1.,
        energy_eV=17000.018, xu_wavelength_A=12398.41984 / 17000.018,
        source_motors={"del": "gamma", "nu": "delta"}, rotation=180,
        control_domain={"del": [5., 40.], "nu": [-10., 10.]},
        domain={"del": [0., 50.], "nu": [-15., 30.]},
        rms_q=.001, condition=100., n_frames=4, n_points=200)
    asset_path = tmp_path / "authored.json"
    asset_path.write_text(json.dumps(asset, sort_keys=True, separators=(",", ":")))
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("geometry_path", str(asset_path))
        assert page._value("source_motors") == "del=gamma, nu=delta"
        assert page._value("camera") == "x+, z-"  # the asset's rotation of 180°: its unmirrored camera
        for key in ("source_motors", "camera", "geometry_kind", "poni_references"):
            assert not page._rows[key].isEnabled()
        # Raw decoding is still an actual input, independent of the fitted asset.
        assert page._rows["raw_dtype"].isEnabled()
        page.set_field("geometry_path", "")
        assert page._value("source_motors") == ""
        assert page._value("camera") == ""
        page._calibrate()
        assert page._calibration_dialog.isVisible()
        page._calibration_dialog.close()
        page.set_field("backend", "pyFAI MultiGeometry")
        assert page._rows["source_motors"].isEnabled()
        assert page._rows["camera"].isEnabled()
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_stop_while_run_checks_the_inputs_stitches_nothing(qapp, tmp_path):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert page.results.count() == 1, page.run_bar.readinessLabel.full_text()
        target = Path(page._owner.prepared.request.module.output.target)
        before = target.read_bytes()
        page.set_field("npt_1d", "80")
        page.run_bar.startButton.click()
        assert page._owner.busy and page.run_bar.stopButton.isEnabled()  # offered while checking
        page.stop()  # the port the host's Run → Stop (Ctrl+Shift+C) calls
        assert not page.run_bar.stopButton.isEnabled()
        _wait(page, qapp)
        assert page.run_bar.readinessLabel.full_text() == "Stopped before stitching"
        assert page._owner.prepared is not None  # the check itself finished
        assert target.read_bytes() == before and page.results.count() == 1  # nothing replaced
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_run_checks_the_inputs_itself_and_stops_when_nothing_is_left_to_stitch(qapp, tmp_path):
    form = _form(tmp_path, "multigeometry")
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("image_stem", "missing_")  # scan 1 has no images under this stem
        _wait(page, qapp)
        assert not page.run_bar.startButton.isEnabled()
        assert "No detector images named *missing_*" in page.run_bar.readinessLabel.full_text()
        page.begin_run()  # direct API admission keeps the exact no-output diagnostic too
        _wait(page, qapp)
        assert page.run_bar.readinessLabel.full_text() == "Nothing to stitch · 1 scan skipped (Preview lists them)"
        assert page._owner.prepared is not None and not page._owner.prepared.requests
        assert page.results.count() == 0 and not page.preview_dialog.isVisible()
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_native_group_queue_skips_unimaged_scan_and_stops_before_next_output(qapp, tmp_path, backend):
    form = _form(tmp_path, backend)
    spec = Path(form.spec_path)
    spec.write_text(spec.read_text() +
        "\n#S 2 ascan del 14 15 1 1\n#D today\n#P0 14 -10 17000.018\n"
        "#N 3\n#L del  nu  energy\n14 -10 17000.018\n15 -10 17000.018\n"
        "\n#S 3 ascan del 14 15 1 1\n#D today\n#P0 14 -10 17000.018\n"
        "#N 3\n#L del  nu  energy\n14 -10 17000.018\n15 -10 17000.018\n")
    if backend == "multigeometry":
        spec.write_text(spec.read_text().replace("17000.018", "12398.41984"))
    for index in range(2):
        source = Path(form.image_dir) / f"sample_scan1_{index:04d}.raw"
        (Path(form.image_dir) / f"sample_scan2_{index:04d}.raw").write_bytes(source.read_bytes())
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("image_stem", "sample_scan{scan}_")
        page.set_field("scan", "1,[1,2],3")
        page.set_field("output_family", "combined")
        page.set_field("unit", "2θ (°)")
        page.begin_preview()
        _wait(page, qapp)
        assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
        requests = page._owner.prepared.requests
        assert len(requests) == 2
        assert len(page._owner.prepared.skipped) == 1
        assert "No detector images" in page.preview_text.toPlainText()
        status = page.run_bar.readinessLabel.full_text()  # skips and warnings stay on the status line
        assert "· 1 scan skipped" in status and status.endswith("(Preview lists them)")
        # the calibration's PILATUS counter limit, a hint and never the field's value (decision 39)
        assert page._value("threshold") is None and page.build_form().threshold is None
        assert page.threshold_range._high.placeholderText() == "Auto · 1048575 (saturation)"
        assert page.build_form().conditioning == StitchConditioning("auto")
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert page.results.count() == 2, page.run_bar.readinessLabel.full_text()
        status = page.run_bar.readinessLabel.full_text()  # skips and warnings stay on the status line
        assert "· 1 scan skipped" in status and status.endswith("(Preview lists them)")
        assert not page._queued_jobs
        grouped = read_analysis_artifact(requests[1].module.output.target)
        assert grouped.native_unit == "2th_deg"
        assert len(grouped.contributing_frames) == 4
        assert [page.frames.item(i).text() for i in range(page.frames.count())] == ["Full", "1-0", "1-1", "2-0", "2-1"]
        assert page.build_form().unit == "2th_deg"
        assert page.scientific.plot_axis.currentData() == "Q"
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, grouped.intensity)
        page.scientific.plot_axis.setCurrentIndex(page.scientific.plot_axis.findData("2theta"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, grouped.radial_axis("2th_deg"))
        page.set_field("threshold_min", "4")
        assert page.build_form().conditioning == StitchConditioning("manual", 4., 1048574.)  # Auto's kept values
        page.set_field("output_path", str(tmp_path / "cancelled"))
        page.begin_preview()
        _wait(page, qapp)
        second_target = Path(page._owner.prepared.requests[1].module.output.target)
        page.begin_run()
        page.stop()
        _wait(page, qapp)
        assert not page._queued_jobs
        assert not second_target.exists()
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("mode", ("1d", "2d"))
def test_append_on_a_blank_page_displays_the_existing_result(qapp, tmp_path, mode):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), mode=mode,
                   npt_rad_2d=32, npt_azim_2d=24)
    request = prepare_stitch_tool(form).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    target = Path(request.module.output.target)
    before = (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes())
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        _wait(page, qapp)  # Source discovery must finish before the real Run button is enabled.
        assert page._current_payload() is None
        assert page.run_bar.startButton.isEnabled()
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        payload = page._current_payload()
        assert payload is not None, (page.run_bar.readinessLabel.full_text(),
                                    page._action_view_revision, page._view_revision)
        assert payload.result_fingerprint == result.payload.result_fingerprint
        assert page.frames.count() > 0 and page.curve.listDataItems()
        assert page.run_bar.readinessLabel.full_text().startswith("Up to date")
        assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == before
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("mode,unit,prefix", (("1d", "q_A^-1", "iq"), ("2d", "q_A^-1", "iq"),
                                            ("1d", "2th_deg", "itth"), ("2d", "2th_deg", "itth")))
def test_stitch_run_exports_stored_profile_and_empty_click_keeps_it(qapp, tmp_path, mode, unit, prefix):
    from tests.xdart.tools.test_stitch_native_values import _native_form
    from xrd_tools.io.export import read_xye
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), mode=mode, unit=unit,
                   npt_rad_2d=32, npt_azim_2d=24)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.resize(1400, 850)
        page.show()
        page.load_form(form)
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        payload = page._current_payload()
        assert payload is not None, page.run_bar.readinessLabel.full_text()
        target = Path(payload.inspection.path)
        xye = target.with_name(f"{prefix}_{target.stem}.xye")
        assert xye.is_file(), page.run_bar.readinessLabel.full_text()
        # Ordinary browsing remains read-only; only Run/Append publishes companions.
        xye_before = xye.stat().st_ino, xye.stat().st_mtime_ns, xye.read_bytes()
        page.open_result(target)
        _wait(page, qapp)
        assert (xye.stat().st_ino, xye.stat().st_mtime_ns, xye.read_bytes()) == xye_before
        profile = payload if mode == "1d" else payload.profile
        x, y, sigma = read_xye(xye)
        radial = profile.axes[0][1]
        for actual, expected in ((x, radial), (y, profile.intensity),
                                 (sigma, np.sqrt(np.abs(np.asarray(profile.intensity, dtype=float)))
                                  if profile.sigma is None else profile.sigma)):
            np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=0, equal_nan=True)
        nexus_before = target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()
        curve_before = [(item.xData.copy(), item.yData.copy()) for item in page.curve.listDataItems()]
        assert curve_before
        chosen = page.browser.current_result(), page.browser.selected_results()
        blank = QtCore.QPoint(15, page.results.viewport().height() - 10)
        assert not page.results.indexAt(blank).isValid()
        QtTest.QTest.mouseClick(page.results.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=blank)
        qapp.processEvents()
        assert (page.browser.current_result(), page.browser.selected_results()) == chosen
        after = [(item.xData, item.yData) for item in page.curve.listDataItems()]
        assert len(after) == len(curve_before)
        for actual, expected in zip(after, curve_before, strict=True):
            for values, prior in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(values, prior)
        xye.unlink()  # Append backfills the export without repeating scientific publication.
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert xye.is_file() and page.run_bar.readinessLabel.full_text().startswith("Up to date")
        assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == nexus_before
        xye.unlink()
        xye.mkdir()  # A real export failure keeps the committed science/display and reports it.
        page.run_bar.startButton.click()
        _wait(page, qapp)
        assert "XYE_EXPORT_FAILED" in page.run_bar.readinessLabel.full_text()
        assert page._current_payload() is not None
        assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == nexus_before
        xye.rmdir()
        if mode == "1d" and unit == "q_A^-1":
            import os
            os.link(target, xye)
            page.run_bar.startButton.click()
            _wait(page, qapp)
            assert "XYE_EXPORT_FAILED" in page.run_bar.readinessLabel.full_text()
            assert (target.stat().st_ino, target.stat().st_mtime_ns, target.read_bytes()) == nexus_before
            assert os.path.samefile(target, xye)
            xye.unlink()
    finally:
        _wait(page, qapp)
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_append_skips_the_outputs_already_stitched_and_replace_stitches_them_again(qapp, tmp_path):
    """Maintainer, 2026-09-29: "the append/overwrite button was disabled. all this should do is skip if
    something has already been processed with the same settings as before, else overwrite"."""
    import os
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1,2")
    page = StitchPage(empty_host_services(_Status()))
    summary = page.run_bar.set_readiness_summary
    try:
        page.load_form(form)
        _wait(page, qapp)
        assert page.run_bar.writeModeButton.isEnabled()
        assert page.run_bar.write_mode() == "Append"
        assert page.run_bar.writeModeButton.toolTip().startswith("Append: skip each output already stitched")
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        targets = [Path(request.module.output.target) for request in page._owner.prepared.requests]
        assert len(targets) == 2 and all(target.is_file() for target in targets)

        def revisions():
            return [(os.stat(target).st_ino, os.stat(target).st_mtime_ns) for target in targets]

        stitched = revisions()
        exports = [target.with_name(f"iq_{target.stem}.xye") for target in targets]
        assert all(path.is_file() for path in exports)
        for path in exports:
            path.unlink()
        shown = []
        page.run_bar.set_readiness_summary = lambda text, **k: (shown.append(str(text)), summary(text, **k))[1]
        page.run_bar.startButton.click()  # nothing changed: nothing is stitched
        _wait(page, qapp)
        assert revisions() == stitched
        assert all(path.is_file() for path in exports)
        assert page.run_bar.readinessLabel.full_text().startswith(
            "Up to date · 2 outputs already stitched with these settings (Replace ⇄ stitches again)")
        # The SPEC file grows by a later scan, and scan 2 now reads another angle: only its output changed.
        spec = Path(form.spec_path)
        text = spec.read_text()
        start, end = text.index("#S 2 "), text.index("#S 3 ")
        spec.write_text(text[:start] + text[start:end].replace("\n15 -10", "\n15.5 -10", 1) + text[end:]
                        + "\n#S 9 ascan del 14 15 1 1\n#D today\n#P0 14 -10\n#N 2\n#L del  nu\n14 -10\n15 -10\n")
        exports[0].unlink()  # A mixed queue also backfills the unchanged output's companion.
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        now = revisions()
        assert now[0] == stitched[0] and now[1] != stitched[1]
        assert all(path.is_file() for path in exports)
        assert any(text.startswith("Running · Stitch 1D · 1/1 · 1 already stitched, skipped") for text in shown), shown
        status = page.run_bar.readinessLabel.full_text()
        assert status.startswith("Complete · 2 frames") and "· 1 already stitched, skipped" in status, status
        page.begin_preview()
        _wait(page, qapp)
        listed = page.preview_text.toPlainText()
        assert listed.startswith("2 stitched outputs · 2 already stitched with these settings (Append skips them)")
        assert listed.count(" · already stitched") == 2
        page.preview_dialog.close()
        targets[0].unlink()  # removed after Preview: the Run that takes Preview's check stitches it again
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert targets[0].is_file() and revisions()[1] == now[1], page.run_bar.readinessLabel.full_text()
        page.run_bar.writeModeButton.click()  # Replace: every output again
        assert page.run_bar.write_mode() == "Overwrite"
        assert page.run_bar.writeModeButton.toolTip().startswith("Replace: stitch every output again")
        before = revisions()
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert all(after != earlier for after, earlier in zip(revisions(), before))
        document = page.settings()  # the choice is saved with the page's settings
        assert document["write_mode"] == "Overwrite"
        document["write_mode"] = "Append"
        page.apply_settings(document)
        assert page.run_bar.write_mode() == "Append"
        assert page.run_bar.writeModeButton.toolTip().startswith("Append:")
    finally:
        page.run_bar.set_readiness_summary = summary
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_a_result_is_shown_once_when_it_is_mounted(qapp, tmp_path):
    """Codex testing-fixes review (startup recommendation 2): mounting a result selected it, which rebuilt
    the frame list and redrew through the list's signals, then did both again (287 frame items twice)."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        _wait(page, qapp)
        page.run_bar.startButton.click()
        _wait(page, qapp)
        target = page._owner.prepared.request.module.output.target
        page.results.clear()  # a fresh list: the reopened result is new to it
        rebuilds, draws = [], []
        clear, draw = page.frames.clear, page._draw
        page.frames.clear = lambda: (rebuilds.append(1), clear())[1]
        page._draw = lambda *args: (draws.append(1), draw(*args))[1]
        page.open_result(target)
        _wait(page, qapp)
        assert sum(page.results.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == str(target)
                   for index in range(page.results.count())) == 1 and page.frames.count() == 3
        assert (len(rebuilds), len(draws)) == (1, 1)
        page.open_result(target)  # already the current result: shown again, once
        _wait(page, qapp)
        assert (len(rebuilds), len(draws)) == (2, 2)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_mask_exact_frame_staging_is_unrotated_and_page_shutdown_reaps_worker(qapp, tmp_path):
    from threading import Event
    import fabio
    from xdart.gui.tabs.calibration.mask import MaskDialog
    form = _form(tmp_path, "multigeometry")
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("camera", "x+, z-")  # turned 180° into Stitch's frame
        source = page._source_spec_for_tools()
        cancel = Event()
        cancel.set()
        staging = tmp_path / "mask-source"
        assert MaskDialog._run((source, 1, str(staging), 180), cancel, lambda *args: None) is None
        images = tuple(staging.glob("frame-*/sample_scan1_0001.edf"))  # named after its source image
        assert len(images) == 1
        original = np.fromfile(Path(form.image_dir) / "sample_scan1_0001.raw", dtype="int32").reshape(form.detector_shape)
        np.testing.assert_array_equal(fabio.open(images[0]).data, original)
        page._make_mask()
        dialog = page._authoring_dialogs[-1]
        dialog.begin()
        assert dialog.active()
        until = time.monotonic() + 10
        receipt = page.shutdown()
        while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
            qapp.processEvents()
            time.sleep(.005)
            receipt = page.shutdown()
        assert receipt.status is PageCleanup.CLEAN
        assert not dialog.active()
        assert not page._value("mask_path")
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()


def test_directory_mask_picker_retains_the_declared_raw_decoder(qapp, tmp_path):
    from threading import Event
    import fabio
    from xdart.gui.tabs.calibration.mask import MaskDialog
    form = _form(tmp_path, "xu_hist")
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("source_kind", "Directory")
        page.set_sources((str(tmp_path),))
        page._make_mask()
        dialog = page._authoring_dialogs[-1]
        image = Path(form.image_dir) / "sample_scan1_0001.raw"
        dialog.set_source_path(image)
        cancelled = Event()
        cancelled.set()
        output = tmp_path / "directory-mask-staging"
        assert MaskDialog._run((dialog._source, 0, str(output), 0), cancelled, lambda *args: None) is None
        staged, = output.glob("frame-*/sample_scan1_0001.edf")  # named after its source image
        assert fabio.open(staged).data.shape == (195, 1475)
        np.testing.assert_array_equal(fabio.open(staged).data, np.full((195, 1475), 11))
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_changed_original_frame_is_labelled_thumbnail_in_popup(qapp, tmp_path):
    import os
    from tests.core.test_analysis_stitch_auxiliary import _publish
    path, raw, project, _receipt = _publish(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        original_y = page.curve.listDataItems()[0].yData.copy()
        source = project / "frame.raw"
        state = source.stat()
        (raw + 100).tofile(source)
        os.utime(source, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
        page._show_raw()
        page.scientific.raw_popup_full.click()
        _wait(page, qapp)
        assert page._raw_frame_view.raw is None
        assert "Thumbnail · raw source changed since processing" in page.scientific.raw_popup_status.text()
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, original_y)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("recorded", (True, False))
def test_reopened_result_marks_the_frames_the_check_excluded(qapp, tmp_path, monkeypatch, recorded):
    from xrd_tools.analysis import stitch_operation
    from tests.core.test_frame_screen import test_grouped_scans_are_screened_only_against_themselves
    if not recorded:  # the writer before verdicts were persisted
        monkeypatch.setattr(stitch_operation, "_record_frame_screen", lambda records, verdicts: None)
    target = test_grouped_scans_are_screened_only_against_themselves(tmp_path)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(target)
        _wait(page, qapp)
        items = [page.frames.item(i) for i in range(1, page.frames.count())]
        status = page.run_bar.readinessLabel.full_text()
        if recorded:
            assert [item.text() for item in items][-2:] == ["dim-3", "dim-4 · excluded"]
            assert "Frame check: empty, excluded from the result" in items[-1].toolTip()
            assert "Frame check: ok" in items[3].toolTip() and "excluded" not in items[3].text()
            assert "1 dropped frame excluded" in status
        else:
            assert not any("·" in item.text() for item in items)
            assert all("Frame check: not recorded" in item.toolTip() for item in items)
            assert "frame check not recorded" in status
    finally:
        page.shutdown()


def test_a_gi_result_opens_on_its_cut_and_offers_its_cuts_beside_q_and_two_theta(qapp, tmp_path):
    """A GI stitched record is Int's (maintainer, 2026-09-29): I(q) with Q and 2θ, and its cuts (here
    Qip + Qoop) as the record's other 1-D modes, chosen on the view's plot axis as Int chooses GI axes;
    the result opens on the cut the Axis chose (here none: the first)."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form

    request = prepare_stitch_tool(_gi_form(tmp_path, "multigeometry", StitchGI(
        ("q_ip", "q_oop"), incidence_motor="eta", sample_motors=_SAMPLE))).request
    assert run_stitch_operation(request).terminal.disposition is ModuleDisposition.COMMITTED
    payload = read_analysis_artifact(request.module.output.target)
    cuts = dict(payload.gi)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(request.module.output.target)
        _wait(page, qapp)
        axis = page.scientific.plot_axis
        assert [axis.itemData(index) for index in range(axis.count())] == ["Q", "2theta", "q_ip", "q_oop"]
        assert axis.currentData() == "q_ip"
        for mode in ("q_ip", "q_oop"):
            axis.setCurrentIndex(axis.findData(mode))
            np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, cuts[mode].axes[0][1])
            np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, cuts[mode].intensity)
        axis.setCurrentIndex(axis.findData("Q"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.intensity)
        # with Qoop chosen on the Axis, a GI result opens on Qoop
        next(button for button in page._gi_host.findChildren(QtWidgets.QPushButton)
             if button.text() == "Grazing").click()
        page.set_field("unit", "Qoop (Å⁻¹)")
        page.open_result(request.module.output.target)
        _wait(page, qapp)
        assert page.scientific.plot_axis.currentData() == "q_oop"
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, cuts["q_oop"].intensity)
        # in Stitch 2D's layout (the mode's, as Int's) the record's cuts stay on offer and shown
        page.run_bar.modeCombo.setCurrentText("Stitch 2D")
        state = page._last_map_state
        assert state.processing_mode == "Int 2D" and state.measurement_mode == "GI"
        assert {"q_ip", "q_oop"} <= {axis.itemData(index) for index in range(axis.count())}
        np.testing.assert_array_equal(state.traces[0].intensity, cuts["q_oop"].intensity)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def _published(form):
    request = prepare_stitch_tool(form).request
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return request.module.output.target, result.payload


def test_background_does_not_show_old_values_when_the_same_output_is_replaced(qapp, tmp_path):
    """Codex's round-2 review F1 reproduction: a background set on a result, the output replaced (another
    polarization) and opened again under the same key: the page shows the new result, not the arrays the
    background was made from (the background is released)."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    form = _native_form(tmp_path, "multigeometry", scans="1")
    path, original = _published(form)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        page.scientific.background.click()
        assert page._background.phase == "ACTIVE"
        np.testing.assert_allclose(page.curve.listDataItems()[0].yData, 0., atol=1e-12)
        replaced_path, replacement = _published(replace(form, polarization_factor=.75))
        assert replaced_path == path
        assert not np.array_equal(original.intensity, replacement.intensity, equal_nan=True)
        page.open_result(path)
        _wait(page, qapp)
        assert page._background.phase in {"EMPTY", "RELEASED"}
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, replacement.intensity)
    finally:
        page._release_background()
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()


def test_clean_shutdown_releases_the_background(qapp, tmp_path):
    """Codex's round-2 review F2 reproduction: a CLEAN shutdown releases the display background."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    path, _payload = _published(_native_form(tmp_path, "multigeometry", scans="1"))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(path)
        _wait(page, qapp)
        page.scientific.background.click()
        assert page._background.phase == "ACTIVE" and page._background.active_bytes > 0
        assert page.shutdown().status is PageCleanup.CLEAN
        assert page._background.phase in {"EMPTY", "RELEASED"}
        assert page._background.active_bytes == 0
    finally:
        page._release_background()
        page.shutdown()
        page.deleteLater()


def test_a_pending_result_keeps_its_retry_clickable(qapp, monkeypatch):
    """Codex's RSM stage-1 review F2, on Stitch's page too: Retry sits in the Controls panel's action row, and
    disabling the whole panel while a result's saving or check was pending disabled Retry with the editors.
    The editors lock and the action row does not (``ControlsPanel.set_editors_enabled``): Retry reaches the
    owner."""
    from xdart.gui.tools.stitch_owner import StitchOwnerFinalization, StitchToolOwner
    retried = []
    monkeypatch.setattr(StitchToolOwner, "finalization",
                        property(lambda _owner: StitchOwnerFinalization.CLEANUP_PENDING))
    monkeypatch.setattr(StitchToolOwner, "begin_retry_cleanup", lambda _owner: retried.append(True))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page._sync_actions()
        assert page.retry_button.isEnabled()
        assert not page.preview_button.isEnabled() and not page.advanced_button.isEnabled()
        assert not page._rows["spec_path"].isEnabled()
        page.retry_button.click()
        assert retried == [True]
    finally:
        monkeypatch.undo()
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()


def test_a_refused_advanced_whole_number_holds_run_and_says_why(qapp, tmp_path):
    """The shared whole-number row on Stitch's Advanced integers: a refused text (a Frame step of 0, a
    fractional First frame) is kept as typed and Source's readiness names it, so Run waits; before, Run was
    ready and the run failed on the text. A blank Last frame is all frames."""
    from tests.xdart.tools.test_stitch_native_values import _native_form
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(_native_form(tmp_path, "multigeometry", scans="1"))
        assert page.run_bar.readinessLabel.full_text().startswith("Ready")
        page.set_field("frame_step", "0")
        assert page.run_bar.readinessLabel.full_text() == "Needs Frame step: must be at least 1"
        assert not page.run_bar.startButton.isEnabled() and page._rows["frame_step"].editor.text() == "0"
        page.set_field("frame_step", "2")
        page.set_field("frame_start", "1.5")
        assert page.run_bar.readinessLabel.full_text() == "Needs First frame: '1.5' is not a whole number"
        page.set_field("frame_start", "0")
        page.set_field("frame_stop", "")
        assert page.run_bar.readinessLabel.full_text().startswith("Ready")
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()


def test_the_mask_windows_fields_are_ints_rows_and_a_refused_frame_launches_nothing(qapp, tmp_path):
    """Codex's Stitch/Int reuse audit R3 (AUD-9 stage 2b G6): the mask window's image and frame are Int's
    ``FormRow``s. The image is chosen with the row's 📁 and shown by name, and the frame is a whole number
    from 0: a refused frame is named and nothing launches."""
    import tifffile
    from xdart.gui.tabs.calibration.mask import MaskDialog
    image = tmp_path / "frame_0001.tif"
    tifffile.imwrite(image, np.zeros((4, 5), dtype=np.uint16))
    dialog = MaskDialog(tmp_path)
    try:
        assert dialog._image_row.objectName() == "controlsFormRow" and dialog.choose is dialog._image_row.browse_button
        assert dialog.frame.objectName() == "controlsLineEdit" and dialog.frame.text() == "0"
        dialog.set_source_path(image)
        assert dialog._image_row.current_value() == str(image)
        dialog.frame.setText("-1")
        dialog.begin()
        assert not dialog.active() and dialog.status.text().startswith("Frame: ")
    finally:
        dialog.shutdown()
        dialog.deleteLater()


def test_stitch_opens_the_shared_calibration_window_with_its_context(qapp, tmp_path):
    """AUD-9 stage 2b G6: Stitch's Calibrate opens the shared calibration window (``tabs/calibration``) with
    Stitch's context, from its own Project."""
    from xdart.gui.tabs.calibration.dialog import CalibrationDialog
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        page._calibrate()
        dialog = page._calibration_dialog
        assert type(dialog) is CalibrationDialog and dialog.context.page == "Stitch"
        assert dialog.project_root == tmp_path.resolve() and dialog.windowTitle() == "Stitch calibration"
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
