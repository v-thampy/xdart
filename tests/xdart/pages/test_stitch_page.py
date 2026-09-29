"""Real worker, numerical owner and artifact round trips for the native page."""

from dataclasses import replace
import json
import re
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

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
from tests.core.test_stitch_operation import _goniometer_record, _ring


class _Status:
    def show(self, text, timeout_ms=0):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(page, qapp, timeout=30):
    until = time.monotonic() + timeout
    qapp.processEvents()
    while (page._owner.busy or page._raw_pending or page._queued_jobs) and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
    qapp.processEvents()
    assert not page._owner.busy, page.run_bar.readinessLabel.full_text()


def _form(tmp_path, backend):
    source = tmp_path / "source"
    source.mkdir()
    spec = source / "sample"
    spec.write_text(
        "#F sample\n#E 1\n#D today\n#O0 del  nu  energy\n\n"
        "#S 1 ascan del 14 15 1 1\n#D today\n#P0 14 -10 17000.018\n"
        "#N 3\n#L del  nu  energy\n14 -10 17000.018\n15 -10 17000.018\n"
    )
    images = source / "images"
    images.mkdir()
    shape = (195, 1475) if backend == "xu_hist" else (195, 487)
    for i in range(2):
        array = np.full(shape, i + 10, dtype=np.int32) if backend == "xu_hist" else _ring(i).astype(np.int32)
        array.tofile(images / f"sample_scan1_{i:04d}.raw")
    geometry = source / "geometry.json"
    if backend == "xu_hist":
        geometry.write_bytes(canonical_surface_resource_bytes())
    else:
        geometry.write_text(json.dumps(_goniometer_record()))
    (tmp_path / "output").mkdir()
    return StitchToolForm(
        project_root=tmp_path, spec_path=spec, scan="1", image_dir=images,
        image_stem="sample_scan1_", frame_selector=StitchFrameSelector(),
        detector_shape=shape, raw_dtype="int32", threshold=800000,
        geometry_path=geometry, geometry_kind=StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(("del", "del"), ("nu", "nu")) if backend == "xu_hist"
            else (("del_angle", "del"), ("nu_angle", "nu")),
        q_range=(1., 5.2), npt_1d=64, output_path=tmp_path / "output" / "sample_stitch1d.nexus",
        backend=backend,
    )


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
        assert page.run_bar.readinessLabel.full_text() == "Choose sources and scans"
        assert all(not widget.toolTip() for widget in (row, row.editor, row.label))
        page.preview_button.click()  # nothing is checked without a source
        assert "source path" in page.run_bar.readinessLabel.full_text() and not page._owner.busy
        # a typed path is the selection, and the readiness line follows it at once
        _commit_source_box(page, qapp, spec[0], key="Return")
        assert page._source_paths == spec and page.settings()["sources"]["Spec"] == list(spec)
        assert page.run_bar.readinessLabel.full_text().startswith("Ready · Stitch 1D · 1 SPEC file")
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
        assert page._value("threshold") == ceiling - 1
        page.set_field("threshold_min", "4")
        policy = page.build_form().conditioning
        raw = np.array([[3, 4, ceiling - 1, ceiling]], dtype=dtype)
        original = raw.copy()
        actual = condition_stitch_frame(raw, policy)
        np.testing.assert_array_equal(np.isfinite(actual), [[False, True, True, False]])
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
        assert [page.frames.item(i).text() for i in range(page.frames.count())] == ["2-0", "2-1", "3-0"]
        original_y = page.curve.listDataItems()[0].yData.copy()
        page.scientific.plot_axis.setCurrentIndex(page.scientific.plot_axis.findData("2theta"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, original_y)
        assert page.curve.getPlotItem().getAxis("bottom").labelText == "2θ"
        page._show_raw()
        _wait(page, qapp)
        assert page._raw_frame_view.thumbnail is not None
        page._raw_popup.mode.setCurrentText("Full Raw")
        _wait(page, qapp)
        np.testing.assert_array_equal(page._raw_frame_view.raw, raw)
        page.frames.setCurrentRow(2)
        assert page._raw_popup.image.canvas.raw_image.size == 0
        assert page._raw_popup.caption.text() == "Loading selected frame…"
        _wait(page, qapp)
        assert "3-0" in page._raw_popup.caption.text()
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, original_y)
        project.rename(tmp_path / "unavailable")
        page.frames.setCurrentRow(1)
        _wait(page, qapp)
        assert page._raw_frame_view.raw is None
        assert "Thumbnail · raw source unavailable" in page._raw_popup.caption.text()
        page._raw_popup.close()
        assert page._raw_frame_view is None
        assert page._raw_popup.image.canvas.raw_image.size == 0
    finally:
        page.shutdown()


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
        assert reopened._log_scale is True and reopened._color_map == "viridis"
        assert reopened._plot_options.intensity_scale == "Log"
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
        assert reopened._plot_options.intensity_scale == "Sqrt"
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
    first, *_ = _publish(first_dir)
    second, *_ = _publish(second_dir)
    page = StitchPage(empty_host_services(_Status()))
    entered, release = Event(), Event()
    original = stitch_owner.read_analysis_artifact
    try:
        page.open_result(first)
        _wait(page, qapp)
        page.open_result(second)
        _wait(page, qapp)
        page.results.setCurrentRow(0)

        def delayed(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        monkeypatch.setattr(stitch_owner, "read_analysis_artifact", delayed)
        page.open_result(first)
        assert entered.wait(5)
        page.results.setCurrentRow(1)
        expected_y = page.curve.listDataItems()[0].yData.copy()
        release.set()
        _wait(page, qapp)
        assert page._current_payload().inspection.path == str(second)
        assert page.results.currentRow() == 1
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
        assert int(page._value("image_rotation")) == 180
        for key in ("source_motors", "image_rotation", "geometry_kind", "poni_references"):
            assert not page._rows[key].isEnabled()
        # Raw decoding is still an actual input, independent of the fitted asset.
        assert page._rows["raw_dtype"].isEnabled()
        page.set_field("geometry_path", "")
        assert page._value("source_motors") == ""
        assert page._value("image_rotation") == ""
        page._calibrate()
        assert page._calibration_dialog.isVisible()
        page._calibration_dialog.close()
        page.set_field("backend", "pyFAI MultiGeometry")
        assert page._rows["source_motors"].isEnabled()
        assert page._rows["image_rotation"].isEnabled()
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
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click()
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
        assert page._value("threshold") == np.iinfo(np.int32).max - 1
        assert page.build_form().conditioning.mode == "auto"
        page.run_bar.startButton.click()
        _wait(page, qapp, timeout=60)
        assert page.results.count() == 2, page.run_bar.readinessLabel.full_text()
        status = page.run_bar.readinessLabel.full_text()  # skips and warnings stay on the status line
        assert "· 1 scan skipped" in status and status.endswith("(Preview lists them)")
        assert not page._queued_jobs
        grouped = read_analysis_artifact(requests[1].module.output.target)
        assert grouped.native_unit == "2th_deg"
        assert len(grouped.contributing_frames) == 4
        assert [page.frames.item(i).text() for i in range(page.frames.count())] == ["1-0", "1-1", "2-0", "2-1"]
        assert page.build_form().unit == "2th_deg"
        assert page.scientific.plot_axis.currentData() == "Q"
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, grouped.intensity)
        page.scientific.plot_axis.setCurrentIndex(page.scientific.plot_axis.findData("2theta"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, grouped.radial_axis("2th_deg"))
        page.set_field("threshold_min", "4")
        assert page.build_form().conditioning == StitchConditioning("manual", 4., float(np.iinfo(np.int32).max - 1))
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


def test_mask_exact_frame_staging_is_unrotated_and_page_shutdown_reaps_worker(qapp, tmp_path):
    from threading import Event
    import fabio
    from xdart.gui.tabs.stitch.mask import StitchMaskDialog
    form = _form(tmp_path, "multigeometry")
    page = StitchPage(empty_host_services(_Status()))
    try:
        _fill_native(page, form)
        page.set_field("image_rotation", "180")
        source = page._source_spec_for_tools()
        cancel = Event()
        cancel.set()
        staging = tmp_path / "mask-source"
        assert StitchMaskDialog._run((source, 1, str(staging), 180), cancel, lambda *args: None) is None
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
    from xdart.gui.tabs.stitch.mask import StitchMaskDialog
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
        assert StitchMaskDialog._run((dialog._source, 0, str(output), 0), cancelled, lambda *args: None) is None
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
        page._raw_popup.mode.setCurrentText("Full Raw")
        _wait(page, qapp)
        assert page._raw_frame_view.raw is None
        assert "Thumbnail · raw source changed since processing" in page._raw_popup.caption.text()
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
        items = [page.frames.item(i) for i in range(page.frames.count())]
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


def test_a_gi_result_offers_its_cut_on_the_plot_axis_beside_q_and_two_theta(qapp, tmp_path):
    """A GI stitched record is Int's (maintainer, 2026-09-29): I(q) with Q and 2θ, and its cut as the
    record's second 1-D mode, chosen on the view's plot axis as Int chooses GI axes."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form

    request = prepare_stitch_tool(_gi_form(tmp_path, "multigeometry", StitchGI(
        "q_ip", incidence_motor="eta", sample_motors=_SAMPLE))).request
    assert run_stitch_operation(request).terminal.disposition is ModuleDisposition.COMMITTED
    payload = read_analysis_artifact(request.module.output.target)
    cut = payload.gi[1]
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.open_result(request.module.output.target)
        _wait(page, qapp)
        axis = page.scientific.plot_axis
        assert [axis.itemData(index) for index in range(axis.count())] == ["Q", "2theta", "q_ip"]
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.intensity)
        axis.setCurrentIndex(axis.findData("q_ip"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, cut.axes[0][1])
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, cut.intensity)
        axis.setCurrentIndex(axis.findData("Q"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.intensity)
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
