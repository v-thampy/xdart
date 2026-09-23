"""Real worker, numerical owner and artifact round trips for the native page."""

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import StitchFrameSelector, StitchToolForm, prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchGeometryKind, run_stitch_operation
from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
from tests.core.test_stitch_operation import _goniometer_record, _ring


class _Status:
    def show(self, text, timeout_ms=0):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(page, qapp, timeout=30):
    until = time.monotonic() + timeout
    while page._owner.busy and time.monotonic() < until:
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


def _fill(page, form):
    values = {key: getattr(form, key) for key in (
        "project_root", "spec_path", "scan", "image_dir", "image_stem", "raw_dtype",
        "raw_header_skip", "geometry_path", "threshold", "npt_1d", "output_path",
    )}
    values.update(
        backend="xrayutilities histogram" if form.backend == "xu_hist" else "pyFAI MultiGeometry",
        detector_rows=form.detector_shape[0], detector_cols=form.detector_shape[1],
        source_motors=", ".join(f"{key}={value}" for key, value in form.source_motors),
        q_min=form.q_range[0], q_max=form.q_range[1],
    )
    for key, value in values.items():
        page.set_field(key, str(value))
    assert page.build_form().fingerprint == form.fingerprint


@pytest.mark.parametrize("backend", ("multigeometry", "xu_hist"))
def test_native_preview_run_repeat_reopen_and_overlay_use_real_owners(qapp, tmp_path, backend):
    form = _form(tmp_path, backend)
    reference_form = replace(form, output_path=tmp_path / "output" / "reference.nexus")
    reference = run_stitch_operation(prepare_stitch_tool(reference_form).request)
    assert reference.terminal.disposition is ModuleDisposition.COMMITTED
    page = StitchPage(empty_host_services(_Status()))
    reopened = None
    try:
        _fill(page, form)
        page.preview_button.click()
        _wait(page, qapp)
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
        page.run_bar.startButton.click()
        _wait(page, qapp)
        payload = read_analysis_artifact(form.output_path, expected_kind=AnalysisArtifactKind.STITCH_1D)
        for name in ("intensity", "coverage", "normalization"):
            np.testing.assert_array_equal(getattr(payload, name), getattr(reference.payload, name))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].xData, payload.axis("q"))
        np.testing.assert_array_equal(page.curve.listDataItems()[0].yData, payload.intensity)
        assert page.results.count() == 1

        page.set_field("npt_1d", "80")
        page.set_field("npt_1d", "64")
        assert not page.run_bar.startButton.isEnabled()
        page.set_field("npt_1d", "80")
        page.begin_preview()
        _wait(page, qapp)
        page.begin_run()
        _wait(page, qapp)
        replacement = read_analysis_artifact(form.output_path)
        assert replacement.intensity.shape == (80,)
        assert page.results.count() == 1
        assert page.shutdown().status is PageCleanup.CLEAN
        (tmp_path / "source").rename(tmp_path / "source-unavailable")

        reopened = StitchPage(empty_host_services(_Status()))
        reopened.open_result(form.output_path)
        _wait(reopened, qapp)
        assert reopened._owner.form is None
        assert reopened._owner.prepared is None
        assert not reopened.run_bar.startButton.isEnabled()
        np.testing.assert_array_equal(reopened.curve.listDataItems()[0].yData, replacement.intensity)
        reopened.open_result(reference_form.output_path)
        _wait(reopened, qapp)
        reopened.display_mode.setCurrentText("Overlay")
        reopened.results.selectAll()
        assert len(reopened.curve.listDataItems()) == 2
        assert reopened.curve.listDataItems()[0].opts["pen"].color().getRgb() == (31, 119, 180, 255)
        reopened.previous.click()
        assert reopened.results.currentRow() == 0
        assert len(reopened.results.selectedItems()) == 2
        assert len(reopened.curve.listDataItems()) == 2
        reopened.next.click()
        assert reopened.results.currentRow() == 1
        assert len(reopened.results.selectedItems()) == 2
        reopened.color_map.setCurrentText("viridis")
        reopened.log_button.click()
        assert reopened._plot_options.intensity_scale == "Log"
        assert len(reopened.curve.listDataItems()) == 2
        reopened._show_options()
        assert reopened._options.waterfall_start.isVisible() is False
        reopened.display_mode.setCurrentText("Single")
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
