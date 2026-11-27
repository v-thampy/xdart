"""Stitch's Advanced Rows/Columns follow the Calibration's detector, as RSM's do. Maintainer, 2026-11-27, at the
beamline (RSM): "the detector was initially set to eiger2 500k. i changed it to pilatus 100k but it still didn't change
the number of rows and columns" and "i thought the rows and columns were supposed to update automatically based on the
detector and images". Stitch decodes the same headerless RAW with its own Rows/Columns (default 195 x 1475), and its
source resolver refused his Pilatus 100K frames with them the same way."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.core.test_stitch_operation import _native_form
from tests.core.test_turned_detector import _rough
from tests.xdart.pages.live_per_scan_support import wait_until
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_rsm_eiger_source import _pilatus_raw_source
from xrd_tools.core.geometry import ImageOrientation


class _Status:
    def show(self, *_args):
        pass


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dispose(page, qapp, timeout=30):
    until = time.monotonic() + timeout
    receipt = page.shutdown()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = page.shutdown()
    assert receipt.status is PageCleanup.CLEAN, receipt
    page.deleteLater()
    qapp.processEvents()


def _shape(page):
    return page._value("detector_rows"), page._value("detector_cols")


@pytest.mark.parametrize("partner", [False, True], ids=["pyfai-file-alone", "with-partner"])
def test_stitch_rows_columns_follow_the_calibration_chosen_saved_and_at_preview(qapp, tmp_path, partner):
    locator = _pilatus_raw_source(tmp_path, partner=partner)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")  # an Eiger's, left over
        page.set_field("geometry_path", str(tmp_path / locator))
        assert _shape(page) == ("195", "487")
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")
        page._calibration_project = str(page._value("project_root"))
        page._adopt_calibration(str(tmp_path / locator), None)  # the window saves under the name the row holds
        assert _shape(page) == ("195", "487")
        page.set_field("detector_rows", "512"); page.set_field("detector_cols", "1028")
        page.begin_preview()  # a partner saved later under the same name: Preview reads it first
        assert _shape(page) == ("195", "487")
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize("single_arm", [False, True])
def test_stitch_uses_a_direct_xu_calibration_under_its_chosen_name(qapp, tmp_path, single_arm):
    from xrd_tools.analysis.xu_stitch_calibration import _canonical_bytes
    from xrd_tools.integrate.xu_transfer import transfer_arm_to_xu

    orientation = ImageOrientation(rotation=90)
    record = _rough("Pilatus100k", orientation, nu_motor=None if single_arm else "nu")
    asset, _ = transfer_arm_to_xu(record, source_motors=("del", None if single_arm else "nu"),
                                 validation_angles=[(0., 0.), (12., 0.), (25., 0.)],
                                 image_rotation=orientation.rotation)
    chosen = tmp_path / ("single.xu.json" if single_arm else "direct.json")
    chosen.write_bytes(_canonical_bytes(asset))
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        page.set_field("backend", "xrayutilities histogram")
        page.set_field("geometry_path", str(chosen))
        assert _shape(page) == ("195", "487")
    finally:
        _dispose(page, qapp)


def test_stitch_typed_calibration_follows_before_any_preview(qapp, tmp_path):
    locator = _pilatus_raw_source(tmp_path, partner=False)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        page.set_field("detector_rows", "512")
        page.set_field("detector_cols", "1028")
        editor = page._rows["geometry_path"].editor
        editor.setText(str(tmp_path / locator))
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
        assert _shape(page) == ("195", "487")
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize("turned", [False, True], ids=["upright", "legacy-camera"])
def test_stitch_restore_admits_raw_and_preview_keeps_the_forms_decoder(qapp, tmp_path, turned):
    from xrd_tools.notebooks.operation_replay import _restore_forms

    form = _native_form(tmp_path, "multigeometry", scans="1")
    if turned:
        orientation = ImageOrientation(rotation=90)
        for path in Path(form.image_dir).glob("*.raw"):
            raw = np.fromfile(path, dtype="<i4").reshape(form.detector_shape)
            orientation.unapply(raw).tofile(path)
        form = replace(form, detector_shape=(487, 195), image_rotation=90)
    page = StitchPage(empty_host_services(_Status()))
    try:
        page.load_form(form)
        assert page.build_form().detector_shape == form.detector_shape
        settings = page.settings()
        recipe = page._capture_run_workflow()
        restored, = _restore_forms(recipe.mode, recipe.configuration, data_root=tmp_path, output_root=tmp_path)
        assert restored.detector_shape == form.detector_shape
        page.begin_preview()
        wait_until(qapp, lambda: not page._owner.busy)
        assert page.build_form().detector_shape == form.detector_shape
        assert page._owner.prepared is not None, page.run_bar.readinessLabel.full_text()
        settings["fields"].update(detector_rows="512", detector_cols="1028")
        page.apply_settings(settings)
        wait_until(qapp, lambda: not page._source_observer.computing)
        assert _shape(page) == tuple(map(str, form.detector_shape))
        assert page.run_bar.startButton.isEnabled(), page.run_bar.readinessLabel.full_text()
    finally:
        _dispose(page, qapp)


def test_pyfai_shape_uses_the_configured_detector_binning(tmp_path):
    from pyFAI.detectors import detector_factory
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_raw_frame

    record = _rough("Pilatus100k")
    record["detector_config"]["binning"] = [2, 2]
    path = tmp_path / "binned.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    expected = detector_factory(record["detector"], record["detector_config"]).shape
    assert expected != (195, 487)
    assert rsm_calibration_raw_frame(path, project_root=tmp_path) == expected


@pytest.mark.parametrize("mode", ["stitch", "rsm"])
def test_make_mask_uses_the_calibration_written_at_the_same_path(qapp, tmp_path, mode):
    from threading import Event
    import fabio
    from tests.xdart.pages.test_rsm_page import _close, _page

    calibration = tmp_path / "calibration" / "cal.json"
    page = _page(qapp) if mode == "rsm" else StitchPage(empty_host_services(_Status()))
    try:
        page.set_field("project_root", str(tmp_path))
        rows, columns = ("adv_rows", "adv_columns") if mode == "rsm" else ("detector_rows", "detector_cols")
        page.set_field(rows, "512")
        page.set_field(columns, "1028")
        page.set_field("geometry_path", str(calibration))  # the calibration has not arrived yet
        _pilatus_raw_source(tmp_path, partner=False)
        page._make_mask()
        dialog = page._authoring_dialogs[-1]
        image = tmp_path / "source" / "images" / "RSM_synth_scan1_0000.raw"
        dialog.set_source_path(image)
        cancelled = Event()
        cancelled.set()  # stage the real image without opening the interactive editor
        staging = tmp_path / "mask-staging"
        dialog._run((dialog._source, 0, str(staging), 0), cancelled, lambda *_: None)
        staged, = staging.glob("frame-*/*.edf")
        assert fabio.open(staged).data.shape == (195, 487)
    finally:
        (_close if mode == "rsm" else _dispose)(page, qapp)
