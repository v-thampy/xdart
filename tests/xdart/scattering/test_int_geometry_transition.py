"""Real Int Run/Stop/Browse and next-run geometry edits preserve presentation."""

from pathlib import Path
from threading import Event
import os

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from tests.xdart.scattering.test_stopped_run_browse import _wait
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.widgets.controls_panel import SegmentedControl
from xrd_tools.io import FrameViewReader
from xrd_tools.io.nexus_record import make_thumbnail_array
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent, ThresholdIntent
from xrd_tools.sources.image import TiffSeriesSource
from xrd_tools.sources.selection import image_series_spec


@pytest.mark.parametrize("gi", (False, True), ids=("standard", "grazing"))
def test_finished_result_survives_next_run_geometry_toggles(tmp_path, gi):
    _exercise(tmp_path, gi=gi, stop=False)


def test_stopped_gi_browse_keeps_raw_threshold_and_mask(tmp_path, monkeypatch):
    _exercise(tmp_path, gi=True, stop=True, monkeypatch=monkeypatch)


@pytest.mark.parametrize("output_mode", ("Append", "Overwrite"), ids=("append", "replace"))
def test_auto_last_stop_after_geometry_changes_keeps_the_saved_curve(
    tmp_path, monkeypatch, output_mode,
):
    """Real paced reads, Run/Stop, partial publication and saved Browse."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = np.arange(195 * 487, dtype=np.uint32).reshape(195, 487) % 1000
    for label in range(1, 81):
        tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif", raw)
    poni = tmp_path / "calibration.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
    )
    entered, release = Event(), Event()
    stop_before = [32]
    read_path = TiffSeriesSource._read_path

    def paced(source, path):
        if Path(path).stem == f"raw_{stop_before[0]:04d}":
            entered.set()
            assert release.wait(20), "Stop pacing not released"
        return read_path(source, path)

    monkeypatch.setattr(TiffSeriesSource, "_read_path", paced)
    output = tmp_path / "processed"
    output.mkdir()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "raw_0001.tif"),
            poni_file=str(poni), project_root=str(tmp_path),
            save_path=str(output), output_mode=output_mode,
            max_cores=1, processing_mode="Int 2D",
            gi=GIIntent(enabled=False, incidence_motor="Manual", th_val=.1, mode_1d="q_ip"),
            bai_1d_args={"npt": 16}, bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )), lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2),
    )
    page.resize(1280, 900)
    page.show()
    try:
        shell = page._shell
        shell.browser.auto_last.setChecked(True)
        for gi, limit in ((True, 32), (False, 40), (True, 56), (False, 64)):
            stop_before[0] = limit
            entered.clear()
            release.clear()
            segment = next(row for row in shell.controls.findChildren(SegmentedControl)
                           if row.path == ("GI", "Grazing"))
            next(button for value, button in segment._segments if value is gi).click()
            prior = page._capture_current_loaded_browse()
            prior_request = None if prior is None else prior.request
            assert _wait(app, shell.run_controls.startButton.isEnabled), page._notice_text
            shell.run_controls.startButton.click()
            assert _wait(app, entered.is_set), (gi, limit, page._notice_text)
            assert page._processed_browser.auto_last
            shell.run_controls.stopButton.click()
            release.set()
            assert _wait(app, lambda: (
                page._capture_current_loaded_browse() is not None
                and page._capture_current_loaded_browse().request is not prior_request
                and shell.run_controls.startButton.isEnabled()
            )), page._notice_text
            current = page._context_controller.navigation.current
            target = page._capture_current_loaded_browse().target
            with FrameViewReader(target) as reader:
                saved = reader.read(current.local_frame_label)
            assert saved.intensity_1d.size > 0
            assert _wait(app, lambda: (
                bool(shell.scientific.curve.listDataItems())
                and page._last_scientific_projection is not None
                and bool(page._last_scientific_projection.traces)
                and page._last_scientific_projection.traces[0].frame is current
            )), (
                gi, output_mode, page._notice_text, page._preferences,
                page._native_plot_axis_transition, page._last_scientific_projection,
            )
            curve = shell.scientific.curve.listDataItems()[0]
            np.testing.assert_allclose(curve.getData()[0], saved.axis_1d.values, rtol=1e-6)
            np.testing.assert_allclose(curve.getData()[1], saved.intensity_1d, rtol=1e-6)
            assert page._preferences.plot_axis == ("q_ip" if gi else "Q")
            assert shell.scientific.raw.canvas.raw_image.size > 0
            assert shell.scientific.cake.canvas.raw_image.size > 0
    finally:
        release.set()
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()


@pytest.mark.skipif(not os.environ.get("XDART_TEST_DATA"), reason="needs real Eiger data")
def test_real_eiger_completed_standard_survives_grazing_controls(tmp_path):
    """The reported 651-frame source/grid, with user Auto Last switched off."""
    root = Path(os.environ["XDART_TEST_DATA"]) / "eiger"
    raw = root / "long/eiger_S069Ta_redo_eta2p0_1_scan001_master.h5"
    poni = root / "LaB6_detxn26_detyn6p5_eta4p5.poni"
    if not raw.is_file() or not poni.is_file():
        pytest.skip("reported Eiger source/calibration absent")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(raw), poni_file=str(poni),
            project_root=str(tmp_path), save_path=str(tmp_path / "result.nexus"),
            output_mode="Overwrite", max_cores=4, processing_mode="Int 2D",
            threshold=ThresholdIntent(apply_threshold=True, threshold_min=0, threshold_max=10000),
            gi=GIIntent(enabled=False, incidence_motor="Manual", th_val=.1,
                        sample_orientation=4, mode_1d="q_ip"),
            bai_1d_args={"numpoints": 1000, "unit": "q_A^-1", "method": "csr",
                         "chi_offset": 90., "polarization_factor": 0.},
            bai_2d_args={"npt_rad": 500, "npt_azim": 500, "unit": "q_A^-1",
                         "method": "csr", "chi_offset": 90., "polarization_factor": 0.},
        )), lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2),
    )
    page.resize(1280, 900)
    page.show()
    try:
        shell = page._shell
        assert _wait(app, shell.run_controls.startButton.isEnabled), page._notice_text
        shell.run_controls.startButton.click()
        assert _wait(app, lambda: page._capture_current_loaded_browse() is not None,
                     timeout=120), page._notice_text
        assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())), page._notice_text
        current = page._context_controller.navigation.current
        assert current.work_ordinal == 651  # HDF5 local labels start at zero.
        x, y = (values.copy() for values in shell.scientific.curve.listDataItems()[0].getData())
        shell.browser.auto_last.setChecked(False)
        for enabled in (True, False):
            segment = next(row for row in shell.controls.findChildren(SegmentedControl)
                           if row.path == ("GI", "Grazing"))
            next(button for value, button in segment._segments if value is enabled).click()
            app.processEvents()
            assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())), (
                page._notice_text, page._preferences, page._last_scientific_projection,
            )
            assert page._context_controller.navigation.current is current
            curve = shell.scientific.curve.listDataItems()[0]
            np.testing.assert_allclose(curve.getData()[0], x, rtol=1e-6)
            np.testing.assert_allclose(curve.getData()[1], y, rtol=1e-6)
    finally:
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()


def _exercise(tmp_path, *, gi, stop, monkeypatch=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    raw = np.full((195, 487), 20, dtype=np.uint32)
    raw[20:60, 20:60] = 10000
    raw[80:120, 20:60] = 0
    raw[140:180, 20:60] = 50  # inclusive upper boundary survives
    mask = np.zeros(raw.shape, dtype=np.uint8)
    mask[20:60, 100:140] = 1
    for label in range(1, 7 if stop else 73):
        tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif", raw)
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    poni = tmp_path / "calibration.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
    )
    entered, release = Event(), Event()
    if stop:
        original = TiffSeriesSource._read_path

        def paced(source, path):
            if Path(path).stem == "raw_0003":
                entered.set()
                assert release.wait(20), "Stop pacing not released"
            return original(source, path)

        monkeypatch.setattr(TiffSeriesSource, "_read_path", paced)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(tmp_path / "raw_0001.tif"),
            poni_file=str(poni), mask_file=str(mask_path),
            project_root=str(tmp_path), save_path=str(tmp_path / "result.nexus"),
            output_mode="Overwrite", max_cores=1, processing_mode="Int 2D",
            threshold=ThresholdIntent(apply_threshold=True, threshold_min=5, threshold_max=50),
            gi=GIIntent(enabled=gi, incidence_motor="Manual", th_val=.1, mode_1d="q_ip"),
            bai_1d_args={"npt": 16}, bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )), lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2),
    )
    page.resize(1280, 900)
    page.show()
    try:
        shell = page._shell
        assert _wait(app, shell.run_controls.startButton.isEnabled), page._notice_text
        shell.run_controls.startButton.click()
        if stop:
            assert _wait(app, entered.is_set), page._notice_text
            shell.run_controls.stopButton.click()
            release.set()
        assert _wait(app, lambda: page._capture_current_loaded_browse() is not None), page._notice_text
        assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())
                     and shell.scientific.raw.canvas.raw_image.size > 0), page._notice_text
        captured = page._capture_current_loaded_browse()
        frame = page._context_controller.navigation.current
        with FrameViewReader(captured.target) as reader:
            saved = reader.read(frame.local_frame_label)
        curve = shell.scientific.curve.listDataItems()[0]
        np.testing.assert_allclose(curve.getData()[1], saved.intensity_1d)
        if stop:
            expected = make_thumbnail_array(raw, mask=mask.astype(bool) | (raw < 5) | (raw > 50))
            np.testing.assert_array_equal(np.isnan(saved.thumbnail), np.isnan(expected))
            assert np.nanmax(saved.thumbnail) <= 50
            np.testing.assert_array_equal(shell.scientific.raw.canvas.raw_image,
                                          saved.thumbnail.T[:, ::-1])
        else:
            x, y = (values.copy() for values in curve.getData())
            before_raw = shell.scientific.raw.canvas.raw_image.copy()
            before_cake = shell.scientific.cake.canvas.raw_image.copy()
            for enabled in (not gi, gi):
                segment = next(row for row in shell.controls.findChildren(SegmentedControl)
                               if row.path == ("GI", "Grazing"))
                next(button for value, button in segment._segments if value is enabled).click()
                app.processEvents()
                assert page._intents.snapshot().thaw().gi.enabled is enabled
                # The raw image is read again from the saved output, asynchronously: a slower machine can show
                # the curve first (hosted runs 40293959972, 40294962178: raw shape (0,) at the comparison).
                assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())
                             and shell.scientific.raw.canvas.raw_image.size > 0), (
                    page._notice_text, page._last_scientific_projection,
                )
                shown = shell.scientific.curve.listDataItems()[0]
                np.testing.assert_allclose(shown.getData()[0], x, rtol=1e-6)
                np.testing.assert_allclose(shown.getData()[1], y, rtol=1e-6)
                assert page._context_controller.navigation.current is frame
                # Finish can retain its live float32 preview until the next edit
                # adopts the persisted quantized thumbnail. Mask topology stays exact.
                np.testing.assert_array_equal(np.isnan(shell.scientific.raw.canvas.raw_image),
                                              np.isnan(before_raw))
                np.testing.assert_allclose(shell.scientific.raw.canvas.raw_image, before_raw,
                                           atol=30 / 255)
                np.testing.assert_allclose(shell.scientific.cake.canvas.raw_image, before_cake,
                                           rtol=1e-6)
            if not gi:
                # Reuse the actual workspace across GI and Standard runs; a
                # next-run edit must not strand the prior run's plot axis.
                for enabled in (True, False):
                    segment = next(row for row in shell.controls.findChildren(SegmentedControl)
                                   if row.path == ("GI", "Grazing"))
                    next(button for value, button in segment._segments if value is enabled).click()
                    prior_identity = page._context_controller.navigation.current.run_identity
                    assert _wait(app, shell.run_controls.startButton.isEnabled), page._notice_text
                    shell.run_controls.startButton.click()
                    assert _wait(app, lambda: (
                        page._capture_current_loaded_browse() is not None
                        and page._context_controller.navigation.current.run_identity is not prior_identity
                    )), page._notice_text
                    assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())), (
                        page._notice_text, page._preferences,
                    )
                    current = page._context_controller.navigation.current
                    target = page._capture_current_loaded_browse().target
                    with FrameViewReader(target) as reader:
                        expected = reader.read(current.local_frame_label)
                    curve = shell.scientific.curve.listDataItems()[0]
                    np.testing.assert_allclose(curve.getData()[0], expected.axis_1d.values, rtol=1e-6)
                    np.testing.assert_allclose(curve.getData()[1], expected.intensity_1d, rtol=1e-6)
                    assert page._preferences.plot_axis == ("q_ip" if enabled else "Q")
                curve = shell.scientific.curve.listDataItems()[0]
                x, y = (values.copy() for values in curve.getData())
                for enabled in (True, False):
                    next(button for value, button in segment._segments if value is enabled).click()
                    app.processEvents()
                    assert _wait(app, lambda: bool(shell.scientific.curve.listDataItems())), page._notice_text
                    shown = shell.scientific.curve.listDataItems()[0]
                    np.testing.assert_allclose(shown.getData()[0], x, rtol=1e-6)
                    np.testing.assert_allclose(shown.getData()[1], y, rtol=1e-6)
    finally:
        release.set()
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
