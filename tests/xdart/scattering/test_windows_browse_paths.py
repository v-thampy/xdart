"""Windows path spelling must not hide real Browse frame identities."""

import ntpath
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
import tifffile

from xdart.gui.tabs.scattering import shell_projection, shell_values
from xdart.gui.tabs.scattering.browser_catalog import BrowserCatalogEntry
from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity


@pytest.mark.parametrize("spelling", ("lowercase", "forward-slashes"))
def test_windows_catalog_spelling_keeps_frames_visible(monkeypatch, spelling):
    # Exercise the actual Windows string comparison on every host; neither
    # the projection nor its Qt consumer is replaced with a fake.
    monkeypatch.setattr(shell_values, "normcase", ntpath.normcase, raising=False)
    monkeypatch.setattr(shell_projection, "normcase", ntpath.normcase, raising=False)
    path = r"C:\Users\Beamline\Data\Combi4_int2d.nexus"
    artifact = path.lower() if spelling == "lowercase" else path.replace("\\", "/")
    identity = RunIdentity(1, "windows-paths")
    frames = tuple(DisplayFrameKey(identity, "Combi4", artifact, k, k)
                   for k in range(1, 17))
    navigation = shell_values.FrameNavigationProjection(frames, frames[-1], (frames[-1],))
    catalog = (BrowserCatalogEntry(path, "Combi4_int2d.nexus", 1),)
    state = shell_projection.build_browser_projection(
        contexts=(), selection=None, navigation=navigation,
        browser_directory=ntpath.dirname(path), date_sorted=False,
        auto_last=True, catalog=catalog,
    )
    assert state.selected_scan == path
    assert len(state.frames) == 16
    assert all(a is b for a, b in zip(state.frames, frames, strict=True))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    browser = BrowserView()
    try:
        browser.reconcile(state, navigation, plot_mode="Single")
        assert browser.frame_model.rowCount() == 16
        assert browser.scans.currentItem().text() == "Combi4_int2d.nexus"
        assert browser.frames.currentIndex().data(QtCore.Qt.ItemDataRole.UserRole) is frames[-1]
    finally:
        browser.deleteLater()
        app.processEvents()


def test_mixed_case_run_and_reopened_browse_keep_all_frames(tmp_path):
    """Real Run -> terminal Browse -> reopened Browse, also run on Windows CI."""
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    from xrd_tools.io import FrameViewReader
    from xrd_tools.session.intent_store import RunIntentStore
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import image_series_spec

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    root = tmp_path / "MixedCaseProject"
    root.mkdir()
    # Stay below saturation so the independent live-thumbnail oracle has no
    # detector-value exclusions to reconstruct.
    image = (np.arange(195 * 487) % 1000).astype(np.uint16).reshape(195, 487)
    for label in range(1, 17):
        tifffile.imwrite(root / f"Combi4_{label:04d}.tif", image + label)
    poni = root / "calibration.poni"
    poni.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
        encoding="utf-8",
    )
    output = root / "Combi4_int2d.nexus"
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(root / "Combi4_0001.tif"),
            poni_file=str(poni), project_root=str(root),
            save_path=str(root / "Combi4.nexus"), output_mode="Overwrite",
            max_cores=1, processing_mode="Int 2D",
            bai_1d_args={"npt": 16}, bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2.0),
    )

    def wait(predicate):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        pytest.fail(page._notice_text or "Run/Browse did not settle")

    def ready():
        return (page._capture_current_loaded_browse() is not None
                and not page._context_controller.browse_pending
                and page._shell.browser.frame_model.rowCount() == 16
                and not page._scientific_repaint_pending)

    def assert_images(label, *, allow_live=False):
        # 1-D cache readiness does not establish that the independent image
        # hydration completed. Check actual image items and their frame data.
        science = page._shell.scientific
        wait(lambda: (science.raw.image.image is not None
                      and science.cake.image.image is not None
                      and page._last_scientific_projection.heavy is not None
                      and page._last_scientific_projection.heavy.frame.local_frame_label == label))
        with FrameViewReader(output) as reader:
            expected = reader.read(label)
        heavy = page._last_scientific_projection.heavy
        # Terminal Browse may retain the already-painted live thumbnail.
        # The saved thumbnail is percentile-clipped and quantized; both are
        # valid here, but a fresh Browse must paint the saved representation.
        from xrd_tools.io.nexus_record import make_thumbnail_array
        live = make_thumbnail_array(image + label) if allow_live else None
        if not (allow_live and np.array_equal(heavy.raw, live, equal_nan=True)):
            np.testing.assert_array_equal(heavy.raw, expected.thumbnail)
        np.testing.assert_array_equal(heavy.cake, expected.intensity_2d)
        np.testing.assert_array_equal(science.raw.image.image, heavy.raw.T[:, ::-1])
        np.testing.assert_array_equal(science.cake.image.image, expected.intensity_2d.T)

    try:
        page._shell.run_controls.startButton.click()
        wait(ready)
        capture = page._capture_current_loaded_browse()
        assert capture.labels == tuple(range(1, 17))
        assert capture.context.scalar_catalog.artifact_path == capture.request.source_path
        assert page._context_controller.navigation.current.local_frame_label == 16
        assert_images(16, allow_live=True)
        # Explicit reopen forces a fresh Browse instead of reusing acquisition.
        page._select_scan(str(output), reopen=True)
        wait(ready)
        assert_images(page._context_controller.navigation.current.local_frame_label)
        navigation = page._context_controller.navigation
        frame = navigation.frames[4]
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.SELECT_FRAME, frame=frame, frames=(frame,),
        ))
        wait(lambda: (not page._scientific_repaint_pending
                      and page._shell.scientific.navigation_current_key is frame))
        with FrameViewReader(output) as reader:
            expected = reader.read(5)
        curves = page._shell.scientific.curve.listDataItems()
        assert len(curves) == 1
        np.testing.assert_array_equal(curves[0].xData, expected.axis_1d.values)
        np.testing.assert_array_equal(curves[0].yData, expected.intensity_1d)
        assert_images(5)
    finally:
        wait(lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
