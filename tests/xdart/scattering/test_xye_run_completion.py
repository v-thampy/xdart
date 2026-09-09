"""A completed XYE-only Run shows its terminal curve and browsable output folder."""

import hashlib
from pathlib import Path
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
import tifffile

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


def _wait(app, predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.mark.parametrize("source_kind,batch,unit,prefix", [
    ("tiff", False, "q_A^-1", "iq"),
    ("hdf", True, "2th_deg", "itth"),
    ("tiff-many", False, "q_A^-1", "iq"),
])
def test_completed_xye_run_opens_terminal_file_and_output_folder(
    tmp_path, monkeypatch, source_kind, batch, unit, prefix,
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    count = 257 if source_kind == "tiff-many" else 3
    shape = (8, 8) if count > 256 else (195, 487)
    images = np.stack([np.full(shape, value, np.uint16)
                       for value in range(1, count + 1)])
    if source_kind.startswith("tiff"):
        for label, image in enumerate(images, 1):
            tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif", image)
        raw = tmp_path / "raw_0001.tif"
        raw_paths = tuple(sorted(tmp_path.glob("raw_*.tif")))
    else:
        raw = tmp_path / "raw_master.h5"
        member = tmp_path / "raw_data_000001.h5"
        with h5py.File(member, "w") as document:
            document.create_dataset("entry/data/data", data=images)
        with h5py.File(raw, "w") as document:
            document["entry/data/data_000001"] = h5py.ExternalLink(
                member.name, "/entry/data/data")
        raw_paths = (raw, member)
    raw_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in raw_paths}
    poni = tmp_path / "calibration.poni"
    detector = (
        'Detector: Detector\nDetector_config: {"pixel1": 0.000172, '
        '"pixel2": 0.000172, "max_shape": [8, 8]}\n'
        if count > 256 else "Detector: Pilatus100k\nDetector_config: {}\n"
    )
    poni.write_text(
        "poni_version: 2\n" + detector +
        "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    stale = output / ("raw" if source_kind.startswith("tiff") else "raw_master")
    stale.mkdir()
    stale = stale / f"{prefix}_unrelated_9999.xye"
    np.savetxt(stale, np.column_stack((np.arange(3), np.ones(3), np.ones(3))))
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(raw), poni_file=str(poni),
            project_root=str(tmp_path), save_path=str(output / "run.nexus"),
            output_mode="Overwrite", processing_mode="Int 1D (XYE)",
            batch_mode=batch, max_cores=1, bai_1d_args={"npt": 16, "unit": unit},
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2.0),
    )
    events = []
    original_drain = page._run_executor.drain_events

    def observe_events():
        update = original_drain()
        events.extend(update)
        return update

    monkeypatch.setattr(page._run_executor, "drain_events", observe_events)
    scientific = trace = None
    try:
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Waterfall"))
        page._shell.run_controls.startButton.click()
        assert _wait(app, lambda: any(event.kind in {
            StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED,
        } for event in events)), page._notice_text
        terminal = next(event for event in events if event.kind in {
            StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED,
        })
        assert terminal.kind is StandardEventKind.FINISHED, (terminal.detail, terminal.primary)
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.completed == terminal.total == count
        generated = tuple(sorted(path for path in output.rglob("*.xye") if path != stale))
        assert len(generated) == count
        assert all(path.name.startswith(prefix + "_") for path in generated)
        assert not tuple(output.rglob("*.nexus"))
        assert not tuple(output.rglob("*.h5"))
        assert not Path(terminal.artifact).exists()
        assert {path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in raw_paths} == raw_hashes
        assert page._intents.snapshot().thaw().processing_mode == "Int 1D (XYE)"
        assert _wait(app, lambda: (
            (context := page._context_controller.viewer_1d_context) is not None
            and context.state.value == "ready"
        )), page._notice_text
        page._refresh_shell()
        context = page._context_controller.viewer_1d_context
        assert context.paths == (str(generated[-1]),)
        assert context.current_path == str(generated[-1])
        navigation = page._context_controller.navigation
        assert len(navigation.frames) == len(navigation.selected) == 1
        scientific = page._last_scientific_projection
        assert len(scientific.traces) == 1
        assert Path(context.current_path).name in scientific.title
        assert raw.name not in scientific.title
        assert Path(context.current_path).name in page._shell.scientific.status.text()
        assert "raw_master.h5" not in page._shell.scientific.status.text()
        assert not page._context_controller.browse_pending
        path_by_frame = dict(zip(navigation.frames, context.paths, strict=True))
        for trace in scientific.traces:
            saved = np.loadtxt(path_by_frame[trace.frame])
            np.testing.assert_allclose(trace.axis.values, saved[:, 0])
            np.testing.assert_allclose(trace.intensity, saved[:, 1])
        assert len(page._shell.scientific.curve.getPlotItem().listDataItems()) == 1
        assert page._processed_browser.projection().directory == str(generated[-1].parent)
        assert _wait(app, lambda: {str(path) for path in generated}.issubset(
            {entry.artifact for entry in page._processed_browser.catalog}))
        scientific = trace = None
        # Another generated file remains reachable through the actual browser
        # selection signal; completion need not eagerly overlay every output.
        scans = page._shell.browser.scans
        row = next(index for index in range(scans.count())
                   if scans.item(index).data(QtCore.Qt.ItemDataRole.UserRole) == str(generated[0]))
        scans.setCurrentRow(row, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        assert _wait(app, lambda: (
            (context := page._context_controller.viewer_1d_context) is not None
            and context.state.value == "ready" and context.paths == (str(generated[0]),)
        )), page._notice_text
        assert page._intents.snapshot().thaw().processing_mode == "Int 1D (XYE)"
        assert page._start_permitted()[0]
        assert page._shell.run_controls.startButton.isEnabled()
        if source_kind == "tiff":
            # Run retains integration authority after browsing the output.
            events.clear()
            page._shell.run_controls.startButton.click()
            assert _wait(app, lambda: any(event.kind in {
                StandardEventKind.FINISHED, StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            } for event in events)), page._notice_text
            repeated = next(event for event in events if event.kind in {
                StandardEventKind.FINISHED, StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            })
            assert repeated.kind is StandardEventKind.FINISHED
            assert repeated.run_identity is not terminal.run_identity
            assert repeated.completed == count
            assert repeated.cleanup_status is CleanupStatus.CLEANED
            assert page._intents.snapshot().thaw().processing_mode == "Int 1D (XYE)"
            assert {path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in raw_paths} == raw_hashes
    finally:
        # These are borrowed production projections, not test-owned arrays.
        scientific = trace = None
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
