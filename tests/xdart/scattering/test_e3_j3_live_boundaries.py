"""Joined E3 live-boundary evidence through the mounted public shell."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
from threading import Event, Lock, Timer, current_thread
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters import (
    run_executor as executor_module,
)
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
)
from xdart.gui.tabs.scattering.adapters.source import (
    FilesystemSourceAdapter,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
    ShellProjection,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.modules.display_context import ContextKind
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io import (
    ProcessedScan,
    iter_frame_records,
    read_xye,
    write_frame_records,
)
from xrd_tools.reduction import core as reduction_core
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources import image as source_image_module
from xrd_tools.sources import nexus as nexus_source_module
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
)

from tests.xdart.scattering.test_e3_join_oracle import (
    _mount,
    _pause,
    _run,
    _wait,
)


_B_SHA256 = (
    "6110bedb3bb14c1c30978f84b43716339ff099856ab55f96e4ca7f2d9c56a3c6"
)
_XYE_SHA256 = (
    "4e61db7153772697d9cf7a5e728314ffdf4abb3150c5254d2372ca27e99a175b"
)


class _RecordingExecutor(StandardRunExecutor):
    """Observe the exact event stream without becoming another event owner."""

    def __init__(self) -> None:
        super().__init__(max_display_items=2, join_timeout=60.0)
        self.observed_events = []

    def drain_events(self):
        events = super().drain_events()
        self.observed_events.extend(events)
        return events


def _real_data_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        raise RuntimeError("XDART_TEST_DATA is required for J3 live evidence")
    root = Path(configured)
    if not root.is_dir():
        raise RuntimeError(f"J3 real-data root is unavailable: {root}")
    return root


@lru_cache(maxsize=1)
def _accepted_browse_nxs() -> Path:
    path = (
        _real_data_root()
        / "xdart_processed_data"
        / "Combi4_Angledependence_samz_4p9_03271005.nxs"
    )
    if not path.is_file():
        raise RuntimeError(f"accepted Browse fixture is unavailable: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _B_SHA256:
        raise RuntimeError(
            f"accepted Browse fixture has unexpected sha256: {digest}"
        )
    return path


def _real_intent(mode: str, tmp_path: Path) -> RunIntent:
    root = _real_data_root()
    if mode in {"standard", "gi"}:
        tiff = root / "Tiff"
        selected = (
            tiff
            / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
        )
        poni = tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"
        if not selected.is_file() or not poni.is_file():
            raise RuntimeError(
                f"accepted TIFF/PONI fixtures are unavailable under {tiff}"
            )
        return RunIntent(
            source_spec=image_series_spec(selected),
            poni_file=str(poni),
            project_root=str(tiff),
            save_path=str(tmp_path / f"{mode}.nxs"),
            output_mode="Overwrite",
            max_cores=1,
            bai_1d_args={"npt": 32, "method": "csr"},
            bai_2d_args={
                "npt_rad": 32,
                "npt_azim": 24,
                "method": "csr",
            },
            gi=GIIntent(
                enabled=mode == "gi",
                incidence_motor="th",
                th_val=0.15,
                mode_1d="q_total",
                mode_2d="qip_qoop",
            ),
        )

    if mode != "eiger":
        raise ValueError(f"unsupported J3 real-data mode: {mode}")
    fixture_root = root / "eiger" / "short"
    raw = tmp_path / "eiger-raw"
    raw.mkdir()
    for stem in (
        "Eiger_NbN_1_thin_test__200mdeg_scan001",
        "Eiger_NbN_2_thin_test__200mdeg_scan001",
    ):
        for suffix in ("_master.h5", "_data_000001.h5"):
            fixture = fixture_root / f"{stem}{suffix}"
            if not fixture.is_file():
                raise RuntimeError(
                    f"accepted Eiger fixture is unavailable: {fixture}"
                )
            (raw / fixture.name).symlink_to(fixture)
    poni = root / "eiger" / "LaB6_detxn26_detyn6p5_eta3.poni"
    if not poni.is_file():
        raise RuntimeError(f"accepted Eiger PONI is unavailable: {poni}")
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".h5",)),
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(tmp_path / "eiger-processed"),
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 32, "method": "csr"},
        bai_2d_args={
            "npt_rad": 32,
            "npt_azim": 24,
            "method": "csr",
        },
    )


def _clean_close(
    app: QtWidgets.QApplication,
    page: ScatteringWorkspace,
):
    receipt = page.close_workspace()
    deadline = time.monotonic() + 60.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.005)
        receipt = page.close_workspace()
    return receipt


def _accepted_xye() -> Path:
    root = os.environ.get("XDART_TEST_DATA")
    if not root:
        raise RuntimeError("XDART_TEST_DATA is required for J3 live evidence")
    path = Path(root) / (
        "test_relative_path/xdart_processed_data/"
        "eiger_S069Ta_redo_eta2p0_1_scan001/"
        "iq_eiger_S069Ta_redo_eta2p0_1_scan001_0044.xye"
    )
    if not path.is_file():
        raise RuntimeError(f"accepted XYE fixture is unavailable: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _XYE_SHA256:
        raise RuntimeError(
            f"accepted XYE fixture has unexpected sha256: {digest}"
        )
    return path


def _write_xye_only_b(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = _accepted_xye()
    q, intensity, sigma = read_xye(source)
    view = FrameView(
        label=44,
        axis_1d=Axis("Q", "q_A^-1", values=q),
        intensity_1d=intensity,
        sigma_1d=sigma,
        source_path=source,
        source_frame_index=0,
        raw=None,
        thumbnail=None,
        axis_2d_x=None,
        axis_2d_y=None,
        intensity_2d=None,
    )
    record = FrameRecord.from_view(view)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        write_frame_records(entry, [record])
    return (
        np.asarray(q, dtype=np.float32).astype(float),
        np.asarray(intensity, dtype=np.float32).astype(float),
    )


@pytest.mark.parametrize("mode", ("standard", "gi", "eiger"))
def test_j3_real_mounted_run_pause_browse_resume_stop_close(
    mode: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pay the joined public lifecycle with real sources and real reduction."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    gui_thread = current_thread().name
    raw_facts: list[tuple[str, str, str]] = []
    reduction_facts: list[tuple[str, str]] = []
    browse_facts: list[tuple[str, str, str]] = []
    c_entered, release_c = Event(), Event()
    source_stream_lock = Lock()
    eiger_stream_windows = 0
    second_eiger_stream = Event()

    real_open_source = executor_module.open_source
    real_read_image = source_image_module.read_image
    real_iter_cursor_chunks = (
        nexus_source_module.NexusStackSource._iter_cursor_chunks
    )
    real_reduce_frame = reduction_core._reduce_frame

    def traced_open_source(source):
        raw_facts.append(
            ("open-source", current_thread().name, str(source.uri))
        )
        return real_open_source(source)

    def traced_read_image(path, *args, **kwargs):
        raw_facts.append(
            ("read-image", current_thread().name, str(path))
        )
        return real_read_image(path, *args, **kwargs)

    def traced_iter_cursor_chunks(self, cursor, chunk_size):
        nonlocal eiger_stream_windows
        with source_stream_lock:
            eiger_stream_windows += 1
            if eiger_stream_windows == 2:
                second_eiger_stream.set()
        raw_facts.append(
            ("read-eiger-blocks", current_thread().name, str(self.path))
        )
        yield from real_iter_cursor_chunks(self, cursor, chunk_size)

    monkeypatch.setattr(executor_module, "open_source", traced_open_source)
    monkeypatch.setattr(
        source_image_module, "read_image", traced_read_image
    )
    monkeypatch.setattr(
        nexus_source_module.NexusStackSource,
        "_iter_cursor_chunks",
        traced_iter_cursor_chunks,
    )

    initial_entered, release_initial = Event(), Event()
    resume_armed, resumed_entered, release_resumed = (
        Event(),
        Event(),
        Event(),
    )
    latch_lock = Lock()
    initial_claimed = False
    resumed_claimed = False

    def latched_reduce(*args, **kwargs):
        nonlocal initial_claimed, resumed_claimed
        cancel_token = (
            kwargs.get("cancel_token")
            if "cancel_token" in kwargs
            else args[6] if len(args) > 6 else None
        )
        streaming = cancel_token is not None
        reduction_facts.append(
            (
                "stream-reduce" if streaming else "scout-reduce",
                current_thread().name,
            )
        )
        gate = None
        with latch_lock:
            if (
                streaming
                and not initial_claimed
                and (
                    mode != "eiger"
                    or second_eiger_stream.is_set()
                )
            ):
                initial_claimed = True
                initial_entered.set()
                gate = release_initial
            elif (
                streaming
                and resume_armed.is_set()
                and not resumed_claimed
            ):
                resumed_claimed = True
                resumed_entered.set()
                gate = release_resumed
        if gate is not None and not gate.wait(timeout=60.0):
            raise TimeoutError("J3 deterministic reduction latch timed out")
        return real_reduce_frame(*args, **kwargs)

    monkeypatch.setattr(
        reduction_core, "_reduce_frame", latched_reduce
    )

    def open_browse(source):
        browse_facts.append(
            ("open-scan", current_thread().name, str(source))
        )
        return ProcessedScan(source)

    def read_browse(source):
        browse_facts.append(
            ("read-records", current_thread().name, str(source))
        )
        if Path(source) == replacement_path:
            c_entered.set()
            if not release_c.wait(timeout=60.0):
                raise TimeoutError("J3 Browse C latch timed out")
        yield from iter_frame_records(source)

    loader = BrowseLoader(
        max_items=32,
        join_timeout=60.0,
        open_scan=open_browse,
        read_records=read_browse,
    )
    monkeypatch.setattr(
        page_module, "BrowseLoader", lambda **_kwargs: loader
    )

    # Resolve and authenticate the 40 MiB accepted artifact before the Qt
    # composition exists.  Test-owned hashing is not part of the live GUI I/O
    # evidence window.
    browse_path = _accepted_browse_nxs()
    replacement_path = tmp_path / "accepted-browse-c.nxs"
    replacement_path.symlink_to(browse_path)
    intent = _real_intent(mode, tmp_path)
    expected_image_paths = (
        frozenset(
            str(path)
            for path in intent.source_spec.options.get("files", ())
        )
        if mode in {"standard", "gi"}
        else frozenset()
    )
    lifecycle = ScatteringCoordinator()
    executor = _RecordingExecutor()
    sources = FilesystemSourceAdapter()
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent),
        lifecycle=lifecycle,
        sources=sources,
        executor=executor,
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    assert page._browse_loader is loader
    assert page._run_executor is executor
    assert page._pipeline._sources is sources
    applied: list[ShellProjection] = []
    apply_state = shell.apply_state

    def observe(state: ShellProjection) -> None:
        applied.append(state)
        apply_state(state)

    monkeypatch.setattr(shell, "apply_state", observe)

    initial_timer = None
    resumed_timer = None
    terminal_close = None
    page.show()
    app.processEvents()
    try:
        _wait(
            app,
            lambda: shell.run_controls.startButton.isEnabled(),
            timeout=180.0,
            diagnostic=lambda: shell.controls.readinessLabel.full_text(),
        )
        shell.run_controls.startButton.click()
        _wait(
            app,
            lambda: (
                page._context_controller.acquisition_context is not None
                and initial_entered.is_set()
            ),
            timeout=180.0,
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"notice={page._notice_text!r}"
            ),
        )

        controller = page._context_controller
        identity = controller.run_identity
        acquisition = controller.acquisition_context
        assert identity is not None
        assert acquisition is not None
        run = executor._exact_run(identity)
        assert run is not None
        assert executor.acquisition_context(identity) is acquisition
        assert acquisition.run_configuration is run.configuration
        assert acquisition.current_display_scan is run.scan
        if mode == "eiger":
            assert acquisition.scan is not run.scan
            assert acquisition.scan.name.startswith("Eiger_NbN_1_")
            assert run.scan.name.startswith("Eiger_NbN_2_")
        else:
            assert acquisition.scan is run.scan
        assert acquisition.record_store is run.display
        assert acquisition.publication_store is run.display
        if mode == "gi":
            assert run.configuration.gi.enabled is True
            assert run.configuration.gi.incidence_motor == "th"
            assert run.configuration.gi.effective_motor == "th"
            assert run.configuration.gi.mode_1d == "q_total"
            assert run.configuration.gi.mode_2d == "qip_qoop"
        else:
            assert run.configuration.gi.enabled is False

        _wait(
            app,
            lambda: (
                lifecycle.phase is RunPhase.RUNNING
                and "Pause" in shell.run_controls.startButton.text()
            ),
            timeout=30.0,
        )
        initial_timer = Timer(0.2, release_initial.set)
        initial_timer.daemon = True
        initial_timer.start()
        shell.run_controls.startButton.click()
        _wait(
            app,
            lambda: lifecycle.phase is RunPhase.PAUSED,
            timeout=60.0,
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"notice={page._notice_text!r}"
            ),
        )
        _wait(
            app,
            lambda: bool(controller.frame_keys),
            timeout=60.0,
        )

        a_configuration = acquisition.run_configuration
        a_scan = acquisition.scan
        a_display_scan = acquisition.current_display_scan
        a_records = acquisition.record_store
        a_publications = acquisition.publication_store
        a_catalog = a_publications.catalog_snapshot()
        a_residency = a_publications.residency_snapshot()
        a_keys = controller.frame_keys
        a_current = controller.navigation.current
        assert a_keys
        assert a_current is not None
        assert any(key is a_current for key in a_keys)
        assert controller.selection.names(acquisition)
        if mode == "gi":
            member = shell.scientific.title.text()
            assert Path(member).name == member
            assert member.lower().endswith((".tif", ".tiff"))
            assert shell.scientific.status.text() == member
            assert shell.scientific.raw.image.image is not None
            assert shell.scientific.cake.image.image is not None
            assert shell.scientific.curve.listDataItems()

        shell.commandRequested.emit(
            ShellCommand(
                ShellCommandKind.SELECT_SCAN, str(browse_path)
            )
        )
        _wait(
            app,
            lambda: (
                controller.selection is not None
                and controller.selection.kind is ContextKind.BROWSE
                and controller.browse_context is not None
            ),
            timeout=60.0,
            diagnostic=lambda: (
                f"browse_pending={controller.browse_pending}; "
                f"notice={page._notice_text!r}"
            ),
        )
        browse_b = controller.browse_context
        assert browse_b is not None
        b_key = controller.navigation.current
        assert b_key is not None
        assert b_key.run_identity is identity
        assert b_key is not a_current
        assert b_key.artifact == str(browse_path)
        assert controller.selection.names(browse_b)
        assert shell.scientific.frame_selector.currentData() is b_key
        assert browse_b.requested_path == str(browse_path)
        assert browse_b.record_store is not a_records
        assert browse_b.publication_store is not a_publications
        assert a_publications.catalog_snapshot() == a_catalog
        assert a_publications.residency_snapshot() == a_residency
        assert browse_facts == [
            ("open-scan", "scattering-browse", str(browse_path)),
            ("read-records", "scattering-browse", str(browse_path)),
        ]

        b_selection = controller.selection
        b_navigation = controller.navigation
        b_records = browse_b.record_store
        b_publications = browse_b.publication_store
        b_title = shell.scientific.title.text()
        b_raw = np.array(shell.scientific.raw.image.image, copy=True)
        b_cake = np.array(shell.scientific.cake.image.image, copy=True)
        b_traces = tuple(
            (
                np.array(item.xData, copy=True),
                np.array(item.yData, copy=True),
            )
            for item in shell.scientific.curve.listDataItems()
        )
        replacement_projection_start = len(applied)
        shell.commandRequested.emit(
            ShellCommand(
                ShellCommandKind.SELECT_SCAN,
                str(replacement_path),
            )
        )
        _wait(app, c_entered.is_set, timeout=60.0)
        assert browse_b.released is True
        assert len(b_records) == 0
        assert len(b_publications) == 0
        assert browse_b.invalidated is True
        assert controller.browse_context is None
        assert controller.retained_contexts == (acquisition,)
        assert controller.projectable_contexts == (acquisition,)
        assert controller.selection is b_selection
        assert controller.navigation is b_navigation
        assert controller.navigation.current is b_key
        repaint_start = len(applied)
        shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.SET_COLOR_MAP, "viridis")
        )
        app.processEvents()
        assert len(applied) > repaint_start
        repaint = applied[-1]
        assert repaint.navigation.current is b_key
        assert repaint.scientific.heavy is None
        assert repaint.scientific.retain_display is True
        assert all(
            state.navigation.current is b_key
            for state in applied[replacement_projection_start:]
        )
        assert shell.scientific.frame_selector.currentData() is b_key
        assert shell.scientific.title.text() == b_title
        np.testing.assert_array_equal(
            shell.scientific.raw.image.image, b_raw
        )
        np.testing.assert_array_equal(
            shell.scientific.cake.image.image, b_cake
        )
        held_traces = shell.scientific.curve.listDataItems()
        assert len(held_traces) == len(b_traces)
        for item, (x, y) in zip(held_traces, b_traces):
            np.testing.assert_array_equal(item.xData, x)
            np.testing.assert_array_equal(item.yData, y)

        release_c.set()
        _wait(
            app,
            lambda: (
                controller.browse_context is not None
                and controller.browse_context.requested_path
                == str(replacement_path)
                and controller.selection.names(
                    controller.browse_context
                )
            ),
            timeout=60.0,
        )
        browse_c = controller.browse_context
        c_key = controller.navigation.current
        assert browse_c is not None
        assert c_key is not None and c_key is not b_key
        assert c_key.artifact == str(replacement_path)
        assert shell.scientific.frame_selector.currentData() is c_key
        first_non_b = next(
            state
            for state in applied[replacement_projection_start:]
            if state.navigation.current is not b_key
        )
        assert first_non_b.navigation.current is c_key
        assert browse_facts == [
            ("open-scan", "scattering-browse", str(browse_path)),
            ("read-records", "scattering-browse", str(browse_path)),
            ("open-scan", "scattering-browse", str(replacement_path)),
            (
                "read-records",
                "scattering-browse",
                str(replacement_path),
            ),
        ]

        resume_armed.set()
        shell.run_controls.startButton.click()
        _wait(
            app,
            lambda: (
                lifecycle.phase is RunPhase.RUNNING
                and resumed_entered.is_set()
            ),
            timeout=60.0,
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"notice={page._notice_text!r}"
            ),
        )
        assert controller.selection.names(acquisition)
        assert controller.navigation.current is a_current
        assert shell.scientific.frame_selector.currentData() is a_current
        assert browse_c.invalidated is True
        assert controller.acquisition_context is acquisition
        assert acquisition.run_configuration is a_configuration
        assert acquisition.scan is a_scan
        assert acquisition.current_display_scan is a_display_scan
        assert acquisition.record_store is a_records
        assert acquisition.publication_store is a_publications
        assert a_publications.catalog_snapshot() == a_catalog
        assert a_publications.residency_snapshot() == a_residency
        assert all(
            key.artifact
            not in {browse_b.requested_path, browse_c.requested_path}
            for key in a_publications.catalog_snapshot().entries
        )

        resumed_timer = Timer(0.2, release_resumed.set)
        resumed_timer.daemon = True
        resumed_timer.start()
        shell.run_controls.stopButton.click()
        _wait(
            app,
            lambda: lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED},
            timeout=120.0,
            diagnostic=lambda: (
                f"phase={lifecycle.phase.value}; "
                f"notice={page._notice_text!r}"
            ),
        )
        assert lifecycle.phase is RunPhase.IDLE
        assert page._notice_text == "Standard run stopped."
        assert "Standard run stopped." in shell.scientific.status.text()
        assert controller.browse_pending is False
        _wait(
            app,
            lambda: (
                run.worker is None or not run.worker.is_alive()
            ),
            timeout=30.0,
        )
        assert run.display.hydration_thread is None
        assert loader._worker is None or not loader._worker.is_alive()

        terminals = [
            event
            for event in executor.observed_events
            if event.run_identity is identity
            and event.kind
            in {
                StandardEventKind.STOPPED,
                StandardEventKind.FINISHED,
                StandardEventKind.FAILED,
            }
        ]
        assert len(terminals) == 1
        assert terminals[0].kind is StandardEventKind.STOPPED
        assert terminals[0].cleanup_status is CleanupStatus.CLEANED
        assert all(
            event.kind is not StandardEventKind.FAILED
            for event in executor.observed_events
            if event.run_identity is identity
        )

        assert reduction_facts
        assert all(
            thread != gui_thread for _operation, thread in reduction_facts
        )
        assert raw_facts
        assert all(
            thread != gui_thread
            for _operation, thread, _path in raw_facts
        )
        if mode == "eiger":
            assert second_eiger_stream.is_set()
            assert [
                operation for operation, _thread, _path in raw_facts
            ] == ["read-eiger-blocks", "read-eiger-blocks"]
            assert len(
                {path for _operation, _thread, path in raw_facts}
            ) == 2
            assert all(
                thread == "scattering-standard"
                for _operation, thread, _path in raw_facts
            )
        else:
            opened = [
                (thread, path)
                for operation, thread, path in raw_facts
                if operation == "open-source"
            ]
            read_images = [
                (thread, path)
                for operation, thread, path in raw_facts
                if operation == "read-image"
            ]
            assert len(opened) == 1
            assert opened[0][0] == "scattering-standard"
            assert expected_image_paths
            assert 1 <= len(read_images) <= len(expected_image_paths)
            assert {
                path for _thread, path in read_images
            } <= expected_image_paths
            assert not any(
                operation == "read-eiger-blocks"
                for operation, _thread, _path in raw_facts
            )

        terminal_close = _clean_close(app, page)
        duplicate = page.close_workspace()
        assert terminal_close is duplicate
        assert terminal_close.cleanup_status is CleanupStatus.CLEANED
        assert terminal_close.cleanup_identity is identity
        assert lifecycle.closed is True
    finally:
        release_c.set()
        release_initial.set()
        release_resumed.set()
        if initial_timer is not None:
            initial_timer.join(timeout=1.0)
        if resumed_timer is not None:
            resumed_timer.join(timeout=1.0)
        if terminal_close is None:
            _clean_close(app, page)
        page.close()
        page.deleteLater()
        app.processEvents()


def test_j3_mounted_qualified_xye_only_replaces_images_and_renders_trace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    b_path = tmp_path / "accepted-xye-only.nxs"
    q, intensity = _write_xye_only_b(b_path)
    rig = _mount(
        monkeypatch,
        tmp_path / "acquisition",
        labels=tuple(range(1, 160)),
        reduction_delay=0.0015,
    )
    shell = rig.shell
    applied: list[ShellProjection] = []
    apply_state = shell.apply_state

    def observe(state: ShellProjection) -> None:
        applied.append(state)
        apply_state(state)

    monkeypatch.setattr(shell, "apply_state", observe)
    try:
        _run(rig)
        _wait(
            rig.app,
            lambda: (
                shell.scientific.raw.image.image is not None
                and shell.scientific.cake.image.image is not None
                and bool(shell.scientific.curve.listDataItems())
            ),
        )
        _pause(rig)
        acquisition = rig.controller.acquisition_context
        assert acquisition is not None

        def acquisition_navigation_is_caught_up() -> bool:
            entries = (
                acquisition.publication_store.catalog_snapshot().entries
            )
            keys = rig.controller.frame_keys
            return (
                bool(keys)
                and len(keys) == len(entries)
                and all(
                    key is entries[index]
                    for index, key in enumerate(keys)
                )
            )

        _wait(rig.app, acquisition_navigation_is_caught_up)
        a_catalog = acquisition.publication_store.catalog_snapshot()
        a_residency = acquisition.publication_store.residency_snapshot()

        rig.command(
            ShellCommand(ShellCommandKind.SELECT_SCAN, str(b_path))
        )
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
            ),
        )
        browse = rig.controller.browse_context
        assert browse is not None
        assert browse.requested_path == str(b_path)
        b_key = rig.controller.frame_keys[0]
        assert b_key.local_frame_label == 44
        _wait(
            rig.app,
            lambda: (
                shell.scientific.frame_selector.currentData() is b_key
                and shell.scientific.title.text()
                == "Browse · accepted-xye-only · frame 44"
                and shell.scientific.raw.image.image is None
                and shell.scientific.cake.image.image is None
                and len(shell.scientific.curve.listDataItems()) == 1
            ),
        )

        projection = next(
            state
            for state in reversed(applied)
            if state.navigation.current is b_key
            and state.scientific.title
            == "Browse · accepted-xye-only · frame 44"
        )
        assert projection.navigation.selected == (b_key,)
        assert projection.scientific.heavy is None
        assert projection.scientific.retain_display is False
        assert len(projection.scientific.traces) == 1
        trace = projection.scientific.traces[0]
        assert trace.frame is b_key
        np.testing.assert_array_equal(trace.axis.values, q)
        np.testing.assert_array_equal(trace.intensity, intensity)

        rendered = shell.scientific.curve.listDataItems()
        assert len(rendered) == 1
        np.testing.assert_array_equal(rendered[0].xData, q)
        np.testing.assert_array_equal(rendered[0].yData, intensity)
        axis = shell.scientific.curve.getPlotItem().getAxis("bottom")
        assert axis.labelText == "Q"
        assert axis.labelUnits == "Å⁻¹"
        assert (
            acquisition.publication_store.catalog_snapshot()
            == a_catalog
        )
        assert (
            acquisition.publication_store.residency_snapshot()
            == a_residency
        )
    finally:
        rig.close()
        rig.page.close()
        rig.page.deleteLater()
        rig.app.processEvents()
